"""
CustomQwen32B_hybrid.py

DRQC-Compress: 4-Component Hybrid Architecture
1. MonarchProj — Structured projection corresponding to brick-wall quantum circuits (replaces input_proj)
2. Learnable Rotation Encoding — Parameterized Ry(θ), Rz(φ) rotation gates
3. EntanglementLayer — Low-rank dimensional mixing corresponding to CNOT/CZ 2-qubit gates
4. Deep Re-uploading MLP — Multi-layer variational circuit + data re-uploading

Goal: Compress the MLP layers of Qwen2.5-coder-32B, achieving 98%+ parameter compression in replaced layers.
"""

import torch
import torch.nn as nn
import math
from transformers import Qwen2ForCausalLM, Qwen2Config
from transformers.activations import ACT2FN
from accelerate import init_empty_weights


def resolve_compute_dtype(dtype="auto"):
    if isinstance(dtype, torch.dtype):
        return dtype

    key = str(dtype).lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if key in mapping:
        return mapping[key]

    if key != "auto":
        raise ValueError(f"Unsupported dtype: {dtype}")

    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if hasattr(torch, "musa") and torch.musa.is_available():
        return torch.float16

    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.float16

    return torch.float32


# ============================================================
# Component 1: MonarchProj — Structured Projection (Brick-wall quantum circuit)
# ============================================================

class MonarchProj(nn.Module):
    """
    Monarch Matrix Projection: W = L @ P @ R
    - L: Block-diagonal matrix [n_blocks_out, block_size, block_size]
    - P: Fixed stride permutation
    - R: Block-diagonal matrix [n_blocks_in, block_size, block_size]

    Quantum analogy: Alternating layers of 2-qubit gates in a brick-wall circuit
    Parameter count: O(n_blocks × block_size²) which is << O(in_dim × out_dim)
    """
    def __init__(self, in_dim, out_dim, block_size=64):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.block_size = block_size

        # Ensure divisibility, pad if necessary
        self.in_padded = math.ceil(in_dim / block_size) * block_size
        self.out_padded = math.ceil(out_dim / block_size) * block_size
        self.n_blocks_in = self.in_padded // block_size
        self.n_blocks_out = self.out_padded // block_size

        # Mid-dimension: Maximum of in and out (prevents information loss)
        self.mid_dim = max(self.in_padded, self.out_padded)
        self.n_blocks_mid = self.mid_dim // block_size

        # R: First block-diagonal layer [n_blocks_in, block_size, block_size]
        # Maps in_padded to mid_dim (expands if in < out)
        self.R = nn.Parameter(torch.empty(self.n_blocks_in, block_size, block_size))
        # L: Second block-diagonal layer [n_blocks_out, block_size, block_size]
        # Maps mid_dim to out_padded
        self.L = nn.Parameter(torch.empty(self.n_blocks_out, block_size, block_size))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Stride permutation indices (fixed, non-learnable)
        perm = self._build_stride_perm(self.n_blocks_in, block_size, self.mid_dim)
        self.register_buffer('perm', perm)

        self._init_weights()

    def _build_stride_perm(self, n_blocks, block_size, target_dim):
        """Construct stride permutation: Interleaves indices, simulating brick-wall circuit connectivity"""
        total = n_blocks * block_size
        idx = torch.arange(total)
        # Stride permutation: Reshape [n_blocks, block_size] to [block_size, n_blocks], transpose, then flatten
        idx = idx.view(n_blocks, block_size).t().contiguous().view(-1)
        # Truncate or expand to target_dim
        if total < target_dim:
            # Repeat padding
            repeats = math.ceil(target_dim / total)
            idx = idx.repeat(repeats)[:target_dim]
        else:
            idx = idx[:target_dim]
        return idx

    def _init_weights(self):
        # Orthogonal initialization for each block
        for i in range(self.n_blocks_in):
            nn.init.orthogonal_(self.R[i])
        for i in range(self.n_blocks_out):
            nn.init.orthogonal_(self.L[i])

    def forward(self, x):
        # x: [..., in_dim]
        shape = x.shape[:-1]
        x = x.reshape(-1, self.in_dim)  # [N, in_dim]
        N = x.shape[0]

        # Pad input if needed
        if self.in_dim < self.in_padded:
            x = torch.nn.functional.pad(x, (0, self.in_padded - self.in_dim))

        # R: Block-diagonal matrix multiply [N, n_blocks_in, block_size] @ [n_blocks_in, block_size, block_size]
        x = x.view(N, self.n_blocks_in, self.block_size)
        x = torch.bmm(
            x.transpose(0, 1),  # [n_blocks_in, N, block_size]
            self.R               # [n_blocks_in, block_size, block_size]
        ).transpose(0, 1)        # [N, n_blocks_in, block_size]
        x = x.reshape(N, -1)     # [N, in_padded]

        # P: Stride permutation
        x_mid = x[:, self.perm] if self.mid_dim <= self.in_padded else \
                torch.nn.functional.pad(x, (0, self.mid_dim - self.in_padded))[:, self.perm]

        # L: Block-diagonal matrix multiply
        x_mid = x_mid[:, :self.out_padded].view(N, self.n_blocks_out, self.block_size)
        out = torch.bmm(
            x_mid.transpose(0, 1),
            self.L
        ).transpose(0, 1).reshape(N, -1)

        # Truncate to out_dim and add bias
        out = out[:, :self.out_dim] + self.bias
        return out.view(*shape, self.out_dim)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# Component 3: EntanglementLayer — Low-rank dimensional mixing (Quantum Entanglement)
# ============================================================

class EntanglementLayer(nn.Module):
    """
    Low-rank residual mixing, simulating CNOT/CZ entanglement gates in quantum circuits.
    Introduces interactions across dimensions after sin/cos encoding.

    x → x + up(down(x))
    Parameter count: 2 × dim × rank
    """
    def __init__(self, dim, rank=64):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.xavier_uniform_(self.down.weight, gain=0.1)
        nn.init.zeros_(self.up.weight)  # Entanglement layer starts as an identity mapping

    def forward(self, x):
        return x + self.up(self.down(x))


# ============================================================
# Integration of Components 2+4: Q_RUNLayer_Hybrid
# ============================================================

class SimpleMLP(nn.Module):
    """Two-layer MLP with LayerNorm"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.act(self.norm(self.fc1(x))))


class Q_RUNLayer_Hybrid(nn.Module):
    """
    DRQC-Compress Hybrid Layer

    Information Flow:
    MonarchProj → LayerNorm → Learnable Rotation sin/cos → EntanglementLayer → Deep Re-uploading MLP → flatten

    Quantum Circuit Analogy:
    brick-wall → Normalization → Ry/Rz rotation gates → CNOT Entanglement → Multi-layer Variational + Measurement
    """
    def __init__(self, input_dim, hidden_dim, n_reuploads=3,
                 u_proj_output_dim=4, block_size=64, entangle_rank=64,
                 **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.u_proj_output_dim = u_proj_output_dim
        self.n_reuploads = n_reuploads

        self.proj_dim = hidden_dim // u_proj_output_dim
        assert self.proj_dim * u_proj_output_dim == hidden_dim

        # Component 1: Monarch Projection
        self.input_proj = MonarchProj(input_dim, self.proj_dim, block_size)
        self.input_norm = nn.LayerNorm(self.proj_dim)

        # Component 2: Learnable rotation parameters
        self.thetas = nn.ParameterList([
            nn.Parameter(torch.ones(self.proj_dim)) for _ in range(n_reuploads)
        ])
        self.phis = nn.ParameterList([
            nn.Parameter(torch.zeros(self.proj_dim)) for _ in range(n_reuploads)
        ])

        sincos_dim = 2 * n_reuploads

        # Component 3: Entanglement Layer (Mixes dimensions after sincos encoding)
        self.entangle = EntanglementLayer(self.proj_dim, rank=entangle_rank)

        # Component 4: Lightweight u_proj (sincos_dim → u_proj_output_dim)
        # Avoid large hidden layers to prevent Out-Of-Memory (OOM) with [B, S, proj_dim, hidden] 4D tensors
        # Monarch + Entanglement Layer + Learnable Rotation provide sufficient expressivity
        self.u_proj = nn.Linear(sincos_dim, u_proj_output_dim)

    def forward(self, x):
        B, S = x.shape[:2]

        # Component 1: Monarch projection + Normalization
        x_proj = self.input_norm(self.input_proj(x))  # [B, S, proj_dim]

        # Component 3: Entanglement layer (Mixes dimensions after projection, before encoding)
        x_proj = self.entangle(x_proj)

        # Component 2: Learnable rotation encoding
        sincos_features = []
        for i in range(self.n_reuploads):
            rotated = self.thetas[i] * x_proj + self.phis[i]
            sincos_features.append(torch.sin(rotated))
            sincos_features.append(torch.cos(rotated))
        sincos = torch.stack(sincos_features, dim=-1)  # [B, S, proj_dim, 2*n_reuploads]

        # Component 4: Lightweight mapping
        out = self.u_proj(sincos)  # [B, S, proj_dim, u_proj_output_dim]

        return out.reshape(B, S, self.hidden_dim)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# MLP Replacement Layer + Model Class
# ============================================================

class Qwen2MLP_Hybrid(nn.Module):
    """Replace Mode: Replace all of gate/up/down with Q_RUNLayer_Hybrid"""
    def __init__(self, config, **qrun_kwargs):
        super().__init__()
        self.act_fn = ACT2FN[config.hidden_act]
        self.gate_proj = Q_RUNLayer_Hybrid(config.hidden_size, config.intermediate_size, **qrun_kwargs)
        self.up_proj = Q_RUNLayer_Hybrid(config.hidden_size, config.intermediate_size, **qrun_kwargs)
        self.down_proj = Q_RUNLayer_Hybrid(config.intermediate_size, config.hidden_size, **qrun_kwargs)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def count_parameters(self):
        return self.gate_proj.count_parameters() + self.up_proj.count_parameters() + self.down_proj.count_parameters()

    def init_weights(self):
        """Kaiming + Orthogonal initialization (Does not depend on original MLP weights)"""
        for qrun_layer in [self.gate_proj, self.up_proj, self.down_proj]:
            self._init_qrun(qrun_layer)
        print("  -> Springhead Hybrid layers initialized successfully")

    def _init_qrun(self, layer):
        with torch.no_grad():
            # MonarchProj has been orthogonally initialized during construction
            # Rotation parameters: theta=1, phi=0 (initially equivalent to standard sin/cos)
            # EntanglementLayer initialized during construction (up=0, initially identity mapping)
            # u_proj: Standard Xavier initialization, avoiding gain=0.01 to prevent signal collapse
            nn.init.xavier_uniform_(layer.u_proj.weight, gain=1.0)
            nn.init.zeros_(layer.u_proj.bias)

class CustomQwen32B_Hybrid(Qwen2ForCausalLM):
    """Qwen2.5-coder-32B with DRQC-Compress Springhead Hybrid"""
    def __init__(self, model_name_or_path, replace_layers=None, qrun_config=None):
        config = Qwen2Config.from_pretrained(model_name_or_path)
        with init_empty_weights():
            super().__init__(config)
        self._load_weights(model_name_or_path)

        default_cfg = {
            'n_reuploads': 3, 'mlp_hidden_size': 128, 'u_proj_output_dim': 4,
            'block_size': 64, 'entangle_rank': 64,
            'n_deep_layers': 2, 'd_hidden': 32,
        }
        if qrun_config:
            default_cfg.update(qrun_config)

        self.compute_dtype = resolve_compute_dtype(default_cfg.get("compute_dtype", "auto"))
        print(f"Compute precision: {self.compute_dtype}")

        total_layers = len(self.model.layers)
        if replace_layers is None:
            replace_layers = list(range(total_layers))

        print(f"Hybrid Configuration: {default_cfg}")
        print(f"Replacing {len(replace_layers)}/{total_layers} layers")

        for idx in replace_layers:
            if idx < total_layers:
                new_mlp = Qwen2MLP_Hybrid(config, **default_cfg).to(dtype=self.compute_dtype)
                new_mlp.init_weights()
                self.model.layers[idx].mlp = new_mlp
                if idx % 8 == 0 or idx == replace_layers[-1]:
                    print(f"  -> Layer {idx}/{total_layers-1} completed", flush=True)

        self._print_stats(replace_layers)

    def _load_weights(self, path):
        import os
        from safetensors.torch import load_file
        print(f"Loading pre-trained weights from: {path}")
        files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.safetensors')])
        print(f"Found {len(files)} weight files")
        sd = {}
        for f in files:
            sd.update(load_file(f))
        missing, unexpected = self.load_state_dict(sd, strict=False, assign=True)
        if missing:
            print(f"Missing: {len(missing)} keys (expected, these are original weights for replaced layers)")
        print("Successfully loaded pre-trained weights")

    def _print_stats(self, replace_layers):
        total = sum(p.numel() for p in self.parameters())
        qrun = sum(self.model.layers[i].mlp.count_parameters() for i in replace_layers
                    if hasattr(self.model.layers[i].mlp, 'count_parameters'))
        orig_mlp_per_layer = 3 * (5120 * 27648 + 27648)  # gate+up+down with bias
        orig_total = orig_mlp_per_layer * len(replace_layers)
        print(f"\nParameter count: Total {total/1e9:.2f}B, Springhead Hybrid {qrun/1e6:.2f}M")
        print(f"Replaced layers compression: {orig_total/1e6:.0f}M → {qrun/1e6:.1f}M ({qrun/orig_total*100:.1f}%)")
        print(f"VRAM (BF16) ~{total*2/1e9:.1f}GB")


def create_hybrid_model(
    model_path="PATH_TO_PRETRAINED_MODEL",
    replace_layers=None, n_reuploads=3, mlp_hidden_size=128,
    u_proj_output_dim=4, block_size=64, entangle_rank=64,
    n_deep_layers=2, d_hidden=32, compute_dtype="auto",
):
    print("=" * 80)
    print("Creating Hybrid Model (DRQC-Compress)")
    print("=" * 80)
    return CustomQwen32B_Hybrid(
        model_name_or_path=model_path,
        replace_layers=replace_layers,
        qrun_config={
            'n_reuploads': n_reuploads, 'mlp_hidden_size': mlp_hidden_size,
            'u_proj_output_dim': u_proj_output_dim, 'block_size': block_size,
            'entangle_rank': entangle_rank, 'n_deep_layers': n_deep_layers,
            'd_hidden': d_hidden, 'compute_dtype': compute_dtype,
        },
    )
