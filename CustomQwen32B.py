"""
Qwen2.5-Coder-32B Q-RUN 实现

用于生成任务的CausalLM模型，支持8卡4090分布式训练
"""

import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM, Qwen2Config
from transformers.activations import ACT2FN
from accelerate import init_empty_weights


class SimpleMLP(nn.Module):
    """两层MLP（共享参数）"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Q_RUNLayer(nn.Module):
    """
    Q-RUN Layer for 32B模型
    
    配置针对Qwen2.5-Coder-32B优化:
    - hidden_size: 5120
    - intermediate_size: 27648
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_reuploads=3,
        use_independent_reuploads=False,
        mlp_hidden_size=32
    ):
        super().__init__()
        self.n_reuploads = n_reuploads
        self.hidden_dim = hidden_dim
        self.use_independent_reuploads = use_independent_reuploads

        # 降维投影: 压缩到 hidden_dim//2
        if use_independent_reuploads:
            self.input_projs = nn.ModuleList([
                nn.Linear(input_dim, hidden_dim//2)
                for _ in range(n_reuploads)
            ])
        else:
            self.input_proj = nn.Linear(input_dim, hidden_dim//2)

        # 共享MLP
        mlp_input_size = 2 * n_reuploads
        self.u_proj = SimpleMLP(mlp_input_size, mlp_hidden_size, 2)

    def forward(self, x):
        # 数据重上传
        reupload_features = []

        if self.use_independent_reuploads:
            for i in range(self.n_reuploads):
                x_proj = self.input_projs[i](x)
                cos_x = torch.cos(x_proj)
                sin_x = torch.sin(x_proj)
                reupload_features.extend([sin_x, cos_x])
        else:
            x_proj = self.input_proj(x)
            cos_x = torch.cos(x_proj)
            sin_x = torch.sin(x_proj)
            for _ in range(self.n_reuploads):
                reupload_features.extend([sin_x, cos_x])

        # 堆叠特征
        out = torch.stack(reupload_features, dim=-1)

        # 共享MLP处理
        out = self.u_proj(out)

        # 展平恢复维度
        return out.flatten(start_dim=-2)

    def count_parameters(self):
        total = 0
        if self.use_independent_reuploads:
            for proj in self.input_projs:
                total += sum(p.numel() for p in proj.parameters())
        else:
            total += sum(p.numel() for p in self.input_proj.parameters())
        total += sum(p.numel() for p in self.u_proj.parameters())
        return total


class Qwen2MLP_withQRUN(nn.Module):
    """
    用Q-RUN替换Qwen2的MLP层（完整替换版）
    
    改进点：
    1. 复现原始 MLP 的计算图 (gate, up, act, down)
    2. 把 gate_up_proj 拆分为 gate_proj 和 up_proj 分别初始化
    3. down_proj 同样用原始权重主成分初始化
    """
    def __init__(
        self,
        config,
        n_reuploads=3,
        use_independent_reuploads=False,
        mlp_hidden_size=32
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate_proj = Q_RUNLayer(
            self.hidden_size,
            self.intermediate_size,
            n_reuploads=n_reuploads,
            use_independent_reuploads=use_independent_reuploads,
            mlp_hidden_size=mlp_hidden_size
        )

        self.up_proj = Q_RUNLayer(
            self.hidden_size,
            self.intermediate_size,
            n_reuploads=n_reuploads,
            use_independent_reuploads=use_independent_reuploads,
            mlp_hidden_size=mlp_hidden_size
        )

        self.down_proj = Q_RUNLayer(
            self.intermediate_size,
            self.hidden_size,
            n_reuploads=n_reuploads,
            use_independent_reuploads=use_independent_reuploads,
            mlp_hidden_size=mlp_hidden_size
        )

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = self.act_fn(gate) * up
        x = self.down_proj(x)
        return x

    def count_parameters(self):
        return (self.gate_proj.count_parameters() +
                self.up_proj.count_parameters() +
                self.down_proj.count_parameters())

    def init_from_original_mlp(self, original_mlp):
        """
        用原始 gate_proj / up_proj / down_proj 的权重
        分别初始化对应的 Q-RUN 层。
        """
        W_gate = original_mlp.gate_proj.weight.data   # [interm, hidden]
        W_up   = original_mlp.up_proj.weight.data     # [interm, hidden]
        W_down = original_mlp.down_proj.weight.data   # [hidden, interm]

        self._init_qrun_layer_from_weight(self.gate_proj, W_gate)
        self._init_qrun_layer_from_weight(self.up_proj, W_up)
        self._init_qrun_layer_from_weight(self.down_proj, W_down)
        print("  -> 已用原始 gate/up/down 权重的主成分分别初始化 Q-RUN 层")

    def _init_qrun_layer_from_weight(self, qrun_layer, W):
        """利用权重矩阵的 SVD 主成分初始化 input_proj（优先用GPU加速）"""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        W_dtype = W.dtype
        W = W.float().to(device)
        out_f, in_f = W.shape
        target_dim = qrun_layer.hidden_dim // 2

        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            if out_f >= in_f:
                init_w = U[:target_dim]
            else:
                init_w = Vh[:target_dim]

            if target_dim > init_w.shape[0]:
                pad = torch.randn(target_dim - init_w.shape[0], init_w.shape[1], dtype=init_w.dtype, device=device)
                init_w = torch.cat([init_w, pad], dim=0)
            elif target_dim < init_w.shape[0]:
                init_w = init_w[:target_dim]

            qrun_layer.input_proj.weight.copy_(init_w.to(W_dtype).cpu())
            nn.init.xavier_uniform_(qrun_layer.u_proj.fc1.weight, gain=0.1)
            nn.init.zeros_(qrun_layer.u_proj.fc1.bias)
            nn.init.xavier_uniform_(qrun_layer.u_proj.fc2.weight, gain=0.01)
            nn.init.zeros_(qrun_layer.u_proj.fc2.bias)


class Qwen2MLP_withQRUN_Hybrid(nn.Module):
    """
    【Hybrid模式】只替换 down_proj 为 Q-RUN，
    gate_proj 和 up_proj 严格保留原始权重。
    
    优势:
    1. gate 和 up 是 MLP 中决定非线性特征的关键，保留原始权重可最大程度维持模型能力。
    2. 只替换 down_proj，虽然压缩率不如全替换，但零样本可用性大大提升。
    3. 计算图与原始 MLP 严格一致。
    """
    def __init__(
        self,
        config,
        original_mlp,
        n_reuploads=3,
        use_independent_reuploads=False,
        mlp_hidden_size=32
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = original_mlp.act_fn

        # 严格保留原始 gate_proj 和 up_proj
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj

        # 只把 down_proj 替换为 Q-RUN
        self.down_proj = Q_RUNLayer(
            self.intermediate_size,
            self.hidden_size,
            n_reuploads=n_reuploads,
            use_independent_reuploads=use_independent_reuploads,
            mlp_hidden_size=mlp_hidden_size
        )

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = self.act_fn(gate) * up
        x = self.down_proj(x)
        return x

    def count_parameters(self):
        # 只统计新增的 Q-RUN 参数
        return self.down_proj.count_parameters()


class Qwen2MLP_withQRUN_Adapter(nn.Module):
    """
    【Adapter模式】保留原始 gate_up_proj / down_proj 权重，
    仅在 MLP 前加入一个轻量 Q-RUN Adapter 作为可学习残差。
    
    优势:
    1. 严格保留所有原始预训练权重。
    2. 当 scale=0 时，输出严格等于原始 MLP，无需训练即可直接推理。
    3. 训练时只需微调 Adapter 参数，原始 MLP 可冻结。
    """
    def __init__(
        self,
        config,
        original_mlp,
        n_reuploads=3,
        use_independent_reuploads=False,
        mlp_hidden_size=32
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # 保留原始权重 (transformers 5.x 中为独立的 gate_proj / up_proj / down_proj)
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn

        # Q-RUN Adapter
        self.qrun_adapter = Q_RUNLayer(
            self.hidden_size,
            self.hidden_size,
            n_reuploads=n_reuploads,
            use_independent_reuploads=use_independent_reuploads,
            mlp_hidden_size=mlp_hidden_size
        )

        # 初始化为 0
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        adapter_out = self.qrun_adapter(x) * torch.tanh(self.scale)
        x_adapted = x + adapter_out

        gate = self.gate_proj(x_adapted)
        up = self.up_proj(x_adapted)
        x_act = self.act_fn(gate) * up
        x_out = self.down_proj(x_act)
        return x_out

    def count_parameters(self):
        return sum(p.numel() for p in self.qrun_adapter.parameters()) + 1


class CustomQwen32BForCausalLM(Qwen2ForCausalLM):
    """
    Qwen2.5-Coder-32B with Q-RUN
    
    Args:
        model_name_or_path: 模型路径
        replace_ffn: 是否替换MLP层
        replace_layers: 指定替换的层索引列表 (None=全部)
        qrun_config: Q-RUN配置
        replacement_mode: 'replace', 'hybrid', 或 'adapter'
            - 'replace': gate/up/down 全部替换为 Q-RUN，用各自原始权重主成分初始化
            - 'hybrid':  只替换 down_proj 为 Q-RUN，gate/up 严格保留原始权重
            - 'adapter': 保留全部原始 MLP，前接 Q-RUN Adapter（无需训练即可推理）
    """
    def __init__(
        self,
        model_name_or_path,
        replace_ffn=True,
        replace_layers=None,
        qrun_config=None,
        replacement_mode='replace'
    ):
        config = Qwen2Config.from_pretrained(model_name_or_path)
        # 不强制设置 torch_dtype，让模型从 config.json 自然继承

        with init_empty_weights():
            super().__init__(config)

        # 加载预训练权重（包含MLP权重）
        self.load_pretrained_weights(model_name_or_path)

        if replace_ffn:
            default_qrun_config = {
                'n_reuploads': 3,
                'use_independent_reuploads': False,
                'mlp_hidden_size': 32
            }
            if qrun_config is not None:
                default_qrun_config.update(qrun_config)

            total_layers = len(self.model.layers)
            if replace_layers is None:
                replace_layers = list(range(total_layers))

            print(f"Q-RUN配置: {default_qrun_config}")
            print(f"替换模式: {replacement_mode}")
            print(f"将处理 {len(replace_layers)}/{total_layers} 层的MLP")

            for layer_idx in replace_layers:
                if layer_idx < total_layers:
                    original_mlp = self.model.layers[layer_idx].mlp

                    if replacement_mode == 'adapter':
                        new_mlp = Qwen2MLP_withQRUN_Adapter(
                            config, original_mlp, **default_qrun_config
                        ).to(torch.bfloat16)
                    elif replacement_mode == 'hybrid':
                        new_mlp = Qwen2MLP_withQRUN_Hybrid(
                            config, original_mlp, **default_qrun_config
                        ).to(torch.bfloat16)
                        self._init_qrun_layer_from_weight(
                            new_mlp.down_proj, original_mlp.down_proj.weight.data
                        )
                        if layer_idx % 8 == 0 or layer_idx == replace_layers[-1]:
                            print(f"  -> 已完成层 {layer_idx}/{total_layers-1} 的 Q-RUN 初始化", flush=True)
                    else:  # 'replace'
                        new_mlp = Qwen2MLP_withQRUN(
                            config, **default_qrun_config
                        ).to(torch.bfloat16)
                        new_mlp.init_from_original_mlp(original_mlp)
                        if layer_idx % 8 == 0 or layer_idx == replace_layers[-1]:
                            print(f"  -> 已完成层 {layer_idx}/{total_layers-1} 的 Q-RUN 初始化", flush=True)

                    self.model.layers[layer_idx].mlp = new_mlp

            self._print_parameter_stats(replace_layers)

    def load_pretrained_weights(self, model_path):
        """加载预训练权重（保留MLP权重以便后续初始化）"""
        import os
        from safetensors.torch import load_file

        print(f"加载预训练权重: {model_path}")

        safetensors_files = []
        for f in os.listdir(model_path):
            if f.endswith('.safetensors'):
                safetensors_files.append(os.path.join(model_path, f))

        if not safetensors_files:
            raise ValueError(f"在 {model_path} 中未找到safetensors文件")

        safetensors_files.sort()
        print(f"找到 {len(safetensors_files)} 个权重文件")

        state_dict = {}
        for f in safetensors_files:
            state_dict.update(load_file(f))

        missing, unexpected = self.load_state_dict(state_dict, strict=False, assign=True)

        if missing:
            print(f"缺失的权重: {len(missing)} 个")
            for m in missing[:10]:
                print(f"  - {m}")
            if len(missing) > 10:
                print(f"  ... 还有 {len(missing)-10} 个")

        if unexpected:
            print(f"未预期的权重: {len(unexpected)} 个")

        print(f"成功加载预训练权重")

    def _init_qrun_layer_from_weight(self, qrun_layer, W):
        """利用权重矩阵的 SVD 主成分初始化 input_proj（GPU加速）"""
        W_dtype = W.dtype
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        W = W.float().to(device)
        out_f, in_f = W.shape
        target_dim = qrun_layer.hidden_dim // 2

        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            if out_f >= in_f:
                init_w = U[:target_dim]
            else:
                init_w = Vh[:target_dim]

            if target_dim > init_w.shape[0]:
                pad = torch.randn(target_dim - init_w.shape[0], init_w.shape[1], dtype=init_w.dtype, device=init_w.device)
                init_w = torch.cat([init_w, pad], dim=0)
            elif target_dim < init_w.shape[0]:
                init_w = init_w[:target_dim]

            qrun_layer.input_proj.weight.copy_(init_w.to(W_dtype).cpu())
            nn.init.xavier_uniform_(qrun_layer.u_proj.fc1.weight, gain=0.1)
            nn.init.zeros_(qrun_layer.u_proj.fc1.bias)
            nn.init.xavier_uniform_(qrun_layer.u_proj.fc2.weight, gain=0.01)
            nn.init.zeros_(qrun_layer.u_proj.fc2.bias)

    def _print_parameter_stats(self, replaced_layers):
        """打印参数量统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        qrun_params = 0
        for layer_idx in replaced_layers:
            mlp = self.model.layers[layer_idx].mlp
            if hasattr(mlp, 'count_parameters'):
                qrun_params += mlp.count_parameters()

        print(f"\n参数量统计:")
        print(f"  总参数量:     {total_params/1e9:.2f}B")
        print(f"  可训练参数:   {trainable_params/1e9:.2f}B")
        print(f"  新增/Q-RUN参数: {qrun_params/1e6:.2f}M")
        print(f"  显存占用(BF16): ~{total_params*2/1e9:.1f} GB")

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_qrun_32b_model(
    model_path="/home/tju_mxd/QQwen32B/Qwen2___5-Coder-32B-Instruct",
    replace_layers=None,
    n_reuploads=3,
    replacement_mode='replace'
):
    """
    创建Q-RUN 32B模型的便捷函数
    """
    print("=" * 80)
    print("创建 Qwen2.5-Coder-32B Q-RUN 模型")
    print("=" * 80)

    model = CustomQwen32BForCausalLM(
        model_name_or_path=model_path,
        replace_ffn=True,
        replace_layers=replace_layers,
        qrun_config={
            'n_reuploads': n_reuploads,
            'use_independent_reuploads': False,
            'mlp_hidden_size': 32
        },
        replacement_mode=replacement_mode
    )

    return model


if __name__ == "__main__":
    print("测试Q-RUN 32B模型创建...")
    model = create_qrun_32b_model(
        model_path="/home/tju_mxd/QQwen32B/Qwen2___5-Coder-32B-Instruct",
        replace_layers=list(range(4)),
        n_reuploads=3,
        replacement_mode='hybrid'
    )
    print("\n测试通过！")
