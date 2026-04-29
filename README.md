<div align="center">

# 🌀 Springhead

<h3>Quantum-Classical Hybrid LLM · Massive Compression · Multi-GPU Ready</h3>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub stars](https://img.shields.io/github/stars/THeWakeSystems/Springhead?style=social)](https://github.com/THeWakeSystems/Springhead)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/THeWakeSystems/Springhead/pulls)

---

<p align="center">
  <a href="#-about"><b>About</b></a> ·
  <a href="#-quick-start"><b>Quick Start</b></a> ·
  <a href="#-architecture"><b>Architecture</b></a> ·
  <a href="#-training"><b>Training</b></a> ·
  <a href="#-benchmarks"><b>Benchmarks</b></a> ·
  <a href="#-project-structure"><b>Structure</b></a> ·
  <a href="#-contributing"><b>Contributing</b></a>
</p>

</div>

---

## 🧬 About

**Springhead** is a quantum-classical hybrid language model built on Qwen2.5-Coder-32B by **TheWakeSystems**. It replaces a subset of classical transformer layers with proprietary Springhead Hybrid quantum-informed modules, achieving extreme parameter compression while preserving core reasoning capabilities.

### ✨ Highlights

- 🌀 **Extreme Compression** — 3,398M → 43.7M trainable parameters (≈ 1.3%), dramatically reducing memory footprint
- ⚛️ **Quantum-Classical Hybrid** — replaces 8 of 64 transformer blocks with quantum-informed tensor network layers (`MonarchProj` + `EntanglementLayer`)
- 🖥️ **Multi-GPU Ready** — automated device dispatch across up to 16 GPUs via `accelerate`, with intelligent memory load balancing
- 🧠 **Reasoning Preserved** — retains mathematical and logical reasoning performance from the base Qwen2.5-Coder-32B
- 🔌 **Drop-in Compatible** — works with standard Hugging Face Transformers pipelines

### 🎯 Use Cases

| ✅ Recommended | ❌ Not Recommended |
|:---|:---|
| Quantum-classical hybrid NN research | Production code generation (without further fine-tuning) |
| Hardware-constrained inference testing | High-risk decision-making |
| Knowledge Distillation experiments | Zero-shot critical reasoning |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU(s) (BF16 recommended)
- ~60 GB total GPU memory (single or multi-GPU)

### Installation

```bash
git clone https://github.com/THeWakeSystems/Springhead.git
cd Springhead

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Inference

```python
import torch
from transformers import AutoTokenizer
from scripts.benchmark_hybrid import load_model, generate

MODEL_PATH = "/path/to/Qwen2.5-Coder-32B"
CHECKPOINT = "checkpoints/checkpoints_hybrid_v2/epoch_2.pt"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = load_model(CHECKPOINT, MODEL_PATH, device="cuda", dtype="bf16")

response = generate(model, tokenizer, "Write a Python function for binary search.")
print(response)
```

### Run Full Benchmark

```bash
python scripts/benchmark_hybrid.py \
    --model_path /path/to/Qwen2.5-Coder-32B \
    --checkpoint checkpoints/checkpoints_hybrid_v2/epoch_2.pt \
    --device cuda \
    --dtype bf16
```

---

## 🏗 Architecture

Springhead targets layers 48–63 of the 64-layer Qwen2.5-Coder-32B backbone. Each replaced MLP is substituted with:

```
Original MLP  →  MonarchProj  →  EntanglementLayer  →  Hybrid Output
```

### Model Specifications

| Metric | Value |
|:---|:---|
| Base Model | Qwen2.5-Coder-32B |
| Total Layers | 64 |
| Hybrid Layers | 8 (layers 48–63) |
| Trainable Parameters | **43.7M** |
| Original Target Parameters | 3,398M |
| Compression Ratio | **≈ 1.3%** |
| `u_proj_output_dim` | 4 |
| `block_size` / `entangle_rank` | 64 |
| Recommended Hardware | 16× CUDA GPUs (BF16), ~58.8 GB total VRAM |

---

## 🎓 Training

### Knowledge Distillation Pipeline

The hybrid layers are trained via Knowledge Distillation to match the original layer outputs:

```bash
python scripts/train_hybrid.py \
    --model_path /path/to/Qwen2.5-Coder-32B \
    --output_dir checkpoints/ \
    --num_epochs 3 \
    --batch_size 2 \
    --learning_rate 1e-3
```

- **Freezes** all base model parameters
- **Trains only** the injected quantum-informed projections
- Supports both SFT and KD loss modes

---

## 📊 Benchmarks

Run the integrated benchmark suite across 5 task categories:

| Category | Status |
|:---|:---|
| 🧮 Math Reasoning | ✅ Stable |
| 🔢 Logic | ✅ Stable |
| 💻 Code Generation | ⚠️ Degraded (token repetition) |
| 🌐 Commonsense | ⚠️ Partial degradation |
| 🌍 Multilingual | ⚠️ Not systematically evaluated |

> **Note:** At the current 1.3% compression ratio, code generation exhibits semantic breaks and token repetition. For production deployments, consider increasing `entangle_rank` or reducing the number of replaced layers.

---

## 📁 Project Structure

```
Springhead/
├── model/
│   └── CustomQwen32B_hybrid.py    # Hybrid model architecture
├── scripts/
│   ├── train_hybrid.py            # Training / KD pipeline
│   ├── benchmark_hybrid.py        # Multi-task benchmark suite
│   └── benchmark_results/         # Saved benchmark outputs
├── examples/
│   └── simple_inference.py        # Minimal inference example
├── checkpoints/
│   └── checkpoints_hybrid_v2/     # Pretrained hybrid weights (Git LFS)
├── MODEL_CARD.md                  # Detailed model card
├── RELEASE_NOTES.md               # Version changelog
└── requirements.txt               # Python dependencies
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-idea`)
3. **Commit** your changes (`git commit -m 'Add amazing idea'`)
4. **Push** to your branch (`git push origin feature/amazing-idea`)
5. **Open** a Pull Request

Please read [MODEL_CARD.md](./MODEL_CARD.md) for model-specific considerations before submitting PRs that modify the architecture.

---

## ⚠️ Known Limitations

- **Code Generation**: Extreme compression may cause token repetition and semantic discontinuities
- **Memory Footprint**: Despite parameter compression, overall GPU memory requirement remains ~58.8 GB in BF16
- **Model Parity**: Exact behavioral parity with the upstream Qwen2.5-Coder-32B base model is not guaranteed

See [RELEASE_NOTES.md](./RELEASE_NOTES.md) for the full list and mitigation roadmap.

---

## 📄 License

This project is licensed under the **Apache 2.0** License — see [LICENSE](./LICENSE) for details.

---

<div align="center">

**Built with ❤️ by [TheWakeSystems](https://github.com/THeWakeSystems)**

</div>