# Springhead-v1.0


<div align="center">
    <h3>Powered by Springhead Hybrid</h3>
</div>


[License](https://opensource.org/licenses/Apache-2.0) | [GitHub](https://github.com/study233333)


Optimized for Extreme Inference Efficiency · Massive Parameter Reduction · Quantum-Classical Hybrid Architecture


## Table of Contents


- [Highlights](#highlights)
- [Model Overview](#model-overview)
- [Key Characteristics](#key-characteristics)
- [Quick Start](#quick-start)
- [What's New in Springhead-v1.0](#whats-new-in-springhead-v10)
- [Training & Fine-Tuning](#training--fine-tuning)
- [Architecture](#architecture)
- [Evaluation & Benchmarks](#evaluation--benchmarks)
- [Languages](#languages)
- [Intended Use](#intended-use)
- [Safety & Limitations](#safety--limitations)
- [Model Information](#model-information)


## Highlights

- **Extreme Parameter Compression:** Reduced trainable parameters from 3398M to 43.7M (≈ 1.3%), heavily minimizing memory footprint.
- **Quantum-Classical Hybrid Layer:** Leverages proprietary Springhead Hybrid architecture by TheWakeSystems to replace specific transformer layers with quantum-informed modules.
- **Enterprise Multi-GPU Deployment:** Designed for 16× CUDA GPUs (BF16) with automated device mapping and memory load balancing.


## Model Overview

Springhead-v1.0 is a model developed by TheWakeSystems. This version is built on Qwen2.5-coder-32B and uses TheWakeSystems' proprietary Springhead Hybrid technology, reducing parameter count and memory requirements drastically while aiming to preserve core structural capabilities.

The model is intended for constrained hardware environments requiring multi-GPU distributed inference where traditional memory footprints are prohibitive.


## Key Characteristics

| Feature | Description |
| :--- | :--- |
| **Base model** | Qwen2.5-coder-32B |
| **Target Workloads** | Mathematical reasoning, logic, code generation |
| **Parameters** | 43.7M trainable parameters after Springhead Hybrid compression (reduced vs. base 3398M active layers) |
| **Architecture** | Quantum-Classical Hybrid Decoder-only Transformer |
| **Compression** | Springhead Hybrid (proprietary quantum-informed layer replacement) |
| **Primary language** | English / Chinese |
| **Recommended Hardware** | 16× CUDA GPUs (BF16), total effective VRAM ≈ 58.8 GB |


## Quick Start

The repository provides automated scripts for standard Transformers loading and inference benchmarking.

### Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Inference Generation

You can load the model seamlessly using our provided `CustomQwen32B_hybrid` wrapper:

```python
import torch
from transformers import AutoTokenizer
from scripts.benchmark_hybrid import load_model, generate

model_path = "/path/to/base/Qwen2.5-coder-32B"
checkpoint_path = "checkpoints/checkpoints_hybrid_v2/epoch_2.pt"

# Initialize Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
)

# Load the Hybrid Model (Auto-dispatches to available GPUs)
model = load_model(
    checkpoint_path=checkpoint_path,
    model_path=model_path,
    device="cuda",
    dtype="bf16",
    max_memory_per_device="20GiB"
)

response = generate(model, tokenizer, "Write a Python function to compute the greatest common divisor.")
print(response)
```

For a full benchmark suite, run the integrated script:

```bash
python scripts/benchmark_hybrid.py \
    --model_path /path/to/base/Qwen2.5-coder-32B \
    --checkpoint checkpoints/checkpoints_hybrid_v2/epoch_2.pt \
    --device cuda \
    --dtype bf16
```


## What's New in Springhead-v1.0

### Summary

- **Model developed as Springhead-v1.0:** Provides a compact architecture for coding and reasoning workloads.
- **Quantum-Classical Entanglement:** Uses `MonarchProj` and `EntanglementLayer` modules in place of standard MLP layers (e.g., layers 48 to 63 target replacement).
- **Automated Device Dispatch:** The inference script seamlessly charts GPU memory and distributes the hybrid model symmetrically using the `accelerate` library.


## Training & Fine-Tuning

### Base Model: Qwen2.5-coder-32B
The model is trained for code, mathematics, and high-quality text-oriented workloads.

### Springhead Hybrid Compression & Knowledge Distillation
- **Compression:** TheWakeSystems' Springhead Hybrid architecture substitutes classical layers with quantum-informed tensor networks.
- **Fine-tuning:** The replaced layers are trained via Knowledge Distillation (KD) or standard SFT to match the original outputs, mapping massive parameter spaces into highly compact representations.
- **Training Script:** We provide `scripts/train_hybrid.py` to replicate the SFT / KD behavior, freezing the unchanged base model parameters and updating only the lightweight quantum-informed projections.


## Architecture

### Model Specifications

| Metric | Value |
| :--- | :--- |
| **Base model** | Qwen2.5-coder-32B |
| **Trainable parameters** | 43.7M |
| **Original target parameters** | 3398M |
| **Compression Ratio** | ≈ 1.3% |
| **Replacement Layers** | Default targets layers 48 to 63 |
| **u_proj_output_dim** | 4 |
| **block_size & entangle_rank** | 64 |


## Evaluation & Benchmarks

The benchmark script (`scripts/benchmark_hybrid.py`) evaluates the model across varying tasks: Code, Math, Logic, Commonsense, and Multilingual tasks.

Currently, due to the extreme nature of the 1.3% compression ratio, generation capabilities experience significant degradation (e.g. repeated tokens, loss of coherent reasoning). Proceed with domain-specific fine-tuning or scaling up the `entangle_rank` for production deployments.


## Languages

- **Primary languages:** English, Chinese
- **Other languages:** Supported, but performance under Springhead Hybrid compression has not been systematically measured.


## Intended Use

### Recommended Use Cases
- Research into Quantum-Classical Hybrid Neural Networks.
- Hardware-constrained inference environment testing.
- Base architecture for extreme Knowledge Distillation experiments.

### Out-of-Scope Uses
- Production-grade code generation without further fine-tuning.
- High-risk decision-making or zero-shot critical reasoning.
- Any use that violates applicable safety laws or regulations.


## Safety & Limitations

### Known Limitations
- **Generation Quality Regressions:** Extreme compression strategies introduce task-specific degradations. The current iteration may exhibit token repetition and semantic discontinuities.
- **Model format:** Exact parity with upstream baselines is not guaranteed.

### Recommendations
- Perform task-specific evaluation prior to deployment.
- Consider adjusting the `replace_layers` count or the `entangle_rank` in `create_hybrid_model` to balance speed/memory against model accuracy.


## Model Information

| Attribute | Details |
| :--- | :--- |
| **Model name** | Springhead-v1.0 |
| **Based on** | Qwen2.5-coder-32B |
| **Developed by** | TheWakeSystems |
| **License** | Apache 2.0 |
| **Architecture** | Springhead Hybrid |


---
Built by TheWakeSystems. For detailed model specifications, refer to `MODEL_CARD.md`.
