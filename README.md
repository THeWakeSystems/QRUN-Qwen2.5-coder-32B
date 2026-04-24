---
language:
  - zh
  - en
license: other
license_name: "Qwen License + Project Additional Terms"
license_link: "./LICENSE"
pipeline_tag: text-generation
library_name: transformers
tags:
  - qwen2
  - qwen2.5-coder
  - quantum-inspired
  - model-compression
  - code-generation
  - pytorch
base_model:
  - Qwen/Qwen2.5-Coder-32B-Instruct
model-index:
  - name: quantum-qwen2.5-coder-32b-compressed
    results:
      - task:
          type: code-generation
          name: HumanEval
        dataset:
          type: openai_humaneval
          name: HumanEval
        metrics:
          - type: pass@1
            name: Pass@1
            value: null
      - task:
          type: code-generation
          name: MBPP
        dataset:
          type: mbpp
          name: MBPP
        metrics:
          - type: pass@1
            name: Pass@1
            value: null
---

# quantum-qwen2.5-coder-32b-compressed

Qwen2.5-Coder-32B 的VQC变分量子线路启发式压缩版本（Q-RUN）。

## Model Details

- Model name: `quantum-qwen2.5-coder-32b-compressed`
- Base model: `Qwen/Qwen2.5-Coder-32B-Instruct`
- Architecture: Qwen2 CausalLM with Q-RUN-compressed FFN variants
- Compression method: Q-RUN (Quantum Re-Upload Network)
- Supported replacement modes:
  - `replace`: replace `gate_proj/up_proj/down_proj`
  - `hybrid`: replace only `down_proj`
  - `adapter`: keep original MLP and prepend Q-RUN adapter
- Primary use: code generation and code reasoning research

## Intended Uses

### Direct Use

- 代码补全与函数生成
- 压缩模型推理研究
- 压缩策略（`replace/hybrid/adapter`）对比实验

### Out-of-Scope Use

- 高风险决策场景（医疗、法律、金融自动决策）
- 未经额外安全评估的生产级自主代理

## Training and Compression Method

该仓库实现了对 Qwen2.5-Coder-32B FFN 层的量子启发式压缩：

1. 将输入投影到低维子空间。
2. 对投影特征进行 `sin/cos` 重上传编码。
3. 通过共享 MLP 聚合并回映射到目标维度。
4. 使用原始权重 SVD 主成分进行初始化（用于提升稳定性）。

更多方法细节见 [quantum_compression_report.md](quantum_compression_report.md)。

## How To Use

### Requirements

```bash
pip install torch transformers accelerate safetensors
```

### Inference Example

```bash
python examples/inference_example.py \
  --model-path /path/to/this/repo \
  --base-model-path /path/to/Qwen2.5-Coder-32B-Instruct \
  --mode hybrid
```

实现入口见 [CustomQwen32B.py](CustomQwen32B.py) 和示例脚本 [examples/inference_example.py](examples/inference_example.py)。

## Benchmark

下表为建议披露格式。请在发布前填入最终结果，并将原始日志放入 [evaluation_results/README.md](evaluation_results/README.md) 指向的数据文件。

| Model Variant | HumanEval Pass@1 | MBPP Pass@1 | EvalPlus | MMLU | Peak VRAM (GB) | Throughput (tok/s) |
|---|---:|---:|---:|---:|---:|---:|
| Base Qwen2.5-Coder-32B-Instruct | TBD | TBD | TBD | TBD | TBD | TBD |
| Q-RUN replace | TBD | TBD | TBD | TBD | TBD | TBD |
| Q-RUN hybrid | TBD | TBD | TBD | TBD | TBD | TBD |
| Q-RUN adapter | TBD | TBD | TBD | TBD | TBD | TBD |

## Evaluation Data

- Raw results and scripts: [evaluation_results/README.md](evaluation_results/README.md)
- Compression report: [quantum_compression_report.md](quantum_compression_report.md)

## Limitations

- 压缩后模型可能在长链路推理与复杂代码规划任务上出现能力退化。
- 不同替换模式在稳定性和压缩率上存在权衡。
- 基准分数对提示模板、采样参数、评测版本敏感。

## License

本项目采用组合许可模型：

- Base model weights and original model constraints: follow Qwen official license.
- Q-RUN implementation, scripts, and documentation: see [LICENSE](LICENSE).

使用前请确保你的分发和商用方式同时满足基础模型许可与本仓库附加条款。

## Citation

如果该模型或实现对你的工作有帮助，请引用本仓库，并同时引用 Qwen2.5-Coder 官方模型。

```bibtex
@misc{quantum_qwen25_coder_32b_compressed,
  title        = {quantum-qwen2.5-coder-32b-compressed},
  author       = {THeWakeSystems},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/THeWakeSystems/quantum-qwen2.5-coder-32b-compressed}
}
```

## Repository Structure

```text
.
├── config.json
├── model.safetensors.index.json
├── README.md
├── tokenizer.json
├── tokenizer_config.json
├── generation_config.json
├── LICENSE
├── quantum_compression_report.md
├── evaluation_results/
│   └── README.md
├── examples/
│   └── inference_example.py
└── CustomQwen32B.py
```
