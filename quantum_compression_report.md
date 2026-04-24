# Quantum Compression Report

## 1. 背景与目标

本项目旨在使用量子启发式结构（Q-RUN）对 Qwen2.5-Coder-32B 的 FFN 模块进行压缩，降低可训练参数规模与显存开销，同时尽量保持代码任务能力。

## 2. 方法概述

### 2.1 Q-RUN Layer

- 输入通过线性投影降维到 `hidden_dim // 2`
- 通过 `sin/cos` 数据重上传形成特征
- 使用共享两层 MLP 聚合后展平输出

### 2.2 三种替换策略

- `replace`：替换 `gate_proj/up_proj/down_proj`
- `hybrid`：仅替换 `down_proj`
- `adapter`：保留原 MLP，在前置位置插入 Q-RUN 残差适配器

### 2.3 初始化策略

- 对原始权重做 SVD
- 取主成分初始化 Q-RUN 投影层
- MLP 层使用 Xavier 初始化

## 3. 实验设置

- 基础模型：Qwen2.5-Coder-32B
- 硬件：8 x RTX 4090（可按实际替换）
- 精度：BF16
- 训练框架：PyTorch + Transformers + Accelerate

## 4. 评测协议

- 代码生成：HumanEval / MBPP / EvalPlus
- 推理能力：MMLU（可选）
- 速度与资源：吞吐、延迟、显存峰值

## 5. 结果摘要（待补充）

| 模式 | 压缩后可训练参数 | HumanEval Pass@1 | MBPP Pass@1 | 显存峰值 |
|---|---:|---:|---:|---:|
| replace | TBD | TBD | TBD | TBD |
| hybrid | TBD | TBD | TBD | TBD |
| adapter | TBD | TBD | TBD | TBD |

## 6. 结论（待补充）

- 在不同比例压缩下的能力-成本折中结论
- 推荐默认发布模式与超参

## 7. 复现说明

1. 固定随机种子并记录依赖版本
2. 提供完整训练与评测脚本
3. 在 [evaluation_results/README.md](evaluation_results/README.md) 提供原始日志和汇总表
