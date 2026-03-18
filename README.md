# Representations in Time Series Foundation Models

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2409.12915&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2409.12915)
[![Python: 3.10](https://img.shields.io/badge/Python-3.10-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/license/MIT)

This repository accompanies our ICML 2025 paper **“Exploring Representations and Interventions in Time Series Foundation Models.”**

## Overview

Time series foundation models (TSFMs) are powerful tools for various applications, but their internal representations and learned concepts are not well understood. In this study, we:

1. **Analyze representation similarity**: Investigate the structure and redundancy of representations across various TSFMs
2. **Perform model pruning**: Leverage redundancy in representations to prune layers and improve efficiency
3. **Identify and localize concepts**: Explore what concepts (periodicity, trends) are learned by these models
4. **Implement concept steering**: Manipulate latent space to influence model behavior

## Repository Structure

```
representations-in-tsfms/
├── efficiency/                # Representation similarity & pruning tools
│   ├── chronos-forecasting/   # Upstream Chronos implementation
│   ├── tsfm_similarity/       # Similarity metrics & experiments
│   ├── produce_similarity_maps.sh
│   ├── produce_and_time_models.sh
│   └── evaluate_chronos_variants.sh
├── steering/                  # Concept discovery & steering
│   ├── configs/               # YAML experiment configs
│   ├── steertool/             # Steering library & CLI
│   ├── run_steering_experiments.sh
│   └── run_separability_analysis.sh
├── environment.yml            # Conda environment specification
└── create_env.sh              # Helper script for env creation
```
路径	核心功能
efficiency/	核心模块：负责表示相似性分析和模型剪枝相关的工具、实验脚本
├─ chronos-forecasting/	依赖的上游模块：Chronos的实现代码
├─ tsfm_similarity/	核心工具：包含表示相似性的度量方法、相关实验的核心代码
├─ produce_similarity_maps.sh	实验脚本：生成不同 TSFMs 各层之间的相似性图谱
├─ produce_and_time_models.sh	实验脚本：生成剪枝后的模型，并测试剪枝后模型的运行耗时
├─ evaluate_chronos_variants.sh	实验脚本：评估剪枝后 Chronos 模型的性能表现
steering/	核心模块：负责概念发现和潜空间干预（Steering）相关工作
├─ configs/	配置文件：以 YAML 格式存储实验的各项配置参数
├─ steertool/	核心工具：包含概念操控的库代码和命令行工具（CLI）
├─ run_separability_analysis.sh	实验脚本：分析模型学到的概念的可分离性，结果生成可视化图表
├─ run_steering_experiments.sh	实验脚本：运行潜空间干预实验，验证对模型输出的操控效果
environment.yml	环境配置：Conda 环境的依赖清单，指定了 Python 3.10 等核心依赖
create_env.sh	辅助脚本：一键创建符合要求的 Conda 环境，简化环境配置流程
## Installation

1. Clone the repository and initialize submodules:
```bash
git clone --recurse-submodules git@github.com:moment-timeseries-foundation-model/representations-in-tsfms.git
cd representations-in-tsfms
```

2. Make sure that you have `conda` installed and create the environment:
```bash
bash create_env.sh
conda activate reps-tsfm
```

## Experiments

### Representation Analysis and Pruning

Analyze model representation similarity and prune redundant layers in TSFMs:

```bash
# Generate similarity maps between layers of TSFMs
cd efficiency
./produce_similarity_maps.sh

# Produce pruned models and time them
./produce_and_time_models.sh

# Evaluate pruned Chronos model variants
./evaluate_chronos_variants.sh
```

Results will be available in the `results` directory.

### Concept Identification and Steering

Analyze concept separability and steer model behaviour using the provided CLI utilities:

```bash
# move to the steering module
cd steering

# Run separability analysis (creates figures under steering/results)
./run_separability_analysis.sh

# Run steering experiments (latent intervention)
./run_steering_experiments.sh
```

## Citation

```bibtex
@inproceedings{wilinski2025exploring,
  title={Exploring Representations and Interventions in Time Series Foundation Models},
  author={Micha{\l} Wili{\'n}ski and Mononito Goswami and Willa Potosnak and Nina {\.{Z}}ukowska and Artur Dubrawski},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=goVzfYtj58}
}
```

## License

This project is licensed under the MIT License (see the `LICENSE` file for details).
