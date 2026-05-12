# 两阶段从头鉴定模型训练 (基于 MSFlow)

## 概述

本方案直接利用现有的 MSFlow/DiffMS 流程，只需要额外用 SIRIUS 生成 SpectraVerse 数据集的碎裂树。

## 流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        两阶段训练流程                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SpectraVerse MSP                                               │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────┐                                                │
│  │   SIRIUS    │  生成碎裂树 (subformulae)                      │
│  └─────────────┘                                                │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 阶段1: 编码器训练 (使用 DiffMS/MIST)                     │   │
│  │                                                          │   │
│  │ 质谱 + Tree → SpectraEncoderGrowing → CDDD (512维)      │   │
│  │                                                          │   │
│  │ 代码路径: /stor1/AIMS4Meta/源代码/DiffMS-master/         │   │
│  └─────────────────────────────────────────────────────────┘   │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 阶段2: 解码器训练 (使用 MSFlow)                          │   │
│  │                                                          │   │
│  │ CDDD → Flow Matching Decoder → SMILES                   │   │
│  │                                                          │   │
│  │ 代码路径: /stor1/AIMS4Meta/源代码/MSFlow-main/           │   │
│  └─────────────────────────────────────────────────────────┘   │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 推理: 质谱 → 编码器 → CDDD → 解码器 → SMILES            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
2_phase/
├── scripts/
│   └── run_sirius.py      # SIRIUS 碎裂树生成脚本
├── configs/
│   └── spectraverse.yaml  # SpectraVerse 数据配置
└── README.md              # 本文件
```

## 使用步骤

### 1. 生成 SIRIUS 碎裂树

```bash
cd /stor1/AIMS4Meta/模型训练/从头鉴定/2_phase

python scripts/run_sirius.py \
    --input /stor1/AIMS4Meta/数据库/spectraverse/spectraverse-1.0.1-pos_cleaned.msp \
    --output ./sirius_output \
    --num_workers 4 \
    --batch_size 100
```

### 2. 阶段1: 训练编码器 (使用 DiffMS)

```bash
# 进入 DiffMS 目录
cd /stor1/AIMS4Meta/源代码/DiffMS-master

# 修改配置文件，指向 SpectraVerse 数据
# configs/dataset/spectraverse.yaml

# 运行训练
python src/mist/train_mist.py --config configs/spectraverse.yaml
```

**关键配置：**
- `form_embedder: "peakformula"` - 使用 SIRIUS 的 subformulae
- `output_size: 512` - 输出 CDDD 维度

### 3. 阶段2: 训练解码器 (使用 MSFlow)

```bash
# 进入 MSFlow 目录
cd /stor1/AIMS4Meta/源代码/MSFlow-main

# 准备数据: SMILES + CDDD
# 使用编码器生成 CDDD

# 运行训练
python cfg_pretrain.py --config configs/spectraverse.yaml
```

### 4. 推理

```bash
# 使用 MSFlow 的推理脚本
python inference.py \
    --encoder checkpoints/encoder_best.pt \
    --decoder checkpoints/decoder_best.pt \
    --input test_spectra.msp
```

## 现有代码路径

| 组件 | 路径 |
|------|------|
| SIRIUS | `/stor1/AIMS4Meta/源代码/SIRIUS/sirius-6.3.3-linux-x64/sirius/bin/sirius` |
| DiffMS (编码器) | `/stor1/AIMS4Meta/源代码/DiffMS-master/` |
| MSFlow (解码器) | `/stor1/AIMS4Meta/源代码/MSFlow-main/` |
| SpectraVerse | `/stor1/AIMS4Meta/数据库/spectraverse/` |

## 关键文件

### DiffMS 编码器

- `src/mist/models/spectra_encoder.py` - SpectraEncoderGrowing 模型
- `src/mist/models/modules.py` - FormulaTransformer 模块
- `src/mist/data/` - 数据处理

### MSFlow 解码器

- `cfg_pretrain.py` - 解码器训练脚本
- `modules/cond_lit_model.py` - 条件 Flow Matching 模型
- `models/cfg_molbert.py` - 分子生成模型

## 与端到端方案对比

| 方面 | 端到端方案 | 两阶段方案 (本方案) |
|------|-----------|-------------------|
| 代码 | 需要新实现 | 直接使用现有代码 |
| Tree 依赖 | 不需要 | 需要 SIRIUS |
| 训练复杂度 | 中等 | 简单 |
| 错误累积 | 无 | 有 |
| 数据利用 | 需要质谱-SMILES对 | 解码器可用更多数据 |

## 注意事项

1. **SIRIUS 处理时间**: SpectraVerse 有 48.8 万谱图，SIRIUS 处理需要较长时间
2. **内存需求**: SIRIUS 输出可能较大，需要足够存储空间
3. **数据格式**: 确保 SIRIUS 输出格式与 DiffMS 兼容

## 创建时间

2026-03-13
