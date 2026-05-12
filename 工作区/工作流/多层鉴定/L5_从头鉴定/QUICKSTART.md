# L5 从头鉴定 - 快速开始指南

## 📁 项目结构

每个工具都有独立的完整目录:

```
L5_从头鉴定/
├── MS-BART/          # MS-BART完整流程
├── DiffMS/           # DiffMS完整流程
├── MSFlow/           # MSFlow完整流程
└── scripts/          # 统一对比脚本
```

## 🚀 快速开始

### 1. MS-BART (推荐首选)

```bash
cd /stor3/AIMS4Meta/工作区/工作流/多层鉴定/L5_从头鉴定/MS-BART

# 数据预处理
python scripts/L5_MSBART_01_preprocess_data.py

# 训练模型
python scripts/L5_MSBART_02_train.py

# 评估模型
python scripts/L5_MSBART_03_evaluate.py
```

**优势:**
- ✅ 完整的训练流程
- ✅ 推理速度快
- ✅ 适合精确匹配

### 2. DiffMS

```bash
cd /stor3/AIMS4Meta/工作区/工作流/多层鉴定/L5_从头鉴定/DiffMS

# 数据预处理
python scripts/L5_DiffMS_01_preprocess_data.py

# 训练模型 (需要完整配置)
python scripts/L5_DiffMS_02_train.py

# 评估模型
python scripts/L5_DiffMS_03_evaluate.py
```

**注意:**
- ⚠️ 需要完整的DiffMS配置
- ⚠️ 训练时间较长
- ✅ 生成多样性高

### 3. MSFlow

```bash
cd /stor3/AIMS4Meta/工作区/工作流/多层鉴定/L5_从头鉴定/MSFlow

# 数据预处理
python scripts/L5_MSFlow_01_preprocess_data.py

# 训练模型
python scripts/L5_MSFlow_02_train.py

# 评估模型
python scripts/L5_MSFlow_03_evaluate.py
```

**注意:**
- ⚠️ 需要从头训练
- ⚠️ 训练时间最长
- ✅ 生成质量最高

### 4. 模型对比

```bash
cd /stor3/AIMS4Meta/工作区/工作流/多层鉴定/L5_从头鉴定

# 对比三个模型的性能
python scripts/L5_compare_models.py
```

## 📊 评估指标

- **Tanimoto相似度**: 分子结构相似性 (0-1)
- **Top-1准确率**: 相似度≥0.9的比例
- **Top-5准确率**: 相似度≥0.7的比例
- **Top-10准确率**: 相似度≥0.5的比例
- **有效性**: 生成的SMILES是否有效

## 🎯 推荐流程

**第一步: MS-BART**
- 最容易上手
- 训练时间短
- 效果稳定

**第二步: MSFlow**
- 性能最优
- 需要更多资源

**第三步: DiffMS**
- 多样性最好
- 训练最复杂

## 📝 数据集

**SpectraVerse**
- 48.8万条谱图
- 3.68万个分子
- 平均92个碎片
- 数据质量优秀

## ⚙️ 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (推荐)
- 16GB+ GPU内存

## 📚 详细文档

查看 `README.md` 获取完整文档。

---

**开始时间:** 2025-03-09
**预计完成:** 根据资源情况
