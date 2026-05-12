# AIMS4Meta — 配方颗粒质谱向量智能分析平台

## 项目概述

AIMS4Meta 是一个面向中药配方颗粒的质谱数据智能分析平台，核心目标是构建「配方颗粒质谱向量数据集」并实现自动化多层鉴定流程。

**核心技术栈**：DreaMS 深度学习质谱嵌入（1024 维向量）+ 14 个质谱/代谢组学开源工具集成 + 多层鉴定工作流。

**当前数据规模**：
- 1030 个样本（正离子 514 / 负离子 516 / 空白 2）
- 涵盖 234 个配方颗粒品种（百部、白芍、白术、当归、大黄等）
- 数据格式：NPZ（向量）+ 元数据（化合物名、SMILES、InChIKey、前体质量、加合物、CCS、分类、分子式）

---

## 项目结构

```
AIMS4Meta/
├── README.md                           # 本文件
├── auto_backup.sh                      # 每日自动备份脚本（cron: 02:00）
├── .gitignore                          # Git 忽略规则
│
├── 配方颗粒质谱向量数据集申报材料_详细版.md   # 数据集知识产权申报文档
├── 登记申请书-模版.docx                  # 项目登记模板
├── 岭南中药材HS-GC-IMS指纹图谱.docx      # 岭南药材气相离子迁移谱研究
├── 实例.docx                           # 实例文档
│
├── 源代码/                              # 集成工具仓库（14 个）
│   ├── cfm-id-code/                    # CFM-ID — 代谢物结构预测
│   ├── classyfire_api/                 # ClassyFire — 化合物分类
│   ├── DiffMS-master/                  # DiffMS — 质谱差异分析
│   ├── DreaMS/                         # DreaMS — 深度学习质谱嵌入（核心）
│   ├── fiora/                          # FIORA — 计算机内碎裂预测
│   ├── MassSpecGym-main/              # MassSpecGym — 质谱模型基准测试
│   ├── matchms-master/                # matchms — 质谱相似度计算
│   ├── mist-main_v2/                   # MIST — 代谢物鉴定工具
│   ├── MolForge-main/                 # MolForge — 分子生成
│   ├── MS-BART-main/                  # MS-BART — 质谱语言模型
│   ├── MSFlow-main/                   # MSFlow — 质谱流式模型
│   ├── recetox-msfinder/              # MS-FINDER (RECETOX fork)
│   ├── RT_Transformer_Smiles/         # 保留时间预测 Transformer
│   └── SIRIUS/                         # SIRIUS — 代谢物鉴定
│
├── 工作流/                              # 数据分析流程
│   ├── 差异分析/                        # 样本间差异分析
│   ├── 多层鉴定/                        # 多工具串联鉴定管线（9 步骤）
│   └── 配方颗粒相似度分析.py            # 基于向量的相似度计算
│
├── 数据库/                              # 化合物与质谱数据库
│   ├── 模拟数据库/                      # 计算机模拟质谱数据
│   ├── 实验数据库/                      # 实验采集质谱数据
│   ├── DreaMS_Atlas/                   # DreaMS 官方图谱库
│   ├── NPASS3.0/                       # NPASS 天然产物活性数据库
│   ├── supplement_compounds/           # 补充化合物库
│   └── TCM_databases/                 # 中药专属数据库
│
├── 训练器/                              # 模型训练与权重
│   ├── 官方权重/                        # 预训练模型权重文件
│   └── MSFlow/                         # MSFlow 模型训练代码
│
├── 文档/                                # 项目文档
│   └── SOP/                            # 标准操作流程
│
├── docs/                               # 参考文献
│   ├── DreaMS_paper.pdf               # DreaMS 论文
│   ├── DreaMS_官方功能分析与挖掘建议.md  # DreaMS 功能分析
│   ├── dreams_features_documentation.md # DreaMS 特性文档
│   ├── FIDDLE paper.pdf               # 分子式预测论文
│   ├── fiora.pdf / MolForge.pdf       # 各工具论文
│   ├── msbart.pdf / msflow.pdf        # 各工具论文
│   └── scipharm-84-00447.pdf          # 西地那非类似物研究
│
├── logs/                               # 运行日志
└── .git/                               # Git 仓库（自动备份至 GitHub）
```

---

## 数据处理管线

```
  Waters 质谱仪 (.raw)
        │
        ▼
  Progenesis QI (峰提取、对齐、质量校正)
        │
        ▼
  MSP 格式导出
        │
        ▼
  DreaMS 模型向量化 (batch_size=256)
        │
        ▼
  NPZ 存储 + 元数据
        │
        ▼
  ┌─────────────────────────────────────┐
  │  多层鉴定管线                         │
  │  CFM-ID → SIRIUS → MS-FINDER → ...  │
  │  多工具投票 / 融合                    │
  └─────────────────────────────────────┘
        │
        ▼
  配方颗粒相似度分析 & 差异分析
```

---

## 快速开始

### 环境要求

- Python 3.8+
- CUDA 兼容 GPU（推荐，用于 DreaMS 等深度学习模型推理）
- Git LFS（大型数据库文件）
- 各工具依赖见 `源代码/<工具名>/` 下的各自文档

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd AIMS4Meta

# 安装 DreaMS（核心依赖）
cd 源代码/DreaMS
pip install -e .

# 安装 matchms（质谱数据处理）
cd ../matchms-master
pip install -e .

# 其他工具按需安装，参见各子目录文档
```

### 基本用法

```python
# 质谱向量化
python 源代码/DreaMS/encode.py --input data.msp --output embeddings.npz

# 配方颗粒相似度分析
python 工作流/配方颗粒相似度分析.py

# 多层鉴定
cd 工作流/多层鉴定
python run_pipeline.py --input sample.msp
```

---

## 数据库说明

| 数据库 | 类型 | 用途 |
|--------|------|------|
| `数据库/实验数据库/` | 实测质谱 | 配方颗粒真实样本采集 |
| `数据库/模拟数据库/` | 模拟质谱 | 计算机模拟生成，扩增训练数据 |
| `数据库/NPASS3.0/` | 天然产物 | 天然产物活性数据参考 |
| `数据库/TCM_databases/` | 中药专属 | 中药化合物专属知识库 |
| `数据库/DreaMS_Atlas/` | 质谱图谱 | DreaMS 官方参考图谱 |
| `数据库/supplement_compounds/` | 补充化合物 | 额外化合物结构信息 |

---

## 工作流说明

### 多层鉴定
多工具串联的代谢物鉴定管线，整合 CFM-ID、SIRIUS、MS-FINDER 等工具，通过投票/融合策略提高鉴定准确率。详见 `工作流/多层鉴定/`。

### 差异分析
样本组间差异化合物筛选，用于质量控制和标志物发现。详见 `工作流/差异分析/`。

### 配方颗粒相似度分析
基于 DreaMS 1024 维向量，计算配方颗粒间化学组成相似度，支持批次一致性评估和品种聚类。

---

## 自动备份

项目配置了每日自动备份（cron: `0 2 * * * /stor3/AIMS4Meta/auto_backup.sh`），将变更推送至 GitHub 远程仓库。

---

## 路线图

- [ ] 建立工具编排框架（统一输入/输出接口）
- [ ] 数据库版本化（DVC / Git LFS）
- [ ] 工作流文档化（每个流程 README + 示例数据）
- [ ] 微调管线闭环（领域数据 -> 微调 -> 评估 -> 部署）
- [ ] CI/CD 集成（自动测试工作流可复现性）
- [ ] Web 可视化界面（向量空间可视化、鉴定结果展示）

---

## 参考论文

- DreaMS: *Deep Learning Embeddings for Mass Spectrometry* — `docs/DreaMS_paper.pdf`
- FIORA: *In-Silico Fragmentation* — `docs/fiora.pdf`
- MolForge: *Molecular Generation from MS/MS* — `docs/MolForge.pdf`
- MS-BART: *Mass Spectrometry Language Model* — `docs/msbart.pdf`
- MSFlow: *Flow-based MS Analysis* — `docs/msflow.pdf`
- FIDDLE: *Formula Prediction from MS/MS* — `docs/FIDDLE paper.pdf`

---

## 维护者

- 项目负责人：[待补充]
- 联系方式：[待补充]
- 最后更新：2026-05-07
