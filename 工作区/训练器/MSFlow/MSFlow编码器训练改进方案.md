# MSFlow编码器训练改进方案

> 针对中药物质基础领域的MSFlow编码器权重训练优化指南

---

## 目录

0. [总体架构：三阶段层级训练方案](#总体架构三阶段层级训练方案)
1. [第一阶段：SpectraVerse 泛化性微调](#第一阶段spectraverse-泛化性微调)
2. [第二阶段：中药领域适应训练](#第二阶段中药领域适应训练)
3. [模拟数据误差传播分析与缓解策略](#模拟数据误差传播分析与缓解策略)
4. [真实与模拟数据混合训练策略](#真实与模拟数据混合训练策略)
5. [进阶优化策略（可选）](#进阶优化策略可选)
6. [实施路线图](#实施路线图)

---

## 总体架构：三阶段层级训练方案

> 核心思路：从通用到专用，逐步收窄领域，避免灾难性遗忘。

```
官方预训练权重 (encoder_msg_cddd.pt)
         ↓ 第一阶段：泛化性微调（当前优先任务）
SpectraVerse 真实谱微调
正负离子分开训练，各自独立权重
         ↓
encoder_sv_neg.pt          encoder_sv_pos.pt
         ↓ 第二阶段：中药领域适应
TCM 化合物训练（FIORA 模拟谱为主，真实谱补充）
         ↓
encoder_tcm_neg.pt         encoder_tcm_pos.pt
         ↓ 第三阶段（可选）：LoRA 类别适配器
黄酮 / 生物碱 / 萜类 / 皂苷 各自独立 LoRA 层
         ↓
encoder_tcm_neg.pt + lora_flavonoid_neg.pt
encoder_tcm_neg.pt + lora_alkaloid_neg.pt
...
```

### 为什么不训练类别专属编码器？

直觉上，为黄酮、生物碱等各自训练独立编码器似乎更精准，但存在一个根本性的**推理时路由问题**：

> 对于未知化合物，我们无法事先知道它属于哪个类别，因此无法选择对应的编码器。

LoRA 适配器方案解决了这个问题：
- 主编码器（`encoder_tcm_neg.pt`）处理所有化合物，保留通用表征能力
- LoRA 层参数量极小（< 1% 主编码器参数），可在推理时按需加载/切换
- 当类别已知时（如已知样品为黄酮类），加载对应 LoRA 层提升精度
- 当类别未知时，直接使用主编码器，不损失通用性

### 三阶段权重继承关系

| 阶段 | 输入权重 | 训练数据 | 输出权重 |
|------|---------|---------|---------|
| 第一阶段 | `encoder_msg_cddd.pt`（官方） | SV neg 129,166 条（真实谱） | `encoder_sv_neg.pt` |
| 第一阶段 | `encoder_msg_cddd.pt`（官方） | SV pos 139,693 条（真实谱） | `encoder_sv_pos.pt` |
| 第二阶段 | `encoder_sv_neg.pt` | TCM neg（FIORA 模拟 + 真实谱） | `encoder_tcm_neg.pt` |
| 第二阶段 | `encoder_sv_pos.pt` | TCM pos（FIORA 模拟 + 真实谱） | `encoder_tcm_pos.pt` |
| 第三阶段 | `encoder_tcm_neg.pt`（冻结） | 黄酮/生物碱/萜类/皂苷（各类别） | `lora_flavonoid_neg.pt` 等 |

> **当前优先任务**：第一阶段负离子微调（补齐负离子短板）。训练脚本已就绪：
> `trainers/denovo/2_phase/scripts/training/run_train_mist_sv.sh`（`MODE=neg`）

---

## 第二阶段：中药领域适应训练

### 2.1 背景与目标

**问题分析**：
- MSFlow官方训练数据（CANOPUS、MassSpecGym）为通用代谢物/药物数据集
- 中药成分具有独特的结构骨架（黄酮、生物碱、萜类、皂苷等）

**目标**：在第一阶段 SV 权重基础上，用 TCM 化合物数据进行领域适应，提升编码器对中药成分的表征能力

### 2.2 数据源规划

#### 2.2.1 中药化合物数据源

| 数据源 | 路径 | 化合物数量 | 特点 |
|--------|------|-----------|------|
| HERB数据库 | `/stor1/AIMS4Meta/databases/TCM_databases/herb/HERB_ingredient_info_v1.txt` | ~49,258 | 中药成分-靶点关联，`Ingredient_Smile` 列 |
| TCMBank | `/stor1/AIMS4Meta/databases/TCM_databases/tcmbank/ingredient_all.xlsx` | 61,966 | 中药复方信息，`Smiles` 列 |

> **不纳入 LOTUS**：LOTUS 是通用天然产物库，不是中药专用，不纳入 TCM 训练集。

#### 2.2.2 质谱数据生成流程

```
化合物 SMILES (HERB/TCMBank)
         ↓ RDKit 计算 InChIKey
与 SpectraVerse InChIKey 比对
         ↓
┌────────────────┬──────────────────────┐
│ SV 中已有真实谱  │  SV 中无真实谱        │
│ → 直接使用      │  → FIORA 模拟         │
│                │  成功(峰数≥5) → 保留  │
│                │  失败 → 丢弃          │
└────────────────┴──────────────────────┘
```

> **关键决策**：
> - **真实谱优先**：SV 中已有真实谱的 TCM 化合物直接使用，无需模拟
> - **FIORA 唯一模拟工具**：对 SV 之外的 TCM 化合物用 FIORA 模拟；FIORA 失败则直接丢弃
> - **放弃 CFM-ID 兜底**：FIORA 失败通常意味着结构过于复杂，CFM-ID 即使能生成也无法验证可信度，引入这类噪声得不偿失
> - **放弃 LOTUS**：LOTUS 是通用天然产物库，不是中药专用，不纳入 TCM 训练集

### 2.3 训练集构建流程

#### Step 1: 从 SpectraVerse 提取 TCM 子集

```python
from rdkit import Chem
import pandas as pd
from pathlib import Path

# 读取 SV InChIKey 集合
sv_neg = pd.read_csv('/stor1/AIMS4Meta/datasets/msflow_datasets/SpectraVerse/spectraverse_neg/labels.tsv', sep='\t')
sv_pos = pd.read_csv('/stor1/AIMS4Meta/datasets/msflow_datasets/spectraverse_pos/labels.tsv', sep='\t')
sv_inchikeys = set(sv_neg['inchikey'].dropna()) | set(sv_pos['inchikey'].dropna())
print(f"SV 唯一 InChIKey: {len(sv_inchikeys)}")

# 计算 HERB InChIKey（参考 generate_combined_smiles.py）
herb = pd.read_csv('/stor1/AIMS4Meta/databases/TCM_databases/herb/HERB_ingredient_info_v1.txt', sep='\t')
herb_inchikeys = {}
for _, row in herb.iterrows():
    smi = row['Ingredient_Smile']
    if pd.notna(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            ik = Chem.MolToInchiKey(mol)
            herb_inchikeys[ik] = row['Ingredient_id']

# 计算 TCMBank InChIKey
tcmbank = pd.read_excel('/stor1/AIMS4Meta/databases/TCM_databases/tcmbank/ingredient_all.xlsx')
tcmbank_inchikeys = {}
for _, row in tcmbank.iterrows():
    smi = row['Smiles']
    if pd.notna(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            ik = Chem.MolToInchiKey(mol)
            tcmbank_inchikeys[ik] = row['TCMBank_ID']

tcm_inchikeys = set(herb_inchikeys) | set(tcmbank_inchikeys)
print(f"TCM 数据库唯一 InChIKey: {len(tcm_inchikeys)}")

# 取交集（SV 中已有真实谱的 TCM 化合物）
sv_tcm_iks = sv_inchikeys & tcm_inchikeys
# 取差集（TCM 中无真实谱，需要模拟）
tcm_only_iks = tcm_inchikeys - sv_inchikeys

print(f"SV 中已有真实谱的 TCM 化合物: {len(sv_tcm_iks)}")
print(f"需要 FIORA 模拟的 TCM 化合物: {len(tcm_only_iks)}")

# 提取 SV_TCM 子集的谱图记录
sv_tcm_neg = sv_neg[sv_neg['inchikey'].isin(sv_tcm_iks)]
sv_tcm_pos = sv_pos[sv_pos['inchikey'].isin(sv_tcm_iks)]
```

#### Step 2: FIORA 模拟（仅对 SV 之外的 TCM 化合物）

```bash
# 对 tcm_only_iks 对应的 SMILES 运行 FIORA（GPU加速，多碰撞能量 15/30/45 NCE%）
# 接口参考: L2_FIORA_模拟数据库构建.py
fiora-predict \
    -i /stor1/AIMS4Meta/datasets/tcm_training/tcm_only_smiles.csv \
    -o /stor1/AIMS4Meta/datasets/tcm_training/fiora_output.msp \
    --dev cuda:0 --min_prob 0.001
```

```python
# 过滤：只保留峰数 ≥ 5 的模拟谱，FIORA 失败的直接丢弃
def filter_fiora_output(msp_path, min_peaks=5):
    """过滤 FIORA 输出，丢弃峰数不足的谱图"""
    # 解析 MSP，统计每个化合物的峰数
    # 峰数 < min_peaks → 丢弃（不用 CFM-ID 兜底）
    pass
```

| 对比维度 | CFM-ID 4 | FIORA |
|---------|----------|-------|
| 算法原理 | 概率碎片图模型（规则+ML） | 图神经网络键断裂预测 |
| 发表/验证 | 较早，广泛基准测试 | 2025 Nature Communications |
| 碎片准确性 | 中等 | 更高（局部邻域建模） |
| 运行环境 | CPU多进程 | GPU加速 |
| 碰撞能量 | 10V/20V/40V | 15/30/45 NCE% |
| **本方案角色** | **不使用** | **唯一模拟工具** |

#### Step 3: SIRIUS碎片树生成

```bash
# 使用已有的SIRIUS批量处理脚本
python /stor1/AIMS4Meta/trainers/denovo/2_phase/scripts/data_preprocessing/run_sirius_spectraverse.py \
    --input /stor1/AIMS4Meta/datasets/tcm_training/simulated_msp/ \
    --output /stor1/AIMS4Meta/datasets/tcm_training/fragmentation_trees/ \
    --mode both \
    --num_workers 24
```

#### Step 4: 转换为subformulae格式

```bash
python /stor1/AIMS4Meta/trainers/denovo/2_phase/scripts/data_preprocessing/sirius_to_subformulae.py \
    --input /stor1/AIMS4Meta/datasets/tcm_training/fragmentation_trees/ \
    --output /stor1/AIMS4Meta/datasets/tcm_training/subformulae/ \
    --use_running_service \
    --port 8901
```

### 2.4 数据增强策略

#### 2.4.1 结构骨架增强

| 中药成分类型 | 结构特征 | 增强策略 |
|-------------|---------|---------|
| 黄酮类 | C6-C3-C6骨架 | RDA裂解模式增强 |
| 生物碱 | 含氮杂环 | 氮导向裂解建模 |
| 萜类 | 异戊二烯单元 | 中性丢失增强 |
| 皂苷 | 糖链+苷元 | 多步裂解建模 |
| 醌类 | 醌环结构 | 特征碎片增强 |

#### 2.4.2 质谱数据增强（subformulae级别）

> **重要**：MSFlow编码器的实际输入是 **subformulae**（由SIRIUS碎片树提取的分子式列表），
> 而非原始 m/z + intensity。因此数据增强应在 **subformulae 级别** 或
> **SIRIUS处理前的MSP级别** 进行，而非对已提取的subformulae直接操作。

```python
import numpy as np
import copy

# === 方法A: MSP级别增强（SIRIUS处理前） ===
def augment_msp_before_sirius(mz_list, intensity_list, augmentation_type='noise'):
    """在送入SIRIUS生成碎片树之前对MSP原始谱图增强
    
    原理：对同一化合物生成多个微扰动MSP，
    经SIRIUS处理后产生略有差异的subformulae，达到增强效果
    """
    mz = np.array(mz_list, dtype=float)
    intensity = np.array(intensity_list, dtype=float)
    
    if augmentation_type == 'noise':
        # m/z加小扰动（模拟仪器波动，不影响分子式匹配）
        mz += np.random.normal(0, 0.001, len(mz))
        
    elif augmentation_type == 'intensity':
        # 强度扰动 (±15%)，模拟实际仪器采集波动
        intensity *= np.random.uniform(0.85, 1.15, len(intensity))
        
    elif augmentation_type == 'peak_drop':
        # 随机丢弃低强度峰（模拟检测限以下信号丢失）
        low_peaks = intensity < np.percentile(intensity, 20)
        drop_mask = ~(low_peaks & (np.random.random(len(intensity)) < 0.3))
        mz, intensity = mz[drop_mask], intensity[drop_mask]
    
    return mz.tolist(), np.clip(intensity, 0, None).tolist()

# === 方法B: subformulae级别增强（直接操作训练数据） ===
def augment_subformulae(subform_data):
    """对已生成的subformulae JSON进行增强
    
    可安全操作的字段：ms2_inten（强度扰动）
    不可修改的字段：formula, mz, mono_mass（这些是离散分子式信息）
    """
    aug = copy.deepcopy(subform_data)
    intensities = np.array(aug['output_tbl']['ms2_inten'])
    
    # 强度扰动: 只调整相对强度，保留分子式信息不变
    intensities *= np.random.uniform(0.8, 1.2, len(intensities))
    intensities = np.clip(intensities, 0, 1)
    
    # 重新归一化
    if intensities.max() > 0:
        intensities = intensities / intensities.max()
    
    aug['output_tbl']['ms2_inten'] = intensities.tolist()
    return aug
```

### 2.5 训练集质量控制

#### 2.5.1 质量过滤标准

> **注意**：模拟谱经SIRIUS处理后的质量门控应严于真实谱，因为模拟谱已经存在第一层误差，
> 如果SIRIUS解析质量再低，误差会进一步放大。

| 指标 | 阈值 | 说明 |
|------|------|------|
| SIRIUS公式评分 | > **0.8** | 模拟谱需更高阈值（真实谱可用>0.7） |
| 碎片树评分 | > **0.6** | 确保碎片树结构可信 |
| 峰数量 | ≥ 5 | 最少碎片数 |
| 母离子丰度 | > 0 | 母离子存在 |

#### 2.5.2 模拟谱质量预验证

> **关键步骤**：在大规模生成模拟数据前，先对已有真实谱的化合物进行小规模验证。

```python
def validate_simulation_quality(real_subformulae_dir, simulated_subformulae_dir, 
                                 min_cosine=0.5, sample_size=200):
    """模拟谱质量预验证
    
    选择一批同时有真实谱和模拟谱的化合物，
    比较其经SIRIUS处理后的subformulae差异，确认模拟质量达标
    """
    from matchms import Spectrum
    from matchms.similarity import ModifiedCosine
    
    real_files = {f.stem: f for f in Path(real_subformulae_dir).glob('*.json')}
    sim_files = {f.stem: f for f in Path(simulated_subformulae_dir).glob('*.json')}
    
    common = set(real_files.keys()) & set(sim_files.keys())
    if len(common) < sample_size:
        print(f"警告: 仅找到 {len(common)} 个共同化合物（目标 {sample_size}）")
    
    scores = []
    for name in list(common)[:sample_size]:
        real = json.load(open(real_files[name]))
        sim = json.load(open(sim_files[name]))
        
        # 比较subformulae中的分子式重合度
        real_formulas = set(real['output_tbl']['formula'])
        sim_formulas = set(sim['output_tbl']['formula'])
        jaccard = len(real_formulas & sim_formulas) / max(len(real_formulas | sim_formulas), 1)
        scores.append(jaccard)
    
    avg_score = np.mean(scores)
    print(f"模拟谱质量验证: 平均分子式Jaccard相似度 = {avg_score:.3f}")
    print(f"  样本数: {len(scores)}, 中位数: {np.median(scores):.3f}")
    
    if avg_score < min_cosine:
        print(f"  ⚠️ 质量不达标（阈值 {min_cosine}），建议检查模拟参数")
        return False
    return True
```

#### 2.5.3 去重策略

```python
# 基于InChIKey去重
def deduplicate_by_inchikey(data_list):
    """基于InChIKey去重，保留质量最高的记录"""
    seen = {}
    for item in data_list:
        key = item['inchikey'].split('-')[0]  # 使用第一部分
        if key not in seen or item['score'] > seen[key]['score']:
            seen[key] = item
    return list(seen.values())
```

### 2.6 预期训练集规模

| 数据类型 | 目标数量 | 当前状态 |
|---------|---------|---------|
| 中药成分（FIORA 模拟） | 50,000+ | 需生成 |
| 正负离子模式 | 各50% | 需平衡 |
| 碰撞能量分布 | 10-60 eV | 需覆盖 |

---

## 第一阶段：SpectraVerse 泛化性微调

### 1.1 SpectraVerse数据集概述

#### 1.1.1 数据规模

| 离子模式 | subformulae数量 | 路径 |
|---------|----------------|------|
| 负离子 | 129,166 | `/stor1/AIMS4Meta/datasets/msflow_datasets/SpectraVerse/spectraverse_neg/subformulae/` |
| 正离子 | 139,693 | `/stor1/AIMS4Meta/datasets/msflow_datasets/spectraverse_pos/subformulae/` |
| **总计** | **268,859** | - |

#### 1.1.2 数据特点

- **来源**：SpectraVerse数据库（高质量实验质谱）
- **覆盖范围**：天然产物、代谢物、药物
- **数据格式**：MSFlow兼容的subformulae JSON
- **字段完整**：mz, ms2_inten, mono_mass, abs_mass_diff, mass_diff, formula, ions

#### 1.1.3 编码器如何使用 subformulae 字段

> 编码器（`FormulaTransformer`）实际只使用 subformulae JSON 中的三个字段，其余均忽略：
>
> | 字段 | 是否使用 | 说明 |
> |------|---------|------|
> | `formula` | ✓ | 每个碎片的分子式，转为向量后送入 Transformer |
> | `ms2_inten` | ✓ | 峰强度，**两处使用**（见下） |
> | `ions` | ✓ | 离子类型（如 `[M+H]+`），one-hot 编码 |
> | `mz` | ✗ | 未使用（分子式已隐含质量信息） |
> | `mono_mass`, `abs_mass_diff`, `mass_diff` | ✗ | 未使用 |
>
> **`ms2_inten` 的两处作用**（代码来源：`modules.py`）：
>
> 1. **输入特征**（`modules.py:169-199`）：强度值被拼接进每个峰的特征向量，与分子式向量、离子类型、加合物一起送入 Transformer。支持多种变换（`float`/`log`/`cat`/`zero`），默认 `float`。
>
> 2. **加权池化**（`modules.py:261-265`，`set_pooling="intensity"`）：Transformer 输出后，用归一化强度对所有峰的表示做加权平均，得到最终谱图嵌入。高强度峰对嵌入贡献更大。
>
> 因此，强度信息既影响每个峰的表示，也决定哪些峰主导最终嵌入——与 DreAMS 等工具考虑峰强度的思路一致，只是 MIST 在 SIRIUS 解析后的 subformulae 层面操作，而非原始 m/z 层面。

### 1.2 微调策略设计

#### 1.2.1 微调 vs 从头训练

| 策略 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| 从头训练 | 完全定制 | 需大量数据 | 领域差异极大 |
| 微调（推荐） | 保留通用知识 | 可能过拟合 | 领域相关性强 |
| 混合训练 | 平衡通用与专精 | 超参调优复杂 | 最佳效果 |

#### 1.2.2 基线权重选择

官方提供两套编码器权重，应先评估再选择：

| 权重 | 训练数据 | 模型规模 | 建议 |
|------|---------|---------|------|
| `encoder_msg_cddd.pt` | MassSpecGym (MSG) | hidden=256/512，较小 | 微调灵活、收敛快 |
| `encoder_canopus_cddd.pt` | CANOPUS | hidden=512/2048，较大 | 容量大、表征丰富 |

> **建议**：在 SpectraVerse 测试集上先评估两套权重的零样本性能（CDDD 余弦相似度），
> 选择更优的作为微调基线。若无明显差异，优先选 CANOPUS 权重（模型容量更大）。

#### 1.2.3 推荐微调方案

```
基线权重选择 (先评估 MSG vs CANOPUS)
         ↓
    SpectraVerse 真实数据微调（正负离子分开，各自独立权重）
         ↓
encoder_sv_neg.pt / encoder_sv_pos.pt
         ↓ 进入第二阶段
    中药领域适应训练
```

> **核心原则**：
> - 真实数据为主，模拟数据为辅（见第4章详细论证）
> - 编码器输出必须保持 **512维**，不得修改输出维度，否则破坏与解码器的兼容性

### 1.3 微调流程详解

#### Step 1: 数据准备

> **注意**：subformulae JSON 文件只含 `cand_form`、`cand_ion`、`output_tbl` 等质谱字段，
> **不含 InChIKey**。InChIKey 在 `labels.tsv` 中（列：`spec, name, formula, smiles, inchikey, ionization, precursor_mz, collision_energy`）。
> 划分时应读取 TSV，输出 split TSV（不复制 JSON 文件，避免 40 万+文件的磁盘开销）。

```python
# 数据集划分脚本（基于 TSV，不复制 JSON）
import pandas as pd
import random
from pathlib import Path
from collections import defaultdict

def split_spectraverse_dataset(labels_tsv, subformulae_dir, output_dir,
                                train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """按 InChIKey 骨架层分组划分 SpectraVerse 数据集

    关键：同一化合物可能有多条谱图（不同碰撞能、不同离子模式），
    必须按 InChIKey 第一段（骨架层）分组，确保同一化合物的所有谱图
    要么全在 train，要么全在 val/test，防止数据泄漏。

    输出：split_train.tsv / split_val.tsv / split_test.tsv（引用原始 JSON 路径，不复制文件）
    """
    df = pd.read_csv(labels_tsv, sep='\t')
    df['subformulae_path'] = df['spec'].apply(
        lambda s: str(Path(subformulae_dir) / f"{s}.json")
    )

    # 按 InChIKey 骨架层（第一段，14字符）分组
    df['skeleton'] = df['inchikey'].fillna('').str.split('-').str[0]

    inchikey_groups = defaultdict(list)
    ungrouped_idx = []
    for idx, row in df.iterrows():
        if row['skeleton'] and len(row['skeleton']) == 14:
            inchikey_groups[row['skeleton']].append(idx)
        else:
            ungrouped_idx.append(idx)

    # 按 InChIKey 组划分
    group_keys = list(inchikey_groups.keys())
    random.seed(42)
    random.shuffle(group_keys)

    n_groups = len(group_keys)
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)

    split_idx = {
        'train': [i for k in group_keys[:n_train] for i in inchikey_groups[k]] + ungrouped_idx,
        'val':   [i for k in group_keys[n_train:n_train+n_val] for i in inchikey_groups[k]],
        'test':  [i for k in group_keys[n_train+n_val:] for i in inchikey_groups[k]],
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for split_name, indices in split_idx.items():
        out_path = Path(output_dir) / f"split_{split_name}.tsv"
        df.loc[indices].to_csv(out_path, sep='\t', index=False)
        print(f"  {split_name}: {len(indices)} 条 → {out_path}")

    print(f"数据集划分完成 (按 InChIKey 骨架层分组):")
    print(f"  化合物组: {n_groups} 个唯一骨架")
    if ungrouped_idx:
        print(f"  无InChIKey（归入train）: {len(ungrouped_idx)} 条")

# 执行划分（负离子）
split_spectraverse_dataset(
    labels_tsv='/stor1/AIMS4Meta/datasets/msflow_datasets/SpectraVerse/spectraverse_neg/labels.tsv',
    subformulae_dir='/stor1/AIMS4Meta/datasets/msflow_datasets/SpectraVerse/spectraverse_neg/subformulae/',
    output_dir='/stor1/AIMS4Meta/datasets/msflow_datasets/SpectraVerse/spectraverse_neg/splits/'
)

# 执行划分（正离子）
split_spectraverse_dataset(
    labels_tsv='/stor1/AIMS4Meta/datasets/msflow_datasets/spectraverse_pos/labels.tsv',
    subformulae_dir='/stor1/AIMS4Meta/datasets/msflow_datasets/spectraverse_pos/subformulae/',
    output_dir='/stor1/AIMS4Meta/datasets/msflow_datasets/spectraverse_pos/splits/'
)
```

#### Step 2: CDDD表示生成

MSFlow编码器输出为CDDD嵌入（512维），需要预先计算目标CDDD：

```python
# 使用CDDD模型计算分子嵌入
from cddd import CDDDModel

cddd_model = CDDDModel()

def compute_cddd_embeddings(smiles_list, output_path):
    """计算SMILES的CDDD嵌入"""
    embeddings = []
    for smiles in smiles_list:
        try:
            emb = cddd_model.predict(smiles)
            embeddings.append(emb)
        except:
            embeddings.append(None)

    import pandas as pd
    df = pd.DataFrame({
        'smiles': smiles_list,
        'cddd': embeddings
    })
    df.to_parquet(output_path)
    return df
```

> **CDDD 值域与编码器输出对齐（重要）**：
>
> CDDD 模型最后一层使用 **sigmoid 激活**，输出值域严格为 **[0, 1]**。
> 编码器输出层必须同样使用 sigmoid（或在输出后 clamp 到 [0, 1]），
> 否则 MSE 损失会因值域不匹配（如 tanh 输出 [-1,1]）导致训练不稳定。
>
> 验证方法（计算 CDDD 嵌入后立即检查）：
> ```python
> import numpy as np
> emb_array = np.stack([e for e in embeddings if e is not None])
> assert emb_array.min() >= 0.0 and emb_array.max() <= 1.0, \
>     f"CDDD 值域异常: [{emb_array.min():.4f}, {emb_array.max():.4f}]，期望 [0, 1]"
> print(f"✓ CDDD 值域正常: [{emb_array.min():.4f}, {emb_array.max():.4f}]")
> ```
>
> 若使用官方 DiffMS/MIST 编码器权重，其输出层已包含 sigmoid，无需额外修改。
> 若自定义编码器架构，需在 `output_size=512` 的线性层后添加 `nn.Sigmoid()`。

#### Step 3: 损失函数选择

编码器的监督信号是 CDDD 嵌入（512维向量），损失函数的选择直接影响训练效果：

| 损失函数 | 特点 | 适用场景 |
|---------|------|----------|
| MSE Loss | 逐元素回归，对异常值敏感 | 默认选择，要求精确匹配各维度 |
| Cosine Similarity Loss | 只关注方向、忽略幅度 | CDDD 语义与方向相关性更强时 |
| MSE + Cosine 混合 | 兼顾精确度和方向一致性 | **推荐**，多数嵌入回归任务的最佳实践 |
| Contrastive Loss | 拉近同化合物、推远不同化合物 | 多条谱图对应同一化合物时的辅助损失 |

> **建议**：使用 **MSE + Cosine 混合损失**，并在实验中对比。
> **需要查证**：DiffMS/MIST 原始代码具体使用的损失函数，建议先复用官方实现再调优。

```python
import torch
import torch.nn as nn

class CombinedEmbeddingLoss(nn.Module):
    """编码器训练损失函数: MSE + Cosine Similarity"""
    def __init__(self, mse_weight=1.0, cosine_weight=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineSimilarity(dim=-1)
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
    
    def forward(self, pred_cddd, target_cddd):
        mse_loss = self.mse(pred_cddd, target_cddd)
        cosine_loss = 1.0 - self.cosine(pred_cddd, target_cddd).mean()  # 1 - cos_sim
        return self.mse_weight * mse_loss + self.cosine_weight * cosine_loss
```

#### Step 4: 微调配置

> **重要说明**：MSFlow 是两阶段架构，编码器和解码器使用**不同的训练脚本**：
> - **编码器**：使用 DiffMS/MIST 框架训练（`train_mist.py`），输入是 subformulae，监督信号是 CDDD 嵌入
> - **解码器**：使用 MSFlow 框架训练（`cfg_pretrain.py`），输入是 CDDD，生成 SMILES
> - 解码器权重可直接复用，无需重训。本方案只训练编码器。

修改 DiffMS 编码器配置（参考 `configs/spectraverse.yaml`）：

```yaml
# 编码器微调配置 (DiffMS/MIST 框架)
encoder:
  model_name: "SpectraEncoderGrowing"
  form_embedder: "peakformula"    # 使用 SIRIUS subformulae
  output_size: 512                # CDDD 维度（不得修改！解码器依赖此维度）
  hidden_size: 512                # 隐藏层维度
  spectra_dropout: 0.1
  top_layers: 2
  refine_layers: 3

  # 微调关键参数
  learning_rate: 1e-5             # 较小学习率（防止灾难性遗忘）
  epochs: 50                      # 微调轮数（比从头训练少）
  weight_decay: 0.01
  batch_size: 64
  
  # 早停 + 渐进解冻策略
  early_stopping:
    monitor: "val_loss"
    patience: 5
    mode: "min"
  
  # 渐进式解冻：前10 epoch冻结底层，仅训练顶层
  freeze_strategy:
    freeze_epochs: 10              # 前10 epoch冻结FormulaTransformer
    unfreeze_lr_factor: 0.1        # 解冻后底层用更小学习率
```

微调数据配置：

```yaml
# 数据路径配置
data:
  # 使用合并后的 TSV（含 subformulae_dir 列，指向各自原始目录）
  labels_tsv: '/stor1/AIMS4Meta/datasets/msflow_datasets/spectraverse_combined_labels.tsv'
  cddd_embeddings: '/stor1/AIMS4Meta/datasets/msflow_datasets/spectraverse_cddd_embeddings.parquet'

  split:
    train: 0.8
    val: 0.1
    test: 0.1
  random_seed: 42
```

#### Step 5: 执行微调

```bash
#!/bin/bash
# finetune_encoder.sh
# 注意：编码器使用 DiffMS/MIST 框架训练，不是 cfg_pretrain.py

# 加载环境
conda activate flow

# 设置GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 阶段1：编码器微调（从官方权重初始化）
cd /stor1/AIMS4Meta/code/DiffMS-master
python src/mist/train_mist.py \
    --config /stor1/AIMS4Meta/trainers/denovo/2_phase/configs/spectraverse.yaml \
    --resume_from /stor1/AIMS4Meta/code/MSFlow-main/checkpoints/MSFlow/Encoder/encoder_msg_cddd.pt \
    --learning_rate 1e-5

# 解码器无需重训，直接复用官方权重
```

> **关键区分**：
> - `cfg_pretrain.py` 是 **解码器**（CondFlowMolBERT）的训练脚本，不是编码器的
> - `train_mist.py` 是 **编码器**（SpectraEncoderGrowing）的训练脚本
> - 解码器已基于大规模分子数据库（ChEMBL/MOSES）充分训练，学习了 CDDD→SMILES 的通用映射，权重可直接复用

### 1.4 微调监控与评估

#### 1.4.1 训练过程监控指标

| 指标 | 说明 | 目标值 | 阶段 |
|------|------|--------|------|
| train_loss | 训练损失 (MSE+Cosine) | 持续下降 | 训练过程 |
| val_loss | 验证损失 | 低于初始值 | 训练过程 |
| cddd_cosine_sim | 预测CDDD与真实CDDD的余弦相似度 | > 0.85 | 训练+评估 |
| cddd_mse | 预测CDDD与真实CDDD的均方误差 | 持续下降 | 训练+评估 |
| retrieval_top1 | 用预测CDDD在候选库中检索Top-1准确率 | > 0.3 | 评估 |
| retrieval_top10 | 用预测CDDD在候选库中检索Top-10准确率 | > 0.7 | 评估 |

#### 1.4.2 端到端评估指标（可选，需解码器参与）

| 指标 | 说明 | 目标值 |
|------|------|--------|
| top1_accuracy | 编码器+解码器端到端 Top-1 | > 0.2 |
| top10_accuracy | 编码器+解码器端到端 Top-10 | > 0.5 |
| tanimoto_similarity | 生成分子与真实分子的 Tanimoto 相似度 | > 0.6 |

### 1.5 正负离子分开训练

#### 1.5.1 数据合并策略（both 模式）

> **注意**：正负离子数据共 268,859 条 JSON 文件，文件复制方式极慢且浪费磁盘。
> 推荐 **TSV 合并方案**：只合并 labels.tsv，添加 `subformulae_dir` 列指向各自原始目录，
> 训练时 DataLoader 按 `subformulae_dir/spec.json` 路径读取，无需移动任何文件。

```python
import pandas as pd
from pathlib import Path

NEG_SUBFORMULAE_DIR = '/stor1/AIMS4Meta/datasets/msflow_datasets/SpectraVerse/spectraverse_neg/subformulae/'
POS_SUBFORMULAE_DIR = '/stor1/AIMS4Meta/datasets/msflow_datasets/spectraverse_pos/subformulae/'

def merge_pos_neg_tsv(neg_tsv, pos_tsv, output_tsv):
    """合并正负离子 TSV，添加 subformulae_dir 列指向各自目录

    不复制 JSON 文件，DataLoader 按 subformulae_dir + spec + '.json' 读取
    """
    neg_df = pd.read_csv(neg_tsv, sep='\t')
    neg_df['subformulae_dir'] = NEG_SUBFORMULAE_DIR

    pos_df = pd.read_csv(pos_tsv, sep='\t')
    pos_df['subformulae_dir'] = POS_SUBFORMULAE_DIR

    merged = pd.concat([neg_df, pos_df], ignore_index=True)
    merged.to_csv(output_tsv, sep='\t', index=False)
    print(f"合并完成: {len(neg_df)} 负离子 + {len(pos_df)} 正离子 = {len(merged)} 条")
    print(f"输出: {output_tsv}")

# 执行合并
merge_pos_neg_tsv(
    neg_tsv='/stor1/AIMS4Meta/datasets/msflow_datasets/SpectraVerse/spectraverse_neg/labels.tsv',
    pos_tsv='/stor1/AIMS4Meta/datasets/msflow_datasets/spectraverse_pos/labels.tsv',
    output_tsv='/stor1/AIMS4Meta/datasets/msflow_datasets/spectraverse_combined_labels.tsv'
)
```

#### 1.5.2 离子模式感知（保持512维输出不变）

> **核心约束**：编码器输出必须严格保持 **512维**，这是解码器（CondFlowMolBERT）
> 的 `COND_DIM=512` 输入约束。改变输出维度会直接破坏与解码器的兼容性。
>
> 离子模式信息应在编码器**内部**融合，而非改变输出维度。

```python
# 方案A：在FormulaTransformer输入层融合离子模式（推荐）
class SpectraEncoderIonAware(nn.Module):
    """512维输出不变，离子模式在内部融合"""
    def __init__(self, base_encoder, ion_embed_dim=16):
        super().__init__()
        self.base_encoder = base_encoder
        # 离子模式嵌入添加到内部特征空间
        self.ion_embedding = nn.Embedding(2, ion_embed_dim)  # 0=positive, 1=negative
        # 投影层：将 hidden_size+ion_embed_dim 映射回 hidden_size
        hidden_size = base_encoder.spectra_encoder[0].hidden_size  # 假设可访问
        self.ion_proj = nn.Linear(hidden_size + ion_embed_dim, hidden_size)
        
    def forward(self, batch, ion_mode_idx):
        """ion_mode_idx: 0=positive, 1=negative"""
        # 编码器主体前向
        encoder_output, aux_out = self.base_encoder.spectra_encoder[0](batch, return_aux=True)
        
        # 融合离子模式信息到内部表示
        B = encoder_output.size(0)
        ion_emb = self.ion_embedding(ion_mode_idx)  # [B, ion_embed_dim]
        ion_emb = ion_emb.unsqueeze(1).expand(-1, encoder_output.size(1), -1) if encoder_output.dim() == 3 else ion_emb
        fused = torch.cat([encoder_output, ion_emb], dim=-1)
        encoder_output = self.ion_proj(fused)
        
        # 后续层保持不变，最终输出仍为 512 维
        pred_frag_fps = self.base_encoder.spectra_encoder[1](aux_out["peak_tensor"])
        output = self.base_encoder.spectra_encoder[2](encoder_output)
        
        return output, {"pred_frag_fps": pred_frag_fps, "h0": encoder_output}

# 方案B：更简单的方式——直接用投影层确保512维输出
class SpectraEncoderIonProjection(nn.Module):
    """ion mode信息拼接后用投影层压回512维"""
    def __init__(self, base_encoder, ion_embed_dim=32):
        super().__init__()
        self.base_encoder = base_encoder
        self.ion_embedding = nn.Embedding(2, ion_embed_dim)
        # 512 + 32 → 512 的投影层
        self.output_proj = nn.Sequential(
            nn.Linear(512 + ion_embed_dim, 512),
            nn.Sigmoid()  # 保持与原始编码器输出范围一致
        )
        
    def forward(self, batch, ion_mode_idx):
        output, aux_outputs = self.base_encoder(batch)
        # output: [B, 512]
        ion_emb = self.ion_embedding(ion_mode_idx)  # [B, 32]
        fused = torch.cat([output, ion_emb], dim=-1)  # [B, 544]
        output = self.output_proj(fused)  # [B, 512] ← 保持512维
        return output, aux_outputs
```

### 2.6 领域自适应微调（第一阶段 → 第二阶段衔接）

```
encoder_sv_neg.pt / encoder_sv_pos.pt
         ↓
    中药化合物数据微调（第二阶段）
         ↓
encoder_tcm_neg.pt / encoder_tcm_pos.pt
```

---

## 模拟数据误差传播分析与缓解策略

### 3.1 误差传播链路

第二阶段使用模拟数据训练存在三重误差累积，必须充分认识并采取缓解措施：

```
第一层: 模拟谱误差          第二层: SIRIUS解析误差         第三层: 域漂移
──────────────────        ──────────────────         ─────────────

SMILES → FIORA            模拟MSP → SIRIUS碎片树      训练: 模拟谱的碎片模式
• 缺失峰、错误强度       • SIRIUS基于真实谱训练     • 推理: 真实谱的碎片模式
• 缺少重排反应产物     • 对模拟谱解析质量降低   • 域差异导致性能下降
• 无真实仪器噪声       • 同位素模式不自然      • 可能比原始权重更差
         ↓                        ↓                       ↓
   累积误差 ────────→ 累积误差 ────────→ 累积误差
```

### 3.2 各层误差详解

| 误差层 | 来源 | 影响 | 严重程度 |
|--------|------|------|--------|
| 第一层 | FIORA预测缺陷 | 缺失真实碎片、产生伪峰 | ★★★ |
| 第二层 | SIRIUS对模拟谱解析偏差 | 碎片树/分子式质量下降 | ★★★★ |
| 第三层 | 训练vs推理谱图分布差异 | 编码器学到偏差表征 | ★★★★★ |

### 3.3 缓解策略

#### 3.3.1 严格质量门控（已在 1.5.1 配置）

模拟数据的SIRIUS碎片树质量阈值应严于真实数据：
- 分子式评分 > 0.8（而非通常0.7）
- 碎片树评分 > 0.6（而非通常0.5）

#### 3.3.2 小规模预验证（已在 1.5.2 配置）

大规模生成前，先对已有真实谱的化合物做对比实验，量化模拟谱经SIRIUS处理后的subformulae质量。

---

## 真实与模拟数据混合训练策略

### 4.1 核心原则：真实数据为主，模拟数据为辅

> 模拟数据存在固有误差（见第3章分析），因此不应作为训练主体。
> 策略：**SpectraVerse 真实数据（278,810条）为主体**，模拟数据仅作补充，
> 混合比例不超过 30%。

### 4.2 混合比例建议

| 训练阶段 | 真实数据比例 | 模拟数据比例 | 说明 |
|---------|-----------|-----------|------|
| 第一阶段（SV微调） | 100% | 0% | 纯真实数据建立基线 |
| 第二阶段（TCM适应） | ≥70% | ≤30% | 模拟数据仅填充覆盖缺口 |

### 4.3 混合数据集构建

> **注意**：同样采用 TSV 合并方案，不复制 JSON 文件。
> 模拟数据 TSV 中添加 `data_source=simulated` 列，训练时通过样本加权区分。

```python
def build_mixed_dataset_tsv(real_tsv, real_subformulae_dir,
                             sim_tsv, sim_subformulae_dir,
                             sim_ratio=0.3, output_tsv=None, seed=42):
    """构建真实+模拟混合数据集（TSV合并，不复制文件）

    Args:
        real_tsv: 真实数据 labels.tsv
        real_subformulae_dir: 真实数据 subformulae 目录
        sim_tsv: 模拟数据 labels.tsv
        sim_subformulae_dir: 模拟数据 subformulae 目录
        sim_ratio: 模拟数据占比上限（默认0.3）
        output_tsv: 输出 TSV 路径
    """
    real_df = pd.read_csv(real_tsv, sep='\t')
    sim_df = pd.read_csv(sim_tsv, sep='\t')

    # 计算模拟数据上限
    max_sim = int(len(real_df) * sim_ratio / (1 - sim_ratio))
    if len(sim_df) > max_sim:
        sim_df = sim_df.sample(max_sim, random_state=seed)

    real_df['subformulae_dir'] = real_subformulae_dir
    real_df['data_source'] = 'real'
    sim_df['subformulae_dir'] = sim_subformulae_dir
    sim_df['data_source'] = 'simulated'

    mixed = pd.concat([real_df, sim_df], ignore_index=True)
    actual_ratio = len(sim_df) / len(mixed)

    print(f"真实数据: {len(real_df)} 条")
    print(f"模拟数据: {len(sim_df)} 条 (占比 {actual_ratio:.1%})")
    print(f"总计: {len(mixed)} 条")

    if output_tsv:
        mixed.to_csv(output_tsv, sep='\t', index=False)
        print(f"输出: {output_tsv}")

    return mixed
```

### 4.4 实验对比设计

建议设计对照实验，量化模拟数据的实际贡献：

| 实验组 | 训练数据 | 目的 |
|--------|---------|------|
| 基线组 | 官方预训练权重（不微调） | 参照 |
| 实验A | 仅SpectraVerse真实数据 | 真实数据微调效果 |
| 实验B | SpectraVerse + 30%模拟 | 模拟数据是否有帮助 |
| 实验C | 仅模拟数据 | 模拟数据单独训练的上限 |

评估指标：在相同测试集（真实谱）上比较 CDDD 余弦相似度、检索 Top-1/Top-10 准确率和 Tanimoto 相似度。

---

## 进阶优化策略（可选）

> 以下策略可在基线方案稳定后逐步引入，不建议一开始全部采用。

### 5.1 对比学习辅助任务

**动机**：SpectraVerse 中同一化合物可能有多条谱图（不同碰撞能、不同离子模式），这些谱图在嵌入空间中应该彼此接近。

> **注意**：同一 InChIKey 的谱图在不同碰撞能下 subformulae 差异可能很大（低能量碎片少、高能量碎片多），
> 直接将所有同 InChIKey 谱图作为正样本对会引入噪声。
> 建议加 **Jaccard 相似度阈值**过滤，只保留 formula 重合度 ≥ 0.3 的谱图对作为正样本。

```python
import torch
import torch.nn.functional as F

def compute_formula_jaccard_matrix(subformulae_list):
    """计算 batch 内所有谱图对的 formula Jaccard 相似度矩阵

    Args:
        subformulae_list: list of list[str]，每条谱图的 formula 列表
    Returns:
        jaccard_matrix: [B, B] tensor
    """
    B = len(subformulae_list)
    formula_sets = [set(f) for f in subformulae_list]
    matrix = torch.zeros(B, B)
    for i in range(B):
        for j in range(B):
            inter = len(formula_sets[i] & formula_sets[j])
            union = len(formula_sets[i] | formula_sets[j])
            matrix[i, j] = inter / union if union > 0 else 0.0
    return matrix


def contrastive_loss(embeddings, labels, subformulae_list=None,
                     temperature=0.1, min_jaccard=0.3):
    """监督对比损失：同一InChIKey且Jaccard≥min_jaccard的谱图互为正样本对

    Args:
        embeddings: [B, 512] CDDD嵌入
        labels: [B] 化合物ID（同一InChIKey对应相同ID）
        subformulae_list: list of list[str]，用于 Jaccard 过滤（可选）
        temperature: 温度参数
        min_jaccard: 正样本对的最低 formula Jaccard 相似度（默认0.3）
    """
    sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
    sim_matrix = sim_matrix / temperature

    # 正样本掩码：同一化合物的不同谱图
    positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    positive_mask.fill_diagonal_(0)  # 排除自身

    # Jaccard 质量过滤：只保留 formula 重合度足够高的正样本对
    if subformulae_list is not None:
        jaccard_matrix = compute_formula_jaccard_matrix(subformulae_list).to(embeddings.device)
        quality_mask = (jaccard_matrix >= min_jaccard).float()
        positive_mask = positive_mask * quality_mask

    # InfoNCE
    exp_sim = torch.exp(sim_matrix)
    exp_sim.fill_diagonal_(0)

    pos_sum = (exp_sim * positive_mask).sum(dim=1)
    all_sum = exp_sim.sum(dim=1)

    loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)
    valid = positive_mask.sum(dim=1) > 0
    return loss[valid].mean() if valid.any() else torch.tensor(0.0, device=embeddings.device)
```

**用法**：作为辅助损失，与主损失（MSE+Cosine）加权混合：

```
total_loss = main_loss + 0.1 * contrastive_loss
```

### 5.2 模拟数据样本加权

**动机**：比硬性 30% 比例限制更灵活，通过降低模拟数据的损失权重来减弱噪声影响。

```python
def weighted_loss(pred, target, data_source, sim_weight=0.5):
    """根据数据来源加权
    
    Args:
        data_source: 'real' or 'simulated' 标记
        sim_weight: 模拟数据的损失权重 (默认0.5，真实数据为1.0)
    """
    per_sample_loss = F.mse_loss(pred, target, reduction='none').mean(dim=-1)
    weights = torch.where(
        data_source == 'real',
        torch.ones_like(per_sample_loss),
        torch.full_like(per_sample_loss, sim_weight)
    )
    return (per_sample_loss * weights).mean()
```

### 5.3 碎片树拓扑信息利用（实验性）

> **背景**：当前编码器只用了碎片节点的分子式列表（展平的 subformulae），
> **忽略了碎片树的父子关系和 neutral loss 信息**。
> 这部分信息反映了分子裂解路径，对结构解析有潜在价值。

**潜在方案**：
- 将 neutral loss 作为额外特征拼接到 subformulae 输入
- 使用图神经网络 (GNN) 编码碎片树结构
- 将树结构作为 Transformer 的注意力掩码

> **注意**：这需要修改编码器架构，工作量较大，建议在基线方案稳定后再尝试。
> 目前 **需要查证** DiffMS/MIST 架构是否支持注入树结构信息。

---

## 实施路线图

> 按三阶段层级方案推进，每阶段完成后评估再进入下一阶段。

### 第一阶段：SpectraVerse 泛化性微调（当前任务）

**目标**：补齐负离子短板，建立正负离子各自的通用编码器基线。

| 任务 | 优先级 | 说明 |
|------|--------|------|
| 负离子数据划分 | **最高** | `split_dataset.py --mode neg`，已就绪 |
| 负离子编码器微调 | **最高** | `run_train_mist_sv.sh`（`MODE=neg`），从 `encoder_msg_cddd.pt` 初始化 |
| 正离子编码器微调 | 高 | `run_train_mist_sv.sh`（`MODE=pos`），与负离子并行或顺序执行 |
| 评估两套权重 | 高 | 在各自测试集上比较 val_loss、cosine 相似度 |

**输出**：`checkpoints/mist_sv_neg/` → `encoder_sv_neg.pt`，`checkpoints/mist_sv_pos/` → `encoder_sv_pos.pt`

---

### 第二阶段：中药领域适应训练

**目标**：在第一阶段权重基础上，用 TCM 化合物数据进行领域适应。

| 任务 | 优先级 | 说明 |
|------|--------|------|
| 提取 SV 中 TCM 子集 | **最高** | 用 RDKit 计算 HERB/TCMBank SMILES 的 InChIKey，与 SV 取交集，直接使用真实谱 |
| FIORA 模拟谱生成 | **最高** | 对 SV 之外的 TCM 化合物，GPU 加速，多碰撞能量（15/30/45 NCE%）；失败直接丢弃 |
| 模拟谱质量预验证 | **最高** | 200 个有真实谱的化合物对照，Jaccard ≥ 0.3 才通过 |
| SIRIUS 碎片树生成 + 质量门控 | 高 | 分子式评分 > 0.8，碎片树评分 > 0.6 |
| TCM 编码器微调（neg/pos 各一套） | **最高** | 从 `encoder_sv_neg/pos.pt` 初始化，LR 降至 1e-5 |
| 对照实验（纯真实 vs 混合模拟） | 高 | 量化模拟数据的实际贡献 |

**输出**：`encoder_tcm_neg.pt`，`encoder_tcm_pos.pt`

---

### 第三阶段：LoRA 类别适配器（可选）

**目标**：为黄酮、生物碱、萜类、皂苷等类别训练轻量级 LoRA 适配器，在类别已知时提升精度。

| 任务 | 优先级 | 说明 |
|------|--------|------|
| 按结构类别划分 TCM 化合物 | 中 | 基于 ClassyFire 或 LOTUS 分类标签 |
| 各类别 LoRA 训练 | 中 | 冻结主编码器，只训练 LoRA 层（< 1% 参数） |
| 推理时路由逻辑 | 低 | 类别已知时加载对应 LoRA，未知时使用主编码器 |

**输出**：`lora_flavonoid_neg.pt`，`lora_alkaloid_neg.pt` 等

---

### 各阶段评估指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| `val_loss`（cosine） | 验证集 cosine 损失 | 持续下降，低于初始值 |
| `val_cos_loss` | 预测指纹与真实指纹的余弦距离 | < 0.15 |
| `val_bce_loss` | 二元交叉熵（辅助参考） | 持续下降 |
| 端到端 Top-1 | 编码器+解码器联合评估 | > 0.2（可选，需解码器参与） |

---

## 附录

### A. 关键文件路径

```
/stor1/AIMS4Meta/
├── databases/
│   ├── TCM_databases/          # 中药数据库
│   │   ├── herb/               # HERB数据库
│   │   ├── lotus/              # LOTUS天然产物
│   │   └── tcmbank/            # TCMBank
│   └── spectraverse/           # SpectraVerse原始数据
│
├── datasets/
│   └── msflow_datasets/
│       ├── SpectraVerse/
│       │   └── spectraverse_neg/
│       │       ├── labels.tsv          # 129,166 条（含 inchikey, smiles 等）
│       │       ├── subformulae/        # 129,166 个 JSON（spectraverse_neg_*.json）
│       │       └── splits/             # 划分后的 TSV（待生成）
│       └── spectraverse_pos/
│           ├── labels.tsv              # 139,693 条
│           ├── subformulae/            # 139,693 个 JSON（spectraverse_pos_*.json）
│           └── splits/                 # 划分后的 TSV（待生成）
│
├── code/
│   ├── DiffMS-master/
│   │   └── src/mist/
│   │       ├── models/spectra_encoder.py  # 编码器模型 (SpectraEncoderGrowing)
│   │       └── train_mist.py              # 编码器训练脚本 (← 用这个!)
│   └── MSFlow-main/
│       ├── checkpoints/               # 模型权重
│       ├── configs/                   # 解码器配置
│       ├── models/cfg_molbert.py       # 解码器模型 (CondFlowMolBERT)
│       └── cfg_pretrain.py            # 解码器训练脚本 (← 不是编码器的!)
│
└── trainers/
    └── denovo/2_phase/
        ├── configs/spectraverse.yaml           # 训练配置
        ├── MSFlow编码器训练改进方案.md       # 本文档
        └── scripts/data_preprocessing/
            ├── msp_to_json_and_tsv.py      # MSP→JSON+TSV 主流程（已用于生成上述数据）
            ├── run_sirius_batch.py         # SIRIUS批量处理
            └── sirius_to_subformulae.py    # 碎片树→subformulae
```

### B. 架构图：编码器 vs 解码器训练分工

```
┌───────────────────────────────────────────────────────────┐
│ 编码器训练 (DiffMS/MIST)                                   │
│                                                           │
│ subformulae JSON → SpectraEncoderGrowing → 指纹 (morgan)  │
│                                                           │
│ 训练脚本: train_mist.py                                    │
│ 监督信号: 分子指纹（morgan4096）                           │
└───────────────────────────────────────────────────────────┘
                        │ 512维编码器输出
                        ▼
┌───────────────────────────────────────────────────────────┐
│ 解码器训练 (MSFlow)   [权重可复用，无需重训]               │
│                                                           │
│ CDDD (512维) → CondFlowMolBERT → SMILES                  │
│                                                           │
│ 训练脚本: cfg_pretrain.py                                  │
│ 已基于ChEMBL/MOSES充分训练，直接复用权重                  │
└───────────────────────────────────────────────────────────┘
```

### C. 参考资源

1. MSFlow GitHub: https://github.com/ghaith-mq/MSFlow
2. DiffMS (编码器框架): https://github.com/samgoldman98/diffms
3. FIORA (模拟工具): https://github.com/BAMeScience/fiora
4. CDDD (分子嵌入): https://github.com/jrwnter/cddd
5. SIRIUS文档: https://boecker-lab.github.io/docs.sirius.github.io/
6. CFM-ID 4: https://cfmid4.wishartlab.com/

---

*文档生成时间: 2026-03-17*
*最后更新: 2026-03-20 (重构文档结构；第二阶段TCM数据策略改为两步走（SV真实谱优先+FIORA模拟补充，放弃CFM-ID兜底和LOTUS）；新增编码器峰强度使用说明（ms2_inten作为输入特征+加权池化权重）)*
*适用版本: MSFlow v1.0*
