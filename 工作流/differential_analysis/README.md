# 中药正品与伪品差异化合物分析工作流

## 概述

本工作流用于识别导致中药正品(Group A)与伪品(Group B)之间差异的关键化合物,基于DreaMS深度学习模型进行质谱数据分析。

---

## 📊 完整数据处理流程图

```
原始仪器数据 (.raw, .d, .wiff, .mzML等)
    ↓
[峰提取与峰对齐]
    使用 MS-DIAL / MZmine / XCMS
    ↓
MS-DIAL 输出 (所有样本合并)
├── all_spectra.msp          # 所有MS/MS谱图
└── alignment_result.txt     # 峰对齐定量表
    ↓
[Step 0] 数据预处理
    00_preprocess_msdial_output.py
    ↓
按组分开的MSP文件
├── group_A_authentic/       # 正品样本
└── group_B_counterfeit/     # 伪品样本
    ↓
[Step 1-4] 差异化合物分析
    ↓
最终报告与可视化
```

---

## 🔧 Step 0: 原始数据预处理

### 选项 A: 使用 MS-DIAL (推荐) ⭐⭐⭐⭐⭐

**1. MS-DIAL 峰检测与对齐**

```
MS-DIAL 操作流程:

1. File → New Project
   - 选择数据类型: LC-MS/MS
   - 离子模式: Positive/Negative

2. File → Import
   - 导入所有 .raw/.d/.wiff 文件
   - 设置样本信息

3. Process → Peak detection
   - Minimum peak height: 1000
   - Mass slice width: 0.1 Da
   - Smoothing level: 3

4. Process → Alignment
   - RT tolerance: ±0.1 min
   - m/z tolerance: ±0.015 Da

5. Export → Export spectra
   - Format: MSP
   - 保存为: all_spectra.msp

6. Export → Alignment result
   - 保存为: alignment_result.txt
```

**2. 准备样本元数据**

创建 `sample_metadata.csv`:

```csv
sample_id,group,replicate,collection_date,notes
sample_A1,A,1,2025-01-01,Authentic batch 1
sample_A2,A,2,2025-01-02,Authentic batch 1
sample_A3,A,3,2025-01-03,Authentic batch 2
sample_B1,B,1,2025-01-01,Counterfeit source 1
sample_B2,B,2,2025-01-02,Counterfeit source 1
sample_B3,B,3,2025-01-03,Counterfeit source 2
```

**必需字段**:
- `sample_id`: 样本ID (必须与MS-DIAL中的样本名称匹配)
- `group`: 组别 (`A` = 正品, `B` = 伪品)

**3. 运行预处理脚本**

```bash
python 00_preprocess_msdial_output.py \
    --msp-file /path/to/msdial_output/all_spectra.msp \
    --feature-table /path/to/msdial_output/alignment_result.txt \
    --metadata sample_metadata.csv \
    --output ./preprocessed_data
```

**输出**:
```
preprocessed_data/
├── group_A_authentic/
│   ├── sample_A1.msp
│   ├── sample_A2.msp
│   └── sample_A3.msp
├── group_B_counterfeit/
│   ├── sample_B1.msp
│   ├── sample_B2.msp
│   └── sample_B3.msp
├── feature_table_A.csv      # Group A 定量表
└── feature_table_B.csv      # Group B 定量表
```

---

### 选项 B: 使用 MZmine 3

```
MZmine 3 操作流程:

1. Raw data methods → Import MS data
   - 选择所有原始文件

2. Feature detection → Mass detection
   - Noise level: 1E3

3. Feature detection → Chromatogram builder
   - Min height: 1E3
   - m/z tolerance: 10 ppm

4. Feature detection → Chromatogram deconvolution
   - Local minimum search

5. Feature list methods → Alignment
   - Join aligner
   - m/z tolerance: 10 ppm
   - RT tolerance: 0.1 min

6. Feature list methods → Export
   - GNPS format → MGF file
   - Feature quantification → CSV
```

**转换为MSP格式**:
```bash
# 如果MZmine输出MGF,可以转换为MSP
# 或直接在Step 1中使用MGF文件
```

---

### 选项 C: 使用 XCMS (R package)

```r
library(xcms)

# 1. 读取数据
files <- list.files("raw_data/", pattern = "\\.mzML$", full.names = TRUE)
data <- readMSData(files, mode = "onDisk")

# 2. 峰检测
cwp <- CentWaveParam(ppm = 10, peakwidth = c(5, 20), snthresh = 5)
data <- findChromPeaks(data, param = cwp)

# 3. 峰对齐
group_info <- rep(c("A", "B"), each = 3)  # 3个A组, 3个B组
pdp <- PeakDensityParam(sampleGroups = group_info, bw = 5)
data <- groupChromPeaks(data, param = pdp)

# 4. 导出
feature_table <- featureValues(data, value = "into")
write.csv(feature_table, "feature_table.csv")
```

---

## 工作流程

### Step 1: DreaMS 嵌入向量生成
**脚本**: `01_generate_dreams_embeddings.py`

**功能**:
- 合并 Group A (正品) 的所有 MSP 文件
- 合并 Group B (伪品) 的所有 MSP 文件
- 生成统一的 DreaMS 1024维嵌入向量
- 创建包含分组信息的元数据表

**输入**:
```
group_a_dir/              # Group A (正品) MSP 文件目录
├── sample_A1.msp
├── sample_A2.msp
└── sample_A3.msp

group_b_dir/              # Group B (伪品) MSP 文件目录
├── sample_B1.msp
├── sample_B2.msp
└── sample_B3.msp
```

**输出**:
```
output_dir/
├── merged_spectra_A.msp          # 合并的 A 组谱图
├── merged_spectra_B.msp          # 合并的 B 组谱图
├── merged_all.msp                # 所有谱图合并
├── spectrum_metadata.csv         # 谱图元数据 (含分组标签)
└── embeddings/
    ├── embeddings.npz            # DreaMS 嵌入向量
    └── spectra.mgf               # MGF 格式谱图
```

**命令**:
```bash
python 01_generate_dreams_embeddings.py \
    --group-a /path/to/group_a_dir \
    --group-b /path/to/group_b_dir \
    --output /path/to/output_dir
```

---

### Step 2: 差异化合物筛选
**脚本**: `02_differential_analysis.py`

**功能**:
1. **PCA 分析**: 降维可视化,观察两组的整体分离
2. **特有化合物识别**: 找出每组特有的化合物
   - 策略: 组内相似度高 + 组间相似度低
3. **定量差异分析** (可选): 基于峰面积的统计检验
4. **可视化**: PCA 图、火山图 (如果有定量数据)

**输入**:
- `embeddings.npz` (来自 Step 1)
- `spectrum_metadata.csv` (来自 Step 1)
- (可选) `feature_table_A.csv` 和 `feature_table_B.csv` 用于定量分析

**输出**:
```
output_dir/
├── statistics/
│   ├── pca_coordinates.csv              # PCA 坐标
│   ├── unique_to_authentic.csv          # 正品特有化合物
│   └── unique_to_counterfeit.csv        # 伪品特有化合物
├── visualization/
│   ├── pca_plot.png                     # PCA 可视化
│   └── volcano_plot.png                 # 火山图 (如有定量数据)
└── DIFFERENTIAL_SUMMARY_REPORT.txt      # 初步统计报告
```

**关键参数**:
- `--similarity-threshold 0.85`: 判断特有化合物的相似度阈值
- `--feature-table-a`: A 组定量数据 (可选)
- `--feature-table-b`: B 组定量数据 (可选)

**命令**:
```bash
python 02_differential_analysis.py \
    --embeddings output_dir/embeddings/embeddings.npz \
    --metadata output_dir/spectrum_metadata.csv \
    --output output_dir \
    --similarity-threshold 0.85
```

---

### Step 3: 差异化合物结构鉴定
**脚本**: `03_identify_differential_compounds.py`

**功能**:
1. 提取差异化合物的谱图
2. 运行三层鉴定流程:
   - L1: MSDial 谱图库匹配
   - L2: 自定义数据库搜索 (tcmbank, herb, coconut 等)
   - L3: SIRIUS 全库搜索 + 骨架约束
3. 标注鉴定结果的差异类型 (正品特有 vs 伪品特有)

**输入**:
- `merged_all.msp` (来自 Step 1)
- `unique_to_authentic.csv` (来自 Step 2)
- `unique_to_counterfeit.csv` (来自 Step 2)
- `config.yaml` (DreaMS 配置文件)

**输出**:
```
output_dir/
├── identification/
│   ├── differential_compounds.msp       # 提取的差异化合物谱图
│   └── final_report/
│       └── detailed_identifications.csv # 三层鉴定结果
└── FINAL_DIFFERENTIAL_COMPOUNDS_IDENTIFIED.csv  # 标注了差异类型的最终结果
```

**命令**:
```bash
python 03_identify_differential_compounds.py \
    --results-dir output_dir \
    --merged-msp output_dir/merged_all.msp \
    --unique-a output_dir/statistics/unique_to_authentic.csv \
    --unique-b output_dir/statistics/unique_to_counterfeit.csv \
    --config config.yaml \
    --l2-databases tcmbank herb coconut
```

---

### Step 4: 生成综合报告
**脚本**: `04_generate_final_report.py`

**功能**:
1. 生成详细的文本报告:
   - 数据集概览
   - 统计摘要
   - Top 10 正品标志化合物
   - Top 10 伪品标志化合物
   - 化学分类分析
   - 鉴别建议
   - 质量控制流程建议
2. 生成 Excel 汇总表 (多个工作表)

**输入**:
- `FINAL_DIFFERENTIAL_COMPOUNDS_IDENTIFIED.csv` (来自 Step 3)
- `unique_to_authentic.csv` (来自 Step 2)
- `unique_to_counterfeit.csv` (来自 Step 2)
- `spectrum_metadata.csv` (来自 Step 1)

**输出**:
```
output_dir/
├── COMPREHENSIVE_DIFFERENTIAL_REPORT.txt    # 详细中英文报告
└── DIFFERENTIAL_ANALYSIS_SUMMARY.xlsx       # Excel 汇总表
    ├── [Sheet 1] Overview                   # 总览
    ├── [Sheet 2] Authentic_Markers          # 正品标志物
    ├── [Sheet 3] Counterfeit_Markers        # 伪品标志物
    └── [Sheet 4] Top_Identified             # Top 鉴定结果
```

**命令**:
```bash
python 04_generate_final_report.py \
    --results-dir output_dir
```

---

## 一键运行 (Master Script)

**脚本**: `run_differential_analysis.sh`

**使用方法**:
```bash
bash run_differential_analysis.sh \
    --group-a /path/to/authentic_samples \
    --group-b /path/to/counterfeit_samples \
    --output /path/to/results \
    --similarity-threshold 0.85 \
    --l2-databases "tcmbank herb coconut" \
    --config config.yaml
```

**参数说明**:
- `--group-a`: Group A (正品) MSP 文件目录 (必需)
- `--group-b`: Group B (伪品) MSP 文件目录 (必需)
- `--output`: 结果输出目录 (默认: ./differential_analysis_results)
- `--similarity-threshold`: 相似度阈值 (默认: 0.85)
- `--l2-databases`: L2 搜索数据库列表 (默认: tcmbank herb coconut)
- `--config`: DreaMS 配置文件 (默认: config.yaml)

---

## 结果解读

### 1. 正品特有化合物 (Authentic-specific Markers)
这些化合物**应该**在正品样本中存在:
- 高组内相似度 (与其他正品谱图相似)
- 低组间相似度 (与伪品谱图不相似)
- **用途**: 正品鉴别的**阳性指标**

### 2. 伪品特有化合物 (Counterfeit-specific Markers)
这些化合物**不应该**在正品样本中存在:
- 高组内相似度 (与其他伪品谱图相似)
- 低组间相似度 (与正品谱图不相似)
- **用途**: 正品鉴别的**阴性指标** (伪品排除标志)

### 3. Specificity Score (特异性评分)
`specificity = sim_to_own_group - sim_to_other_group`
- 评分越高,越具有组特异性
- 推荐使用 specificity > 0.2 的化合物作为标志物

---

## 质量控制应用

基于分析结果建立质量控制标准:

### 判定标准 (示例)

**正品 (Authentic)**:
- ≥3 个正品标志物检出
- 无伪品标志物检出

**伪品 (Counterfeit)**:
- <2 个正品标志物检出
- 或 ≥1 个伪品标志物检出

**不确定 (Uncertain)**:
- 2 个正品标志物检出
- 且无伪品标志物检出
- → 建议补充检测

---

## 输出文件总览

```
differential_analysis_results/
├── merged_spectra_A.msp                             # Step 1: A 组合并谱图
├── merged_spectra_B.msp                             # Step 1: B 组合并谱图
├── merged_all.msp                                   # Step 1: 所有谱图
├── spectrum_metadata.csv                            # Step 1: 元数据
├── embeddings/
│   ├── embeddings.npz                               # Step 1: DreaMS 嵌入
│   └── spectra.mgf                                  # Step 1: MGF 格式
├── statistics/
│   ├── pca_coordinates.csv                          # Step 2: PCA 坐标
│   ├── unique_to_authentic.csv                      # Step 2: 正品标志物
│   └── unique_to_counterfeit.csv                    # Step 2: 伪品标志物
├── visualization/
│   └── pca_plot.png                                 # Step 2: PCA 图
├── identification/
│   ├── differential_compounds.msp                   # Step 3: 差异化合物谱图
│   └── final_report/
│       └── detailed_identifications.csv             # Step 3: 鉴定结果
├── FINAL_DIFFERENTIAL_COMPOUNDS_IDENTIFIED.csv      # Step 3: 最终标注结果
├── COMPREHENSIVE_DIFFERENTIAL_REPORT.txt            # Step 4: 综合报告
└── DIFFERENTIAL_ANALYSIS_SUMMARY.xlsx               # Step 4: Excel 汇总
```

---

## 依赖环境

### Conda 环境
```bash
conda activate dreams
```

### Python 依赖
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn
- openpyxl (for Excel output)

### DreaMS 组件
- `core/embeddings_generator.py`
- `core/msp_parser.py`
- `workflows/three_layer_id/three_layer_workflow.py`

---

## 常见问题

### Q1: 如果没有定量数据 (feature table) 怎么办?
A: 可以只进行定性分析,基于 DreaMS 相似度识别特有化合物。定量差异分析会被跳过。

### Q2: 如何调整特异性阈值?
A: 使用 `--similarity-threshold` 参数:
- 较高 (0.9): 更严格,找到的标志物更少但更可靠
- 较低 (0.8): 更宽松,找到的标志物更多但可能有假阳性

### Q3: 如果鉴定率很低怎么办?
A:
1. 检查是否选择了合适的 L2 数据库 (如中药专用数据库)
2. 使用 L3 全库搜索 (已默认开启)
3. 考虑使用 de novo 从头预测 (需要额外配置)

### Q4: 可以分析多于两组的数据吗?
A: 当前工作流设计用于 A vs B 两组比较。如需多组比较,可以:
- 两两比较 (A vs B, A vs C, B vs C)
- 或修改脚本支持多组 ANOVA 分析

---

## 引用

如果使用本工作流,请引用:

**DreaMS**:
- Bushuiev et al. "DreaMS: Deep representations empowering the annotation of mass spectra." *Nature Biotechnology* (2025).

**SIRIUS** (如果使用 L2/L3):
- Dührkop et al. "SIRIUS 4: a rapid tool for turning tandem mass spectra into metabolite structure information." *Nature Methods* (2019).

---

## 联系与支持

- DreaMS GitHub: https://github.com/pluskal-lab/DreaMS
- DreaMS 文档: https://dreams-docs.readthedocs.io/
- 问题反馈: 提交 GitHub Issue

---

## 更新日志

**v1.0** (2026-01-10):
- 初始版本
- 支持正品 vs 伪品差异分析
- 集成 DreaMS 嵌入向量 + 三层鉴定
- 自动生成综合报告
