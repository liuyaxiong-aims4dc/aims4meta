# DreaMS 功能介绍说明文档

## 1. 核心功能：DreaMS 1024 维光谱表示
- **功能**：通过自监督学习训练的深度神经网络，将质谱转换为 1024 维向量表示
- **用途**：用于光谱相似性计算、聚类、分子网络构建等下游任务
- **实现**：`dreams_embeddings()` 函数，加载 `embedding_model.ckpt` 权重文件

## 2. 光谱库匹配
- **功能**：使用 DreaMS embedding 替代传统方法（如 dot product、modified cosine）进行光谱库匹配
- **优势**：基于深度表示的相似性通常比传统方法更准确
- **实现**：在 tutorials/library_matching.ipynb 中有示例

## 3. 分子网络构建
- **功能**：基于 DreaMS 相似度构建 k-NN 分子网络
- **实现**：使用 NN-Descent 算法高效构建网络，可导出为 GraphML 等格式用于 Cytoscape 可视化
- **自定义脚本**：`build_dreams_network.py`（当前项目中）

## 4. DreaMS Atlas
- **功能**：预构建的 201 亿光谱全球分子网络（3-NN 图）
- **用途**：新数据可投影到此 atlas 上，快速获得全局上下文
- **实现**：在 tutorials/atlas.ipynb 中有示例，支持本地投影和 TMAP 可视化

## 5. 预训练 + 微调框架
- **SSL Backbone**：`ssl_model.ckpt` - 自监督预训练的骨干网络
- **Fine-tuning Heads**：针对不同任务的头部网络，如：
  - RegressionHead：分子性质预测
  - ClassificationHead：分子分类
- **用途**：支持将预训练表征迁移到特定下游任务

## 6. 分子性质预测
- **功能**：预测 11 个 RDKit 分子描述符，包括：
  - AtomicLogP（脂水分配系数）
  - NumHAcceptors（氢键受体数）
  - NumHDonors（氢键供体数）
  - PolarSurfaceArea（极性表面积）
  - NumRotatableBonds（可旋转键数）
  - NumAromaticRings（芳香环数）
  - NumAliphaticRings（脂肪环数）
  - FractionCSP3（sp3 杂化碳分数）
  - QED（定量估计药物相似性）
  - SyntheticAccessibility（合成可及性）
  - BertzComplexity（Bertz 复杂度）
- **实现**：通过 RegressionHead 微调实现

## 7. Attention 分析
- **功能**：提取模型中间层表示和注意力权重
- **用途**：
  - 可解释性分析：理解模型关注的光谱区域
  - 通过 MACCS fingerprint probing 测量模型对分子结构的"觉知"
  - 分析注意力权重是否指向碎片关系
- **实现**：在 tutorials/attention_heads_analysis/ 中有示例

## 8. LSH 大规模聚类
- **功能**：使用局部敏感哈希（Locality Sensitive Hashing）进行快速光谱聚类
- **用途**：处理大规模数据集时的近似聚类，提高计算效率
- **实现**：在 `dreams/algorithms/lsh/` 目录下

## 9. MSData 数据格式管理
- **功能**：统一的质谱数据管理格式，基于 HDF5
- **优势**：高效存储和访问大规模质谱数据
- **实现**：在 `dreams/utils/data.py` 和相关模块中

## 10. Murcko 模式的结构感知数据划分
- **功能**：基于 Murcko 骨架的训练-验证-测试数据划分
- **优势**：避免分子结构相似性导致的数据泄露问题
- **实现**：在 `dreams/algorithms/murcko_hist/` 目录下

## 11. 光谱质量评估
- **功能**：评估单个光谱和 LC-MS 数据的质量指标
- **用途**：数据预处理阶段的质量控制
- **实现**：在 tutorials/spectral_quality.ipynb 中有示例

## 12. GeMS 数据集与 Atlas 构建
- **功能**：从原始质谱数据构建 GeMS（Global Exploration of Molecular Structures）数据集
- **实现**：
  - 数据清洗和预处理脚本
  - 网络构建和聚类工具
  - 在 experiments/dreams_atlas/ 目录下

## 13. HF Spaces Web App
- **功能**：在 Hugging Face Spaces 上的 Web 应用
- **用途**：一键进行光谱库检索演示
- **特点**：无需本地安装，快速体验 DreaMS 功能