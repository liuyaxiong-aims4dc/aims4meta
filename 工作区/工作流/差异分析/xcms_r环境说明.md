# XCMS_R环境使用说明

## 环境信息
- **环境名称**: xcms_r
- **R版本**: 4.5.3
- **XCMS版本**: 4.8.0
- **CAMERA版本**: 1.66.0
- **来源**: 从192.168.200.17服务器打包

## 主要依赖包
- **核心分析包**:
  - xcms==4.8.0
  - CAMERA==1.66.0
  - Biobase
  - limma
  - MSnbase
  - mzR
  - Spectra
  - QFeatures

- **数据处理包**:
  - tidyverse
  - dplyr
  - ggplot2
  - readr
  - writexl

- **质谱专用包**:
  - MassSpecWavelet
  - MScoreutils
  - MSnbase
  - mzR
  - Spectra

## 激活环境
```bash
conda activate xcms_r
```

## 运行差异分析脚本

### 1. 两组差异分析
```bash
cd /stor3/AIMS4Meta/工作流/差异分析
./run_differential.sh
```

或手动运行：
```bash
conda activate xcms_r
Rscript /stor3/AIMS4Meta/工作流/差异分析/xcms_differential_analysis.R
```

### 2. 多组差异分析
```bash
cd /stor3/AIMS4Meta/工作流/差异分析
./run_multigroup.sh
```

或手动运行：
```bash
conda activate xcms_r
Rscript /stor3/AIMS4Meta/工作流/差异分析/xcms_multigroup_analysis.R
```

## 环境验证
```bash
conda activate xcms_r
Rscript -e "
cat('✅ XCMS_R环境验证成功！\n')
cat(sprintf('📦 R版本: %s\n', R.version.string))
if(requireNamespace('xcms', quietly=TRUE)) cat(sprintf('📦 xcms版本: %s\n', packageVersion('xcms')))
if(requireNamespace('CAMERA', quietly=TRUE)) cat(sprintf('📦 CAMERA版本: %s\n', packageVersion('CAMERA')))
"
```

## 脚本说明

### xcms_differential_analysis.R
用于两组样本的差异分析，包括：
- 原始数据导入
- 峰检测和对齐
- 样本归一化
- 统计分析
- 结果可视化

### xcms_multigroup_analysis.R
用于多组样本的差异分析，包括：
- 多组数据导入
- 批量峰检测
- 多重比较校正
- 组间差异分析
- 热图和PCA图

## 注意事项
1. 确保在运行脚本前已激活xcms_r环境
2. 输入数据格式应为mzML或mzXML
3. 样本信息文件需要包含组别信息
4. 环境已包含所有必需的R包，无需额外安装

## 环境管理
```bash
# 查看环境信息
conda env list

# 查看已安装的R包
conda activate xcms_r
Rscript -e "installed.packages()[,c('Package','Version')]"

# 导出环境配置
conda env export -n xcms_r > xcms_r_env_backup.yml

# 更新环境
# conda env update -f /tmp/xcms_r_env.yml --prune
```

## 故障排除
如果遇到问题，请检查：
1. 环境是否正确激活: `conda env list | grep xcms_r`
2. R版本是否正确: `R --version`
3. 关键包是否安装: `Rscript -e "packageVersion('xcms')"`
4. 输入文件路径是否正确
5. 文件权限是否正确

## 常见错误解决

### 错误: 找不到xcms包
```bash
conda activate xcms_r
Rscript -e "if(!requireNamespace('xcms', quietly=TRUE)) install.packages('xcms', repos='https://cloud.r-project.org')"
```

### 错误: 内存不足
在R脚本中增加内存限制：
```r
memory.limit(size = 16000)  # Windows
# 或在Linux中通过ulimit设置
```

### 错误: 路径问题
确保所有路径使用绝对路径，且路径中没有中文或特殊字符。

## 性能优化建议
1. 对于大型数据集，建议使用多核处理
2. 可以调整XCMS参数以提高精度或速度
3. 定期清理临时文件释放磁盘空间
4. 使用SSD存储可以提高数据处理速度
