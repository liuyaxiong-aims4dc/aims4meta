# Numba碎片鉴定模块说明

## 功能概述

`numba_fragment_matching.py` 是一个使用Numba JIT编译加速的碎片峰鉴定模块，用于在质谱鉴定过程中获取详细的碎片鉴定信息。

## 核心功能

### 1. 碎片峰鉴定
- 找出查询光谱和库光谱之间鉴定的碎片峰
- 支持自定义质量容差（tolerance）
- 支持母离子偏移补偿（precursor shift）

### 2. 详细鉴定信息
返回每个鉴定碎片的完整信息：
- `query_mz`: 查询峰的m/z值
- `lib_mz`: 库峰的m/z值
- `query_intensity`: 查询峰的强度
- `lib_intensity`: 库峰的强度
- `mz_diff`: m/z差异（绝对值）

### 3. 性能优化
- 使用Numba JIT编译加速
- 自动检测Numba可用性
- 提供CPU fallback实现

## 与matchMS的关系

### matchMS提供：
- 余弦相似度分数
- 鉴定峰数量

### Numba碎片鉴定提供：
- 每个鉴定峰的详细m/z信息
- 每个鉴定峰的强度信息
- m/z差异值

**两者互补**：matchMS计算整体相似度，Numba碎片鉴定提供详细鉴定信息。

## 使用方法

```python
from numba_fragment_matching import find_matched_fragments

# 准备数据
query_peaks = [(100.0, 50.0), (150.0, 80.0), (200.0, 100.0)]  # (m/z, intensity)
lib_peaks = [(100.05, 45.0), (150.02, 85.0), (200.1, 90.0)]

# 执行鉴定
matched_fragments = find_matched_fragments(
    query_peaks=query_peaks,
    lib_peaks=lib_peaks,
    tolerance=0.05,  # 质量容差 (Da)
    query_pmz=300.0,  # 查询母离子m/z
    lib_pmz=300.0     # 库母离子m/z
)

# 查看结果
for frag in matched_fragments:
    print(f"Query m/z: {frag['query_mz']:.4f}")
    print(f"Lib m/z: {frag['lib_mz']:.4f}")
    print(f"m/z差异: {frag['mz_diff']:.4f} Da")
    print(f"Query强度: {frag['query_intensity']:.2f}")
    print(f"Lib强度: {frag['lib_intensity']:.2f}")
```

## 性能数据

基于测试结果（10,000次鉴定）：

| 指标 | 数值 |
|------|------|
| 速度 | ~6,700 次/秒 |
| 加速比 | 比纯Python快约10倍 |
| 内存占用 | 低（Numba优化） |

## 在多层鉴定流程中的作用

### 调用位置
- `L1_matchMS鉴定.py` - L1真实数据库鉴定
- `L2_matchMS鉴定.py` - L2模拟数据库鉴定

### 工作流程
1. matchMS计算余弦相似度分数
2. 如果分数超过阈值，调用Numba碎片鉴定
3. 获取详细的碎片鉴定信息
4. 将鉴定信息添加到结果中

### 结果展示
鉴定信息以字符串形式存储在结果CSV中：
```
100.0000/100.0500(Δ0.0500Da); 150.0000/150.0200(Δ0.0200Da)
```

## 技术细节

### Numba加速原理
- 使用`@njit`装饰器进行JIT编译
- 将Python代码编译为机器码
- 避免Python解释器开销

### 母离子偏移补偿
当查询和库的母离子m/z不同时，自动计算偏移量并尝试鉴定：
```python
pmz_diff = query_pmz - lib_pmz
# 尝试直接鉴定和偏移鉴定
diff_shifted = abs(query_mz - lib_mz + pmz_diff)
```

### 贪婪鉴定策略
- 每个库峰只能鉴定一次
- 优先鉴定m/z差异最小的峰
- 确保鉴定的唯一性

## 依赖

- **必需**: numpy
- **可选**: numba (用于加速，无则自动降级到CPU版本)

## 注意事项

1. **首次运行较慢**: Numba需要编译时间，首次调用会稍慢
2. **内存管理**: 大规模鉴定时注意内存使用
3. **容差设置**: 根据仪器精度调整tolerance参数
   - Q-TOF: 0.05 Da
   - Orbitrap: 0.01 Da

## 更新历史

- 2024-02: 初始版本
- 2024-03: 添加母离子偏移补偿
- 2025-03: 优化性能，添加详细文档

## 作者

多层鉴定项目组
