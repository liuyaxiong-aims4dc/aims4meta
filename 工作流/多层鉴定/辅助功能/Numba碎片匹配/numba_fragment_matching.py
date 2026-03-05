#!/usr/bin/env python3
"""
Numba加速的碎片匹配模块

使用Numba JIT编译加速碎片匹配,预期性能提升10-100倍
"""

import numpy as np
from numba import jit, prange
from typing import List, Tuple, Dict, Optional
import time

# 尝试导入numba
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    # 只在主进程打印一次
    import os
    if not os.environ.get('NUMBA_INFO_PRINTED'):
        print("[INFO] Numba可用,将使用JIT加速")
        os.environ['NUMBA_INFO_PRINTED'] = '1'
except ImportError:
    NUMBA_AVAILABLE = False
    import os
    if not os.environ.get('NUMBA_INFO_PRINTED'):
        print("[WARNING] Numba不可用,使用纯Python版本")
        os.environ['NUMBA_INFO_PRINTED'] = '1'


@jit(nopython=True)
def find_matched_fragments_numba_core(
    query_mz: np.ndarray,
    query_int: np.ndarray,
    lib_mz: np.ndarray,
    lib_int: np.ndarray,
    tolerance: float,
    pmz_diff: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba加速的碎片匹配核心函数
    
    参数:
        query_mz: 样品峰m/z数组
        query_int: 样品峰强度数组
        lib_mz: 候选物峰m/z数组
        lib_int: 候选物峰强度数组
        tolerance: 质量容差 (Da)
        pmz_diff: 母离子差值
    
    返回:
        (query_indices, lib_indices, mz_diffs)
    """
    n_query = len(query_mz)
    n_lib = len(lib_mz)
    
    # 预分配结果数组 (最大可能匹配数)
    max_matches = min(n_query, n_lib)
    query_indices = np.empty(max_matches, dtype=np.int64)
    lib_indices = np.empty(max_matches, dtype=np.int64)
    mz_diffs = np.empty(max_matches, dtype=np.float64)
    
    # 记录已使用的lib峰
    used_lib = np.zeros(n_lib, dtype=np.bool_)
    
    match_count = 0
    
    for i in range(n_query):
        if query_int[i] <= 0:
            continue
        
        best_j = -1
        best_diff = tolerance + 1.0
        
        for j in range(n_lib):
            if used_lib[j] or lib_int[j] <= 0:
                continue
            
            # 直接匹配
            diff = abs(query_mz[i] - lib_mz[j])
            # Precursor shift匹配
            diff_shifted = abs(query_mz[i] - lib_mz[j] + pmz_diff)
            
            if diff <= tolerance and diff < best_diff:
                best_j = j
                best_diff = diff
            elif diff_shifted <= tolerance and diff_shifted < best_diff:
                best_j = j
                best_diff = diff_shifted
        
        if best_j >= 0:
            used_lib[best_j] = True
            query_indices[match_count] = i
            lib_indices[match_count] = best_j
            mz_diffs[match_count] = best_diff
            match_count += 1
    
    # 返回实际匹配数
    return query_indices[:match_count], lib_indices[:match_count], mz_diffs[:match_count]


def find_matched_fragments_numba(
    query_peaks: List[Tuple[float, float]],
    lib_peaks: List[Tuple[float, float]],
    tolerance: float,
    query_pmz: float,
    lib_pmz: float
) -> List[Dict]:
    """
    Numba加速的碎片匹配 (兼容旧接口)
    
    参数:
        query_peaks: 样品峰列表 [(mz, intensity), ...]
        lib_peaks: 候选物峰列表 [(mz, intensity), ...]
        tolerance: 质量容差 (Da)
        query_pmz: 样品母离子
        lib_pmz: 候选物母离子
    
    返回:
        匹配列表 [{'query_mz': ..., 'lib_mz': ..., ...}, ...]
    """
    if not query_peaks or not lib_peaks:
        return []
    
    # 转换为NumPy数组
    query_mz = np.array([p[0] for p in query_peaks], dtype=np.float64)
    query_int = np.array([p[1] for p in query_peaks], dtype=np.float64)
    lib_mz = np.array([p[0] for p in lib_peaks], dtype=np.float64)
    lib_int = np.array([p[1] for p in lib_peaks], dtype=np.float64)
    
    pmz_diff = query_pmz - lib_pmz if query_pmz and lib_pmz else 0.0
    
    # 调用Numba核心函数
    query_idx, lib_idx, mz_diff = find_matched_fragments_numba_core(
        query_mz, query_int, lib_mz, lib_int, tolerance, pmz_diff
    )
    
    # 转换为旧格式
    matched = []
    for i, j, diff in zip(query_idx, lib_idx, mz_diff):
        matched.append({
            "query_mz": round(query_mz[i], 4),
            "lib_mz": round(lib_mz[j], 4),
            "query_intensity": round(query_int[i], 4),
            "lib_intensity": round(lib_int[j], 4),
            "mz_diff": round(diff, 4),
        })
    
    return matched


def find_matched_fragments_cpu(
    query_peaks: List[Tuple[float, float]],
    lib_peaks: List[Tuple[float, float]],
    tolerance: float,
    query_pmz: float,
    lib_pmz: float
) -> List[Dict]:
    """
    CPU碎片匹配 (原始Python实现,作为fallback)
    """
    matched = []
    pmz_diff = query_pmz - lib_pmz if query_pmz and lib_pmz else 0
    used_lib = set()
    
    for q_mz, q_int in query_peaks:
        if q_int <= 0:
            continue
        
        best_match = None
        best_diff = float('inf')
        
        for i, (l_mz, l_int) in enumerate(lib_peaks):
            if i in used_lib or l_int <= 0:
                continue
            
            # 直接匹配
            diff = abs(q_mz - l_mz)
            # Precursor shift匹配
            diff_shifted = abs(q_mz - l_mz + pmz_diff)
            
            if diff <= tolerance and diff < best_diff:
                best_match = (i, l_mz, l_int, diff)
                best_diff = diff
            elif diff_shifted <= tolerance and diff_shifted < best_diff:
                best_match = (i, l_mz, l_int, diff_shifted)
                best_diff = diff_shifted
        
        if best_match:
            i, l_mz, l_int, diff = best_match
            used_lib.add(i)
            matched.append({
                "query_mz": round(q_mz, 4),
                "lib_mz": round(l_mz, 4),
                "query_intensity": round(q_int, 4),
                "lib_intensity": round(l_int, 4),
                "mz_diff": round(diff, 4),
            })
    
    return matched


def find_matched_fragments(
    query_peaks: List[Tuple[float, float]],
    lib_peaks: List[Tuple[float, float]],
    tolerance: float,
    query_pmz: float,
    lib_pmz: float,
    use_numba: bool = True
) -> List[Dict]:
    """
    碎片匹配 (自动选择最优实现)
    
    参数:
        query_peaks: 样品峰列表 [(mz, intensity), ...]
        lib_peaks: 候选物峰列表 [(mz, intensity), ...]
        tolerance: 质量容差 (Da)
        query_pmz: 样品母离子
        lib_pmz: 候选物母离子
        use_numba: 是否使用Numba加速
    
    返回:
        匹配列表 [{'query_mz': ..., 'lib_mz': ..., ...}, ...]
    """
    if use_numba and NUMBA_AVAILABLE:
        try:
            return find_matched_fragments_numba(
                query_peaks, lib_peaks, tolerance,
                query_pmz, lib_pmz
            )
        except Exception as e:
            print(f"[WARNING] Numba匹配失败,回退到CPU: {e}")
            return find_matched_fragments_cpu(
                query_peaks, lib_peaks, tolerance,
                query_pmz, lib_pmz
            )
    else:
        return find_matched_fragments_cpu(
            query_peaks, lib_peaks, tolerance,
            query_pmz, lib_pmz
        )


if __name__ == "__main__":
    # 性能测试
    print("=" * 60)
    print("Numba碎片匹配性能测试")
    print("=" * 60)
    
    # 生成测试数据
    np.random.seed(42)
    
    # 测试不同峰数
    for n_peaks in [50, 100, 200, 500]:
        print(f"\n【测试: {n_peaks}个峰】")
        
        query_peaks = [(100.0 + i*10 + np.random.rand(), np.random.rand()) 
                       for i in range(n_peaks)]
        lib_peaks = [(100.5 + i*10 + np.random.rand(), np.random.rand()) 
                     for i in range(n_peaks)]
        tolerance = 0.5
        query_pmz = 500.0
        lib_pmz = 499.9
        
        # 预热Numba (首次编译)
        if n_peaks == 50:
            _ = find_matched_fragments_numba(query_peaks, lib_peaks, tolerance, query_pmz, lib_pmz)
        
        # 测试CPU版本
        start = time.time()
        for _ in range(100):
            cpu_results = find_matched_fragments_cpu(query_peaks, lib_peaks, tolerance, query_pmz, lib_pmz)
        cpu_time = (time.time() - start) / 100
        
        # 测试Numba版本
        start = time.time()
        for _ in range(100):
            numba_results = find_matched_fragments_numba(query_peaks, lib_peaks, tolerance, query_pmz, lib_pmz)
        numba_time = (time.time() - start) / 100
        
        print(f"  CPU:  {cpu_time*1000:.2f} ms, {len(cpu_results)} 个匹配")
        print(f"  Numba: {numba_time*1000:.2f} ms, {len(numba_results)} 个匹配")
        print(f"  加速比: {cpu_time/numba_time:.2f}x")
        
        # 验证结果一致性
        if len(cpu_results) == len(numba_results):
            print(f"  ✓ 匹配数量一致")
        else:
            print(f"  ⚠ 匹配数量不一致: CPU {len(cpu_results)} vs Numba {len(numba_results)}")
