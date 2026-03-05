#!/usr/bin/env python3
"""
同位素相似度计算脚本

根据分子式计算理论同位素分布，与实测同位素分布比较，计算相似度
"""

import argparse
import json
import os
import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm


# 同位素自然丰度（%）
ISOTOPE_ABUNDANCE = {
    'H': {'H1': 99.9885, 'H2': 0.0115},
    'C': {'C12': 98.93, 'C13': 1.07},
    'N': {'N14': 99.632, 'N15': 0.368},
    'O': {'O16': 99.757, 'O17': 0.038, 'O18': 0.205},
    'S': {'S32': 94.93, 'S33': 0.76, 'S34': 4.29, 'S36': 0.02},
    'Cl': {'Cl35': 75.78, 'Cl37': 24.22},
    'Br': {'Br79': 50.69, 'Br81': 49.31},
    'P': {'P31': 100.0},
    'F': {'F19': 100.0},
    'I': {'I127': 100.0},
}


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="同位素相似度计算与预处理")
    
    # 模式1：相似度计算模式
    parser.add_argument("--input", help="输入CSV文件（鉴定结果，含matched_formula列）")
    parser.add_argument("--output", help="输出CSV文件（默认覆盖原文件）")
    parser.add_argument("--formula_col", default="matched_formula", 
                       help="分子式列名")
    parser.add_argument("--isotope_dist_col", default="isotope_distribution",
                       help="实测同位素分布列名")
    
    # 模式2：预处理模式（从原始样品CSV提取同位素分布，导出JSON映射）
    parser.add_argument("--sample_csv", help="原始样品CSV文件（MSDIAL导出，用于预处理）")
    parser.add_argument("--export_json", help="导出同位素映射JSON文件（预处理模式）")
    
    return parser.parse_args()


def parse_formula(formula: str) -> dict:
    """解析分子式，返回元素计数字典
    示例: 'C6H12O6' -> {'C': 6, 'H': 12, 'O': 6}
    """
    if not formula or not isinstance(formula, str):
        return {}
    
    formula = formula.strip()
    if not formula:
        return {}
    
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    
    element_counts = {}
    for element, count in matches:
        count = int(count) if count else 1
        element_counts[element] = element_counts.get(element, 0) + count
    
    return element_counts


def calculate_theoretical_isotope_distribution(formula: str, max_mz_diff: int = 4) -> List[Tuple[int, float]]:
    """
    根据分子式计算理论同位素分布
    
    参数:
        formula: 分子式，如 'C6H12O6'
        max_mz_diff: 最大m/z差值（默认计算M+0到M+4）
    
    返回:
        [(mz_diff, relative_abundance), ...]，相对丰度归一化到100
    """
    element_counts = parse_formula(formula)
    if not element_counts:
        return []
    
    from math import comb
    
    distribution = [1.0] + [0.0] * max_mz_diff
    
    for element, count in element_counts.items():
        if element not in ISOTOPE_ABUNDANCE:
            continue
        
        new_distribution = [0.0] * (max_mz_diff + 1)
        
        for i in range(max_mz_diff + 1):
            if distribution[i] == 0:
                continue
            
            if element == 'C' and count > 0:
                p_c13 = 0.0107
                for k in range(min(count, max_mz_diff - i) + 1):
                    prob = comb(count, k) * (p_c13 ** k) * ((1 - p_c13) ** (count - k))
                    if i + k <= max_mz_diff:
                        new_distribution[i + k] += distribution[i] * prob
            
            elif element == 'H' and count > 0:
                p_h2 = 0.000115
                for k in range(min(count, max_mz_diff - i) + 1):
                    prob = comb(count, k) * (p_h2 ** k) * ((1 - p_h2) ** (count - k))
                    if i + k <= max_mz_diff:
                        new_distribution[i + k] += distribution[i] * prob
            
            elif element == 'N' and count > 0:
                p_n15 = 0.00368
                for k in range(min(count, max_mz_diff - i) + 1):
                    prob = comb(count, k) * (p_n15 ** k) * ((1 - p_n15) ** (count - k))
                    if i + k <= max_mz_diff:
                        new_distribution[i + k] += distribution[i] * prob
            
            elif element == 'O' and count > 0:
                p_o18 = 0.00205
                for k in range(min(count, max_mz_diff - i) + 1):
                    prob = comb(count, k) * (p_o18 ** k) * ((1 - p_o18) ** (count - k))
                    if i + 2 * k <= max_mz_diff:
                        new_distribution[i + 2 * k] += distribution[i] * prob
            
            elif element == 'S' and count > 0:
                p_s34 = 0.0429
                for k in range(min(count, max_mz_diff - i) + 1):
                    prob = comb(count, k) * (p_s34 ** k) * ((1 - p_s34) ** (count - k))
                    if i + 2 * k <= max_mz_diff:
                        new_distribution[i + 2 * k] += distribution[i] * prob
            
            elif element == 'Cl' and count > 0:
                p_cl37 = 0.2422
                for k in range(min(count, max_mz_diff - i) + 1):
                    prob = comb(count, k) * (p_cl37 ** k) * ((1 - p_cl37) ** (count - k))
                    if i + 2 * k <= max_mz_diff:
                        new_distribution[i + 2 * k] += distribution[i] * prob
            
            elif element == 'Br' and count > 0:
                p_br81 = 0.4931
                for k in range(min(count, max_mz_diff - i) + 1):
                    prob = comb(count, k) * (p_br81 ** k) * ((1 - p_br81) ** (count - k))
                    if i + 2 * k <= max_mz_diff:
                        new_distribution[i + 2 * k] += distribution[i] * prob
            
            else:
                new_distribution[i] += distribution[i]
        
        distribution = new_distribution
    
    max_val = max(distribution) if distribution else 1
    if max_val > 0:
        distribution = [d / max_val * 100 for d in distribution]
    
    return [(i, distribution[i]) for i in range(len(distribution)) if distribution[i] > 0.01]


def parse_measured_isotope_distribution(dist_str: str) -> List[Tuple[int, float]]:
    """
    解析实测同位素分布字符串
    
    MSDIAL格式示例: "100 - 89.5 - 63 - 6.71 - 2.01"
    返回: [(0, 100), (1, 89.5), (2, 63), (3, 6.71), (4, 2.01)]
    """
    if not dist_str or not isinstance(dist_str, str):
        return []
    
    try:
        parts = [p.strip() for p in str(dist_str).replace(',', '-').split('-')]
        values = []
        for i, part in enumerate(parts):
            if part:
                try:
                    val = float(part)
                    values.append((i, val))
                except ValueError:
                    continue
        return values
    except Exception:
        return []


def calculate_isotope_similarity(theoretical: List[Tuple[int, float]], 
                                 measured: List[Tuple[int, float]]) -> float:
    """
    计算理论同位素分布与实测分布的相似度
    
    使用余弦相似度计算
    返回: 0-100的相似度分数
    """
    if not theoretical or not measured:
        return 0.0
    
    max_mz = max(max(mz for mz, _ in theoretical), max(mz for mz, _ in measured))
    
    vec_theo = [0.0] * (max_mz + 1)
    vec_meas = [0.0] * (max_mz + 1)
    
    for mz, val in theoretical:
        vec_theo[mz] = val
    for mz, val in measured:
        vec_meas[mz] = val
    
    vec_theo = np.array(vec_theo)
    vec_meas = np.array(vec_meas)
    
    norm_theo = np.linalg.norm(vec_theo)
    norm_meas = np.linalg.norm(vec_meas)
    
    if norm_theo == 0 or norm_meas == 0:
        return 0.0
    
    cosine_sim = np.dot(vec_theo, vec_meas) / (norm_theo * norm_meas)
    similarity = max(0, min(100, cosine_sim * 100))
    
    return round(similarity, 2)


def calculate_isotope_similarity_for_df(df: pd.DataFrame, 
                                       formula_col: str,
                                       isotope_dist_col: str) -> pd.DataFrame:
    """为DataFrame计算同位素相似度"""
    
    # 查找实际列名
    actual_formula_col = None
    for col in df.columns:
        if col.lower() == formula_col.lower():
            actual_formula_col = col
            break
    
    # 同位素分布列名可能有多种形式，尝试多种匹配方式
    actual_dist_col = None
    # 1. 精确匹配（忽略大小写）
    for col in df.columns:
        if col.lower() == isotope_dist_col.lower():
            actual_dist_col = col
            break
    
    # 2. 尝试常见变体
    if not actual_dist_col:
        possible_dist_cols = [
            'Isotope Distribution',
            'isotope_distribution', 
            'Isotope_Distribution',
            'isotope distribution'
        ]
        for possible in possible_dist_cols:
            for col in df.columns:
                if col.lower() == possible.lower():
                    actual_dist_col = col
                    break
            if actual_dist_col:
                break
    
    # 3. 模糊匹配：包含"isotope"和"distrib"的列
    if not actual_dist_col:
        for col in df.columns:
            col_lower = col.lower()
            if 'isotope' in col_lower and 'distrib' in col_lower:
                actual_dist_col = col
                break
    
    if not actual_formula_col:
        print(f"[WARNING] 未找到分子式列: {formula_col}")
        df['isotope_similarity'] = ''
        return df
    
    if not actual_dist_col:
        print(f"[WARNING] 未找到同位素分布列: {isotope_dist_col}")
        df['isotope_similarity'] = ''
        return df
    
    print(f"[INFO] 使用分子式列: {actual_formula_col}")
    print(f"[INFO] 使用同位素分布列: {actual_dist_col}")
    
    similarities = []
    calculated_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="[同位素相似度计算]", unit="条", ncols=80):
        formula = row.get(actual_formula_col, '')
        measured_dist = row.get(actual_dist_col, '')
        
        if formula and measured_dist and str(formula).strip() and str(measured_dist).strip():
            try:
                theoretical_dist = calculate_theoretical_isotope_distribution(str(formula))
                measured_dist_parsed = parse_measured_isotope_distribution(str(measured_dist))
                
                if theoretical_dist and measured_dist_parsed:
                    similarity = calculate_isotope_similarity(theoretical_dist, measured_dist_parsed)
                    similarities.append(similarity)
                    calculated_count += 1
                else:
                    similarities.append('')
            except Exception as e:
                similarities.append('')
        else:
            similarities.append('')
    
    df['isotope_similarity'] = similarities
    
    valid_count = sum(1 for s in similarities if s != '')
    print(f"[INFO] 同位素相似度计算完成: {valid_count}/{len(df)} ({valid_count/len(df)*100:.1f}%)")
    
    return df


def load_isotope_from_sample_csv(sample_csv: str) -> Dict[str, str]:
    """
    从MSDIAL原始样品CSV加载同位素分布映射
    
    参数:
        sample_csv: MSDIAL导出的样品CSV文件路径
    
    返回:
        {compound_id: isotope_distribution_str} 映射字典
    """
    if not sample_csv or not os.path.exists(sample_csv):
        print(f"[WARNING] 样品CSV不存在: {sample_csv}")
        return {}
    
    isotope_map = {}
    try:
        # MSDIAL导出CSV有3行表头，header=2跳过前两行
        sample_df = pd.read_csv(sample_csv, header=2, encoding='utf-8-sig')
        
        if 'Compound' not in sample_df.columns or 'Isotope Distribution' not in sample_df.columns:
            print(f"[WARNING] CSV缺少 Compound 或 Isotope Distribution 列")
            return {}
        
        for _, row in sample_df.iterrows():
            cid = str(row.get('Compound', '')).strip()
            iso = str(row.get('Isotope Distribution', '')).strip()
            
            if cid and iso and iso != 'nan':
                isotope_map[cid] = iso
        
        print(f"[INFO] 从CSV加载 {len(isotope_map)} 条同位素分布")
        
    except Exception as e:
        print(f"[ERROR] 加载CSV失败: {e}")
    
    return isotope_map


def export_isotope_map(sample_csv: str, output_json: str) -> bool:
    """
    导出同位素分布映射为JSON文件（供L4 SIRIUS预处理使用）
    
    参数:
        sample_csv: 样品CSV文件路径
        output_json: 输出JSON文件路径
    
    返回:
        是否成功
    """
    print(f"\n{'='*60}")
    print("同位素分布预处理（导出JSON映射）")
    print(f"{'='*60}")
    print(f"[INFO] 输入CSV: {sample_csv}")
    print(f"[INFO] 输出JSON: {output_json}")
    
    # 加载同位素分布
    isotope_map = load_isotope_from_sample_csv(sample_csv)
    
    if not isotope_map:
        print("[WARNING] 未找到任何同位素分布数据")
        isotope_map = {}
    
    # 验证并统计
    valid_count = 0
    for cid, iso in isotope_map.items():
        parsed = parse_measured_isotope_distribution(iso)
        if len(parsed) >= 2:  # 至少需要M0和M1
            valid_count += 1
    
    print(f"[INFO] 有效同位素分布: {valid_count}/{len(isotope_map)}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    # 写入JSON
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(isotope_map, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 成功导出同位素映射: {output_json}")
        return True
    except Exception as e:
        print(f"[ERROR] 导出JSON失败: {e}")
        return False


def main():
    """主函数"""
    args = parse_arguments()
    
    # 模式判断
    if args.sample_csv and args.export_json:
        # 模式2：预处理模式
        success = export_isotope_map(args.sample_csv, args.export_json)
        return 0 if success else 1
    
    elif args.input:
        # 模式1：相似度计算模式
        print("=" * 60)
        print("同位素相似度计算")
        print("=" * 60)
        
        # 加载数据
        print(f"[INFO] 加载输入文件: {args.input}")
        df = pd.read_csv(args.input)
        print(f"[INFO] 共 {len(df)} 条记录")
        
        # 计算同位素相似度
        df = calculate_isotope_similarity_for_df(df, args.formula_col, args.isotope_dist_col)
        
        # 保存结果
        output_file = args.output or args.input
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n结果保存至: {output_file}")
        print("=" * 60)
        return 0
    
    else:
        print("[ERROR] 请指定模式：")
        print("  模式1（相似度计算）: --input <CSV> [--output <CSV>]")
        print("  模式2（预处理）: --sample_csv <CSV> --export_json <JSON>")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
