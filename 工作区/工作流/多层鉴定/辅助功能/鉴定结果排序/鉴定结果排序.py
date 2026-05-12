#!/usr/bin/env python3
"""
鉴定结果排序工具

功能：
- 对L1和L2鉴定结果进行综合排序
- 计算综合评分，考虑多个指标
- 支持Top-K结果筛选
- 输出排序后的结果
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
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


def calculate_theoretical_isotope_distribution(formula: str, max_mz_diff: int = 4) -> list:
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


def parse_measured_isotope_distribution(dist_str: str) -> list:
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


def calculate_isotope_similarity(theoretical: list, 
                                 measured: list) -> float:
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


def calculate_isotope_similarity_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """为DataFrame计算同位素相似度"""
    
    # 查找实际列名
    actual_formula_col = None
    for col in df.columns:
        if 'formula' in col.lower() and 'matched' in col.lower():
            actual_formula_col = col
            break
    
    # 同位素分布列名可能有多种形式，尝试多种鉴定方式
    actual_dist_col = None
    # 1. 精确鉴定（忽略大小写）
    for col in df.columns:
        if 'isotope' in col.lower() and 'distrib' in col.lower():
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
    
    if not actual_formula_col:
        print(f"[WARNING] 未找到分子式列")
        df['isotope_similarity'] = 0.0
        return df
    
    if not actual_dist_col:
        # 各层独立排序时尚未执行CSV关联，同位素分布列天然缺失
        # 最终汇总阶段会统一重新计算，此处静默跳过
        df['isotope_similarity'] = 0.0
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
                    similarities.append(0.0)
            except Exception as e:
                similarities.append(0.0)
        else:
            similarities.append(0.0)
    
    df['isotope_similarity'] = similarities
    
    valid_count = sum(1 for s in similarities if s > 0)
    print(f"[INFO] 同位素相似度计算完成: {valid_count}/{len(df)} ({valid_count/len(df)*100:.1f}%)")
    
    return df

def calculate_comprehensive_score(row, level):
    """
    计算综合评分
    
    Args:
        row: 数据行
        level: 鉴定层级 (L1/L2)
        
    Returns:
        综合评分
    """
    # 检查是否有实测CCS
    has_measured_ccs = False
    measured_ccs = None
    predicted_ccs = None
    ccs_deviation_pct = None
    
    if 'CCS (angstrom^2)' in row:
        ccs_val = row['CCS (angstrom^2)']
        if pd.notna(ccs_val) and ccs_val != '' and str(ccs_val).strip() != '':
            try:
                measured_ccs = float(ccs_val)
                has_measured_ccs = True
            except (ValueError, TypeError):
                pass
    
    if 'predicted_ccs' in row:
        pred_val = row['predicted_ccs']
        if pd.notna(pred_val) and pred_val != '' and str(pred_val).strip() != '':
            try:
                predicted_ccs = float(pred_val)
            except (ValueError, TypeError):
                pass
    
    if 'predicted_ccs_deviation_pct' in row:
        dev_val = row['predicted_ccs_deviation_pct']
        if pd.notna(dev_val) and dev_val != '' and str(dev_val).strip() != '':
            try:
                ccs_deviation_pct = float(dev_val)
            except (ValueError, TypeError):
                pass
    
    # 根据是否有实测CCS和RT调整权重
    if has_measured_ccs:
        # 有实测CCS时，引入CCS偏差项，重新分配权重
        weights = {
                  'cosine_score': 0.25,
                  'matched_peaks': 0.15,
                  'precursor_ppm_diff': 0.30,
                  'isotope_similarity': 0.15,
                  'ccs_deviation': 0.15
              }
    else:
        # 无实测CCS时，保持原权重，但加入RT评分
        weights = {
            'cosine_score': 0.30,
            'matched_peaks': 0.20,
            'precursor_ppm_diff': 0.30,
            'isotope_similarity': 0.20
        }
    
    # 初始化评分
    score = 0.0
    
    # 1. 余弦相似度 (0-1，越高越好)
    if 'cosine_score' in row:
        cosine_score = float(row['cosine_score'])
        score += cosine_score * weights['cosine_score']
    
    # 2. 匹配峰数 (归一化，越高越好)
    if 'matched_peaks_ratio' in row:
        # 新格式：匹配数/总峰数
        ratio_str = str(row['matched_peaks_ratio'])
        if '/' in ratio_str:
            try:
                matched, total = ratio_str.split('/')
                matched_peaks = int(matched)
                # 假设10个峰为满分
                peaks_score = min(1.0, matched_peaks / 10.0)
                score += peaks_score * weights.get('matched_peaks', 0.15)
            except (ValueError, ZeroDivisionError):
                pass
    elif 'matched_peaks' in row:
        # 旧格式：直接是数字
        matched_peaks = int(row['matched_peaks'])
        # 假设10个峰为满分
        peaks_score = min(1.0, matched_peaks / 10.0)
        score += peaks_score * weights['matched_peaks']
    
    # 3. 母离子质量差 (越小越好)
    if 'precursor_ppm_diff' in row:
        ppm_diff = abs(float(row['precursor_ppm_diff']))
        # ±5ppm以内为满分，±5ppm-20ppm依次递减，超过±20ppm为0分
        if ppm_diff <= 5.0:
            ppm_score = 1.0
        elif ppm_diff <= 20.0:
            ppm_score = 1.0 - (ppm_diff - 5.0) / 15.0
        else:
            ppm_score = 0.0
        score += ppm_score * weights['precursor_ppm_diff']
    
    # 4. 同位素相似度 (如果有，越高越好)
    if 'isotope_similarity' in row and pd.notna(row['isotope_similarity']):
        isotope_score = float(row['isotope_similarity']) / 100.0  # 归一化到0-1
        score += isotope_score * weights['isotope_similarity']
    
    # 5. CCS偏差 (仅当有实测CCS时)
    if has_measured_ccs and 'ccs_deviation' in weights:
        if ccs_deviation_pct is not None:
            # CCS偏差：±2%以内为满分，2-10%依次递减，超过10%为0分
            ccs_dev_abs = abs(ccs_deviation_pct)
            if ccs_dev_abs <= 2.0:
                ccs_score = 1.0
            elif ccs_dev_abs <= 10.0:
                ccs_score = 1.0 - (ccs_dev_abs - 2.0) / 8.0
            else:
                ccs_score = 0.0
        elif measured_ccs is not None and predicted_ccs is not None:
            # 计算CCS偏差百分比
            ccs_dev_abs = abs((predicted_ccs - measured_ccs) / measured_ccs * 100)
            if ccs_dev_abs <= 2.0:
                ccs_score = 1.0
            elif ccs_dev_abs <= 10.0:
                ccs_score = 1.0 - (ccs_dev_abs - 2.0) / 8.0
            else:
                ccs_score = 0.0
        else:
            ccs_score = 0.0
        score += ccs_score * weights['ccs_deviation']
    

    
    return score


def sort_identification_results(input_csv, output_csv, level, top_k=10):
    """
    对鉴定结果进行排序
    
    Args:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径
        level: 鉴定层级 (L1/L2)
        top_k: 保留Top-K结果
        
    Returns:
        bool: 操作是否成功
    """
    try:
        # 读取输入文件
        df = pd.read_csv(input_csv)

        # 空数据检查
        if len(df) == 0:
            print(f"[INFO] 输入文件无数据，跳过排序")
            df.to_csv(output_csv, index=False, encoding='utf-8')
            return True

        # 计算同位素相似度
        df = calculate_isotope_similarity_for_df(df)
        

        
        # 计算综合评分
        df['comprehensive_score'] = df.apply(
            lambda row: calculate_comprehensive_score(row, level), axis=1
        )
        
        # 按样品化合物分组，每个样品化合物保留Top-K结果
        if 'query_name' in df.columns:
            # 按样品化合物分组，每组内按综合评分降序排序
            df_sorted = df.sort_values(['query_name', 'comprehensive_score'], ascending=[True, False])
            
            # 基于InChIKey去重，每个化合物只保留得分最高的一个
            deduplicated_rows = []
            
            # 查找InChIKey列
            inchikey_col = None
            for col in df.columns:
                if 'inchikey' in col.lower() and ('matched' in col.lower() or 'inchikey' == col.lower()):
                    inchikey_col = col
                    break
            
            if inchikey_col:
                print(f"[INFO] 使用InChIKey列进行去重: {inchikey_col}")
                
                # 按query_name分组
                for query_name, group in df_sorted.groupby('query_name'):
                    # 按InChIKey分组，保留每个InChIKey得分最高的结果
                    seen_inchikeys = set()
                    for _, row in group.iterrows():
                        inchikey = str(row.get(inchikey_col, '')).strip()
                        # 跳过空InChIKey
                        if inchikey and inchikey not in seen_inchikeys:
                            seen_inchikeys.add(inchikey)
                            deduplicated_rows.append(row)
            else:
                print("[WARNING] 未找到InChIKey列，跳过去重")
                # 直接使用排序结果
                deduplicated_rows = df_sorted.to_dict('records')
            
            # 转换回DataFrame
            df_deduplicated = pd.DataFrame(deduplicated_rows)
            
            # 再次按query_name分组，保留Top-K结果
            if 'query_name' in df_deduplicated.columns:
                df_topk = df_deduplicated.sort_values(['query_name', 'comprehensive_score'], ascending=[True, False])
                df_topk = df_topk.groupby('query_name').head(top_k).reset_index(drop=True)
            else:
                df_topk = df_deduplicated.sort_values('comprehensive_score', ascending=False).head(top_k)
        else:
            # 如果没有query_name列，则全局排序并保留Top-K
            df_sorted = df.sort_values('comprehensive_score', ascending=False)
            df_topk = df_sorted.head(top_k)
        
        # 保存结果
        df_topk.to_csv(output_csv, index=False, encoding='utf-8')
        
        print(f"排序完成: {len(df)} -> {len(df_topk)} 条结果")
        print(f"输出文件: {output_csv}")
        
        return True
    except Exception as e:
        print(f"排序失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="鉴定结果排序工具")
    parser.add_argument("--input", required=True, help="输入CSV文件路径")
    parser.add_argument("--output", required=True, help="输出CSV文件路径")
    parser.add_argument("--level", required=True, choices=['L1', 'L2'], help="鉴定层级")
    parser.add_argument("--top_k", type=int, default=10, help="保留Top-K结果")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 执行排序
    success = sort_identification_results(
        args.input, args.output, args.level, args.top_k
    )
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
