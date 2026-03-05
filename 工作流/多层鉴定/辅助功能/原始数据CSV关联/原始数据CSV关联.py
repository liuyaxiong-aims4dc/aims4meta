#!/usr/bin/env python3
"""
原始数据CSV关联脚本 - 关联鉴定结果与原始样本数据

功能：
1. 关联鉴定结果CSV与原始样本CSV（原有功能）
2. 提取同位素分布信息并输出为JSON（新增功能，供L4使用）
"""

import argparse
import os
import sys
import pandas as pd
import re
import json
from typing import Optional


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="原始数据CSV关联")
    
    # 模式1：关联鉴定结果
    parser.add_argument("--input", help="输入鉴定结果CSV文件")
    parser.add_argument("--sample_csv", required=True, help="原始样本CSV文件")
    parser.add_argument("--output", help="输出CSV文件（默认在原文件名后加_suffix）")
    parser.add_argument("--query_name_pattern", default=r'\(([^)]+)\)', 
                       help="从query_name提取Compound ID的正则表达式")
    
    # 模式2：提取同位素信息
    parser.add_argument("--extract_isotope", action="store_true",
                       help="提取同位素分布信息模式")
    parser.add_argument("--output_json", help="输出JSON文件路径（提取同位素模式）")
    
    return parser.parse_args()


def extract_compound_id(query_name: str, pattern: str) -> Optional[str]:
    """从query_name中提取Compound ID"""
    if not isinstance(query_name, str):
        return None
    
    match = re.search(pattern, query_name)
    if match:
        return match.group(1)
    return None


def extract_isotope_distribution(sample_csv: str, output_json: str) -> bool:
    """
    从原始CSV提取同位素分布信息（供L4使用）
    
    参数:
        sample_csv: 原始样本CSV文件路径
        output_json: 输出JSON文件路径
    
    返回:
        是否成功
    """
    print(f"\n{'='*60}")
    print("提取同位素分布信息")
    print(f"{'='*60}")
    
    # 1. 加载原始样本数据
    print(f"[INFO] 加载原始样本数据: {sample_csv}")
    try:
        # MSDIAL导出CSV有3行表头，需要跳过前2行
        sample_df = pd.read_csv(sample_csv, header=2, encoding='utf-8')
        print(f"[INFO] 原始样本: {len(sample_df)} 条")
    except Exception as e:
        print(f"[ERROR] 加载原始样本数据失败: {e}")
        return False
    
    # 2. 查找Compound列和Isotope Distribution列
    compound_col = None
    isotope_col = None
    
    for col in sample_df.columns:
        col_lower = col.lower().strip()
        if col_lower == 'compound' or col_lower == 'compound id':
            compound_col = col
        if 'isotope' in col_lower and 'distribution' in col_lower:
            isotope_col = col
    
    if not compound_col:
        print(f"[ERROR] 未找到Compound列")
        print(f"[INFO] 可用列: {list(sample_df.columns[:10])}")
        return False
    
    if not isotope_col:
        print(f"[ERROR] 未找到Isotope Distribution列")
        print(f"[INFO] 可用列: {list(sample_df.columns[:10])}")
        return False
    
    print(f"[INFO] 使用列: '{compound_col}' 和 '{isotope_col}'")
    
    # 3. 提取同位素分布
    isotope_map = {}
    total_rows = len(sample_df)
    valid_rows = 0
    
    for _, row in sample_df.iterrows():
        compound_id = str(row[compound_col]).strip()
        iso_str = str(row[isotope_col]).strip()
        
        if not compound_id or compound_id == 'nan' or compound_id == '':
            continue
        
        if not iso_str or iso_str == 'nan' or iso_str == '':
            continue
        
        # 解析 "100 - 89.5 - 63" 格式
        try:
            parts = [float(x.strip()) for x in iso_str.replace('−', '-').split('-') if x.strip()]
            if parts:
                isotope_map[compound_id] = parts
                valid_rows += 1
        except ValueError:
            continue
    
    print(f"[INFO] 提取同位素分布: {len(isotope_map)} 条有效 (共{total_rows}行，{valid_rows}行有效)")
    
    # 4. 保存为JSON
    print(f"[INFO] 保存JSON: {output_json}")
    try:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(isotope_map, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 成功保存 {len(isotope_map)} 条同位素分布")
        return True
    except Exception as e:
        print(f"[ERROR] 保存失败: {e}")
        return False


def link_csv_data(input_csv: str, sample_csv: str, output_csv: str, pattern: str) -> bool:
    """
    关联鉴定结果CSV与原始样本CSV
    
    参数:
        input_csv: 鉴定结果CSV文件路径
        sample_csv: 原始样本CSV文件路径
        output_csv: 输出CSV文件路径
        pattern: 提取Compound ID的正则表达式
    
    返回:
        是否成功
    """
    print(f"\n{'='*60}")
    print("原始数据CSV关联")
    print(f"{'='*60}")
    
    # 1. 加载鉴定结果
    print(f"[INFO] 加载鉴定结果: {input_csv}")
    try:
        result_df = pd.read_csv(input_csv)
        print(f"[INFO] 鉴定结果: {len(result_df)} 条")
    except Exception as e:
        print(f"[ERROR] 加载鉴定结果失败: {e}")
        return False
    
    # 2. 加载原始样本数据
    print(f"[INFO] 加载原始样本数据: {sample_csv}")
    try:
        # MSDIAL导出CSV有3行表头：
        # 第1行：列类型（Normalised abundance/Raw abundance）
        # 第2行：条件名称
        # 第3行：实际列名
        sample_df = pd.read_csv(sample_csv, header=2)
        
        # 同时读取第1行（列类型）用于识别峰强度列
        header_df = pd.read_csv(sample_csv, header=0, nrows=0)
        header_types = list(header_df.columns)
        
        print(f"[INFO] 原始样本: {len(sample_df)} 条")
    except Exception as e:
        print(f"[ERROR] 加载原始样本数据失败: {e}")
        return False
    
    # 3. 从鉴定结果中提取Compound ID
    # 支持多种列名：query_name（L1-L3标准）, id/featureId（L4 SIRIUS）
    print("[INFO] 提取Compound ID...")
    
    # 确定用于提取 compound_id 的源列
    name_col = None
    for candidate in ['query_name', 'id', 'featureId', 'feature_id', 'compoundId']:
        if candidate in result_df.columns:
            name_col = candidate
            break
    
    if name_col:
        result_df['compound_id'] = result_df[name_col].apply(
            lambda x: extract_compound_id(x, pattern)
        )
        extracted_count = result_df['compound_id'].notna().sum()
        print(f"[INFO] 从 '{name_col}' 提取Compound ID: {extracted_count}/{len(result_df)} 条")
    else:
        print(f"[ERROR] 未找到 query_name/id 等列，无法提取 Compound ID")
        print(f"[INFO] 可用列: {list(result_df.columns)}")
        return False
    
    # 4. 关联数据
    print("[INFO] 关联原始数据...")
    
    # 确定样本CSV中的ID列名
    id_column = None
    possible_id_columns = ['Compound', 'Compound ID', 'compound_id', 'ID', 'id', 'Peak ID', 'peak_id']
    for col in possible_id_columns:
        if col in sample_df.columns:
            id_column = col
            break
    
    if id_column is None:
        print(f"[ERROR] 无法在样本CSV中找到ID列，可用列: {list(sample_df.columns)}")
        return False
    
    print(f"[INFO] 使用ID列: {id_column}")
    
    # 通过 compound_id 精确匹配（MSP与CSV一一对应，精确匹配即可100%关联）
    merged_df = result_df.merge(
        sample_df,
        left_on='compound_id',
        right_on=id_column,
        how='left',
        suffixes=('', '_sample')
    )
    matched_count = merged_df[id_column + '_sample'].notna().sum() if id_column + '_sample' in merged_df.columns else merged_df[id_column].notna().sum()
    print(f"[INFO] Compound ID 精确匹配: {matched_count}/{len(result_df)} 条")
    
    # 确保基础信息列存在（即使原始数据中缺失）
    base_columns = [
        'Retention time (min)',
        'CCS (angstrom^2)',
        'Isotope Distribution'
    ]
    for col in base_columns:
        if col not in merged_df.columns:
            merged_df[col] = ''
    
    # 5. 处理关联后的列
    # 删除重复的ID列
    if id_column in merged_df.columns and id_column != 'compound_id':
        merged_df = merged_df.drop(columns=[id_column])
    
    # 删除用于关联的临时列
    if 'compound_id' in merged_df.columns:
        merged_df = merged_df.drop(columns=['compound_id'])
    
    # 删除不需要的原始样本列
    columns_to_drop = [
        'Neutral mass (Da)', 'm/z', 'Charge',
        'Chromatographic peak width (min)', 'Identifications',
        'Minimum CV%',
        'Accepted Compound ID', 'Accepted Description',
        'Adducts', 'Formula', 'Score', 'Fragmentation Score',
        'Mass Error (ppm)', 'Isotope Similarity',
        'Retention Time Error (mins)', 'dCCS (angstrom^2)', 'Compound Link'
    ]
    
    for col in list(merged_df.columns):
        # 删除明确不需要的列
        if col in columns_to_drop or col.endswith('_sample'):
            if col in merged_df.columns:
                merged_df = merged_df.drop(columns=[col])
    
    # 6. 保存结果
    print(f"[INFO] 保存关联结果: {output_csv}")
    try:
        merged_df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"[INFO] 关联完成，共 {len(merged_df)} 条记录，{len(merged_df.columns)} 列")
        return True
    except Exception as e:
        print(f"[ERROR] 保存失败: {e}")
        return False


def main():
    """主函数"""
    args = parse_arguments()
    
    # 模式2：提取同位素信息
    if args.extract_isotope:
        if not args.output_json:
            print("[ERROR] 提取同位素模式需要指定 --output_json")
            return 1
        success = extract_isotope_distribution(args.sample_csv, args.output_json)
        return 0 if success else 1
    
    # 模式1：关联鉴定结果
    if not args.input:
        print("[ERROR] 关联模式需要指定 --input")
        return 1
    
    # 确定输出文件路径
    if args.output:
        output_csv = args.output
    else:
        # 默认在原文件名后加_suffix
        base, ext = os.path.splitext(args.input)
        output_csv = f"{base}_linked{ext}"
    
    success = link_csv_data(args.input, args.sample_csv, output_csv, args.query_name_pattern)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
