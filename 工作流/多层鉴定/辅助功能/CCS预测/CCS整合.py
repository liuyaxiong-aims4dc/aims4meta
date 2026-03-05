#!/usr/bin/env python3
"""
CCS整合脚本 - 合并SigmaCCS和CCSBase预测结果
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CCS结果整合")
    parser.add_argument("--sigma_input", required=True, help="SigmaCCS预测结果CSV")
    parser.add_argument("--ccsbase_input", required=True, help="CCSBase预测结果CSV")
    parser.add_argument("--output", help="输出CSV文件（默认在sigma文件名后加_suffix）")
    parser.add_argument("--primary_source", default="sigma", choices=["sigma", "ccsbase"], 
                       help="主要来源（当两者都有预测时优先选择）")
    return parser.parse_args()


def load_results(file_path, source_name):
    """加载预测结果"""
    if not os.path.exists(file_path):
        print(f"[WARNING] {source_name}结果文件不存在: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    print(f"[INFO] 加载{source_name}结果: {len(df)}条记录")
    return df


def integrate_ccs_results(sigma_df, ccsbase_df, primary_source="sigma"):
    """
    整合CCS预测结果
    
    列说明:
    - library_ccs: MSDIAL数据库中的实测CCS值（原始值，不被覆盖）
    - predicted_ccs: SigmaCCS/CCSBase预测值
    - prediction_source: 预测来源(SigmaCCS/CCSBase/None)
    """
    if sigma_df is None and ccsbase_df is None:
        raise ValueError("两个输入文件都不存在")
    
    # 如果只有一个有结果，直接使用
    if sigma_df is None:
        return ccsbase_df
    
    if ccsbase_df is None:
        return sigma_df
    
    # 两个都有结果，需要整合
    print("[INFO] 整合两个来源的CCS预测结果...")
    
    # 确保两个DataFrame有相同的索引
    if len(sigma_df) != len(ccsbase_df):
        print("[WARNING] 两个结果文件行数不一致，可能存在数据错位")
    
    # 创建结果DataFrame（以ccsbase_df为准，因为它包含SigmaCCS的结果）
    result_df = ccsbase_df.copy()
    
    # 整合逻辑：CCSBase只补充SigmaCCS失败的记录
    sigma_success = result_df['prediction_source'] == 'SigmaCCS'
    ccsbase_success = result_df['prediction_source'] == 'CCSBase'
    
    # 统计信息
    sigma_count = sigma_success.sum()
    ccsbase_count = ccsbase_success.sum()
    none_count = (result_df['prediction_source'] == '').sum()
    total = len(result_df)
    
    # 统计已有数据库CCS的记录
    has_library_ccs = result_df['library_ccs'].notna().sum() if 'library_ccs' in result_df.columns else 0
    
    print(f"整合统计:")
    print(f"  有数据库CCS值:     {has_library_ccs} ({has_library_ccs/total*100:.1f}%)")
    print(f"  SigmaCCS预测成功:  {sigma_count} ({sigma_count/total*100:.1f}%)")
    print(f"  CCSBase预测成功:   {ccsbase_count} ({ccsbase_count/total*100:.1f}%)")
    print(f"  无预测:            {none_count} ({none_count/total*100:.1f}%)")
    
    return result_df


def calculate_ccs_deviations(df, measured_ccs_col="sample_ccs"):
    """
    计算CCS相对偏差
    
    计算两列偏差:
    - library_ccs_deviation_pct: (数据库CCS - 样品CCS) / 样品CCS * 100%
    - predicted_ccs_deviation_pct: (预测CCS - 样品CCS) / 样品CCS * 100%
    """
    
    # 自动识别实测CCS列名（支持多种命名）
    possible_ccs_cols = ['sample_ccs', 'CCS (angstrom^2)', 'CCS', 'Measured CCS']
    actual_ccs_col = None
    for col in possible_ccs_cols:
        if col in df.columns:
            actual_ccs_col = col
            break
    
    if actual_ccs_col is None:
        print(f"[INFO] 缺少实测CCS列，跳过偏差计算（需先完成CSV关联）")
        return df
    
    if actual_ccs_col != measured_ccs_col:
        print(f"[INFO] 使用实测CCS列: {actual_ccs_col}")
    
    meas_ccs = pd.to_numeric(df[actual_ccs_col], errors='coerce')
    valid_meas = meas_ccs.notna() & (meas_ccs > 0)
    
    if valid_meas.sum() == 0:
        print("[INFO] 无有效的实测CCS数据，跳过偏差计算")
        return df
    
    print(f"\n{'='*60}")
    print("CCS偏差计算")
    print(f"{'='*60}")
    
    # 1. 计算数据库CCS偏差 (library_ccs vs sample_ccs)
    if 'library_ccs' in df.columns:
        lib_ccs = pd.to_numeric(df['library_ccs'], errors='coerce')
        lib_valid = lib_ccs.notna() & valid_meas
        
        if lib_valid.sum() > 0:
            df.loc[lib_valid, 'library_ccs_deviation_pct'] = (
                (lib_ccs[lib_valid] - meas_ccs[lib_valid]) / meas_ccs[lib_valid] * 100
            ).round(2)
            
            lib_dev = df.loc[lib_valid, 'library_ccs_deviation_pct']
            print(f"数据库CCS偏差:")
            print(f"  有效记录: {lib_valid.sum()}")
            print(f"  平均偏差: {lib_dev.mean():.2f}%")
            print(f"  中位数偏差: {lib_dev.median():.2f}%")
            print(f"  平均绝对偏差: {lib_dev.abs().mean():.2f}%")
        else:
            print("数据库CCS偏差: 无有效数据")
    
    # 2. 计算预测CCS偏差 (predicted_ccs vs sample_ccs)
    if 'predicted_ccs' in df.columns:
        pred_ccs = pd.to_numeric(df['predicted_ccs'], errors='coerce')
        pred_valid = pred_ccs.notna() & valid_meas
        
        if pred_valid.sum() > 0:
            df.loc[pred_valid, 'predicted_ccs_deviation_pct'] = (
                (pred_ccs[pred_valid] - meas_ccs[pred_valid]) / meas_ccs[pred_valid] * 100
            ).round(2)
            
            pred_dev = df.loc[pred_valid, 'predicted_ccs_deviation_pct']
            print(f"\n预测CCS偏差:")
            print(f"  有效记录: {pred_valid.sum()}")
            print(f"  平均偏差: {pred_dev.mean():.2f}%")
            print(f"  中位数偏差: {pred_dev.median():.2f}%")
            print(f"  平均绝对偏差: {pred_dev.abs().mean():.2f}%")
        else:
            print("\n预测CCS偏差: 无有效数据")
    
    print(f"{'='*60}\n")
    
    return df


def main():
    """主函数"""
    args = parse_arguments()
    
    print("=" * 60)
    print("CCS预测结果整合")
    print("=" * 60)
    
    # 1. 加载结果
    sigma_results = load_results(args.sigma_input, "SigmaCCS")
    ccsbase_results = load_results(args.ccsbase_input, "CCSBase")
    
    # 2. 整合结果
    combined_results = integrate_ccs_results(sigma_results, ccsbase_results, args.primary_source)
    
    # 3. 计算CCS偏差（计算数据库CCS和预测CCS与样品CCS的偏差）
    combined_results = calculate_ccs_deviations(combined_results)
    
    # 4. 保存结果
    output_file = args.output or args.sigma_input.replace('_sigmaCCS.csv', '_ccs_combined.csv')
    combined_results.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n整合完成!")
    print(f"结果保存至: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()