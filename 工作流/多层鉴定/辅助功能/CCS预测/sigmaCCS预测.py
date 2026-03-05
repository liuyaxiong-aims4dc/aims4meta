#!/usr/bin/env python3
"""
SigmaCCS预测脚本 - 离线CCS预测（使用SigmaCCS Python API）
"""

import argparse
import os
import sys
import pandas as pd
import tempfile
from pathlib import Path
from tqdm import tqdm

# SigmaCCS源代码路径
SIGMACC_DIR = "/stor3/AIMS4Meta/源代码/SigmaCCS"
SIGMA_SUPPORTED_ADDUCTS = ['[M+H]+', '[M+Na]+', '[M-H]-']

# 添加SigmaCCS路径
sys.path.insert(0, SIGMACC_DIR)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="SigmaCCS离线预测")
    parser.add_argument("--input", required=True, help="输入CSV文件")
    parser.add_argument("--output", help="输出CSV文件（默认在原文件名后加_suffix）")
    parser.add_argument("--smiles_col", default="smiles", help="SMILES列名")
    parser.add_argument("--adduct_col", default="adduct", help="加合物列名")
    return parser.parse_args()


def predict_with_sigma_ccs(input_csv, smiles_col="smiles", adduct_col="adduct"):
    """使用SigmaCCS Python API进行CCS预测"""
    
    # 读取数据
    df = pd.read_csv(input_csv)
    print(f"[INFO] 读取 {len(df)} 条记录")
    
    # 初始化结果列（使用统一列名）
    df['predicted_ccs'] = ""
    df['prediction_source'] = ""
    
    # 确定实际使用的SMILES列
    smiles_colnames = [smiles_col, 'matched_smiles', 'SMILES', 'candidate_smiles']
    actual_smiles_col = None
    for col in smiles_colnames:
        if col in df.columns:
            actual_smiles_col = col
            break
    
    if not actual_smiles_col:
        print(f"[ERROR] 未找到SMILES列（尝试过: {smiles_colnames}）")
        return df
    
    print(f"[INFO] 使用 '{actual_smiles_col}' 作为SMILES列")
    
    # 筛选有SMILES的数据
    valid_mask = df[actual_smiles_col].notna() & (df[actual_smiles_col] != "")
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) == 0:
        print("[WARN] 无有效SMILES数据")
        return df
    
    # 准备输入文件（SigmaCCS read_data() 要求 CSV 含 SMILES,Adduct,True CCS 三列）
    temp_dir = Path(tempfile.mkdtemp())
    input_file = temp_dir / "input.csv"
    output_file = temp_dir / "output.csv"
    
    # 检查加合物列
    actual_adduct_col = None
    for col in [adduct_col, 'adduct', 'Adduct', 'precursor_type', 'matched_adduct']:
        if col in df.columns:
            actual_adduct_col = col
            break
    
    # 写入CSV输入文件（含header，True CCS填0占位）
    rows_for_sigma = []
    unsupported_indices = set()
    for idx, row in valid_df.iterrows():
        smiles = row[actual_smiles_col]
        adduct = '[M+H]+'  # 默认
        if actual_adduct_col:
            adduct_val = row.get(actual_adduct_col, '[M+H]+')
            if pd.notna(adduct_val) and adduct_val:
                adduct = str(adduct_val)
        # 检查加合物是否被SigmaCCS支持
        if adduct not in SIGMA_SUPPORTED_ADDUCTS:
            unsupported_indices.add(idx)
            df.loc[idx, 'prediction_source'] = "unsupported_adduct"
            continue
        rows_for_sigma.append({'SMILES': smiles, 'Adduct': adduct, 'True CCS': 0})
    
    if not rows_for_sigma:
        print("[WARN] 无有效化合物可提交SigmaCCS预测")
        return df
    
    sigma_input_df = pd.DataFrame(rows_for_sigma)
    sigma_input_df.to_csv(input_file, index=False)
    
    # 统计信息
    total_input = len(df)
    no_smiles = total_input - len(valid_df)
    unsupported_adduct = len(unsupported_indices)
    ready_for_prediction = len(rows_for_sigma)
    
    print(f"\n{'='*60}")
    print("SigmaCCS 预测统计")
    print(f"{'='*60}")
    print(f"  输入总数:           {total_input:4d}")
    print(f"  - 无SMILES:         {no_smiles:4d} (跳过)")
    print(f"  - 不支持加合物:     {unsupported_adduct:4d} (跳过)")
    print(f"  待预测:             {ready_for_prediction:4d}")
    print(f"{'='*60}")
    
    try:
        # 使用SigmaCCS Python API
        from sigma.sigma import Model_prediction
        
        # 参数和模型路径
        param_path = f"{SIGMACC_DIR}/parameter/parameter.pkl"
        model_path = f"{SIGMACC_DIR}/model/model.h5"
        
        print(f"[INFO] 开始SigmaCCS预测 (含3D构象生成)...")
        
        # 执行预测
        Model_prediction(
            ifile=str(input_file),
            ParameterPath=param_path,
            mfileh5=model_path,
            ofile=str(output_file),
            Isevaluate=0
        )
        
        # 读取结果
        if output_file.exists():
            sigma_results = pd.read_csv(output_file)
            
            # 建立SMILES到CCS的映射
            smiles_to_ccs = {}
            for i, row in sigma_results.iterrows():
                smiles = row['SMILES']
                ccs_val = row.get('Predicted CCS', '')
                if ccs_val and pd.notna(ccs_val):
                    smiles_to_ccs[smiles] = ccs_val
            
            # 更新结果
            success_count = 0
            conformation_failed = 0
            for idx, row in valid_df.iterrows():
                smiles = row[actual_smiles_col]
                if smiles in smiles_to_ccs:
                    df.loc[idx, 'predicted_ccs'] = smiles_to_ccs[smiles]
                    df.loc[idx, 'prediction_source'] = "SigmaCCS"
                    success_count += 1
                else:
                    # 有SMILES但SigmaCCS未返回结果（3D构象生成失败等）
                    if idx not in unsupported_indices:
                        df.loc[idx, 'prediction_source'] = "conformation_failed"
                        conformation_failed += 1
            
            # 打印详细统计
            print(f"\n{'='*60}")
            print("SigmaCCS 预测结果")
            print(f"{'='*60}")
            print(f"  3D构象生成成功:     {len(sigma_results):4d}/{ready_for_prediction}")
            print(f"  CCS预测成功:        {success_count:4d}/{ready_for_prediction}")
            print(f"  - 构象生成失败:     {conformation_failed:4d}")
            print(f"  - 其他失败:         {ready_for_prediction - len(sigma_results):4d}")
            print(f"{'='*60}")
            print(f"  总体成功率:         {success_count}/{total_input} ({success_count/total_input*100:.1f}%)")
            print(f"{'='*60}\n")
        else:
            print("[WARN] SigmaCCS未生成输出文件")
            df.loc[valid_mask, 'prediction_source'] = "prediction_failed"
    
    except Exception as e:
        print(f"[ERROR] SigmaCCS预测失败: {e}")
        df.loc[valid_mask, 'prediction_source'] = f"error: {str(e)[:50]}"
    
    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return df


def main():
    """主函数"""
    args = parse_arguments()
    
    print("=" * 60)
    print("SigmaCCS离线预测")
    print("=" * 60)
    
    # 执行预测
    result_df = predict_with_sigma_ccs(args.input, args.smiles_col, args.adduct_col)
    
    # 保存结果
    output_file = args.output or args.input.replace('.csv', '_sigmaCCS.csv')
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    
    # 统计信息
    total = len(result_df)
    success = (result_df['prediction_source'] == 'SigmaCCS').sum()
    unsupported = (result_df['prediction_source'] == 'unsupported_adduct').sum()
    no_smiles = (result_df['prediction_source'] == '').sum()
    conf_failed = (result_df['prediction_source'] == 'conformation_failed').sum()
    other_failed = total - success - unsupported - no_smiles - conf_failed
    
    print(f"\n预测完成!")
    print(f"总化合物数: {total}")
    print(f"预测成功: {success} ({success/total*100:.1f}%)")
    if unsupported > 0:
        print(f"不支持加合物: {unsupported} ({unsupported/total*100:.1f}%)")
    if no_smiles > 0:
        print(f"无SMILES: {no_smiles} ({no_smiles/total*100:.1f}%)")
    if conf_failed > 0:
        print(f"3D构象失败: {conf_failed} ({conf_failed/total*100:.1f}%)")
    if other_failed > 0:
        print(f"其他失败: {other_failed} ({other_failed/total*100:.1f}%)")
    print(f"结果保存至: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
