#!/usr/bin/env python3
"""
CCSBase预测脚本 - 在线CCS预测（仅补充SigmaCCS失败的化合物）
"""

import argparse
import os
import sys
import requests
import pandas as pd
import time
from typing import List, Tuple

# CCSBase API配置
CCSBASE_MULTI_PRED_URL = "https://ccsbase.net/multi_pred"
CCSBASE_MAX_WORKERS = 5


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CCSBase在线预测（补充SigmaCCS失败项）")
    parser.add_argument("--input", required=True, help="输入CSV文件（SigmaCCS输出结果）")
    parser.add_argument("--output", help="输出CSV文件")
    parser.add_argument("--smiles_col", default="smiles", help="SMILES列名")
    parser.add_argument("--adduct_col", default="adduct", help="加合物列名")
    parser.add_argument("--name_col", default="compound_name", help="化合物名列")
    return parser.parse_args()


def find_column(df, candidates, label=""):
    """自动检测列名，返回实际列名或None"""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def prepare_ccsbase_csv(df: pd.DataFrame, smiles_col: str, adduct_col: str, 
                       name_col: str) -> str:
    """准备CCSBase批量提交的CSV内容"""
    # 自动检测实际列名
    actual_smiles_col = find_column(df, [smiles_col, 'matched_smiles', 'SMILES', 'smiles', 'candidate_smiles'])
    actual_adduct_col = find_column(df, [adduct_col, 'adduct', 'Adduct', 'precursor_type', 'matched_adduct'])
    actual_name_col = find_column(df, [name_col, 'query_name', 'matched_name', 'compound_name', 'Name'])
    
    if not actual_smiles_col:
        print(f"[ERROR] 未找到SMILES列")
        return ""
    
    print(f"[INFO] SMILES列: '{actual_smiles_col}', 加合物列: '{actual_adduct_col}', 名称列: '{actual_name_col}'")
    
    csv_lines = ["Adduct,Smiles,Name"]
    
    for idx, row in df.iterrows():
        smiles = str(row[actual_smiles_col]) if actual_smiles_col and pd.notna(row.get(actual_smiles_col)) else ""
        adduct = str(row[actual_adduct_col]) if actual_adduct_col and pd.notna(row.get(actual_adduct_col)) else "[M+H]+"
        name = str(row[actual_name_col]) if actual_name_col and pd.notna(row.get(actual_name_col)) else f"compound_{idx}"
        
        if smiles and smiles.strip() and smiles != 'nan':
            # CCSBase CSV中逗号需要处理（SMILES中一般无逗号，Name可能有）
            name = name.replace(',', ';')
            csv_lines.append(f"{adduct},{smiles},{name}")
    
    return "\n".join(csv_lines)


def submit_to_ccsbase(csv_content: str) -> str:
    """提交到CCSBase并获取预测结果"""
    try:
        response = requests.post(
            CCSBASE_MULTI_PRED_URL,
            files={'file': ('input.csv', csv_content, 'text/csv')},
            timeout=300  # 5分钟超时
        )
        
        if response.status_code == 200:
            return response.text
        else:
            print(f"[ERROR] CCSBase请求失败: {response.status_code}")
            return ""
            
    except Exception as e:
        print(f"[ERROR] CCSBase请求异常: {e}")
        return ""


def parse_ccsbase_response(response_text: str) -> pd.DataFrame:
    """解析CCSBase返回的CSV响应"""
    if not response_text.strip():
        return pd.DataFrame()
    
    try:
        # 直接从文本读取CSV
        from io import StringIO
        df = pd.read_csv(StringIO(response_text))
        return df
    except Exception as e:
        print(f"[ERROR] 解析CCSBase响应失败: {e}")
        return pd.DataFrame()


def predict_with_ccsbase(input_csv: str, smiles_col: str = "smiles", 
                        adduct_col: str = "adduct", name_col: str = "compound_name") -> pd.DataFrame:
    """使用CCSBase进行预测（仅补充SigmaCCS失败的化合物）"""
    df = pd.read_csv(input_csv)
    
    # 初始化结果列（如果还不存在）
    if 'predicted_ccs' not in df.columns:
        df['predicted_ccs'] = ""
    if 'prediction_source' not in df.columns:
        df['prediction_source'] = ""
    
    # 筛选需要CCSBase补充预测的行（SigmaCCS未成功的）
    if 'prediction_source' in df.columns:
        need_predict_mask = df['prediction_source'] != 'SigmaCCS'
        sigma_success_count = (~need_predict_mask).sum()
        print(f"[INFO] SigmaCCS已成功: {sigma_success_count} 条，跳过")
        print(f"[INFO] 需CCSBase补充预测: {need_predict_mask.sum()} 条")
    else:
        need_predict_mask = pd.Series(True, index=df.index)
        print("[INFO] 未检测到prediction_source列，对全部化合物预测")
    
    # 提取待预测子集
    predict_df = df[need_predict_mask].copy()
    
    if len(predict_df) == 0:
        print("[INFO] SigmaCCS已全部成功，无需CCSBase补充预测")
        return df
    
    # 自动检测实际列名
    actual_smiles_col = find_column(predict_df, [smiles_col, 'matched_smiles', 'SMILES', 'smiles', 'candidate_smiles'])
    actual_name_col = find_column(predict_df, [name_col, 'query_name', 'matched_name', 'compound_name', 'Name'])
    
    if not actual_smiles_col:
        print(f"[ERROR] 未找到SMILES列")
        return df
    
    # 过滤有SMILES的数据
    has_smiles = predict_df[actual_smiles_col].notna() & (predict_df[actual_smiles_col].astype(str).str.strip() != '') & (predict_df[actual_smiles_col].astype(str) != 'nan')
    submit_df = predict_df[has_smiles]
    
    if len(submit_df) == 0:
        print("[WARN] 待预测化合物无有效SMILES")
        return df
    
    # 准备CSV内容
    csv_content = prepare_ccsbase_csv(submit_df, smiles_col, adduct_col, name_col)
    
    if not csv_content.strip() or csv_content.count('\n') < 1:
        print("[WARNING] 没有有效的化合物用于预测")
        return df
    
    valid_count = csv_content.count('\n')
    print(f"[INFO] 准备提交 {valid_count} 个化合物到CCSBase...")
    
    # 提交预测
    print("[INFO] 提交到CCSBase进行预测...")
    response_text = submit_to_ccsbase(csv_content)
    
    if not response_text:
        df.loc[need_predict_mask, 'prediction_source'] = "ccsbase_submission_failed"
        return df
    
    # 解析结果
    ccsbase_results = parse_ccsbase_response(response_text)
    
    if ccsbase_results.empty:
        df.loc[need_predict_mask, 'prediction_source'] = "ccsbase_parsing_failed"
        return df
    
    print(f"[INFO] CCSBase返回 {len(ccsbase_results)} 条结果")
    print(f"[INFO] CCSBase返回列: {list(ccsbase_results.columns)}")
    
    # 自动检测CCS结果列（兼容 Predicted CCS (Å) 等各种命名）
    ccs_col = None
    for col in ccsbase_results.columns:
        col_lower = col.lower()
        if 'ccs' in col_lower and ('predict' in col_lower or col_lower == 'ccs'):
            ccs_col = col
            break
    if not ccs_col:
        # 再宽松匹配
        for col in ccsbase_results.columns:
            if 'ccs' in col.lower():
                ccs_col = col
                break
    
    # 检测SMILES结果列
    smi_result_col = None
    for col in ['SMI', 'Smiles', 'smiles', 'SMILES']:
        if col in ccsbase_results.columns:
            smi_result_col = col
            break
    
    # 检测Name结果列
    name_result_col = None
    for col in ['Name', 'name']:
        if col in ccsbase_results.columns:
            name_result_col = col
            break
    
    if not ccs_col:
        print(f"[WARNING] CCSBase返回中未找到CCS列，返回列: {list(ccsbase_results.columns)}")
        df.loc[need_predict_mask, 'prediction_source'] = "ccsbase_format_error"
        return df
    
    print(f"[INFO] CCS结果列: '{ccs_col}', SMILES列: '{smi_result_col}', Name列: '{name_result_col}'")
    
    # 构建SMILES→CCS映射（最可靠）
    smiles_ccs_map = {}
    if smi_result_col:
        for _, row in ccsbase_results.iterrows():
            smi = str(row[smi_result_col]).strip()
            val = row[ccs_col]
            if smi and pd.notna(val) and str(val).strip():
                smiles_ccs_map[smi] = val
    
    # 构建Name→CCS映射（备选）
    name_ccs_map = {}
    if name_result_col:
        for _, row in ccsbase_results.iterrows():
            name = str(row[name_result_col]).strip()
            val = row[ccs_col]
            if name and pd.notna(val) and str(val).strip():
                name_ccs_map[name] = val
    
    print(f"[INFO] CCSBase成功预测: SMILES映射{len(smiles_ccs_map)}条, Name映射{len(name_ccs_map)}条")
    
    # 映射回原始数据（仅映射need_predict_mask范围内的行）
    for idx in predict_df.index:
        if not has_smiles.get(idx, False):
            continue
        
        matched = False
        
        # 优先用SMILES匹配
        if actual_smiles_col and smiles_ccs_map:
            smi = str(df.loc[idx, actual_smiles_col]).strip()
            if smi in smiles_ccs_map:
                df.loc[idx, 'predicted_ccs'] = smiles_ccs_map[smi]
                df.loc[idx, 'prediction_source'] = "CCSBase"
                matched = True
        
        # 备选：Name匹配
        if not matched and actual_name_col and name_ccs_map:
            name_val = str(df.loc[idx, actual_name_col]).replace(',', ';').strip()
            if name_val in name_ccs_map:
                df.loc[idx, 'predicted_ccs'] = name_ccs_map[name_val]
                df.loc[idx, 'prediction_source'] = "CCSBase"
    
    return df


def main():
    """主函数"""
    args = parse_arguments()
    
    print("=" * 60)
    print("CCSBase在线预测")
    print("=" * 60)
    
    # 执行预测
    result_df = predict_with_ccsbase(args.input, args.smiles_col, args.adduct_col, args.name_col)
    
    # 保存结果
    output_file = args.output or args.input.replace('.csv', '_ccsbase.csv')
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    
    # 统计信息
    total = len(result_df)
    success = (result_df['prediction_source'] == 'CCSBase').sum()
    skipped = (result_df['prediction_source'] == 'SigmaCCS').sum()
    failed = total - success - skipped
    
    print(f"\n预测完成!")
    print(f"总化合物数: {total}")
    print(f"SigmaCCS已成功(跳过): {skipped} ({skipped/total*100:.1f}%)")
    print(f"CCSBase补充成功: {success} ({success/total*100:.1f}%)")
    print(f"预测失败: {failed} ({failed/total*100:.1f}%)")
    print(f"结果保存至: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()