#!/usr/bin/env python3
"""CCSBase2在线CCS预测（分批提交，按加合物+SMILES精确映射）"""
import requests
import pandas as pd
import sys
import time
from io import StringIO

CCSBASE_URL = "https://ccsbase.net/multi_pred-ccsbase2"
BATCH_SIZE = 500  # 每批提交化合物数（避免504超时）
MAX_RETRIES = 2   # 失败重试次数


def predict_ccsbase2(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # 查找SMILES和adduct列
    smiles_col = next((c for c in ['matched_smiles', 'SMILES', 'smiles'] if c in df.columns), None)
    adduct_col = next((c for c in ['adduct', 'Adduct'] if c in df.columns), None)

    if not smiles_col:
        print("[ERROR] 未找到SMILES列")
        return

    # 准备提交数据：记录每行对应的(smiles, adduct)用于精确映射
    row_info = []  # [(idx, smiles, adduct), ...]
    for idx, row in df.iterrows():
        smiles = str(row[smiles_col]) if pd.notna(row.get(smiles_col)) else ""
        adduct = str(row[adduct_col]) if adduct_col and pd.notna(row.get(adduct_col)) else "[M+H]+"
        if smiles and smiles.strip() and smiles != 'nan':
            row_info.append((idx, smiles.strip(), adduct.strip()))

    if not row_info:
        print("[INFO] 无有效SMILES，跳过CCS预测")
        df.to_csv(output_csv, index=False)
        return

    # 分批提交
    all_results = {}  # (smiles, adduct) -> ccs_value
    total = len(row_info)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[INFO] 共 {total} 个化合物，分 {n_batches} 批提交（每批{BATCH_SIZE}）...")

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)
        batch = row_info[start:end]

        # 构建CSV
        csv_lines = ["Adduct,Smiles,Name"]
        for i, (idx, smiles, adduct) in enumerate(batch):
            csv_lines.append(f"{adduct},{smiles},compound_{start + i}")
        csv_content = "\n".join(csv_lines)

        # 提交（带重试）
        batch_success = False
        for attempt in range(MAX_RETRIES + 1):
            try:
                print(f"[INFO] 批次 {batch_idx + 1}/{n_batches}（{len(batch)}个化合物）...")
                response = requests.post(
                    CCSBASE_URL,
                    files={'file': ('input.csv', csv_content, 'text/csv')},
                    timeout=300
                )
                if response.status_code == 200:
                    batch_success = True
                    break
                elif response.status_code == 504:
                    print(f"[WARN] 批次 {batch_idx + 1} 超时(504)，重试 {attempt + 1}/{MAX_RETRIES}")
                    if attempt < MAX_RETRIES:
                        time.sleep(10)
                else:
                    print(f"[ERROR] 批次 {batch_idx + 1} 请求失败: {response.status_code}")
                    break
            except requests.exceptions.Timeout:
                print(f"[WARN] 批次 {batch_idx + 1} 连接超时，重试 {attempt + 1}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES:
                    time.sleep(10)
            except Exception as e:
                print(f"[ERROR] 批次 {batch_idx + 1} 异常: {e}")
                break

        if not batch_success:
            print(f"[WARN] 批次 {batch_idx + 1} 失败，跳过")
            continue

        # 解析结果：用(smiles, adduct)组合key映射
        try:
            results = pd.read_csv(StringIO(response.text))
            ccs_col = 'Predicted CCS (\u00c5\u00b2)'
            for _, rrow in results.iterrows():
                if pd.notna(rrow.get(ccs_col)):
                    key = (str(rrow['SMI']).strip(), str(rrow.get('Adduct', '')).strip())
                    all_results[key] = rrow[ccs_col]
        except Exception as e:
            print(f"[WARN] 批次 {batch_idx + 1} 结果解析失败: {e}")
            continue

    # 映射回原数据（按(smiles, adduct)精确匹配）
    df['predicted_ccs'] = ""
    for idx, smiles, adduct in row_info:
        # 优先精确匹配(smiles, adduct)
        key = (smiles, adduct)
        if key in all_results:
            df.loc[idx, 'predicted_ccs'] = all_results[key]
        else:
            # 回退：仅用SMILES匹配（兼容结果中无Adduct列的情况）
            for k, v in all_results.items():
                if k[0] == smiles:
                    df.loc[idx, 'predicted_ccs'] = v
                    break

    # 计算偏差
    df['predicted_ccs_deviation_pct'] = ""
    if 'CCS (angstrom^2)' in df.columns:
        for idx in df.index:
            measured = df.loc[idx, 'CCS (angstrom^2)']
            predicted = df.loc[idx, 'predicted_ccs']
            if pd.notna(measured) and predicted != '' and predicted != '':
                try:
                    deviation = (float(predicted) - float(measured)) / float(measured) * 100
                    df.loc[idx, 'predicted_ccs_deviation_pct'] = round(deviation, 2)
                except:
                    pass

    # 创建 ccs_combined 列（实测/预测CCS(偏差%)）
    df['ccs_combined'] = ""
    ccs_col = 'CCS (angstrom^2)'
    if ccs_col in df.columns:
        for idx in df.index:
            measured = df.loc[idx, ccs_col]
            predicted = df.loc[idx, 'predicted_ccs']
            has_measured = pd.notna(measured) and str(measured).strip() not in ('', 'nan')
            has_predicted = predicted not in ('', None) and str(predicted).strip() != ''
            if has_measured and has_predicted:
                deviation = abs(float(predicted) - float(measured)) / float(measured) * 100
                df.loc[idx, 'ccs_combined'] = f"{float(measured):.2f}/{float(predicted):.2f} ({deviation:.1f}%)"
            elif has_predicted:
                df.loc[idx, 'ccs_combined'] = f"-/{float(predicted):.2f}"
            elif has_measured:
                df.loc[idx, 'ccs_combined'] = f"{float(measured):.2f}/-"
    
    df.to_csv(output_csv, index=False)
    success = (df['predicted_ccs'] != "").sum()
    print(f"[INFO] CCS预测完成: {success}/{len(df)}（ccs_combined 已创建）")


if __name__ == "__main__":
    predict_ccsbase2(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
