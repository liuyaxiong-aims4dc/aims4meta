#!/usr/bin/env python3
"""
SpectraVerse labels.tsv 按 InChIKey 骨架分组，划分 train/val/test。

用法:
    # 只处理负离子（补齐负离子编码器）
    python split_dataset.py --mode neg

    # 只处理正离子
    python split_dataset.py --mode pos

    # 合并正负离子（混合训练）
    python split_dataset.py --mode both

输出（以 neg 为例）:
    data/splits/neg/labels.tsv       完整标签
    data/splits/neg/mist_split.tsv   MIST 格式 split 文件（name/split 两列）
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# 数据路径
NEG_TSV = "/stor1/AIMS4Meta/datasets/msflow_datasets/SpectraVerse/spectraverse_neg/labels.tsv"
NEG_SUBFORM_DIR = "/stor1/AIMS4Meta/datasets/msflow_datasets/SpectraVerse/spectraverse_neg/subformulae"
POS_TSV = "/stor1/AIMS4Meta/datasets/msflow_datasets/spectraverse_pos/labels.tsv"
POS_SUBFORM_DIR = "/stor1/AIMS4Meta/datasets/msflow_datasets/spectraverse_pos/subformulae"

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_BASE = SCRIPT_DIR / "data" / "splits"


def parse_args():
    parser = argparse.ArgumentParser(description="划分 SpectraVerse 数据集")
    parser.add_argument("--mode", choices=["neg", "pos", "both"], default="neg",
                        help="离子模式（默认: neg）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = str(OUTPUT_BASE / args.mode)

    # 按 mode 加载数据
    if args.mode == "neg":
        df = pd.read_csv(NEG_TSV, sep="\t", dtype=str)
        df["subformulae_dir"] = NEG_SUBFORM_DIR
    elif args.mode == "pos":
        df = pd.read_csv(POS_TSV, sep="\t", dtype=str)
        df["subformulae_dir"] = POS_SUBFORM_DIR
    else:  # both
        neg_df = pd.read_csv(NEG_TSV, sep="\t", dtype=str)
        neg_df["subformulae_dir"] = NEG_SUBFORM_DIR
        pos_df = pd.read_csv(POS_TSV, sep="\t", dtype=str)
        pos_df["subformulae_dir"] = POS_SUBFORM_DIR
        df = pd.concat([neg_df, pos_df], ignore_index=True)

    combined = df
    print(f"[{args.mode}] 总条数: {len(combined)}")

    # 过滤掉没有 subformulae JSON 的条目
    def json_exists(row):
        return (Path(row["subformulae_dir"]) / f"{row['spec']}.json").exists()

    print("检查 subformulae JSON 文件是否存在...")
    mask = combined.apply(json_exists, axis=1)
    missing = (~mask).sum()
    if missing > 0:
        print(f"  警告: {missing} 条缺少 JSON 文件，已跳过")
    combined = combined[mask].reset_index(drop=True)
    print(f"有效条数: {len(combined)}")

    # 按 InChIKey 骨架分组
    combined["skeleton"] = (
        combined["inchikey"].fillna("").str.split("-").str[0]
    )

    # 按 skeleton 划分（以 skeleton 为单位，防止数据泄露）
    skeletons = combined["skeleton"].unique()
    np.random.shuffle(skeletons)
    n = len(skeletons)
    n_train = int(n * args.train)
    n_val = int(n * args.val)

    train_sk = set(skeletons[:n_train])
    val_sk = set(skeletons[n_train : n_train + n_val])
    test_sk = set(skeletons[n_train + n_val :])

    train_df = combined[combined["skeleton"].isin(train_sk)].drop(columns=["skeleton", "subformulae_dir"])
    val_df   = combined[combined["skeleton"].isin(val_sk)].drop(columns=["skeleton", "subformulae_dir"])
    test_df  = combined[combined["skeleton"].isin(test_sk)].drop(columns=["skeleton", "subformulae_dir"])

    # 验证无交叉
    train_ik = set(train_df["inchikey"].dropna())
    val_ik   = set(val_df["inchikey"].dropna())
    test_ik  = set(test_df["inchikey"].dropna())
    assert len(train_ik & val_ik) == 0
    assert len(train_ik & test_ik) == 0
    assert len(val_ik & test_ik) == 0

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    combined.drop(columns=["skeleton"]).to_csv(f"{output_dir}/labels.tsv", sep="\t", index=False)

    # MIST 格式 split 文件（name/split 两列）
    split_rows = (
        [(r["spec"], "train") for _, r in train_df.iterrows()] +
        [(r["spec"], "val")   for _, r in val_df.iterrows()] +
        [(r["spec"], "test")  for _, r in test_df.iterrows()]
    )
    pd.DataFrame(split_rows, columns=["name", "split"]).to_csv(
        f"{output_dir}/mist_split.tsv", sep="\t", index=False
    )

    print(f"\n划分结果 [{args.mode}]:")
    print(f"  train: {len(train_df):>7} 条  ({len(train_sk)} skeletons)")
    print(f"  val:   {len(val_df):>7} 条  ({len(val_sk)} skeletons)")
    print(f"  test:  {len(test_df):>7} 条  ({len(test_sk)} skeletons)")
    print(f"\n输出: {output_dir}/")


if __name__ == "__main__":
    main()
