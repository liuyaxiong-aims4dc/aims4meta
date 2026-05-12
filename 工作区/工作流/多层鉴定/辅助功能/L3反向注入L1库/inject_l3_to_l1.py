#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L3 高置信度结构鉴定结果反向注入 L1 自建累积库

设计要点：
1. 阈值基于 Hoffmann et al. Nat Biotech 2022 (COSMIC 论文) FDR 映射表，
   默认 confidence_exact >= 0.64（≈ FDR 10%），论文实际使用值。
2. 累积库文件：/stor3/AIMS4Meta/数据库/实验数据库/L3_auto_library/L3_auto_library_{POS|NEG}.msp
   （正负模式分文件存储，与 L1 实验库的正负分离规范对齐）
3. InChIKey：SIRIUS/CSI:FingerID 仅输出 block1（14 字符骨架哈希）；
   本脚本用 RDKit 从 SMILES 重算完整 27 字符 InChIKey 写入，去重键仍用 block1。
4. 去重：按 InChIKey block1 唯一；跨样品重复命中保留 confidence_exact 最高版本。
5. MS2 谱图来源：L3_processed.msp（样品实测 MS2）。
6. 每条记录 COMMENT 含 source/sample/confidence/date/ion_mode，便于追溯与事后清理。

用法：
    python inject_l3_to_l1.py \
        --l3_identified_csv <path>/L3_results/L3_identified.csv \
        --l3_processed_msp  <path>/L3_results/L3_processed.msp \
        --sample_name 大黄 --ion_mode POS
"""
import os
import re
import sys
import argparse
import logging
from datetime import datetime

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_LIB_DIR = "/stor3/AIMS4Meta/数据库/实验数据库/L3_auto_library"
DEFAULT_THRESHOLD = 0.64  # Hoffmann et al. 2022 Nat Biotech, COSMIC FDR ~10%

# RDKit 可选依赖：用于从 SMILES 补全完整 InChIKey
try:
    from rdkit import Chem as _RDChem
    from rdkit.Chem import inchi as _RDInchi
    from rdkit import RDLogger as _RDLogger
    _RDLogger.DisableLog('rdApp.*')
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False


def smiles_to_full_inchikey(smiles: str) -> str:
    """从 SMILES 计算完整 27 字符 InChIKey，失败返回空串。"""
    if not _HAS_RDKIT or not smiles:
        return ''
    try:
        mol = _RDChem.MolFromSmiles(smiles)
        if mol is None:
            return ''
        ik = _RDInchi.MolToInchiKey(mol)
        return ik if ik else ''
    except Exception:
        return ''


def inchikey_block1(ik: str) -> str:
    """取 InChIKey 第一段（14 字符骨架哈希），兼容只给 block1 的输入。"""
    if not ik:
        return ''
    return ik.split('-')[0].strip().upper()


def extract_rt_from_query_name(qname: str) -> str:
    """从 'Unknown (9.96_445.1095m/z)' / 'xxx (21.37_374.2972n)' 提取 RT（分钟）"""
    if not qname:
        return ''
    m = re.search(r'\(([\d.]+)_[\d.]+[mn]', qname)
    return m.group(1) if m else ''


def strip_prefix(qname: str) -> str:
    """去掉 L3/L4 mappingFeatureId 前缀，得到与 MSP NAME 一致的短名。"""
    if not qname:
        return ''
    for sep in ('_L3_processed_', '_L4_processed_', '_processed_'):
        if sep in qname:
            return qname.split(sep, 1)[-1].strip()
    return qname.strip()


def parse_msp_to_dict(msp_path: str) -> dict:
    """读 MSP，返回 {NAME: [peak_line, ...]}"""
    peaks_map = {}
    if not os.path.exists(msp_path):
        return peaks_map

    current_name = None
    current_peaks = []
    collecting = False

    def _flush():
        if current_name and current_peaks:
            peaks_map[current_name] = list(current_peaks)

    with open(msp_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.rstrip('\n').rstrip('\r')
            if not line.strip():
                _flush()
                current_name = None
                current_peaks = []
                collecting = False
                continue
            up = line.upper()
            if up.startswith('NAME:'):
                _flush()
                current_name = line.split(':', 1)[1].strip()
                current_peaks = []
                collecting = False
            elif up.startswith('NUM PEAKS:'):
                collecting = True
            elif collecting:
                current_peaks.append(line)
    _flush()
    return peaks_map


def normalize_adduct(adduct_str) -> str:
    """'[M + H]+' -> '[M+H]+'"""
    if adduct_str is None or (isinstance(adduct_str, float) and pd.isna(adduct_str)):
        return ''
    return str(adduct_str).replace(' ', '').strip()


def _safe_str(v) -> str:
    if v is None:
        return ''
    if isinstance(v, float) and pd.isna(v):
        return ''
    s = str(v).strip()
    return '' if s.lower() == 'nan' else s


def build_msp_entry(row, peaks, sample_name, ion_mode) -> str:
    """构造单条 MSP 记录字符串（以空行结尾）"""
    name = _safe_str(row.get('matched_name')) or 'Unknown'
    smiles = _safe_str(row.get('matched_smiles'))
    raw_ik = _safe_str(row.get('matched_inchikey'))
    # 优先从 SMILES 用 RDKit 重算完整 27 字符 InChIKey；失败时 fallback 到 SIRIUS 给的 block1
    full_ik = smiles_to_full_inchikey(smiles)
    inchikey = full_ik if full_ik else raw_ik
    formula = _safe_str(row.get('matched_formula'))
    mz_val = row.get('precursor_mz')
    try:
        mz_str = f"{float(mz_val):.6f}" if mz_val is not None and not (isinstance(mz_val, float) and pd.isna(mz_val)) else ''
    except (TypeError, ValueError):
        mz_str = ''
    adduct = normalize_adduct(row.get('adduct'))
    rt = extract_rt_from_query_name(_safe_str(row.get('query_name')))
    ontology = _safe_str(row.get('matched_ontology'))
    conf = row.get('confidence_exact', '')
    try:
        conf_str = f"{float(conf):.3f}"
    except (TypeError, ValueError):
        conf_str = str(conf)
    source_db = _safe_str(row.get('source_database')) or 'CSI:FingerID'
    today = datetime.now().strftime('%Y-%m-%d')

    lines = [
        f"NAME: {name}",
        f"PRECURSORMZ: {mz_str}",
        f"PRECURSORTYPE: {adduct}",
        f"IONMODE: {'Positive' if ion_mode == 'POS' else 'Negative'}",
        f"FORMULA: {formula}",
        f"SMILES: {smiles}",
        f"INCHIKEY: {inchikey}",
    ]
    if rt:
        lines.append(f"RETENTIONTIME: {rt}")
    if ontology:
        lines.append(f"ONTOLOGY: {ontology}")
    comment = "; ".join([
        "source=L3_auto_accumulated",
        f"sample={sample_name}",
        f"ion_mode={ion_mode}",
        f"confidence_exact={conf_str}",
        f"via={source_db}",
        f"date={today}",
    ])
    lines.append(f"COMMENT: {comment}")
    lines.append(f"Num Peaks: {len(peaks)}")
    lines.extend(peaks)
    return '\n'.join(lines) + '\n\n'


def load_existing_library(msp_path: str) -> dict:
    """读累积库 -> {inchikey_block1: (confidence_float, full_record_str)}
    去重键统一用 block1（14 字符），兼容已有完整/block1-only 两种写法。"""
    existing = {}
    if not os.path.exists(msp_path):
        return existing

    buf = []
    cur_ikey_block1 = None
    cur_conf = -1.0

    def _flush():
        if buf and cur_ikey_block1:
            existing[cur_ikey_block1] = (cur_conf, '\n'.join(buf) + '\n\n')

    with open(msp_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.rstrip('\n').rstrip('\r')
            if not s.strip():
                _flush()
                buf = []
                cur_ikey_block1 = None
                cur_conf = -1.0
                continue
            buf.append(s)
            up = s.upper()
            if up.startswith('INCHIKEY:'):
                cur_ikey_block1 = inchikey_block1(s.split(':', 1)[1].strip())
            elif up.startswith('COMMENT:'):
                m = re.search(r'confidence_exact=([\d.]+)', s)
                if m:
                    try:
                        cur_conf = float(m.group(1))
                    except ValueError:
                        pass
    _flush()
    return existing


def main():
    parser = argparse.ArgumentParser(description="L3 高置信度结构鉴定结果反向注入 L1 自建累积库")
    parser.add_argument("--l3_identified_csv", required=True, help="L3_identified.csv 路径")
    parser.add_argument("--l3_processed_msp", required=True, help="L3_processed.msp 路径（提供 MS2 谱图）")
    parser.add_argument("--sample_name", required=True, help="样品名称（用于来源标注）")
    parser.add_argument("--ion_mode", choices=['POS', 'NEG'], required=True)
    parser.add_argument("--output_dir", default=DEFAULT_LIB_DIR,
                        help=f"累积库目录（默认 {DEFAULT_LIB_DIR}）")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"confidence_exact 入库阈值（默认 {DEFAULT_THRESHOLD}，"
                             f"对应 Hoffmann et al. 2022 FDR ~10%）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_msp = os.path.join(args.output_dir, f"L3_auto_library_{args.ion_mode}.msp")

    logger.info("=== L3 高置信度结果反向注入 L1 自建累积库 ===")
    logger.info(f"样品={args.sample_name}  |  模式={args.ion_mode}  |  阈值={args.threshold}")
    logger.info(f"目标库: {output_msp}  |  RDKit 补全 InChIKey: {'启用' if _HAS_RDKIT else '未安装(保留14位block1)'}")

    if not os.path.exists(args.l3_identified_csv):
        logger.warning(f"L3_identified.csv 不存在: {args.l3_identified_csv} —— 跳过注入")
        return 0
    if not os.path.exists(args.l3_processed_msp):
        logger.warning(f"L3_processed.msp 不存在: {args.l3_processed_msp} —— 跳过注入")
        return 0

    # 1. 读命中结果
    df = pd.read_csv(args.l3_identified_csv)
    total = len(df)
    logger.info(f"L3 命中总数: {total}")

    if 'confidence_exact' not in df.columns or 'matched_inchikey' not in df.columns:
        logger.warning("L3_identified.csv 缺少 confidence_exact / matched_inchikey 列，跳过")
        return 0

    # 2. 筛选阈值 + 有效 InChIKey
    conf = pd.to_numeric(df['confidence_exact'], errors='coerce').replace(
        [float('-inf'), float('inf')], float('nan'))
    df = df.assign(_conf=conf)
    df = df[df['_conf'] >= args.threshold]
    df = df.dropna(subset=['matched_inchikey'])
    df = df[df['matched_inchikey'].astype(str).str.strip().ne('')]
    df = df[df['matched_inchikey'].astype(str).str.strip().str.lower().ne('nan')]

    # 样品内按 InChIKey 去重（取最高 confidence）
    df = df.sort_values('_conf', ascending=False).drop_duplicates(
        subset='matched_inchikey', keep='first')
    logger.info(f"通过阈值 + 样品内去重后: {len(df)} 条")

    if len(df) == 0:
        logger.info("本样品无满足阈值的高置信度结果，未写入。")
        return 0

    # 3. MS2 谱图
    peaks_map = parse_msp_to_dict(args.l3_processed_msp)
    logger.info(f"L3_processed.msp 谱图数: {len(peaks_map)}")

    # 4. 已有累积库
    existing = load_existing_library(output_msp)
    before = len(existing)
    logger.info(f"累积库既有条数: {before}")

    # 5. 合并（去重键用 block1）
    added = 0
    updated = 0
    skipped_no_peaks = 0
    for _, row in df.iterrows():
        qname_full = _safe_str(row.get('query_name'))
        short_name = strip_prefix(qname_full)
        peaks = peaks_map.get(short_name) or peaks_map.get(qname_full)
        if not peaks:
            skipped_no_peaks += 1
            continue
        ikey_raw = _safe_str(row['matched_inchikey'])
        key = inchikey_block1(ikey_raw)
        if not key:
            continue
        new_conf = float(row['_conf'])
        row_for_build = row.drop('_conf', errors='ignore')
        row_for_build['confidence_exact'] = new_conf
        entry = build_msp_entry(row_for_build, peaks, args.sample_name, args.ion_mode)
        if key in existing:
            old_conf, _ = existing[key]
            if new_conf > old_conf:
                existing[key] = (new_conf, entry)
                updated += 1
        else:
            existing[key] = (new_conf, entry)
            added += 1

    logger.info(f"新增 {added} 条 | 覆盖更新（更高置信度）{updated} 条 | 无匹配MS2跳过 {skipped_no_peaks} 条")

    # 6. 写回（按 confidence 降序）
    with open(output_msp, 'w', encoding='utf-8') as f:
        for key, (conf_val, entry) in sorted(existing.items(),
                                              key=lambda kv: -kv[1][0]):
            f.write(entry)

    logger.info(f"累积库当前条数: {len(existing)}（之前 {before}，本次净增 {len(existing) - before}）")
    logger.info(f"已写出: {output_msp}")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
