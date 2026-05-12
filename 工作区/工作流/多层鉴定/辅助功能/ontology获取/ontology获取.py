#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIRIUS CANOPUS Ontology 分类获取脚本
用于获取化合物的分类信息（如黄酮类、甾体类等）

来源（无外部 API 依赖）：
  1. SIRIUS CANOPUS TSV 直读（本轮 L3 结果）
  2. ontology_sirius_cache.json 持久化缓存（跨 run 积累）
  3. CANOPUS 实时计算（为 L1/L2 来源但未进入 L3 的化合物单独运行 SIRIUS formula+canopus）
     启用条件：同时传入 --canopus_fallback --sample_msp --ion_mode

模式：
  - CSV 模式： --input_csv + --output_csv，为鉴定结果 CSV 填充 ontology 列
  - 缓存模式： --results_dir + --cache_only，扫描历史结果填充持久化缓存
"""

import json
import logging
from typing import Dict, Optional
import argparse
import pandas as pd
import os
import sys
import tempfile
import subprocess

# SIRIUS 默认路径（与 L3_SIRIUS_CLI.py 保持一致）
DEFAULT_SIRIUS_BIN = "/stor3/AIMS4Meta/源代码/SIRIUS/sirius-6.3.5-linux-x64/sirius/bin/sirius"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ontology_fetch.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# SIRIUS CANOPUS 列名映射（CANOPUS TSV 列名 → 分类层级）
CANOPUS_FIELD_MAP = {
    'superclass':    'ClassyFire#superclass',
    'class':         'ClassyFire#class',
    'subclass':      'ClassyFire#subclass',
    'direct_parent': 'ClassyFire#most specific class',
}

# SIRIUS 累积缓存文件
SIRIUS_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ontology_sirius_cache.json")


def _load_sirius_cache() -> dict:
    """加载 SIRIUS CANOPUS 累积缓存

    缓存格式: {f"{smiles}_{field}_sirius": "Flavonoids"}
    """
    if not os.path.exists(SIRIUS_CACHE_FILE):
        return {}
    try:
        with open(SIRIUS_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载 SIRIUS 缓存失败: {e}")
        return {}


def _save_sirius_cache(cache: dict):
    """保存 SIRIUS 累积缓存"""
    try:
        with open(SIRIUS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存 SIRIUS 缓存失败: {e}")


def load_sirius_canopus_ontology(sirius_results_dir: str, classification_field: str = "class") -> Dict[str, str]:
    """从 SIRIUS CANOPUS 输出加载分类

    - 优先 canopus_structure_summary.tsv（结构级）
    - fallback canopus_formula_summary.tsv（formula 级）
    - 只取 formulaRank==1 的主候选
    - 返回 {query_name: classification}，query_name 形如 "Unknown (RT_mz)"

    不需联网；秒级完成；覆盖率 = L3 跑过 SIRIUS 的 compound 数量
    """
    col_name = CANOPUS_FIELD_MAP.get(classification_field)
    if not col_name:
        logger.info(f"[CANOPUS] 分类层级 '{classification_field}' 在 CANOPUS TSV 中无对应列，跳过")
        return {}
    if not sirius_results_dir or not os.path.isdir(sirius_results_dir):
        return {}

    mapping: Dict[str, str] = {}

    def _extract_query_name(feat_id: str) -> str:
        """从 mappingFeatureId 提取 query_name"""
        if not feat_id:
            return ''
        for sep in ('_L3_processed_', '_L4_processed_', '_processed_'):
            if sep in feat_id:
                return feat_id.split(sep)[-1].strip()
        return feat_id.strip()

    def _load_tsv(tsv_path: str, tag: str):
        if not os.path.exists(tsv_path):
            return 0
        try:
            tdf = pd.read_csv(tsv_path, sep='\t', dtype=str, keep_default_na=False)
        except Exception as e:
            logger.warning(f"[CANOPUS] 解析 {tag} 失败: {e}")
            return 0
        if col_name not in tdf.columns or 'mappingFeatureId' not in tdf.columns:
            logger.warning(f"[CANOPUS] {tag} 缺少必要列（需要 {col_name} 和 mappingFeatureId）")
            return 0
        if 'formulaRank' in tdf.columns:
            tdf = tdf[tdf['formulaRank'].astype(str).str.strip() == '1']
        added = 0
        for _, row in tdf.iterrows():
            cls = str(row.get(col_name, '')).strip()
            if not cls:
                continue
            feat_id = str(row.get('mappingFeatureId', '')).strip()
            if not feat_id:
                continue
            qname = _extract_query_name(feat_id)
            for k in (feat_id, qname):
                if k and k not in mapping:
                    mapping[k] = cls
                    added += 1
        return added

    struct_added = _load_tsv(
        os.path.join(sirius_results_dir, 'canopus_structure_summary.tsv'),
        'canopus_structure_summary.tsv')
    if struct_added:
        logger.info(f"[CANOPUS] 从 canopus_structure_summary.tsv 载入 {struct_added} 条 {classification_field} 分类（结构级）")

    formula_added = _load_tsv(
        os.path.join(sirius_results_dir, 'canopus_formula_summary.tsv'),
        'canopus_formula_summary.tsv')
    if formula_added:
        logger.info(f"[CANOPUS] 从 canopus_formula_summary.tsv 追加 {formula_added} 条 {classification_field} 分类（formula 级 fallback）")

    return mapping


def load_canopus_tsv_all(results_dir: str) -> pd.DataFrame:
    """从多层鉴定结果目录加载 CANOPUS TSV（搜索子目录）

    返回合并后的 DataFrame，优先结构级，formula 级补充。
    """
    dfs = []
    for tsv_name in ["canopus_structure_summary.tsv", "canopus_formula_summary.tsv"]:
        found = False
        for root, dirs, files in os.walk(results_dir):
            if tsv_name in files:
                path = os.path.join(root, tsv_name)
                try:
                    tdf = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
                    dfs.append(tdf)
                    logger.info(f"  [CANOPUS] 加载 {path} ({len(tdf)} 行)")
                    found = True
                except Exception as e:
                    logger.warning(f"  [CANOPUS] 读取 {path} 失败: {e}")
                break
        if found:
            break  # 优先结构级，找到即停
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['mappingFeatureId'], keep='first')
    return df


def extract_all_classifications(canopus_df: pd.DataFrame) -> dict:
    """从 CANOPUS TSV 提取所有层级分类

    Returns: {mappingFeatureId: {"superclass": "...", "class": "...", ...}}
    """
    results = {}
    for _, row in canopus_df.iterrows():
        feat_id = str(row.get("mappingFeatureId", "")).strip()
        if not feat_id:
            continue
        entry = {}
        for field, col in CANOPUS_FIELD_MAP.items():
            val = str(row.get(col, "")).strip()
            if val and val not in ("nan", "NaN", ""):
                entry[field] = val
        if entry:
            results[feat_id] = entry
    return results


def build_name_to_smiles_from_csv(results_dir: str) -> dict:
    """从 L1/L2/L3/总结果 CSV 中提取 NAME → SMILES 映射"""
    name_to_smiles = {}
    for fname in ["L1_results.csv", "L2_results.csv", "L3_results.csv", "多层鉴定总结果.csv"]:
        csv_path = os.path.join(results_dir, fname)
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", dtype=str, keep_default_na=False)
        except Exception:
            continue
        name_col = None
        for c in ["query_name", "matched_name", "target_name"]:
            if c in df.columns:
                name_col = c
                break
        smiles_col = None
        for c in ["matched_smiles", "target_smiles", "smiles"]:
            if c in df.columns:
                smiles_col = c
                break
        if not name_col or not smiles_col:
            continue
        for _, row in df.iterrows():
            name = str(row[name_col]).strip()
            smiles = str(row[smiles_col]).strip()
            if name and name != "nan" and smiles and smiles != "nan":
                if name not in name_to_smiles:
                    name_to_smiles[name] = smiles
    return name_to_smiles


def build_name_to_smiles_from_msp(results_dir: str) -> dict:
    """从 L3 SIRIUS 输入 MSP 中提取 NAME → SMILES 映射"""
    name_to_smiles = {}
    msp_files = []
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.endswith(".msp") and "sirius" in f.lower():
                path = os.path.join(root, f)
                msp_files.append(path)
                logger.info(f"  [MSP] 发现: {path}")
    for msp_path in msp_files:
        current_name = ""
        current_smiles = ""
        try:
            with open(msp_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    stripped = line.strip()
                    lower = stripped.lower()
                    if lower.startswith("name:"):
                        current_name = stripped[5:].strip()
                        current_smiles = ""
                    elif lower.startswith("smiles:"):
                        current_smiles = stripped[7:].strip()
                    elif lower.startswith("num peaks:") or lower.startswith("num peaks"):
                        if current_name and current_smiles:
                            if current_name not in name_to_smiles:
                                name_to_smiles[current_name] = current_smiles
                        current_name = ""
                        current_smiles = ""
                if current_name and current_smiles:
                    if current_name not in name_to_smiles:
                        name_to_smiles[current_name] = current_smiles
        except Exception as e:
            logger.warning(f"  [WARN] 读取 MSP {msp_path} 失败: {e}")
    return name_to_smiles


def build_cache_from_results(results_dirs: list, dry_run: bool = False) -> int:
    """
    扫描多层鉴定结果目录，将 CANOPUS 分类写入持久化缓存

    Args:
        results_dirs: 多个结果目录路径
        dry_run: 仅预览，不写缓存

    Returns:
        新增缓存条目数
    """
    sirius_cache = _load_sirius_cache()
    logger.info(f"[缓存模式] 当前 SIRIUS 缓存: {len(sirius_cache)} 条")
    total_added = 0

    for results_dir in results_dirs:
        if not os.path.isdir(results_dir):
            logger.warning(f"[SKIP] 目录不存在: {results_dir}")
            continue

        logger.info(f"{'='*60}")
        logger.info(f"[处理] {results_dir}")

        # 1. 加载 CANOPUS TSV
        canopus_df = load_canopus_tsv_all(results_dir)
        if canopus_df.empty:
            logger.info(f"  [INFO] 未找到 CANOPUS TSV 文件，跳过（该目录可能未跑 L3）")
            continue

        classifications = extract_all_classifications(canopus_df)
        logger.info(f"  [CANOPUS] 提取到 {len(classifications)} 个化合物分类")

        # 2. 构建 NAME → SMILES 映射
        name_to_smiles = {}
        csv_map = build_name_to_smiles_from_csv(results_dir)
        name_to_smiles.update(csv_map)
        logger.info(f"  [CSV]  NAME→SMILES: {len(csv_map)} 条")
        msp_map = build_name_to_smiles_from_msp(results_dir)
        name_to_smiles.update(msp_map)
        logger.info(f"  [ALL]  合并 NAME→SMILES: {len(name_to_smiles)} 条")

        if not name_to_smiles:
            logger.warning(f"  [WARN] 未找到任何 NAME→SMILES 映射，跳过")
            continue

        # 3. 合并到缓存
        added = 0
        for feat_id, entry in classifications.items():
            compound_name = feat_id
            smiles = name_to_smiles.get(compound_name)
            if not smiles:
                continue
            for field, classification in entry.items():
                if not classification:
                    continue
                cache_key = f"{smiles}_{field}_sirius"
                if cache_key not in sirius_cache or not sirius_cache[cache_key]:
                    sirius_cache[cache_key] = classification
                    added += 1

        logger.info(f"  [合并] 新增 {added} 条缓存记录")
        total_added += added

    if dry_run:
        logger.info(f"\n[DRY RUN] 共可新增 {total_added} 条，未写入缓存")
    elif total_added > 0:
        _save_sirius_cache(sirius_cache)
        logger.info(f"\n[保存] SIRIUS 缓存 → {SIRIUS_CACHE_FILE} ({len(sirius_cache)} 条)")
    else:
        logger.info(f"\n[无变化] 缓存无需更新")

    return total_added


# ============================================================
# CANOPUS 实时计算（为 L1/L2 来源但未进 L3 的化合物单独跑 SIRIUS）
# ============================================================

def _parse_sample_msp(path: str) -> dict:
    """解析样品 MSP 文件，返回 {name: {key: value, 'peaks': [(mz, int), ...]}}"""
    entries = {}
    current_name = None
    current = {}
    current_peaks = []
    in_peaks = False
    if not os.path.exists(path):
        return entries
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.rstrip('\n').rstrip('\r')
            if not s.strip():
                if current_name and current_peaks:
                    current['peaks'] = list(current_peaks)
                    entries[current_name] = dict(current)
                current_name = None
                current = {}
                current_peaks = []
                in_peaks = False
                continue
            lower = s.lower()
            if lower.startswith('name:'):
                if current_name and current_peaks:
                    current['peaks'] = list(current_peaks)
                    entries[current_name] = dict(current)
                current_name = s.split(':', 1)[1].strip()
                current = {'_raw_name': current_name}
                current_peaks = []
                in_peaks = False
            elif lower.startswith('precursormz:') or lower.startswith('precursor_mz:'):
                current['precursormz'] = s.split(':', 1)[1].strip()
            elif lower.startswith('precursortype:') or lower.startswith('adduct:'):
                current['precursortype'] = s.split(':', 1)[1].strip()
            elif lower.startswith('charge:'):
                current['charge'] = s.split(':', 1)[1].strip()
            elif lower.startswith('formula:'):
                current['formula'] = s.split(':', 1)[1].strip()
            elif lower.startswith('smiles:'):
                current['smiles'] = s.split(':', 1)[1].strip()
            elif lower.startswith('num peaks:') or lower.startswith('num_peaks:'):
                in_peaks = True
            elif in_peaks:
                parts = s.split()
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        current_peaks.append((mz, intensity))
                    except ValueError:
                        pass
        if current_name and current_peaks:
            current['peaks'] = list(current_peaks)
            entries[current_name] = dict(current)
    return entries


def _match_query_name(query_name: str, msp_names: set) -> Optional[str]:
    """在 MSP NAME 列表中匹配 query_name"""
    qn = str(query_name).strip()
    if not qn:
        return None
    # 精确匹配
    if qn in msp_names:
        return qn
    # 去掉前缀后匹配（例如 _L3_processed_Unknown (1.34_407)m/z → Unknown (1.34_407)m/z）
    for sep in ('_L3_processed_', '_L4_processed_', '_processed_'):
        if sep in qn:
            short = qn.split(sep, 1)[-1].strip()
            if short in msp_names:
                return short
    # 提取 RT_m/z 做模糊匹配
    import re
    m = re.search(r'\(([\d.]+)_([\d.]+)[mn]', qn)
    if m:
        rt, mz_val = m.group(1), m.group(2)
        for name in msp_names:
            if rt in name and mz_val in name:
                return name
    return None


def run_canopus_fallback(unmatched_df, sirius_bin, sample_msp, ion_mode,
                         instrument, classification_field, identifier_col):
    """
    为未命中的化合物运行 SIRIUS formula + canopus，将结果写入持久化缓存。

    仅运行 formula + fingerprint + canopus（跳 structure），~10-15s/compound。
    formula 直接从 L1/L2 已知分子式写入 MSP（无搜索），大幅加速。
    """
    logger.info("="*60)
    logger.info(f"[CANOPUS 实时计算] 开始为 {len(unmatched_df)} 个化合物运行 SIRIUS CANOPUS")

    if not os.path.exists(sirius_bin) or not os.access(sirius_bin, os.X_OK):
        logger.warning(f"[CANOPUS 实时计算] SIRIUS 不可用: {sirius_bin}")
        return 0

    if not os.path.exists(sample_msp):
        logger.warning(f"[CANOPUS 实时计算] 样品 MSP 不存在: {sample_msp}")
        return 0

    # 1. 解析样品 MSP
    logger.info(f"[CANOPUS 实时计算] 解析 MSP: {sample_msp}")
    msp_entries = _parse_sample_msp(sample_msp)
    msp_names = set(msp_entries.keys())
    logger.info(f"[CANOPUS 实时计算] MSP 中共 {len(msp_names)} 条谱图")

    # 2. 匹配 query_name → MSP NAME
    query_to_info = {}
    for _, row in unmatched_df.iterrows():
        qname = str(row.get('query_name', '')).strip()
        if not qname:
            continue
        matched = _match_query_name(qname, msp_names)
        if matched:
            formula = str(row.get(identifier_col.replace('smiles', 'formula').replace('SMILES', 'formula'), ''))
            if not formula or formula == 'nan':
                # 尝试从 matched_formula 列获取
                formula = str(row.get('matched_formula', ''))
            smiles = str(row.get(identifier_col, '')).strip()
            query_to_info[qname] = {
                'msp_name': matched,
                'formula': formula if formula and formula != 'nan' else '',
                'smiles': smiles if smiles and smiles != 'nan' else '',
            }

    if not query_to_info:
        logger.warning(f"[CANOPUS 实时计算] 无化合物可匹配到 MSP 谱图，跳过")
        return 0

    logger.info(f"[CANOPUS 实时计算] 匹配到 {len(query_to_info)}/{len(unmatched_df)} 个化合物")

    # 3. 构建 mini MSP（SIRIUS 兼容格式）
    adduct = '[M+H]+,[M+Na]+' if ion_mode == 'POS' else '[M-H]-'
    ion_mode_str = 'Positive' if ion_mode == 'POS' else 'Negative'
    msp_lines = []
    for qname, info in query_to_info.items():
        entry = msp_entries.get(info['msp_name'])
        if not entry or not entry.get('peaks'):
            continue
        msp_lines.append(f"NAME: {info['msp_name']}")
        msp_lines.append(f"PRECURSORMZ: {entry.get('precursormz', '')}")
        msp_lines.append(f"PRECURSORTYPE: {adduct}")
        msp_lines.append(f"IONMODE: {ion_mode_str}")
        if info['formula']:
            msp_lines.append(f"FORMULA: {info['formula']}")
        if info['smiles']:
            msp_lines.append(f"SMILES: {info['smiles']}")
        msp_lines.append(f"Num Peaks: {len(entry['peaks'])}")
        for mz, intensity in entry['peaks']:
            msp_lines.append(f"{mz} {intensity}")
        msp_lines.append("")

    if not msp_lines:
        logger.warning(f"[CANOPUS 实时计算] 构建 MSP 后无有效条目，跳过")
        return 0

    msp_content = '\n'.join(msp_lines)
    logger.info(f"[CANOPUS 实时计算] 构建 mini MSP: {len([l for l in msp_lines if l.startswith('NAME:')])} 个化合物")

    # 4. 写入临时文件并运行 SIRIUS
    profile = 'orbitrap' if instrument == 'orbitrap' else 'qtof'
    with tempfile.TemporaryDirectory(prefix='ontology_canopus_') as tmpdir:
        mini_msp = os.path.join(tmpdir, 'mini.msp')
        project = os.path.join(tmpdir, 'project.sirius')
        with open(mini_msp, 'w', encoding='utf-8') as f:
            f.write(msp_content)

        # Formula
        logger.info(f"[CANOPUS 实时计算] 步骤1/3: Formula 预测（公式已约束）...")
        cmd_formula = [
            sirius_bin, '--cores', str(min(8, os.cpu_count() or 4)),
            '-i', mini_msp, '-o', project,
            'formula', '-c', '10', '-p', profile,
            '-I', adduct, '-i', '',
        ]
        try:
            subprocess.run(cmd_formula, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            logger.warning("[CANOPUS 实时计算] Formula 步骤超时")
            return 0
        except Exception as e:
            logger.warning(f"[CANOPUS 实时计算] Formula 步骤失败: {e}")
            return 0

        # Fingerprint
        logger.info(f"[CANOPUS 实时计算] 步骤2/3: Fingerprint 预测...")
        cmd_fp = [sirius_bin, '-i', project, '-o', project, 'fingerprint']
        try:
            subprocess.run(cmd_fp, capture_output=True, text=True, timeout=300)
        except Exception as e:
            logger.warning(f"[CANOPUS 实时计算] Fingerprint 步骤失败: {e}")
            return 0

        # CANOPUS
        logger.info(f"[CANOPUS 实时计算] 步骤3/3: CANOPUS 分类...")
        cmd_cano = [sirius_bin, '-i', project, '-o', project, 'canopus']
        try:
            result = subprocess.run(cmd_cano, capture_output=True, text=True, timeout=300)
        except Exception as e:
            logger.warning(f"[CANOPUS 实时计算] CANOPUS 步骤失败: {e}")
            return 0

        # 5. 读取 CANOPUS TSV
        tsv_path = os.path.join(project, 'canopus_formula_summary.tsv')
        if not os.path.exists(tsv_path):
            logger.warning("[CANOPUS 实时计算] canopus_formula_summary.tsv 未生成")
            return 0

        col_name = CANOPUS_FIELD_MAP.get(classification_field)
        if not col_name:
            return 0

        try:
            tdf = pd.read_csv(tsv_path, sep='\t', dtype=str, keep_default_na=False)
        except Exception as e:
            logger.warning(f"[CANOPUS 实时计算] 读取 TSV 失败: {e}")
            return 0

        if col_name not in tdf.columns:
            logger.warning(f"[CANOPUS 实时计算] TSV 缺少 {col_name} 列")
            return 0

        # 6. 写入缓存：SMILES → classification
        sirius_cache = _load_sirius_cache()
        added = 0
        for _, row in tdf.iterrows():
            cls = str(row.get(col_name, '')).strip()
            if not cls:
                continue
            # 找对应的 SMILES（通过 mappingFeatureId）
            feat_id = str(row.get('mappingFeatureId', '')).strip()
            smiles_found = None
            for qname, info in query_to_info.items():
                msp_name = info['msp_name']
                if msp_name in feat_id or feat_id in msp_name or msp_name.replace('Unknown (','').rstrip(')').replace(' ','') in feat_id:
                    smiles_found = info['smiles']
                    break
            if not smiles_found:
                continue
            cache_key = f"{smiles_found}_{classification_field}_sirius"
            if cache_key not in sirius_cache:
                sirius_cache[cache_key] = cls
                added += 1

        if added:
            _save_sirius_cache(sirius_cache)
            logger.info(f"[CANOPUS 实时计算] 写入缓存 {added} 条新记录（共 {len(sirius_cache)} 条）")
        else:
            logger.info("[CANOPUS 实时计算] 未产生新的缓存记录")

        return added


# ============================================================

def process_csv(input_csv: str, output_csv: str, classification_field: str = "class",
                force_refresh: bool = False, sirius_results_dir: Optional[str] = None,
                canopus_fallback: bool = False, sample_msp_path: Optional[str] = None,
                sirius_bin: Optional[str] = None, ion_mode: Optional[str] = None,
                instrument: str = "qtof"):
    """
    处理CSV文件，为每个化合物获取 ontology 分类

    级联策略（全部本地，不联网）：
      1. SIRIUS CANOPUS TSV 直读（本轮 L3 结果，秒级）
      2. SIRIUS 持久化缓存（跨 run 积累，毫秒级）

    每次 CANOPUS 直读命中时自动写入持久化缓存，供无 L3 的后续运行使用。
    """
    sirius_cache = _load_sirius_cache()
    logger.info(f"加载 SIRIUS 缓存: {len(sirius_cache)} 条")

    if not os.path.exists(input_csv):
        logger.warning(f"输入文件不存在: {input_csv}")
        return

    if os.path.getsize(input_csv) == 0:
        logger.warning(f"输入文件为空: {input_csv}")
        return

    try:
        df = pd.read_csv(input_csv, encoding='utf-8')
    except pd.errors.EmptyDataError:
        logger.warning(f"输入文件格式错误(空文件): {input_csv}")
        return

    if df.empty:
        logger.warning(f"输入文件无数据: {input_csv}")
        return

    logger.info(f"读取 {len(df)} 条记录")

    identifier_col = None
    for col in ['matched_smiles', 'target_smiles', 'smiles', 'SMILES']:
        if col in df.columns:
            identifier_col = col
            break

    if not identifier_col:
        for col in ['matched_inchikey', 'target_inchikey', 'inchikey', 'InChIKey', 'INCHIKEY']:
            if col in df.columns:
                identifier_col = col
                break

    if not identifier_col:
        logger.warning("未找到 SMILES 或 InChIKey 列，无法获取 ontology")
        df.to_csv(output_csv, index=False, encoding='utf-8')
        return

    logger.info(f"使用标识符列: {identifier_col}")

    if 'matched_ontology' not in df.columns:
        df['matched_ontology'] = ""

    # 优先级 1：SIRIUS CANOPUS TSV 直读
    canopus_filled = 0
    if (not force_refresh) and sirius_results_dir and ('query_name' in df.columns):
        canopus_map = load_sirius_canopus_ontology(sirius_results_dir, classification_field)
        if canopus_map:
            for idx, row in df.iterrows():
                qname = str(row.get('query_name', '')).strip()
                if not qname:
                    continue
                current = str(row.get('matched_ontology', '')).strip()
                if current and current != 'nan':
                    continue
                cls = canopus_map.get(qname)
                if cls:
                    df.at[idx, 'matched_ontology'] = cls
                    canopus_filled += 1
                    smiles = str(row.get(identifier_col, '')).strip()
                    if smiles and smiles != 'nan':
                        cache_key = f"{smiles}_{classification_field}_sirius"
                        sirius_cache[cache_key] = cls
            if canopus_filled:
                logger.info(f"[来源: CANOPUS] 直接填充 {canopus_filled} 条 ontology")
                _save_sirius_cache(sirius_cache)

    # 优先级 2：SIRIUS 持久化缓存
    has_ontology = df['matched_ontology'].apply(
        lambda x: bool(x) and str(x).strip() != '' and str(x) != 'nan'
    )
    skipped_count = has_ontology.sum()
    needs_query = ~has_ontology
    logger.info(f"总记录: {len(df)}, CANOPUS已填: {canopus_filled}, 已有ontology跳过: {skipped_count}, 需查询: {needs_query.sum()}")

    cache_filled = 0
    for idx in df[needs_query].index:
        smiles = str(df.at[idx, identifier_col]).strip()
        if not smiles or smiles == 'nan':
            continue
        cache_key = f"{smiles}_{classification_field}_sirius"
        cls = sirius_cache.get(cache_key)
        if cls:
            df.at[idx, 'matched_ontology'] = cls
            cache_filled += 1

    if cache_filled:
        logger.info(f"[来源: SIRIUS缓存] {cache_filled} 条 ontology 从持久化缓存命中")

    fail_count = needs_query.sum() - cache_filled

    # 优先级 3：CANOPUS 实时计算（为 L1/L2 来源但未进入 L3 的化合物单独运行 SIRIUS formula+canopus）
    candidate_mask = needs_query & df[identifier_col].notna() & (df[identifier_col] != '') & (df[identifier_col] != 'nan')
    if fail_count > 0 and canopus_fallback and sample_msp_path and sirius_bin and ion_mode:
        if candidate_mask.sum() > 0:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"[来源: CANOPUS 实时计算] 为 {candidate_mask.sum()} 个有 SMILES 的未命中化合物运行 SIRIUS CANOPUS...")
                added = run_canopus_fallback(
                    unmatched_df=df[candidate_mask].copy(),
                    sirius_bin=sirius_bin or DEFAULT_SIRIUS_BIN,
                    sample_msp=sample_msp_path,
                    ion_mode=ion_mode,
                    instrument=instrument,
                    classification_field=classification_field,
                    identifier_col=identifier_col,
                )
                if added:
                    # 重新加载缓存并回填
                    sirius_cache = _load_sirius_cache()
                    cache_filled2 = 0
                    for idx in df[candidate_mask].index:
                        smiles = str(df.at[idx, identifier_col]).strip()
                        if not smiles or smiles == 'nan':
                            continue
                        cache_key = f"{smiles}_{classification_field}_sirius"
                        cls = sirius_cache.get(cache_key)
                        if cls:
                            current = str(df.at[idx, 'matched_ontology']).strip()
                            if not current or current == 'nan':
                                df.at[idx, 'matched_ontology'] = cls
                                cache_filled2 += 1
                    logger.info(f"[来源: CANOPUS 新鲜] CANOPUS 实时计算完成，新填充 {cache_filled2} 条 ontology")
                    fail_count -= cache_filled2
            except Exception as e:
                logger.warning(f"[来源: CANOPUS 实时计算] 运行异常（不影响主流程）: {e}")
        else:
            logger.info("[来源: CANOPUS 实时计算] 未命中化合物均缺少 SMILES，无法运行 CANOPUS")

    if fail_count > 0:
        logger.warning(f"未命中: {fail_count} 条（CANOPUS 缓存无记录，可能来自L1/L2鉴定且未进入L3流程）")

    df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"处理完成: CANOPUS={canopus_filled}, 缓存命中={cache_filled}, 未命中={fail_count}")
    logger.info(f"结果保存至: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="SIRIUS CANOPUS Ontology 分类获取")
    parser.add_argument("--input_csv", default=None,
                       help="输入 CSV 文件路径（CSV 模式）")
    parser.add_argument("--output_csv", default=None,
                       help="输出 CSV 文件路径（CSV 模式）")
    parser.add_argument("--field", default="class",
                       choices=["kingdom", "superclass", "class", "subclass", "direct_parent"],
                       help="分类层级 (默认: class)")
    parser.add_argument("--force_refresh", action="store_true",
                       help="强制刷新，跳过 CANOPUS TSV 直读")
    parser.add_argument("--sirius_results_dir", default=None,
                       help="SIRIUS CANOPUS 输出目录（CSV 模式下优先从此填充）")
    parser.add_argument("--results_dir", nargs="+", default=None,
                       help="多层鉴定结果目录（可多个），缓存模式：扫描历史结果填充持久化缓存")
    parser.add_argument("--dry_run", action="store_true",
                       help="仅预览（缓存模式），不写缓存")
    parser.add_argument("--canopus_fallback", action="store_true",
                       help="为未命中化合物实时运行 SIRIUS CANOPUS（需配合 --sample_msp --ion_mode）")
    parser.add_argument("--sample_msp", default=None,
                       help="样品 MSP 路径（CANOPUS 实时计算用，提供 MS2 谱图）")
    parser.add_argument("--sirius_bin", default=DEFAULT_SIRIUS_BIN,
                       help="SIRIUS 可执行文件路径")
    parser.add_argument("--ion_mode", default=None, choices=["POS", "NEG"],
                       help="离子模式（CANOPUS 实时计算用）")
    parser.add_argument("--instrument", default="qtof", choices=["orbitrap", "qtof"],
                       help="仪器类型（CANOPUS 实时计算用，默认 qtof）")

    args = parser.parse_args()

    if args.results_dir:
        # 缓存模式：扫描历史结果 → 填充持久化缓存
        build_cache_from_results(args.results_dir, dry_run=args.dry_run)
    elif args.input_csv:
        # CSV 模式：为鉴定结果 CSV 填充 ontology 列
        if not args.output_csv:
            parser.error("CSV 模式需要 --output_csv")
        process_csv(args.input_csv, args.output_csv, args.field, args.force_refresh,
                    args.sirius_results_dir,
                    canopus_fallback=args.canopus_fallback,
                    sample_msp_path=args.sample_msp,
                    sirius_bin=args.sirius_bin,
                    ion_mode=args.ion_mode,
                    instrument=args.instrument)
    else:
        parser.error("请指定 --input_csv（CSV 模式） 或 --results_dir（缓存模式）")


if __name__ == "__main__":
    main()
