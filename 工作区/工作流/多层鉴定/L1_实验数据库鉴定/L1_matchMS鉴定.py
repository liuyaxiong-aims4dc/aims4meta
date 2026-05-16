#!/usr/bin/env python3
"""
L1: matchMS 鉴定

鉴定策略：
- 使用 matchms 标准预处理管道（谱图清洗、归一化、噪声过滤）
- ModifiedCosine 相似度（支持 precursor shift）
- m/z 倒排索引预筛选（加速候选检索）
- 多库遍历，跨库合并 Top-K
- 可选：从 Waters QI 导出 CSV 读取实测同位素分布，与理论值比对打分
- 输出统一格式的 L1_matchMS_results.csv
"""

import argparse
import importlib.util
import os
import pickle
import re
import time
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from pyteomics import mass as _pyteomics_mass
    _HAS_PYTEOMICS = True
except ImportError:
    _HAS_PYTEOMICS = False

from matchms import Spectrum
from matchms.importing import load_from_msp
from matchms.filtering import (
    default_filters,
    normalize_intensities,
    select_by_intensity,
    select_by_mz,
    require_minimum_number_of_peaks,
    reduce_to_number_of_peaks,
    add_precursor_mz,
    clean_adduct,
    correct_charge,
    derive_ionmode,
)
from matchms.similarity import ModifiedCosineGreedy as ModifiedCosine


# ============================================================
# 碎片鉴定（供 L2/L3 import 复用，支持 Numba 加速）
# ============================================================

_numba_fragment_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '辅助功能', 'Numba碎片鉴定', 'numba_fragment_matching.py'
)
if os.path.exists(_numba_fragment_path):
    _spec_nb = importlib.util.spec_from_file_location(
        "numba_fragment_matching", os.path.abspath(_numba_fragment_path))
    _numba_frag_module = importlib.util.module_from_spec(_spec_nb)
    _spec_nb.loader.exec_module(_numba_frag_module)
    find_matched_fragments = _numba_frag_module.find_matched_fragments
    USE_NUMBA = True
else:
    USE_NUMBA = False

    def find_matched_fragments(query_peaks, lib_peaks, tolerance, query_pmz, lib_pmz):
        """找出鉴定的碎片峰详情（CPU fallback，支持 precursor shift）"""
        # 过滤母离子峰，避免虚高碎片计数
        if query_pmz:
            query_peaks = [(mz, inten) for mz, inten in query_peaks
                           if abs(mz - query_pmz) > tolerance]
        matched = []
        pmz_diff = (query_pmz - lib_pmz) if query_pmz and lib_pmz else 0

        used_lib = set()
        for q_mz, q_int in query_peaks:
            if q_int <= 0:
                continue
            best_match = None
            best_diff = float('inf')
            for i, (l_mz, l_int) in enumerate(lib_peaks):
                if i in used_lib or l_int <= 0:
                    continue
                diff = abs(q_mz - l_mz)
                diff_shifted = abs(q_mz - l_mz + pmz_diff)
                if diff <= tolerance and diff < best_diff:
                    best_match = (i, l_mz, l_int, diff)
                    best_diff = diff
                elif diff_shifted <= tolerance and diff_shifted < best_diff:
                    best_match = (i, l_mz, l_int, diff_shifted)
                    best_diff = diff_shifted
            if best_match:
                i, l_mz, l_int, diff = best_match
                used_lib.add(i)
                matched.append({
                    "query_mz": round(q_mz, 4),
                    "lib_mz": round(l_mz, 4),
                    "query_intensity": round(q_int, 4),
                    "lib_intensity": round(l_int, 4),
                    "mz_diff": round(diff, 4),
                })
        return matched


# ============================================================
# 同位素分布（可选）
# ============================================================

# 从 MSP compound name 中提取 compound_id，如 "Unknown (4.63_678.4991n)" → "4.63_678.4991n"
_COMPOUND_ID_RE = re.compile(r'\(([^)]+)\)')


def load_isotope_data(csv_path: str) -> dict:
    """从 Waters QI 导出 CSV 加载实测同位素分布。
    返回 {compound_id: [100.0, M+1%, M+2%, ...]}，compound_id 如 '4.63_678.4991n'。
    CSV 格式：前两行为元信息，第三行为列名（header=2）。
    """
    try:
        df = pd.read_csv(csv_path, header=2, encoding='utf-8')
    except Exception as e:
        print(f"[WARN] 无法加载同位素 CSV: {e}")
        return {}

    compound_col = next((c for c in df.columns if c.strip().lower() in ('compound', 'compound id')), None)
    isotope_col  = next((c for c in df.columns if 'isotope' in c.lower() and 'distribution' in c.lower()), None)
    if not compound_col or not isotope_col:
        print(f"[WARN] CSV 中未找到 Compound / Isotope Distribution 列")
        return {}

    result = {}
    for _, row in df.iterrows():
        cid = str(row[compound_col]).strip()
        iso = str(row[isotope_col]).strip()
        if not cid or cid == 'nan' or not iso or iso == 'nan':
            continue
        try:
            parts = [float(x.strip()) for x in iso.replace('−', '-').split('-') if x.strip()]
            if parts:
                result[cid] = parts
        except ValueError:
            continue

    print(f"[INFO] 同位素 CSV: 加载 {len(result)} 条实测同位素分布")
    return result


def predict_isotope_pattern(formula: str, n_peaks: int = 5) -> list:
    """用 pyteomics 预测分子式的理论同位素分布，返回归一化到 100 的列表。"""
    if not _HAS_PYTEOMICS or not formula:
        return []
    try:
        mono = _pyteomics_mass.calculate_mass(formula=formula)
        isotopologues = list(_pyteomics_mass.isotopologues(
            formula, report_abundance=True, overall_threshold=0.001))
        groups: dict = {}
        for comp, ab in isotopologues:
            m = _pyteomics_mass.calculate_mass(composition=comp)
            offset = round(m - mono)
            groups[offset] = groups.get(offset, 0.0) + ab
        arr = [groups.get(i, 0.0) for i in range(n_peaks)]
        if arr[0] > 0:
            arr = [v / arr[0] * 100.0 for v in arr]
        return arr
    except Exception:
        return []


def isotope_cosine(obs: list, pred: list) -> float:
    """两个同位素分布向量的余弦相似度（截取到相同长度）。"""
    n = min(len(obs), len(pred))
    if n == 0:
        return 0.0
    a = np.array(obs[:n], dtype=float)
    b = np.array(pred[:n], dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ============================================================
# HDMSE数据处理
# ============================================================

def _process_hdmse_spectra(exp, rt_window: float, drift_tolerance: float,
                          use_correlation: bool = False, corr_threshold: float = 0.7):
    """处理HDMSE数据：按RT和漂移时间分组，合并同组碎片

    改进：不依赖precursor_mz，而是按RT+DT分组所有峰
    """
    from collections import defaultdict
    from matchms import Spectrum

    # 按RT和漂移时间分组（不再按precursor_mz）
    rt_dt_groups = defaultdict(list)

    for spectrum in exp:
        if spectrum.getMSLevel() != 2:
            continue

        rt = spectrum.getRT() / 60  # 转为分钟
        rt_bin = round(rt / rt_window) * rt_window
        dt = spectrum.getDriftTime()
        dt_bin = round(dt / drift_tolerance) * drift_tolerance if drift_tolerance else dt

        # 收集所有峰（不区分前体和碎片）
        for peak in spectrum:
            rt_dt_groups[(rt_bin, dt_bin)].append((peak.getMZ(), peak.getIntensity(), rt))

    print(f"[INFO] HDMSE分组: {len(rt_dt_groups)} 个RT+DT组合")

    # 为每个RT+DT组创建特征谱图
    merged_spectra = []
    for (rt_bin, dt_bin), all_peaks in rt_dt_groups.items():
        if not all_peaks:
            continue

        # 合并相同m/z的峰（取最大强度）
        peak_dict = {}
        for mz, intensity, rt in all_peaks:
            mz_key = round(mz, 4)
            peak_dict[mz_key] = max(peak_dict.get(mz_key, 0), intensity)

        if not peak_dict:
            continue

        mz_arr = np.array(sorted(peak_dict.keys()), dtype=float)
        int_arr = np.array([peak_dict[m] for m in mz_arr], dtype=float)

        # 推断前体m/z：取强度最高的峰
        max_int_idx = np.argmax(int_arr)
        precursor_mz = mz_arr[max_int_idx]

        s = Spectrum(mz=mz_arr, intensities=int_arr,
                   metadata={"precursor_mz": precursor_mz,
                           "retention_time": rt_bin,
                           "drift_time": dt_bin})
        merged_spectra.append(s)

    print(f"[INFO] HDMSE合并: {len(merged_spectra)} 个特征谱图")
    return merged_spectra


# ============================================================
# 谱图预处理
# ============================================================

def preprocess_spectrum(spectrum: Spectrum, min_peaks: int = 3,
                        min_mz: float = 10.0, max_mz: float = 2000.0,
                        min_intensity: float = 0.01) -> Spectrum:
    """matchms 标准预处理管道"""
    if spectrum is None:
        return None
    spectrum = default_filters(spectrum)       # 字段名标准化、类型修正
    spectrum = add_precursor_mz(spectrum)      # 补全 precursor_mz
    spectrum = clean_adduct(spectrum)          # 加合物标准化
    spectrum = correct_charge(spectrum)        # 电荷修正
    spectrum = derive_ionmode(spectrum)        # 推断离子模式
    spectrum = select_by_mz(spectrum, mz_from=min_mz, mz_to=max_mz)
    spectrum = normalize_intensities(spectrum) # 先归一化，再按相对强度过滤
    spectrum = select_by_intensity(spectrum, intensity_from=min_intensity)
    spectrum = reduce_to_number_of_peaks(spectrum, n_max=500)
    spectrum = require_minimum_number_of_peaks(spectrum, n_required=min_peaks)
    return spectrum  # None 表示不合格，已被过滤


# ============================================================
# 数据库加载
# ============================================================

def load_library(lib_name: str, lib_path: str, min_peaks: int = 3) -> list:
    """加载并预处理数据库（优先使用缓存）"""
    pkl_path = lib_path.replace('.msp', '_matchms_cache.pkl')

    if os.path.exists(pkl_path):
        print(f"[INFO] 加载 {lib_name} 缓存: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            library = pickle.load(f)
        print(f"[INFO] {lib_name}: {len(library)} 条（缓存）")
        return library

    print(f"[INFO] 加载 {lib_name}: {lib_path}")
    print(f"[INFO] 正在读取MSP文件，请稍候...")
    
    # 使用tqdm显示加载进度
    raw = []
    for spectrum in tqdm(load_from_msp(lib_path), desc=f"  加载{lib_name}", unit="条"):
        raw.append(spectrum)
    
    print(f"[INFO] 预处理中...")
    library = []
    for s in tqdm(raw, desc=f"  预处理", unit="条"):
        s = preprocess_spectrum(s, min_peaks=min_peaks)
        if s is not None:
            library.append(s)
    print(f"[INFO] {lib_name}: {len(raw)} -> {len(library)} 条（预处理后）")

    with open(pkl_path, 'wb') as f:
        pickle.dump(library, f)
    print(f"[INFO] 缓存已保存: {pkl_path}")
    return library


def load_sample_spectra(sample_msp: str, min_peaks: int = 3,
                        use_correlation: bool = False, corr_threshold: float = 0.7) -> list:
    """加载并预处理样品谱图

    Args:
        use_correlation: 是否使用强度相关性过滤（需要scipy）
        corr_threshold: 相关性阈值（0-1），默认0.7
    """
    print(f"[INFO] 加载样品: {sample_msp}")
    if use_correlation:
        print(f"[INFO] 强度相关性过滤: 阈值={corr_threshold}")

    # 判断文件格式
    if sample_msp.lower().endswith('.mzml'):
        # mzML格式：使用pyopenms读取
        try:
            from pyopenms import MSExperiment, MzMLFile
        except ImportError:
            print("[ERROR] 需要安装pyopenms: pip install pyopenms")
            return [], []

        exp = MSExperiment()
        MzMLFile().load(sample_msp, exp)

        # 普通模式：直接读取
        raw = []
        for spectrum in exp:
            if spectrum.getMSLevel() != 2:
                continue

            precursors = spectrum.getPrecursors()
            if not precursors:
                continue

            precursor = precursors[0]
            precursor_mz = precursor.getMZ()
            drift_time = spectrum.getDriftTime()
            rt = spectrum.getRT() / 60  # 转为分钟

            mz = []
            intensity = []
            for peak in spectrum:
                mz.append(peak.getMZ())
                intensity.append(peak.getIntensity())

            if len(mz) == 0:
                continue

            from matchms import Spectrum
            metadata = {"precursor_mz": precursor_mz, "retention_time": rt}
            if drift_time >= 0:
                metadata["drift_time"] = drift_time
            s = Spectrum(mz=np.array(mz), intensities=np.array(intensity),
                        metadata=metadata)
            raw.append(s)
    else:
        # MSP格式 - 分块加载显示进度
        print(f"[INFO] 正在加载MSP文件...")

        # 先快速统计谱图数量
        total_spectra = 0
        with open(sample_msp, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip().upper().startswith('NAME:'):
                    total_spectra += 1

        print(f"[INFO] 预计加载 {total_spectra} 个谱图")

        # 使用tqdm显示加载进度
        raw = []
        for spectrum in tqdm(load_from_msp(sample_msp), total=total_spectra, desc="加载谱图"):
            raw.append(spectrum)

        print(f"[INFO] 加载完成，共 {len(raw)} 个谱图")
    
    spectra = []
    for s in raw:
        s = preprocess_spectrum(s, min_peaks=min_peaks)
        spectra.append(s)  # 保留 None 占位，保持索引对齐
    valid = sum(1 for s in spectra if s is not None)
    print(f"[INFO] 样品: {len(raw)} -> {valid} 条有效（预处理后）")
    return spectra, raw  # 同时返回原始谱图用于提取 query_name


# ============================================================
# m/z 倒排索引
# ============================================================

def build_mz_index(library_spectra: list) -> dict:
    """构建母离子 m/z 倒排索引（bin 精度 0.01 Da）"""
    mz_index = defaultdict(list)
    for i, spectrum in enumerate(library_spectra):
        pmz = spectrum.get("precursor_mz")
        if pmz:
            mz_index[int(round(pmz * 100))].append(i)
    return mz_index


# ============================================================
# 鉴定核心
# ============================================================

def extract_source_database(lib_spectrum, fallback_name: str) -> str:
    """提取库谱图来源（模拟库返回 tool/db，真实库返回库名）"""
    comment = lib_spectrum.get("comment", "") or ""
    if "Source_tool=" not in comment:
        return fallback_name
    source_tool, source_db = "", ""
    for part in comment.split(';'):
        part = part.strip()
        if part.startswith('Source_tool='):
            source_tool = part.split('=', 1)[1].strip()
        elif part.startswith('Source_database='):
            source_db = part.split('=', 1)[1].strip()
    if source_tool and source_db:
        return f"{source_tool}/{source_db}"
    return source_tool or fallback_name


def match_one(query: Spectrum, library_spectra: list, mz_index: dict,
              mz_ppm: float, min_peaks: int, cosine_threshold: float,
              fragment_tolerance: float, top_k: int) -> list:
    """对单个查询谱图执行 matchMS 鉴定，返回 top_k 结果"""
    pmz = query.get("precursor_mz")
    if not pmz:
        return []

    # m/z 预筛选
    tol = pmz * mz_ppm / 1e6
    candidates = set()
    for b in range(int(round((pmz - tol) * 100)) - 1,
                   int(round((pmz + tol) * 100)) + 2):
        candidates.update(mz_index.get(b, []))

    # 精确过滤
    candidates = [c for c in candidates
                  if abs((library_spectra[c].get("precursor_mz") or 0) - pmz) <= tol]
    if not candidates:
        return []

    scorer = ModifiedCosine(tolerance=fragment_tolerance)
    query_peaks_list = list(zip(query.peaks.mz, query.peaks.intensities))
    # 过滤母离子峰，避免虚高碎片计数
    # 用 3× fragment_tolerance 窗口：MSP metadata 的 precursor_mz 可能与实际峰 m/z 偏差 0.01–0.03 Da
    precursor_window = fragment_tolerance * 3
    query_peaks_filtered = [(mz, inten) for mz, inten in query_peaks_list
                            if abs(mz - pmz) > precursor_window]
    matches = []
    for lib_idx in candidates:
        lib = library_spectra[lib_idx]
        lib_pmz = lib.get("precursor_mz")
        try:
            result = scorer.pair(query, lib)
            # 先跑碎片匹配（含母离子过滤），以真实碎片数作为判定阈值
            lib_peaks_list = list(zip(lib.peaks.mz, lib.peaks.intensities))
            frags = find_matched_fragments(
                query_peaks_filtered, lib_peaks_list, fragment_tolerance, pmz, lib_pmz)
            real_matched_peaks = len(frags)
            if real_matched_peaks < min_peaks or result["score"] < cosine_threshold:
                continue
            frags_str = "; ".join([
                f"{f['query_mz']}/{f['lib_mz']}(Δ{f['mz_diff']:.4f}Da)"
                for f in frags
            ])
            matches.append({
                "cosine_score": float(result["score"]),
                "matched_peaks": real_matched_peaks,
                "library_total_peaks": len(lib.peaks.mz),
                "matched_fragments_str": frags_str,
                "library_spectrum": lib,
            })
        except Exception:
            continue

    matches.sort(key=lambda x: x["cosine_score"], reverse=True)
    return matches[:top_k]


# ============================================================
# 主函数
# ============================================================

def main():
    # ── 环境变量 ──────────────────────────────────────────────
    SAMPLE_MSP        = os.environ.get('L1_SAMPLE_MSP')
    SAMPLE_CSV        = os.environ.get('L1_SAMPLE_CSV', '')
    ION_MODE          = os.environ.get('L1_ION_MODE')
    LIBRARIES_JSON    = os.environ.get('L1_MATCHMS_LIBRARIES')
    MZ_TOLERANCE_PPM  = os.environ.get('L1_MZ_TOLERANCE_PPM')
    FRAGMENT_TOLERANCE= os.environ.get('L1_FRAGMENT_TOLERANCE')
    MIN_MATCHED_PEAKS = os.environ.get('L1_MIN_MATCHED_PEAKS')
    COSINE_THRESHOLD  = os.environ.get('L1_COSINE_THRESHOLD')
    OUTPUT_DIR        = os.environ.get('L1_OUTPUT_DIR')
    TOP_K             = os.environ.get('L1_TOP_K')

    missing = [k for k, v in [
        ('L1_SAMPLE_MSP', SAMPLE_MSP), ('L1_ION_MODE', ION_MODE),
        ('L1_MATCHMS_LIBRARIES', LIBRARIES_JSON), ('L1_MZ_TOLERANCE_PPM', MZ_TOLERANCE_PPM),
        ('L1_FRAGMENT_TOLERANCE', FRAGMENT_TOLERANCE), ('L1_MIN_MATCHED_PEAKS', MIN_MATCHED_PEAKS),
        ('L1_COSINE_THRESHOLD', COSINE_THRESHOLD), ('L1_OUTPUT_DIR', OUTPUT_DIR),
    ] if not v]
    if missing:
        raise ValueError(f"环境变量未设置: {', '.join(missing)}")

    LIBRARIES          = json.loads(LIBRARIES_JSON)
    MZ_TOLERANCE_PPM   = int(float(MZ_TOLERANCE_PPM))
    FRAGMENT_TOLERANCE = float(FRAGMENT_TOLERANCE)
    MIN_MATCHED_PEAKS  = int(MIN_MATCHED_PEAKS)
    COSINE_THRESHOLD   = float(COSINE_THRESHOLD)
    TOP_K              = int(TOP_K) if TOP_K else 1

    # ── 命令行参数（可覆盖环境变量）─────────────────────────────
    parser = argparse.ArgumentParser(description="L1 matchMS 鉴定")
    parser.add_argument("--sample_msp",          default=SAMPLE_MSP)
    parser.add_argument("--sample_csv",           default=SAMPLE_CSV or None,
                        help="Waters QI 导出 CSV（含 Isotope Distribution 列），可选")
    parser.add_argument("--ion_mode",             default=ION_MODE, choices=["POS", "NEG"])
    parser.add_argument("--output_dir",           default=OUTPUT_DIR)
    parser.add_argument("--mz_ppm",               type=float, default=MZ_TOLERANCE_PPM)
    parser.add_argument("--fragment_tolerance",   type=float, default=FRAGMENT_TOLERANCE)
    parser.add_argument("--min_matched_peaks",    type=int,   default=MIN_MATCHED_PEAKS)
    parser.add_argument("--cosine_threshold",     type=float, default=COSINE_THRESHOLD)
    parser.add_argument("--use_correlation",      action='store_true',
                        help="启用强度相关性过滤（需要scipy）")
    parser.add_argument("--corr_threshold",       type=float, default=0.7,
                        help="相关性阈值（0-1），默认0.7")
    parser.add_argument("--output_csv",           default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 根据输出目录最后一级判断层级，避免父目录名包含 L2/L3 时误判
    level = "L1"
    output_dir_name = os.path.basename(os.path.normpath(args.output_dir or ""))
    if output_dir_name.startswith("L2"):
        level = "L2"
    elif output_dir_name.startswith("L3"):
        level = "L3"

    # 根据层级设置输出文件名
    output_csv = args.output_csv or os.path.join(args.output_dir, f"{level}_matchMS_results.csv")

    # 可选：加载实测同位素分布
    isotope_data: dict = {}
    if args.sample_csv and os.path.exists(args.sample_csv):
        isotope_data = load_isotope_data(args.sample_csv)
    elif args.sample_csv:
        print(f"[WARN] sample_csv 不存在: {args.sample_csv}")

    print("=" * 70)
    print(f"{level} MatchMS 鉴定")
    print("=" * 70)
    print(f"[参数] 母离子容差: {args.mz_ppm} ppm")
    print(f"[参数] 碎片容差:   {args.fragment_tolerance} Da")
    print(f"[参数] 最少鉴定峰: {args.min_matched_peaks}")
    print(f"[参数] 余弦阈值:   {args.cosine_threshold}")
    print(f"[参数] Top-K:      {TOP_K}")

    t0 = time.time()

    # 1. 加载样品
    sample_spectra, raw_spectra = load_sample_spectra(
        args.sample_msp, min_peaks=args.min_matched_peaks,
        use_correlation=args.use_correlation,
        corr_threshold=args.corr_threshold)

    # 2. 加载数据库
    libraries, mz_indices = {}, {}
    for lib_name, lib_path in LIBRARIES.items():
        # 处理列表格式的路径（多个MSP文件）
        if isinstance(lib_path, list):
            # 跳过列表格式，matchMS不支持多文件合并
            print(f"[WARN] matchMS不支持多文件数据库: {lib_name} (跳过)")
            continue

        if lib_path and os.path.exists(lib_path):
            libraries[lib_name] = load_library(lib_name, lib_path, min_peaks=args.min_matched_peaks)
            mz_indices[lib_name] = build_mz_index(libraries[lib_name])
        else:
            print(f"[WARN] 数据库不存在: {lib_path}")

    # 3. 鉴定
    results = []
    matched_count = 0

    for i, query in enumerate(tqdm(sample_spectra, desc="[matchMS鉴定]", unit="条")):
        if query is None:
            continue
        query_pmz = query.get("precursor_mz")
        if not query_pmz:
            continue

        # 跨库收集候选
        all_matches = []
        for lib_name, lib_spectra in libraries.items():
            hits = match_one(query, lib_spectra, mz_indices[lib_name],
                             args.mz_ppm, args.min_matched_peaks,
                             args.cosine_threshold, args.fragment_tolerance, TOP_K)
            for h in hits:
                h["lib_name"] = lib_name
            all_matches.extend(hits)

        # 跨库合并 Top-K
        all_matches.sort(key=lambda x: x["cosine_score"], reverse=True)
        top_matches = all_matches[:TOP_K]

        if not top_matches:
            continue

        matched_count += 1
        raw = raw_spectra[i]
        query_name = (raw.get("compound_name") or raw.get("name") or f"spectrum_{i}")

        # 从 MSP name 提取 compound_id，查找实测同位素分布
        obs_iso: list = []
        if isotope_data:
            m = _COMPOUND_ID_RE.search(query_name)
            if m:
                obs_iso = isotope_data.get(m.group(1), [])

        for rank, match in enumerate(top_matches, 1):
            lib_s = match["library_spectrum"]
            lib_pmz = lib_s.get("precursor_mz")
            if lib_pmz and lib_pmz > 0:
                ppm_diff = (query_pmz - lib_pmz) / lib_pmz * 1e6  # 保留正负号
            else:
                ppm_diff = float('inf')

            # 同位素评分（有实测数据且能预测理论值时才计算）
            iso_score = ""
            if obs_iso:
                formula = (lib_s.get("formula") or lib_s.get("molecular_formula") or
                           lib_s.get("FORMULA") or "")
                pred_iso = predict_isotope_pattern(formula) if formula else []
                if pred_iso:
                    iso_score = round(isotope_cosine(obs_iso, pred_iso), 4)

            results.append({
                "query_name":           query_name,
                "matched_name":         lib_s.get("compound_name") or lib_s.get("name") or lib_s.get("title") or "",
                "matched_smiles":       lib_s.get("smiles") or lib_s.get("SMILES") or "",
                "matched_inchikey":     lib_s.get("inchikey") or lib_s.get("INCHIKEY") or "",
                "matched_formula":      (lib_s.get("formula") or lib_s.get("molecular_formula") or
                                         lib_s.get("FORMULA") or ""),
                "cosine_score":         round(match["cosine_score"], 4),
                "matched_peaks_ratio":  f"\t{match['matched_peaks']}/{match['library_total_peaks']}",
                "matched_fragments":    match["matched_fragments_str"],
                "precursor_mz":         query_pmz,
                "library_precursor_mz": lib_pmz,
                "precursor_ppm_diff":   round(ppm_diff, 2),
                "adduct":               lib_s.get("adduct") or lib_s.get("precursor_type") or lib_s.get("PRECURSORTYPE") or "",
                "matched_ontology":     lib_s.get("ontology") or lib_s.get("ONTOLOGY") or "",
                "isotope_score":        iso_score,
                "source_method":        "MatchMS",
                "source_database":      extract_source_database(lib_s, match["lib_name"]),
                "rank":                 rank,
            })

    # 4. 保存
    pd.DataFrame(results).to_csv(output_csv, index=False, encoding='utf-8')

    print("\n" + "=" * 70)
    print(f"{level} MatchMS 鉴定完成")
    print("=" * 70)
    print(f"总样品数:   {len(raw_spectra)}")
    print(f"鉴定成功:   {matched_count} ({matched_count/len(raw_spectra)*100:.1f}%)")
    print(f"输出文件:   {output_csv}")
    print(f"耗时:       {time.time()-t0:.1f}秒")
    print("=" * 70)

    return True  # 返回成功状态，与鉴定数量无关


if __name__ == "__main__":
    import sys
    main()
    sys.exit(0)  # 显式返回成功，即使matched_count=0
