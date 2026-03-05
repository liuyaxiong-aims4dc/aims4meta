#!/usr/bin/env python3
"""
L1: ModifiedCosine匹配（方法导向）

匹配策略：ModifiedCosine方法遍历所有数据库
- 支持多数据库：MSDIAL、SpectraTraverse
- 对每个样品谱图，在所有库中寻找最佳匹配
- 输出统一格式的 L1_MC_results.csv
"""

import argparse
import os
import sys
import time
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from matchms.importing import load_from_msp
from matchms.similarity import ModifiedCosine
from matchms.filtering import normalize_intensities
from tqdm import tqdm

# 导入Numba碎片匹配模块
numba_fragment_path = os.path.join(os.path.dirname(__file__), '..', '辅助功能', 'Numba碎片匹配', 'numba_fragment_matching.py')
if os.path.exists(numba_fragment_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("numba_fragment_matching", numba_fragment_path)
    numba_frag_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(numba_frag_module)
    find_matched_fragments_numba = numba_frag_module.find_matched_fragments
    USE_NUMBA = True
else:
    USE_NUMBA = False


def extract_source_database(lib_spectrum, fallback_name):
    """从库谱图元数据中提取来源信息
    
    对于模拟库谱图（Comment含Source_tool=），返回如 'CFM-ID/FDA_approved_drugs_positive'
    对于真实库谱图，直接返回 fallback_name（如 'MSDIAL'、'Spectraverse'）
    """
    if lib_spectrum is None:
        return fallback_name
    comment = lib_spectrum.get("comment", "") or ""
    if not comment or "Source_tool=" not in comment:
        return fallback_name
    # 模拟库：提取 Source_tool/Source_database
    source_tool = ""
    source_db = ""
    for part in comment.split(';'):
        part = part.strip()
        if part.startswith('Source_tool='):
            source_tool = part.split('=', 1)[1].strip()
        elif part.startswith('Source_database='):
            source_db = part.split('=', 1)[1].strip()
    if source_tool and source_db:
        return f"{source_tool}/{source_db}"
    elif source_tool:
        return source_tool
    return fallback_name


def load_sample_spectra(sample_msp):
    """加载样品质谱数据"""
    print(f"[INFO] 加载样品MSP: {sample_msp}")
    spectra = list(load_from_msp(sample_msp))
    print(f"[INFO] 成功加载 {len(spectra)} 个样品谱图")
    return spectra


def load_library(lib_name, lib_path):
    """加载单个数据库（优先使用缓存）"""
    import pickle
    
    # 检查是否有谱图缓存
    pkl_path = lib_path.replace('.msp', '_spectra_cache.pkl')
    
    if os.path.exists(pkl_path):
        print(f"[INFO] 加载{lib_name}缓存: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            library = pickle.load(f)
        print(f"[INFO] 成功加载 {len(library)} 个参考谱图（缓存）")
        return library
    
    # 无缓存，从MSP加载
    print(f"[INFO] 加载{lib_name}数据库: {lib_path}")
    library = list(load_from_msp(lib_path))
    print(f"[INFO] 成功加载 {len(library)} 个参考谱图")
    return library


def build_mz_index(library_spectra):
    """构建母离子m/z倒排索引（整数bin键）"""
    mz_index = defaultdict(list)
    for i, spectrum in enumerate(library_spectra):
        precursor_mz = spectrum.get("precursor_mz")
        if precursor_mz:
            bin_key = int(round(precursor_mz * 100))
            mz_index[bin_key].append(i)
    return mz_index


def find_matched_fragments(query_peaks, lib_peaks, tolerance, query_pmz, lib_pmz):
    """找出匹配的碎片峰详情 (支持Numba加速)"""
    # 如果已加载Numba模块,使用加速版本
    if USE_NUMBA:
        return find_matched_fragments_numba(query_peaks, lib_peaks, tolerance, query_pmz, lib_pmz, use_numba=True)

    # Fallback: CPU版本
    matched = []
    pmz_diff = query_pmz - lib_pmz if query_pmz and lib_pmz else 0
    used_lib = set()

    for q_mz, q_int in query_peaks:
        if q_int <= 0:
            continue

        best_match = None
        best_diff = float('inf')

        for i, (l_mz, l_int) in enumerate(lib_peaks):
            if i in used_lib or l_int <= 0:
                continue

            # 直接匹配
            diff = abs(q_mz - l_mz)
            # Precursor shift匹配
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


def match_spectra_mc(query_spectrum, library_spectra, mz_index, 
                     mz_tolerance_ppm, min_matched_peaks, 
                     cosine_threshold, fragment_tolerance, top_k=1):
    """
    使用ModifiedCosine匹配单个查询谱图与库谱图
    
    返回: (matches_list, stats) - matches_list 是按分数排序的匹配列表，最多 top_k 个
    """
    precursor_mz = query_spectrum.get("precursor_mz")
    if not precursor_mz:
        return [], {"filter_reason": "no_precursor"}
    
    # m/z预筛选
    mz_tolerance = precursor_mz * mz_tolerance_ppm / 1e6
    min_bin = int(round((precursor_mz - mz_tolerance) * 100)) - 1
    max_bin = int(round((precursor_mz + mz_tolerance) * 100)) + 1
    
    candidates = set()
    for bin_key in range(min_bin, max_bin + 1):
        if bin_key in mz_index:
            candidates.update(mz_index[bin_key])
    
    if not candidates:
        return [], {"candidates_count": 0, "filter_reason": "no_candidates"}
    
    # 精确过滤
    valid_candidates = []
    for lib_idx in candidates:
        lib_precursor = library_spectra[lib_idx].get("precursor_mz")
        if lib_precursor and abs(precursor_mz - lib_precursor) <= mz_tolerance:
            valid_candidates.append(lib_idx)
    
    if not valid_candidates:
        return [], {"candidates_count": len(candidates), "filter_reason": "no_valid_candidates"}
    
    # 归一化查询谱图
    query_norm = normalize_intensities(query_spectrum)
    if query_norm is None:
        return [], {"candidates_count": len(valid_candidates), "filter_reason": "normalize_failed"}
    
    query_peaks = list(zip(query_norm.peaks.mz, query_norm.peaks.intensities))
    modified_cosine = ModifiedCosine(tolerance=fragment_tolerance)
    
    # 收集所有满足条件的匹配
    all_matches = []
    
    for lib_idx in valid_candidates:
        lib_spectrum = library_spectra[lib_idx]
        lib_norm = normalize_intensities(lib_spectrum)
        if lib_norm is None:
            continue
        
        try:
            score = modified_cosine.pair(query_norm, lib_norm)
            
            if score["matches"] >= min_matched_peaks and score["score"] >= cosine_threshold:
                lib_peaks = list(zip(lib_norm.peaks.mz, lib_norm.peaks.intensities))
                matched_frags = find_matched_fragments(
                    query_peaks, lib_peaks, fragment_tolerance,
                    precursor_mz, lib_spectrum.get("precursor_mz")
                )
                
                all_matches.append({
                    "library_index": lib_idx,
                    "cosine_score": score["score"],
                    "matched_peaks": score["matches"],
                    "matched_fragments": matched_frags,
                    "library_spectrum": lib_spectrum
                })
        except Exception:
            continue
    
    # 按分数排序，取 Top-K
    all_matches.sort(key=lambda x: x["cosine_score"], reverse=True)
    return all_matches[:top_k], {"candidates_count": len(valid_candidates), "filter_reason": "success" if all_matches else "low_score"}


def main():
    """主函数 - ModifiedCosine匹配，遍历所有数据库"""

    ###########################################################################
    # 读取环境变量 + 验证
    ###########################################################################
    SAMPLE_MSP = os.environ.get('L1_SAMPLE_MSP')
    ION_MODE = os.environ.get('L1_ION_MODE')
    LIBRARIES_JSON = os.environ.get('L1_MC_LIBRARIES')
    MZ_TOLERANCE_PPM = os.environ.get('L1_MZ_TOLERANCE_PPM')
    FRAGMENT_TOLERANCE = os.environ.get('L1_FRAGMENT_TOLERANCE')
    MIN_MATCHED_PEAKS = os.environ.get('L1_MIN_MATCHED_PEAKS')
    COSINE_THRESHOLD = os.environ.get('L1_COSINE_THRESHOLD')
    OUTPUT_DIR = os.environ.get('L1_OUTPUT_DIR')
    TOP_K = os.environ.get('L1_TOP_K')  # L1/L2 共用此参数

    _missing = []
    for var_name, var_val in [
        ('L1_SAMPLE_MSP', SAMPLE_MSP), ('L1_ION_MODE', ION_MODE),
        ('L1_MC_LIBRARIES', LIBRARIES_JSON), ('L1_MZ_TOLERANCE_PPM', MZ_TOLERANCE_PPM),
        ('L1_FRAGMENT_TOLERANCE', FRAGMENT_TOLERANCE), ('L1_MIN_MATCHED_PEAKS', MIN_MATCHED_PEAKS),
        ('L1_COSINE_THRESHOLD', COSINE_THRESHOLD), ('L1_OUTPUT_DIR', OUTPUT_DIR),
    ]:
        if not var_val:
            _missing.append(var_name)
    if _missing:
        raise ValueError(f"错误：以下环境变量未设置，必须由总控脚本提供：{', '.join(_missing)}")

    LIBRARIES = json.loads(LIBRARIES_JSON)
    MZ_TOLERANCE_PPM = int(float(MZ_TOLERANCE_PPM))
    FRAGMENT_TOLERANCE = float(FRAGMENT_TOLERANCE)
    MIN_MATCHED_PEAKS = int(MIN_MATCHED_PEAKS)
    COSINE_THRESHOLD = float(COSINE_THRESHOLD)
    TOP_K = int(TOP_K) if TOP_K else 1

    ###########################################################################
    # 命令行参数（可覆盖环境变量）
    ###########################################################################
    parser = argparse.ArgumentParser(description="L1 ModifiedCosine匹配（方法导向，多库遍历）")
    parser.add_argument("--sample_msp", default=SAMPLE_MSP, help="样品MSP文件路径")
    parser.add_argument("--ion_mode", default=ION_MODE, choices=["POS", "NEG"], help="离子模式")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--sample_csv", help="原始样本CSV文件")
    parser.add_argument("--mz_ppm", type=float, default=MZ_TOLERANCE_PPM, help="母离子容差(ppm)")
    parser.add_argument("--fragment_tolerance", type=float, default=FRAGMENT_TOLERANCE, help="碎片容差(Da)")
    parser.add_argument("--min_matched_peaks", type=int, default=MIN_MATCHED_PEAKS, help="最少匹配碎片数")
    parser.add_argument("--cosine_threshold", type=float, default=COSINE_THRESHOLD, help="余弦阈值")
    parser.add_argument("--output_csv", help="输出CSV文件路径（默认：{output_dir}/L1_MC_results.csv）")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 输出文件路径
    if args.output_csv:
        output_csv = args.output_csv
    else:
        output_csv = os.path.join(args.output_dir, "L1_MC_results.csv")
    
    print("=" * 70)
    print("L1 ModifiedCosine匹配（方法导向，多库遍历）")
    print("=" * 70)
    print(f"[参数] 母离子容差: {args.mz_ppm} ppm")
    print(f"[参数] 碎片容差: {args.fragment_tolerance} Da")
    print(f"[参数] 最少匹配碎片: {args.min_matched_peaks}")
    print(f"[参数] 余弦阈值: {args.cosine_threshold}")
    print(f"[参数] Top-K: {TOP_K}")
    
    start_time = time.time()
    
    # 1. 加载样品
    sample_spectra = load_sample_spectra(args.sample_msp)
    
    # 2. 加载所有数据库并构建索引
    libraries = {}
    mz_indices = {}
    
    for lib_name, lib_path in LIBRARIES.items():
        if lib_path and os.path.exists(lib_path):
            libraries[lib_name] = load_library(lib_name, lib_path)
            mz_indices[lib_name] = build_mz_index(libraries[lib_name])
        else:
            print(f"[WARN] 数据库路径不存在: {lib_path}")
    
    # 3. 执行匹配
    results = []  # 存储所有匹配结果
    matched_samples = set()  # 已匹配的样品索引
    
    print(f"\n[INFO] 开始匹配 {len(sample_spectra)} 个样品谱图...")
    
    for i, query_spectrum in enumerate(tqdm(sample_spectra, desc="[MC匹配]", unit="条", ncols=80)):
        query_pmz = query_spectrum.get("precursor_mz")
        if not query_pmz:
            continue
        
        # 在所有库中收集匹配结果
        all_matches = []
        
        for lib_name in libraries:
            lib_spectra = libraries[lib_name]
            mz_index = mz_indices[lib_name]
            
            matches, _ = match_spectra_mc(
                query_spectrum, lib_spectra, mz_index,
                args.mz_ppm, args.min_matched_peaks,
                args.cosine_threshold, args.fragment_tolerance,
                top_k=TOP_K  # 每个库取 Top-K
            )
            
            for match in matches:
                match["lib_name"] = lib_name
                all_matches.append(match)
        
        # 合并所有库的结果，按分数排序取 Top-K
        all_matches.sort(key=lambda x: x["cosine_score"], reverse=True)
        top_matches = all_matches[:TOP_K]
        
        if top_matches:
            matched_samples.add(i)
            query_name = query_spectrum.get("compound_name") or query_spectrum.get("name") or f"spectrum_{i}"
            
            for rank, match in enumerate(top_matches, 1):
                lib_spectrum = match["library_spectrum"]
                lib_pmz = lib_spectrum.get("precursor_mz")
                # 保留正负号: 正值表示样品母离子大于库母离子,负值表示样品母离子小于库母离子
                ppm_diff = (query_pmz - lib_pmz) / query_pmz * 1e6 if query_pmz and lib_pmz else 0
                
                matched_name = (
                    lib_spectrum.get("compound_name") or 
                    lib_spectrum.get("name") or 
                    lib_spectrum.get("title") or ""
                )
                
                matched_frags_str = "; ".join([
                    f"{f['query_mz']}/{f['lib_mz']}(Δ{f['mz_diff']:.4f}Da)"
                    for f in match["matched_fragments"]
                ])
                
                lib_smiles = lib_spectrum.get("smiles") or lib_spectrum.get("SMILES") or ""
                lib_inchikey = lib_spectrum.get("inchikey") or lib_spectrum.get("INCHIKEY") or ""
                
                # 提取分子式
                lib_formula = (lib_spectrum.get("formula") or lib_spectrum.get("FORMULA") or 
                              lib_spectrum.get("molecular_formula") or lib_spectrum.get("FORMULA_MOLECULAR") or "")
                
                results.append({
                    "query_name": query_name,
                    "matched_name": matched_name,
                    "matched_smiles": lib_smiles,
                    "matched_inchikey": lib_inchikey,
                    "matched_formula": lib_formula,
                    "cosine_score": round(float(match["cosine_score"]), 4),
                    "matched_peaks": match["matched_peaks"],
                    "precursor_mz": query_pmz,
                    "library_precursor_mz": lib_pmz,
                    "precursor_ppm_diff": round(ppm_diff, 2),
                    "adduct": lib_spectrum.get("adduct", ""),
                    "library_ccs": lib_spectrum.get("ccs") or lib_spectrum.get("CCS") or "",
                    "matched_ontology": lib_spectrum.get("ontology") or lib_spectrum.get("ONTOLOGY") or "",
                    "matched_fragments": matched_frags_str,
                    "source_method": "MC",
                    "source_database": extract_source_database(lib_spectrum, match["lib_name"]),
                    "rank": rank
                })
    
    # 4. 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False, encoding='utf-8')
    
    # 5. 输出统计
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("L1 ModifiedCosine匹配完成")
    print("=" * 70)
    print(f"总样品数:       {len(sample_spectra)}")
    print(f"匹配成功:       {len(matched_samples)} ({len(matched_samples)/len(sample_spectra)*100:.1f}%)")
    print(f"输出文件:       {output_csv}")
    print(f"耗时:           {elapsed_time:.1f}秒")
    print("=" * 70)
    
    return len(matched_samples)


if __name__ == "__main__":
    main()
