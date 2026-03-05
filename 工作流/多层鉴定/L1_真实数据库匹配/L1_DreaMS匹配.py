#!/usr/bin/env python3
"""
L1: DreaMS Embedding匹配（方法导向）

匹配策略：DreaMS embedding cosine相似度方法遍历所有数据库
- 支持多数据库：MSDIAL、SpectraTraverse
- 需要：样品embedding + 库embedding缓存
- 输出统一格式的 L1_DreaMS_results.csv
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
from matchms.similarity import CosineGreedy
from matchms.filtering import normalize_intensities
from tqdm import tqdm


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

# DreaMS需要PYTHONPATH
DREAMS_SRC = "/stor3/AIMS4Meta/源代码/DreaMS"
if DREAMS_SRC not in sys.path:
    sys.path.insert(0, DREAMS_SRC)


def load_sample_spectra(sample_msp):
    """加载样品质谱数据"""
    print(f"[INFO] 加载样品MSP: {sample_msp}")
    spectra = list(load_from_msp(sample_msp))
    print(f"[INFO] 成功加载 {len(spectra)} 个样品谱图")
    return spectra


def load_sample_embeddings(sample_emb_path):
    """加载样品的DreaMS embedding"""
    if sample_emb_path and os.path.exists(sample_emb_path):
        print(f"[INFO] 加载样品embedding: {sample_emb_path}")
        return np.load(sample_emb_path)['embeddings'].astype(np.float32)
    print(f"[ERROR] 样品embedding不存在: {sample_emb_path}")
    return None


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
    print(f"[INFO] 加载{lib_name}数据库（解析MSP）: {lib_path}")
    library = list(load_from_msp(lib_path))
    print(f"[INFO] 成功加载 {len(library)} 个参考谱图")
    return library


def load_library_from_npz(emb_path):
    """
    从NPZ加载库的embedding和元数据
    返回: (embeddings, metadata_dict, has_metadata)
    """
    if not os.path.exists(emb_path):
        return None, None, False
    
    data = np.load(emb_path, allow_pickle=True)
    embeddings = data['embeddings'].astype(np.float32)
    
    # 检查是否有元数据
    has_metadata = 'names' in data
    
    if has_metadata:
        metadata = {
            'names': data['names'],
            'smiles': data['smiles'],
            'inchikeys': data['inchikeys'],
            'precursor_mzs': data['precursor_mzs'],
            'adducts': data['adducts'],
            'ccs_list': data['ccs_list'],
            'ontologies': data['ontologies'],
            'formulas': data['formulas'] if 'formulas' in data else np.array([''] * len(data['names']), dtype=object)
        }
        return embeddings, metadata, True
    
    return embeddings, None, False


def load_library_embeddings(emb_path):
    """加载库的DreaMS embedding（旧接口，保持兼容）"""
    if os.path.exists(emb_path):
        print(f"[INFO] 加载库embedding: {emb_path}")
        return np.load(emb_path)['embeddings'].astype(np.float32)
    print(f"[WARN] 库embedding不存在: {emb_path}")
    return None


def build_mz_index_from_metadata(precursor_mzs):
    """从元数据构建母离子m/z倒排索引（快速）"""
    mz_index = defaultdict(list)
    for i, pmz in enumerate(precursor_mzs):
        if pmz and pmz > 0:
            bin_key = int(round(pmz * 100))
            mz_index[bin_key].append(i)
    return mz_index


def build_mz_index(library_spectra):
    """构建母离子m/z倒排索引（从MSP谱图，较慢）"""
    mz_index = defaultdict(list)
    for i, spectrum in enumerate(library_spectra):
        precursor_mz = spectrum.get("precursor_mz")
        if precursor_mz:
            bin_key = int(round(precursor_mz * 100))
            mz_index[bin_key].append(i)
    return mz_index


def find_matched_fragments(query_peaks, lib_peaks, tolerance, query_pmz, lib_pmz):
    """找出匹配的碎片峰详情 - 统一使用L1_MC的Numba加速版本"""
    # 导入L1_MC的碎片匹配函数 (所有层级统一使用)
    l1_mc_path = os.path.join(os.path.dirname(__file__), 'L1_MC匹配.py')
    if os.path.exists(l1_mc_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("L1_MC", l1_mc_path)
        l1_mc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(l1_mc)
        return l1_mc.find_matched_fragments(query_peaks, lib_peaks, tolerance, query_pmz, lib_pmz)

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


def match_with_dreams_fast(query_emb, candidates, lib_embs_norm, lib_metadata,
                           cosine_threshold, query_pmz, top_k=1):
    """
    快速DreaMS embedding匹配（无需谱图，仅用元数据）
    
    返回: (matches_list, stats) - matches_list 是按分数排序的匹配列表，最多 top_k 个
    """
    if not candidates:
        return [], {"filter_reason": "no_candidates"}
    
    # 归一化查询embedding
    from numpy.linalg import norm
    q_norm = norm(query_emb)
    if q_norm == 0:
        return [], {"filter_reason": "zero_embedding"}
    q_emb_n = query_emb / q_norm
    
    # 计算embedding相似度
    sims = lib_embs_norm[candidates] @ q_emb_n
    
    # 收集所有满足阈值的匹配
    all_matches = []
    sorted_idx = np.argsort(sims)[::-1]
    
    for local_idx in sorted_idx:
        emb_score = sims[local_idx]
        if emb_score < cosine_threshold:
            break
        
        lib_idx = candidates[local_idx]
        all_matches.append({
            "lib_idx": lib_idx,
            "cosine_score": emb_score,
            "lib_name": str(lib_metadata['names'][lib_idx]) if lib_metadata['names'][lib_idx] else "",
            "lib_smiles": str(lib_metadata['smiles'][lib_idx]) if lib_metadata['smiles'][lib_idx] else "",
            "lib_inchikey": str(lib_metadata['inchikeys'][lib_idx]) if lib_metadata['inchikeys'][lib_idx] else "",
            "lib_pmz": float(lib_metadata['precursor_mzs'][lib_idx]),
            "lib_adduct": str(lib_metadata['adducts'][lib_idx]) if lib_metadata['adducts'][lib_idx] else "",
            "lib_ccs": str(lib_metadata['ccs_list'][lib_idx]) if lib_metadata['ccs_list'][lib_idx] else "",
            "lib_ontology": str(lib_metadata['ontologies'][lib_idx]) if lib_metadata['ontologies'][lib_idx] else "",
            "lib_formula": str(lib_metadata['formulas'][lib_idx]) if lib_metadata['formulas'][lib_idx] else "",
            "matched_peaks": 0,
            "matched_fragments": []
        })
    
    return all_matches[:top_k], {"filter_reason": "success" if all_matches else "low_score"}


def match_with_dreams(query_spectrum, query_emb, candidates, lib_embs_norm,
                      library_spectra, fragment_tolerance, min_matched_peaks, cosine_threshold, top_k=1):
    """
    使用DreaMS embedding相似度匹配
    
    返回: (matches_list, stats) - matches_list 是按分数排序的匹配列表，最多 top_k 个
    """
    if not candidates:
        return [], {"filter_reason": "no_candidates"}
    
    # 归一化查询embedding
    from numpy.linalg import norm
    q_norm = norm(query_emb)
    if q_norm == 0:
        return [], {"filter_reason": "zero_embedding"}
    q_emb_n = query_emb / q_norm
    
    # 计算embedding相似度
    sims = lib_embs_norm[candidates] @ q_emb_n
    
    # 归一化查询谱图
    query_norm = normalize_intensities(query_spectrum)
    if query_norm is None:
        return [], {"filter_reason": "normalize_failed"}
    
    query_peaks = list(zip(query_norm.peaks.mz, query_norm.peaks.intensities))
    counter = CosineGreedy(tolerance=fragment_tolerance)
    
    # 收集所有满足条件的匹配
    all_matches = []
    
    # 按embedding相似度排序检查
    sorted_idx = np.argsort(sims)[::-1]
    for local_idx in sorted_idx:
        lib_idx = candidates[local_idx]
        emb_score = sims[local_idx]
        
        if emb_score < cosine_threshold:
            break
        
        lib_spectrum = library_spectra[lib_idx]
        lib_norm = normalize_intensities(lib_spectrum)
        if lib_norm is None:
            continue
        
        try:
            r = counter.pair(query_norm, lib_norm)
            if r["matches"] >= min_matched_peaks:
                lib_peaks = list(zip(lib_norm.peaks.mz, lib_norm.peaks.intensities))
                matched_frags = find_matched_fragments(
                    query_peaks, lib_peaks, fragment_tolerance,
                    query_spectrum.get("precursor_mz"),
                    lib_spectrum.get("precursor_mz")
                )
                
                all_matches.append({
                    "library_index": lib_idx,
                    "cosine_score": emb_score,
                    "matched_peaks": r["matches"],
                    "matched_fragments": matched_frags,
                    "library_spectrum": lib_spectrum
                })
        except Exception:
            continue
    
    # 按分数排序，取 Top-K
    all_matches.sort(key=lambda x: x["cosine_score"], reverse=True)
    return all_matches[:top_k], {"filter_reason": "success" if all_matches else "low_score"}


def main():
    """主函数 - DreaMS Embedding匹配，遍历所有数据库"""

    ###########################################################################
    # 读取环境变量 + 验证
    ###########################################################################
    SAMPLE_MSP = os.environ.get('L1_SAMPLE_MSP')
    SAMPLE_EMB = os.environ.get('L1_SAMPLE_EMB')
    ION_MODE = os.environ.get('L1_ION_MODE')
    LIBRARIES_JSON = os.environ.get('L1_LIBRARIES')
    MZ_TOLERANCE_PPM = os.environ.get('L1_MZ_TOLERANCE_PPM')
    FRAGMENT_TOLERANCE = os.environ.get('L1_FRAGMENT_TOLERANCE')
    MIN_MATCHED_PEAKS = os.environ.get('L1_MIN_MATCHED_PEAKS')
    COSINE_THRESHOLD = os.environ.get('L1_COSINE_THRESHOLD')
    OUTPUT_DIR = os.environ.get('L1_OUTPUT_DIR', './L1_DreaMS_results')
    TOP_K = os.environ.get('L1_TOP_K')  # L1/L2 共用此参数

    _missing = []
    for var_name, var_val in [
        ('L1_SAMPLE_MSP', SAMPLE_MSP), ('L1_SAMPLE_EMB', SAMPLE_EMB),
        ('L1_ION_MODE', ION_MODE), ('L1_LIBRARIES', LIBRARIES_JSON),
        ('L1_MZ_TOLERANCE_PPM', MZ_TOLERANCE_PPM), ('L1_FRAGMENT_TOLERANCE', FRAGMENT_TOLERANCE),
        ('L1_MIN_MATCHED_PEAKS', MIN_MATCHED_PEAKS), ('L1_COSINE_THRESHOLD', COSINE_THRESHOLD),
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
    parser = argparse.ArgumentParser(description="L1 DreaMS Embedding匹配（方法导向，多库遍历）")
    parser.add_argument("--sample_msp", default=SAMPLE_MSP, help="样品MSP文件路径")
    parser.add_argument("--sample_emb", default=SAMPLE_EMB, help="样品DreaMS embedding文件路径（NPZ）")
    parser.add_argument("--ion_mode", default=ION_MODE, choices=["POS", "NEG"], help="离子模式")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--sample_csv", help="原始样本CSV文件")
    parser.add_argument("--mz_ppm", type=float, default=MZ_TOLERANCE_PPM, help="母离子容差(ppm)")
    parser.add_argument("--fragment_tolerance", type=float, default=FRAGMENT_TOLERANCE, help="碎片容差(Da)")
    parser.add_argument("--min_matched_peaks", type=int, default=MIN_MATCHED_PEAKS, help="最少匹配碎片数")
    parser.add_argument("--cosine_threshold", type=float, default=COSINE_THRESHOLD, help="余弦阈值")
    parser.add_argument("--output_csv", help="输出CSV文件路径（默认：{output_dir}/L1_DreaMS_results.csv）")
    args = parser.parse_args()

    if not args.sample_emb:
        raise ValueError("错误：必须提供 --sample_emb 参数或设置 L1_SAMPLE_EMB 环境变量")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 输出文件路径
    if args.output_csv:
        output_csv = args.output_csv
    else:
        output_csv = os.path.join(args.output_dir, "L1_DreaMS_results.csv")
    
    print("=" * 70)
    print("L1 DreaMS Embedding匹配（方法导向，多库遍历）")
    print("=" * 70)
    print(f"[参数] 母离子容差: {args.mz_ppm} ppm")
    print(f"[参数] 碎片容差: {args.fragment_tolerance} Da")
    print(f"[参数] 最少匹配碎片: {args.min_matched_peaks}")
    print(f"[参数] 余弦阈值: {args.cosine_threshold}")
    print(f"[参数] Top-K: {TOP_K}")
    
    start_time = time.time()
    
    # 1. 加载样品
    sample_spectra = load_sample_spectra(args.sample_msp)
    sample_embs = load_sample_embeddings(args.sample_emb)
    
    if sample_embs is None:
        print("[ERROR] 无法加载样品embedding，退出")
        return 0
    
    if len(sample_embs) != len(sample_spectra):
        print(f"[WARN] embedding数量({len(sample_embs)})与谱图数量({len(sample_spectra)})不匹配")
    
    # 2. 加载所有数据库
    libraries = {}        # 谱图列表（仅当npz无元数据时使用）
    lib_metadata = {}     # 元数据（从npz加载）
    mz_indices = {}
    lib_embeddings = {}
    lib_embs_norm = {}
    
    for lib_name, lib_config in LIBRARIES.items():
        msp_path = lib_config["msp"]  # 直接是字符串路径
        emb_path = lib_config["emb"]  # 直接是字符串路径
        
        if not (emb_path and os.path.exists(emb_path)):
            print(f"[WARN] {lib_name} embedding不存在，跳过")
            continue
        
        # 优先从NPZ加载（快速）
        emb, meta, has_meta = load_library_from_npz(emb_path)
        
        if has_meta:
            # NPZ有元数据，但仍需加载MSP做碎片匹配
            print(f"[INFO] {lib_name} 从NPZ加载embedding+元数据: {emb.shape}")
            lib_embeddings[lib_name] = emb
            lib_metadata[lib_name] = meta
            mz_indices[lib_name] = build_mz_index_from_metadata(meta['precursor_mzs'])
            # 加载MSP谱图用于碎片匹配
            if msp_path and os.path.exists(msp_path):
                print(f"[INFO] {lib_name} 加载MSP谱图用于碎片匹配...")
                libraries[lib_name] = load_library(lib_name, msp_path)
            else:
                print(f"[WARN] {lib_name} MSP文件不存在，将无法进行碎片匹配")
        elif msp_path and os.path.exists(msp_path):
            # NPZ无元数据，回退到解析MSP
            print(f"[INFO] {lib_name} NPZ无元数据，回退到解析MSP...")
            libraries[lib_name] = load_library(lib_name, msp_path)
            mz_indices[lib_name] = build_mz_index(libraries[lib_name])
            lib_embeddings[lib_name] = emb
        else:
            print(f"[WARN] {lib_name} 无法加载（无元数据且MSP缺失），跳过")
            continue
        
        # 预计算归一化embedding
        from numpy.linalg import norm
        ln = norm(lib_embeddings[lib_name], axis=1, keepdims=True)
        ln[ln == 0] = 1
        lib_embs_norm[lib_name] = lib_embeddings[lib_name] / ln
    
    # 3. 执行匹配
    results = []
    matched_samples = set()
    
    # 检查是否有可用的库
    all_lib_names = set(lib_metadata.keys()) | set(libraries.keys())
    if not all_lib_names:
        print("[ERROR] 没有可用的数据库")
        return 0
    
    print(f"\n[INFO] 开始匹配 {len(sample_spectra)} 个样品谱图...")
    
    mz_tol = args.mz_ppm / 1e6
    
    for i, query_spectrum in enumerate(tqdm(sample_spectra, desc="[DreaMS匹配]", unit="条", ncols=80)):
        query_pmz = query_spectrum.get("precursor_mz")
        if not query_pmz:
            continue
        
        if i >= len(sample_embs):
            continue
        
        query_emb = sample_embs[i]
        
        # 在所有库中收集匹配结果
        all_matches = []
        
        for lib_name in all_lib_names:
            mz_index = mz_indices[lib_name]
            lib_emb_n = lib_embs_norm[lib_name]
            
            # 获取候选
            tol = query_pmz * mz_tol
            candidates = []
            for b in range(int(round((query_pmz-tol)*100))-1, int(round((query_pmz+tol)*100))+2):
                if b in mz_index:
                    candidates.extend(mz_index[b])
            
            if not candidates:
                continue
            
            # 根据是否有谱图选择匹配方法
            if lib_name in libraries:
                # 有谱图，做完整匹配（含碎片匹配）
                lib_spectra = libraries[lib_name]
                # 精确过滤候选
                candidates = [c for c in set(candidates) 
                             if abs(query_pmz - lib_spectra[c].get("precursor_mz", 0)) <= tol]
                if not candidates:
                    continue
                matches, _ = match_with_dreams(
                    query_spectrum, query_emb, candidates, lib_emb_n,
                    lib_spectra, args.fragment_tolerance,
                    args.min_matched_peaks, args.cosine_threshold,
                    top_k=TOP_K
                )
            elif lib_name in lib_metadata:
                # 无谱图但有元数据，快速匹配（无碎片匹配）
                matches, _ = match_with_dreams_fast(
                    query_emb, candidates, lib_emb_n, lib_metadata[lib_name],
                    args.cosine_threshold, query_pmz, top_k=TOP_K
                )
            else:
                continue
            
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
                # 从匹配结果提取信息
                lib_key = match["lib_name"]  # 库字典key（如 "simulated"、"MSDIAL"）
                
                if "lib_idx" in match:
                    # 快速匹配结果（来自元数据）
                    matched_name = match["lib_name"]
                    lib_smiles = match["lib_smiles"]
                    lib_inchikey = match["lib_inchikey"]
                    lib_pmz = match["lib_pmz"]
                    lib_adduct = match["lib_adduct"]
                    lib_ccs = match["lib_ccs"]
                    lib_ontology = match["lib_ontology"]
                    lib_formula = match.get("lib_formula", "")
                    matched_frags_str = ""
                    # 提取 source_database：尝试从库谱图获取工具/来源信息
                    _lib_spec = None
                    if lib_key in libraries and match["lib_idx"] < len(libraries[lib_key]):
                        _lib_spec = libraries[lib_key][match["lib_idx"]]
                    source_db = extract_source_database(_lib_spec, lib_key)
                else:
                    # 传统匹配结果（来自谱图）
                    lib_spectrum = match["library_spectrum"]
                    lib_pmz = lib_spectrum.get("precursor_mz")
                    matched_name = (
                        lib_spectrum.get("compound_name") or 
                        lib_spectrum.get("name") or 
                        lib_spectrum.get("title") or ""
                    )
                    lib_smiles = lib_spectrum.get("smiles") or lib_spectrum.get("SMILES") or ""
                    lib_inchikey = lib_spectrum.get("inchikey") or lib_spectrum.get("INCHIKEY") or ""
                    lib_adduct = lib_spectrum.get("adduct", "")
                    lib_ccs = lib_spectrum.get("ccs") or lib_spectrum.get("CCS") or ""
                    lib_ontology = lib_spectrum.get("ontology") or lib_spectrum.get("ONTOLOGY") or ""
                    # 提取分子式
                    lib_formula = (lib_spectrum.get("formula") or lib_spectrum.get("FORMULA") or 
                                  lib_spectrum.get("molecular_formula") or lib_spectrum.get("FORMULA_MOLECULAR") or "")
                    matched_frags_str = "; ".join([
                        f"{f['query_mz']}/{f['lib_mz']}(Δ{f['mz_diff']:.4f}Da)"
                        for f in match.get("matched_fragments", [])
                    ])
                    source_db = extract_source_database(lib_spectrum, lib_key)
                
                # 保留正负号: 正值表示样品母离子大于库母离子,负值表示样品母离子小于库母离子
                ppm_diff = (query_pmz - lib_pmz) / query_pmz * 1e6 if query_pmz and lib_pmz else 0
                
                results.append({
                    "query_name": query_name,
                    "matched_name": matched_name,
                    "matched_smiles": lib_smiles,
                    "matched_inchikey": lib_inchikey,
                    "matched_formula": lib_formula,
                    "cosine_score": round(float(match["cosine_score"]), 4),
                    "matched_peaks": match.get("matched_peaks", 0),
                    "precursor_mz": query_pmz,
                    "library_precursor_mz": lib_pmz,
                    "precursor_ppm_diff": round(ppm_diff, 2),
                    "adduct": lib_adduct,
                    "library_ccs": lib_ccs,
                    "matched_ontology": lib_ontology,
                    "matched_fragments": matched_frags_str,
                    "source_method": "DreaMS",
                    "source_database": source_db,
                    "rank": rank
                })
    
    # 4. 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False, encoding='utf-8')
    
    # 5. 输出统计
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("L1 DreaMS Embedding匹配完成")
    print("=" * 70)
    print(f"总样品数:       {len(sample_spectra)}")
    print(f"匹配成功:       {len(matched_samples)} ({len(matched_samples)/len(sample_spectra)*100:.1f}%)")
    print(f"输出文件:       {output_csv}")
    print(f"耗时:           {elapsed_time:.1f}秒")
    print("=" * 70)
    
    return len(matched_samples)


if __name__ == "__main__":
    main()
