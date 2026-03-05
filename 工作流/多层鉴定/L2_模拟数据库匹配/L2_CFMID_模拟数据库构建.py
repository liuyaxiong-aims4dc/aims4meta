#!/usr/bin/env python3
"""
L2: CFM-ID 模拟谱图预测（独立脚本，在 dreams 环境运行）

功能：
1. 加载候选化合物库中所有 SMILES
2. 应用 SMILES 修复（除盐、去电荷、括号修复等）
3. 以 CFM-ID 版本号为缓存标签，已有缓存则跳过
4. 无缓存 → 多进程并行运行 CFM-ID 预测 → 保存 MSP（带 Source_tool: CFM-ID 标签）

输出：
- {candidate_lib_dir}/simulated_cfmid_v{X}_{ION_MODE}.msp  (缓存，直接作为最终输出)
"""

###############################################################################
# 导入模块
###############################################################################

import os
import sys
import subprocess
import time
import json
import multiprocessing as mp
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# 导入 SMILES 修复工具（辅助功能）
_smiles_fix_path = Path(__file__).parent.parent / "辅助功能" / "SMILES修复"
sys.path.insert(0, str(_smiles_fix_path))
from SMILES修复 import fix_smiles, calculate_precursor_mz, generate_inchikey, generate_formula

###############################################################################
# 配置参数区（全部由总控脚本环境变量注入）
###############################################################################

ION_MODE = os.environ.get('L2_ION_MODE')
OUTPUT_DIR = os.environ.get('L2_OUTPUT_DIR')
CANDIDATE_LIBRARY = os.environ.get('L2_CANDIDATE_LIBRARY')
CFMID_DIR = os.environ.get('L2_CFMID_DIR')
CFMID_WORKERS = os.environ.get('L2_CFMID_WORKERS')
FORCE_REGENERATE = os.environ.get('L2_FORCE_REGENERATE', '0')

# 验证必需参数
_missing = []
if not ION_MODE:
    _missing.append('L2_ION_MODE')
if not OUTPUT_DIR:
    _missing.append('L2_OUTPUT_DIR')
if not CANDIDATE_LIBRARY:
    _missing.append('L2_CANDIDATE_LIBRARY')
if not CFMID_DIR:
    _missing.append('L2_CFMID_DIR')
if not CFMID_WORKERS:
    _missing.append('L2_CFMID_WORKERS')

if _missing:
    raise ValueError(f"错误：以下环境变量未设置：{', '.join(_missing)}")

# 类型转换
CFMID_WORKERS = int(CFMID_WORKERS)
FORCE_REGENERATE = FORCE_REGENERATE.lower() in ('1', 'true', 'yes')

# CFM-ID 路径
CFMID_PREDICT = os.path.join(CFMID_DIR, "cfm", "build", "bin", "cfm-predict")
CFMID_RDKIT_LIB = os.path.join(CFMID_DIR, "rdkit-2017.09.3", "lib")
CFMID_LIB = os.path.join(CFMID_DIR, "cfm", "build", "lib")

###############################################################################
# === 候选化合物加载 ===
###############################################################################

def load_all_candidates(library_path, precursor_type):
    """加载候选化合物库中所有 SMILES（全量加载 + SMILES 修复）"""
    lib_path = Path(library_path)
    candidates = []
    seen_smiles = set()
    fix_stats = {'total': 0, 'fixed': 0, 'failed': 0}

    if lib_path.is_file() and lib_path.suffix == '.csv':
        files = [lib_path]
    elif lib_path.is_dir():
        files = list(lib_path.glob('*.csv'))
    else:
        print(f"  [WARNING] 候选库路径无效: {library_path}")
        return []

    for csv_file in files:
        try:
            df = pd.read_csv(csv_file)
            col_map = {col.lower(): col for col in df.columns}
            if 'smiles' not in col_map:
                print(f"  [SKIP] {csv_file.name}: 无 SMILES 列")
                continue

            smiles_col = col_map['smiles']
            name_col = col_map.get('name', '')
            inchikey_col = col_map.get('inchikey', '')
            formula_col = col_map.get('formula', col_map.get('molecular_formula', ''))

            for _, row in df.iterrows():
                smi = row.get(smiles_col)
                if pd.isna(smi) or not str(smi).strip():
                    continue

                smi = str(smi).strip()
                fix_stats['total'] += 1

                # 应用 SMILES 修复
                fixed_smi, fix_msg = fix_smiles(smi)
                if fixed_smi is None:
                    fix_stats['failed'] += 1
                    continue
                if fix_msg != "无需修复":
                    fix_stats['fixed'] += 1

                if fixed_smi in seen_smiles:
                    continue
                seen_smiles.add(fixed_smi)

                precursor_mz = calculate_precursor_mz(fixed_smi, precursor_type)
                if not precursor_mz:
                    continue

                # InChIKey / Formula：CSV 有则用，无则从 SMILES 自动生成
                inchikey = str(row.get(inchikey_col, '')) if inchikey_col else ''
                if not inchikey or inchikey == 'nan':
                    inchikey = generate_inchikey(fixed_smi)
                formula = str(row.get(formula_col, '')) if formula_col else ''
                if not formula or formula == 'nan':
                    formula = generate_formula(fixed_smi)

                candidates.append({
                    'smiles': fixed_smi,
                    'original_smiles': smi,
                    'name': str(row.get(name_col, 'Unknown')) if name_col else 'Unknown',
                    'source': csv_file.stem,
                    'precursor_mz': precursor_mz,
                    'inchikey': inchikey,
                    'formula': formula,
                })
        except Exception as e:
            print(f"  [ERROR] 加载 {csv_file.name} 失败: {e}")

    print(f"  SMILES 修复统计: 总计{fix_stats['total']}, 修复{fix_stats['fixed']}, 失败{fix_stats['failed']}")
    return candidates


###############################################################################
# === 版本缓存 ===
###############################################################################

def get_cfmid_version():
    """获取 CFM-ID 版本号（从预训练模型目录名推断）"""
    models_base = os.path.join(CFMID_DIR, "cfm-pretrained-models")
    if os.path.isdir(models_base):
        for name in os.listdir(models_base):
            if name.startswith("cfmid"):
                return name.replace("cfmid", "")
    return "unknown"


###############################################################################
# === CFM-ID 预测 ===
###############################################################################

def _cfmid_predict_single(args):
    """单个 CFM-ID 预测（多进程 worker）"""
    pred_id, smiles, ion_mode, output_dir, cfmid_predict, cfmid_dir, rdkit_lib, cfm_lib = args

    output_file = os.path.join(output_dir, f"{pred_id}.log")
    models_base = os.path.join(cfmid_dir, "cfm-pretrained-models", "cfmid4")

    if ion_mode == "POS":
        param_file = os.path.join(models_base, "[M+H]+", "param_output.log")
        config_file = os.path.join(models_base, "[M+H]+", "param_config.txt")
    else:
        param_file = os.path.join(models_base, "[M-H]-", "param_output.log")
        config_file = os.path.join(models_base, "[M-H]-", "param_config.txt")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{rdkit_lib}:{cfm_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    cmd = [cfmid_predict, smiles, "0.001", param_file, config_file, "0", output_file]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60, env=env)
        if result.returncode == 0 and os.path.exists(output_file):
            return pred_id, output_file
        return pred_id, None
    except Exception:
        return pred_id, None


def parse_cfmid_output(output_file):
    """解析 CFM-ID 输出（返回三个能量的谱图列表）"""
    if not output_file or not os.path.exists(output_file):
        return {}

    energy_peaks = {}
    current_energy = None

    with open(output_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('energy'):
                current_energy = line
                energy_peaks[current_energy] = []
                continue
            if current_energy is None:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    energy_peaks[current_energy].append((float(parts[0]), float(parts[1])))
                except Exception:
                    pass

    return energy_peaks


def merge_energy_spectra(spectra_list):
    """合并多个能量的谱图，相同m/z取最大强度
    
    参数:
        spectra_list: [[(mz, intensity), ...], ...] 多个能量的谱图
    返回:
        [(mz, intensity), ...] 合并后的谱图列表
    """
    if not spectra_list:
        return []
    
    # 收集所有峰
    peak_dict = {}
    for peaks in spectra_list:
        for mz, intensity in peaks:
            mz_key = round(mz, 4)
            if mz_key not in peak_dict or intensity > peak_dict[mz_key]:
                peak_dict[mz_key] = intensity
    
    # 排序并返回
    sorted_peaks = sorted(peak_dict.items(), key=lambda x: x[0])
    return [(p[0], p[1]) for p in sorted_peaks]


def format_spectrum_msp_cfmid(cand, peaks, cfmid_ver, ion_mode):
    """格式化单个 CFM-ID 谱图为 MSP 文本
    
    参数:
        cand: 候选化合物信息字典
        peaks: [(mz, intensity), ...] 峰列表
        cfmid_ver: CFM-ID 版本号
        ion_mode: 离子模式 (POS/NEG)
    
    返回:
        MSP 格式的文本字符串
    """
    if not peaks:
        return None
    
    precursor_type = '[M+H]+' if ion_mode == 'POS' else '[M-H]-'
    ionmode_str = 'Positive' if ion_mode == 'POS' else 'Negative'
    
    lines = []
    lines.append(f"NAME: {cand.get('name', 'Unknown')}")
    lines.append(f"PRECURSORMZ: {cand.get('precursor_mz', 0):.6f}")
    lines.append(f"PRECURSORTYPE: {precursor_type}")
    lines.append(f"FORMULA: {cand.get('formula', '')}")
    lines.append(f"Ontology: ")
    lines.append(f"INCHIKEY: {cand.get('inchikey', '')}")
    lines.append(f"SMILES: {cand.get('smiles', '')}")
    lines.append(f"RETENTIONTIME: ")
    lines.append(f"CCS: ")
    lines.append(f"IONMODE: {ionmode_str}")
    lines.append(f"INSTRUMENTTYPE: In-silico")
    lines.append(f"INSTRUMENT: CFM-ID")
    lines.append(f"COLLISIONENERGY: energy0/energy1/energy2")
    lines.append(f"Comment: Source_tool=CFM-ID; Source_tool_version=CFMID_v{cfmid_ver}; Source_database={cand.get('source', '')}")
    lines.append(f"Num Peaks: {len(peaks)}")
    
    for mz, intensity in peaks:
        lines.append(f"{mz:.6f}\t{intensity:.4f}")
    
    return "\n".join(lines)


def run_cfmid_prediction(candidates, ion_mode, work_dir, output_msp, cfmid_ver, workers=5, progress_file=None, resume=True):
    """运行 CFM-ID 多进程预测（支持断点续传 + 实时写入）
    
    参数:
        candidates: 候选化合物列表
        ion_mode: 离子模式 (POS/NEG)
        work_dir: 工作目录
        output_msp: 输出 MSP 文件路径（实时追加写入）
        cfmid_ver: CFM-ID 版本号
        workers: 并行进程数
        progress_file: 进度文件路径
        resume: 是否启用断点续传
    """
    # 断点续传：从 JSON 加载已完成的 InChIKey
    completed_ids = set()
    if resume and progress_file and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                completed_ids = set(progress_data.get('completed', []))
            if completed_ids:
                print(f"  断点续传: 已完成 {len(completed_ids)} 个化合物")
        except Exception:
            pass
    
    cfmid_out_dir = work_dir / 'cfmid_output'
    cfmid_out_dir.mkdir(exist_ok=True)

    # 过滤掉已完成的候选（基于 InChIKey/pred_id）
    pending_candidates = [c for c in candidates if c['pred_id'] not in completed_ids]
    
    if not pending_candidates:
        print(f"  所有候选已完成预测！")
        return completed_ids
    
    print(f"  CFM-ID 预测: {len(pending_candidates)} 个待处理 / {len(candidates)} 总计, {workers} 进程...")

    args_list = [
        (c['pred_id'], c['smiles'], ion_mode, str(cfmid_out_dir),
         CFMID_PREDICT, CFMID_DIR, CFMID_RDKIT_LIB, CFMID_LIB)
        for c in pending_candidates
    ]

    cand_map = {c['pred_id']: c for c in candidates}
    success_count = 0
    fail_count = 0
    newly_completed = []
    
    # 首次创建时写入 BOM（兼容 WPS/Excel），后续追加用普通 UTF-8
    if not os.path.exists(output_msp) or os.path.getsize(output_msp) == 0:
        with open(output_msp, 'w', encoding='utf-8-sig') as f:
            pass  # 仅写入 BOM 头
    
    with open(output_msp, 'a', encoding='utf-8') as f_out:
        with mp.Pool(workers) as pool:
            for result in tqdm(
                pool.imap(_cfmid_predict_single, args_list),
                total=len(args_list), desc="  CFM-ID", ncols=80
            ):
                pred_id, output_file = result
                if output_file:
                    energy_peaks = parse_cfmid_output(output_file)
                    if energy_peaks:
                        # 合并三个能量的谱图（energy0/energy1/energy2）
                        spectra_list = []
                        for key in ('energy0', 'energy1', 'energy2'):
                            if key in energy_peaks and energy_peaks[key]:
                                spectra_list.append(energy_peaks[key])
                        if spectra_list:
                            peaks = merge_energy_spectra(spectra_list)
                            # 格式化并立即写入 MSP
                            cand = cand_map.get(pred_id, {})
                            spectrum_text = format_spectrum_msp_cfmid(cand, peaks, cfmid_ver, ion_mode)
                            if spectrum_text:
                                f_out.write(spectrum_text)
                                f_out.write("\n\n")
                                success_count += 1
                            completed_ids.add(pred_id)
                            newly_completed.append(pred_id)
                else:
                    fail_count += 1
                
                # 定期保存进度并刷新文件（每 20 个化合物）
                if len(newly_completed) % 20 == 0 and newly_completed:
                    f_out.flush()
                    if progress_file:
                        try:
                            with open(progress_file, 'w') as f:
                                json.dump({'completed': list(completed_ids)}, f)
                        except Exception:
                            pass

    # 保存最终进度
    if progress_file:
        try:
            with open(progress_file, 'w') as f:
                json.dump({'completed': list(completed_ids)}, f)
        except Exception:
            pass
    
    print(f"  CFM-ID 完成: {success_count} 成功, {fail_count} 失败 / {len(pending_candidates)} 待处理")
    return completed_ids


###############################################################################
# === MSP 输出 ===
###############################################################################

def save_cfmid_msp(predictions, candidates, output_path, cfmid_ver, ion_mode, mode='w'):
    """保存 CFM-ID 预测结果为 MSP 格式（MSDIAL 兼容表头）
    
    参数:
        predictions: 预测结果字典
        candidates: 候选化合物列表
        output_path: 输出文件路径
        cfmid_ver: CFM-ID 版本号
        ion_mode: 离子模式 (POS/NEG)
        mode: 写入模式 ('w' 覆盖, 'a' 追加)
    """
    cand_map = {c['pred_id']: c for c in candidates}
    count = 0

    precursor_type = '[M+H]+' if ion_mode == 'POS' else '[M-H]-'
    ionmode_str = 'Positive' if ion_mode == 'POS' else 'Negative'

    # 使用 UTF-8 with BOM，让 WPS/Excel 等软件自动识别编码
    encoding = 'utf-8-sig' if mode == 'w' else 'utf-8'
    with open(output_path, mode, encoding=encoding) as f:
        for pred_id, peaks in predictions.items():
            cand = cand_map.get(pred_id, {})
            if not peaks:
                continue

            name = cand.get('name', 'Unknown')
            f.write(f"NAME: {name}\n")
            f.write(f"PRECURSORMZ: {cand.get('precursor_mz', 0):.6f}\n")
            f.write(f"PRECURSORTYPE: {precursor_type}\n")
            f.write(f"FORMULA: {cand.get('formula', '')}\n")
            f.write(f"Ontology:\n")
            f.write(f"INCHIKEY: {cand.get('inchikey', '')}\n")
            f.write(f"SMILES: {cand.get('smiles', '')}\n")
            f.write(f"RETENTIONTIME:\n")
            f.write(f"CCS:\n")
            f.write(f"IONMODE: {ionmode_str}\n")
            f.write(f"INSTRUMENTTYPE: In-silico\n")
            f.write(f"INSTRUMENT: CFM-ID\n")
            f.write(f"COLLISIONENERGY: energy0/energy1/energy2\n")
            f.write(f"Comment: Source_tool=CFM-ID; Source_tool_version=CFMID_v{cfmid_ver}; Source_database={cand.get('source', '')}\n")
            f.write(f"Num Peaks: {len(peaks)}\n")

            for mz, intensity in peaks:
                f.write(f"{mz:.6f}\t{intensity:.4f}\n")
            f.write("\n")
            count += 1

    print(f"  已保存: {output_path} ({count} 条谱图)")
    return count


###############################################################################
# === 主函数 ===
###############################################################################

def main():
    """主函数"""
    start_time = time.time()

    print("=" * 70)
    print("L2: CFM-ID 模拟谱图预测")
    print("=" * 70)
    print(f"  离子模式: {ION_MODE}")
    print(f"  候选库: {CANDIDATE_LIBRARY}")
    print(f"  并行进程: {CFMID_WORKERS}")
    print(f"  强制重新生成: {FORCE_REGENERATE}")
    print("=" * 70)

    # 确定目录
    candidate_lib_path = Path(CANDIDATE_LIBRARY)
    candidate_lib_dir = str(candidate_lib_path.parent if candidate_lib_path.is_file() else candidate_lib_path)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 版本标签
    print("\n[1/3] 获取 CFM-ID 版本...")
    cfmid_ver = get_cfmid_version()
    version_tag = f"cfmid_v{cfmid_ver}"
    print(f"  版本: {version_tag}")

    # 2. 检查缓存
    cached_msp = os.path.join(candidate_lib_dir, f"simulated_{version_tag}_{ION_MODE}.msp")
    print(f"  缓存路径: {cached_msp}")

    if os.path.exists(cached_msp) and not FORCE_REGENERATE:
        with open(cached_msp, 'r', encoding='utf-8', errors='ignore') as f:
            count = sum(1 for line in f if line.strip().lower().startswith('name:'))
        print(f"\n[CACHE HIT] 发现缓存，跳过预测 ({count} 条谱图)")

        # 输出 MSP 路径文件（供总控脚本读取）
        msp_path_file = os.path.join(OUTPUT_DIR, "cfmid_msp_path.txt")
        with open(msp_path_file, 'w') as f:
            f.write(cached_msp)

        print(f"\n耗时: {time.time() - start_time:.1f} 秒")
        return True

    # 3. 加载候选 + 预测
    print("\n[2/3] 加载候选化合物（含 SMILES 修复）...")
    precursor_type = '[M+H]+' if ION_MODE == 'POS' else '[M-H]-'
    all_candidates = load_all_candidates(CANDIDATE_LIBRARY, precursor_type)
    print(f"  有效候选: {len(all_candidates)} 个")

    if not all_candidates:
        print("[WARNING] 无候选化合物，生成空库")
        with open(cached_msp, 'w') as f:
            f.write("")
        msp_path_file = os.path.join(OUTPUT_DIR, "cfmid_msp_path.txt")
        with open(msp_path_file, 'w') as f:
            f.write(cached_msp)
        return True

    for i, c in enumerate(all_candidates):
        # 使用 InChIKey 作为唯一标识（不受候选库顺序变化影响）
        c['pred_id'] = c.get('inchikey') or f"CFMID_{i}"
        c['idx'] = i

    print("\n[3/3] 运行 CFM-ID 预测...")
    # 中间文件直接放在候选库目录下（与MSP缓存在一起，便于统一管理）
    work_dir = Path(candidate_lib_dir)
    
    # 进度文件放在子目录中（与 FIORA 统一）
    cfmid_out_dir = work_dir / 'cfmid_output'
    cfmid_out_dir.mkdir(exist_ok=True)
    progress_file = cfmid_out_dir / 'cfmid_progress.json'

    # 强制重新生成时，清空已有文件和进度
    if FORCE_REGENERATE:
        if os.path.exists(cached_msp):
            os.remove(cached_msp)
        if progress_file.exists():
            progress_file.unlink()
    
    # 安全检查：进度文件存在但 MSP 文件不存在，说明数据不一致，重置进度
    if progress_file.exists() and (not os.path.exists(cached_msp) or os.path.getsize(cached_msp) == 0):
        print("  [WARNING] 进度文件存在但 MSP 文件缺失，重置进度...")
        progress_file.unlink()

    completed_ids = run_cfmid_prediction(
        all_candidates, ION_MODE, work_dir, cached_msp, cfmid_ver,
        workers=CFMID_WORKERS, progress_file=str(progress_file),
        resume=not FORCE_REGENERATE
    )
    
    # 统计最终文件中的谱图数
    count = 0
    if os.path.exists(cached_msp):
        with open(cached_msp, 'r', encoding='utf-8', errors='ignore') as f:
            count = sum(1 for line in f if line.strip().startswith('NAME:'))

    # 输出 MSP 路径文件（供总控脚本读取）
    msp_path_file = os.path.join(OUTPUT_DIR, "cfmid_msp_path.txt")
    with open(msp_path_file, 'w') as f:
        f.write(cached_msp)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("CFM-ID 预测完成")
    print("=" * 70)
    print(f"  版本: {version_tag}")
    print(f"  缓存: {cached_msp}")
    print(f"  谱图: {count} 条 / {len(all_candidates)} 候选")
    print(f"  耗时: {elapsed:.1f} 秒")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
