#!/usr/bin/env python3
"""
L2: FIORA 模拟谱图预测（独立脚本，在 fiora 环境运行）

功能：
1. 加载候选化合物库中所有 SMILES
2. 应用 SMILES 修复（除盐、去电荷、括号修复等）
3. 以 FIORA 版本号为缓存标签，已有缓存则跳过
4. 无缓存 → 运行 FIORA 预测 → 保存 MSP（带 Source_tool: FIORA 标签）

输出：
- {candidate_lib_dir}/simulated_fiora_v{X}_{ION_MODE}.msp  (缓存，直接作为最终输出)
"""

###############################################################################
# 导入模块
###############################################################################

import os
import sys
import subprocess
import time
import json
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
FIORA_DEVICE = os.environ.get('L2_FIORA_DEVICE', 'cuda:0')
FORCE_REGENERATE = os.environ.get('L2_FORCE_REGENERATE', '0')

# 验证必需参数
_missing = []
if not ION_MODE:
    _missing.append('L2_ION_MODE')
if not OUTPUT_DIR:
    _missing.append('L2_OUTPUT_DIR')
if not CANDIDATE_LIBRARY:
    _missing.append('L2_CANDIDATE_LIBRARY')

if _missing:
    raise ValueError(f"错误：以下环境变量未设置：{', '.join(_missing)}")

# 类型转换
FORCE_REGENERATE = FORCE_REGENERATE.lower() in ('1', 'true', 'yes')

# FIORA 二进制路径
FIORA_BIN = '/home/lyx/miniconda3/envs/fiora/bin/fiora-predict'

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

def get_fiora_version():
    """获取 FIORA 版本号"""
    try:
        result = subprocess.run(
            ['/home/lyx/miniconda3/envs/fiora/bin/pip', 'show', 'fiora'],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.splitlines():
            if line.startswith('Version:'):
                return line.split(':')[1].strip()
    except Exception:
        pass
    return "unknown"


###############################################################################
# === FIORA 预测 ===
###############################################################################

def parse_existing_predictions(mgf_path):
    """解析已有的 MGF 输出，返回已预测的 pred_id 集合
    
    注意：此函数仅用于兼容旧版断点续传（无 JSON 进度文件时）
    推荐使用 JSON 进度文件进行断点续传
    """
    if not os.path.exists(mgf_path):
        return set()
    
    completed = set()
    try:
        with open(mgf_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('TITLE='):
                    pred_id = line.split('=', 1)[1].strip()
                    completed.add(pred_id)
    except Exception:
        pass
    return completed


def merge_energy_spectra(spectra_list):
    """合并多个能量的谱图，相同m/z取最大强度
    
    参数:
        spectra_list: [[(mz, intensity), ...], ...] 多个能量的谱图
    返回:
        (mz_list, intensity_list) 合并后的谱图
    """
    if not spectra_list:
        return [], []
    
    # 收集所有峰
    peak_dict = {}
    for peaks in spectra_list:
        for mz, intensity in peaks:
            mz_key = round(mz, 4)
            if mz_key not in peak_dict or intensity > peak_dict[mz_key]:
                peak_dict[mz_key] = intensity
    
    # 排序并返回
    sorted_peaks = sorted(peak_dict.items(), key=lambda x: x[0])
    mz_out = [p[0] for p in sorted_peaks]
    int_out = [p[1] for p in sorted_peaks]
    return mz_out, int_out


def format_spectrum_msp(cand, peaks, fiora_ver, ion_mode):
    """格式化单个谱图为 MSP 文本
    
    参数:
        cand: 候选化合物信息字典
        peaks: [(mz, intensity), ...] 峰列表
        fiora_ver: FIORA 版本号
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
    lines.append(f"INSTRUMENT: FIORA")
    lines.append(f"COLLISIONENERGY: 15/30/45")
    lines.append(f"Comment: Source_tool=FIORA; Source_tool_version=FIORA_v{fiora_ver}; Source_database={cand.get('source', '')}")
    lines.append(f"Num Peaks: {len(peaks)}")
    
    for mz, intensity in peaks:
        lines.append(f"{mz:.6f}\t{intensity:.4f}")
    
    return "\n".join(lines)


def run_fiora_prediction(candidates, precursor_type, work_dir, output_msp, fiora_ver, ion_mode, device="cuda:0", resume=True, progress_file=None, batch_size=10):
    """运行 FIORA 预测（多能量 + 分批调用 + 实时写入 + tqdm 进度条 + 断点续传）
    
    对每个化合物预测 3 个能量（15, 30, 45），然后合并为单条谱图。
    每批完成后立即写入 MSP 文件（实时更新）。
    
    参数:
        candidates: 候选化合物列表
        precursor_type: 前体类型
        work_dir: 工作目录
        output_msp: 输出 MSP 文件路径
        fiora_ver: FIORA 版本号
        ion_mode: 离子模式 (POS/NEG)
        device: GPU 设备
        resume: 是否启用断点续传
        progress_file: 进度记录文件路径（JSON格式）
        batch_size: 每批处理的化合物数量（默认10）
    """
    # 创建 FIORA 输出子目录（与 CFM-ID 统一）
    fiora_out_dir = work_dir / 'fiora_output'
    fiora_out_dir.mkdir(exist_ok=True)
    
    input_csv = fiora_out_dir / 'fiora_input.csv'
    output_mgf = fiora_out_dir / 'fiora_output.mgf'  # 单能量临时输出
    
    # 多能量配置
    collision_energies = [15, 30, 45]  # NCE %
    
    # 断点续传：优先从 JSON 进度文件加载
    completed_ids = set()
    
    if resume:
        if progress_file and os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    completed_ids = set(progress_data.get('completed', []))
                if completed_ids:
                    print(f"  断点续传: 已完成 {len(completed_ids)} 个化合物")
            except Exception as e:
                print(f"  警告: 进度文件加载失败 ({e})")
                completed_ids = set()
    
    # 过滤掉已完成的候选
    pending_candidates = [c for c in candidates if c['pred_id'] not in completed_ids]
    
    if not pending_candidates:
        print(f"  所有候选已完成预测！")
        return completed_ids
    
    print(f"  FIORA 预测: {len(pending_candidates)} 个待处理 / {len(candidates)} 总计...")
    print(f"  多能量: {collision_energies} (NCE%)")
    print(f"  输出: {output_msp}")
    
    # 分批
    batches = []
    for i in range(0, len(pending_candidates), batch_size):
        batches.append(pending_candidates[i:i+batch_size])
    print(f"  分 {len(batches)} 批处理 (每批 {batch_size} 个)")
    
    success_count = 0
    fail_count = 0
    
    # 打开输出文件（追加模式，实现实时写入）
    # 首次创建时写入 BOM（兼容 WPS/Excel），后续追加用普通 UTF-8
    if not os.path.exists(output_msp) or os.path.getsize(output_msp) == 0:
        with open(output_msp, 'w', encoding='utf-8-sig') as f:
            pass  # 仅写入 BOM 头
    with open(output_msp, 'a', encoding='utf-8') as f_out:
        for batch_idx, batch in enumerate(tqdm(batches, desc="  FIORA", ncols=80)):
            batch_predictions = {}  # 本批结果 {pred_id: [(mz,int), ...]}
            
            # 对每个能量分别预测
            for ce in collision_energies:
                # 准备本能量输入 CSV
                rows = []
                for c in batch:
                    rows.append({
                        'Name': c['pred_id'],
                        'SMILES': c['smiles'],
                        'Precursor_type': precursor_type,
                        'CE': ce,
                        'Instrument_type': 'HCD'
                    })
                pd.DataFrame(rows).to_csv(input_csv, index=False)
                
                # 清理临时输出
                if output_mgf.exists():
                    output_mgf.unlink()
                
                cmd = [FIORA_BIN, '-i', str(input_csv), '-o', str(output_mgf),
                       '--dev', device, '--min_prob', '0.001']
                
                try:
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                        text=True
                    )
                    try:
                        returncode = proc.wait(timeout=300)  # 单能量单批最多5分钟
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                        tqdm.write(f"  批次 {batch_idx+1} CE={ce} 超时")
                        continue
                    
                    if returncode != 0:
                        tqdm.write(f"  批次 {batch_idx+1} CE={ce} 失败")
                        continue
                except Exception as e:
                    tqdm.write(f"  批次 {batch_idx+1} CE={ce} 异常: {e}")
                    continue
                
                # 解析本能量输出
                energy_predictions = parse_fiora_output(output_mgf)
                
                # 收集各能量的谱图
                for pred_id, peaks in energy_predictions.items():
                    if pred_id not in batch_predictions:
                        batch_predictions[pred_id] = []
                    batch_predictions[pred_id].append(peaks)
            
            # 合并多能量谱图并实时写入
            cand_map = {c['pred_id']: c for c in batch}
            for pred_id, spectra_list in batch_predictions.items():
                if spectra_list:
                    merged_mz, merged_int = merge_energy_spectra(spectra_list)
                    peaks = list(zip(merged_mz, merged_int))
                    
                    # 格式化并立即写入
                    cand = cand_map.get(pred_id, {})
                    spectrum_text = format_spectrum_msp(cand, peaks, fiora_ver, ion_mode)
                    if spectrum_text:
                        f_out.write(spectrum_text)
                        f_out.write("\n\n")  # MSP 谱图间空行分隔
                        success_count += 1
                        completed_ids.add(pred_id)
            
            # 每批写入后刷新到磁盘
            f_out.flush()
            
            # 更新 JSON 进度文件
            if progress_file and completed_ids:
                try:
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump({'completed': list(completed_ids)}, f)
                except Exception:
                    pass
            
            # 统计失败
            batch_success = len([p for p in batch_predictions.values() if p])
            fail_count += len(batch) - batch_success
    
    # 清理临时文件
    for tmp in [input_csv, output_mgf]:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
    
    print(f"  FIORA 预测完成: {success_count} 成功, {fail_count} 失败 / {len(pending_candidates)} 待处理")
    return completed_ids


def parse_fiora_output(output_mgf):
    """解析 FIORA 输出 MGF 文件"""
    predictions = {}
    if not output_mgf.exists():
        return predictions

    current_id = None
    current_peaks = []

    with open(output_mgf, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == 'BEGIN IONS':
                current_id = None
                current_peaks = []
            elif line.startswith('TITLE='):
                current_id = line.split('=', 1)[1].strip()
            elif line == 'END IONS':
                if current_id and current_peaks:
                    predictions[current_id] = current_peaks
                current_id = None
                current_peaks = []
            elif line and not line.startswith('#') and '=' not in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        current_peaks.append((float(parts[0]), float(parts[1])))
                    except Exception:
                        pass
    
    return predictions


###############################################################################
# === MSP 输出 ===
###############################################################################

def save_fiora_msp(predictions, candidates, output_path, fiora_ver, ion_mode, mode='w'):
    """保存 FIORA 预测结果为 MSP 格式（MSDIAL 兼容表头）
    
    参数:
        predictions: 预测结果字典 {pred_id: (mz_list, int_list)} 或 {pred_id: [(mz, int), ...]}
        candidates: 候选化合物列表
        output_path: 输出文件路径
        fiora_ver: FIORA 版本号
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

            # 处理两种格式: tuple (mz_list, int_list) 或 list [(mz, int), ...]
            if isinstance(peaks, tuple) and len(peaks) == 2:
                mz_list, int_list = peaks
                peak_pairs = list(zip(mz_list, int_list))
            elif isinstance(peaks, list):
                peak_pairs = peaks
            else:
                continue

            if not peak_pairs:
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
            f.write(f"INSTRUMENT: FIORA\n")
            f.write(f"COLLISIONENERGY: 15/30/45\n")  # 多能量合并
            f.write(f"Comment: Source_tool=FIORA; Source_tool_version=FIORA_v{fiora_ver}; Source_database={cand.get('source', '')}\n")
            f.write(f"Num Peaks: {len(peak_pairs)}\n")

            for mz, intensity in peak_pairs:
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
    print("L2: FIORA 模拟谱图预测")
    print("=" * 70)
    print(f"  离子模式: {ION_MODE}")
    print(f"  候选库: {CANDIDATE_LIBRARY}")
    print(f"  GPU 设备: {FIORA_DEVICE}")
    print(f"  强制重新生成: {FORCE_REGENERATE}")
    print("=" * 70)

    # 确定目录
    candidate_lib_path = Path(CANDIDATE_LIBRARY)
    candidate_lib_dir = str(candidate_lib_path.parent if candidate_lib_path.is_file() else candidate_lib_path)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 版本标签
    print("\n[1/3] 获取 FIORA 版本...")
    fiora_ver = get_fiora_version()
    version_tag = f"fiora_v{fiora_ver}"
    print(f"  版本: {version_tag}")

    # 2. 检查缓存
    cached_msp = os.path.join(candidate_lib_dir, f"simulated_{version_tag}_{ION_MODE}.msp")
    print(f"  缓存路径: {cached_msp}")

    if os.path.exists(cached_msp) and not FORCE_REGENERATE:
        with open(cached_msp, 'r', encoding='utf-8', errors='ignore') as f:
            count = sum(1 for line in f if line.strip().lower().startswith('name:'))
        print(f"\n[CACHE HIT] 发现缓存，跳过预测 ({count} 条谱图)")

        # 输出 MSP 路径文件（供总控脚本读取）
        msp_path_file = os.path.join(OUTPUT_DIR, "fiora_msp_path.txt")
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
        msp_path_file = os.path.join(OUTPUT_DIR, "fiora_msp_path.txt")
        with open(msp_path_file, 'w') as f:
            f.write(cached_msp)
        return True

    for i, c in enumerate(all_candidates):
        # 使用 InChIKey 作为唯一标识（不受候选库顺序变化影响）
        c['pred_id'] = c.get('inchikey') or f"FIORA_{i}"
        c['idx'] = i

    print("\n[3/3] 运行 FIORA 预测...")
    # 中间文件直接放在候选库目录下（与MSP缓存在一起，便于统一管理）
    work_dir = Path(candidate_lib_dir)
    
    # 进度文件放在子目录中（与 CFM-ID 统一）
    fiora_out_dir = work_dir / 'fiora_output'
    fiora_out_dir.mkdir(exist_ok=True)
    progress_file = fiora_out_dir / 'fiora_progress.json'
    
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
    
    # 运行预测（实时写入 MSP）
    completed_set = run_fiora_prediction(
        all_candidates, precursor_type, work_dir, cached_msp, fiora_ver, ION_MODE,
        FIORA_DEVICE, resume=not FORCE_REGENERATE, progress_file=str(progress_file)
    )
    
    # 统计最终文件中的谱图数
    count = 0
    if os.path.exists(cached_msp):
        with open(cached_msp, 'r', encoding='utf-8', errors='ignore') as f:
            count = sum(1 for line in f if line.strip().startswith('NAME:'))
    
    # 输出 MSP 路径文件（供总控脚本读取）
    msp_path_file = os.path.join(OUTPUT_DIR, "fiora_msp_path.txt")
    with open(msp_path_file, 'w') as f:
        f.write(cached_msp)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("FIORA 预测完成")
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
