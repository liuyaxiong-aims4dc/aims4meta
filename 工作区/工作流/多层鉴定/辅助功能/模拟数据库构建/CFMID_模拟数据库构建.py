#!/usr/bin/env python3
"""
L2: CFM-ID 模拟谱图预测（独立脚本，在 cfmid 环境运行）

功能：
1. 加载候选化合物库中所有 SMILES
2. 应用 SMILES 修复（除盐、去电荷、括号修复等）
3. 以 CFM-ID 版本号为缓存标签，已有缓存则跳过
4. 无缓存 → 多进程并行运行 CFM-ID 预测 → 保存 MSP（带 Source_tool: CFM-ID 标签）

使用方式：
    # 1. 修改下方"配置参数区"的参数
    # 2. 运行脚本
    python CFMID_模拟数据库构建.py

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

# ===== Conda 环境自动激活 =====
def _ensure_conda_env():
    """确保在 aims4meta conda 环境中运行，否则自动重启动"""
    required_env = "aims4meta"
    conda_python = os.path.expanduser(f"~/miniconda3/envs/{required_env}/bin/python")
    # 用 sys.executable 路径判断是否已切换（不用 CONDA_DEFAULT_ENV，避免子进程死循环）
    if os.path.realpath(sys.executable) == os.path.realpath(conda_python):
        return
    if not os.path.isfile(conda_python):
        print(f"[WARNING] 未找到 {required_env} 环境: {conda_python}")
        print(f"  请手动运行: conda activate {required_env}")
        return
    print(f"[INFO] 自动切换到 {required_env} 环境...")
    try:
        proc = subprocess.run([conda_python] + sys.argv)
        sys.exit(proc.returncode)
    except Exception as e:
        print(f"[ERROR] 切换失败: {e}")
        sys.exit(1)

_ensure_conda_env()
# =============================

import pandas as pd
from tqdm import tqdm

# 导入 SMILES 修复工具（辅助功能）
_smiles_fix_path = Path(__file__).parent.parent / "SMILES修复"
sys.path.insert(0, str(_smiles_fix_path))
from SMILES修复 import preprocess_smiles

###############################################################################
# 配置参数区（用户在此修改参数）
###############################################################################

# ----- 基础配置 -----
ION_MODE = "NEG"  # 离子模式: "POS" 或 "NEG"
CANDIDATE_LIBRARY = "/stor3/AIMS4Meta/数据库/模拟数据库/TCM/coconut_csv_lite-04-2026.csv"  # 候选化合物库CSV文件路径

# ----- CFM-ID 配置 -----
CFMID_DIR = "/stor3/AIMS4Meta/源代码/cfm-id-code"  # CFM-ID安装目录

# ----- 输出配置 -----
OUTPUT_DIR = ""  # 输出目录（留空则使用候选库所在目录）

# ----- 性能配置 -----
CFMID_WORKERS = ""  # 并行进程数（留空则自动检测CPU核心数的80%, 最多128）

# ----- 缓存配置 -----
FORCE_REGENERATE = False  # True=从头开始，False=续传

###############################################################################
# 参数验证与初始化
###############################################################################

# 验证必需参数
if not os.path.exists(CANDIDATE_LIBRARY):
    raise ValueError(f"错误：候选库文件不存在: {CANDIDATE_LIBRARY}")

if not os.path.exists(CFMID_DIR):
    raise ValueError(f"错误：CFM-ID目录不存在: {CFMID_DIR}")

# 自动检测CPU核心数
if not CFMID_WORKERS:
    cpu_count = mp.cpu_count()
    CFMID_WORKERS = min(int(cpu_count * 0.8), 128)  # 使用80%的核心，最多128
    print(f"  自动检测: 使用 {CFMID_WORKERS} 个worker (总核心数: {cpu_count})")
else:
    CFMID_WORKERS = int(CFMID_WORKERS)

# 如果未指定输出目录，使用候选库所在目录
if not OUTPUT_DIR:
    candidate_path = Path(CANDIDATE_LIBRARY)
    if candidate_path.is_file():
        OUTPUT_DIR = str(candidate_path.parent)
    else:
        OUTPUT_DIR = CANDIDATE_LIBRARY

# CFM-ID 路径
CFMID_PREDICT = os.path.join(CFMID_DIR, "cfm", "build", "bin", "cfm-predict")
CFMID_RDKIT_LIB = os.path.join(CFMID_DIR, "rdkit-2017.09.3", "lib")
CFMID_LIB = os.path.join(CFMID_DIR, "cfm", "build", "lib")

###############################################################################
# === 候选化合物加载 ===
###############################################################################

def load_all_candidates(library_path, precursor_type):
    """加载候选化合物库（CSV加载 + 预处理）"""
    lib_path = Path(library_path)
    filter_stats = {'duplicate': 0}

    if lib_path.is_file() and lib_path.suffix == '.csv':
        files = [lib_path]
    elif lib_path.is_dir():
        files = list(lib_path.glob('*.csv'))
    else:
        print(f"  [WARNING] 候选库路径无效: {library_path}")
        return []

    # 第一步：收集所有SMILES和元数据
    all_smiles_data = []
    for csv_file in files:
        try:
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    print(f"  [INFO] 成功加载 {csv_file.name} (编码: {encoding})")
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                print(f"  [ERROR] 无法解析 {csv_file.name}: 所有编码尝试失败")
                continue
            
            col_map = {col.lower(): col for col in df.columns}
            smiles_col = col_map.get('smiles') or col_map.get('canonical_smiles')
            if not smiles_col:
                print(f"  [SKIP] {csv_file.name}: 无 SMILES 列")
                continue
            
            name_col = col_map.get('name', '')
            
            for idx, row in df.iterrows():
                smi = row.get(smiles_col)
                if pd.isna(smi) or not str(smi).strip():
                    continue
                all_smiles_data.append({
                    'smiles': str(smi).strip(),
                    'name': str(row.get(name_col, 'Unknown')) if name_col else 'Unknown',
                    'source': csv_file.stem,
                })
        except Exception as e:
            print(f"  [ERROR] 加载 {csv_file.name} 失败: {e}")

    if not all_smiles_data:
        return []
    
    # 第二步：预处理（去重 + 修复）
    print(f"  [INFO] 收集到 {len(all_smiles_data)} 个SMILES，开始预处理...")
    smiles_list = [item['smiles'] for item in all_smiles_data]
    preprocessed, dedup_stats, smiles_to_index = preprocess_smiles(smiles_list, precursor_type, show_progress=True)
    print(f"  [INFO] 预处理完成: 总数={dedup_stats['total']}, 唯一={dedup_stats['unique']}, 重复={dedup_stats['duplicate']}")
    filter_stats['duplicate'] = dedup_stats['duplicate']
    
    # 第三步：构建候选列表（通过索引映射关联元数据，避免丢失）
    candidates = []
    for item in all_smiles_data:
        if item['smiles'] not in smiles_to_index:
            continue
        
        idx = smiles_to_index[item['smiles']]
        p = preprocessed[idx]
        candidates.append({
            'smiles': p['fixed_smiles'],
            'original_smiles': item['smiles'],
            'name': item['name'],
            'source': item['source'],
            'precursor_mz': p['precursor_mz'],
            'inchikey': p['inchikey'],
            'formula': p['formula'],
        })

    print(f"  最终候选数: {len(candidates)}")
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

    # 参数说明：
    # prob_thresh=0.001: 强度阈值（默认值）
    # annotate_fragments=0: 不输出碎片注释
    # apply_postproc=1: 使用CFM-ID默认后处理
    cmd = [cfmid_predict, smiles, "0.001", param_file, config_file, "0", output_file, "1"]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=int(os.environ.get("CFMID_TIMEOUT", 300)), env=env)
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
                    # 只取前两个值：m/z和强度
                    energy_peaks[current_energy].append((float(parts[0]), float(parts[1])))
                except Exception:
                    pass

    return energy_peaks


def format_spectrum_msp_cfmid(cand, peaks, cfmid_ver, ion_mode, collision_energy="10V/20V/40V"):
    """格式化单个 CFM-ID 谱图为 MSP 文本
    
    参数:
        cand: 候选化合物信息字典
        peaks: [(mz, intensity), ...] 峰列表
        cfmid_ver: CFM-ID 版本号
        ion_mode: 离子模式 (POS/NEG)
        collision_energy: 碰撞能量字符串
    
    返回:
        MSP 格式的文本字符串
    """
    if not peaks:
        return None
    
    precursor_type = '[M+H]+' if ion_mode == 'POS' else '[M-H]-'
    ionmode_str = 'Positive' if ion_mode == 'POS' else 'Negative'
    
    # NAME/SMILES/INCHIKEY等字段去换行（避免MSP解析器将续行误判为峰数据）
    name_val = str(cand.get('name', 'Unknown')).replace('\n', ' ').replace('\r', '')
    smiles_val = str(cand.get('smiles', '')).replace('\n', ' ').replace('\r', '')
    inchikey_val = str(cand.get('inchikey', '')).replace('\n', ' ').replace('\r', '')
    formula_val = str(cand.get('formula', '')).replace('\n', ' ').replace('\r', '')
    source_val = str(cand.get('source', '')).replace('\n', ' ').replace('\r', '')
    
    lines = []
    lines.append(f"NAME: {name_val}")
    lines.append(f"PRECURSORMZ: {cand.get('precursor_mz', 0):.6f}")
    lines.append(f"PRECURSORTYPE: {precursor_type}")
    lines.append(f"FORMULA: {formula_val}")
    lines.append(f"Ontology: ")
    lines.append(f"INCHIKEY: {inchikey_val}")
    lines.append(f"SMILES: {smiles_val}")
    lines.append(f"RETENTIONTIME: ")
    lines.append(f"CCS: ")
    lines.append(f"IONMODE: {ionmode_str}")
    lines.append(f"INSTRUMENTTYPE: In-silico")
    lines.append(f"INSTRUMENT: CFM-ID")
    lines.append(f"COLLISIONENERGY: {collision_energy}")
    lines.append(f"Comment: Source_tool=CFM-ID; Source_tool_version=CFMID_v{cfmid_ver}; Source_database={source_val}")
    lines.append(f"Num Peaks: {len(peaks)}")
    
    for mz, intensity in peaks:
        lines.append(f"{mz:.6f}\t{intensity:.4f}")
    
    return "\n".join(lines)


def run_cfmid_prediction(candidates, ion_mode, work_dir, output_msp, cfmid_ver, workers=5, progress_file=None, resume=True, initial_completed=None):
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
        initial_completed: 初始已完成的compound ID集合（从MSP解析或JSON恢复）
    """
    # 断点续传：从 JSON 或 MSP 加载已完成的 InChIKey
    completed_ids = set() if initial_completed is None else set(initial_completed)
    
    # 如果没有传入初始已完成列表，尝试从JSON恢复
    if not completed_ids and resume and progress_file and os.path.exists(progress_file):
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
                total=len(args_list), desc="  CFM-ID", ncols=100, dynamic_ncols=True, leave=True
            ):
                pred_id, output_file = result
                if output_file:
                    energy_peaks = parse_cfmid_output(output_file)
                    if energy_peaks:
                        # 为每个能量生成独立的谱图条目
                        cand = cand_map.get(pred_id, {})
                        energy_map = {
                            'energy0': '10V',
                            'energy1': '20V', 
                            'energy2': '40V'
                        }
                        
                        has_valid_spectrum = False
                        for key, ce_label in energy_map.items():
                            if key in energy_peaks and energy_peaks[key]:
                                peaks = energy_peaks[key]
                                # 格式化并立即写入 MSP
                                spectrum_text = format_spectrum_msp_cfmid(cand, peaks, cfmid_ver, ion_mode, collision_energy=ce_label)
                                if spectrum_text:
                                    f_out.write(spectrum_text)
                                    f_out.write("\n\n")
                                    has_valid_spectrum = True
                        
                        if has_valid_spectrum:
                            success_count += 1
                            completed_ids.add(pred_id)
                            newly_completed.append(pred_id)
                else:
                    fail_count += 1
                
                # 每完成一个化合物立即保存进度（避免中断导致JSON落后于MSP）
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


def parse_completed_from_msp(msp_file):
    """从MSP文件解析已完成的compound，返回{pred_id: energy_count}
    
    使用INCHIKEY作为唯一标识，因为NAME可能为'nan'
    """
    completed = {}  # {pred_id: energy_count}
    if not os.path.exists(msp_file) or os.path.getsize(msp_file) == 0:
        return completed
    
    current_inchikey = None
    energy_count = 0
    
    try:
        with open(msp_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line.startswith('INCHIKEY:'):
                    # 保存上一个compound
                    if current_inchikey and energy_count > 0:
                        completed[current_inchikey] = energy_count
                    # 开始新的compound
                    current_inchikey = line[9:].strip()
                    energy_count = 0
                elif line.startswith('COLLISIONENERGY:'):
                    energy_count += 1
        
        # 保存最后一个compound
        if current_inchikey and energy_count > 0:
            completed[current_inchikey] = energy_count
    except Exception as e:
        print(f"  警告: MSP解析失败 ({e})")
    
    return completed


def get_resume_completed_ids(candidates, msp_file, progress_file, resume=True):
    """获取已完成的compound ID集合（智能恢复）
    
    优先级：
    1. JSON进度文件存在且有效 → 使用JSON
    2. JSON不存在或无效 → 从MSP文件解析
    """
    # 创建candidate ID到pred_id的映射
    cand_map = {}
    for c in candidates:
        pid = c.get('inchikey') or c.get('pred_id', '')
        if pid:
            cand_map[pid] = c.get('pred_id', pid)
    
    # 方法1：尝试从JSON进度文件恢复
    if resume and progress_file and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                completed_ids = set(progress_data.get('completed', []))
            if completed_ids:
                print(f"  续传[JSON]: 发现 {len(completed_ids)} 个已完成化合物")
                return completed_ids
        except Exception:
            pass
    
    # 方法2：从MSP文件解析（恢复丢失的进度）
    msp_completed = parse_completed_from_msp(msp_file)
    if msp_completed:
        print(f"  续传[MSP]: 发现 {len(msp_completed)} 个化合物 ({sum(msp_completed.values())} 个能量)")
        return set(msp_completed.keys())
    
    return set()


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

            # NAME/SMILES等字段去换行（避免MSP解析器将续行误判为峰数据）
            name = str(cand.get('name', 'Unknown')).replace('\n', ' ').replace('\r', '')
            smiles = str(cand.get('smiles', '')).replace('\n', ' ').replace('\r', '')
            inchikey = str(cand.get('inchikey', '')).replace('\n', ' ').replace('\r', '')
            formula = str(cand.get('formula', '')).replace('\n', ' ').replace('\r', '')
            source = str(cand.get('source', '')).replace('\n', ' ').replace('\r', '')
            
            f.write(f"NAME: {name}\n")
            f.write(f"PRECURSORMZ: {cand.get('precursor_mz', 0):.6f}\n")
            f.write(f"PRECURSORTYPE: {precursor_type}\n")
            f.write(f"FORMULA: {formula}\n")
            f.write(f"Ontology:\n")
            f.write(f"INCHIKEY: {inchikey}\n")
            f.write(f"SMILES: {smiles}\n")
            f.write(f"RETENTIONTIME:\n")
            f.write(f"CCS:\n")
            f.write(f"IONMODE: {ionmode_str}\n")
            f.write(f"INSTRUMENTTYPE: In-silico\n")
            f.write(f"INSTRUMENT: CFM-ID\n")
            f.write(f"COLLISIONENERGY: 10V/20V/40V\n")
            f.write(f"Comment: Source_tool=CFM-ID; Source_tool_version=CFMID_v{cfmid_ver}; Source_database={source}\n")
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

    # 提取CSV文件名（不含扩展名）
    csv_basename = candidate_lib_path.stem if candidate_lib_path.is_file() else "library"

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 版本标签
    print("\n[1/3] 获取 CFM-ID 版本...")
    cfmid_ver = get_cfmid_version()
    version_tag = f"cfmid_v{cfmid_ver}"
    print(f"  版本: {version_tag}")
    
    # 2. 加载候选化合物（需要先加载才能检查缓存完整性）
    print("\n[2/3] 加载候选化合物（含 SMILES 修复）...")
    precursor_type = '[M+H]+' if ION_MODE == 'POS' else '[M-H]-'
    all_candidates = load_all_candidates(CANDIDATE_LIBRARY, precursor_type)
    print(f"  有效候选: {len(all_candidates)} 个")
    
    if not all_candidates:
        print("[WARNING] 无候选化合物，生成空库")
        cached_msp = os.path.join(candidate_lib_dir, f"{csv_basename}_simulated_{version_tag}_{ION_MODE}.msp")
        with open(cached_msp, 'w') as f:
            f.write("")
        msp_path_file = os.path.join(OUTPUT_DIR, "cfmid_msp_path.txt")
        with open(msp_path_file, 'w') as f:
            f.write(cached_msp)
        return True

    # 3. 检查缓存（文件名包含CSV名称）
    cached_msp = os.path.join(candidate_lib_dir, f"{csv_basename}_simulated_{version_tag}_{ION_MODE}.msp")
    print(f"  缓存路径: {cached_msp}")
    
    # 为候选设置pred_id
    for i, c in enumerate(all_candidates):
        c['pred_id'] = c.get('inchikey') or f"CFMID_{i}"
        c['idx'] = i

    if os.path.exists(cached_msp) and not FORCE_REGENERATE:
        # 使用智能解析计算已完成的compound数（考虑多能量）
        msp_completed = parse_completed_from_msp(cached_msp)
        count = len(msp_completed)
        total_energy = sum(msp_completed.values()) if msp_completed else 0
        
        if count > 0:
            print(f"\n[CACHE HIT] 发现缓存，已完成 {count} 个化合物 ({total_energy} 个能量谱图)")
            
            # 检查是否所有化合物都已完成
            if count >= len(all_candidates):
                print(f"  所有 {count} 个化合物已完成，跳过预测")
                
                # 输出 MSP 路径文件（供总控脚本读取）
                msp_path_file = os.path.join(OUTPUT_DIR, "cfmid_msp_path.txt")
                with open(msp_path_file, 'w') as f:
                    f.write(cached_msp)

                print(f"\n耗时: {time.time() - start_time:.1f} 秒")
                return True
            else:
                print(f"  剩余 {len(all_candidates) - count} 个化合物待预测，将继续")
        else:
            print(f"\n[CACHE] 发现缓存但解析失败，将重新生成")
    
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
    
    # 智能续传：优先从JSON恢复，否则从MSP解析
    resume_ids = get_resume_completed_ids(
        all_candidates, cached_msp, str(progress_file), 
        resume=not FORCE_REGENERATE
    )
    
    # 将恢复的ID传入预测函数
    completed_ids = run_cfmid_prediction(
        all_candidates, ION_MODE, work_dir, cached_msp, cfmid_ver,
        workers=CFMID_WORKERS, progress_file=str(progress_file),
        resume=True,  # 始终启用续传
        initial_completed=resume_ids  # 传入已恢复的ID
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
