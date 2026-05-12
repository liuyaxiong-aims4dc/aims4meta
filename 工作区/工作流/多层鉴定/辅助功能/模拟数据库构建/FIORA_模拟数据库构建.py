#!/usr/bin/env python3
"""
L2: FIORA 模拟谱图预测（独立脚本，在 fiora 环境运行）

功能：
1. 加载候选化合物库中所有 SMILES
2. 应用 SMILES 修复（除盐、去电荷、括号修复等）
3. 以 FIORA 版本号为缓存标签，已有缓存则跳过
4. 无缓存 → 运行 FIORA 预测 → 保存 MSP（带 Source_tool: FIORA 标签）

使用方式：
    # 1. 修改下方"配置参数区"的参数
    # 2. 运行脚本
    python FIORA_模拟数据库构建.py

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

# ===== Conda 环境自动激活 =====
def _ensure_conda_env():
    """确保在 aims4meta conda 环境中运行，否则自动重启动"""
    required_env = "aims4meta"
    conda_python = os.path.expanduser(f"~/miniconda3/envs/{required_env}/bin/python")
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
ION_MODE = "POS"  # 离子模式: "POS" 或 "NEG"
CANDIDATE_LIBRARY = "/stor3/AIMS4Meta/数据库/模拟数据库/TCM/coconut_csv_lite-04-2026.csv"  # 候选化合物库CSV文件路径

# ----- FIORA 配置 -----
FIORA_DEVICE = "cuda:0"  # GPU设备: "cuda:0" 或 "cpu"

# ----- 输出配置 -----
OUTPUT_DIR = ""  # 输出目录（留空则使用候选库所在目录）

# ----- 缓存配置 -----
FORCE_REGENERATE = False

###############################################################################
# 参数验证与初始化
###############################################################################

# 验证必需参数
if not os.path.exists(CANDIDATE_LIBRARY):
    raise ValueError(f"错误：候选库文件不存在: {CANDIDATE_LIBRARY}")

# 如果未指定输出目录，使用候选库所在目录
if not OUTPUT_DIR:
    candidate_path = Path(CANDIDATE_LIBRARY)
    if candidate_path.is_file():
        OUTPUT_DIR = str(candidate_path.parent)
    else:
        OUTPUT_DIR = CANDIDATE_LIBRARY

# FIORA 二进制路径
FIORA_BIN = '/home/lyx/miniconda3/envs/fiora/bin/fiora-predict'

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


def parse_completed_from_msp(msp_file):
    """从MSP文件解析已完成的compound，返回{pred_id: energy_count}
    
    使用INCHIKEY作为唯一标识，因为NAME可能为'nan'
    """
    completed = {}
    if not os.path.exists(msp_file) or os.path.getsize(msp_file) == 0:
        return completed
    
    current_inchikey = None
    energy_count = 0
    
    try:
        with open(msp_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line.startswith('INCHIKEY:'):
                    if current_inchikey and energy_count > 0:
                        completed[current_inchikey] = energy_count
                    current_inchikey = line[9:].strip()
                    energy_count = 0
                elif line.startswith('COLLISIONENERGY:'):
                    energy_count += 1
        
        if current_inchikey and energy_count > 0:
            completed[current_inchikey] = energy_count
    except Exception as e:
        print(f"  警告: MSP解析失败 ({e})")
    
    return completed


def get_resume_completed_ids(candidates, msp_file, progress_file, mgf_file=None, resume=True):
    """获取已完成的compound ID集合（智能恢复）
    
    优先级：
    1. JSON进度文件存在且有效 → 使用JSON
    2. MSP文件存在 → 从MSP解析
    3. MGF文件存在 → 从MGF解析（兼容旧版）
    """
    # 方法1：尝试从JSON进度文件恢复
    if resume and progress_file and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                completed_ids = set(progress_data.get('completed', []))
            if completed_ids:
                print(f"  续传[JSON]: 发现 {len(completed_ids)} 个已完成化合物")
                return completed_ids
        except Exception:
            pass
    
    # 方法2：从MSP文件解析
    msp_completed = parse_completed_from_msp(msp_file)
    if msp_completed:
        print(f"  续传[MSP]: 发现 {len(msp_completed)} 个化合物 ({sum(msp_completed.values())} 个能量)")
        return set(msp_completed.keys())
    
    # 方法3：从MGF文件解析（兼容旧版）
    if mgf_file:
        mgf_completed = parse_existing_predictions(mgf_file)
        if mgf_completed:
            print(f"  续传[MGF]: 发现 {len(mgf_completed)} 个已完成化合物")
            return mgf_completed
    
    return set()


def merge_energy_spectra(spectra_list):
    """合并多个能量的谱图，相同m/z累加强度，然后归一化
    
    参数:
        spectra_list: [[(mz, intensity), ...], ...] 多个能量的谱图
    返回:
        (mz_list, intensity_list) 合并并归一化后的谱图
    """
    if not spectra_list:
        return [], []
    
    # 收集所有峰（相同m/z累加强度）
    peak_dict = {}
    for peaks in spectra_list:
        for mz, intensity in peaks:
            mz_key = round(mz, 4)
            if mz_key not in peak_dict:
                peak_dict[mz_key] = 0.0
            peak_dict[mz_key] += intensity
    
    if not peak_dict:
        return [], []
    
    # 归一化：以最强峰为100
    max_intensity = max(peak_dict.values())
    if max_intensity > 0:
        for mz_key in peak_dict:
            peak_dict[mz_key] = (peak_dict[mz_key] / max_intensity) * 100.0
    
    # 排序并返回
    sorted_peaks = sorted(peak_dict.items(), key=lambda x: x[0])
    mz_out = [p[0] for p in sorted_peaks]
    int_out = [p[1] for p in sorted_peaks]
    return mz_out, int_out


def format_spectrum_msp(cand, peaks, fiora_ver, ion_mode, collision_energy="10eV/20eV/40eV"):
    """格式化单个谱图为 MSP 文本
    
    参数:
        cand: 候选化合物信息字典
        peaks: [(mz, intensity), ...] 峰列表
        fiora_ver: FIORA 版本号
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
    lines.append(f"INSTRUMENT: FIORA")
    lines.append(f"COLLISIONENERGY: {collision_energy}")
    lines.append(f"Comment: Source_tool=FIORA; Source_tool_version=FIORA_v{fiora_ver}; Source_database={source_val}")
    lines.append(f"Num Peaks: {len(peaks)}")
    
    for mz, intensity in peaks:
                lines.append(f"{mz:.6f}\t{intensity:.8f}")
    
    return "\n".join(lines)


def run_fiora_prediction(candidates, precursor_type, work_dir, output_msp, fiora_ver, ion_mode, device="cuda:0", resume=True, progress_file=None, batch_size=10, initial_completed=None):
    """运行 FIORA 预测（多能量 + 分批调用 + 实时写入 + tqdm 进度条 + 断点续传）
    
    对每个化合物预测 3 个能量（10, 20, 40），每个能量输出独立谱图条目。
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
        batch_size: 每批处理的化合物数量（默认10，复杂SMILES建议更小）
        initial_completed: 初始已完成的compound ID集合（从MSP或JSON恢复）
    """
    # 创建 FIORA 输出子目录（与 CFM-ID 统一）
    fiora_out_dir = work_dir / 'fiora_output'
    fiora_out_dir.mkdir(exist_ok=True)
    
    input_csv = fiora_out_dir / 'fiora_input.csv'
    output_mgf = fiora_out_dir / 'fiora_output.mgf'  # 单能量临时输出
    
    # 多能量配置
    collision_energies = [10, 20, 40]  # eV
    
    # 智能续传：优先从JSON恢复，否则从MSP/MGF解析
    completed_ids = set() if initial_completed is None else set(initial_completed)
    
    if not completed_ids and resume:
        # 尝试从JSON进度文件恢复
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
        
        # 如果JSON恢复失败，尝试从MSP解析
        if not completed_ids:
            msp_completed = parse_completed_from_msp(str(output_msp))
            if msp_completed:
                completed_ids = set(msp_completed.keys())
                print(f"  续传[MSP]: 发现 {len(completed_ids)} 个化合物")
    
    # 过滤掉已完成的候选
    pending_candidates = [c for c in candidates if c['pred_id'] not in completed_ids]
    
    if not pending_candidates:
        print(f"  所有候选已完成预测！")
        return completed_ids
    
    print(f"  FIORA 预测: {len(pending_candidates)} 个待处理 / {len(candidates)} 总计...")
    print(f"  多能量: {collision_energies} (eV)")
    print(f"  输出: {output_msp}")

    # 动态分批：根据分子量自适应调整batch_size和timeout
    # 分子量越大，batch_size越小，timeout越长
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    for c in pending_candidates:
        try:
            mol = Chem.MolFromSmiles(c['smiles'])
            if mol:
                c['mol_weight'] = Descriptors.MolWt(mol)
            else:
                c['mol_weight'] = 0  # 无效SMILES，放到最后处理
        except:
            c['mol_weight'] = 0

    # 按分子量排序（从小到大）
    pending_candidates.sort(key=lambda x: x['mol_weight'])

    # 动态分批策略：根据分子量决定batch_size和timeout
    # 先按档位分组
    def get_tier(mw):
        if mw == 0:
            return 0  # 无效SMILES
        elif mw < 1000:
            return 1  # 小分子
        elif mw < 1200:
            return 2  # 中等分子
        else:
            return 3  # 大分子 (1200-1500, >1500已在SMILES修复中过滤)

    # 按档位分组
    tier_groups = {0: [], 1: [], 2: [], 3: []}
    for c in pending_candidates:
        tier = get_tier(c['mol_weight'])
        tier_groups[tier].append(c)

    # 为每个档位生成批次
    batches = []
    tier_configs = {
        0: (1, 60),    # 无效SMILES: batch=1, timeout=60s
        1: (50, 120),  # 小分子 <1000: batch=50, timeout=120s
        2: (5, 180),   # 中等分子 1000-1200: batch=5, timeout=180s
        3: (3, 300),   # 大分子 1200-1500: batch=3, timeout=300s
    }

    # 从小到大处理，把难的放后面，无效SMILES放最后
    for tier in [1, 2, 3, 0]:
        candidates = tier_groups[tier]
        if not candidates:
            continue

        batch_size, timeout = tier_configs[tier]

        # 分批
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            batches.append((batch, timeout))

    # 统计各档位的批次数和分子量范围
    batch_stats = {}
    for batch, timeout in batches:
        batch_size = len(batch)
        # 计算该批次的分子量范围
        mw_values = [c['mol_weight'] for c in batch]
        mw_min = min(mw_values)
        mw_max = max(mw_values)

        # 根据分子量范围确定档位标签
        if mw_max == 0:
            mw_range = "无效SMILES"
        elif mw_max < 1000:
            mw_range = "<1000 Da (小分子)"
        elif mw_max < 1200:
            mw_range = "1000-1200 Da (中等分子)"
        else:
            mw_range = "1200-1500 Da (大分子)"

        key = (mw_range, batch_size, timeout)
        if key not in batch_stats:
            batch_stats[key] = {'count': 0, 'mw_min': mw_min, 'mw_max': mw_max}
        batch_stats[key]['count'] += 1
        batch_stats[key]['mw_min'] = min(batch_stats[key]['mw_min'], mw_min)
        batch_stats[key]['mw_max'] = max(batch_stats[key]['mw_max'], mw_max)

    print(f"  分 {len(batches)} 批处理 (动态分批策略，从小到大):")
    # 按batch_size从大到小排序（即从小分子到大分子）
    for key in sorted(batch_stats.keys(), key=lambda x: x[1], reverse=True):
        mw_range, batch_size, timeout = key
        stats = batch_stats[key]
        mw_detail = f"{stats['mw_min']:.1f}-{stats['mw_max']:.1f} Da"
        print(f"    {mw_range:30s} (batch={batch_size:3d}, timeout={timeout:3d}s, 实际{mw_detail}): {stats['count']:4d} 批")
    
    success_count = 0
    fail_count = 0
    
    # 打开输出文件（追加模式，实现实时写入）
    # 首次创建时写入 BOM（兼容 WPS/Excel），后续追加用普通 UTF-8
    if not os.path.exists(output_msp) or os.path.getsize(output_msp) == 0:
        with open(output_msp, 'w', encoding='utf-8-sig') as f:
            pass  # 仅写入 BOM 头
    with open(output_msp, 'a', encoding='utf-8') as f_out:
        for batch_idx, (batch, timeout) in enumerate(tqdm(batches, desc="  FIORA", ncols=100, dynamic_ncols=True, leave=True)):
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

                # 强度阈值0.001
                cmd = [FIORA_BIN, '-i', str(input_csv), '-o', str(output_mgf),
                       '--dev', device, '--min_prob', '0.001']

                max_retries = 0  # 不重试
                
                for retry in range(max_retries + 1):
                    try:
                        # 使用字节模式避免文本编码问题
                        proc = subprocess.Popen(
                            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                        )
                        try:
                            returncode = proc.wait(timeout=timeout)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                            if retry < max_retries:
                                tqdm.write(f"  批次 {batch_idx+1} CE={ce} 超时，重试 ({retry+1}/{max_retries})...")
                                continue
                            else:
                                tqdm.write(f"  批次 {batch_idx+1} CE={ce} 超时，放弃")
                                break
                        
                        if returncode != 0:
                            # 读取stderr（字节模式）
                            stderr_bytes = proc.stderr.read()
                            stderr_msg = stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else ''
                            if retry < max_retries:
                                tqdm.write(f"  批次 {batch_idx+1} CE={ce} 失败 (rc={returncode})，重试 ({retry+1}/{max_retries})...")
                                continue
                            else:
                                tqdm.write(f"  批次 {batch_idx+1} CE={ce} 失败 (rc={returncode})")
                                if stderr_msg:
                                    # 显示完整stderr（最多2000字符）
                                    tqdm.write(f"    stderr: {stderr_msg[:2000]}")
                                break
                        else:
                            # 成功，跳出重试循环
                            break
                    except Exception as e:
                        stderr_bytes = proc.stderr.read() if proc.stderr else b''
                        stderr_msg = stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else ''
                        if retry < max_retries:
                            tqdm.write(f"  批次 {batch_idx+1} CE={ce} 异常，重试 ({retry+1}/{max_retries})...")
                            continue
                        else:
                            tqdm.write(f"  批次 {batch_idx+1} CE={ce} 异常: {e}")
                            if stderr_msg:
                                tqdm.write(f"    stderr: {stderr_msg[:2000]}")
                            break
                else:
                    # 所有重试都失败，跳过本能量
                    continue
                
                # 解析本能量输出
                energy_predictions = parse_fiora_output(output_mgf)
                
                # 收集各能量的谱图
                for pred_id, peaks in energy_predictions.items():
                    if pred_id not in batch_predictions:
                        batch_predictions[pred_id] = []
                    batch_predictions[pred_id].append(peaks)
            
            # 为每个能量生成独立的谱图条目并实时写入
            cand_map = {c['pred_id']: c for c in batch}
            energy_labels = {10: '10eV', 20: '20eV', 40: '40eV'}
            
            for pred_id, spectra_list in batch_predictions.items():
                if spectra_list and len(spectra_list) == len(collision_energies):
                    cand = cand_map.get(pred_id, {})
                    has_valid_spectrum = False
                    
                    # 为每个能量生成独立的谱图
                    for i, ce in enumerate(collision_energies):
                        peaks = spectra_list[i]
                        if peaks:
                            # 对每个能量的谱图进行归一化
                            # 计算最大强度
                            max_intensity = max(intensity for _, intensity in peaks)
                            if max_intensity > 0:
                                # 归一化到100
                                normalized_peaks = [(mz, (intensity / max_intensity) * 100.0) for mz, intensity in peaks]
                            else:
                                normalized_peaks = peaks
                                
                            ce_label = energy_labels.get(ce, f"{ce}eV")
                            spectrum_text = format_spectrum_msp(cand, normalized_peaks, fiora_ver, ion_mode, collision_energy=ce_label)
                            if spectrum_text:
                                f_out.write(spectrum_text)
                                f_out.write("\n\n")  # MSP 谱图间空行分隔
                                has_valid_spectrum = True
                    
                    if has_valid_spectrum:
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
            f.write(f"INSTRUMENT: FIORA\n")
            f.write(f"COLLISIONENERGY: 10eV/20eV/40eV\n")  # 多能量合并
            f.write(f"Comment: Source_tool=FIORA; Source_tool_version=FIORA_v{fiora_ver}; Source_database={source}\n")
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

    # 提取CSV文件名（不含扩展名）
    csv_basename = candidate_lib_path.stem if candidate_lib_path.is_file() else "library"

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 版本标签
    print("\n[1/3] 获取 FIORA 版本...")
    fiora_ver = get_fiora_version()
    version_tag = f"fiora_v{fiora_ver}"
    print(f"  版本: {version_tag}")

    # 2. 加载候选化合物（提前加载以确定期望数量）
    print("\n[2/3] 加载候选化合物（含 SMILES 修复）...")
    precursor_type = '[M+H]+' if ION_MODE == 'POS' else '[M-H]-'
    all_candidates = load_all_candidates(CANDIDATE_LIBRARY, precursor_type)
    print(f"  有效候选: {len(all_candidates)} 个")
    
    # 计算期望的谱图数量（每个化合物3个能量）
    expected_count = len(all_candidates) * 3
    
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
    
    # 3. 检查缓存（文件名包含CSV名称）
    cached_msp = os.path.join(candidate_lib_dir, f"{csv_basename}_simulated_{version_tag}_{ION_MODE}.msp")
    print(f"  缓存路径: {cached_msp}")
    print(f"  期望谱图: {expected_count} 条")
    
    if os.path.exists(cached_msp) and not FORCE_REGENERATE:
        with open(cached_msp, 'r', encoding='utf-8', errors='ignore') as f:
            count = sum(1 for line in f if line.strip().lower().startswith('name:'))
        
        # 检查缓存完整性：只有谱图数>=期望数才认为缓存完整
        if count >= expected_count:
            print(f"\n[CACHE HIT] 缓存完整，跳过预测 ({count} 条谱图)")
            # 输出 MSP 路径文件（供总控脚本读取）
            msp_path_file = os.path.join(OUTPUT_DIR, "fiora_msp_path.txt")
            with open(msp_path_file, 'w') as f:
                f.write(cached_msp)
            print(f"\n耗时: {time.time() - start_time:.1f} 秒")
            return True
        else:
            print(f"\n[CACHE PARTIAL] 缓存不完整 ({count}/{expected_count} 条)，启用续传...")
    else:
        print(f"\n[NEW] 无缓存或强制重新生成，开始预测...")

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
    
    # 智能续传：优先从JSON恢复，否则从MSP解析
    fiora_mgf_file = fiora_out_dir / 'fiora_output.mgf'
    resume_ids = get_resume_completed_ids(
        all_candidates, str(cached_msp), str(progress_file), 
        mgf_file=str(fiora_mgf_file),
        resume=not FORCE_REGENERATE
    )
    
    # 运行预测（实时写入 MSP）
    completed_set = run_fiora_prediction(
        all_candidates, precursor_type, work_dir, cached_msp, fiora_ver, ION_MODE,
        FIORA_DEVICE, resume=True, progress_file=str(progress_file),
        batch_size=1, initial_completed=resume_ids
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
