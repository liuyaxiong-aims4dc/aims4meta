#!/usr/bin/env python3
"""
L3_SIRIUS CLI模式鉴定脚本

基于SIRIUS命令行工具进行分子式预测和结构鉴定
相比REST模式更简单、更稳定，与GUI行为一致

用法:
    python L3_SIRIUS_CLI.py \
        --sample_msp /path/to/input.msp \
        --output_dir /path/to/output \
        --instrument orbitrap \
        --ion_mode POS
"""

import os
import sys
import re
import signal
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd

# SIRIUS配置
DEFAULT_SIRIUS_BIN = "/stor3/AIMS4Meta/源代码/SIRIUS/sirius-6.3.5-linux-x64/sirius/bin/sirius"
SIRIUS_USERNAME = os.environ.get('L3_SIRIUS_USERNAME', "fanhl@whut.edu.cn")
SIRIUS_PASSWORD = os.environ.get('L3_SIRIUS_PASSWORD', "Kongtong@518936")

_sirius_process = None

def _signal_handler(sig, frame):
    global _sirius_process
    if _sirius_process and _sirius_process.poll() is None:
        print("\n  [中止] 正在终止SIRIUS进程...")
        _sirius_process.terminate()
        try:
            _sirius_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _sirius_process.kill()
    sys.exit(1)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# 默认参数
DEFAULT_INSTRUMENT = 'qtof'  # qtof 或 orbitrap
DEFAULT_ION_MODE = 'POS'
DEFAULT_MZ_THRESHOLD = 1500
DEFAULT_DATABASES = []  # 默认不使用在线库，需由总控脚本显式传入

# SIRIUS本地自定义数据库（custom-db）
# 所有本地库通过 --databases 参数传入，脚本自动检查注册状态
# 路径：/stor3/AIMS4Meta/数据库/模拟数据库/sirius/

# 所有已知本地数据库的路径映射（用于注册检查）
KNOWN_LOCAL_DBS = {
    'tcmbank_herb':       '/stor3/AIMS4Meta/数据库/模拟数据库/sirius/tcmbank_herb.siriusdb',
    'coconut_0425':       '/stor3/AIMS4Meta/数据库/模拟数据库/sirius/coconut_0425.siriusdb',
    'drugbank':           '/stor3/AIMS4Meta/数据库/模拟数据库/sirius/drugbank.siriusdb',
    'fda_approved':       '/stor3/AIMS4Meta/数据库/模拟数据库/sirius/fda_approved.siriusdb',
    'pfas':               '/stor3/AIMS4Meta/数据库/模拟数据库/sirius/pfas.siriusdb',
    'mycotoxin':          '/stor3/AIMS4Meta/数据库/模拟数据库/sirius/mycotoxin.siriusdb',
    'public_spectra_2506': '/stor3/AIMS4Meta/数据库/模拟数据库/sirius/public_spectra_2506.siriusdb',
}


def _load_intensity_dict(csv_path: str):
    """从 Progenesis QI 导出 CSV 读取 {(rt, mz): Maximum Abundance}
    - CSV 前2行为元信息（skiprows=2）
    - Compound 列格式例：'1.13_448.1722n'
    - Maximum Abundance 列为所有样品的最大丰度
    任一环节缺失则返回 None（调用方需返回 None 即停用丰度过滤）
    """
    if not csv_path or not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, skiprows=2)
    except Exception as e:
        print(f"  警告: 读取 CSV 失败（丰度过滤停用）: {e}")
        return None
    if 'Compound' not in df.columns or 'Maximum Abundance' not in df.columns:
        print(f"  警告: CSV 缺少 'Compound' 或 'Maximum Abundance' 列，丰度过滤停用")
        return None
    d = {}
    for _, row in df.iterrows():
        cid = str(row['Compound'])
        m = re.match(r'([\d.]+)_([\d.]+)[mn]', cid)
        if not m:
            continue
        try:
            rt = round(float(m.group(1)), 2)
            mz = round(float(m.group(2)), 4)
        except ValueError:
            continue
        ab = pd.to_numeric(row['Maximum Abundance'], errors='coerce')
        if pd.notna(ab):
            d[(rt, mz)] = float(ab)
    return d


def preprocess_msp(input_file: str, output_file: str, mz_threshold: float, ion_mode: str = 'POS', min_peaks: int = 6,
                   intensity_dict=None, min_intensity: float = 0) -> tuple:
    """预处理MSP：过滤多电荷/超质量/低质量谱图，补全Precursor_type
    可选：根据配套 CSV 的 Maximum Abundance 过滤信号太弱的谱图
    """
    default_adduct = "[M+H]+" if ion_mode == 'POS' else "[M-H]-"
    stats = {'total': 0, 'kept': 0, 'skipped_mz': 0, 'skipped_charge': 0,
             'skipped_peaks': 0, 'skipped_intensity': 0}
    use_intensity_filter = (intensity_dict is not None and min_intensity > 0)

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        lines, ptype_idx, charge_idx, charge_val, pmz, pmz_idx = [], -1, -1, None, None, -1
        num_peaks = 0
        name_val = None

        def process():
            nonlocal stats
            stats['total'] += 1
            if pmz is None or pmz >= mz_threshold:
                stats['skipped_mz'] += 1
                return
            if charge_val in ["2+", "3+", "4+", "5+", "6+", "7+", "2-", "3-", "4-", "5-", "6-", "7-"]:
                stats['skipped_charge'] += 1
                return
            if num_peaks < min_peaks:
                stats['skipped_peaks'] += 1
                return
            # 信号强度过滤（基于配套 CSV 的 Maximum Abundance）
            if use_intensity_filter and name_val:
                m = re.search(r'\(([\d.]+)_([\d.]+)[mn]/z\)', name_val)
                if m:
                    try:
                        key = (round(float(m.group(1)), 2), round(float(m.group(2)), 4))
                        max_ab = intensity_dict.get(key)
                        if max_ab is not None and max_ab < min_intensity:
                            stats['skipped_intensity'] += 1
                            return
                    except ValueError:
                        pass
            if ptype_idx < 0:
                insert = pmz_idx + 1 if pmz_idx >= 0 else len(lines)
                lines.insert(insert, f"Precursor_type: {default_adduct}")
            for l in lines:
                outfile.write(l + "\n")
            stats['kept'] += 1

        for line in infile:
            s = line.rstrip()
            if not s:
                if lines:
                    lines.append("")
                    process()
                lines, ptype_idx, charge_idx, charge_val, pmz, pmz_idx = [], -1, -1, None, None, -1
                num_peaks = 0
                name_val = None
                continue
            lines.append(s)
            u = s.upper()
            if u.startswith('NAME:'):
                name_val = s.split(':', 1)[1].strip()
            if u.startswith('PRECURSORTYPE:') or u.startswith('ADDUCT:') or u.startswith('PRECURSOR_TYPE:'):
                ptype_idx = len(lines) - 1
            m = re.search(r'^Charge\s*:\s*([0-9]+)([-+])', s, re.I)
            if m:
                charge_idx = len(lines) - 1
                charge_val = f"{m.group(1)}{m.group(2)}"
            m = re.search(r'^PRECURSORMZ\s*:\s*([0-9.]+)', s, re.I)
            if m:
                pmz_idx = len(lines) - 1
                try:
                    pmz = float(m.group(1))
                except ValueError:
                    pass
            # 统计碎片峰数量
            if re.match(r'^\d+\.?\d*\s+\d+', s):
                num_peaks += 1

        if lines:
            process()

    print(f"  预处理: {stats['total']} -> {stats['kept']} 条")
    filter_msg = f"m/z>{mz_threshold}: {stats['skipped_mz']}, 多电荷: {stats['skipped_charge']}, 碎片<{min_peaks}: {stats['skipped_peaks']}"
    if use_intensity_filter:
        filter_msg += f", 丰度<{int(min_intensity)}: {stats['skipped_intensity']}"
    print(f"    过滤: {filter_msg}")
    return output_file, stats['kept'], stats


def msp_to_mgf(msp_file: str, mgf_file: str) -> int:
    """将MSP转换为MGF格式"""
    spectra = []
    current_spectrum = {}

    with open(msp_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Name:'):
                if current_spectrum:
                    spectra.append(current_spectrum)
                current_spectrum = {'name': line.split(':', 1)[1].strip(), 'peaks': []}
            elif line.startswith('PrecursorMZ:') or line.startswith('PRECURSORMZ:'):
                current_spectrum['precursor_mz'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('Precursor_type:') or line.startswith('PRECURSORTYPE:') or line.startswith('Adduct:'):
                current_spectrum['adduct'] = line.split(':', 1)[1].strip()
            elif re.match(r'^\d+\.?\d*\s+\d+', line):
                parts = line.split()
                if len(parts) >= 2:
                    current_spectrum['peaks'].append((float(parts[0]), float(parts[1])))

        if current_spectrum:
            spectra.append(current_spectrum)

    # 写入MGF
    with open(mgf_file, 'w') as f:
        for spec in spectra:
            if 'precursor_mz' not in spec or not spec['peaks']:
                continue

            f.write("BEGIN IONS\n")
            f.write(f"TITLE={spec['name']}\n")
            f.write(f"PEPMASS={spec['precursor_mz']}\n")
            if 'adduct' in spec:
                f.write(f"CHARGE={spec['adduct']}\n")
            else:
                f.write("CHARGE=1+\n")

            for mz, intensity in spec['peaks']:
                f.write(f"{mz} {intensity}\n")

            f.write("END IONS\n\n")

    print(f"  MGF转换: {len(spectra)} 个谱图")
    return len(spectra)


def check_login(sirius_bin: str) -> bool:
    """检查SIRIUS是否已登录且token有效"""
    try:
        result = subprocess.run(
            [sirius_bin, 'login', '--show'],
            capture_output=True, text=True, timeout=60
        )
        output = result.stdout + result.stderr
        # 已登录: 输出包含 "Logged in as:"
        if 'Logged in as:' in output:
            return True
        return False
    except Exception:
        return False


def login_sirius(sirius_bin: str) -> bool:
    """登录SIRIUS（先检查是否已登录，如已登录则跳过）"""
    # 先检查是否已有有效登录
    if check_login(sirius_bin):
        print("  已登录，跳过重复登录")
        return True

    print(f"  登录SIRIUS账号: {SIRIUS_USERNAME}")
    try:
        env = os.environ.copy()
        env['SIRIUS_USERNAME'] = SIRIUS_USERNAME
        env['SIRIUS_PASSWORD'] = SIRIUS_PASSWORD

        result = subprocess.run(
            [sirius_bin, 'login', '--user-env', 'SIRIUS_USERNAME', '--password-env', 'SIRIUS_PASSWORD'],
            capture_output=True,
            text=True,
            timeout=120,
            env=env
        )
        if result.returncode == 0 or 'Login successful' in result.stdout or 'Academic' in result.stdout:
            print("  登录成功")
            return True
        else:
            print(f"  登录失败: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print("  登录超时，尝试检查是否已登录...")
        return check_login(sirius_bin)
    except Exception as e:
        print(f"  登录异常: {e}")
        return False


def ensure_local_dbs_registered(sirius_bin: str, databases: list) -> None:
    """确保所有本地自定义数据库已在SIRIUS中注册"""
    for db_name in databases:
        if db_name not in KNOWN_LOCAL_DBS:
            continue  # 在线库（BIO等）不需要注册
        db_path = KNOWN_LOCAL_DBS[db_name]
        if not os.path.exists(db_path):
            print(f"  警告: 本地数据库文件不存在: {db_path}")
            continue
        # 检查是否已注册
        try:
            result = subprocess.run(
                [sirius_bin, 'custom-db', 'show', '--db', db_name],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and db_name in result.stdout:
                continue  # 已注册
            # 未注册，尝试注册
            print(f"  注册本地数据库: {db_name} <- {db_path}")
            add_result = subprocess.run(
                [sirius_bin, 'custom-db', 'add', '--location', db_path],
                capture_output=True, text=True, timeout=60
            )
            if add_result.returncode == 0:
                print(f"  注册成功: {db_name}")
            else:
                print(f"  注册失败: {add_result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"  注册超时: {db_name}")
        except Exception as e:
            print(f"  注册异常: {db_name} - {e}")


def run_sirius_analysis(sirius_bin: str, input_file: str, output_dir: str,
                       instrument: str, databases: list, ion_mode: str = 'POS') -> bool:
    """运行SIRIUS分析（分步执行：formula → fingerprint → canopus → structure）"""
    profile = 'orbitrap' if instrument == 'orbitrap' else 'qtof'
    db_str = ','.join(databases)

    # 确保本地库已注册
    ensure_local_dbs_registered(sirius_bin, databases)

    sirius_project = os.path.join(output_dir, 'sirius_project.sirius')

    # 续传检测：若项目文件已存在，SIRIUS 会自动跳过已完成的 compound/step
    if os.path.exists(sirius_project):
        size_mb = os.path.getsize(sirius_project) / (1024 * 1024)
        print(f"  [续传] 检测到 sirius_project.sirius 已存在（{size_mb:.1f} MB），将在其基础上续跑，跳过已完成的 compound")
    else:
        print(f"  [新建] 未检测到历史项目文件，将从头运行")

    # 获取CPU核心数配置（默认使用80%核心）
    import multiprocessing
    max_cores = multiprocessing.cpu_count()
    cores = int(os.environ.get('L3_SIRIUS_CORES', max_cores * 0.8))

    print(f"  运行SIRIUS分析（分步执行）...")
    print(f"  仪器类型: {profile}")
    print(f"  数据库: {db_str}")
    print(f"  CPU核心数: {cores}/{max_cores}")

    # 步骤1: formula预测
    # 加合物控制：正离子允许 [M+H]+ 和 [M+Na]+（QI 对未知加合物留白，两种都常见，均需考虑），负离子只用 [M-H]-
    # 同时清空detectable列表，禁止SIRIUS自动扩展其他加合物（K/NH4/2H 等）
    # -c 10: 每个 instance 保留 Top-10 formula 候选（SIRIUS 官方默认）。曾收紧到 5 为加速下游，
    #     但实验发现缩敦后最优 formula 常落在 rank 6-10，导致置信度偏低 → 改回默认 10
    # compound-timeout 保持 SIRIUS 官方默认 300秒（不显式设置）
    #   历史教训：--compound-timeout 设为 15/60 都会导致大量 instance formula 超时跳过 → Computed 0
    # 注：-I 仅对未声明 PRECURSORTYPE 的 compound 做 fallback；MSP 已声明的加合物会被 SIRIUS 尊重保留
    enforced_adduct = '[M+H]+,[M+Na]+' if ion_mode == 'POS' else '[M-H]-'
    print(f"\n  [步骤1/4] Formula预测（强制加合物: {enforced_adduct}，禁止自动扩展）...")
    print(f"    formula候选数: Top-10（官方默认） | 单instance超时: SIRIUS默认(300秒)")
    cmd1 = [
        sirius_bin,
        '--cores', str(cores),
        '-i', input_file,
        '-o', sirius_project,
        'formula',
        '-c', '10',                       # formula候选数上限（官方默认）
        '-p', profile,
        '-I', enforced_adduct,    # 强制只用指定加合物
        '-i', ''                   # 清空可检测加合物列表，禁止自动扩展
    ]
    
    if not _run_sirius_step(cmd1, "Formula"):
        return False

    # 步骤2: fingerprint预测
    print(f"\n  [步骤2/4] Fingerprint预测...")
    cmd2 = [
        sirius_bin,
        '--cores', str(cores),
        '-i', sirius_project,
        '-o', sirius_project,
        'fingerprint'
    ]

    if not _run_sirius_step(cmd2, "Fingerprint"):
        return False

    # 步骤3: CANOPUS化合物分类（重要：提升结构预测准确性）
    print(f"\n  [步骤3/4] CANOPUS化合物分类...")
    cmd3 = [
        sirius_bin,
        '--cores', str(cores),
        '-i', sirius_project,
        '-o', sirius_project,
        'canopus'
    ]

    if not _run_sirius_step(cmd3, "CANOPUS"):
        print("  警告: CANOPUS失败，继续尝试structure搜索...")

    # 步骤4: structure搜索（需联网，token过期会静默返回空结果）
    print(f"\n  [步骤4/4] Structure搜索...")
    # structure步骤前验证登录状态，防止token过期导致静默失败
    if not check_login(sirius_bin):
        print("  Token已过期，重新登录...")
        if not login_sirius(sirius_bin):
            print("  警告: 重登录失败，structure步骤可能返回空结果")
    cmd4 = [
        sirius_bin,
        '--cores', str(cores),
        '-i', sirius_project,
        '-o', sirius_project,
        'structure',
        '--db', db_str
    ]

    return _run_sirius_step(cmd4, "Structure")


def _run_sirius_step(cmd: list, step_name: str) -> bool:
    """运行单个SIRIUS步骤"""
    print(f"  命令: {' '.join(cmd)}")
    print("  " + "=" * 60)

    try:
        global _sirius_process
        env = os.environ.copy()
        # JDK 25 默认用 posix_spawn 替代 fork，某些环境下 AOT 编译子进程会报 EACCES
        # 回退到传统 fork 机制消除 "权限不够" 警告（不影响功能，仅去噪）
        env['SIRIUS_OPTS'] = env.get('SIRIUS_OPTS', '') + ' -Djdk.lang.Process.launchMechanism=FORK'
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        _sirius_process = process

        # 逐行读取并显示输出
        import re as _re_out
        last_status = ""
        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue
            # 过滤索引重建日志
            if 'Re)building index' in line or 'initIndex' in line:
                continue
            # 过滤 Java 堆栈行（"at xxx.method(...)" 或 "... N more"），避免刷屏
            if _re_out.match(r'^\s*at\s', line) or _re_out.search(r'\.\.\. \d+ more\s*$', line):
                continue
            # 超时/取消警告：原位刷新（不换行）
            if 'TimeoutException' in line or 'CancellationException' in line or 'CANCELED' in line:
                m = _re_out.search(r'Unknown \([^)]+\)', line)
                if m:
                    status = f"  [超时跳过] {m.group(0)}"
                else:
                    # 无化合物上下文的纯网络异常（如 Structure 阶段 BIO 在线查询超时）
                    status = "  [网络超时] 正在重试..."
                sys.stdout.write(f"\r{status:<80}")
                sys.stdout.flush()
                last_status = status
            else:
                # 其他信息：正常换行显示
                if last_status:
                    sys.stdout.write("\n")
                    last_status = ""
                print(f"  {line}")

        process.wait(timeout=7200)  # 2小时超时

        if process.returncode == 0:
            print("  " + "=" * 60)
            print(f"  {step_name}完成")
            return True
        else:
            print("  " + "=" * 60)
            print(f"  {step_name}失败 (退出码: {process.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print("  " + "=" * 60)
        print(f"  {step_name}超时（2小时）")
        if process:
            process.kill()
        return False
    except Exception as e:
        print("  " + "=" * 60)
        print(f"  {step_name}异常: {e}")
        return False


def export_results(sirius_bin: str, sirius_project: str, output_dir: str) -> bool:
    """导出SIRIUS结果，验证是否真正生成了结构鉴定文件"""
    results_dir = os.path.join(output_dir, 'sirius_results')
    # 清空旧结果，避免残留误导
    if os.path.exists(results_dir):
        import shutil
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # SIRIUS 6 使用 'summaries' 子命令（非旧版 'write-summaries'）
    cmd = [
        sirius_bin,
        '--project', sirius_project,
        'summaries',
        '-o', results_dir
    ]

    print(f"  导出结果到: {results_dir}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"  结果导出失败: {result.stderr}")
            return False

        # 验证是否真正生成了结构鉴定结果文件
        struct_file = os.path.join(results_dir, 'structure_identifications.tsv')
        if not os.path.exists(struct_file):
            # SIRIUS 6 summaries 可能输出到子目录或不同文件名，尝试搜索
            import glob as _glob
            tsv_files = _glob.glob(os.path.join(results_dir, '**', '*structure*'), recursive=True)
            if tsv_files:
                struct_file = tsv_files[0]
                print(f"  找到结构鉴定文件: {struct_file}")
            else:
                result_files = os.listdir(results_dir) if os.path.exists(results_dir) else []
                print(f"  错误: summaries未生成structure_identifications.tsv")
                print(f"  导出目录文件: {result_files if result_files else '(空)'}")
                print(f"  可能原因: SIRIUS在线服务通信失败或未登录")
                return False

        print("  结果导出完成")
        return True

    except Exception as e:
        print(f"  结果导出异常: {e}")
        return False


def process_results(output_dir: str) -> tuple:
    """处理SIRIUS结果，生成L3标准输出"""
    results_dir = os.path.join(output_dir, 'sirius_results')
    struct_file = os.path.join(results_dir, 'structure_identifications.tsv')

    if not os.path.exists(struct_file):
        print(f"  警告: 未找到结构鉴定结果文件: {struct_file}")
        return 0, 0

    # 读取结构鉴定结果
    df = pd.read_csv(struct_file, sep='\t')
    total_compounds = len(df)

    # 过滤有效结果（有SMILES）
    df_hit = df[df['smiles'].notna() & (df['smiles'] != '')].copy()
    n_hits = len(df_hit)

    # 三级置信度策略：高置信全保留（多候选供选择），中/低仅保留最佳
    # 高：≥0.64 全保留 | 中：0.32-0.64 仅最佳 | 低：<0.32/-Inf 仅最佳
    has_conf_col = 'ConfidenceScoreExact' in df_hit.columns
    has_rank_col = 'structurePerIdRank' in df_hit.columns
    compound_col = 'mappingFeatureId' if 'mappingFeatureId' in df_hit.columns else (
        'alignedFeatureId' if 'alignedFeatureId' in df_hit.columns else None)

    def _safe_conf(val):
        """安全的置信度数值提取，-Infinity/NaN 返回 -inf"""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return float('-inf')
        try:
            return float(val)
        except (ValueError, TypeError):
            return float('-inf')

    if has_conf_col and compound_col:
        df_hit['_conf_num'] = df_hit['ConfidenceScoreExact'].apply(_safe_conf)
        kept_rows = []
        for _, group in df_hit.groupby(compound_col):
            has_high = (group['_conf_num'] >= 0.64).any()
            if has_high:
                kept_rows.append(group[group['_conf_num'] >= 0.64])
            else:
                if has_rank_col:
                    best = group[group['structurePerIdRank'] == group['structurePerIdRank'].min()].head(1)
                else:
                    best = group.loc[[group['_conf_num'].idxmax()]]
                if len(best) > 0:
                    kept_rows.append(best)
        df_output = pd.concat(kept_rows, ignore_index=True) if kept_rows else pd.DataFrame(columns=df_hit.columns)
        df_hit.drop(columns=['_conf_num'], inplace=True)
        if '_conf_num' in df_output.columns:
            df_output.drop(columns=['_conf_num'], inplace=True)
    else:
        df_output = df_hit
    n_output = len(df_output)

    if n_hits == 0:
        print("  警告: 结构鉴定结果中无有效SMILES，可能Zodiac/CSI:FingerID在线服务未成功")
        # 仍输出L3文件（仅有分子式和评分，无结构）- 统一列名格式
        df_l4 = pd.DataFrame({
            'query_name': df['mappingFeatureId'] if 'mappingFeatureId' in df.columns else (
                df['alignedFeatureId'] if 'alignedFeatureId' in df.columns else df.index),
            'matched_name': '',
            'matched_smiles': '',
            'matched_inchikey': '',
            'matched_formula': df['molecularFormula'] if 'molecularFormula' in df.columns else '',
            'precursor_formula': df['precursorFormula'] if 'precursorFormula' in df.columns else '',
            'cosine_score': '',
            'matched_peaks_ratio': '',
            'matched_fragments': '',
            'precursor_mz': df['ionMass'] if 'ionMass' in df.columns else '',
            'library_precursor_mz': '',
            'precursor_ppm_diff': '',
            'adduct': df['adduct'] if 'adduct' in df.columns else '',
            'matched_ontology': '',
            'source_method': 'SIRIUS',
            'source_database': 'CSI:FingerID',
            'rank': 1,  # 无结构时无多候选
            'isotope_similarity': '',
            'comprehensive_score': '',
            'csi_score': '',
            'confidence_exact': '',
            'confidence_approx': '',
            'zodiac_score': df['ZodiacScore'] if 'ZodiacScore' in df.columns else '',
            'sirius_score': df['SiriusScore'] if 'SiriusScore' in df.columns else '',
            'identification_source': 'L3_SIRIUS_no_structure',
            'structure_confidence': '无结构'
        })
        l3_file = os.path.join(output_dir, 'L3_identified.csv')
        df_l4.to_csv(l3_file, index=False, encoding='utf-8')
        print(f"  L3结果(无结构): {l3_file} ({total_compounds} 条，仅分子式)")
        return total_compounds, 0

    # 判断置信度等级（COSMIC论文 FDR 映射：≥0.64 → ~10%, ≥0.32 → ~20%）
    def _classify_confidence(val):
        if val is None or (isinstance(val, float) and (pd.isna(val) or val == float('-inf'))):
            return '低置信度'
        try:
            v = float(val)
        except (ValueError, TypeError):
            return '低置信度'
        if v >= 0.64:
            return '高置信度'
        elif v >= 0.32:
            return '中置信度'
        else:
            return '低置信度'

    df_l4 = pd.DataFrame({
        'query_name': df_output['mappingFeatureId'] if 'mappingFeatureId' in df_output.columns else (
            df_output['alignedFeatureId'] if 'alignedFeatureId' in df_output.columns else df_output.index),
        'matched_name': df_output['name'] if 'name' in df_output.columns else '',
        'matched_smiles': df_output['smiles'],
        'matched_inchikey': df_output['InChIkey2D'] if 'InChIkey2D' in df_output.columns else '',
        'matched_formula': df_output['molecularFormula'] if 'molecularFormula' in df_output.columns else '',
        'precursor_formula': df_output['precursorFormula'] if 'precursorFormula' in df_output.columns else '',  # SIRIUS 已融合了加合物原子的 precursor 化学式
        'cosine_score': '',  # L3无余弦相似度
        'matched_peaks_ratio': '',  # L3无碎片匹配比例
        'matched_fragments': '',  # L3无碎片详情
        'precursor_mz': df_output['ionMass'] if 'ionMass' in df_output.columns else '',
        'library_precursor_mz': '',  # 汇总脚本中由 precursor_formula 精确质量计算
        'precursor_ppm_diff': '',  # 汇总脚本中计算
        'adduct': df_output['adduct'] if 'adduct' in df_output.columns else '',
        'matched_ontology': '',  # L3无Ontology分类
        'source_method': 'SIRIUS',
        'source_database': 'CSI:FingerID',
        'rank': df_output['structurePerIdRank'] if 'structurePerIdRank' in df_output.columns else 1,
        'isotope_similarity': '',  # L3无同位素相似度
        'comprehensive_score': df_output['ConfidenceScoreExact'] if 'ConfidenceScoreExact' in df_output.columns else '',
        # L3特有列
        'csi_score': df_output['CSI:FingerIDScore'] if 'CSI:FingerIDScore' in df_output.columns else '',
        'confidence_exact': df_output['ConfidenceScoreExact'] if 'ConfidenceScoreExact' in df_output.columns else '',
        'confidence_approx': df_output['ConfidenceScoreApproximate'] if 'ConfidenceScoreApproximate' in df_output.columns else '',
        'zodiac_score': df_output['ZodiacScore'] if 'ZodiacScore' in df_output.columns else '',
        'sirius_score': df_output['SiriusScore'] if 'SiriusScore' in df_output.columns else '',
        'identification_source': 'L3_SIRIUS',
        'structure_confidence': df_output['ConfidenceScoreExact'].apply(_classify_confidence)
            if 'ConfidenceScoreExact' in df_output.columns else '低置信度'
    })

    l3_file = os.path.join(output_dir, 'L3_identified.csv')
    df_l4.to_csv(l3_file, index=False, encoding='utf-8')
    print(f"  L3结果(高置信度): {l3_file} ({n_output} 条，总结构命中 {n_hits}，过滤后 {n_output})")

    return total_compounds, n_hits


def write_summary(output_dir: str, args, n_compounds: int, n_hits: int, success: bool):
    """写入统计摘要"""
    summary_file = os.path.join(output_dir, 'L3_summary.txt')

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("L3 SIRIUS CLI模式鉴定统计\n")
        f.write("=" * 60 + "\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入: {args.sample_msp}\n")
        f.write(f"输出: {output_dir}\n")
        f.write(f"仪器: {args.instrument}\n")
        f.write(f"离子模式: {args.ion_mode}\n")
        f.write(f"数据库: {', '.join(args.databases)}\n\n")
        f.write(f"化合物数: {n_compounds}\n")
        f.write(f"结构命中: {n_hits}\n")
        if n_compounds > 0:
            f.write(f"命中率: {n_hits/n_compounds*100:.1f}%\n")
        f.write(f"\n分析状态: {'成功' if success else '失败'}\n")
        f.write("=" * 60 + "\n")

    print(f"  统计摘要: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='L3_SIRIUS CLI模式鉴定')
    parser.add_argument('--sample_msp', required=True, help='输入MSP文件')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--instrument', default=DEFAULT_INSTRUMENT,
                       choices=['orbitrap', 'qtof'], help=f'仪器类型 (默认: {DEFAULT_INSTRUMENT})')
    parser.add_argument('--ion_mode', default=DEFAULT_ION_MODE,
                       choices=['POS', 'NEG'], help=f'离子模式 (默认: {DEFAULT_ION_MODE})')
    parser.add_argument('--mz_threshold', type=float, default=DEFAULT_MZ_THRESHOLD,
                       help=f'm/z上限 (默认: {DEFAULT_MZ_THRESHOLD})')
    parser.add_argument('--min_peaks', type=int, default=6,
                       help='最小碎片峰数量 (默认: 6)')
    parser.add_argument('--sample_csv', default='',
                       help='配套 Progenesis QI CSV（用于 Maximum Abundance 信号过滤，空则不过滤）')
    parser.add_argument('--min_intensity', type=float, default=0,
                       help='信号强度阈值，化合物 Maximum Abundance 低于该值则跳过（0 表示停用，默认 0）')
    parser.add_argument('--databases', nargs='+', default=DEFAULT_DATABASES,
                       help=f'结构数据库列表，支持在线库(BIO等)和本地库(tcmbank_herb等) (默认: {" ".join(DEFAULT_DATABASES)})')
    parser.add_argument('--sirius_bin', default=DEFAULT_SIRIUS_BIN, help='SIRIUS路径')
    args = parser.parse_args()

    if not os.path.exists(args.sample_msp):
        print(f"错误: 文件不存在: {args.sample_msp}")
        return False
    if not os.path.exists(args.sirius_bin):
        print(f"错误: SIRIUS不存在: {args.sirius_bin}")
        return False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("L3_SIRIUS CLI模式鉴定")
    print(f"  输入: {args.sample_msp}")
    print(f"  输出: {output_dir}")
    print(f"  仪器: {args.instrument} | 离子模式: {args.ion_mode}")
    print(f"  数据库: {', '.join(args.databases)}")
    # 区分本地库和在线库
    local_dbs = [d for d in args.databases if d in KNOWN_LOCAL_DBS]
    online_dbs = [d for d in args.databases if d not in KNOWN_LOCAL_DBS]
    if local_dbs:
        print(f"  本地库: {', '.join(local_dbs)}")
    if online_dbs:
        print(f"  在线库: {', '.join(online_dbs)}")
    print(f"  质量筛选: 碎片≥{args.min_peaks}, m/z≤{args.mz_threshold}")
    print("=" * 70)

    # [1] MSP预处理
    print("\n[1/5] MSP预处理...")
    processed_msp = str(output_dir / 'L3_processed.msp')
    # 可选信号强度过滤：加载配套 CSV 的 (rt, mz) -> Maximum Abundance 映射
    intensity_dict = None
    if args.sample_csv and args.min_intensity > 0:
        intensity_dict = _load_intensity_dict(args.sample_csv)
        if intensity_dict is not None:
            print(f"  信号强度过滤已启用：阈值={int(args.min_intensity)}, 丰度寻查表={len(intensity_dict)} 条")
    processed_msp, n_compounds, _ = preprocess_msp(
        args.sample_msp, processed_msp, args.mz_threshold, args.ion_mode, args.min_peaks,
        intensity_dict=intensity_dict, min_intensity=args.min_intensity)

    if n_compounds == 0:
        print("  无化合物需要分析")
        write_summary(str(output_dir), args, 0, 0, True)
        return True

    # [2] 登录SIRIUS
    print("\n[2/5] 登录SIRIUS...")
    if not login_sirius(args.sirius_bin):
        print("错误: SIRIUS登录失败")
        write_summary(str(output_dir), args, n_compounds, 0, False)
        return False

    # [3] 运行SIRIUS分析（直接使用MSP）
    # 若项目已存在（续传），跳过全部 SIRIUS 计算，零增长
    # 注意：sirius_results/ 会被 clean 清理，不能通过它判断；仅靠 .sirius 文件大小
    sirius_project = os.path.join(str(output_dir), 'sirius_project.sirius')
    if os.path.exists(sirius_project) and os.path.getsize(sirius_project) > 10 * 1024 * 1024:
        size_mb = os.path.getsize(sirius_project) / (1024 * 1024)
        print(f"\n[3/5] SIRIUS分析（续传 - 项目已存在 {size_mb:.1f} MB，跳过全部计算）...")
        print(f"  [跳过] Formula/Fingerprint/CANOPUS/Structure 均已存在，直接复用")
    else:
        print("\n[3/5] 运行SIRIUS分析...")
        if not run_sirius_analysis(args.sirius_bin, processed_msp, str(output_dir),
                                   args.instrument, args.databases, args.ion_mode):
            print("错误: SIRIUS分析失败")
            write_summary(str(output_dir), args, n_compounds, 0, False)
            return False
            return False

    # [4] 导出结果（失败时重试structure步骤）
    max_export_retries = 2
    export_ok = False

    # summaries导出需要有效token，先验证登录
    if not check_login(args.sirius_bin):
        print("  Token已过期，重新登录...")
        login_sirius(args.sirius_bin)

    for attempt in range(max_export_retries + 1):
        print(f"\n[4/5] 导出结果{'(重试 ' + str(attempt) + '/' + str(max_export_retries) + ')' if attempt > 0 else ''}...")
        if export_results(args.sirius_bin, sirius_project, str(output_dir)):




            export_ok = True
            break

        if attempt < max_export_retries:
            # 重新登录（可能登录过期导致在线服务失败）
            print("  尝试重新登录SIRIUS...")
            if login_sirius(args.sirius_bin):
                # 仅重跑structure步骤（formula/fingerprint结果已在project中）
                print("  重跑structure步骤...")
                db_str = ','.join(args.databases)
                import multiprocessing
                max_cores = multiprocessing.cpu_count()
                cores = int(os.environ.get('L3_SIRIUS_CORES', max_cores * 0.8))
                retry_structure_cmd = [
                    args.sirius_bin,
                    '--cores', str(cores),
                    '-i', sirius_project,
                    '-o', sirius_project,
                    'structure',
                    '--db', db_str
                ]
                if _run_sirius_step(retry_structure_cmd, "Structure(重试)"):
                    continue  # 重试导出
            print("  重登录或structure重跑失败")

    if not export_ok:
        print("错误: 结构鉴定结果导出失败（已重试" + str(max_export_retries) + "次）")
        write_summary(str(output_dir), args, n_compounds, 0, False)
        return False

    # [5] 处理结果
    print("\n[5/5] 处理结果...")
    total, n_hits = process_results(str(output_dir))

    write_summary(str(output_dir), args, n_compounds, n_hits, True)

    print("\n" + "=" * 70)
    pct = n_hits / n_compounds * 100 if n_compounds > 0 else 0
    print(f"完成 | 化合物: {n_compounds} | 命中: {n_hits} ({pct:.1f}%)")
    print("=" * 70)

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
