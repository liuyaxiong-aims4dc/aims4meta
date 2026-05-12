#!/usr/bin/env python3
"""
SIRIUS 批量处理 MSP 文件生成碎片树
支持任意 MSP 格式质谱数据集

用法:
    # 处理单个 MSP 文件
    python run_sirius_batch.py --input /path/to/data.msp --output /path/to/output
    
    # 处理正负模式（指定两个文件）
    python run_sirius_batch.py --mode both \
        --input_pos /path/to/pos.msp \
        --input_neg /path/to/neg.msp \
        --output /path/to/output
"""
import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import shutil

# SIRIUS 路径
SIRIUS_PATH = "/stor1/AIMS4Meta/code/SIRIUS/sirius-6.3.3-linux-x64/sirius/bin/sirius"

# 默认数据路径（用于快速启动）
DEFAULT_POS = "/stor1/AIMS4Meta/databases/spectraverse/spectraverse-1.0.1-pos.msp"
DEFAULT_NEG = "/stor1/AIMS4Meta/databases/spectraverse/spectraverse-1.0.1-neg.msp"


def parse_args():
    parser = argparse.ArgumentParser(
        description='SIRIUS 批量处理 MSP 文件 (支持 24+ 并行进程)'
    )
    parser.add_argument(
        '--mode', type=str, default='single',
        choices=['single', 'pos', 'neg', 'both'],
        help='处理模式: single(单文件), pos(正离子), neg(负离子), both(两者)'
    )
    parser.add_argument(
        '--input', type=str,
        help='输入 MSP 文件路径 (用于 single 模式)'
    )
    parser.add_argument(
        '--input_pos', type=str, default=DEFAULT_POS,
        help=f'正离子 MSP 文件路径 (默认: {DEFAULT_POS})'
    )
    parser.add_argument(
        '--input_neg', type=str, default=DEFAULT_NEG,
        help=f'负离子 MSP 文件路径 (默认: {DEFAULT_NEG})'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='输出目录根路径'
    )
    parser.add_argument(
        '--num_workers', type=int, default=1,
        help='并行进程数 (默认: 1, SIRIUS 6.3+ 内部已用 36 线程, 多进程会导致工作区冲突静默失败)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=500,
        help='每批处理的谱图数 (默认: 500, 大批次可充分利用 SIRIUS 内部并行, 每批约 40 秒)'
    )
    parser.add_argument(
        '--profile', type=str, default='qtof',
        choices=['qtof', 'orbitrap', 'fticr'],
        help='仪器类型 (默认: qtof)'
    )
    parser.add_argument(
        '--candidates', type=int, default=1,
        help='每个谱图保留的候选分子式数量 (默认: 1)'
    )
    parser.add_argument(
        '--timeout', type=int, default=600,
        help='每个批次的超时时间(秒) (默认: 600, 即 10 分钟)'
    )
    parser.add_argument(
        '--keep_temp', action='store_true',
        help='保留临时文件 (用于调试)'
    )
    parser.add_argument(
        '--sirius_path', type=str, default=SIRIUS_PATH,
        help=f'SIRIUS 可执行文件路径 (默认: {SIRIUS_PATH})'
    )
    parser.add_argument(
        '--auto-convert', action='store_true',
        help='碎片树生成完成后自动转换为 subformulae JSON 格式'
    )
    parser.add_argument(
        '--no-auto-convert', action='store_false', dest='auto_convert',
        help='禁用自动 JSON 转换'
    )
    parser.set_defaults(auto_convert=True)
    parser.add_argument(
        '--json_output', type=str, default=None,
        help='JSON 输出目录 (默认: 输出目录/../subformulae/{pos|neg})'
    )
    parser.add_argument(
        '--json_port', type=int, default=8889,
        help='JSON 转换使用的 REST 端口 (默认: 8889)'
    )
    return parser.parse_args()


def check_sirius(sirius_path: str = SIRIUS_PATH):
    """检查 SIRIUS 是否可用"""
    if not os.path.exists(sirius_path):
        print(f"错误: SIRIUS 不存在于 {sirius_path}")
        print("请检查 SIRIUS 安装路径")
        return False
    
    # 测试 SIRIUS 版本
    try:
        result = subprocess.run(
            [sirius_path, '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # 提取版本信息 (SIRIUS 输出有很多日志，需要提取关键行)
            for line in result.stdout.split('\n'):
                if 'SIRIUS' in line and 'lib' not in line.lower():
                    print(f"SIRIUS 版本: {line.strip()}")
                    break
            return True
    except Exception as e:
        print(f"SIRIUS 检查失败: {e}")
    
    return False


def check_sirius_login(sirius_path: str = SIRIUS_PATH) -> bool:
    """
    检查 SIRIUS 是否已登录
    
    Returns:
        bool: 是否已登录
    """
    try:
        result = subprocess.run(
            [sirius_path, 'login', '--show'],
            capture_output=True,
            text=True,
            timeout=10
        )
        # 检查输出中是否包含登录成功信息
        if 'Active Subscription' in result.stdout or 'Academic License' in result.stdout:
            return True
        return False
    except Exception:
        return False


def login_sirius(sirius_path: str = SIRIUS_PATH) -> bool:
    """
    自动登录 SIRIUS
    
    使用项目中的登录凭证
    
    Returns:
        bool: 登录是否成功
    """
    # 账号密码（来自项目登录脚本）
    username = "fanhl@whut.edu.cn"
    password = "Kongtong@518936"
    
    print(f"正在登录 SIRIUS (账号: {username})...")
    
    try:
        # 设置环境变量
        env = os.environ.copy()
        env['SIRIUS_USERNAME'] = username
        env['SIRIUS_PASSWORD'] = password
        
        result = subprocess.run(
            [sirius_path, 'login', '--user-env', 'SIRIUS_USERNAME', '--password-env', 'SIRIUS_PASSWORD'],
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )
        
        if result.returncode == 0 and 'Login successful' in result.stdout:
            print("✓ SIRIUS 登录成功")
            return True
        else:
            print(f"✗ SIRIUS 登录失败")
            if 'error' in result.stderr.lower():
                print(f"错误: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"登录异常: {e}")
        return False


def check_input_files(mode: str, input_pos: str = DEFAULT_POS, input_neg: str = DEFAULT_NEG):
    """检查输入文件是否存在"""
    files_to_check = []
    
    if mode in ['pos', 'both']:
        files_to_check.append(('正离子', input_pos))
    if mode in ['neg', 'both']:
        files_to_check.append(('负离子', input_neg))
    
    for ion_mode, filepath in files_to_check:
        if not os.path.exists(filepath):
            print(f"错误: {ion_mode}数据文件不存在: {filepath}")
            return False
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✓ {ion_mode}数据: {filepath} ({size_mb:.1f} MB)")
    
    return True


def parse_spectrum_metadata(spectrum_lines):
    """
    从谱图行中提取元信息
    
    Returns:
        dict: 包含 name, formula, smiles, inchikey, ionization, precursor_mz 等
    """
    metadata = {
        'name': '',
        'formula': '',
        'smiles': '',
        'inchikey': '',
        'ionization': '',
        'precursor_mz': '',
        'collision_energy': ''
    }
    
    for line in spectrum_lines:
        line = line.strip()
        if line.startswith('Name:'):
            metadata['name'] = line.split(':', 1)[1].strip()
        elif line.startswith('Formula:'):
            metadata['formula'] = line.split(':', 1)[1].strip()
        elif line.startswith('SMILES:'):
            metadata['smiles'] = line.split(':', 1)[1].strip()
        elif line.startswith('InChIKey:'):
            metadata['inchikey'] = line.split(':', 1)[1].strip()
        elif line.startswith('PrecursorType:'):
            metadata['ionization'] = line.split(':', 1)[1].strip()
        elif line.startswith('PrecursorMZ:'):
            metadata['precursor_mz'] = line.split(':', 1)[1].strip()
        elif line.startswith('CollisionEnergy:'):
            metadata['collision_energy'] = line.split(':', 1)[1].strip()
    
    return metadata


def split_msp(msp_path: str, output_dir: str, batch_size: int = 100):
    """
    将 MSP 文件分割成多个小文件，并提取 labels.tsv
    
    Args:
        msp_path: MSP 文件路径
        output_dir: 输出目录
        batch_size: 每批的谱图数
    
    Returns:
        分割后的文件列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取 MSP 文件
    spectra = []
    current_spectrum = []
    all_metadata = []  # 收集所有元信息
    
    print(f"读取 {msp_path}...")
    with open(msp_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '':
                if current_spectrum:
                    spectra.append(current_spectrum)
                    # 提取元信息
                    metadata = parse_spectrum_metadata(current_spectrum)
                    all_metadata.append(metadata)
                    current_spectrum = []
            else:
                current_spectrum.append(line)
        
        if current_spectrum:
            spectra.append(current_spectrum)
            metadata = parse_spectrum_metadata(current_spectrum)
            all_metadata.append(metadata)
    
    print(f"总共 {len(spectra)} 个谱图")
    
    # 保存 labels.tsv
    labels_file = os.path.join(os.path.dirname(output_dir), 'labels.tsv')
    print(f"保存元信息到 {labels_file}...")
    
    import csv
    with open(labels_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'formula', 'smiles', 'inchikey', 
                                                'ionization', 'precursor_mz', 'collision_energy'],
                                delimiter='\t')
        writer.writeheader()
        for meta in all_metadata:
            writer.writerow(meta)
    
    print(f"✓ 已保存 {len(all_metadata)} 条元信息")
    
    # 分割
    batch_files = []
    for i in range(0, len(spectra), batch_size):
        batch = spectra[i:i + batch_size]
        batch_file = os.path.join(output_dir, f'batch_{i//batch_size:05d}.msp')
        
        with open(batch_file, 'w', encoding='utf-8') as f:
            for spectrum in batch:
                f.writelines(spectrum)
                f.write('\n')
        
        batch_files.append(batch_file)
    
    print(f"分割成 {len(batch_files)} 个批次")
    return batch_files


# 全局计数器，用于定期刷新登录
_batch_counter = [0]  # 使用列表以便在函数内修改
_LOGIN_REFRESH_INTERVAL = 20  # 每 20 批刷新一次登录


def run_sirius_batch(msp_file: str, output_dir: str, profile: str = 'qtof', 
                     candidates: int = 1, timeout: int = 600, sirius_path: str = SIRIUS_PATH):
    """
    对一个批次的谱图运行 SIRIUS
    
    Args:
        msp_file: MSP 文件路径
        output_dir: 输出目录
        profile: 仪器类型
        candidates: 候选分子式数量
        timeout: 超时时间(秒)
        sirius_path: SIRIUS 可执行文件路径
    
    Returns:
        (success: bool, batch_name: str)
    """
    global _batch_counter, _LOGIN_REFRESH_INTERVAL
    batch_name = Path(msp_file).stem
    # SIRIUS 6.3+ 输出为 .sirius 文件，不需要子目录
    batch_output = os.path.join(output_dir, f"{batch_name}.sirius")
    
    # 跳过已存在的 .sirius 文件
    if os.path.exists(batch_output):
        return True, batch_name
    
    # 定期刷新登录状态（每 N 批）
    _batch_counter[0] += 1
    if _batch_counter[0] % _LOGIN_REFRESH_INTERVAL == 1:
        login_sirius(sirius_path)
    
    # 构建 SIRIUS 命令
    cmd = [
        sirius_path,
        '--input', msp_file,
        '--output', batch_output,
        'formula',
        '--profile', profile,
        '--candidates', str(candidates),
        '--no-recalibration'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            error_msg = result.stderr[:200] if result.stderr else "未知错误"
            print(f"  [错误] {batch_name}: {error_msg}")
            return False, batch_name
        
        # 验证输出文件是否实际生成
        if not os.path.exists(batch_output):
            # 可能是登录过期，尝试重新登录并重试一次
            print(f"  [无输出] {batch_name}: 尝试重新登录...")
            if login_sirius(sirius_path):
                # 重试一次
                result2 = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                if result2.returncode == 0 and os.path.exists(batch_output):
                    print(f"  [重试成功] {batch_name}")
                    return True, batch_name
            print(f"  [失败] {batch_name}: SIRIUS 无法生成碎片树")
            return False, batch_name
        
        return True, batch_name
    
    except subprocess.TimeoutExpired:
        # 超时后检查文件是否已生成（SIRIUS 可能在超时前已完成写入）
        if os.path.exists(batch_output) and os.path.getsize(batch_output) > 0:
            print(f"  [超时但文件已生成] {batch_name}")
            return True, batch_name
        print(f"  [超时] {batch_name}")
        return False, batch_name
    except Exception as e:
        # 异常后也检查文件是否已生成
        if os.path.exists(batch_output) and os.path.getsize(batch_output) > 0:
            print(f"  [异常但文件已生成] {batch_name}")
            return True, batch_name
        print(f"  [异常] {batch_name}: {e}")
        return False, batch_name


def process_batch_wrapper(args):
    """处理单个批次 (用于并行)"""
    msp_file, output_dir, profile, candidates, timeout, sirius_path = args
    return run_sirius_batch(msp_file, output_dir, profile, candidates, timeout, sirius_path)


def process_mode(mode: str, output_root: str, args):
    """
    处理单个离子模式
    
    Args:
        mode: 'pos' 或 'neg'
        output_root: 输出根目录
        args: 命令行参数
    """
    # 确定输入文件
    if mode == 'pos':
        input_file = args.input_pos
        ion_mode_name = '正离子'
    else:
        input_file = args.input_neg
        ion_mode_name = '负离子'
    
    # 创建输出目录
    output_dir = os.path.join(output_root, 'fragmentation_trees')
    temp_dir = os.path.join(output_root, f'temp_batches_{mode}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 警告多进程风险
    if args.num_workers > 1:
        print(f"⚠ 警告: num_workers={args.num_workers}, SIRIUS 6.3+ 多进程并行会导致静默失败!")
        print(f"  SIRIUS 内部已使用 36 线程, 建议 num_workers=1, batch_size=500")
    
    print(f"\n{'='*60}")
    print(f"处理 {ion_mode_name} 模式")
    print(f"{'='*60}")
    print(f"输入: {input_file}")
    print(f"输出: {output_dir}")
    print(f"临时目录: {temp_dir}")
    print(f"并行进程: {args.num_workers}, 批次大小: {args.batch_size}")
    
    # 统计已有批次
    existing_count = len([f for f in os.listdir(output_dir) if f.endswith('.sirius')])
    if existing_count > 0:
        print(f"已有 {existing_count} 个 .sirius 文件, 将自动跳过")
    
    # 分割 MSP 文件
    print(f"\n[1/3] 分割 MSP 文件...")
    batch_files = split_msp(input_file, temp_dir, args.batch_size)
    
    # 运行 SIRIUS
    print(f"\n[2/3] 运行 SIRIUS (并行数: {args.num_workers})...")
    
    # 估算时间 (batch_size=500 时每批约 40 秒)
    est_batches = len(batch_files) - existing_count
    est_time_per_batch = 40 if args.batch_size >= 500 else max(4, args.batch_size * 0.08)
    est_time_hours = (est_batches * est_time_per_batch) / 3600
    print(f"总批次: {len(batch_files)}, 待处理: ~{est_batches}, 预计时间: ~{est_time_hours:.1f} 小时")
    
    process_args = [
        (f, output_dir, args.profile, args.candidates, args.timeout, args.sirius_path)
        for f in batch_files
    ]
    
    results = []
    with mp.Pool(args.num_workers) as pool:
        for result in tqdm(
            pool.imap(process_batch_wrapper, process_args),
            total=len(batch_files),
            desc=f"{ion_mode_name}处理进度"
        ):
            results.append(result)
    
    # 统计结果
    success_count = sum(1 for r, _ in results if r)
    failed_batches = [name for r, name in results if not r]
    
    print(f"\n[3/3] 处理完成!")
    print(f"  成功: {success_count}/{len(batch_files)} 批次")
    
    if failed_batches:
        print(f"  失败批次: {len(failed_batches)}")
        failed_log = os.path.join(output_root, f'failed_batches_{mode}.txt')
        with open(failed_log, 'w') as f:
            for name in failed_batches:
                f.write(f"{name}\n")
        print(f"  已保存到: {failed_log}")
    
    # 清理临时文件
    if not args.keep_temp:
        print(f"  清理临时文件...")
        shutil.rmtree(temp_dir)
    
    return success_count, len(batch_files)


def process_single_file(input_file: str, output_root: str, args):
    """
    处理单个 MSP 文件
    
    Args:
        input_file: 输入 MSP 文件路径
        output_root: 输出根目录
        args: 命令行参数
    """
    output_dir = os.path.join(output_root, 'fragmentation_trees')
    temp_dir = os.path.join(output_root, 'temp_batches')
    os.makedirs(output_dir, exist_ok=True)
    
    # 警告多进程风险
    if args.num_workers > 1:
        print(f"⚠ 警告: num_workers={args.num_workers}, SIRIUS 6.3+ 多进程并行会导致静默失败!")
        print(f"  SIRIUS 内部已使用 36 线程, 建议 num_workers=1, batch_size=500")
    
    print(f"\n{'='*60}")
    print(f"处理单个文件")
    print(f"{'='*60}")
    print(f"输入: {input_file}")
    print(f"输出: {output_dir}")
    print(f"并行进程: {args.num_workers}, 批次大小: {args.batch_size}")
    
    # 统计已有批次
    existing_count = len([f for f in os.listdir(output_dir) if f.endswith('.sirius')])
    if existing_count > 0:
        print(f"已有 {existing_count} 个 .sirius 文件, 将自动跳过")
    
    # 分割 MSP 文件
    print(f"\n[1/3] 分割 MSP 文件...")
    batch_files = split_msp(input_file, temp_dir, args.batch_size)
    
    # 运行 SIRIUS
    print(f"\n[2/3] 运行 SIRIUS (并行数: {args.num_workers})...")
    
    process_args = [
        (f, output_dir, args.profile, args.candidates, args.timeout, args.sirius_path)
        for f in batch_files
    ]
    
    results = []
    with mp.Pool(args.num_workers) as pool:
        for result in tqdm(
            pool.imap(process_batch_wrapper, process_args),
            total=len(batch_files),
            desc="处理进度"
        ):
            results.append(result)
    
    # 统计结果
    success_count = sum(1 for r, _ in results if r)
    failed_batches = [name for r, name in results if not r]
    
    print(f"\n[3/3] 处理完成!")
    print(f"  成功: {success_count}/{len(batch_files)} 批次")
    
    if failed_batches:
        print(f"  失败批次: {len(failed_batches)}")
        failed_log = os.path.join(output_root, 'failed_batches.txt')
        with open(failed_log, 'w') as f:
            for name in failed_batches:
                f.write(f"{name}\n")
        print(f"  已保存到: {failed_log}")
    
    # 清理临时文件
    if not args.keep_temp:
        print(f"  清理临时文件...")
        shutil.rmtree(temp_dir)
    
    return success_count, len(batch_files)


def main():
    args = parse_args()
    
    # 检查 SIRIUS
    print("检查 SIRIUS...")
    if not check_sirius(args.sirius_path):
        sys.exit(1)
    
    # 检查 SIRIUS 登录状态
    print("\n检查 SIRIUS 登录状态...")
    if not check_sirius_login(args.sirius_path):
        print("SIRIUS 未登录，尝试自动登录...")
        if not login_sirius(args.sirius_path):
            print("错误: SIRIUS 登录失败，无法继续")
            print("请检查账号密码或网络连接")
            sys.exit(1)
    else:
        print("✓ SIRIUS 已登录")
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 处理模式
    if args.mode == 'single':
        # 单文件模式
        if not args.input:
            print("错误: single 模式需要指定 --input 参数")
            sys.exit(1)
        if not os.path.exists(args.input):
            print(f"错误: 输入文件不存在: {args.input}")
            sys.exit(1)
        success, total = process_single_file(args.input, args.output, args)
        print(f"\n完成! 成功: {success}/{total} 批次")
        
    else:
        # 多模式处理
        print("\n检查输入文件...")
        if not check_input_files(args.mode, args.input_pos, args.input_neg):
            sys.exit(1)
        
        modes_to_process = []
        if args.mode in ['pos', 'both']:
            modes_to_process.append('pos')
        if args.mode in ['neg', 'both']:
            modes_to_process.append('neg')
        
        # 处理每个模式
        total_stats = {}
        for mode in modes_to_process:
            success, total = process_mode(mode, args.output, args)
            total_stats[mode] = {'success': success, 'total': total}
        
        # 最终统计
        print(f"\n{'='*60}")
        print("最终统计")
        print(f"{'='*60}")
        for mode, stats in total_stats.items():
            ion_name = '正离子' if mode == 'pos' else '负离子'
            print(f"{ion_name}: {stats['success']}/{stats['total']} 批次成功")
    
    print(f"\n碎片树输出目录: {args.output}")
    print("SIRIUS 处理完成!")
    
    # 自动转换为 JSON
    if args.auto_convert:
        print("\n" + "="*60)
        print("自动转换为 subformulae JSON 格式")
        print("="*60)
        auto_convert_to_json(args)


def auto_convert_to_json(args):
    """碎片树生成完成后自动转换为 JSON"""
    import subprocess
    import time
    
    # 获取脚本目录
    script_dir = Path(__file__).parent
    convert_script = script_dir / "sirius_to_subformulae.py"
    
    if not convert_script.exists():
        print(f"错误: 转换脚本不存在: {convert_script}")
        return
    
    # 确定碎片树目录和 JSON 输出目录
    fragtrees_dir = Path(args.output) / "fragmentation_trees"
    
    if args.json_output:
        json_output = Path(args.json_output)
    else:
        # 默认输出到与 fragmentation_trees 同级的 subformulae 目录
        # 例如: fragmentation_trees/pos -> subformulae/pos
        #       fragmentation_trees/neg -> subformulae/neg
        json_output = Path(args.output).parent.parent / "subformulae" / Path(args.output).name
    
    if args.mode == 'single':
        # 单文件模式：json_output 已经是正确的路径（从 output 参数解析得到）
        json_output.mkdir(parents=True, exist_ok=True)
        print(f"碎片树目录: {fragtrees_dir}")
        print(f"JSON 输出目录: {json_output}")
        
        cmd = [
            "python3", str(convert_script),
            "--input", str(fragtrees_dir),
            "--output", str(json_output),
            "--port", str(args.json_port)
        ]
        
        print(f"\n执行: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode == 0:
            json_count = len(list(json_output.glob("*.json")))
            print(f"\nJSON 转换完成! 生成 {json_count} 个文件")
        else:
            print(f"\nJSON 转换失败!")
            
    elif args.mode in ['pos', 'neg', 'both']:
        # 多模式处理
        modes = ['pos'] if args.mode == 'pos' else (['neg'] if args.mode == 'neg' else ['pos', 'neg'])
        
        for mode in modes:
            mode_frag_dir = Path(args.output) / mode / "fragmentation_trees"
            if not mode_frag_dir.exists():
                print(f"跳过 {mode}: 碎片树目录不存在")
                continue
            
            if args.json_output:
                mode_json_output = Path(args.json_output) / mode
            else:
                mode_json_output = Path(args.output).parent / "subformulae" / mode
            
            mode_json_output.mkdir(parents=True, exist_ok=True)
            
            print(f"\n--- {('正离子' if mode == 'pos' else '负离子')} JSON 转换 ---")
            print(f"碎片树目录: {mode_frag_dir}")
            print(f"JSON 输出目录: {mode_json_output}")
            
            # 每个模式使用不同端口
            port = args.json_port + (1 if mode == 'neg' else 0)
            
            cmd = [
                "python3", str(convert_script),
                "--input", str(mode_frag_dir),
                "--output", str(mode_json_output),
                "--port", str(port)
            ]
            
            print(f"执行: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=False)
            
            if result.returncode == 0:
                json_count = len(list(mode_json_output.glob("*.json")))
                print(f"完成! 生成 {json_count} 个 JSON 文件")
            else:
                print(f"转换失败!")


if __name__ == '__main__':
    main()
