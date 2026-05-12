"""
SIRIUS 碎裂树生成脚本
"""
import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp


# SIRIUS 路径
SIRIUS_PATH = "/stor1/AIMS4Meta/源代码/SIRIUS/sirius-6.3.3-linux-x64/sirius/bin/sirius"


def parse_args():
    parser = argparse.ArgumentParser(description='运行 SIRIUS 生成碎裂树')
    parser.add_argument('--input', type=str, required=True,
                       help='输入 MSP 文件路径')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='并行进程数')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='每批处理的谱图数')
    parser.add_argument('--profile', type=str, default='qtof',
                       choices=['qtof', 'orbitrap', 'fticr'],
                       help='仪器类型')
    return parser.parse_args()


def split_msp(msp_path: str, output_dir: str, batch_size: int = 100):
    """
    将 MSP 文件分割成多个小文件
    
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
    
    with open(msp_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                if current_spectrum:
                    spectra.append(current_spectrum)
                    current_spectrum = []
            else:
                current_spectrum.append(line)
        
        if current_spectrum:
            spectra.append(current_spectrum)
    
    print(f"总共 {len(spectra)} 个谱图")
    
    # 分割
    batch_files = []
    for i in range(0, len(spectra), batch_size):
        batch = spectra[i:i + batch_size]
        batch_file = os.path.join(output_dir, f'batch_{i//batch_size:05d}.msp')
        
        with open(batch_file, 'w') as f:
            for spectrum in batch:
                f.writelines(spectrum)
                f.write('\n')
        
        batch_files.append(batch_file)
    
    print(f"分割成 {len(batch_files)} 个批次")
    return batch_files


def run_sirius_batch(msp_file: str, output_dir: str, profile: str = 'qtof'):
    """
    对一个批次的谱图运行 SIRIUS
    
    Args:
        msp_file: MSP 文件路径
        output_dir: 输出目录
        profile: 仪器类型
    """
    batch_name = Path(msp_file).stem
    batch_output = os.path.join(output_dir, batch_name)
    os.makedirs(batch_output, exist_ok=True)
    
    # 构建 SIRIUS 命令
    cmd = [
        SIRIUS_PATH,
        '--input', msp_file,
        '--output', batch_output,
        'formula',
        '--profile', profile,
        '--candidates', '1',
        '--no-recalibration'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        if result.returncode != 0:
            print(f"SIRIUS 错误 ({batch_name}): {result.stderr}")
            return False
        
        return True
    
    except subprocess.TimeoutExpired:
        print(f"SIRIUS 超时 ({batch_name})")
        return False
    except Exception as e:
        print(f"SIRIUS 异常 ({batch_name}): {e}")
        return False


def process_batch(args):
    """处理单个批次 (用于并行)"""
    msp_file, output_dir, profile = args
    return run_sirius_batch(msp_file, output_dir, profile)


def main():
    args = parse_args()
    
    # 检查 SIRIUS
    if not os.path.exists(SIRIUS_PATH):
        print(f"错误: SIRIUS 不存在于 {SIRIUS_PATH}")
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 分割 MSP 文件
    print("分割 MSP 文件...")
    temp_dir = os.path.join(args.output, 'temp_batches')
    batch_files = split_msp(args.input, temp_dir, args.batch_size)
    
    # 并行运行 SIRIUS
    print(f"运行 SIRIUS (并行数: {args.num_workers})...")
    
    process_args = [(f, args.output, args.profile) for f in batch_files]
    
    with mp.Pool(args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_batch, process_args),
            total=len(batch_files),
            desc="处理批次"
        ))
    
    # 统计结果
    success_count = sum(results)
    print(f"\n完成! 成功: {success_count}/{len(batch_files)}")
    
    # 清理临时文件
    print("清理临时文件...")
    import shutil
    shutil.rmtree(temp_dir)
    
    print("SIRIUS 处理完成!")


if __name__ == '__main__':
    main()
