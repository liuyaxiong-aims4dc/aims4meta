#!/usr/bin/env python3
"""
L3 SIRIUS 参数续传校验辅助脚本

在总控脚本调用 L3 SIRIUS 前后执行：
  --mode check  : 运行前检查参数是否变化，若变化则删除旧项目
  --mode save   : 运行成功后保存参数，供下次比对

用法:
  # 运行前检查
  python L3参数续传校验.py --mode check \
      --output_dir /path/to/L3_results \
      --sample_msp /path/to/input.msp \
      --ion_mode POS --instrument orbitrap \
      --databases BIO drugbank \
      --mz_threshold 1500 --min_peaks 6

  # 运行成功后保存
  python L3参数续传校验.py --mode save \
      --output_dir /path/to/L3_results \
      --sample_msp /path/to/input.msp \
      ... (同样的参数)

追踪的参数：MSP身份、ion_mode、instrument、databases、mz_threshold、
min_peaks、min_intensity、sample_csv、加合物类型、formula候选数
"""

import argparse
import hashlib
import json
import os
import shutil
import sys


def _get_msp_identity(msp_path: str) -> str:
    """MSP 文件身份标识（路径 + 大小 + 修改时间）"""
    if not msp_path or not os.path.exists(msp_path):
        return f"nonexistent:{msp_path}"
    stat = os.stat(msp_path)
    return f"{os.path.abspath(msp_path)}:{stat.st_size}:{stat.st_mtime}"


def _compute_hash(args) -> str:
    """计算当前参数 SHA256 哈希"""
    params = {
        'msp_identity': _get_msp_identity(args.sample_msp),
        'instrument': args.instrument,
        'ion_mode': args.ion_mode,
        'databases': sorted(args.databases),
        'mz_threshold': args.mz_threshold,
        'min_peaks': args.min_peaks,
        'min_intensity': args.min_intensity,
        'sample_csv': args.sample_csv or '',
        'enforced_adduct': '[M+H]+,[M+Na]+' if args.ion_mode == 'POS' else '[M-H]-',
        'formula_candidates': 10,
    }
    params_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(params_str.encode('utf-8')).hexdigest(), params


def _check_mode(args):
    """check 模式：比对参数，变化时删除旧项目"""
    params_hash, params = _compute_hash(args)
    params_file = os.path.join(args.output_dir, 'sirius_params.json')
    sirius_project = os.path.join(args.output_dir, 'sirius_project.sirius')

    if not os.path.exists(sirius_project):
        print("[L3参数校验] 无旧 SIRIUS 项目，将新建")
        return

    if not os.path.exists(params_file):
        print("[L3参数校验] 未找到参数文件（可能为旧版运行结果），删除旧项目确保准确")
        shutil.rmtree(sirius_project)
        return

    try:
        with open(params_file, 'r', encoding='utf-8') as f:
            old_data = json.load(f)
        old_hash = old_data.get('params_hash', '')
        old_params = old_data.get('params', {})

        if old_hash == params_hash:
            size_mb = os.path.getsize(sirius_project) / (1024 * 1024)
            print(f"[L3参数校验] ✓ 参数未变化，续传（{size_mb:.1f} MB）")
            return

        # 参数变化，打印差异
        print("[L3参数校验] ⚠ 参数变更，自动删除旧项目重新计算：")
        for key in sorted(set(list(params.keys()) + list(old_params.keys()))):
            old_val = old_params.get(key, '<缺失>')
            new_val = params.get(key, '<缺失>')
            if str(old_val) != str(new_val):
                if key == 'msp_identity':
                    # 只显示变化要点
                    print(f"  输入MSP已变化")
                else:
                    print(f"  {key}: {old_val} → {new_val}")

    except Exception as e:
        print(f"[L3参数校验] 读取参数文件异常: {e}，删除旧项目确保准确")

    if os.path.exists(params_file):
        os.remove(params_file)
    if os.path.exists(sirius_project):
        shutil.rmtree(sirius_project)
    print("[L3参数校验] 已清理旧项目，将从头计算")


def _save_mode(args):
    """save 模式：保存当前参数哈希"""
    params_hash, params = _compute_hash(args)
    params_file = os.path.join(args.output_dir, 'sirius_params.json')

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump({'params_hash': params_hash, 'params': params}, f,
                      ensure_ascii=False, indent=2)
        print(f"[L3参数校验] 参数已保存: {params_file}")
    except Exception as e:
        print(f"[L3参数校验] 警告: 无法保存参数文件: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='L3 SIRIUS 参数续传校验')
    parser.add_argument('--mode', required=True, choices=['check', 'save'],
                        help='check=运行前检查 / save=运行后保存')
    parser.add_argument('--output_dir', required=True, help='L3 输出目录')
    parser.add_argument('--sample_msp', required=True, help='L3 输入 MSP 文件')
    parser.add_argument('--ion_mode', required=True, choices=['POS', 'NEG'])
    parser.add_argument('--instrument', required=True, choices=['orbitrap', 'qtof'])
    parser.add_argument('--databases', nargs='+', default=[],
                        help='结构数据库列表')
    parser.add_argument('--mz_threshold', type=float, default=1500)
    parser.add_argument('--min_peaks', type=int, default=6)
    parser.add_argument('--min_intensity', type=float, default=0)
    parser.add_argument('--sample_csv', default='')

    args = parser.parse_args()

    if args.mode == 'check':
        _check_mode(args)
    else:
        _save_mode(args)


if __name__ == '__main__':
    main()
