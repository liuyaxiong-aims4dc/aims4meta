#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DiffMS数据预处理脚本
将SpectraVerse数据转换为DiffMS所需的格式
"""

import os
import sys
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def parse_msp(msp_file):
    """解析MSP文件"""
    spectra = []
    current = {}
    peaks = []

    with open(msp_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Name:'):
                if current and 'smiles' in current and len(peaks) > 0:
                    current['peaks'] = peaks
                    spectra.append(current.copy())
                current = {'name': line.split(':', 1)[1].strip()}
                peaks = []
            elif line.startswith('SMILES:'):
                current['smiles'] = line.split(':', 1)[1].strip()
            elif line.startswith('InChIKey:'):
                current['inchikey'] = line.split(':', 1)[1].strip()
            elif line.startswith('Formula:'):
                current['formula'] = line.split(':', 1)[1].strip()
            elif line.startswith('PrecursorMZ:'):
                current['precursor_mz'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('PrecursorType:'):
                current['adduct'] = line.split(':', 1)[1].strip()
            elif line.startswith('Num Peaks:'):
                pass
            elif line and '\t' in line:
                parts = line.split('\t')
                if len(parts) == 2:
                    try:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        peaks.append((mz, intensity))
                    except:
                        pass

        # 保存最后一个
        if current and 'smiles' in current and len(peaks) > 0:
            current['peaks'] = peaks
            spectra.append(current.copy())

    return spectra


def compute_morgan_fp(smiles, radius=2, nBits=4096):
    """计算Morgan指纹"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        fp_array = np.zeros((nBits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        return fp_array
    except:
        return None


def normalize_spectrum(peaks, max_mz=1000.0):
    """归一化质谱数据"""
    mz = np.array([p[0] for p in peaks], dtype=np.float32)
    intensity = np.array([p[1] for p in peaks], dtype=np.float32)

    # 归一化mz到[0, 1]
    mz = mz / max_mz

    # 归一化强度到[0, 1]
    intensity = intensity / intensity.max()

    return mz, intensity


def main():
    parser = argparse.ArgumentParser(description='DiffMS数据预处理')
    parser.add_argument('--msp_pos', type=str,
                       default='/stor3/AIMS4Meta/数据库/实验数据库/spectraverse/spectraverse-1.0.1-pos.msp',
                       help='正离子MSP文件')
    parser.add_argument('--msp_neg', type=str,
                       default='/stor3/AIMS4Meta/数据库/实验数据库/spectraverse/spectraverse-1.0.1-neg.msp',
                       help='负离子MSP文件')
    parser.add_argument('--output_dir', type=str,
                       default='/stor3/AIMS4Meta/工作区/工作流/多层鉴定/L5_从头鉴定/DiffMS/data',
                       help='输出目录')
    parser.add_argument('--min_peaks', type=int, default=10,
                       help='最小碎片数')
    parser.add_argument('--max_peaks', type=int, default=500,
                       help='最大碎片数')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--max_mz', type=float, default=1000.0,
                       help='最大m/z值')

    args = parser.parse_args()

    print("=" * 80)
    print("DiffMS数据预处理")
    print("=" * 80)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 解析数据
    print(f"\n解析正离子数据: {args.msp_pos}")
    spectra_pos = parse_msp(args.msp_pos)
    print(f"  解析到 {len(spectra_pos)} 条谱图")

    print(f"\n解析负离子数据: {args.msp_neg}")
    spectra_neg = parse_msp(args.msp_neg)
    print(f"  解析到 {len(spectra_neg)} 条谱图")

    # 合并数据
    all_spectra = spectra_pos + spectra_neg
    print(f"\n总计: {len(all_spectra)} 条谱图")

    # 过滤数据
    print(f"\n过滤碎片数 {args.min_peaks}-{args.max_peaks} 的谱图...")
    filtered_spectra = [
        s for s in all_spectra
        if args.min_peaks <= len(s['peaks']) <= args.max_peaks
    ]
    print(f"  过滤后: {len(filtered_spectra)} 条谱图")

    # 处理数据
    print("\n处理数据...")
    processed_data = []
    for spectrum in tqdm(filtered_spectra):
        smiles = spectrum['smiles']

        # 计算指纹
        fp = compute_morgan_fp(smiles)
        if fp is None:
            continue

        # 归一化质谱
        mz, intensity = normalize_spectrum(spectrum['peaks'], args.max_mz)

        # 保存
        processed_data.append({
            'name': spectrum['name'],
            'smiles': smiles,
            'fingerprint': fp,
            'mz': mz,
            'intensity': intensity,
            'num_peaks': len(spectrum['peaks']),
            'precursor_mz': spectrum.get('precursor_mz', 0.0),
            'adduct': spectrum.get('adduct', ''),
            'inchikey': spectrum.get('inchikey', ''),
            'formula': spectrum.get('formula', ''),
            'ion_mode': 'positive' if spectrum in spectra_pos else 'negative'
        })

    print(f"  成功处理: {len(processed_data)} 条数据")

    # 划分数据集
    print("\n划分数据集...")
    np.random.seed(42)
    np.random.shuffle(processed_data)

    train_size = int(len(processed_data) * args.train_ratio)
    val_size = int((len(processed_data) - train_size) / 2)

    train_data = processed_data[:train_size]
    val_data = processed_data[train_size:train_size+val_size]
    test_data = processed_data[train_size+val_size:]

    print(f"  训练集: {len(train_data)}")
    print(f"  验证集: {len(val_data)}")
    print(f"  测试集: {len(test_data)}")

    # 保存数据
    print("\n保存数据...")
    for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        output_file = os.path.join(args.output_dir, f'diffms_{name}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"  保存到: {output_file}")

    # 保存统计信息
    stats = {
        'total_spectra': len(all_spectra),
        'filtered_spectra': len(filtered_spectra),
        'processed_spectra': len(processed_data),
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'min_peaks': args.min_peaks,
        'max_peaks': args.max_peaks,
        'max_mz': args.max_mz
    }

    stats_file = os.path.join(args.output_dir, 'preprocess_stats.pkl')
    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f)

    print(f"\n统计信息保存到: {stats_file}")
    print("\n预处理完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
