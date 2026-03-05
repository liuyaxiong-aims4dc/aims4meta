#!/usr/bin/env python3
"""
Script: 03_extract_differential_spectra.py
Purpose: Extract spectra for differential compounds from original MSP file
Author: DreaMS Workflow
Date: 2026-01-26
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

# Add DreaMS to path
sys.path.append('/stor1/aims4dc/aims4dc_scripts/DreaMS')

from core import MSPParser


def extract_differential_spectra(
    original_msp: str,
    differential_csv: str,
    output_dir: str,
    extract_all: bool = False,
    extract_authentic_specific: bool = True,
    extract_counterfeit_specific: bool = True
):
    """
    Extract spectra for differential compounds

    Args:
        original_msp: Original MSP file containing all spectra
        differential_csv: CSV file with differential compounds
        output_dir: Output directory
        extract_all: Extract all differential compounds
        extract_authentic_specific: Extract authentic-specific markers
        extract_counterfeit_specific: Extract counterfeit-specific markers
    """
    print("=" * 80)
    print("提取差异化合物谱图")
    print("=" * 80)
    print()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load differential compounds
    print("[1] 加载差异化合物列表")
    df_diff = pd.read_csv(differential_csv)
    print(f"  总差异化合物数: {len(df_diff)}")

    # Filter by specificity
    import numpy as np
    authentic_specific = df_diff[df_diff['Log2FC'] == np.inf]
    counterfeit_specific = df_diff[df_diff['Log2FC'] == -np.inf]

    print(f"  正品特异性标志物: {len(authentic_specific)}")
    print(f"  伪品特异性标志物: {len(counterfeit_specific)}")
    print()

    # 2. Parse original MSP
    print("[2] 解析原始MSP文件")
    parser = MSPParser(original_msp)
    all_spectra = parser.parse()
    print(f"  总谱图数: {len(all_spectra)}")
    print()

    # Create compound name to spectrum mapping
    print("[3] 创建化合物名称到谱图的映射")
    name_to_spectrum = {}
    for spec in all_spectra:
        name = spec.get('name', '')
        if name:
            # Store both full name and extracted name (for "Unknown (xxx)" format)
            name_to_spectrum[name] = spec

            # Extract name from "Unknown (xxx)" format
            if 'Unknown (' in name and ')' in name:
                extracted_name = name.split('(')[1].split(')')[0]
                name_to_spectrum[extracted_name] = spec

    print(f"  映射条目数: {len(name_to_spectrum)}")
    print()

    # 4. Extract spectra
    print("[4] 提取差异化合物谱图")

    def extract_and_save(df_subset, output_filename, label):
        """Extract spectra for a subset of compounds"""
        if len(df_subset) == 0:
            print(f"  {label}: 无化合物需要提取")
            return

        extracted_spectra = []
        not_found = []

        for idx, row in df_subset.iterrows():
            compound_name = row['Compound']

            if compound_name in name_to_spectrum:
                extracted_spectra.append(name_to_spectrum[compound_name])
            else:
                not_found.append(compound_name)

        print(f"  {label}:")
        print(f"    提取成功: {len(extracted_spectra)}")
        print(f"    未找到: {len(not_found)}")

        if not_found and len(not_found) <= 10:
            print(f"    未找到的化合物: {', '.join(not_found[:10])}")

        # Save to MSP
        if extracted_spectra:
            output_file = output_path / output_filename
            with open(output_file, 'w') as f:
                for spec in extracted_spectra:
                    # Write spectrum in MSP format
                    f.write(f"NAME: {spec.get('name', '')}\n")

                    if 'precursor_mz' in spec:
                        f.write(f"PRECURSORMZ: {spec['precursor_mz']}\n")
                    if 'precursor_type' in spec:
                        f.write(f"PRECURSORTYPE: {spec['precursor_type']}\n")
                    if 'ion_mode' in spec:
                        f.write(f"IONMODE: {spec['ion_mode']}\n")
                    if 'formula' in spec:
                        f.write(f"FORMULA: {spec['formula']}\n")
                    if 'smiles' in spec:
                        f.write(f"SMILES: {spec['smiles']}\n")
                    if 'inchikey' in spec:
                        f.write(f"INCHIKEY: {spec['inchikey']}\n")
                    if 'retention_time' in spec:
                        f.write(f"RETENTIONTIME: {spec['retention_time']}\n")
                    if 'ccs' in spec:
                        f.write(f"CCS: {spec['ccs']}\n")
                    if 'collision_energy' in spec:
                        f.write(f"COLLISIONENERGY: {spec['collision_energy']}\n")
                    if 'instrument' in spec:
                        f.write(f"INSTRUMENT: {spec['instrument']}\n")
                    if 'instrument_type' in spec:
                        f.write(f"INSTRUMENTTYPE: {spec['instrument_type']}\n")
                    if 'comment' in spec:
                        f.write(f"COMMENT: {spec['comment']}\n")

                    # Write peaks
                    peaks = spec.get('peaks', [])
                    f.write(f"Num Peaks: {len(peaks)}\n")
                    for mz, intensity in peaks:
                        f.write(f"{mz}\t{intensity}\n")
                    f.write("\n")

            print(f"    保存至: {output_file}")
        print()

    # Extract different subsets
    if extract_all:
        extract_and_save(df_diff, "differential_compounds_all.msp", "所有差异化合物")

    if extract_authentic_specific:
        extract_and_save(authentic_specific, "differential_compounds_authentic_specific.msp",
                        "正品特异性标志物")

    if extract_counterfeit_specific:
        extract_and_save(counterfeit_specific, "differential_compounds_counterfeit_specific.msp",
                        "伪品特异性标志物")

    print("=" * 80)
    print("提取完成")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Extract spectra for differential compounds'
    )
    parser.add_argument(
        '--original-msp',
        type=str,
        required=True,
        help='Original MSP file containing all spectra'
    )
    parser.add_argument(
        '--differential-csv',
        type=str,
        required=True,
        help='CSV file with differential compounds'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--extract-all',
        action='store_true',
        help='Extract all differential compounds'
    )
    parser.add_argument(
        '--extract-authentic-specific',
        action='store_true',
        default=True,
        help='Extract authentic-specific markers (default: True)'
    )
    parser.add_argument(
        '--extract-counterfeit-specific',
        action='store_true',
        default=True,
        help='Extract counterfeit-specific markers (default: True)'
    )

    args = parser.parse_args()

    extract_differential_spectra(
        original_msp=args.original_msp,
        differential_csv=args.differential_csv,
        output_dir=args.output_dir,
        extract_all=args.extract_all,
        extract_authentic_specific=args.extract_authentic_specific,
        extract_counterfeit_specific=args.extract_counterfeit_specific
    )


if __name__ == '__main__':
    main()
