#!/usr/bin/env python3
"""
Script: 04_L1_library_matching.py
Purpose: L1 identification - MSDial library matching for differential compounds
Author: DreaMS Workflow
Date: 2026-01-26
"""

import sys
import argparse
from pathlib import Path

# Add DreaMS to path
sys.path.append('/stor1/aims4dc/aims4dc_scripts/DreaMS')

from core.spectral_matching import run_library_matching


def run_l1_identification(
    query_msp: str,
    library_msp: str,
    library_npz: str,
    output_dir: str,
    top_k: int = 10,
    precursor_ppm_tolerance: float = 10.0,
    dreams_threshold: float = 0.7,
    modified_cosine_threshold: float = 0.6,
    fragment_mz_tolerance: float = 0.05
):
    """
    Run L1 identification using MSDial library matching

    Args:
        query_msp: Query MSP file (differential compounds)
        library_msp: Library MSP file (MSDial)
        library_npz: Library embeddings file
        output_dir: Output directory
        top_k: Number of top matches to return
        precursor_ppm_tolerance: Precursor m/z tolerance in ppm
        dreams_threshold: Minimum DreaMS similarity score
        modified_cosine_threshold: Minimum Modified Cosine score
        fragment_mz_tolerance: Fragment m/z tolerance in Da
    """
    print("=" * 80)
    print("L1 鉴定：MSDial 谱图库匹配")
    print("=" * 80)
    print()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"查询文件: {query_msp}")
    print(f"库文件: {library_msp}")
    print(f"库向量: {library_npz}")
    print(f"输出目录: {output_dir}")
    print()

    print("匹配参数:")
    print(f"  Top-K: {top_k}")
    print(f"  前体 ppm 容差: ±{precursor_ppm_tolerance}")
    print(f"  DreaMS 阈值: {dreams_threshold}")
    print(f"  Modified Cosine 阈值: {modified_cosine_threshold}")
    print(f"  碎片 m/z 容差: ±{fragment_mz_tolerance} Da")
    print()

    # Run library matching
    df_results, query_embeddings, query_spectra, mgf_file = run_library_matching(
        query_msp=query_msp,
        library_npz=library_npz,
        library_msp=library_msp,
        output_dir=output_dir,
        top_k=top_k,
        precursor_ppm_tolerance=precursor_ppm_tolerance,
        dreams_threshold=dreams_threshold,
        modified_cosine_threshold=modified_cosine_threshold,
        fragment_mz_tolerance=fragment_mz_tolerance,
        compute_modified_cosine=True,
        enable_isotope_scoring=False,  # No isotope data in MSP
        enable_ccs_scoring=False,  # No CCS data in MSP
        output_format='csv',
        save_query_embeddings=True,
        keep_mgf=True
    )

    print()
    print("=" * 80)
    print("L1 鉴定完成")
    print("=" * 80)
    print()
    print(f"总匹配数: {len(df_results)}")
    print(f"查询化合物数: {len(query_spectra)}")

    if len(df_results) > 0:
        # Count unique queries with matches
        unique_queries = df_results['query_name'].nunique()
        print(f"有匹配的查询化合物数: {unique_queries}")

        # Count high-confidence matches
        high_conf = df_results[
            (df_results['DreaMS_similarity'] >= 0.8) &
            (df_results['modified_cosine_similarity'] >= 0.7)
        ]
        print(f"高置信度匹配数 (DreaMS≥0.8, ModCos≥0.7): {len(high_conf)}")

        # Show top matches
        print()
        print("Top 10 最佳匹配:")
        print("-" * 80)
        top_matches = df_results.nsmallest(10, 'rank')
        for idx, row in top_matches.iterrows():
            print(f"{row['query_name']}")
            print(f"  → {row['library_name']}")
            print(f"     DreaMS: {row['DreaMS_similarity']:.3f}, "
                  f"ModCos: {row['modified_cosine_similarity']:.3f}, "
                  f"ppm: {row['precursor_ppm_error']:.2f}")
            print()

    return df_results


def main():
    parser = argparse.ArgumentParser(
        description='L1 identification using MSDial library matching'
    )
    parser.add_argument(
        '--query-msp',
        type=str,
        required=True,
        help='Query MSP file (differential compounds)'
    )
    parser.add_argument(
        '--library-msp',
        type=str,
        required=True,
        help='Library MSP file (MSDial)'
    )
    parser.add_argument(
        '--library-npz',
        type=str,
        required=True,
        help='Library embeddings file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of top matches to return (default: 10)'
    )
    parser.add_argument(
        '--precursor-ppm-tolerance',
        type=float,
        default=10.0,
        help='Precursor m/z tolerance in ppm (default: 10.0)'
    )
    parser.add_argument(
        '--dreams-threshold',
        type=float,
        default=0.7,
        help='Minimum DreaMS similarity score (default: 0.7)'
    )
    parser.add_argument(
        '--modified-cosine-threshold',
        type=float,
        default=0.6,
        help='Minimum Modified Cosine score (default: 0.6)'
    )
    parser.add_argument(
        '--fragment-mz-tolerance',
        type=float,
        default=0.05,
        help='Fragment m/z tolerance in Da (default: 0.05)'
    )

    args = parser.parse_args()

    run_l1_identification(
        query_msp=args.query_msp,
        library_msp=args.library_msp,
        library_npz=args.library_npz,
        output_dir=args.output_dir,
        top_k=args.top_k,
        precursor_ppm_tolerance=args.precursor_ppm_tolerance,
        dreams_threshold=args.dreams_threshold,
        modified_cosine_threshold=args.modified_cosine_threshold,
        fragment_mz_tolerance=args.fragment_mz_tolerance
    )


if __name__ == '__main__':
    main()
