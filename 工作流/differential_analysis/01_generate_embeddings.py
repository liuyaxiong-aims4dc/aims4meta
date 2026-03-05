#!/usr/bin/env python3
"""
Script: 01_generate_dreams_embeddings.py
Purpose: Generate DreaMS embeddings for Group A (authentic) and Group B (counterfeit) samples
Author: DreaMS Workflow
Date: 2026-01-10
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add DreaMS to path
sys.path.append('/stor1/aims4dc/aims4dc_scripts/DreaMS')

from core import generate_dreams_embeddings, MSPParser


class DifferentialEmbeddingsGenerator:
    """Generate DreaMS embeddings for differential analysis"""

    def __init__(self, group_a_dir: str, group_b_dir: str, output_dir: str):
        """
        Initialize the generator

        Args:
            group_a_dir: Directory containing Group A (authentic) MSP files
            group_b_dir: Directory containing Group B (counterfeit) MSP files
            output_dir: Output directory for embeddings and metadata
        """
        self.group_a_dir = Path(group_a_dir)
        self.group_b_dir = Path(group_b_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def merge_group_spectra(self, group_dir: Path, group_label: str, output_msp: Path) -> list:
        """
        Merge all MSP files from a group

        Args:
            group_dir: Directory containing MSP files
            group_label: Group label ('A' for authentic, 'B' for counterfeit)
            output_msp: Output merged MSP file path

        Returns:
            List of spectrum dictionaries with metadata
        """
        print(f"\n=== Merging {group_label} group spectra ===")
        all_spectra = []

        msp_files = sorted(group_dir.glob('*.msp'))
        if not msp_files:
            raise FileNotFoundError(f"No MSP files found in {group_dir}")

        print(f"Found {len(msp_files)} MSP files")

        for msp_file in msp_files:
            print(f"  Processing: {msp_file.name}")
            parser = MSPParser(str(msp_file))
            spectra = parser.parse()

            sample_id = msp_file.stem

            # Add sample and group information
            for spectrum in spectra:
                spectrum['sample_id'] = sample_id
                spectrum['group'] = group_label
                all_spectra.append(spectrum)

        print(f"Total spectra in group {group_label}: {len(all_spectra)}")

        # Save merged MSP
        self._save_merged_msp(all_spectra, output_msp)

        return all_spectra

    def _save_merged_msp(self, spectra: list, output_path: Path):
        """Save merged spectra to MSP file"""
        with open(output_path, 'w') as f:
            for spectrum in spectra:
                # Write metadata
                f.write(f"NAME: {spectrum.get('name', 'Unknown')}\n")
                f.write(f"PRECURSORMZ: {spectrum.get('precursor_mz', 0)}\n")
                f.write(f"PRECURSORTYPE: {spectrum.get('precursor_type', '[M+H]+')}\n")

                if 'retention_time' in spectrum:
                    f.write(f"RETENTIONTIME: {spectrum['retention_time']}\n")
                if 'ionmode' in spectrum:
                    f.write(f"IONMODE: {spectrum['ionmode']}\n")
                if 'sample_id' in spectrum:
                    f.write(f"SAMPLE_ID: {spectrum['sample_id']}\n")
                if 'group' in spectrum:
                    f.write(f"GROUP: {spectrum['group']}\n")

                # Write peaks
                peaks = spectrum.get('peaks', [])
                f.write(f"Num Peaks: {len(peaks)}\n")
                for mz, intensity in peaks:
                    f.write(f"{mz}\t{intensity}\n")
                f.write("\n")

        print(f"Saved merged MSP: {output_path}")

    def generate_embeddings(self, msp_file: Path) -> tuple:
        """
        Generate DreaMS embeddings for merged spectra

        Args:
            msp_file: Input MSP file

        Returns:
            Tuple of (embeddings, spectrum_names, metadata)
        """
        print(f"\n=== Generating DreaMS embeddings ===")
        print(f"Input: {msp_file}")

        embeddings, spectrum_names, mgf_path = generate_dreams_embeddings(
            input_file=str(msp_file),
            output_dir=str(self.output_dir / 'embeddings'),
            output_format='npz',
            keep_mgf=True
        )

        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Generated MGF: {mgf_path}")

        return embeddings, spectrum_names, mgf_path

    def create_metadata_table(self, spectra_a: list, spectra_b: list) -> pd.DataFrame:
        """
        Create metadata table for all spectra

        Args:
            spectra_a: List of Group A spectra
            spectra_b: List of Group B spectra

        Returns:
            DataFrame with spectrum metadata
        """
        print("\n=== Creating metadata table ===")

        all_spectra = spectra_a + spectra_b

        metadata = []
        for i, spectrum in enumerate(all_spectra):
            metadata.append({
                'index': i,
                'name': spectrum.get('name', f'spectrum_{i}'),
                'sample_id': spectrum.get('sample_id', 'unknown'),
                'group': spectrum.get('group', 'unknown'),
                'precursor_mz': spectrum.get('precursor_mz', 0),
                'precursor_type': spectrum.get('precursor_type', '[M+H]+'),
                'retention_time': spectrum.get('retention_time', 0),
                'ionmode': spectrum.get('ionmode', 'Positive')
            })

        df_metadata = pd.DataFrame(metadata)

        # Save metadata
        metadata_path = self.output_dir / 'spectrum_metadata.csv'
        df_metadata.to_csv(metadata_path, index=False)
        print(f"Saved metadata: {metadata_path}")

        # Print summary
        print(f"\nMetadata summary:")
        print(f"  Total spectra: {len(df_metadata)}")
        print(f"  Group A (authentic): {len(df_metadata[df_metadata['group'] == 'A'])}")
        print(f"  Group B (counterfeit): {len(df_metadata[df_metadata['group'] == 'B'])}")

        return df_metadata

    def run(self):
        """Run the complete embedding generation workflow"""
        print("=" * 80)
        print("DreaMS Differential Analysis - Step 1: Embedding Generation")
        print("=" * 80)

        # Step 1: Merge Group A spectra
        print("\n[Step 1/5] Merging Group A (authentic) spectra...")
        spectra_a = self.merge_group_spectra(
            self.group_a_dir,
            'A',
            self.output_dir / 'merged_spectra_A.msp'
        )

        # Step 2: Merge Group B spectra
        print("\n[Step 2/5] Merging Group B (counterfeit) spectra...")
        spectra_b = self.merge_group_spectra(
            self.group_b_dir,
            'B',
            self.output_dir / 'merged_spectra_B.msp'
        )

        # Step 3: Merge all spectra
        print("\n[Step 3/5] Merging all spectra...")
        all_spectra = spectra_a + spectra_b
        merged_all_path = self.output_dir / 'merged_all.msp'
        self._save_merged_msp(all_spectra, merged_all_path)
        print(f"Total combined spectra: {len(all_spectra)}")

        # Step 4: Generate DreaMS embeddings
        print("\n[Step 4/5] Generating DreaMS embeddings...")
        embeddings, spectrum_names, mgf_path = self.generate_embeddings(merged_all_path)

        # Step 5: Create metadata table
        print("\n[Step 5/5] Creating metadata table...")
        metadata_df = self.create_metadata_table(spectra_a, spectra_b)

        # Final summary
        print("\n" + "=" * 80)
        print("✓ Embedding generation complete!")
        print("=" * 80)
        print("\nOutput files:")
        print(f"  - Merged spectra (A): {self.output_dir / 'merged_spectra_A.msp'}")
        print(f"  - Merged spectra (B): {self.output_dir / 'merged_spectra_B.msp'}")
        print(f"  - Merged spectra (All): {merged_all_path}")
        print(f"  - DreaMS embeddings: {self.output_dir / 'embeddings' / 'embeddings.npz'}")
        print(f"  - MGF file: {mgf_path}")
        print(f"  - Metadata: {self.output_dir / 'spectrum_metadata.csv'}")
        print()

        return {
            'embeddings': embeddings,
            'metadata': metadata_df,
            'mgf_path': mgf_path,
            'merged_msp': merged_all_path
        }


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate DreaMS embeddings for differential analysis'
    )
    parser.add_argument(
        '--group-a',
        type=str,
        required=True,
        help='Directory containing Group A (authentic) MSP files'
    )
    parser.add_argument(
        '--group-b',
        type=str,
        required=True,
        help='Directory containing Group B (counterfeit) MSP files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./differential_analysis_results',
        help='Output directory (default: ./differential_analysis_results)'
    )

    args = parser.parse_args()

    # Run workflow
    generator = DifferentialEmbeddingsGenerator(
        group_a_dir=args.group_a,
        group_b_dir=args.group_b,
        output_dir=args.output
    )

    results = generator.run()

    print("Next step: Run 02_differential_analysis.py")


if __name__ == '__main__':
    main()
