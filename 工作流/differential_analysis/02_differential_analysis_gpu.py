#!/usr/bin/env python3
"""
Script: 02_differential_analysis_gpu.py
Purpose: GPU-accelerated differential analysis between Group A and B
Author: DreaMS Workflow
Date: 2026-01-25
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Add DreaMS to path
sys.path.append('/stor1/aims4dc/aims4dc_scripts/DreaMS')


class DifferentialAnalyzerGPU:
    """GPU-accelerated differential compound analyzer"""

    def __init__(self, embeddings_file: str, metadata_file: str, output_dir: str):
        """
        Initialize analyzer

        Args:
            embeddings_file: Path to embeddings NPZ file
            metadata_file: Path to metadata CSV file
            output_dir: Output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / 'statistics').mkdir(exist_ok=True)
        (self.output_dir / 'visualization').mkdir(exist_ok=True)

        # Check GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Load data
        print("\nLoading embeddings and metadata...")
        self.embeddings = np.load(embeddings_file)['embeddings']
        self.metadata = pd.read_csv(metadata_file)

        print(f"  Embeddings shape: {self.embeddings.shape}")
        print(f"  Metadata rows: {len(self.metadata)}")

        # Group indices
        self.group_a_idx = self.metadata[self.metadata['group'] == 'A'].index.tolist()
        self.group_b_idx = self.metadata[self.metadata['group'] == 'B'].index.tolist()

        print(f"  Group A samples: {len(self.group_a_idx)}")
        print(f"  Group B samples: {len(self.group_b_idx)}")

    def pca_analysis(self):
        """PCA analysis to visualize group separation"""
        print("\n=== PCA Analysis ===")

        # PCA
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(self.embeddings)

        # Save PCA coordinates
        pca_df = pd.DataFrame({
            'PC1': embeddings_pca[:, 0],
            'PC2': embeddings_pca[:, 1],
            'group': self.metadata['group'],
            'sample_id': self.metadata['sample_id']
        })
        pca_df.to_csv(self.output_dir / 'statistics' / 'pca_coordinates.csv', index=False)

        # Visualization
        plt.figure(figsize=(12, 8))

        # Group A
        plt.scatter(
            embeddings_pca[self.group_a_idx, 0],
            embeddings_pca[self.group_a_idx, 1],
            c='#2E86DE',
            label='Group A (Authentic)',
            alpha=0.7,
            s=60,
            edgecolors='white',
            linewidth=0.5
        )

        # Group B
        plt.scatter(
            embeddings_pca[self.group_b_idx, 0],
            embeddings_pca[self.group_b_idx, 1],
            c='#EE5A6F',
            label='Group B (Counterfeit)',
            alpha=0.7,
            s=60,
            edgecolors='white',
            linewidth=0.5
        )

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        plt.title('PCA: Group A vs Group B', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        pca_plot_path = self.output_dir / 'visualization' / 'pca_plot.png'
        plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  PCA plot saved: {pca_plot_path}")

        return embeddings_pca, pca

    def find_unique_compounds_gpu(self, similarity_threshold: float = 0.85, batch_size: int = 500):
        """
        Fully vectorized GPU-accelerated method to find compounds unique to each group

        Strategy: High within-group similarity, low between-group similarity

        Args:
            similarity_threshold: Minimum within-group similarity
            batch_size: Number of spectra to process at once (for memory efficiency)

        Returns:
            Tuple of (unique_A_df, unique_B_df)
        """
        print("\n=== Finding Unique Compounds (Fully Vectorized GPU) ===")
        print(f"Similarity threshold: {similarity_threshold}")
        print(f"Batch size: {batch_size}")

        # Normalize embeddings on CPU first
        print("\nNormalizing embeddings on CPU...")
        embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        # Keep group embeddings on CPU, load to GPU in chunks
        group_a_emb_cpu = embeddings_norm[self.group_a_idx]
        group_b_emb_cpu = embeddings_norm[self.group_b_idx]

        print(f"  Group A embeddings: {group_a_emb_cpu.shape}")
        print(f"  Group B embeddings: {group_b_emb_cpu.shape}")

        # Pre-extract metadata columns for faster access
        print("\nPre-extracting metadata...")
        groups = self.metadata['group'].values
        names = self.metadata['name'].values
        sample_ids = self.metadata['sample_id'].values
        precursor_mzs = self.metadata['precursor_mz'].values
        retention_times = self.metadata['retention_time'].values

        # Create index mappings for O(1) lookup (critical optimization!)
        print("Creating index mappings for O(1) lookup...")
        idx_to_pos_a = {idx: pos for pos, idx in enumerate(self.group_a_idx)}
        idx_to_pos_b = {idx: pos for pos, idx in enumerate(self.group_b_idx)}

        # Initialize arrays to store average similarities
        n_samples = len(self.embeddings)
        avg_sim_to_a = np.zeros(n_samples, dtype=np.float32)
        avg_sim_to_b = np.zeros(n_samples, dtype=np.float32)

        # Process in batches
        n_batches = (n_samples + batch_size - 1) // batch_size

        print(f"\nCalculating similarities for {n_samples} spectra in {n_batches} batches...")
        print("(This is the main computation step)")

        # Convert group embeddings to torch tensors on GPU
        group_a_emb_gpu = torch.from_numpy(group_a_emb_cpu).float().to(self.device)
        group_b_emb_gpu = torch.from_numpy(group_b_emb_cpu).float().to(self.device)

        import sys
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)

            if (batch_idx + 1) % 100 == 0 or batch_idx == n_batches - 1:
                print(f"  Batch {batch_idx + 1}/{n_batches} ({end_idx}/{n_samples} spectra)", flush=True)
                sys.stdout.flush()

            # Get batch embeddings and move to GPU
            batch_emb_cpu = embeddings_norm[start_idx:end_idx]
            batch_emb = torch.from_numpy(batch_emb_cpu).float().to(self.device)

            # Calculate similarities to both groups
            sim_to_a = torch.mm(batch_emb, group_a_emb_gpu.t())
            sim_to_b = torch.mm(batch_emb, group_b_emb_gpu.t())

            # Move results back to CPU
            sim_to_a_cpu = sim_to_a.cpu().numpy()
            sim_to_b_cpu = sim_to_b.cpu().numpy()

            # Clear GPU memory
            del batch_emb, sim_to_a, sim_to_b
            torch.cuda.empty_cache()

            # Vectorized calculation of average similarities
            batch_groups = groups[start_idx:end_idx]

            for i in range(len(batch_emb_cpu)):
                global_idx = start_idx + i

                if batch_groups[i] == 'A':
                    # Exclude self from group A (O(1) lookup!)
                    self_pos = idx_to_pos_a[global_idx]
                    mask = np.ones(len(self.group_a_idx), dtype=bool)
                    mask[self_pos] = False
                    avg_sim_to_a[global_idx] = sim_to_a_cpu[i][mask].mean() if mask.sum() > 0 else 0
                    avg_sim_to_b[global_idx] = sim_to_b_cpu[i].mean()
                elif batch_groups[i] == 'B':
                    # Exclude self from group B (O(1) lookup!)
                    self_pos = idx_to_pos_b[global_idx]
                    mask = np.ones(len(self.group_b_idx), dtype=bool)
                    mask[self_pos] = False
                    avg_sim_to_a[global_idx] = sim_to_a_cpu[i].mean()
                    avg_sim_to_b[global_idx] = sim_to_b_cpu[i][mask].mean() if mask.sum() > 0 else 0

        print("\nFiltering group-specific compounds...", flush=True)

        # Vectorized filtering for Group A (authentic-specific)
        mask_a = (groups == 'A') & (avg_sim_to_a > similarity_threshold) & (avg_sim_to_b < 0.7)
        indices_a = np.where(mask_a)[0]

        unique_a = []
        for idx in indices_a:
            unique_a.append({
                'index': int(idx),
                'name': names[idx],
                'sample_id': sample_ids[idx],
                'precursor_mz': float(precursor_mzs[idx]),
                'retention_time': float(retention_times[idx]),
                'sim_to_own_group': float(avg_sim_to_a[idx]),
                'sim_to_other_group': float(avg_sim_to_b[idx]),
                'specificity': float(avg_sim_to_a[idx] - avg_sim_to_b[idx])
            })

        # Vectorized filtering for Group B (counterfeit-specific)
        mask_b = (groups == 'B') & (avg_sim_to_b > similarity_threshold) & (avg_sim_to_a < 0.7)
        indices_b = np.where(mask_b)[0]

        unique_b = []
        for idx in indices_b:
            unique_b.append({
                'index': int(idx),
                'name': names[idx],
                'sample_id': sample_ids[idx],
                'precursor_mz': float(precursor_mzs[idx]),
                'retention_time': float(retention_times[idx]),
                'sim_to_own_group': float(avg_sim_to_b[idx]),
                'sim_to_other_group': float(avg_sim_to_a[idx]),
                'specificity': float(avg_sim_to_b[idx] - avg_sim_to_a[idx])
            })

        # Convert to DataFrames and sort by specificity
        df_unique_a = pd.DataFrame(unique_a).sort_values('specificity', ascending=False) if unique_a else pd.DataFrame()
        df_unique_b = pd.DataFrame(unique_b).sort_values('specificity', ascending=False) if unique_b else pd.DataFrame()

        # Save results
        df_unique_a.to_csv(
            self.output_dir / 'statistics' / 'unique_to_authentic.csv',
            index=False
        )
        df_unique_b.to_csv(
            self.output_dir / 'statistics' / 'unique_to_counterfeit.csv',
            index=False
        )

        print(f"\n  Authentic-specific compounds: {len(unique_a)}")
        print(f"  Counterfeit-specific compounds: {len(unique_b)}")

        return df_unique_a, df_unique_b

    def quantitative_difference(self, feature_table_a: str = None, feature_table_b: str = None):
        """Quantitative differential analysis (optional)"""
        if feature_table_a is None or feature_table_b is None:
            print("\n=== Quantitative Analysis ===")
            print("  Skipped (no feature tables provided)")
            return None

        print("\n=== Quantitative Differential Analysis ===")
        # Load feature tables
        df_a = pd.read_csv(feature_table_a)
        df_b = pd.read_csv(feature_table_b)

        # Perform statistical tests (t-test, fold change, etc.)
        # ... implementation ...

        return None

    def plot_volcano(self, diff_quant):
        """Plot volcano plot for quantitative differences"""
        if diff_quant is None:
            return

        # ... implementation ...
        pass

    def generate_summary_report(self, unique_a, unique_b):
        """Generate summary report"""
        print("\n=== Generating Summary Report ===")

        report_path = self.output_dir / 'DIFFERENTIAL_SUMMARY_REPORT.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DIFFERENTIAL ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("## Dataset Overview\n")
            f.write(f"Total spectra: {len(self.embeddings)}\n")
            f.write(f"Group A (Authentic): {len(self.group_a_idx)} spectra\n")
            f.write(f"Group B (Counterfeit): {len(self.group_b_idx)} spectra\n\n")

            f.write("## Differential Compounds\n")
            f.write(f"Authentic-specific markers: {len(unique_a)}\n")
            f.write(f"Counterfeit-specific markers: {len(unique_b)}\n\n")

            if len(unique_a) > 0:
                f.write("## Top 10 Authentic-Specific Markers\n")
                for i, row in unique_a.head(10).iterrows():
                    f.write(f"{i+1}. {row['name']}\n")
                    f.write(f"   m/z: {row['precursor_mz']:.4f}\n")
                    f.write(f"   Specificity: {row['specificity']:.3f}\n\n")

            if len(unique_b) > 0:
                f.write("## Top 10 Counterfeit-Specific Markers\n")
                for i, row in unique_b.head(10).iterrows():
                    f.write(f"{i+1}. {row['name']}\n")
                    f.write(f"   m/z: {row['precursor_mz']:.4f}\n")
                    f.write(f"   Specificity: {row['specificity']:.3f}\n\n")

        print(f"  Report saved: {report_path}")

    def run(self, similarity_threshold: float = 0.85,
            feature_table_a: str = None, feature_table_b: str = None,
            batch_size: int = 500):
        """
        Run complete differential analysis workflow

        Args:
            similarity_threshold: Similarity threshold for unique compound detection
            feature_table_a: Optional feature table for Group A
            feature_table_b: Optional feature table for Group B
            batch_size: Batch size for GPU processing

        Returns:
            Dictionary with results
        """
        print("\n" + "=" * 80)
        print("DIFFERENTIAL ANALYSIS WORKFLOW (GPU-ACCELERATED)")
        print("=" * 80)

        # Step 1: PCA analysis
        print("\n[Step 1/4] PCA analysis...")
        pca_coords, pca = self.pca_analysis()

        # Step 2: Find unique compounds (GPU-accelerated)
        print("\n[Step 2/4] Identifying group-specific compounds (GPU)...")
        unique_a, unique_b = self.find_unique_compounds_gpu(similarity_threshold, batch_size)

        # Step 3: Quantitative differential analysis (if feature tables provided)
        print("\n[Step 3/4] Quantitative differential analysis...")
        diff_quant = self.quantitative_difference(feature_table_a, feature_table_b)
        if diff_quant is not None:
            self.plot_volcano(diff_quant)

        # Step 4: Generate summary report
        print("\n[Step 4/4] Generating summary report...")
        self.generate_summary_report(unique_a, unique_b)

        # Final summary
        print("\n" + "=" * 80)
        print("✓ Differential analysis complete!")
        print("=" * 80)
        print("\nOutput files:")
        print(f"  - PCA coordinates: {self.output_dir / 'statistics' / 'pca_coordinates.csv'}")
        print(f"  - Authentic-specific: {self.output_dir / 'statistics' / 'unique_to_authentic.csv'}")
        print(f"  - Counterfeit-specific: {self.output_dir / 'statistics' / 'unique_to_counterfeit.csv'}")
        print(f"  - PCA plot: {self.output_dir / 'visualization' / 'pca_plot.png'}")
        print(f"  - Summary report: {self.output_dir / 'DIFFERENTIAL_SUMMARY_REPORT.txt'}")
        print("\nNext step: Run 03_identify_differential_compounds.py")
        print()

        return {
            'unique_authentic': unique_a,
            'unique_counterfeit': unique_b,
            'pca_coords': pca_coords
        }


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='GPU-accelerated differential compound analysis'
    )
    parser.add_argument(
        '--embeddings',
        type=str,
        required=True,
        help='Path to DreaMS embeddings NPZ file'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='Path to spectrum metadata CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./differential_analysis_results',
        help='Output directory'
    )
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.85,
        help='Similarity threshold for unique compound detection (default: 0.85)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=500,
        help='Batch size for GPU processing (default: 500)'
    )
    parser.add_argument(
        '--feature-table-a',
        type=str,
        default=None,
        help='Optional: Feature table for Group A (for quantitative analysis)'
    )
    parser.add_argument(
        '--feature-table-b',
        type=str,
        default=None,
        help='Optional: Feature table for Group B (for quantitative analysis)'
    )

    args = parser.parse_args()

    # Run analysis
    analyzer = DifferentialAnalyzerGPU(
        embeddings_file=args.embeddings,
        metadata_file=args.metadata,
        output_dir=args.output
    )

    results = analyzer.run(
        similarity_threshold=args.similarity_threshold,
        feature_table_a=args.feature_table_a,
        feature_table_b=args.feature_table_b,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
