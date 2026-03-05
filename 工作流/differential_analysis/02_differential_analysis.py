#!/usr/bin/env python3
"""
Script: 02_differential_analysis.py
Purpose: Statistical analysis to identify differential compounds between Group A and B
Author: DreaMS Workflow
Date: 2026-01-10
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Add DreaMS to path
sys.path.append('/stor1/aims4dc/aims4dc_scripts/DreaMS')


class DifferentialAnalyzer:
    """Analyze differential compounds between two groups"""

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

        # Load data
        print("Loading embeddings and metadata...")
        self.embeddings = np.load(embeddings_file)['embeddings']
        self.metadata = pd.read_csv(metadata_file)

        print(f"  Embeddings shape: {self.embeddings.shape}")
        print(f"  Metadata rows: {len(self.metadata)}")

        # Group indices
        self.group_a_idx = self.metadata[self.metadata['group'] == 'A'].index.tolist()
        self.group_b_idx = self.metadata[self.metadata['group'] == 'B'].index.tolist()

        print(f"  Group A indices: {len(self.group_a_idx)}")
        print(f"  Group B indices: {len(self.group_b_idx)}")

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

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        plt.title('PCA: Authentic vs Counterfeit TCM', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, frameon=True, shadow=True)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()

        pca_plot_path = self.output_dir / 'visualization' / 'pca_plot.png'
        plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  PC1 explained variance: {pca.explained_variance_ratio_[0]:.2%}")
        print(f"  PC2 explained variance: {pca.explained_variance_ratio_[1]:.2%}")
        print(f"  Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        print(f"  PCA plot saved: {pca_plot_path}")

        return embeddings_pca, pca

    def find_unique_compounds(self, similarity_threshold: float = 0.85):
        """
        Find compounds unique to each group

        Strategy: High within-group similarity, low between-group similarity

        Args:
            similarity_threshold: Minimum within-group similarity

        Returns:
            Tuple of (unique_A_df, unique_B_df)
        """
        print("\n=== Finding Unique Compounds ===")
        print(f"Similarity threshold: {similarity_threshold}")

        unique_a = []  # Authentic-specific markers
        unique_b = []  # Counterfeit-specific markers

        # Analyze each spectrum
        for i in range(len(self.embeddings)):
            emb = self.embeddings[i]
            group = self.metadata.loc[i, 'group']

            # Calculate average similarity to each group
            sim_to_a = np.mean([
                1 - cosine(emb, self.embeddings[j])
                for j in self.group_a_idx if j != i
            ]) if len(self.group_a_idx) > 1 else 0

            sim_to_b = np.mean([
                1 - cosine(emb, self.embeddings[j])
                for j in self.group_b_idx if j != i
            ]) if len(self.group_b_idx) > 1 else 0

            # Identify group-specific compounds
            if group == 'A':
                if sim_to_a > similarity_threshold and sim_to_b < 0.7:
                    unique_a.append({
                        'index': i,
                        'name': self.metadata.loc[i, 'name'],
                        'sample_id': self.metadata.loc[i, 'sample_id'],
                        'precursor_mz': self.metadata.loc[i, 'precursor_mz'],
                        'retention_time': self.metadata.loc[i, 'retention_time'],
                        'sim_to_own_group': sim_to_a,
                        'sim_to_other_group': sim_to_b,
                        'specificity': sim_to_a - sim_to_b
                    })

            elif group == 'B':
                if sim_to_b > similarity_threshold and sim_to_a < 0.7:
                    unique_b.append({
                        'index': i,
                        'name': self.metadata.loc[i, 'name'],
                        'sample_id': self.metadata.loc[i, 'sample_id'],
                        'precursor_mz': self.metadata.loc[i, 'precursor_mz'],
                        'retention_time': self.metadata.loc[i, 'retention_time'],
                        'sim_to_own_group': sim_to_b,
                        'sim_to_other_group': sim_to_a,
                        'specificity': sim_to_b - sim_to_a
                    })

        # Convert to DataFrames and sort by specificity
        df_unique_a = pd.DataFrame(unique_a).sort_values('specificity', ascending=False)
        df_unique_b = pd.DataFrame(unique_b).sort_values('specificity', ascending=False)

        # Save results
        df_unique_a.to_csv(
            self.output_dir / 'statistics' / 'unique_to_authentic.csv',
            index=False
        )
        df_unique_b.to_csv(
            self.output_dir / 'statistics' / 'unique_to_counterfeit.csv',
            index=False
        )

        print(f"  Authentic-specific compounds: {len(unique_a)}")
        print(f"  Counterfeit-specific compounds: {len(unique_b)}")

        if len(unique_a) > 0:
            print(f"\n  Top 3 Authentic markers:")
            for i, row in df_unique_a.head(3).iterrows():
                print(f"    - {row['name']} (m/z {row['precursor_mz']:.4f}, specificity: {row['specificity']:.3f})")

        if len(unique_b) > 0:
            print(f"\n  Top 3 Counterfeit markers:")
            for i, row in df_unique_b.head(3).iterrows():
                print(f"    - {row['name']} (m/z {row['precursor_mz']:.4f}, specificity: {row['specificity']:.3f})")

        return df_unique_a, df_unique_b

    def quantitative_difference(self, feature_table_a: str = None, feature_table_b: str = None,
                               fold_change_threshold: float = 2.0, pvalue_threshold: float = 0.05):
        """
        Quantitative differential analysis

        Args:
            feature_table_a: Path to Group A feature table (optional)
            feature_table_b: Path to Group B feature table (optional)
            fold_change_threshold: Minimum fold change
            pvalue_threshold: Maximum p-value

        Returns:
            DataFrame of differential compounds
        """
        print("\n=== Quantitative Differential Analysis ===")

        if feature_table_a is None or feature_table_b is None:
            print("  ⚠ Feature tables not provided, skipping quantitative analysis")
            print("  Note: This requires peak area/intensity information from feature tables")
            return None

        # Check if files exist
        if not Path(feature_table_a).exists() or not Path(feature_table_b).exists():
            print(f"  ⚠ Feature table files not found, skipping")
            return None

        print(f"  Feature table A: {feature_table_a}")
        print(f"  Feature table B: {feature_table_b}")
        print(f"  Fold change threshold: {fold_change_threshold}")
        print(f"  P-value threshold: {pvalue_threshold}")

        # Load feature tables
        ft_a = pd.read_csv(feature_table_a)
        ft_b = pd.read_csv(feature_table_b)

        # TODO: Implement feature matching and statistical testing
        # This requires knowing the exact format of your feature tables
        print("  ⚠ Quantitative analysis implementation requires feature table format specification")

        return None

    def plot_volcano(self, df_diff: pd.DataFrame):
        """Generate volcano plot"""
        if df_diff is None or len(df_diff) == 0:
            return

        plt.figure(figsize=(12, 8))

        # All points
        plt.scatter(
            df_diff['log2_fc'],
            -np.log10(df_diff['p_value']),
            c='gray',
            alpha=0.5,
            s=30,
            label='Not significant'
        )

        # Significant in A
        up_a = df_diff[df_diff['direction'] == 'upregulated_in_A']
        if len(up_a) > 0:
            plt.scatter(
                up_a['log2_fc'],
                -np.log10(up_a['p_value']),
                c='#2E86DE',
                alpha=0.8,
                s=60,
                label='Enriched in Authentic',
                edgecolors='white',
                linewidth=0.5
            )

        # Significant in B
        up_b = df_diff[df_diff['direction'] == 'upregulated_in_B']
        if len(up_b) > 0:
            plt.scatter(
                up_b['log2_fc'],
                -np.log10(up_b['p_value']),
                c='#EE5A6F',
                alpha=0.8,
                s=60,
                label='Enriched in Counterfeit',
                edgecolors='white',
                linewidth=0.5
            )

        # Threshold lines
        plt.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5, linewidth=1)
        plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        plt.axvline(x=-1, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        plt.xlabel('log₂(Fold Change)', fontsize=12)
        plt.ylabel('-log₁₀(p-value)', fontsize=12)
        plt.title('Volcano Plot: Authentic vs Counterfeit TCM', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, frameon=True, shadow=True)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()

        volcano_path = self.output_dir / 'visualization' / 'volcano_plot.png'
        plt.savefig(volcano_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Volcano plot saved: {volcano_path}")

    def generate_summary_report(self, unique_a: pd.DataFrame, unique_b: pd.DataFrame):
        """Generate summary statistics report"""
        print("\n=== Generating Summary Report ===")

        report = []
        report.append("=" * 80)
        report.append("Differential Compound Analysis Summary")
        report.append("=" * 80)
        report.append("")
        report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("## Dataset Overview")
        report.append(f"  Total spectra: {len(self.metadata)}")
        report.append(f"  Group A (Authentic) spectra: {len(self.group_a_idx)}")
        report.append(f"  Group B (Counterfeit) spectra: {len(self.group_b_idx)}")
        report.append("")
        report.append("## Differential Compounds")
        report.append(f"  Authentic-specific compounds: {len(unique_a)}")
        report.append(f"  Counterfeit-specific compounds: {len(unique_b)}")
        report.append("")

        if len(unique_a) > 0:
            report.append("## Top 10 Authentic Markers")
            report.append("  (Compounds found predominantly in authentic samples)")
            report.append("")
            for i, (idx, row) in enumerate(unique_a.head(10).iterrows(), 1):
                report.append(f"  {i}. {row['name']}")
                report.append(f"     m/z: {row['precursor_mz']:.4f}")
                report.append(f"     RT: {row['retention_time']:.2f} min")
                report.append(f"     Specificity score: {row['specificity']:.3f}")
                report.append(f"     Similarity to authentic: {row['sim_to_own_group']:.3f}")
                report.append(f"     Similarity to counterfeit: {row['sim_to_other_group']:.3f}")
                report.append("")

        if len(unique_b) > 0:
            report.append("## Top 10 Counterfeit Markers")
            report.append("  (Compounds found predominantly in counterfeit samples)")
            report.append("")
            for i, (idx, row) in enumerate(unique_b.head(10).iterrows(), 1):
                report.append(f"  {i}. {row['name']}")
                report.append(f"     m/z: {row['precursor_mz']:.4f}")
                report.append(f"     RT: {row['retention_time']:.2f} min")
                report.append(f"     Specificity score: {row['specificity']:.3f}")
                report.append(f"     Similarity to counterfeit: {row['sim_to_own_group']:.3f}")
                report.append(f"     Similarity to authentic: {row['sim_to_other_group']:.3f}")
                report.append("")

        report.append("## Recommendations")
        report.append("")
        report.append("For authentication testing, use the following markers:")
        report.append("")
        report.append("Positive indicators (should be present in authentic):")
        for i, (idx, row) in enumerate(unique_a.head(3).iterrows(), 1):
            report.append(f"  {i}. m/z {row['precursor_mz']:.4f} (RT: {row['retention_time']:.2f} min)")
        report.append("")
        report.append("Negative indicators (should NOT be present in authentic):")
        for i, (idx, row) in enumerate(unique_b.head(3).iterrows(), 1):
            report.append(f"  {i}. m/z {row['precursor_mz']:.4f} (RT: {row['retention_time']:.2f} min)")
        report.append("")
        report.append("=" * 80)

        # Save report
        report_text = "\n".join(report)
        report_path = self.output_dir / 'DIFFERENTIAL_SUMMARY_REPORT.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"  Summary report saved: {report_path}")
        print("\n" + report_text)

    def run(self, similarity_threshold: float = 0.85,
            feature_table_a: str = None, feature_table_b: str = None):
        """
        Run complete differential analysis workflow

        Args:
            similarity_threshold: Threshold for unique compound detection
            feature_table_a: Optional feature table for Group A
            feature_table_b: Optional feature table for Group B
        """
        print("=" * 80)
        print("DreaMS Differential Analysis - Step 2: Statistical Analysis")
        print("=" * 80)

        # Step 1: PCA analysis
        print("\n[Step 1/4] PCA dimensionality reduction...")
        pca_coords, pca_model = self.pca_analysis()

        # Step 2: Find unique compounds
        print("\n[Step 2/4] Identifying group-specific compounds...")
        unique_a, unique_b = self.find_unique_compounds(similarity_threshold)

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
        description='Differential compound analysis between two groups'
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
    analyzer = DifferentialAnalyzer(
        embeddings_file=args.embeddings,
        metadata_file=args.metadata,
        output_dir=args.output
    )

    results = analyzer.run(
        similarity_threshold=args.similarity_threshold,
        feature_table_a=args.feature_table_a,
        feature_table_b=args.feature_table_b
    )


if __name__ == '__main__':
    main()
