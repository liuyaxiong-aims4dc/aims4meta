#!/usr/bin/env python3
"""
Script: 02_quantitative_differential_analysis.py
Purpose: Quantitative differential analysis based on Progenesis QI output
Author: DreaMS Workflow
Date: 2026-01-26
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add DreaMS to path
sys.path.append('/stor1/aims4dc/aims4dc_scripts/DreaMS')


# ============================================================================
# 可调整参数 (Adjustable Parameters)
# ============================================================================

# 差异化合物筛选参数 (Differential compound filtering parameters)
DEFAULT_P_THRESHOLD = 0.05              # p值阈值 (p-value threshold)
DEFAULT_Q_THRESHOLD = 0.05              # q值阈值 (FDR threshold)
DEFAULT_FOLD_CHANGE_THRESHOLD = 50.0    # 倍数变化阈值 (fold change threshold)
                                        # 要求至少50倍差异才算差异化合物

MIN_SIGNAL_INTENSITY = 5000             # 最小信号强度阈值 (minimum signal intensity)
                                        # 差异化合物必须在至少一组中信号强度 ≥ 此值
                                        # 确保检测到的差异化合物信号足够强，可靠检测

# 说明：
# 1. 差异化合物 = 通过统计检验(p<0.05, q<0.05) + 倍数变化≥50 + 信号强度≥5000
# 2. 特异性标志物 = 差异化合物中，Log2FC = ±∞ 的那些（完全特异性）
# 3. 所有化合物都必须先通过统计检验，才能被认为是差异化合物

# ============================================================================


class QuantitativeDifferentialAnalyzer:
    """Quantitative differential analyzer for Progenesis QI output"""

    def __init__(self, progenesis_file: str, output_dir: str):
        """
        Initialize analyzer

        Args:
            progenesis_file: Path to Progenesis QI CSV file
            output_dir: Output directory for results
        """
        self.progenesis_file = Path(progenesis_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / 'statistics').mkdir(exist_ok=True)
        (self.output_dir / 'visualization').mkdir(exist_ok=True)

        print(f"\n{'='*80}")
        print("QUANTITATIVE DIFFERENTIAL ANALYSIS")
        print(f"{'='*80}\n")
        print(f"Input file: {self.progenesis_file}")
        print(f"Output directory: {self.output_dir}\n")

        # Load and parse data
        self.df = None
        self.group_a_cols = []
        self.group_b_cols = []
        self.load_progenesis_data()

    def load_progenesis_data(self):
        """Load and parse Progenesis QI CSV file with multi-row header"""
        print("Loading Progenesis QI data...")

        # Read the file with multi-row header
        # Row 0: "Normalised abundance" / "Raw abundance"
        # Row 1: Group labels (ZHENGPIN / WEIPIN)
        # Row 2: Column names

        # First, read the first 3 rows to understand the structure
        header_df = pd.read_csv(self.progenesis_file, nrows=3, header=None)

        # Read the full data starting from row 2 (0-indexed)
        self.df = pd.read_csv(self.progenesis_file, skiprows=2)

        print(f"  Total compounds: {len(self.df)}")
        print(f"  Total columns: {len(self.df.columns)}")

        # Identify sample columns by reading row 1 (group labels)
        group_labels = pd.read_csv(self.progenesis_file, skiprows=1, nrows=1, header=None).iloc[0]

        # Find columns that belong to each group
        # We need to match column names with group labels
        # The sample columns start after the metadata columns

        # Find where sample data starts (after "Minimum CV%")
        metadata_cols = ['Compound', 'Neutral mass (Da)', 'm/z', 'Charge',
                        'Retention time (min)', 'CCS (angstrom^2)',
                        'Chromatographic peak width (min)', 'Identifications',
                        'Anova (p)', 'q Value', 'Max Fold Change',
                        'Highest Mean', 'Lowest Mean', 'Isotope Distribution',
                        'Maximum Abundance', 'Minimum CV%']

        # Get sample columns (those not in metadata)
        sample_cols = [col for col in self.df.columns if col not in metadata_cols
                      and not col.startswith('Unnamed')]

        print(f"  Sample columns found: {len(sample_cols)}")

        # Parse group labels to identify which samples belong to which group
        # Group labels are in row 1, we need to map them to column indices
        for i, col_name in enumerate(self.df.columns):
            if col_name in sample_cols:
                # Find the corresponding group label
                group_label = str(group_labels.iloc[i]).strip().upper()

                if 'ZHENGPIN' in group_label:
                    self.group_a_cols.append(col_name)
                elif 'WEIPIN' in group_label:
                    self.group_b_cols.append(col_name)

        print(f"  Group A (Authentic) samples: {len(self.group_a_cols)}")
        print(f"  Group B (Counterfeit) samples: {len(self.group_b_cols)}")

        # Clean up data
        # Remove rows with all NaN values in sample columns
        sample_data = self.df[self.group_a_cols + self.group_b_cols]
        valid_rows = sample_data.notna().any(axis=1)
        self.df = self.df[valid_rows].reset_index(drop=True)

        print(f"  Valid compounds after filtering: {len(self.df)}\n")

    def filter_differential_compounds(self, p_threshold: float = DEFAULT_P_THRESHOLD,
                                     q_threshold: float = DEFAULT_Q_THRESHOLD,
                                     fold_change_threshold: float = DEFAULT_FOLD_CHANGE_THRESHOLD,
                                     min_signal_intensity: float = MIN_SIGNAL_INTENSITY):
        """
        Filter compounds based on statistical significance and intensity criteria

        筛选逻辑：
        1. 统计显著性：p < 0.05, q < 0.05
        2. 倍数变化：Fold Change ≥ 50
        3. 信号强度：至少一组的平均信号强度 ≥ 5000
        4. 特异性标志物：满足以上条件且 Log2FC = ±∞（完全特异性）

        Args:
            p_threshold: Maximum p-value for significance
            q_threshold: Maximum q-value (FDR) for significance
            fold_change_threshold: Minimum fold change
            min_signal_intensity: Minimum signal intensity

        Returns:
            DataFrame with differential compounds
        """
        print(f"\n{'='*80}")
        print("FILTERING DIFFERENTIAL COMPOUNDS")
        print(f"{'='*80}\n")
        print(f"筛选标准 (Filtering Criteria):")
        print(f"  1. 统计显著性: p-value < {p_threshold}, q-value < {q_threshold}")
        print(f"  2. 倍数变化: Fold Change ≥ {fold_change_threshold}")
        print(f"  3. 信号强度: 至少一组平均强度 ≥ {min_signal_intensity}")
        print(f"  4. 特异性标志物: 满足以上条件且 Log2FC = ±∞\n")

        # Extract statistical columns
        df_filtered = self.df.copy()

        # Convert statistical columns to numeric
        df_filtered['Anova (p)'] = pd.to_numeric(df_filtered['Anova (p)'], errors='coerce')
        df_filtered['q Value'] = pd.to_numeric(df_filtered['q Value'], errors='coerce')
        df_filtered['Max Fold Change'] = pd.to_numeric(df_filtered['Max Fold Change'], errors='coerce')

        # Step 1: Apply statistical filters
        mask_stats = (
            (df_filtered['Anova (p)'] < p_threshold) &
            (df_filtered['q Value'] < q_threshold) &
            (df_filtered['Max Fold Change'] > fold_change_threshold)
        )

        df_diff = df_filtered[mask_stats].copy()
        print(f"通过统计筛选的化合物: {len(df_diff)}")

        # Step 2: Calculate group means
        df_diff['Mean_A'] = df_diff[self.group_a_cols].mean(axis=1)
        df_diff['Mean_B'] = df_diff[self.group_b_cols].mean(axis=1)

        # Step 3: Apply signal intensity filter
        # At least one group must have mean intensity >= min_signal_intensity
        mask_intensity = (df_diff['Mean_A'] >= min_signal_intensity) | (df_diff['Mean_B'] >= min_signal_intensity)
        df_diff = df_diff[mask_intensity].copy()

        print(f"通过信号强度筛选的化合物: {len(df_diff)}")
        print(f"  ({len(df_diff)/len(self.df)*100:.1f}% of total compounds)\n")

        # Step 4: Calculate Log2FC and identify specificity
        with np.errstate(divide='ignore', invalid='ignore'):
            df_diff['Log2FC'] = np.log2(df_diff['Mean_A'] / df_diff['Mean_B'])

        df_diff['Direction'] = df_diff['Log2FC'].apply(
            lambda x: 'Authentic-enriched' if x > 0 else 'Counterfeit-enriched'
        )

        # Identify specificity markers (Log2FC = ±∞)
        df_diff['Is_Specific'] = False
        df_diff['Specificity_Type'] = ''

        # Authentic-specific (Log2FC = +∞, present only in Group A)
        mask_auth_specific = np.isinf(df_diff['Log2FC']) & (df_diff['Log2FC'] > 0)
        df_diff.loc[mask_auth_specific, 'Is_Specific'] = True
        df_diff.loc[mask_auth_specific, 'Specificity_Type'] = 'Authentic-specific'

        # Counterfeit-specific (Log2FC = -∞, present only in Group B)
        mask_count_specific = np.isinf(df_diff['Log2FC']) & (df_diff['Log2FC'] < 0)
        df_diff.loc[mask_count_specific, 'Is_Specific'] = True
        df_diff.loc[mask_count_specific, 'Specificity_Type'] = 'Counterfeit-specific'

        # Count results
        n_authentic = (df_diff['Direction'] == 'Authentic-enriched').sum()
        n_counterfeit = (df_diff['Direction'] == 'Counterfeit-enriched').sum()
        n_auth_specific = (df_diff['Specificity_Type'] == 'Authentic-specific').sum()
        n_count_specific = (df_diff['Specificity_Type'] == 'Counterfeit-specific').sum()

        print(f"差异化合物分类 (Differential Compound Classification):")
        print(f"  - 正品富集 (Authentic-enriched): {n_authentic}")
        print(f"  - 伪品富集 (Counterfeit-enriched): {n_counterfeit}\n")

        print(f"特异性标志物 (Specificity Markers - Log2FC = ±∞):")
        print(f"  - 正品特异性 (Authentic-specific): {n_auth_specific}")
        print(f"  - 伪品特异性 (Counterfeit-specific): {n_count_specific}")
        print(f"  - 总计 (Total): {n_auth_specific + n_count_specific}\n")

        # Save results
        output_file = self.output_dir / 'statistics' / 'differential_compounds.csv'
        df_diff.to_csv(output_file, index=False)
        print(f"Results saved: {output_file}\n")

        return df_diff

    def plot_volcano(self, df_diff, p_threshold: float = 0.05,
                    fold_change_threshold: float = 2.0):
        """
        Create volcano plot

        Args:
            df_diff: DataFrame with differential compounds
            p_threshold: p-value threshold for coloring
            fold_change_threshold: Fold change threshold for coloring
        """
        print(f"\n{'='*80}")
        print("GENERATING VOLCANO PLOT")
        print(f"{'='*80}\n")

        # Prepare data for all compounds
        df_plot = self.df.copy()
        df_plot['Anova (p)'] = pd.to_numeric(df_plot['Anova (p)'], errors='coerce')
        df_plot['Max Fold Change'] = pd.to_numeric(df_plot['Max Fold Change'], errors='coerce')

        # Calculate log2 fold change and -log10(p-value)
        df_plot['Mean_A'] = df_plot[self.group_a_cols].mean(axis=1)
        df_plot['Mean_B'] = df_plot[self.group_b_cols].mean(axis=1)
        df_plot['Log2FC'] = np.log2(df_plot['Mean_A'] / df_plot['Mean_B'])
        df_plot['NegLog10P'] = -np.log10(df_plot['Anova (p)'] + 1e-300)  # Add small value to avoid log(0)

        # Remove infinite values
        df_plot = df_plot[np.isfinite(df_plot['Log2FC']) & np.isfinite(df_plot['NegLog10P'])]

        # Determine significance
        log2fc_threshold = np.log2(fold_change_threshold)
        neglog10p_threshold = -np.log10(p_threshold)

        df_plot['Significant'] = 'Not significant'
        df_plot.loc[
            (df_plot['NegLog10P'] > neglog10p_threshold) &
            (df_plot['Log2FC'] > log2fc_threshold),
            'Significant'
        ] = 'Authentic-enriched'
        df_plot.loc[
            (df_plot['NegLog10P'] > neglog10p_threshold) &
            (df_plot['Log2FC'] < -log2fc_threshold),
            'Significant'
        ] = 'Counterfeit-enriched'

        # Create plot
        plt.figure(figsize=(12, 8))

        # Plot non-significant points
        ns = df_plot[df_plot['Significant'] == 'Not significant']
        plt.scatter(ns['Log2FC'], ns['NegLog10P'], c='lightgray', alpha=0.5, s=20, label='Not significant')

        # Plot authentic-enriched
        auth = df_plot[df_plot['Significant'] == 'Authentic-enriched']
        plt.scatter(auth['Log2FC'], auth['NegLog10P'], c='#2E86DE', alpha=0.7, s=40, label='Authentic-enriched')

        # Plot counterfeit-enriched
        count = df_plot[df_plot['Significant'] == 'Counterfeit-enriched']
        plt.scatter(count['Log2FC'], count['NegLog10P'], c='#EE5A6F', alpha=0.7, s=40, label='Counterfeit-enriched')

        # Add threshold lines
        plt.axhline(y=neglog10p_threshold, color='black', linestyle='--', linewidth=1, alpha=0.5)
        plt.axvline(x=log2fc_threshold, color='black', linestyle='--', linewidth=1, alpha=0.5)
        plt.axvline(x=-log2fc_threshold, color='black', linestyle='--', linewidth=1, alpha=0.5)

        plt.xlabel('Log2 Fold Change (Authentic / Counterfeit)', fontsize=12)
        plt.ylabel('-Log10(p-value)', fontsize=12)
        plt.title('Volcano Plot: Authentic vs Counterfeit', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / 'visualization' / 'volcano_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Volcano plot saved: {plot_path}\n")
        print(f"Statistics:")
        print(f"  - Total compounds plotted: {len(df_plot)}")
        print(f"  - Authentic-enriched: {len(auth)}")
        print(f"  - Counterfeit-enriched: {len(count)}")
        print(f"  - Not significant: {len(ns)}\n")

    def generate_summary_report(self, df_diff):
        """Generate summary report"""
        print(f"\n{'='*80}")
        print("GENERATING SUMMARY REPORT")
        print(f"{'='*80}\n")

        report_path = self.output_dir / 'QUANTITATIVE_DIFFERENTIAL_REPORT.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("QUANTITATIVE DIFFERENTIAL ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")

            f.write("## Dataset Overview\n")
            f.write(f"Total compounds: {len(self.df)}\n")
            f.write(f"Group A (Authentic) samples: {len(self.group_a_cols)}\n")
            f.write(f"Group B (Counterfeit) samples: {len(self.group_b_cols)}\n\n")

            f.write("## Differential Compounds\n")
            f.write(f"Total differential compounds: {len(df_diff)}\n")

            if len(df_diff) > 0:
                n_authentic = (df_diff['Direction'] == 'Authentic-enriched').sum()
                n_counterfeit = (df_diff['Direction'] == 'Counterfeit-enriched').sum()

                f.write(f"  - Authentic-enriched: {n_authentic}\n")
                f.write(f"  - Counterfeit-enriched: {n_counterfeit}\n\n")

                # Top authentic-enriched compounds
                df_auth = df_diff[df_diff['Direction'] == 'Authentic-enriched'].sort_values('Log2FC', ascending=False)
                if len(df_auth) > 0:
                    f.write("## Top 10 Authentic-Enriched Compounds\n")
                    for i, (idx, row) in enumerate(df_auth.head(10).iterrows(), 1):
                        f.write(f"{i}. {row['Compound']}\n")
                        f.write(f"   m/z: {row['m/z']:.4f}\n")
                        f.write(f"   RT: {row['Retention time (min)']:.2f} min\n")
                        f.write(f"   Log2FC: {row['Log2FC']:.2f}\n")
                        f.write(f"   p-value: {row['Anova (p)']:.2e}\n\n")

                # Top counterfeit-enriched compounds
                df_count = df_diff[df_diff['Direction'] == 'Counterfeit-enriched'].sort_values('Log2FC', ascending=True)
                if len(df_count) > 0:
                    f.write("## Top 10 Counterfeit-Enriched Compounds\n")
                    for i, (idx, row) in enumerate(df_count.head(10).iterrows(), 1):
                        f.write(f"{i}. {row['Compound']}\n")
                        f.write(f"   m/z: {row['m/z']:.4f}\n")
                        f.write(f"   RT: {row['Retention time (min)']:.2f} min\n")
                        f.write(f"   Log2FC: {row['Log2FC']:.2f}\n")
                        f.write(f"   p-value: {row['Anova (p)']:.2e}\n\n")

        print(f"Report saved: {report_path}\n")

    def run(self, p_threshold: float = 0.05, q_threshold: float = 0.05,
            fold_change_threshold: float = 2.0):
        """
        Run complete quantitative differential analysis workflow

        Args:
            p_threshold: Maximum p-value for significance
            q_threshold: Maximum q-value (FDR) for significance
            fold_change_threshold: Minimum fold change

        Returns:
            DataFrame with differential compounds
        """
        print(f"\n{'='*80}")
        print("QUANTITATIVE DIFFERENTIAL ANALYSIS WORKFLOW")
        print(f"{'='*80}\n")

        # Step 1: Filter differential compounds
        print("[Step 1/3] Filtering differential compounds...")
        df_diff = self.filter_differential_compounds(
            p_threshold=p_threshold,
            q_threshold=q_threshold,
            fold_change_threshold=fold_change_threshold
        )

        # Step 2: Generate volcano plot
        print("[Step 2/3] Generating volcano plot...")
        self.plot_volcano(df_diff, p_threshold=p_threshold,
                         fold_change_threshold=fold_change_threshold)

        # Step 3: Generate summary report
        print("[Step 3/3] Generating summary report...")
        self.generate_summary_report(df_diff)

        # Final summary
        print(f"\n{'='*80}")
        print("✓ Quantitative differential analysis complete!")
        print(f"{'='*80}\n")
        print("Output files:")
        print(f"  - Differential compounds: {self.output_dir / 'statistics' / 'differential_compounds.csv'}")
        print(f"  - Volcano plot: {self.output_dir / 'visualization' / 'volcano_plot.png'}")
        print(f"  - Summary report: {self.output_dir / 'QUANTITATIVE_DIFFERENTIAL_REPORT.txt'}")
        print()

        return df_diff


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Quantitative differential analysis based on Progenesis QI output'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to Progenesis QI CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./quantitative_differential_results',
        help='Output directory'
    )
    parser.add_argument(
        '--p-threshold',
        type=float,
        default=0.05,
        help='Maximum p-value for significance (default: 0.05)'
    )
    parser.add_argument(
        '--q-threshold',
        type=float,
        default=0.05,
        help='Maximum q-value (FDR) for significance (default: 0.05)'
    )
    parser.add_argument(
        '--fold-change',
        type=float,
        default=50.0,
        help='Minimum fold change (default: 50.0)'
    )

    args = parser.parse_args()

    # Run analysis
    analyzer = QuantitativeDifferentialAnalyzer(
        progenesis_file=args.input,
        output_dir=args.output
    )

    results = analyzer.run(
        p_threshold=args.p_threshold,
        q_threshold=args.q_threshold,
        fold_change_threshold=args.fold_change
    )


if __name__ == '__main__':
    main()
