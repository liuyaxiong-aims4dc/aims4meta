#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一对比评估脚本
对比MS-BART、DiffMS、MSFlow三个模型的性能
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_dir, model_name):
    """加载模型结果"""
    stats_file = os.path.join(results_dir, f'{model_name}_stats.csv')
    if os.path.exists(stats_file):
        return pd.read_csv(stats_file)
    return None


def main():
    parser = argparse.ArgumentParser(description='对比评估三个模型')
    parser.add_argument('--ms_bart_results', type=str,
                       default='/stor3/AIMS4Meta/工作区/工作流/多层鉴定/L5_从头鉴定/MS-BART/results',
                       help='MS-BART结果目录')
    parser.add_argument('--diffms_results', type=str,
                       default='/stor3/AIMS4Meta/工作区/工作流/多层鉴定/L5_从头鉴定/DiffMS/results',
                       help='DiffMS结果目录')
    parser.add_argument('--msflow_results', type=str,
                       default='/stor3/AIMS4Meta/工作区/工作流/多层鉴定/L5_从头鉴定/MSFlow/results',
                       help='MSFlow结果目录')
    parser.add_argument('--output_dir', type=str,
                       default='/stor3/AIMS4Meta/工作区/工作流/多层鉴定/L5_从头鉴定/results',
                       help='输出目录')

    args = parser.parse_args()

    print("=" * 80)
    print("模型对比评估")
    print("=" * 80)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载结果
    print("\n加载模型结果...")
    ms_bart_stats = load_results(args.ms_bart_results, 'msbart')
    diffms_stats = load_results(args.diffms_results, 'diffms')
    msflow_stats = load_results(args.msflow_results, 'msflow')

    # 汇总结果
    all_stats = []

    if ms_bart_stats is not None:
        ms_bart_stats['model'] = 'MS-BART'
        all_stats.append(ms_bart_stats)
        print("  MS-BART: ✓")
    else:
        print("  MS-BART: ✗ (未找到结果)")

    if diffms_stats is not None:
        diffms_stats['model'] = 'DiffMS'
        all_stats.append(diffms_stats)
        print("  DiffMS: ✓")
    else:
        print("  DiffMS: ✗ (未找到结果)")

    if msflow_stats is not None:
        msflow_stats['model'] = 'MSFlow'
        all_stats.append(msflow_stats)
        print("  MSFlow: ✓")
    else:
        print("  MSFlow: ✗ (未找到结果)")

    if not all_stats:
        print("\n错误: 没有找到任何模型结果!")
        print("请先运行各个模型的评估脚本")
        return

    # 合并结果
    comparison_df = pd.concat(all_stats, ignore_index=True)

    # 选择关键指标
    metrics = [
        'model',
        'total',
        'valid',
        'validity_rate',
        'avg_tanimoto',
        'median_tanimoto',
        'top1_accuracy',
        'top5_accuracy',
        'top10_accuracy',
        'exact_match'
    ]

    comparison_df = comparison_df[metrics]

    # 打印对比表
    print("\n" + "=" * 80)
    print("性能对比")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    # 保存对比结果
    comparison_file = os.path.join(args.output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n对比结果保存到: {comparison_file}")

    # 创建可视化
    print("\n创建可视化...")

    # 设置样式
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    # 1. Tanimoto相似度对比
    ax = axes[0, 0]
    models = comparison_df['model'].tolist()
    avg_tanimoto = comparison_df['avg_tanimoto'].tolist()
    bars = ax.bar(models, avg_tanimoto, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_ylabel('Average Tanimoto Similarity')
    ax.set_title('Average Tanimoto Similarity')
    ax.set_ylim([0, 1])
    for bar, val in zip(bars, avg_tanimoto):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')

    # 2. Top-K准确率对比
    ax = axes[0, 1]
    x = np.arange(len(models))
    width = 0.25
    top1 = comparison_df['top1_accuracy'].tolist()
    top5 = comparison_df['top5_accuracy'].tolist()
    top10 = comparison_df['top10_accuracy'].tolist()

    ax.bar(x - width, top1, width, label='Top-1 (≥0.9)', color='#3498db')
    ax.bar(x, top5, width, label='Top-5 (≥0.7)', color='#e74c3c')
    ax.bar(x + width, top10, width, label='Top-10 (≥0.5)', color='#2ecc71')
    ax.set_ylabel('Accuracy')
    ax.set_title('Top-K Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim([0, 1])

    # 3. 有效性率对比
    ax = axes[0, 2]
    validity = comparison_df['validity_rate'].tolist()
    bars = ax.bar(models, validity, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_ylabel('Validity Rate')
    ax.set_title('SMILES Validity Rate')
    ax.set_ylim([0, 1])
    for bar, val in zip(bars, validity):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2%}', ha='center', va='bottom')

    # 4. 完全匹配数对比
    ax = axes[1, 0]
    exact_match = comparison_df['exact_match'].tolist()
    bars = ax.bar(models, exact_match, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_ylabel('Exact Match Count')
    ax.set_title('Exact Match (Tanimoto=1.0)')
    for bar, val in zip(bars, exact_match):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(int(val)), ha='center', va='bottom')

    # 5. 中位数Tanimoto对比
    ax = axes[1, 1]
    median_tanimoto = comparison_df['median_tanimoto'].tolist()
    bars = ax.bar(models, median_tanimoto, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_ylabel('Median Tanimoto Similarity')
    ax.set_title('Median Tanimoto Similarity')
    ax.set_ylim([0, 1])
    for bar, val in zip(bars, median_tanimoto):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')

    # 6. 综合评分雷达图
    ax = axes[1, 2]
    categories = ['Validity', 'Avg Tanimoto', 'Top-1', 'Top-5', 'Top-10']
    num_vars = len(categories)

    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    plt.xticks(angles[:-1], categories)
    ax.set_ylim(0, 1)

    for idx, model in enumerate(models):
        values = [
            comparison_df[comparison_df['model'] == model]['validity_rate'].values[0],
            comparison_df[comparison_df['model'] == model]['avg_tanimoto'].values[0],
            comparison_df[comparison_df['model'] == model]['top1_accuracy'].values[0],
            comparison_df[comparison_df['model'] == model]['top5_accuracy'].values[0],
            comparison_df[comparison_df['model'] == model]['top10_accuracy'].values[0]
        ]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)

    ax.set_title('Comprehensive Performance')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    # 保存图表
    plot_file = os.path.join(args.output_dir, 'model_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  图表保存到: {plot_file}")

    # 创建总结报告
    print("\n" + "=" * 80)
    print("总结报告")
    print("=" * 80)

    # 找出最佳模型
    best_avg_tanimoto = comparison_df.loc[comparison_df['avg_tanimoto'].idxmax()]
    best_top1 = comparison_df.loc[comparison_df['top1_accuracy'].idxmax()]
    best_validity = comparison_df.loc[comparison_df['validity_rate'].idxmax()]

    print(f"\n最佳平均Tanimoto相似度: {best_avg_tanimoto['model']} ({best_avg_tanimoto['avg_tanimoto']:.4f})")
    print(f"最佳Top-1准确率: {best_top1['model']} ({best_top1['top1_accuracy']:.2%})")
    print(f"最佳有效性率: {best_validity['model']} ({best_validity['validity_rate']:.2%})")

    # 保存总结报告
    summary = {
        'best_avg_tanimoto_model': best_avg_tanimoto['model'],
        'best_avg_tanimoto_value': best_avg_tanimoto['avg_tanimoto'],
        'best_top1_model': best_top1['model'],
        'best_top1_value': best_top1['top1_accuracy'],
        'best_validity_model': best_validity['model'],
        'best_validity_value': best_validity['validity_rate']
    }

    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(args.output_dir, 'comparison_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\n总结报告保存到: {summary_file}")

    print("\n对比评估完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
