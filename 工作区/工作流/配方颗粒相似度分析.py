#!/usr/bin/env python3
"""
配方颗粒相似度分析脚本

功能：
1. 加载DreaMS生成的向量数据
2. 计算不同品种配方颗粒之间的相似度
3. 生成相似度矩阵和可视化结果
4. 生成分析报告，用于更新知识产权登记申报材料

使用方式：
    python 配方颗粒相似度分析.py --emb_dir 向量数据目录 --output_dir 输出目录
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pandas as pd
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="配方颗粒相似度分析脚本")
    parser.add_argument("--emb_dir", required=True, help="DreaMS向量数据目录")
    parser.add_argument("--output_dir", default="./相似度分析结果", help="输出目录")
    parser.add_argument("--n_components", type=int, default=2, help="降维维度")
    parser.add_argument("--figsize", type=int, nargs=2, default=[12, 10], help="图表尺寸")
    return parser.parse_args()


def load_embeddings(emb_dir):
    """加载DreaMS生成的向量数据"""
    embeddings = []
    sample_names = []
    
    # 遍历目录中的所有npz文件
    for file_path in Path(emb_dir).glob("*.npz"):
        try:
            data = np.load(file_path)
            if 'embeddings' in data:
                # 假设每个文件包含一个样品的多个特征的嵌入
                # 取平均值作为样品的整体嵌入
                sample_emb = np.mean(data['embeddings'], axis=0)
                embeddings.append(sample_emb)
                # 从文件名中提取样品名称
                sample_name = file_path.stem
                sample_names.append(sample_name)
                print(f"加载样品: {sample_name}")
        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
    
    if not embeddings:
        raise ValueError("未找到有效的向量数据")
    
    return np.array(embeddings), sample_names


def compute_similarity_matrix(embeddings):
    """计算相似度矩阵"""
    return cosine_similarity(embeddings)


def visualize_similarity_heatmap(similarity_matrix, sample_names, output_dir, figsize):
    """生成相似度热图"""
    plt.figure(figsize=figsize)
    sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", 
                xticklabels=sample_names, yticklabels=sample_names)
    plt.title("配方颗粒相似度热图")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "相似度热图.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"相似度热图已保存到: {output_path}")


def visualize_pca(embeddings, sample_names, output_dir, figsize):
    """PCA降维可视化"""
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)
    
    plt.figure(figsize=figsize)
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    for i, name in enumerate(sample_names):
        plt.annotate(name, (pca_result[i, 0], pca_result[i, 1]))
    plt.title("配方颗粒PCA降维可视化")
    plt.xlabel(f"主成分1 ({pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"主成分2 ({pca.explained_variance_ratio_[1]:.2f})")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "PCA可视化.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"PCA可视化已保存到: {output_path}")


def visualize_tsne(embeddings, sample_names, output_dir, figsize):
    """t-SNE降维可视化"""
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=figsize)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
    for i, name in enumerate(sample_names):
        plt.annotate(name, (tsne_result[i, 0], tsne_result[i, 1]))
    plt.title("配方颗粒t-SNE降维可视化")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "t-SNE可视化.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"t-SNE可视化已保存到: {output_path}")


def visualize_umap(embeddings, sample_names, output_dir, figsize):
    """UMAP降维可视化"""
    reducer = umap.UMAP(random_state=42)
    umap_result = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=figsize)
    plt.scatter(umap_result[:, 0], umap_result[:, 1])
    for i, name in enumerate(sample_names):
        plt.annotate(name, (umap_result[i, 0], umap_result[i, 1]))
    plt.title("配方颗粒UMAP降维可视化")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "UMAP可视化.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"UMAP可视化已保存到: {output_path}")


def generate_analysis_report(similarity_matrix, sample_names, output_dir):
    """生成分析报告"""
    # 计算平均相似度
    upper_tri = np.triu(similarity_matrix, k=1)
    avg_similarity = np.mean(upper_tri[upper_tri > 0])
    
    # 找到最相似和最不相似的样品对
    max_sim = 0
    min_sim = 1
    max_pair = None
    min_pair = None
    
    for i in range(len(sample_names)):
        for j in range(i+1, len(sample_names)):
            sim = similarity_matrix[i, j]
            if sim > max_sim:
                max_sim = sim
                max_pair = (sample_names[i], sample_names[j])
            if sim < min_sim:
                min_sim = sim
                min_pair = (sample_names[i], sample_names[j])
    
    # 生成报告
    report_path = os.path.join(output_dir, "相似度分析报告.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 配方颗粒相似度分析报告\n\n")
        f.write(f"## 分析概览\n\n")
        f.write(f"- 分析样品数量: {len(sample_names)}\n")
        f.write(f"- 平均相似度: {avg_similarity:.4f}\n")
        f.write(f"- 最相似样品对: {max_pair[0]} 与 {max_pair[1]} (相似度: {max_sim:.4f})\n")
        f.write(f"- 最不相似样品对: {min_pair[0]} 与 {min_pair[1]} (相似度: {min_sim:.4f})\n\n")
        
        f.write("## 相似度矩阵\n\n")
        f.write("| 样品名称 | " + " | ".join(sample_names) + " |\n")
        f.write("|---------|" + "|--------" * len(sample_names) + "|\n")
        for i, name in enumerate(sample_names):
            row = [name] + [f"{similarity_matrix[i, j]:.4f}" for j in range(len(sample_names))]
            f.write("| " + " | ".join(row) + " |\n")
        
        f.write("\n## 可视化结果\n\n")
        f.write("### 相似度热图\n\n")
        f.write("![相似度热图](相似度热图.png)\n\n")
        f.write("### PCA降维可视化\n\n")
        f.write("![PCA可视化](PCA可视化.png)\n\n")
        f.write("### t-SNE降维可视化\n\n")
        f.write("![t-SNE可视化](t-SNE可视化.png)\n\n")
        f.write("### UMAP降维可视化\n\n")
        f.write("![UMAP可视化](UMAP可视化.png)\n\n")
        
        f.write("## 分析结论\n\n")
        f.write("1. 配方颗粒样品之间的整体相似度为 {:.4f}，表明不同品种之间具有一定的相似性。\n".format(avg_similarity))
        f.write("2. 最相似的样品对为 {} 与 {}，相似度达到 {:.4f}，说明这两个品种在化学成分上非常接近。\n".format(max_pair[0], max_pair[1], max_sim))
        f.write("3. 最不相似的样品对为 {} 与 {}，相似度为 {:.4f}，表明这两个品种在化学成分上存在较大差异。\n".format(min_pair[0], min_pair[1], min_sim))
        f.write("4. 从降维可视化结果可以看出，样品在特征空间中的分布情况，有助于理解不同品种之间的关系。\n\n")
        
        f.write("## 知识产权登记申报材料更新建议\n\n")
        f.write("1. 在申报材料中添加相似度分析结果，说明不同品种配方颗粒的化学指纹相似性。\n")
        f.write("2. 使用可视化结果作为支持材料，展示配方颗粒的整体特征。\n")
        f.write("3. 基于相似度分析结果，强调配方颗粒的质量一致性和可控性。\n")
        f.write("4. 利用DreaMS向量技术的先进性，提升申报材料的技术含量。\n")
    
    print(f"分析报告已保存到: {report_path}")


def main():
    """主函数"""
    args = parse_arguments()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载向量数据
    print("正在加载向量数据...")
    embeddings, sample_names = load_embeddings(args.emb_dir)
    print(f"成功加载 {len(sample_names)} 个样品的向量数据")
    
    # 计算相似度矩阵
    print("正在计算相似度矩阵...")
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    # 生成可视化结果
    print("正在生成可视化结果...")
    visualize_similarity_heatmap(similarity_matrix, sample_names, args.output_dir, args.figsize)
    visualize_pca(embeddings, sample_names, args.output_dir, args.figsize)
    visualize_tsne(embeddings, sample_names, args.output_dir, args.figsize)
    visualize_umap(embeddings, sample_names, args.output_dir, args.figsize)
    
    # 生成分析报告
    print("正在生成分析报告...")
    generate_analysis_report(similarity_matrix, sample_names, args.output_dir)
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()
