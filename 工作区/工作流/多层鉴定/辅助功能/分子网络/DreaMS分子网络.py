#!/usr/bin/env python3
"""
L4: DreaMS 分子网络（纯样品间 Embedding 互连）

策略：
- 不依赖任何数据库（真实库或模拟库）
- 对所有样品间两两计算 embedding cosine 相似度
- 相似度 ≥ 阈值的样品对连边，构建分子网络
- 连通分量聚类 → 标注结构家族
- 输出网络边表 + 节点属性表

输入：样品 embedding（来自L1向量化）+ 样品 MSP 元数据
输出：L4_network_edges.csv + L4_network_nodes.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# GPU 加速支持
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DREAMS_SRC = "/stor3/AIMS4Meta/源代码/DreaMS"
if DREAMS_SRC not in sys.path:
    sys.path.insert(0, DREAMS_SRC)


###############################################################################
# 连通分量（Union-Find）
###############################################################################

class UnionFind:
    """并查集，用于连通分量聚类"""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


###############################################################################
# 轻量 MSP 解析（复用缓存机制）
###############################################################################

def load_library(lib_name, lib_path):
    """加载单个数据库（优先使用pickle缓存，不依赖matchms）"""
    import pickle
    pkl_path = lib_path.replace('.msp', '_spectra_cache.pkl')

    if os.path.exists(pkl_path):
        print(f"[INFO] 加载{lib_name}缓存: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            library = pickle.load(f)
        print(f"[INFO] 成功加载 {len(library)} 个参考谱图（缓存）")
        return library

    try:
        from matchms.importing import load_from_msp
        print(f"[INFO] 加载{lib_name}数据库（解析MSP）: {lib_path}")
        library = list(load_from_msp(lib_path))
        print(f"[INFO] 成功加载 {len(library)} 个参考谱图")
        return library
    except ImportError:
        print(f"[ERROR] 无缓存且matchms不可用，无法加载: {lib_path}")
        return []


def parse_msp_metadata(msp_path):
    """从 MSP 解析元数据 + 峰列表（复用L1缓存机制）"""
    library = load_library('', msp_path)

    entries = []
    for spec in library:
        name = spec.get('compound_name') or spec.get('name', '')
        entry = {
            'name': name,
            'peaks': list(spec.peaks) if hasattr(spec, 'peaks') else [],
        }
        for key in ['smiles', 'inchikey', 'formula', 'precursor_mz',
                    'ontology', 'adduct', 'retention_time', 'ion_mode',
                    'source_tool', 'source_db', 'comment']:
            val = spec.get(key)
            if val is not None:
                entry[key] = val
        entries.append(entry)

    return entries


###############################################################################
# 核心：样品间 Embedding 互连构建分子网络
###############################################################################

def build_molecular_network(sample_embs, sample_info, threshold):
    """构建样品间分子网络

    对所有样品间两两计算 embedding cosine 相似度，
    相似度 ≥ threshold 的样品对连边。

    参数:
        sample_embs: 样品 embedding 矩阵 (n_sample, dim)
        sample_info: 样品元数据列表
        threshold: cosine 相似度阈值

    返回:
        edges: 边列表 [(source_idx, target_idx, cosine_score), ...]
        clusters: {cluster_id: [sample_indices, ...], ...}
    """
    n_sample = len(sample_embs)
    use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()

    # 归一化 embedding
    if use_gpu:
        tensor = torch.tensor(sample_embs, dtype=torch.float32, device='cuda:0')
        norms = torch.norm(tensor, dim=1, keepdim=True)
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)
        normed = tensor / norms
    else:
        norms = np.linalg.norm(sample_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = sample_embs / norms

    edges = []

    # 分批计算，避免 OOM
    batch_size = 1000 if use_gpu else 500
    pbar = tqdm(total=n_sample, desc="  构建分子网络", ncols=100)

    for i_start in range(0, n_sample, batch_size):
        i_end = min(i_start + batch_size, n_sample)
        batch = normed[i_start:i_end]

        if use_gpu:
            # 计算当前批次与 i_start 之后所有样品的相似度（只算上三角）
            # (batch, dim) @ (n-i_start, dim).T → (batch, n-i_start)
            tail = normed[i_start:]
            sim_block = torch.mm(batch, tail.T)

            for li in range(i_end - i_start):
                gi = i_start + li
                row = sim_block[li]
                # 只取 gi 之后的（上三角，避免重复）
                local_from = li + 1  # 偏移到 gi+1 在 tail 中的位置
                if local_from >= row.shape[0]:
                    continue
                above_mask = row[local_from:] >= threshold
                above_positions = torch.where(above_mask)[0].cpu().numpy()  # 相对 local_from 的偏移
                above_scores = row[local_from:].cpu().numpy()

                for pos in above_positions:
                    gj = i_start + local_from + pos.item() if hasattr(pos, 'item') else i_start + local_from + int(pos)
                    score = float(above_scores[pos])
                    edges.append((gi, gj, round(score, 6)))
        else:
            tail = normed[i_start:]
            sim_block = batch @ tail.T

            for li in range(i_end - i_start):
                gi = i_start + li
                local_from = li + 1
                if local_from >= sim_block.shape[1]:
                    continue
                row = sim_block[li, local_from:]
                positions = np.where(row >= threshold)[0]  # 相对 local_from 的偏移

                for pos in positions:
                    gj = i_start + local_from + pos
                    score = float(row[pos])
                    edges.append((gi, gj, round(score, 6)))

        pbar.update(i_end - i_start)

    pbar.close()

    # 连通分量聚类
    uf = UnionFind(n_sample)
    for si, ti, _ in edges:
        uf.union(si, ti)

    # 收集聚类结果
    cluster_map = defaultdict(list)
    for i in range(n_sample):
        root = uf.find(i)
        cluster_map[root].append(i)

    # 按 cluster 大小排序，大簇优先
    clusters = {}
    for cid, (root, members) in enumerate(
        sorted(cluster_map.items(), key=lambda x: -len(x[1]))
    ):
        clusters[cid] = members

    return edges, clusters


###############################################################################
# 主函数
###############################################################################

def main():
    # ---- 读取 & 验证环境变量 ----
    OUTPUT_DIR    = os.environ.get('L4_OUTPUT_DIR')
    SAMPLE_EMB    = os.environ.get('L4_SAMPLE_EMB')
    SAMPLE_MSP    = os.environ.get('L4_SAMPLE_MSP')
    SIM_THRESHOLD = os.environ.get('L4_SIM_THRESHOLD')

    _missing = []
    for var_name, var_val in [
        ('L4_OUTPUT_DIR', OUTPUT_DIR),
        ('L4_SAMPLE_EMB', SAMPLE_EMB),
        ('L4_SAMPLE_MSP', SAMPLE_MSP),
        ('L4_SIM_THRESHOLD', SIM_THRESHOLD),
    ]:
        if not var_val:
            _missing.append(var_name)
    if _missing:
        raise ValueError(f"错误：以下环境变量未设置：{', '.join(_missing)}")

    SIM_THRESHOLD = float(SIM_THRESHOLD)
    start_time = time.time()

    print("=" * 70)
    print("L4: DreaMS 分子网络（纯样品间 Embedding 互连）")
    print("=" * 70)
    print(f"  样品 Embedding: {SAMPLE_EMB}")
    print(f"  样品 MSP: {SAMPLE_MSP}")
    print(f"  相似度阈值: {SIM_THRESHOLD}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 70)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ======== 1. 加载样品 embedding + MSP ========
    print("\n[1/3] 加载样品数据...")

    if not os.path.exists(SAMPLE_EMB):
        print(f"[ERROR] 样品 embedding 不存在: {SAMPLE_EMB}")
        return False
    if not os.path.exists(SAMPLE_MSP):
        print(f"[ERROR] 样品 MSP 不存在: {SAMPLE_MSP}")
        return False

    sample_data = np.load(SAMPLE_EMB, allow_pickle=True)
    sample_embs_full = sample_data['embeddings'].astype(np.float32)
    sample_msp_entries = parse_msp_metadata(SAMPLE_MSP)

    # 对齐 embedding 和 MSP（通过名称）
    sample_info = []
    sample_emb_indices = []

    if sample_msp_entries and 'names' in sample_data:
        emb_names = list(sample_data['names'])
        emb_pmzs = list(sample_data.get('precursor_mzs', [0.0] * len(emb_names)))
        name_to_idx = {name: i for i, name in enumerate(emb_names)}

        for spec in sample_msp_entries:
            name = spec.get('name', '')
            if name in name_to_idx:
                idx = name_to_idx[name]
                sample_info.append({
                    'name': emb_names[idx],
                    'precursor_mz': float(emb_pmzs[idx]) if idx < len(emb_pmzs) else spec.get('precursor_mz', 0),
                    'adduct': spec.get('adduct', ''),
                    'smiles': spec.get('smiles', ''),
                    'formula': spec.get('formula', ''),
                    'ontology': spec.get('ontology', ''),
                })
                sample_emb_indices.append(idx)

        if sample_emb_indices:
            sample_embs = sample_embs_full[sample_emb_indices]
        else:
            sample_embs = sample_embs_full
            sample_info = [{'name': f'sample_{i}',
                            'precursor_mz': 0,
                            'adduct': '', 'smiles': '', 'formula': '', 'ontology': ''}
                           for i in range(len(sample_embs_full))]
        print(f"  样品: MSP {len(sample_msp_entries)} 条 → Embedding 匹配 {len(sample_info)} 条")
    else:
        sample_embs = sample_embs_full
        sample_info = [{'name': f'sample_{i}',
                        'precursor_mz': 0,
                        'adduct': '', 'smiles': '', 'formula': '', 'ontology': ''}
                       for i in range(len(sample_embs_full))]
        print(f"  样品: {sample_embs.shape[0]} 条（无名称匹配，直接对齐）")

    n_sample = len(sample_embs)

    # ======== 2. 构建分子网络 ========
    print(f"\n[2/3] 构建分子网络（阈值 ≥ {SIM_THRESHOLD}）...")

    edges, clusters = build_molecular_network(
        sample_embs, sample_info, threshold=SIM_THRESHOLD
    )

    # ======== 3. 导出结果 ========
    print(f"\n[3/3] 导出结果...")

    # 3a: 边表
    if edges:
        edge_records = []
        for si, ti, score in edges:
            edge_records.append({
                'source_idx': si,
                'target_idx': ti,
                'source_name': sample_info[si].get('name', f'sample_{si}'),
                'target_name': sample_info[ti].get('name', f'sample_{ti}'),
                'source_precursor_mz': sample_info[si].get('precursor_mz', 0),
                'target_precursor_mz': sample_info[ti].get('precursor_mz', 0),
                'cosine_score': score,
            })
        edges_df = pd.DataFrame(edge_records)
        edges_df = edges_df.sort_values('cosine_score', ascending=False)
        edges_path = str(output_dir / 'L4_network_edges.csv')
        edges_df.to_csv(edges_path, index=False, encoding='utf-8')
        print(f"  边表: {len(edges_df)} 条边 → {edges_path}")
    else:
        edges_path = str(output_dir / 'L4_network_edges.csv')
        pd.DataFrame().to_csv(edges_path, index=False, encoding='utf-8')
        print("  无边（所有样品间相似度均低于阈值）")

    # 3b: 节点属性表（含聚类信息）
    # 构建节点 → cluster_id 映射
    node_cluster = {}
    cluster_sizes = {}
    for cid, members in clusters.items():
        cluster_sizes[cid] = len(members)
        for idx in members:
            node_cluster[idx] = cid

    node_records = []
    for i in range(n_sample):
        info = sample_info[i]
        cid = node_cluster.get(i, -1)
        node_records.append({
            'node_idx': i,
            'name': info.get('name', f'sample_{i}'),
            'precursor_mz': info.get('precursor_mz', 0),
            'adduct': info.get('adduct', ''),
            'smiles': info.get('smiles', ''),
            'formula': info.get('formula', ''),
            'ontology': info.get('ontology', ''),
            'cluster_id': cid,
            'cluster_size': cluster_sizes.get(cid, 1),
        })

    nodes_df = pd.DataFrame(node_records)
    nodes_path = str(output_dir / 'L4_network_nodes.csv')
    nodes_df.to_csv(nodes_path, index=False, encoding='utf-8')

    # 统计聚类信息
    n_clusters = len(clusters)
    n_multi_node = sum(1 for members in clusters.values() if len(members) > 1)
    max_cluster = max(cluster_sizes.values()) if cluster_sizes else 0

    elapsed = time.time() - start_time

    print(f"  节点表: {n_sample} 个节点 → {nodes_path}")
    print(f"  聚类: {n_clusters} 个（多节点: {n_multi_node}，最大: {max_cluster}）")

    print("\n" + "=" * 70)
    print(f"L4 DreaMS 分子网络完成 | 样品: {n_sample} | 边: {len(edges)} | "
          f"聚类: {n_clusters} | 耗时: {elapsed:.1f}s")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
