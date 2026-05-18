#!/usr/bin/env python3
import csv, os, re, sys
from collections import Counter, defaultdict

def extract_bare_name(name):
    m = re.search(r'(Unknown\s*\([\d.]+_[\d.]+[mn]?/z\))', name)
    return m.group(1) if m else name

def load_csv(path):
    if not os.path.exists(path): return []
    with open(path, 'r', encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def save_csv(path, rows, fields):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def load_identifications(input_dir):
    ident = {}
    for r in load_csv(os.path.join(input_dir, 'L3_results', 'L3_identified.csv')):
        key = extract_bare_name(r['query_name'])
        if key not in ident:
            ident[key] = {'smiles': r.get('matched_smiles','').strip(), 'formula': r.get('matched_formula','').strip(), 'matched_name': r.get('matched_name','').strip(), 'ontology': r.get('matched_ontology','').strip(), 'source_level': 'L3_SIRIUS'}
    for fname in ['L1_results.csv', 'L1_matchMS_results.csv', 'L1_DreaMS_results.csv']:
        for r in load_csv(os.path.join(input_dir, 'L1_results', fname)):
            key = extract_bare_name(r.get('query_name',''))
            if key:
                ident[key] = {'smiles': r.get('matched_smiles','').strip(), 'formula': r.get('matched_formula','').strip(), 'matched_name': r.get('matched_name','').strip(), 'ontology': r.get('matched_ontology','').strip(), 'source_level': 'L1'}
    return ident

def propagate_to_nodes(input_dir, ident):
    nodes = load_csv(os.path.join(input_dir, 'L4_results', 'L4_network_nodes.csv'))
    if not nodes: print('No L4_network_nodes.csv'); return nodes
    matched = 0
    for node in nodes:
        key = extract_bare_name(node['name'])
        if key in ident:
            ann = ident[key]
            for k in ['smiles','formula','matched_name','ontology','source_level']: node[k] = ann[k]
            matched += 1
    print(f'Name matched: {matched}/{len(nodes)}')
    cn = defaultdict(list)
    for n in nodes: cn[n['cluster_id']].append(n)
    prop = 0
    for cid, members in cn.items():
        m = [n for n in members if n.get('smiles')]
        if not m: continue
        u = [n for n in members if not n.get('smiles')]
        if not u: continue
        top_f = Counter(n['formula'] for n in m if n.get('formula')).most_common(1)
        top_o = Counter(n['ontology'] for n in m if n.get('ontology')).most_common(1)
        top_l = Counter(n['source_level'] for n in m if n.get('source_level')).most_common(1)
        for node in u:
            if top_f: node['formula'] = top_f[0][0]
            if top_o: node['ontology'] = top_o[0][0]
            if top_l: node['source_level'] = top_l[0][0]+'_propagated'
            node['propagated'] = 'True'
            prop += 1
    print(f'Propagated: {prop}')
    out = os.path.join(input_dir, 'L4_results', 'L4_network_nodes_annotated.csv')
    f = ['node_idx','name','precursor_mz','adduct','smiles','formula','matched_name','ontology','source_level','source_method','source_database','confidence','propagated','cluster_id','cluster_size']
    save_csv(out, nodes, f)
    ann = sum(1 for n in nodes if n.get('smiles'))
    print(f'Annotated: {ann}/{len(nodes)} ({ann*100/len(nodes):.1f}%)')
    return nodes

def update_summary(input_dir, nodes):
    spath = os.path.join(input_dir, 'summary.csv')
    summary = load_csv(spath)
    if not summary: return
    lookup = {}
    for n in nodes:
        k = extract_bare_name(n['name'])
        lookup[k] = {'cluster_id': n['cluster_id'], 'cluster_size': n['cluster_size']}
    f = list(summary[0].keys())
    f += ['l4_cluster_id','l4_cluster_size']
    for row in summary:
        k = extract_bare_name(row['query_name'])
        if k in lookup: row['l4_cluster_id'] = lookup[k]['cluster_id']; row['l4_cluster_size'] = lookup[k]['cluster_size']
    save_csv(spath, summary, f)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    args = ap.parse_args()
    ident = load_identifications(args.input_dir)
    print(f'Loaded {len(ident)} identifications')
    nodes = propagate_to_nodes(args.input_dir, ident)
    update_summary(args.input_dir, nodes)
    print('Done')
