#!/usr/bin/env python3
"""
L4 从头鉴定 —— 薄封装，直接调用 MSFlow 源代码
依赖: /stor3/AIMS4Meta/源代码/MSFlow-main/
      /home/lyx/miniconda3/envs/msflow (torch 2.10)
"""

import sys, os, numpy as np, torch, csv, re

MSFLOW = '/stor3/AIMS4Meta/源代码/MSFlow-main'
sys.path.insert(0, MSFLOW)
sys.path.insert(0, os.path.join(MSFLOW, 'ms_scripts/DiffMS/src'))
os.chdir(MSFLOW)

from mist.models.spectra_encoder import SpectraEncoderGrowing
from modules.cond_lit_model import CondFlowMolBERTLitModule
from flow_matching.path import MixtureDiscreteProbPath, PolynomialConvexScheduler
from utils.sample import cond_generate_mols
from utils.metrics import decode_tokens_to_smiles
from utils.functions import canonicalize
from configs import ID2TOK, TOK2ID, PAD

torch.serialization.add_safe_globals([MixtureDiscreteProbPath, PolynomialConvexScheduler, torch.nn.modules.loss.CrossEntropyLoss])

# ---- Formula helpers (inlined from sirius_formula_helper.py) ----
ELEMENT_ORDER = ['C','H','As','B','Br','Cl','Co','F','Fe','I','K','N','Na','O','P','S','Se','Si']
E_TO_IDX = {e:i for i,e in enumerate(ELEMENT_ORDER)}

def formula_to_counts(f):
    c = np.zeros(18, dtype=np.int32)
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', f):
        n = int(n) if n else 1
        if el in E_TO_IDX: c[E_TO_IDX[el]] = min(n, 255)
    return c

def load_sirius_formulas(path):
    m = {}
    for row in csv.DictReader(open(path), delimiter='\t'):
        fid = row.get('mappingFeatureId', '')
        r = re.search(r'\(([\d.]+)_([\d.]+)m/z\)', fid)
        if r:
            k = f"{r.group(1)}_{r.group(2)}"
            if row.get('molecularFormula'): m[k] = row['molecularFormula']
    return m

def extract_key(name):
    r = re.search(r'\(([\d.]+)_([\d.]+)m/z\)', name)
    return f"{r.group(1)}_{r.group(2)}" if r else name

# ---- Encoder & Decoder ----
device = 'cuda' if torch.cuda.is_available() else 'cpu'

chk = torch.load('checkpoints/Encoder/encoder_msg_cddd.pt', map_location=device)
encoder = SpectraEncoderGrowing(inten_transform='float', inten_prob=0.1,
    remove_prob=0.5, peak_attn_layers=2, num_heads=8, pairwise_featurization=True,
    embed_instrument=False, cls_type='ms1', set_pooling='cls',
    spec_features='peakformula', mol_features='fingerprint',
    form_embedder='pos-cos', output_size=512, hidden_size=512,
    spectra_dropout=0.0, top_layers=1, refine_layers=4, magma_modulo=2048)
encoder.load_state_dict(chk['model_state_dict']); encoder.eval(); encoder.to(device)
for n, c in encoder.named_children():
    if isinstance(c, torch.nn.Sigmoid): setattr(encoder, n, torch.nn.Tanh())

cfm = CondFlowMolBERTLitModule.load_from_checkpoint('checkpoints/Decoder/MSFlow_cddds.ckpt')
decoder = cfm.model; decoder.eval(); decoder.to(device)

def encode_spectrum(precursor_mz, peaks, formula_map, name=''):
    key = extract_key(name)
    formula = formula_map.get(key, '')
    if formula:
        counts = formula_to_counts(formula)
    else:
        n_c = max(1, int(precursor_mz / 14))
        counts = np.zeros(18, dtype=np.int32)
        counts[0] = min(n_c, 255); counts[1] = min(n_c*2, 255); counts[13] = min(n_c//3, 255)
    n = peaks.shape[0]; mp = n + 1
    fv = torch.from_numpy(np.tile(counts, (mp, 1))).float().unsqueeze(0).to(device)
    ty = torch.zeros(1, mp, dtype=torch.long, device=device); ty[0, 0] = 3
    it = torch.zeros(1, mp, device=device)
    if n > 0: it[0, 1:] = torch.from_numpy(peaks[:, 1]).float().to(device)
    batch = {'form_vec': torch.clamp(fv, 0, 255), 'types': ty, 'intens': it,
        'ion_vec': torch.zeros(1, mp, dtype=torch.long, device=device),
        'num_peaks': torch.tensor([mp], device=device),
        'instruments': torch.zeros(1, dtype=torch.long, device=device),
        'names': [name]}
    with torch.no_grad(): cddd, _ = encoder(batch)
    return cddd.squeeze(0)

def parse_msp(path):
    spectra = []; cur = {'peaks': []}
    for line in open(path):
        line = line.strip()
        if not line: continue
        if ':' in line and not line[0].isdigit():
            k, v = line.split(':', 1); k = k.strip().upper()
            if k == 'NAME':
                if cur['peaks']: cur['peaks'] = np.array(cur['peaks']); spectra.append(cur)
                cur = {'name': v.strip(), 'peaks': []}
            elif k == 'PRECURSORMZ': cur['precursor_mz'] = float(v)
        else:
            p = line.split()
            if len(p) >= 2: cur['peaks'].append([float(p[0]), float(p[1])])
    if cur['peaks']: cur['peaks'] = np.array(cur['peaks']); spectra.append(cur)
    return spectra

def main(msp_path, output_dir, formula_tsv=None, num_candidates=5):
    os.makedirs(output_dir, exist_ok=True)
    formula_map = {}
    if formula_tsv and os.path.exists(formula_tsv):
        formula_map = load_sirius_formulas(formula_tsv)
        print(f"SIRIUS formulas: {len(formula_map)} compounds")
    spectra = parse_msp(msp_path)
    csv_path = os.path.join(output_dir, 'L4_denovo_candidates.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['query_name', 'rank', 'smiles', 'score'])
        for spec in spectra:
            cddd = encode_spectrum(spec['precursor_mz'], spec['peaks'], formula_map, spec['name'])
            cond = cddd.unsqueeze(0).expand(num_candidates, -1)
            samples = cond_generate_mols(decoder, cond=cond,
                source_distribution='uniform', num_samples=num_candidates,
                steps=128, guidance_scale=1.5, temperature=1, device=device)
            _, smi = decode_tokens_to_smiles(samples, ID2TOK=ID2TOK, TOK2ID=TOK2ID, PAD=PAD)
            smi = [canonicalize(s) for s in smi if s]; smi = [s for s in smi if s is not None]
            for r, s in enumerate(smi, 1): w.writerow([spec['name'], r, s, 1.0 / r])
    print(f"Saved: {csv_path}")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input_msp', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--formula_tsv', default=None)
    p.add_argument('--num_candidates', type=int, default=5)
    a = p.parse_args()
    main(a.input_msp, a.output_dir, a.formula_tsv, a.num_candidates)

