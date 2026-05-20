#!/usr/bin/env python3
import sys, os, json, csv, re, numpy as np, torch

MSFLOW = '/stor3/AIMS4Meta/源代码/MSFlow-main'
os.chdir(MSFLOW)

sys.path.insert(0, os.path.join(MSFLOW, 'ms_scripts/DiffMS/src'))
from mist.models.spectra_encoder import SpectraEncoderGrowing
sys.path.remove(os.path.join(MSFLOW, 'ms_scripts/DiffMS/src'))

sys.path.insert(0, MSFLOW)
from modules.cond_lit_model import CondFlowMolBERTLitModule
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from utils.sample import cond_generate_mols
from utils.metrics import decode_tokens_to_smiles
from utils.functions import canonicalize
from configs import ID2TOK, TOK2ID, PAD
torch.serialization.add_safe_globals([MixtureDiscreteProbPath, PolynomialConvexScheduler, torch.nn.modules.loss.CrossEntropyLoss])

ELEM = ['C','H','As','B','Br','Cl','Co','F','Fe','I','K','N','Na','O','P','S','Se','Si']
E2I = {e:i for i,e in enumerate(ELEM)}

def formula_to_counts(f):
    c = np.zeros(18, dtype=np.int32)
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', f):
        n = int(n) if n else 1
        if el in E2I: c[E2I[el]] = min(n, 255)
    return c

def load_sirius(path):
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

def load_subformula(key, subform_dir):
    path = os.path.join(subform_dir, f"{key}.json")
    if not os.path.exists(path): return None
    with open(path) as f:
        return json.load(f)

def encode(precursor_mz, peaks, fm, subform_dir, enc, dec, dev, name=''):
    key = extract_key(name)
    n = peaks.shape[0]; mp = n + 1
    fv = np.zeros((mp, 18), dtype=np.int32)
    
    # CLS token gets overall formula
    formula = fm.get(key, '')
    if formula:
        fv[0] = formula_to_counts(formula)
    else:
        n_c = max(1, int(precursor_mz / 14))
        fv[0, 0] = min(n_c, 255); fv[0, 1] = min(n_c*2, 255); fv[0, 13] = min(n_c//3, 255)
    
    # Fragment peaks: load subformulas from SIRIUS tree JSON
    sd = load_subformula(key, subform_dir) if subform_dir else None
    if sd and 'output_tbl' in sd and 'formula' in sd['output_tbl']:
        jfrags = sd['output_tbl']['formula']
        for i, fragf in enumerate(jfrags):
            if i < n: fv[i+1] = formula_to_counts(fragf)
    # Fallback: use mass-based estimation for unfilled peaks
    for i in range(n):
        if np.all(fv[i+1] == 0):
            if formula:
                fv[i+1] = formula_to_counts(formula)
            else:
                fv[i+1] = fv[0]
    
    fv_t = torch.from_numpy(fv).float().unsqueeze(0).to(dev)
    ty = torch.zeros(1, mp, dtype=torch.long, device=dev); ty[0, 0] = 3
    it = torch.zeros(1, mp, device=dev)
    if n > 0: it[0, 1:] = torch.from_numpy(peaks[:, 1]).float().to(dev)
    batch = {'form_vec': torch.clamp(fv_t, 0, 255), 'types': ty, 'intens': it,
        'ion_vec': torch.zeros(1, mp, dtype=torch.long, device=dev),
        'num_peaks': torch.tensor([mp], device=dev),
        'instruments': torch.zeros(1, dtype=torch.long, device=dev), 'names': [name]}
    with torch.no_grad(): cddd, _ = enc(batch)
    return cddd.squeeze(0)

def parse(path):
    sp = []; cur = {'peaks': []}
    for line in open(path):
        line = line.strip()
        if not line: continue
        if ':' in line and not line[0].isdigit():
            k, v = line.split(':', 1); k = k.strip().upper()
            if k == 'NAME':
                if cur['peaks']: cur['peaks'] = np.array(cur['peaks']); sp.append(cur)
                cur = {'name': v.strip(), 'peaks': []}
            elif k == 'PRECURSORMZ': cur['precursor_mz'] = float(v)
        else:
            p = line.split()
            if len(p) >= 2: cur['peaks'].append([float(p[0]), float(p[1])])
    if cur['peaks']: cur['peaks'] = np.array(cur['peaks']); sp.append(cur)
    return sp

def load_models(dev):
    chk = torch.load('checkpoints/Encoder/encoder_msg_cddd.pt', map_location=dev)
    enc = SpectraEncoderGrowing(inten_transform='float', inten_prob=0.1, remove_prob=0.5,
        peak_attn_layers=2, num_heads=8, pairwise_featurization=True, embed_instrument=False,
        cls_type='ms1', set_pooling='cls', spec_features='peakformula', mol_features='fingerprint',
        form_embedder='pos-cos', output_size=512, hidden_size=512, spectra_dropout=0.0,
        top_layers=1, refine_layers=4, magma_modulo=2048)
    enc.load_state_dict(chk['model_state_dict']); enc.eval(); enc.to(dev)
    def _replace_sig(m):
        for name, child in m.named_children():
            if isinstance(child, torch.nn.Sigmoid): setattr(m, name, torch.nn.Tanh())
            else: _replace_sig(child)
    _replace_sig(enc)
    cfm = CondFlowMolBERTLitModule.load_from_checkpoint('checkpoints/Decoder/MSFlow_cddds.ckpt')
    dec = cfm.model; dec.eval(); dec.to(dev)
    return enc, dec

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input_msp', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--formula_tsv', default=None)
    p.add_argument('--subform_dir', default=None)
    p.add_argument('--num_candidates', type=int, default=5)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--ion_mode', default=None)
    p.add_argument('--encoder_path', default=None)
    p.add_argument('--decoder_path', default=None)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    fm = load_sirius(a.formula_tsv) if a.formula_tsv and os.path.exists(a.formula_tsv) else {}
    print(f"FormulaTSV={a.formula_tsv} ({len(fm)} compounds)")
    sp = parse(a.input_msp)
    if fm:
        before = len(sp)
        sp = [s for s in sp if extract_key(s['name']) in fm]
        print(f"Formula filter: {before} -> {len(sp)} spectra")
    print(f"Spectra: {len(sp)}")
    enc, dec = load_models(a.device)
    csv_path = os.path.join(a.output_dir, 'L4_denovo_candidates.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['query_name','rank','smiles','score'])
        for spec in sp:
            cddd = encode(spec['precursor_mz'], spec['peaks'], fm, a.subform_dir, enc, dec, a.device, spec['name'])
            cond = cddd.unsqueeze(0).expand(a.num_candidates, -1)
            samps = cond_generate_mols(dec, cond=cond, source_distribution='uniform',
                num_samples=a.num_candidates, steps=128, guidance_scale=1.5, temperature=1, device=a.device)
            _, smi = decode_tokens_to_smiles(samps, ID2TOK=ID2TOK, TOK2ID=TOK2ID, PAD=PAD)
            smi = [canonicalize(s) for s in smi if s]
            smi = [s for s in smi if s is not None]
            for r, s in enumerate(smi, 1): w.writerow([spec['name'], r, s, 1.0 / r])
            torch.cuda.empty_cache()
    print(f"Saved: {csv_path} ({len(sp)} spectra x {a.num_candidates} candidates)")
