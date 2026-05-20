#!/usr/bin/env python3
"""
MSFlow 从头预测 + 准确性验证脚本
数据集: spectraverse_pos
流程: subformulae JSON -> 编码器(CDDD) -> 解码器(SMILES) -> 与labels.tsv比对
"""
import sys, os, json, csv, re, time, argparse
import numpy as np
import torch
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, DataStructs
RDLogger.DisableLog("rdApp.*")

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

ATOM_MASS = {
    "C":12.0,"H":1.007825,"O":15.994915,"N":14.003074,
    "P":30.973762,"S":31.972071,"F":18.998403,"Cl":35.453,
    "Br":79.904,"I":126.904,"Na":22.989218,"K":38.963707,
    "Si":28.0855,"Se":78.96,"As":74.9216,"B":10.81,"Fe":55.845,"Co":58.933
}
ION_LST = ["[M+H]+","[M+Na]+","[M+K]+","[M-H2O+H]+","[M+H3N+H]+","[M]+","[M-H4O2+H]+"]
ION_REMAP = {
    "[M+NH4]+":"[M+H3N+H]+","M+H":"[M+H]+","M+Na":"[M+Na]+",
    "M+H-H2O":"[M-H2O+H]+","M-H2O+H":"[M-H2O+H]+","M+NH4":"[M+H3N+H]+",
    "M-2H2O+H":"[M-H4O2+H]+","[M-2H2O+H]+":"[M-H4O2+H]+",
}
ION_TO_IDX = {k: i for i, k in enumerate(ION_LST)}

def get_ion_idx(ion_str):
    ion_str = ION_REMAP.get(ion_str, ion_str)
    return ION_TO_IDX.get(ion_str, len(ION_LST) - 1)

def formula_to_counts(f):
    c = np.zeros(18, dtype=np.int32)
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', f):
        n = int(n) if n else 1
        if el in E2I: c[E2I[el]] = min(n, 255)
    return c

def canonical_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol) if mol else None
    except:
        return None

def tanimoto_sim(smi1, smi2):
    try:
        m1 = Chem.MolFromSmiles(smi1)
        m2 = Chem.MolFromSmiles(smi2)
        if m1 is None or m2 is None:
            return 0.0
        fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
        fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0

def _replace_sigmoid_with_tanh(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Sigmoid):
            setattr(module, name, torch.nn.Tanh())
        else:
            _replace_sigmoid_with_tanh(child)

def load_models(device, encoder_type='canopus'):
    print(f"Loading encoder ({encoder_type})...")
    if encoder_type == 'canopus':
        ckpt = torch.load('checkpoints/Encoder/encoder_canpous_cddd.pt', map_location=device)
        enc = SpectraEncoderGrowing(
            inten_transform='float', inten_prob=0.1, remove_prob=0.5,
            peak_attn_layers=2, num_heads=8, pairwise_featurization=True, embed_instrument=False,
            cls_type='ms1', set_pooling='cls', spec_features='peakformula', mol_features='fingerprint',
            form_embedder='pos-cos', output_size=512, hidden_size=256, spectra_dropout=0.0,
            top_layers=1, refine_layers=4, magma_modulo=512)
    else:
        ckpt = torch.load('checkpoints/Encoder/encoder_msg_cddd.pt', map_location=device)
        enc = SpectraEncoderGrowing(
            inten_transform='float', inten_prob=0.1, remove_prob=0.5,
            peak_attn_layers=2, num_heads=8, pairwise_featurization=True, embed_instrument=False,
            cls_type='ms1', set_pooling='cls', spec_features='peakformula', mol_features='fingerprint',
            form_embedder='pos-cos', output_size=512, hidden_size=512, spectra_dropout=0.0,
            top_layers=1, refine_layers=4, magma_modulo=2048)
    enc.load_state_dict(ckpt['model_state_dict'])
    enc.eval()
    enc.to(device)
    _replace_sigmoid_with_tanh(enc)

    print("Loading decoder...")
    cfm = CondFlowMolBERTLitModule.load_from_checkpoint(
        'checkpoints/Decoder/MSFlow_cddds.ckpt', map_location=device, strict=False)
    dec = cfm.model
    dec.eval()
    dec.to(device)
    return enc, dec

def encode_spectrum(spec_id, subform_dir, enc, device, max_peaks=None):
    jpath = os.path.join(subform_dir, f"{spec_id}.json")
    if not os.path.exists(jpath):
        return None
    with open(jpath) as f:
        j = json.load(f)
    tbl = j["output_tbl"]
    root_form = j["cand_form"]
    root_ion = j.get("cand_ion", "[M+H]+")

    if tbl is None or len(tbl.get("formula", [])) == 0:
        return None

    frags_list = list(tbl["formula"])
    intens_list = list(tbl["ms2_inten"])
    ions_list = list(tbl["ions"])

    if max_peaks is not None and len(frags_list) > max_peaks:
        order = np.argsort(intens_list)[::-1]
        cutoff = min(len(intens_list) - 1, max_peaks)
        keep = order[:cutoff]
        frags_list = np.array(frags_list)[keep].tolist()
        intens_list = np.array(intens_list)[keep].tolist()
        ions_list = np.array(ions_list)[keep].tolist()

    n = len(frags_list)
    if n == 0:
        return None

    fv = np.zeros((n + 1, 18), dtype=np.int32)
    for i in range(n):
        fv[i] = formula_to_counts(frags_list[i])
    fv[n] = formula_to_counts(root_form)

    ty = np.zeros(n + 1, dtype=np.int64)
    ty[n] = 3

    it = np.zeros(n + 1, dtype=np.float32)
    it[:n] = np.array(intens_list, dtype=np.float32)
    it[n] = 1.0

    iv = np.zeros(n + 1, dtype=np.float32)
    for i in range(n):
        iv[i] = get_ion_idx(ions_list[i] if i < len(ions_list) else root_ion)
    iv[n] = get_ion_idx(root_ion)

    mp = n + 1
    batch = {
        'form_vec': torch.from_numpy(fv).float().unsqueeze(0).to(device),
        'types': torch.from_numpy(ty).long().unsqueeze(0).to(device),
        'intens': torch.from_numpy(it).float().unsqueeze(0).to(device),
        'ion_vec': torch.from_numpy(iv).float().unsqueeze(0).to(device),
        'num_peaks': torch.tensor([mp], dtype=torch.long, device=device),
        'instruments': torch.zeros(1, dtype=torch.float32, device=device),
        'names': [spec_id]
    }
    with torch.no_grad():
        cddd, _ = enc(batch)
    return cddd.squeeze(0)

def decode_cddd(cddd, dec, device, num_candidates=5):
    cond = cddd.unsqueeze(0).expand(num_candidates, -1)
    with torch.no_grad():
        samps = cond_generate_mols(
            dec, cond=cond, source_distribution='uniform',
            num_samples=num_candidates, steps=128,
            guidance_scale=1.5, temperature=1, device=device)
    _, smi = decode_tokens_to_smiles(samps, ID2TOK=ID2TOK, TOK2ID=TOK2ID, PAD=PAD)
    smi = [canonicalize(s) for s in smi if s]
    smi = [s for s in smi if s is not None]
    return smi

def main():
    parser = argparse.ArgumentParser(description='MSFlow 从头预测 + 准确性验证')
    parser.add_argument('--subform_dir', type=str,
        default='/stor3/AIMS4Meta/数据集/msflow_datasets/spectraverse/spectraverse_pos/subformulae')
    parser.add_argument('--labels_tsv', type=str,
        default='/stor3/AIMS4Meta/数据集/msflow_datasets/spectraverse/spectraverse_pos/labels.tsv')
    parser.add_argument('--output_dir', type=str,
        default='/stor3/AIMS4Meta/工作区/工作流/多层鉴定/L4_从头鉴定/output_spectraverse_pos')
    parser.add_argument('--num_candidates', type=int, default=10)
    parser.add_argument('--max_spectra', type=int, default=0, help='0=全部')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--encoder', type=str, default='canopus', choices=['canopus', 'msg'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("Loading labels...")
    labels_df = pd.read_csv(args.labels_tsv, sep='\t')
    labels_df['canon_smiles'] = labels_df['smiles'].apply(canonical_smiles)
    label_map = {}
    for _, row in labels_df.iterrows():
        spec_id = row['spec']
        if spec_id not in label_map and row['canon_smiles']:
            label_map[spec_id] = {
                'smiles': row['canon_smiles'],
                'formula': row['formula'],
                'name': row['name'],
                'ionization': row['ionization']
            }
    print(f"Loaded {len(label_map)} unique spectra with valid SMILES")

    spec_ids = sorted(label_map.keys())
    if args.max_spectra > 0:
        spec_ids = spec_ids[:args.max_spectra]
    print(f"Will process {len(spec_ids)} spectra")

    enc, dec = load_models(device, args.encoder)

    results = []
    t0_all = time.time()
    for idx, spec_id in enumerate(spec_ids):
        t0 = time.time()
        cddd = encode_spectrum(spec_id, args.subform_dir, enc, device)
        if cddd is None:
            continue
        pred_smiles = decode_cddd(cddd, dec, device, args.num_candidates)
        elapsed = time.time() - t0

        gt = label_map[spec_id]
        gt_smi = gt['smiles']
        top1_hit = 0
        top5_hit = 0
        best_tani = 0.0
        for rank, smi in enumerate(pred_smiles, 1):
            if smi == gt_smi:
                if rank == 1: top1_hit = 1
                top5_hit = 1
            tani = tanimoto_sim(smi, gt_smi)
            if tani > best_tani:
                best_tani = tani
            results.append({
                'spec_id': spec_id, 'rank': rank, 'pred_smiles': smi,
                'gt_smiles': gt_smi, 'exact_match': int(smi == gt_smi),
                'tanimoto': round(tani, 4)
            })

        if (idx + 1) % 100 == 0 or idx < 10:
            print(f"  [{idx+1}/{len(spec_ids)}] {spec_id}: "
                  f"{len(pred_smiles)} cands, top1={'✓' if top1_hit else '✗'}, "
                  f"best_tani={best_tani:.3f}, {elapsed:.1f}s")
        torch.cuda.empty_cache()

    total_time = time.time() - t0_all
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, 'detailed_results.csv'), index=False)

    n_total = len(spec_ids)
    per_spec = results_df.groupby('spec_id')
    top1_acc = (per_spec['exact_match'].max() == 1).sum() / n_total if n_total > 0 else 0
    top5_acc = (per_spec['exact_match'].max() == 1).sum() / n_total if n_total > 0 else 0
    top1_acc_rank1 = (per_spec.apply(lambda g: g[g['rank']==1]['exact_match'].max() if 1 in g['rank'].values else 0) == 1).sum() / n_total if n_total > 0 else 0
    avg_best_tani = per_spec['tanimoto'].max().mean()

    summary = {
        'dataset': 'spectraverse_pos',
        'n_spectra': n_total,
        'n_candidates': args.num_candidates,
        'top1_exact_accuracy': round(top1_acc_rank1, 4),
        'top5_exact_accuracy': round(top5_acc, 4),
        'avg_best_tanimoto': round(avg_best_tani, 4),
        'total_time_s': round(total_time, 1),
        'avg_time_per_spectrum_s': round(total_time / max(n_total, 1), 2)
    }

    print("\n" + "=" * 60)
    print("MSFlow 从头预测验证结果 (SpectraVerse-Pos)")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output_dir}")

if __name__ == '__main__':
    main()
