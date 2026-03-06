import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import torch
import swifter
from modules.cond_lit_model import CondFlowMolBERTLitModule
from utils.metrics import decode_tokens_to_smiles
from utils.sample import cond_generate_mols
from configs import *
from rdkit.Chem import rdFMCS
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import TanimotoSimilarity
from utils.functions import canonicalize, canonicalize_safe
import os
from multiprocessing import Process
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor 
from rdkit.Chem import rdFMCS
from myopic_mces import MCES


steps = 128

def fast_smiles_to_fps(smiles_list, radius=2, fp_size=2048):
    morgan_gen = GetMorganGenerator(radius=radius, fpSize=fp_size)
    if isinstance(smiles_list, str):
        mol = Chem.MolFromSmiles(smiles_list)
        fp = morgan_gen.GetFingerprint(mol)
        return fp

    fps_list = []
    for smi in smiles_list:
        if not isinstance(smi, str) or smi.strip() == "" or smi.lower() == "none":
            fps_list.append(None)
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps_list.append(None)
            continue

        fp = morgan_gen.GetFingerprint(mol)
        fps_list.append(fp)

    if all(fp is None for fp in fps_list):
        return None, None

    fps_np = []
    for fp in fps_list:
        if fp is None:
            fps_np.append(np.zeros((fp_size,), dtype=np.float32))
        else:
            arr = np.zeros((fp_size,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps_np.append(arr)

    return np.array(fps_np), fps_list


morgan_gen1 = GetMorganGenerator(radius=2, fpSize=2048)

def compute_tanimoto_to_reference(smiles_list, reference_smiles):
    ref_mol = Chem.MolFromSmiles(reference_smiles)
    if ref_mol is None:
        return np.zeros(len(smiles_list))

    ref_fp = morgan_gen1.GetFingerprint(ref_mol)
    _, fps_list = fast_smiles_to_fps(smiles_list, radius=2, fp_size=2048)
    if fps_list is None:
        return np.zeros(len(smiles_list))

    sims = []
    for fp in fps_list:
        if fp is None:
            sims.append(0.0)
        else:
            sims.append(DataStructs.TanimotoSimilarity(fp, ref_fp))

    return np.array(sims)




def process_chunk(chunk_df, gpu_id, chunk_id, ckpt_dir, checkpoint_path):
    """Worker process that runs on one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda")

    print(f"[GPU {gpu_id}] Starting chunk {chunk_id} ({len(chunk_df)} samples)")

    # Load model on this GPU
    model = CondFlowMolBERTLitModule.load_from_checkpoint(ckpt_dir + checkpoint_path)
    cfm = model.model
    cfm.eval().to(device)

    
    # cddd
    query_smiles = chunk_df.canon_smiles
    fps_list = chunk_df["cddd"].tolist()
    fps_np = np.array(fps_list, dtype=np.float32)
    conds = torch.tensor(fps_np, dtype=torch.float32).to(device)

    results = []

    for i in tqdm(range(len(query_smiles)), desc=f"GPU {gpu_id} - Chunk {chunk_id}"):

        cond = conds[i]
        samples = cond_generate_mols(
            cfm,
            cond=cond,
            source_distribution='uniform',
            num_samples=100,
            steps=steps, 
            device=device,
            temperature=1,
            guidance_scale=1.5,
        )

        _, smiles = decode_tokens_to_smiles(samples, ID2TOK=ID2TOK, TOK2ID=TOK2ID, PAD=PAD)
        smiles = [canonicalize_safe(canonicalize(s)) for s in smiles if s]
        smiles = [s for s in smiles if s is not None]
        smiles_count = Counter(smiles)
        smiles_ordered = [item for item, _ in smiles_count.most_common()]

        query_smi = canonicalize_safe(canonicalize(query_smiles.iloc[i]))

        if len(smiles) > 0:
            # Compute fingerprint-based Tanimoto similarities
            sims = compute_tanimoto_to_reference(smiles, query_smi)
            sims_sorted = np.sort(sims)[::-1]

            sims_1 = compute_tanimoto_to_reference([smiles_ordered[0]], query_smi)
            sims_10 = compute_tanimoto_to_reference(smiles_ordered[:10], query_smi)

            sim_top1_value = float(sims_1[0]) if len(sims_1) > 0 else 0.0
            sim_top10_value = float(np.max(sims_10)) if len(sims_10) > 0 else 0.0

            # Accuracy
            acc_top1 = 1 if query_smi == smiles_ordered[0] else 0
            acc_top10 = 1 if query_smi in smiles_ordered[:10] else 0
            
            # MCES distances
            mces_top1 = int(MCES(query_smi, smiles_ordered[0])[1]) if smiles_ordered else 0
            mces_top10 = min(
                [int(MCES(query_smi, s)[1]) for s in smiles_ordered[:10]],
                default=0.0
            )

            # Max tanimoto
            max_idx = np.argmax(sims)
            top10_idx = np.argmax(sims_10)

            result = {
                "query_smiles": query_smi,
                "mean_tanimoto": float(sims.mean()),
                "top1_tanimoto": float(sims_sorted[0]),
                "validity": len(smiles),
                "uniqueness": len(set(smiles)),
                "maxsim_smiles": smiles[max_idx],
                "sim_top1": sim_top1_value,
                "sim_top10": sim_top10_value,
                "maxsim_smiles_1": smiles_ordered[0] if smiles_ordered else None,
                "maxsim_smiles_10": smiles_ordered[top10_idx] if smiles_ordered else None,
                "acc_top1": acc_top1,
                "acc_top10": acc_top10,
                "mces_top1": mces_top1,
                "mces_top10": mces_top10,
            }


        else:
            result = {
                "query_smiles": query_smi,
                "mean_tanimoto": 0.0,
                "top1_tanimoto": 0.0,
                "validity": 0,
                "uniqueness": 0,
                "maxsim_smiles": None,
                "sim_top1": 0.0,
                "sim_top10": 0.0,
                "maxsim_smiles_1": None,
                "maxsim_smiles_10": None,
                "acc_top1": 0,
                "acc_top10": 0,
                "mces_top1": 0.0,
                "mces_top10": 0.0,
            }

        results.append(result)

    results_df = pd.DataFrame(results)
    out_path = f"../output/results/msg_chunk_{chunk_id}_{steps}_steps_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"[GPU {gpu_id}] ✅ Saved results to {out_path}")





def main():
    df_test =  pd.read_parquet('../output/data/msg_cddd.parquet') 
    ckpt_dir = '/hpfs/userws/mqawag/output/checkpoints/'
    checkpoint_path = 'MSFlow_cddd.ckpt' # cddd decoder

    # Split into 8 roughly equal chunks
    num_gpus = 6
    chunks = np.array_split(df_test, num_gpus)

    processes = []
    for gpu_id, chunk_df in enumerate(chunks):
        p = Process(target=process_chunk, args=(chunk_df, gpu_id, gpu_id, ckpt_dir, checkpoint_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge results
    all_results = []
    for i in range(num_gpus):
        df_chunk = pd.read_csv(f"../output/results/msg_chunk_{i}_{steps}_steps_results.csv")
        all_results.append(df_chunk)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(f"../output/results/msg_cddd.csv", index=False)
    print("✅ All chunks done and merged.")


if __name__ == "__main__":
    main()
