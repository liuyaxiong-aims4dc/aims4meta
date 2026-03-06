from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset
from transformers import GenerationConfig
import torch
from tqdm import tqdm
from accelerate import PartialState
from accelerate.utils import gather_object
import argparse
import numpy as np
import pandas as pd
import os
from rich.table import Table
from rich.console import Console
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem
from rdkit import Chem
from selfies import decoder
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import sys
sys.path.append(".")
from src.metric import MoleculeEvaluator, TopkMoleculeEvaluator
from src.utils import save_arr



def morgan_fp(mol: Chem.Mol, fp_size=2048, radius=2):
    """
    Compute Morgan fingerprint for a molecule.
    
    Args:
        mol (Chem.Mol): _description_
        fp_size (int, optional): Size of the fingerprint.
        radius (int, optional): Radius of the fingerprint.
    """

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_size)
    return fp


def tanimoto_scores(
    true_selfies: str,
    pred_selfies: str
) -> list:
    try:
        true_smiles = decoder(true_selfies)
        true_mol = Chem.MolFromSmiles(true_smiles)
        if true_mol is None:
            return 0
    except Exception as e:
        return 0
    
    try:
        pred_smiles = decoder(pred_selfies)
        pred_mol = Chem.MolFromSmiles(pred_smiles)
        if pred_mol is None:
            return 0
    except Exception as e:
        return 0
    
    true_fp = morgan_fp(true_mol)
    pred_fp = morgan_fp(pred_mol)

    try:
        reward = TanimotoSimilarity(true_fp, pred_fp)
    except Exception as e:
        print(f"Tanimoto error: {e}")
        reward = 0
    return reward


if __name__ == "__main__":

    #Argument parsing
    parser = argparse.ArgumentParser(description="Generate candidates for feedback training")
    parser.add_argument("--model_path", type=str, default="logs/checkpoints/bart-base-selfies-pretrain-4M", help="Path to the trained BART model")
    parser.add_argument("--train_path",  type=str, default="logs/datasets/MassSpecGym-ALL/MassSpecGym-ALL_fps_selfies_threshold_0.2.tsv", help="Path to the train dataset")    
    parser.add_argument("--temperature", type=float, default=0.4, help="Temperature for sampling")
    parser.add_argument("--num_beams", type=int, default=10, help="Number of beams for beam search. 1 means no beam search.")
    parser.add_argument("--compute_mces", action="store_true", 
                       help="Compute mces molecular metrics")
    args = parser.parse_args()

    # Start up the distributed environment without needing the Accelerator.
    distributed_state = PartialState()
    # Initialize evaluators only on main process
    if distributed_state.is_main_process:
        print("Evaluate and save: ", args)

    model_path = args.model_path
    train_path = args.train_path
    temperature = args.temperature

    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path).to(distributed_state.device)
    model.eval()

    train_dataset = pd.read_csv(train_path, sep='\t')
    train_dataset = train_dataset.to_dict('records')
    debug = False
    if debug: train_dataset = train_dataset[:1004]
        
    batch_size = 16
    if args.num_beams > 20:
        batch_size = 4
    
    n_gpu = distributed_state.num_processes
    rank = distributed_state.process_index
    data_per_gpu = len(train_dataset) // n_gpu
    start_index = rank * data_per_gpu
    end_index = start_index + data_per_gpu if rank < n_gpu - 1 else len(train_dataset)
    train_dataset = train_dataset[start_index:end_index]

    generation_config = GenerationConfig(
        max_new_tokens=256,
        do_sample=True, temperature=temperature, 
        num_return_sequences=10,
        num_beams=args.num_beams,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    completions_per_process = []

    for i in tqdm(range(0, len(train_dataset), batch_size), desc=f"Rank [{distributed_state.process_index}] processing: "):
        batch = train_dataset[i:i + batch_size]
        fps = [item["fps"] for item in batch]
        selfies_true = [item["selfies"] for item in batch]
        adducts = [item["adduct"] for item in batch]
        identifiers = [item["identifier"] for item in batch]
        tip_inputs = tokenizer(fps, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
        tip_inputs = {k: v.to(distributed_state.device) for k, v in tip_inputs.items()}
        with torch.inference_mode():
            tip_completion_ids = model.generate(**tip_inputs, generation_config=generation_config)
        tip_completion_ids = tip_completion_ids.cpu()
        answers = [tokenizer.decode(x, skip_special_tokens=True).replace(" ", "") for x in tip_completion_ids]

        if len(answers) > len(batch):  # Top-K generation
            num_return_sequences = len(answers) // len(batch)
            answers = [answers[i * num_return_sequences:(i + 1) * num_return_sequences] for i in range(len(batch))]
        for st, sp, fp, adduct, identifier in zip(selfies_true, answers, fps, adducts, identifiers):
            completions_per_process.append({
                "fps": fp,
                "selfies_true": st,
                "selfies_pred": sp,
                "adduct": adduct,
                "identifier": identifier
            })
    
    torch.cuda.empty_cache()
    completions_gather = gather_object(completions_per_process)
    
    if distributed_state.is_main_process:
        
        evaluator = MoleculeEvaluator(mces=args.compute_mces)
        topk_evaluator = TopkMoleculeEvaluator(mces=args.compute_mces)
        all_preds = [item["selfies_pred"] for item in completions_gather]
        all_labels = [item["selfies_true"] for item in completions_gather]
        all_fps = [item["fps"] for item in completions_gather]
        all_adducts = [item["adduct"] for item in completions_gather]
        all_identifiers = [item["identifier"] for item in completions_gather]

        # Calculate Top1 metrics
        top1_selfies_pred = [s[0] for s in all_preds]
        top1_results = evaluator.evaluate_de_novo_step_selfies(top1_selfies_pred, all_labels)
        print("Top1 results:", top1_results)
        # generated token and gold token statistics
        top1_preds_token_lens = [len(tokenizer.tokenize(s)) for s in top1_selfies_pred]
        labels_token_lens = [len(tokenizer.tokenize(s)) for s in all_labels]

        console = Console()
        table = Table(title="Token Length Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Generated", style="magenta")
        table.add_column("Labels", style="green")
        table.add_row("Mean", f"{np.mean(top1_preds_token_lens)}", f"{np.mean(labels_token_lens)}")
        table.add_row("Max", f"{np.max(top1_preds_token_lens)}", f"{np.max(labels_token_lens)}")
        table.add_row("Min", f"{np.min(top1_preds_token_lens)}", f"{np.min(labels_token_lens)}")
        table.add_row("Std", f"{np.std(top1_preds_token_lens)}", f"{np.std(labels_token_lens)}")
        console.print(table)

        # Calculate Topk metrics
        topk_results = topk_evaluator.evaluate_de_novo_step_selfies_top_k(all_preds, all_labels)
        print("Topk results:", topk_results)
        
        # Save results
        save_predictions = []
        for i in range(len(all_labels)):
            true_label = all_labels[i]
            candidates = list(map(lambda x: [x, tanimoto_scores(true_label, x)], all_preds[i]))
            sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            save_predictions.append({
                "candidates": sorted_candidates,
                "label": true_label,
                "fps": all_fps[i],
                "adduct": all_adducts[i],
                "identifier": all_identifiers[i]
            })
        dir_name = os.path.dirname(args.train_path)
        base_name = os.path.basename(args.train_path)
        base_name = base_name.rsplit('.', 1)[0]
        save_path = os.path.join(dir_name, "candidates", base_name+ ".jsonl")
        save_arr(save_predictions, save_path)
        print("generate data ok")

# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --main_process_port 29400 src/rank_rl/generate_candidates.py --model_path logs/checkpoints/bart-base-4M-pretrain-ft
# CUDA_VISIBLE_DEVICES=2 python src/rank_rl/generate_candidates.py --model_path logs/checkpoints/bart-base-4M-pretrain-ft
