import typing as T
from rdkit import Chem
import numpy as np
from rdkit.DataStructs import TanimotoSimilarity
from rdkit import RDLogger
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from multiprocessing import cpu_count
from tqdm import tqdm
from selfies import decoder


import sys
sys.path.append('.')
from src.utils import morgan_fp, MyopicMCES

RDLogger.DisableLog('rdApp.*')

class MoleculeEvaluator:
    def __init__(self, mces: bool = False):
        self.mces = mces
        if mces:
            print("MCES is enabled")
            self.myopic_mces = MyopicMCES()
        self.fps_cache = {}

    def calculate_mces(self, pred_true_smiles: T.Tuple[str, str]) -> float:
        pred_smi, true_smi = pred_true_smiles
        if pred_smi is None or true_smi is None:
            return 100
        try:
            mce_val = self.myopic_mces(true_smi, pred_smi)
        except Exception as e:
            print(f'ERROR: {true_smi} {pred_smi} MCES {e}')
            mce_val = 100
        return mce_val

    def calculate_tanimoto(self, pred_true_mols_smiles: T.Tuple[Chem.Mol, Chem.Mol, str, str]) -> float:
        pred_mol, true_mol, pred_smi, true_smi = pred_true_mols_smiles
        if pred_smi is None or true_smi is None:
            return 0.0
        try:
            if pred_smi not in self.fps_cache:
                self.fps_cache[pred_smi] = morgan_fp(pred_mol, to_np=False)
            if true_smi not in self.fps_cache:
                self.fps_cache[true_smi] = morgan_fp(true_mol, to_np=False)
            sim = TanimotoSimilarity(self.fps_cache[true_smi], self.fps_cache[pred_smi])
        except Exception as e:
            print(f"Tanimoto error: {e}")
            sim = 0.0
        return sim

    def is_selfies_valid_with_rdkit(self, selfies_str: str) -> T.Tuple[T.Optional[Chem.Mol], T.Optional[str]]:
        try:
            smiles = decoder(selfies_str)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None, None
            return mol, Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            return None, None

    def calculate_mol_from_selfies(self, pred_true_selfies: T.Tuple[str, str]) -> T.Optional[T.List[T.List]]:
        pred_selfies, true_selfies = pred_true_selfies
        pred_mol, canonical_pred_smiles = self.is_selfies_valid_with_rdkit(pred_selfies)
        true_mol, canonical_true_smiles = self.is_selfies_valid_with_rdkit(true_selfies)
        if pred_mol and true_mol:
            return [[pred_mol, canonical_pred_smiles], [true_mol, canonical_true_smiles]]
        return None

    def evaluate_de_novo_step_selfies(self, selfies_pred: T.List[str], selfies_true: T.List[str]) -> T.Dict[str, float]:
        cpu_count = 20
        print("Multi-processing to calculate mol and canonical smiles: ", cpu_count)

        valid_mols_pred, valid_mols_true, valid_smiles_pred, valid_smiles_true = [], [], [], []

        with ProcessPool(max_workers=cpu_count) as pool:
            mols_future = pool.map(self.calculate_mol_from_selfies, zip(selfies_pred, selfies_true))
            mols_results = mols_future.result()
            with tqdm(total=len(selfies_true), desc="Calculate mol and canonical smiles: ") as progress_bar:
                while True:
                    try:
                        result = next(mols_results)
                        if result:
                            valid_mols_pred.append(result[0][0])
                            valid_mols_true.append(result[1][0])
                            valid_smiles_pred.append(result[0][1])
                            valid_smiles_true.append(result[1][1])
                    except StopIteration:
                        break
                    except TimeoutError as error:
                        print(error, flush=True)
                    except Exception as error:
                        print(error, flush=True)
                        exit()
                    progress_bar.update(1)

        if  len(valid_mols_pred) == 0:
            return {
                "top1_valid_mols": 0,
                "top1_mces_dist": 100,
                "top1_tanimoto_sim": 0,
                "top1_mol_accuracy": 0,
            }

        metric_vals = {
            "top1_valid_mols": len(valid_mols_pred) / len(selfies_pred),
            "top1_mces_dist": 100,
            "top1_tanimoto_sim": 0,
            "top1_mol_accuracy": 0,
        }

        tanimoto_sims = []
        with ProcessPool(max_workers=cpu_count) as pool:
            mols_future = pool.map(self.calculate_tanimoto, zip(valid_mols_pred, valid_mols_true, valid_smiles_pred, valid_smiles_true))
            mols_results = mols_future.result()
            with tqdm(total=len(valid_smiles_true), desc="Calculate valid tanimoto: ") as progress_bar:
                while True:
                    try:
                        result = next(mols_results)
                        tanimoto_sims.append(result)
                    except StopIteration:
                        break
                    except TimeoutError as error:
                        print(error, flush=True)
                    except Exception as error:
                        print(error, flush=True)
                        exit()
                    progress_bar.update(1)
        metric_vals["top1_tanimoto_sim"] =float(np.mean(tanimoto_sims))

        if self.mces:
            mces_dists = []
            with ProcessPool(max_workers=cpu_count) as pool:
                mces_future = pool.map(self.calculate_mces, zip(valid_smiles_pred, valid_smiles_true))
                mces_results = mces_future.result()
                with tqdm(total=len(valid_smiles_true), desc="Calculate MCES: ") as progress_bar:
                    while True:
                        try:
                            result = next(mces_results)
                            mces_dists.append(result)
                        except StopIteration:
                            break
                        except Exception as error:
                            print(error, flush=True)
                            exit()
                        progress_bar.update(1)
            metric_vals["top1_mces_dist"] = float(np.mean(mces_dists))

        correct_cnt = sum(1 for pred_smi, true_smi in zip(valid_smiles_pred, valid_smiles_true) if pred_smi == true_smi)
        metric_vals["top1_mol_accuracy"] = correct_cnt / len(valid_smiles_pred)

        for key in metric_vals:
            metric_vals[key] = round(metric_vals[key], 4)

        return metric_vals

class TopkMoleculeEvaluator:
    def __init__(self, mces: bool = False):
        self.fps_cache = {}
        self.mces = mces
        if mces:
            print("MCES is enabled")
            self.myopic_mces = MyopicMCES()

    def get_canonical_smiles_from_selfies(self, pred_true_selfies):
        def get_canonical_smiles(selfies_str):
            try:
                smiles = decoder(selfies_str)
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None
                return Chem.MolToSmiles(mol, canonical=True)
            except Exception as e:
                return None
        pred_selfies_topk, true_selfies = pred_true_selfies
        pred_smiles_topk = [get_canonical_smiles(selfies) for selfies in pred_selfies_topk]
        true_smiles = get_canonical_smiles(true_selfies)
        if true_smiles is None: return None
        return pred_smiles_topk, true_smiles
    
    def calculate_tanimoto(self, pred_true_smiles):
        pred_smi_topk, true_smi = pred_true_smiles
        if true_smi is None:
            return 0.0
        topk_tanimoto_sims = []
        for pred_smi in pred_smi_topk:
            if pred_smi is None:
                continue
            try:
                if pred_smi not in self.fps_cache:
                    pred_mol = Chem.MolFromSmiles(pred_smi)
                    self.fps_cache[pred_smi] = morgan_fp(pred_mol, to_np=False)
                if true_smi not in self.fps_cache:
                    true_mol = Chem.MolFromSmiles(true_smi)
                    self.fps_cache[true_smi] = morgan_fp(true_mol, to_np=False)
                sim = TanimotoSimilarity(self.fps_cache[true_smi], self.fps_cache[pred_smi])
                topk_tanimoto_sims.append(sim)
            except Exception as e:
                print(f"Tanimoto error: {e}")
        if len(topk_tanimoto_sims) == 0:
            return 0
        return np.max(topk_tanimoto_sims)
    
    def calculate_mces(self, pred_true_smiles):
        pred_smi_topk, true_smi = pred_true_smiles
        if true_smi is None:
            return 100.0
        topk_mces = []
        for pred_smi in pred_smi_topk:
            if pred_smi is None:
                continue
            try:
                mce_val = self.myopic_mces(true_smi, pred_smi)
                topk_mces.append(mce_val)
            except Exception as e:
                print(f'MCES error: {e}')
        if len(topk_mces) == 0:
            return 100
        return np.min(topk_mces)
    
    def evaluate_de_novo_step_selfies_top_k(self, selfies_pred_topk: T.List[T.List[str]], selfies_true: T.List[str]) -> T.Dict[str, float]:
        cpu_count = 20
        print("Multi-processing to calculate canonical smiles: ", cpu_count)

        canonical_smiles_pred_topk, canonical_smiles_true = [], []
        with ProcessPool(max_workers=cpu_count) as pool:
            smiles_future = pool.map(self.get_canonical_smiles_from_selfies, zip(selfies_pred_topk, selfies_true))
            smiles_results = smiles_future.result()
            with tqdm(total=len(selfies_true), desc="Calculate canonical smiles: ") as progress_bar:
                while True:
                    try:
                        result = next(smiles_results)
                        if result:
                            canonical_smiles_pred_topk.append(result[0])
                            canonical_smiles_true.append(result[1])
                    except StopIteration:
                        break
                    except Exception as error:
                        print(error, flush=True)
                        exit()
                    progress_bar.update(1)
        valid_cnt, total_cnt = 0, 0
        for item in canonical_smiles_pred_topk:
            for smiles in item:
                total_cnt += 1
                if smiles is not None:
                    valid_cnt += 1
        if valid_cnt == 0:
            return {
                "topk_valid_mols": 0,
                "topk_mces_dist": 100,
                "topk_tanimoto_sim": 0,
                "topk_mol_accuracy": 0,
            }
        metric_vals = {
            "topk_valid_mols": valid_cnt / total_cnt if total_cnt > 0 else 0,
            "topk_mces_dist": 100,
            "topk_tanimoto_sim": 0,
            "topk_mol_accuracy": 0,
        }
        
        tanimoto_sims = []
        with ProcessPool(max_workers=cpu_count) as pool:
            mols_future = pool.map(self.calculate_tanimoto, zip(canonical_smiles_pred_topk, canonical_smiles_true))
            mols_results = mols_future.result()
            with tqdm(total=len(canonical_smiles_true), desc="Calculate valid tanimoto: ") as progress_bar:
                while True:
                    try:
                        result = next(mols_results)
                        tanimoto_sims.append(result)
                    except StopIteration:
                        break
                    except Exception as error:
                        print(error, flush=True)
                        exit()
                    progress_bar.update(1)
        metric_vals["topk_tanimoto_sim"] =float(np.mean(tanimoto_sims))

        if self.mces:
            mces_dists = []
            with ProcessPool(max_workers=cpu_count) as pool:
                mces_future = pool.map(self.calculate_mces, zip(canonical_smiles_pred_topk, canonical_smiles_true))
                mces_results = mces_future.result()
                with tqdm(total=len(canonical_smiles_true), desc="Calculate MCES: ") as progress_bar:
                    while True:
                        try:
                            result = next(mces_results)
                            mces_dists.append(result)
                        except StopIteration:
                            break
                        except Exception as error:
                            print(error, flush=True)
                            exit()
                        progress_bar.update(1)
            metric_vals["topk_mces_dist"] = float(np.mean(mces_dists))

        correct_cnt = 0
        for smiles_topk, smiles_true in zip(canonical_smiles_pred_topk, canonical_smiles_true):
            if smiles_true in smiles_topk:
                correct_cnt += 1
        metric_vals["topk_mol_accuracy"] = correct_cnt / len(canonical_smiles_true)
        for key in metric_vals:
            metric_vals[key] = round(metric_vals[key], 4)
        
        return metric_vals

if __name__ == "__main__":
    evaluator = MoleculeEvaluator()
    selfies_pred = ["[C][O]", "[C][C][O]"]
    selfies_true = ["[C][O][C]", "[C][C][O]"]
    metrics1 = evaluator.evaluate_de_novo_step_selfies(selfies_pred, selfies_true)
    print(metrics1)

    topk_evaluator = TopkMoleculeEvaluator()
    selfies_pred_topk = [["[C][O]", "[C][C][O]"], ["[C][C][O]", "[C][O]"]]
    selfies_true = ["[C][O][C]", "[C][C][O]"]
    metrics2 = topk_evaluator.evaluate_de_novo_step_selfies_top_k(selfies_pred_topk, selfies_true)
    print(metrics2)

