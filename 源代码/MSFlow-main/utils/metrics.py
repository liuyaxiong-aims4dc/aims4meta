import logging
from typing import List, Tuple, Dict, Collection
import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem
from collections import Counter
from safe import decode as safe_decode, SAFEDecodeError

# Mute RDKit logging
RDLogger.logger().setLevel(RDLogger.CRITICAL)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def is_valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


def get_mols(smiles_list: Collection[str]):
    return [Chem.MolFromSmiles(s) for s in smiles_list if s is not None]


def get_fingerprints(mols: List[Chem.Mol]):
    return [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=1024)
            for m in mols if m is not None]


def calculate_internal_pairwise_similarities(smiles_list: Collection[str]) -> np.ndarray:
    mols = get_mols(smiles_list)
    fps = get_fingerprints(mols)
    n = len(fps)
    similarities = np.zeros((n, n))

    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        similarities[i, :i] = sims
        similarities[:i, i] = sims

    return similarities


def decode_tokens_to_smiles(
    samples: List,
    ID2TOK: Dict[int, str],
    TOK2ID: Dict[str, int],
    PAD: str = ''
) -> Tuple[List[str], List[str]]:
    """
    Decodes a list of token sequences into SMILES strings using SAFE.
    Returns:
        valid_safe_strings: List of successfully decoded SAFE strings
        decoded_smiles: List of valid SMILES
    """
    valid_safe_strings = []
    decoded_smiles = []

    for s in samples:
        try:
            ids = s.tolist() if hasattr(s, 'tolist') else list(s)
            safe_str = "".join(ID2TOK[i] for i in ids if i != TOK2ID[PAD])
            smiles = safe_decode(safe_str, canonical=True)
            valid_safe_strings.append(safe_str)
            decoded_smiles.append(smiles)
        except SAFEDecodeError:
            continue  # skip invalid ones

    return valid_safe_strings, decoded_smiles


def compute_smiles_metrics(
    total_samples: int,
    decoded_smiles: List[str]
) -> Dict[str, float]:
    """
    Computes validity, uniqueness, and diversity for a list of decoded SMILES.

    Validity is computed as:
        valid SMILES / total number of input samples (including SAFE decoding failures)
    """
    if total_samples == 0:
        return {"validity": 0.0, "uniqueness": 0.0, "diversity": 0.0}

    valid_smiles = [s for s in decoded_smiles if is_valid_smiles(s)]
    if not valid_smiles:
        return {"validity": 0.0, "uniqueness": 0.0, "diversity": 0.0}

    unique_smiles = list(set(valid_smiles))
    sim_matrix = calculate_internal_pairwise_similarities(unique_smiles)
    n = len(unique_smiles)
    diversity = 1.0 - (np.sum(sim_matrix) / (n * (n - 1))) if n > 1 else 0.0
    return {
        "validity": len(valid_smiles) / total_samples,
        "uniqueness": len(unique_smiles) / len(valid_smiles),
        "diversity": diversity,
    }



def get_topk_molecules(smiles_list, k=10):
    """
    Given a list of SMILES strings (from DiffMS samples), 
    returns the top-k valid molecules ranked by frequency.
    
    Args:
        smiles_list (list of str): SMILES strings generated for one spectrum.
        k (int): number of top molecules to return.
        
    Returns:
        list of tuples: [(smiles, count), ...] sorted by count (descending).
    """
    valid_smiles = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue  # invalid molecule
        # check if molecule is connected (single fragment)
        if len(Chem.GetMolFrags(mol)) == 1:
            valid_smiles.append(Chem.MolToSmiles(mol))  # canonicalize

    # count frequency
    counts = Counter(valid_smiles)

    # get top-k
    topk = counts.most_common(k)

    return topk