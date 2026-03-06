from rdkit import Chem
import pandas as pd


FUNCTIONAL_GROUPS_SMARTS = {
    # Carbonyl-containing / oxygen-based
    "ester": "C(=O)O[C,O]",
    "amide": "C(=O)N",
    "carboxylate": "[CX3](=O)[O-]",
    "lactone": "O=C1OC1",  # cyclic ester, optional

    # Sulfur-containing
    "sulfoxide": "S(=O)[#6]",
    "sulfone": "S(=O)(=O)[#6]",
    "thioether": "[#16][CX4]",  # R-S-R
    "thiol": "[SX2H]",

    # Phosphorus-containing
    "phosphate": "P(=O)(O)O",
    "phosphonate": "P(=O)(O)[#6]",

    # Nitrogen / reactive
    "nitro": "[NX3](=O)=O",
    "azide": "[N-]=[N+]=N",
    "nitrile": "C#N",

    # Aromatic / heteroaromatic
    "heteroaromatic": "a[!c]",  # any aromatic atom that is not carbon
    "polycyclic_aromatic": "c1cc2ccccc2cc1",  # fused aromatic rings, example SMARTS

    # Oxygen / reactive
    "epoxide": "C1OC1",
    "peroxide": "OO"
}


def extract_functional_groups(smiles, fg_smarts):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    
    present_groups = set()
    for name, smarts in fg_smarts.items():
        pattern = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(pattern):
            present_groups.add(name)
    return present_groups
def fg_scores(true_smiles, pred_smiles, fg_smarts):
    true_fg = extract_functional_groups(true_smiles, fg_smarts)
    pred_fg = extract_functional_groups(pred_smiles, fg_smarts)

    tp = len(true_fg & pred_fg)
    fp = len(pred_fg - true_fg)
    fn = len(true_fg - pred_fg)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )
    return precision, recall, f1

if __name__ == "__main__":
    # Example lists of SMILES strings for testing
    query_smiles_list = ["P(=O)(O)O", "CC(=O)[O-].[Na+]","c1ccc2cc3ccccc3cc2c1"]      # replace by ground truth smiles
    generated_smiles_list = ["P(=O)(O)O", "CC(=O)" ,"c1ccc2cc3ccccc3cc2c1"] # replace by top-1 predicted smiles

    scores = [
        fg_scores(t, p, FUNCTIONAL_GROUPS_SMARTS)
        for t, p in zip(query_smiles_list, generated_smiles_list)
    ]

    df_scores = pd.DataFrame(scores, columns=["fg_precision", "fg_recall", "fg_f1"])
    mean_precision = df_scores["fg_precision"].mean()
    mean_recall = df_scores["fg_recall"].mean()
    mean_f1 = df_scores["fg_f1"].mean()
    std_f1 = df_scores["fg_f1"].std()

    print(f"FG Precision: {mean_precision:.3f}")
    print(f"FG Recall: {mean_recall:.3f}")
    print(f"FG F1: {mean_f1:.3f} ± {std_f1:.3f}")