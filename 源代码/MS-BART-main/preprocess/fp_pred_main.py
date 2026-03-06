import pandas as pd
from pathlib import Path
import copy
import numpy as np
import torch
from tqdm import tqdm
import selfies as sf
from rdkit import Chem

import sys
sys.path.append("./mist/src")
from mist.utils.plot_utils import *
import mist.subformulae.assign_subformulae as assign_subformulae
import mist.models.base as base
import mist.data.datasets as datasets
import mist.data.featurizers as featurizers

class MISTPredictor:
    def __init__(self, fp_ckpt, res_dir, mgf_input, labels):
        self.fp_ckpt = fp_ckpt
        self.res_dir = Path(res_dir)
        self.mgf_input = mgf_input
        self.labels = labels
        self.res_dir.mkdir(exist_ok=True)
        self.subform_dir = self.res_dir / "subforms_fp"
        self.subform_dir.mkdir(exist_ok=True, parents=True)

        self.device = torch.device("cuda:0")
        self.test_dataset = None

        self.load_model()

    def assign_subformulae(self):
        assign_subformulae.assign_subforms(
            spec_files=self.mgf_input,
            labels_file=self.labels,
            output_dir=self.subform_dir,
            mass_diff_thresh=20,
            max_formulae=50,
            num_workers=32,
            feature_id="FEATURE_ID",
            debug=False
        )

    def load_model(self):
        fp_model = torch.load(self.fp_ckpt, map_location=self.device)
        main_hparams = fp_model["hyper_parameters"]
        self.kwargs = copy.deepcopy(main_hparams)
        self.kwargs['device'] = "cuda:0"
        self.kwargs['num_workers'] = 0
        self.kwargs['subform_folder'] = self.subform_dir
        self.kwargs['labels_file'] = self.labels

        self.model = base.build_model(**self.kwargs)
        self.model.load_state_dict(fp_model["state_dict"])
        self.model = self.model.to(self.device)
        self.model = self.model.eval()


    def prepare_dataset(self):
        self.kwargs["spec_features"] = self.model.spec_features(mode="test") # 'peakformula_test'
        self.kwargs['mol_features'] = "none"
        self.kwargs['allow_none_smiles'] = True
        paired_featurizer = featurizers.get_paired_featurizer(**self.kwargs)

        spectra_mol_pairs = datasets.get_paired_spectra(**self.kwargs)
        spectra_mol_pairs = list(zip(*spectra_mol_pairs))

        self.test_dataset = datasets.SpectraMolDataset(
            spectra_mol_list=spectra_mol_pairs, featurizer=paired_featurizer, **self.kwargs
        )

    def predict(self):
        self.prepare_dataset()
        output_preds = (
            self.model.encode_all_spectras(self.test_dataset, no_grad=True, **self.kwargs).cpu().numpy()
        )
        output_names = self.test_dataset.get_spectra_names()
        return output_preds, output_names

# Example usage
if __name__ == "__main__":
    dataset_name = "MassSpecGym" # MassSpecGym/CANOPUS
    fp_ckpt = f"./data/{dataset_name}/mist/mist.ckpt"
    res_dir = f"./data/{dataset_name}/mist/"
    mgf_input = f"./data/{dataset_name}/{dataset_name}.mgf"
    labels = f"./data/{dataset_name}/{dataset_name}_labels.tsv"

    predictor = MISTPredictor(fp_ckpt, res_dir, mgf_input, labels)
    predictor.assign_subformulae()
    output_preds, output_names = predictor.predict()
    print(output_preds.shape, len(output_names))

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        indices_list = [np.where(row > threshold)[0].tolist() for row in output_preds]
        name_fps_keys = {name: fps for name, fps in zip(output_names, indices_list)}

        if dataset_name == "MassSpecGym":
            df = pd.read_csv("MassSpecGym/data/MassSpecGym.tsv", sep='\t')
        elif dataset_name == "CANOPUS":
            canopus_split = pd.read_csv('./data/datasets/CANOPUS/splits/canopus_hplus_100_0.tsv', sep='\t')
            canopus_labels = pd.read_csv('./data/datasets/CANOPUS/labels.tsv', sep='\t')
            canopus_labels["name"] = canopus_labels["spec"]
            df = canopus_labels.merge(canopus_split, on="name")
        else:
            raise ValueError("Unknown dataset name")
        print(f"Before processing, the number of entries is: {len(df)}")
        # Add FPS column/Selfies column
        df['fps'] = ''
        df['selfies'] = ''
        
        # Record the indices of rows that need to be deleted (those where FPS or Selfies generation failed)
        to_drop = []
        for idx, row in df.iterrows():
            identifier_key = "identifier" if dataset_name == "MassSpecGym" else "name"
            identifier = row[identifier_key]
            smiles = row['smiles']
            if identifier not in name_fps_keys:
                to_drop.append(idx)
                print(f"identifier {identifier} not in name_fps_keys")
                continue
            fps = name_fps_keys[identifier]
            df.at[idx, 'fps'] = "".join([f"<fp{fp:04d}>" for fp in fps])
            try:
                # Convert to standard SELFIES
                mol = Chem.MolFromSmiles(smiles)
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                selfies_str = sf.encoder(canonical_smiles)
                df.at[idx, 'selfies'] = selfies_str
            except:
                print(f"Error in encode smiles to selfies: {smiles}")
                to_drop.append(idx)
                continue
        # Delete rows where SELFIES generation failed
        df = df.drop(index=to_drop)
        print(f"Final number of entries: {len(df)}")

        df.to_csv(f"./data/{dataset_name}/{dataset_name}_fps_selfies_threshold_{threshold}.tsv", sep='\t', index=False)
