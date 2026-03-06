import pandas as pd
from pathlib import Path
import copy
import numpy as np
import torch
from tqdm import tqdm

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

        self.device = torch.device("cpu")
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
        self.kwargs['device'] = "cpu"
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
    dataset_name = "msg"
    threshold = 0.5
    fp_ckpt = "../pretrained_models/mist_fp_canopus_pretrain.ckpt"
    res_dir = f"./{dataset_name}/mist/"
    mgf_input = f"./{dataset_name}/{dataset_name}-demo.mgf"
    labels = f"./{dataset_name}/{dataset_name}-labels.tsv"

    predictor = MISTPredictor(fp_ckpt, res_dir, mgf_input, labels)
    predictor.assign_subformulae()
    output_preds, output_names = predictor.predict()
    print(output_preds.shape, len(output_names))
    indices_list = [np.where(row > threshold)[0].tolist() for row in output_preds]
    print(indices_list)
    if dataset_name == "msd":
        for i in range(10):
            print("\n\n")
            print("10ev: ", indices_list[i])
            print("20ev: ", indices_list[i+10])
            print("30ev: ", indices_list[i+20])
    if dataset_name == "msg":
        for i in range(10):
            print("morgan 4096: ", indices_list[i])
    