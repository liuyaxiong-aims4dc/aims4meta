import sys
sys.path.append('MSFlow/ms_scripts/DiffMS/src')# Adjust the path as needed to point to the DiffMS/src directory
import numpy as np
from DiffMS.src.datasets import spec2mol_dataset
from hydra import compose, initialize
import torch
import torch.nn as nn
from DiffMS.src.mist.models.spectra_encoder import SpectraEncoderGrowing
from tqdm import tqdm
from hydra import compose, initialize
import warnings
from rdkit.Chem import  MolFromInchi, MolToSmiles
warnings.filterwarnings('ignore')
from rdkit import rdBase
import pandas as pd
import torch.nn.functional as F

blocker = rdBase.BlockLogs()

def replace_sigmoid_with_tanh(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Sigmoid):
            setattr(module, name, nn.Tanh())
        else:
            replace_sigmoid_with_tanh(child)
def batch_to_device(batch, device):
    """
    Recursively move a batch (dict, list, tuple, tensor) to the given device.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [batch_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(batch_to_device(v, device) for v in batch)
    else:
        return batch  # leave other types (ints, strings) unchanged

with initialize(version_base=None, config_path="DiffMS/configs", job_name="test_app"):
    cfg = compose(config_name="config")

dataset_config = cfg["dataset"]

if dataset_config["name"] not in ("canopus", "msg"):
    raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

datamodule = spec2mol_dataset.Spec2MolDataModule(cfg)
    
            
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = 'MSFlow/checkpoints/MSFlow/Encoder/encoder_msg_cddd.pt' # Better to set absolute path of the downloaded checkpoint checkpoint
cddd_model = torch.load(checkpoint, map_location=torch.device(device))
# encoder_hidden_dim= 256           # Small Model Default (CANOPUS)
# encoder_magma_modulo= 512         # Small Model Default (CANOPUS)
encoder_hidden_dim= 512          # Large Model Default (MSG)
encoder_magma_modulo= 2048       # Large Model Default (MSG)
encoder = SpectraEncoderGrowing(
            inten_transform='float',
            inten_prob=0.1,
            remove_prob=0.5,
            peak_attn_layers=2,
            num_heads=8,
            pairwise_featurization=True,
            embed_instrument=False,
            cls_type='ms1',
            set_pooling='cls',
            spec_features='peakformula',
            mol_features='fingerprint',
            form_embedder='pos-cos',
            output_size=512,
            hidden_size=encoder_hidden_dim,
            spectra_dropout=0.0,
            top_layers=1,
            refine_layers=4,
            magma_modulo=encoder_magma_modulo,
        )
encoder.load_state_dict(cddd_model['model_state_dict'])
replace_sigmoid_with_tanh(encoder)
print(encoder)
results = []
encoder.to(device)
encoder.eval()
for data in tqdm(datamodule.test_dataloader()): 
    data = batch_to_device(data,device)  
    outputs,aux = encoder(data)
    results.append(outputs)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    del outputs
    del data
predictions = torch.cat(results)
predictions = predictions.detach().cpu().numpy()
print(predictions.shape)
inchis = []
smiles = []
for i, _ in tqdm(enumerate(range(len(datamodule.test_dataset)))):
    inchi = datamodule.test_dataset[i]['graph'][0].inchi
    inchis.append(inchi)
    smiles.append(MolToSmiles(MolFromInchi(inchi),isomericSmiles=False))

df_test = pd.DataFrame({'inchi': inchis,
                        'canon_smiles': smiles,
                   'cddd': [row for row in predictions]
                   })
df_test.to_csv('../example_conditions/msg_test_cddd.csv')