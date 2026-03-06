import torch
from torch.utils.data import Dataset
import numpy as np
from utils.functions import smiles_to_fps
from configs.lit_model import COND_DIM

class MolDataset(Dataset):
    def __init__(self, encoded_seqs):
        self.data = torch.tensor(np.array(encoded_seqs),dtype=torch.long)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    


class CondMolDataset(Dataset):
    def __init__(self, encoded_seqs,conditions):
        self.data = torch.tensor(np.array(encoded_seqs),dtype=torch.long)
        if COND_DIM == 4096:
            cond_fp = smiles_to_fps(conditions, radius=2,fp_size=4096)
            self.conditions = torch.tensor(np.array(cond_fp), dtype = torch.float32)
        else:
            self.conditions = torch.tensor(np.array(conditions), dtype = torch.float32)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.conditions[idx]
    