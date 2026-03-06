from modules.cond_lit_model import CondFlowMolBERTLitModule
import torch
from flow_matching.path import MixtureDiscreteProbPath
from utils.sample import cond_generate_mols
from utils.metrics import decode_tokens_to_smiles
from configs import *
import os
import pandas as pd
from utils.functions import canonicalize
from flow_matching.path.scheduler import PolynomialConvexScheduler
torch.serialization.add_safe_globals([MixtureDiscreteProbPath, PolynomialConvexScheduler, torch.nn.modules.loss.CrossEntropyLoss])

cfm_module = CondFlowMolBERTLitModule.load_from_checkpoint('checkpoints/MSFlow/Decoder/MSFlow_cddds.ckpt')
model = cfm_module.model
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
condition_df = pd.read_csv('example_conditions/canopus_cddd_example.csv') 
query_smiles = condition_df.canon_smiles
cddd_list = condition_df.iloc[0,3:].tolist() # specify columns corresponding to cddd  or df['cddd'] depding on format
cddd_np = np.array(cddd_list, dtype=np.float32)
cond = torch.tensor(cddd_np, dtype=torch.float32).to(device)
samples = cond_generate_mols(
    model,
    cond=cond,
    source_distribution='uniform',
    num_samples=100,
    steps=128,
    guidance_scale=1.5,
    temperature=1,
    device=device,
    )
_, smiles = decode_tokens_to_smiles(samples, ID2TOK=ID2TOK, TOK2ID=TOK2ID, PAD=PAD)
smiles = [canonicalize(s) for s in smiles if s]
smiles = [s for s in smiles if s is not None]
print('Example of a generated SMILES:', smiles[0])