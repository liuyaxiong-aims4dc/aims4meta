
import sys, os, numpy as np, torch, pandas as pd
os.chdir("/stor3/AIMS4Meta/\u6e90\u4ee3\u7801/MSFlow-main")
sys.path.insert(0, '.')
from modules.cond_lit_model import CondFlowMolBERTLitModule
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from utils.sample import cond_generate_mols
from utils.metrics import decode_tokens_to_smiles
from utils.functions import canonicalize
from configs import ID2TOK, TOK2ID, PAD
torch.serialization.add_safe_globals([MixtureDiscreteProbPath, PolynomialConvexScheduler, torch.nn.modules.loss.CrossEntropyLoss])
dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Device:', dev)
cfm = CondFlowMolBERTLitModule.load_from_checkpoint('checkpoints/Decoder/MSFlow_cddds.ckpt', map_location=dev, strict=False)
dec = cfm.model; dec.eval(); dec.to(dev)
cddd_all = torch.from_numpy(np.load('/tmp/cddd_encoded.npy')).to(dev)
test_ids = [l.strip() for l in open('/tmp/eval_testset.txt') if l.strip()]
results = []
for i in range(min(len(cddd_all), len(test_ids))):
    cond = cddd_all[i].unsqueeze(0).expand(5, -1)
    samps = cond_generate_mols(dec, cond=cond, source_distribution='uniform',
        num_samples=5, steps=128, guidance_scale=1.5, temperature=1, device=dev)
    _, smi = decode_tokens_to_smiles(samps, ID2TOK=ID2TOK, TOK2ID=TOK2ID, PAD=PAD)
    smi = [canonicalize(s) for s in smi if s]
    smi = [s for s in smi if s is not None]
    for rank, s in enumerate(smi, 1):
        if s: results.append({'query_name': test_ids[i], 'rank': rank, 'smiles': s})
    print(f'  {test_ids[i]}: {len(smi)} cands')
pd.DataFrame(results).to_csv('/tmp/msflow_final_results.csv', index=False)
print(f'Done: {len(results)} total')
