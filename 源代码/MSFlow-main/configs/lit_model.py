import torch
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from torch.nn import CrossEntropyLoss

model_name = 'dfm'
source = 'uniform'  # or 'masked'
d_model= 1535 #767direct  # must be divisble by num of heads, +1 for time embedding
COND_DIM = 512 # 4096 for ECFP
n_layers=12  
n_heads=12  
mlp = 2048  
dropout = 0.3
max_steps =  10000 #200000 cddd
lr = 1e-4 #8e-4
warmup_ratio = 0.08
device = 'cuda' if torch.cuda.is_available() else 'cpu'
scheduler=PolynomialConvexScheduler(n=1.0)
path = MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=1.0))
# loss = MixturePathGeneralizedKL(path=path) # defaultL: CE loss
weighted = False
if(weighted):
    loss = CrossEntropyLoss(reduction='none')
else:
    loss = CrossEntropyLoss()
temperature = 1 #softmax temp
uncond_prob = 0.1 #cfg condition drop probability
T_STEPS = 128
