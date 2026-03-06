from sampling.samplers import  sample_flow
from sampling.cfg_samplers import sample_flow_cond
from configs.lit_model import *
from configs.data import *


# unconditional sampling 
def generate_mols(model, num_samples = 1000,steps = 100,path = path, seq_len=MAX_LEN,source_distribution='uniform',mask_token_id=0, device = 'cuda',temperature = temperature):
    samples = sample_flow(num_samples=num_samples,model=model,steps = steps, path =path, vocab_size=vocab_size,seq_len=seq_len,source_distribution=source_distribution,mask_token_id=mask_token_id,device=device,temperature=temperature)
    return samples

# Sampling with classifier-free-guidance
def cond_generate_mols(model, cond, guidance_scale=1.5, num_samples = 100, steps = 128 ,path = path, seq_len=MAX_LEN,source_distribution='uniform',mask_token_id=0, device = 'cuda',temperature = temperature):
    samples = sample_flow_cond(num_samples=num_samples,model=model,cond = cond,guidance_scale=guidance_scale, steps = steps, path =path, vocab_size=vocab_size,seq_len=seq_len,source_distribution=source_distribution,mask_token_id=mask_token_id,device=device,temperature=temperature)
    return samples