import random, numpy as np, torch
import json

training_data_path = '/path/to/parquet/training_data/'  # example: # data_path = '/path/to/combined_training_data.parquet'
val_data_path = '/path/to/parquet/val_data/'
vocab_path = 'vocab.json'
output_path = '/path/to/checkpoints/'
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
SPECIAL_TOKENS = ['MASK', 'PAD']        
MASK, PAD = SPECIAL_TOKENS
with open(vocab_path, "rb") as f:
    vocab = json.load(f)
TOK2ID, ID2TOK = vocab['tok2id'], vocab['id2tok']
TOK2ID = {k: int(v) for k, v in TOK2ID.items()} 
ID2TOK = {int(k): v for k, v in ID2TOK.items()}
vocab_size = len(TOK2ID) 

batch_size = 256  
MAX_LEN = 128         