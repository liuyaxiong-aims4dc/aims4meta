import pandas as pd
import safe  
import json
from utils.functions import canonicalize
from configs.data import MAX_LEN

SPECIAL_TOKENS = ['MASK', 'PAD']        
MASK, PAD = SPECIAL_TOKENS

# --- Load existing vocab ---
vocab_path = "/path/to/vocab/vocab.json"
with open(vocab_path, "r") as f:
    vocab_data = json.load(f)

TOK2ID = vocab_data["tok2id"]
TOK2ID = {str(k): int(v) for k, v in TOK2ID.items()}


def encode_row(s):
    """Encode a SMILES string into SAFE + tokens."""
    try:
        s = canonicalize(s)
        encoded = safe.encode(s)
        tokens = list(safe.split(encoded))
        return  encoded, tokens, len(tokens)
    except safe.SAFEFragmentationError:
        print("fragm error found")
        # if SAFE fails, skip it
        return None
    except safe.SAFEEncodeError:
        print("encoder error found")
        # if SAFE fails, skip it
        return None
    except safe.SAFEDecodeError:
        print("decoder error found")
        # if SAFE fails, skip it
        return None
    except:
        print("safe error found")
        # if SAFE fails, skip it
        return None

def encode(tokens: list[str], TOK2ID, MAX_LEN) -> list[int]:
    """Convert tokens to IDs, skip sample if any token not in vocab."""
    if any(t not in TOK2ID for t in tokens):
        print("new token found")
        return None
    if(len(tokens)<= MAX_LEN):
        return [TOK2ID[t] for t in tokens] + [TOK2ID[PAD]] * (MAX_LEN - len(tokens))
    else: 
        return None

path = "/path/to/smiles/data/"
df = pd.read_pickle(path)
df['results'] = df['canon_smiles'].swifter.apply(encode_row)
df = df[df['results'].notnull()].copy()
df['SAFE'], df['safe_tokens'], df['seq_len'] = zip(*(df['results']))
df.drop(columns=['results'],inplace= True)
df['encoded'] = df['safe_tokens'].swifter.apply(lambda tokens: encode(tokens,TOK2ID, MAX_LEN))
df = df[df['encoded'].notnull()].copy()
print("file successfully saved")
df.to_parquet('/path/to/output/data/training_data.parquet')
