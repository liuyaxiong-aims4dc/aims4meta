import torch
from rdkit import Chem
import torch.nn as nn
import numpy as np
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from tqdm import tqdm


def smiles_to_fps(smiles_list, radius=3, fp_size=1024):
    # Ensure input is a list
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]

    # Initialize Morgan fingerprint generator
    morgan_gen = GetMorganGenerator(radius=radius, fpSize=fp_size)

    # Convert SMILES to molecules
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    mols = [mol for mol in mols if mol is not None]  # filter invalid

    fps_np = []
    for mol in tqdm(mols, desc='Converting SMILES to bit fingerprints'):
        fp = morgan_gen.GetFingerprint(mol)
        arr = np.zeros((fp_size,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps_np.append(arr)

    return np.array(fps_np)  




def smiles_to_cfps(smiles_list, radius=2, fp_size=4096):
    # Ensure input is a list
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]

    # Convert SMILES to molecules
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    mols = [mol for mol in mols if mol is not None]  # filter invalid

    fps_np = []
    for mol in tqdm(mols, desc='Converting SMILES to count fingerprints'):
        fp = AllChem.GetHashedMorganFingerprint(mol, radius=radius, nBits=fp_size)
        arr = np.zeros((0,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps_np.append(arr)

    return np.array(fps_np)  

def gumbel_sigmoid(logits, temperature=0.5, hard=False):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = torch.sigmoid((torch.log(logits + 1e-10) - torch.log(1 - logits + 1e-10) + gumbel_noise) / temperature)
    if hard:
        y_hard = (y > 0.5).float()
        y = y_hard.detach() - y.detach() + y
    return y


def weighted_bce(preds, targets,pos_weight=5.0):
    eps = 1e-8 
    loss = - (pos_weight * targets *torch.log(preds + eps) + (1 - targets)*torch.log(1 - preds + eps) )
    return loss.mean()

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
        return batch 

def replace_sigmoid_with_tanh(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Sigmoid):
            setattr(module, name, nn.Tanh())
        else:
            replace_sigmoid_with_tanh(child)



def tanimoto_similarity(pred_fp, true_fp, threshold=0.5):
    pred_fp = (pred_fp.detach().cpu().numpy() > threshold).astype(np.uint8)
    true_fp = (true_fp.detach().cpu().numpy() > 0).astype(np.uint8)

    sims = []
    B, d = pred_fp.shape

    for p, t in zip(pred_fp, true_fp):
        # SAFE bitstring construction
        bitstr_p = ''.join('1' if x else '0' for x in p)
        bitstr_t = ''.join('1' if x else '0' for x in t)

        # Debug (optional)
        # print("Expected length:", d, "Actual:", len(bitstr_p))

        assert len(bitstr_p) == d
        assert len(bitstr_t) == d

        p_bits = DataStructs.CreateFromBitString(bitstr_p)
        t_bits = DataStructs.CreateFromBitString(bitstr_t)

        sims.append(DataStructs.FingerprintSimilarity(p_bits, t_bits))

    return np.mean(sims)



def create_finetune_strategy(
    model: nn.Module,
    strategy: str = "full",
    unfreeze_last_n: int = 0,
    verbose: bool = True,
):
    """
    Configure model parameters for different fine-tuning strategies.

    Args:
        model: CondFlowMolBERT model instance.
        strategy: str, one of:
            - "full"              → fine-tune all layers
            - "freeze_embeddings" → freeze token/pos/time embeddings
            - "freeze_encoder"    → train only cond_proj + lm_head
            - "unfreeze_last_n"   → unfreeze last N encoder layers + head
            - "lm_head_only"      → train only lm_head
        unfreeze_last_n: used if strategy='unfreeze_last_n'
        verbose: print summary of trainable parameters

    Returns:
        The modified model (in-place).
    """

    def set_requires_grad(module, flag):
        for p in module.parameters():
            p.requires_grad = flag

    # ---- Reset all ----
    set_requires_grad(model, True)

    # ---- STRATEGIES ----
    if strategy == "full":
        # train everything
        pass

    elif strategy == "freeze_embeddings":
        set_requires_grad(model.tok_emb, False)
        set_requires_grad(model.pos_emb, False)
        set_requires_grad(model.time_emb, False)

    elif strategy == "freeze_encoder":
        set_requires_grad(model.encoder, False)

    elif strategy == "unfreeze_last_n":
        # start frozen
        set_requires_grad(model, False)
        total_layers = len(model.encoder.layers)
        if unfreeze_last_n > total_layers:
            raise ValueError(f"unfreeze_last_n={unfreeze_last_n} > total_layers={total_layers}")
        for layer in model.encoder.layers[-unfreeze_last_n:]:
            set_requires_grad(layer, True)
        set_requires_grad(model.lm_head, True)
        set_requires_grad(model.cond_proj, True)

    elif strategy == "lm_head_only":
        # freeze everything except final head
        set_requires_grad(model, False)
        set_requires_grad(model.lm_head, True)

    else:
        raise ValueError(f"Unknown fine-tuning strategy: {strategy}")

    # ---- Summary ----
    if verbose:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"\n🧠 Fine-tuning strategy: {strategy}")
        print(f"   Trainable params: {trainable:,} / {total:,} ({trainable/total:.2%})")
        print(f"   Frozen params:    {frozen:,}")

    return 'Model weights have been freezed according to strategy'


def zero_cond_to_none(cond: torch.Tensor) -> torch.Tensor | None:
    """
    Converts all-zero condition rows to None for unconditional behavior.

    Args:
        cond: [B, cond_dim] conditioning tensor

    Returns:
        cond with zero rows replaced by None, or None if all rows are zero
    """
    if cond is None:
        return None

    nonzero_mask = (cond.abs().sum(dim=1) > 0)  # True if row is non-zero

    if nonzero_mask.sum() == 0:
        return None
    return cond

def stochastic_drop_condition(cond: torch.Tensor, uncond_prob: float):
    """
    Returns:
      cond_kept: subset of cond with rows kept (conditional ones)
      mask: boolean mask of shape [B], True = keep cond, False = drop cond
    """
    if cond is None:
        return None, None

    B = cond.size(0)
    mask = (torch.rand(B, device=cond.device) > uncond_prob)  # keep some rows
    return cond, mask


def transfer_weights(uncond_model, cond_model, freeze_pretrained=False):
    with torch.no_grad():
        cond_model.model.tok_emb.weight.copy_(uncond_model.tok_emb.weight)
        cond_model.model.pos_emb.weight.copy_(uncond_model.pos_emb.weight)
        cond_model.model.time_emb.weight.copy_(uncond_model.time_emb.weight)
        cond_model.model.encoder.load_state_dict(uncond_model.encoder.state_dict())
        cond_model.model.lm_head.weight.copy_(uncond_model.lm_head.weight)



def transfer_weights_with_adaptive_ln(uncond_model, cond_model, freeze_pretrained=False):
    """
    Transfer pretrained weights and initialize AdaptiveLayerNorm from pretrained LayerNorm.
    """
    # 1️⃣ Embeddings
    cond_model.tok_emb.weight.data.copy_(uncond_model.tok_emb.weight)
    cond_model.pos_emb.weight.data.copy_(uncond_model.pos_emb.weight)

    # Time embedding if shapes match
    if cond_model.time_emb.weight.shape == uncond_model.time_emb.weight.shape:
        cond_model.time_emb.weight.data.copy_(uncond_model.time_emb.weight)
        if hasattr(cond_model.time_emb, 'bias') and hasattr(uncond_model.time_emb, 'bias'):
            cond_model.time_emb.bias.data.copy_(uncond_model.time_emb.bias)

    # 2️⃣ Encoder layers
    for l_pre, l_cond in zip(uncond_model.encoder.layers, cond_model.encoder.layers):
        # Multihead attention
        try:
            l_cond.self_attn.in_proj_weight.data.copy_(l_pre.self_attn.in_proj_weight)
            l_cond.self_attn.in_proj_bias.data.copy_(l_pre.self_attn.in_proj_bias)
            l_cond.self_attn.out_proj.weight.data.copy_(l_pre.self_attn.out_proj.weight)
            l_cond.self_attn.out_proj.bias.data.copy_(l_pre.self_attn.out_proj.bias)
        except Exception as e:
            print("⚠️ Skipping attention weights for a layer:", e)

        # Feedforward layers: can only copy if dimensions match
        if l_cond.linear1.weight.shape == l_pre.linear1.weight.shape:
            l_cond.linear1.weight.data.copy_(l_pre.linear1.weight)
            l_cond.linear1.bias.data.copy_(l_pre.linear1.bias)
        if l_cond.linear2.weight.shape == l_pre.linear2.weight.shape:
            l_cond.linear2.weight.data.copy_(l_pre.linear2.weight)
            l_cond.linear2.bias.data.copy_(l_pre.linear2.bias)

        # Initialize AdaptiveLayerNorm from pretrained LayerNorm
        if hasattr(l_pre, 'norm1') and hasattr(l_cond, 'norm1'):
            l_cond.norm1.ln.weight.data.copy_(l_pre.norm1.weight)
            l_cond.norm1.ln.bias.data.copy_(l_pre.norm1.bias)
        if hasattr(l_pre, 'norm2') and hasattr(l_cond, 'norm2'):
            l_cond.norm2.ln.weight.data.copy_(l_pre.norm2.weight)
            l_cond.norm2.ln.bias.data.copy_(l_pre.norm2.bias)
    try:
        cond_model.lm_head.weight.data.copy_(uncond_model.lm_head.weight)
    except Exception as e:
        print("⚠️ Skipping LM head copy:", e)

    # 4️⃣ Freeze pretrained layers if requested
    if freeze_pretrained:
        for name, param in cond_model.named_parameters():
            # Keep conditional adapters and lm_head trainable
            if ('cond' in name) or ('lm_head' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False


    print("Transfer completed. AdaptiveLayerNorm initialized from pretrained LayerNorm.")


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return None 


