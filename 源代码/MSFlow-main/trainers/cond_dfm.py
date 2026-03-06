import torch
from torch.nn import CrossEntropyLoss
from flow_matching.loss import MixturePathGeneralizedKL
from configs import *
# from utils.functions import stochastic_drop_condition, zero_cond_to_none

def dfm_step(batch, cond, model, source, loss_fn, scheduler, path, device, mask_token_id, weighted=False, uncond_prob=0.1, training = False):
    """
    batch: input token ids, shape [B, L]
    cond: condition tensor, shape [B, cond_dim]
    model: FlowMolBERT_Cond instance
    source: str, 'masked' or other sampling mode
    loss_fn: loss function (CrossEntropyLoss or MixturePathGeneralizedKL)
    scheduler, path, device: as usual
    mask_token_id: token used for masking
    weighted: whether to use weighted loss
    force_uncond_prob: probability to drop condition for unconditional training
    """

    B = batch.size(0)
    x1 = batch.to(device)
    cond = cond.to(device) if cond is not None else None

    # Source distribution
    if source == 'masked':
        x0 = torch.full_like(x1, fill_value=mask_token_id)
    else:
        x0 = torch.randint_like(x1, model.lm_head.out_features)

    # Sample path
    t = torch.rand(B, device=device) * (1 - 1e-3)
    path_sample = path.sample(t=t, x_0=x0, x_1=x1)

    # Handle dropping condition for classifier-free guidance
    if training and cond is not None:
        drop_mask = torch.rand(B, device=device) < uncond_prob
        cond = cond.clone()
        cond[drop_mask] = 0.0  # zero out dropped rows

    # Forward pass: all-zero cond rows are treated as unconditional internally
    logits = model(x=path_sample.x_t, t=path_sample.t, cond=cond)
    if isinstance(loss_fn, CrossEntropyLoss):
        if not weighted:
            loss = loss_fn(logits.view(-1, logits.size(-1)), x1.view(-1)).mean()
        else:
            loss_per_sample = loss_fn(logits.view(-1, logits.size(-1)), x1.view(-1))
            alpha_t, sigma_t = scheduler(t).alpha_t, scheduler(t).sigma_t
            weights = alpha_t / sigma_t
            weights = torch.clamp(weights, min=0.05, max=1.5)
            weights_expanded = weights.unsqueeze(1).repeat(1, x1.size(-1)).view(-1)
            weighted_loss = loss_per_sample * weights_expanded
            loss = weighted_loss.mean()

    elif isinstance(loss_fn, MixturePathGeneralizedKL):
        loss = loss_fn(
            logits=logits,
            x_1=x1,
            x_t=path_sample.x_t,
            t=path_sample.t
        ).mean()
    else:
        raise ValueError(f"Invalid loss function type: {type(loss_fn)}")

    return loss
