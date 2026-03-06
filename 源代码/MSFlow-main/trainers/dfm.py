import torch
from torch.nn import CrossEntropyLoss
from flow_matching.loss import MixturePathGeneralizedKL
from configs import *



def dfm_step(batch, model, source, loss_fn,scheduler, path, device, mask_token_id,weighted=False):
    B = batch.size(0)
    x1 = batch.to(device)
    if source == 'masked':
        x0 = torch.full_like(x1, fill_value=mask_token_id)  # fully masked input
    else:
        x0 = torch.randint_like(x1, vocab)  # uniform source
    t = torch.rand(B, device=device) * (1 - 1e-3)  # avoid t=1
    path_sample = path.sample(t=t, x_0=x0, x_1=x1)

    logits = model(x=path_sample.x_t, t=path_sample.t)

    if isinstance(loss_fn, CrossEntropyLoss):
        if weighted == False:
            loss = loss_fn(logits.view(-1, logits.size(-1)), x1.view(-1)).mean()   # Reshape for CrossEntropy: [B * T, vocab]
        else:
            loss_per_sample = loss_fn(logits.view(-1, logits.size(-1)), x1.view(-1))  #[B*T]
            alpha_t, sigma_t = scheduler(t).alpha_t,scheduler(t).sigma_t
            weights = alpha_t / sigma_t
            weights = torch.clamp(weights, min=0.05, max=1.5)
            weights_expanded = weights.unsqueeze(1).repeat(1, x1.size(-1)).view(-1)  #[B]-----> [B,1] ---> [B,T] ----> [B*T]
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
        raise ValueError("Invalid loss function type: {}".format(type(loss_fn)))

    return loss



























# def mfm_flow_loss(x_clean, x_corrupt, logits, pad_token_id):
#     mask_ratio = (x_corrupt == pad_token_id).float().mean(dim=1, keepdim=True)
#     weight = 1.0 / torch.clamp(1 - mask_ratio, min=1e-3)
#     loss_fn = CrossEntropyLoss(ignore_index=pad_token_id)
#     loss = loss_fn(logits.view(-1, logits.size(-1)), x_clean.view(-1))
#     return weight.mean() * loss

# def mfm_train_step(batch, model, optimizer, device, mask_token_id, pad_token_id):
#     m = random.uniform(0.1, 0.6)
#     x0 = batch.to(device)
#     noise = torch.bernoulli(torch.full_like(x0, m, dtype=torch.float)).bool()
#     xc = torch.where(noise, torch.full_like(x0, mask_token_id), x0)
#     logits = model(xc)
#     loss = mfm_flow_loss(x0, xc, logits, pad_token_id)
#     # optimizer.zero_grad(); loss.backward(); optimizer.step()
#     return loss #loss.item()