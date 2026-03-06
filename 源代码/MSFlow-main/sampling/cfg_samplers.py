import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.utils import ModelWrapper
from flow_matching.solver import MixtureDiscreteEulerSolver

#newly added that requires one forward pass only but twice the memory
class WrappedModelCond(ModelWrapper):
    def __init__(self, model, temperature: float = 1.0, guidance_scale: float = 1.0):
        super().__init__(model)
        self.temperature = temperature
        self.guidance_scale = guidance_scale

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        cond = extras.get("cond", None)

        if self.guidance_scale == 0.0 or cond is None:
            logits = self.model(x, t, cond=cond)
            return torch.softmax(logits / self.temperature, dim=-1)

        # Create batch for single forward pass: first half zeros (uncond), second half original cond
        B = x.size(0)
        zero_cond = torch.zeros_like(cond)
        x_cat = torch.cat([x, x], dim=0)
        t_cat = torch.cat([t, t], dim=0)
        cond_cat = torch.cat([zero_cond, cond], dim=0)
        # Single forward pass
        logits_cat = self.model(x_cat, t_cat, cond=cond_cat)

        # Split logits
        logits_uncond, logits_cond = logits_cat[:B], logits_cat[B:]

        # Classifier-free guidance
        guided_logits = logits_uncond + self.guidance_scale * (logits_cond - logits_uncond)
        return torch.softmax(guided_logits / self.temperature, dim=-1)



@torch.no_grad()
def sample_flow_cond(
    num_samples: int,
    model,
    cond: torch.Tensor,
    path,
    seq_len: int,
    vocab_size: int,
    mask_token_id: int = 0,
    source_distribution: str = "uniform",
    steps: int = 128,
    epsilon: float = 1e-3,
    device: str = "cpu",
    return_intermediates: bool = False,
    temperature: float = 1.0,
    guidance_scale: float = 0.0,
):
    step_size = 1.0 / steps
    time_grid = torch.linspace(0, 1 - epsilon, steps, device=device)

    # Initialize tokens
    if source_distribution == "uniform":
        x_init = torch.randint(high=vocab_size, size=(num_samples, seq_len), device=device)
    elif source_distribution == "masked":
        x_init = torch.full((num_samples, seq_len), mask_token_id, device=device)
    else:
        raise ValueError(f"Unknown source_distribution: {source_distribution}")

    # Prepare conditioning
    cond = cond.to(device)
    if cond.size(0) != num_samples:
        if cond.dim() == 1:
            cond = cond.unsqueeze(0).repeat(num_samples, 1)
        else:
            raise ValueError("Condition batch size must match num_samples or be broadcastable.")

    # Wrap model with guidance
    wrapped_model = WrappedModelCond(model, temperature=temperature, guidance_scale=guidance_scale)

    # Create solver
    solver = MixtureDiscreteEulerSolver(
        model=wrapped_model,
        path=path,
        vocabulary_size=vocab_size,
    )

    # Generate samples
    samples = solver.sample(
        x_init=x_init,
        step_size=step_size,
        time_grid=time_grid,
        return_intermediates=return_intermediates,
        verbose=True,
        cond=cond,
    )

    return samples.detach().cpu()