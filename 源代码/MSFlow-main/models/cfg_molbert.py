import torch
import torch.nn as nn
import torch.nn.functional as F
from .adaptive import  ConditionalTransformerEncoderLayer, ConditionalTransformerEncoder


class CondFlowMolBERT(nn.Module):
    def __init__(self, vocab=173, cond_dim=1449, time_dim=1, d_model=127, n_layers=4, n_heads=4, mlp_dim=256, max_len=72, dropout=0.4):
        super().__init__()
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.time_emb = nn.Linear(1, time_dim)

        # Condition embedding MLP
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, 4096), # for all exps : bits, cdddds 4096, for ms 128
            nn.ReLU(),
            nn.LayerNorm(4096),
            nn.Linear(4096, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        total_dim = d_model + time_dim
        layer = ConditionalTransformerEncoderLayer(
            d_model=total_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            batch_first=True,
            cond_dim=total_dim,  # condition added to d_model+time_dim
        )
        self.encoder = ConditionalTransformerEncoder(layer, n_layers)
        

        self.lm_head = nn.Linear(total_dim, vocab, bias=False)

    def forward(
    self,
    x: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor = None,
) -> torch.Tensor:
        """
        Forward pass with optional classifier-free guidance handling.
        """
        B, L = x.shape

        # Token + position embeddings
        tok_embed = self.tok_emb(x)
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        pos_embed = self.pos_emb(pos_ids)
        x_embed = tok_embed + pos_embed

        # Time embeddings
        t_embed = self.time_emb(t.unsqueeze(-1))
        t_embed = t_embed.unsqueeze(1).expand(-1, L, -1)

        # Combine embeddings
        h = torch.cat([x_embed, t_embed], dim=-1)

        # Detect unconditional rows (all-zero cond)
        if cond is not None:
            uncond_mask = (cond.abs().sum(dim=-1) == 0)  # [B]
            cond_embed = self.cond_proj(cond)

            # Zero-out embeddings for unconditional rows
            cond_embed[uncond_mask] = 0
        else:
            cond_embed = None

        # Pass through transformer
        h = self.encoder(h, condition=cond_embed if cond_embed is not None else None)
        return self.lm_head(h)

