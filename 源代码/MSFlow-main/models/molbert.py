import torch.nn as nn
import torch
from configs.data import MAX_LEN

class FlowMolBERT(nn.Module):
    def __init__(self, vocab, time_dim=1, d_model=127, n_layers=4, n_heads=4, mlp=256, max_len=72):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)  # Learned positional embeddings

        self.time_emb = nn.Linear(1, time_dim)

        layer = nn.TransformerEncoderLayer(d_model + time_dim, n_heads, mlp, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.lm_head = nn.Linear(d_model + time_dim, vocab, bias=False)

    def forward(self, x, t):
        B, L = x.shape

        tok_embed = self.tok_emb(x)  # [B, L, d_model]
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)  # [B, L]
        pos_embed = self.pos_emb(pos_ids)  # [B, L, d_model]

        x_embed = tok_embed + pos_embed  # Add positional embedding

        t_embed = self.time_emb(t.unsqueeze(-1))  # [B, time_dim]
        t_embed = t_embed.unsqueeze(1).expand(-1, L, -1)  # [B, L, time_dim]

        h = torch.cat([x_embed, t_embed], dim=-1)  # [B, L, d_model + time_dim]
        h = self.encoder(h)
        return self.lm_head(h)