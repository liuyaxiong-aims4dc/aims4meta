import torch.nn as nn
import torch.nn.functional as F
import copy

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, normalized_shape, cond_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.ln = nn.LayerNorm(normalized_shape, elementwise_affine=True, eps=eps)
        # condition -> gamma, beta
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, 2 * normalized_shape),
            nn.ReLU(),
            nn.Linear(2 * normalized_shape, 2 * normalized_shape)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, cond):
        # x: [B, L, D], cond: [B, C]
        normalized = self.ln(x)

        #newly added
        uncond_mask = (cond.abs().sum(dim=-1) == 0).float().unsqueeze(-1)

        gamma_beta = self.mlp(cond)  # [B, 2*D]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, D], [B, D]
        
        #newly added
        gamma = gamma * (1 - uncond_mask)
        beta = beta * (1 - uncond_mask)
        
        # expand over sequence length
        gamma = gamma.unsqueeze(1)  # [B, 1, D]
        beta = beta.unsqueeze(1)    # [B, 1, D]

        return normalized * (1 + gamma) + beta



class ConditionalTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, cond_dim=1024, dim_feedforward=2048, dropout=0.1, batch_first=True):
        super().__init__(d_model, nhead, dim_feedforward, dropout, batch_first=batch_first)
        cond_dim = d_model - 1   #dmodel = dmodel + 1 for time embed 

        self.norm1 = AdaptiveLayerNorm(d_model, cond_dim)
        self.norm2 = AdaptiveLayerNorm(d_model, cond_dim)

    def forward(self, src, condition, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # src: [B, L, D], cond: [B, C]
        x = src
        x2 = self.self_attn(x, x, x, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask,
                            need_weights=False, is_causal=is_causal)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x, condition)

        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x, condition)

        return x
    
    
class ConditionalTransformerEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__(encoder_layer, num_layers, norm)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=False, condition=None):
        output = src
        for mod in self.layers:
            output = mod(output, condition=condition,src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask,
                         is_causal=is_causal)
        if self.norm is not None:
            output = self.norm(output, condition) if isinstance(self.norm, AdaptiveLayerNorm) else self.norm(output)
        return output
