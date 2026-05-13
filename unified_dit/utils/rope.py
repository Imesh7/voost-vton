from torch import nn
import torch


class RoPE(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        
    def get_pos_emb(self, x):
        seq_len = x.size(1)
        device = x.device
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        ).to(device)
        pos_seq = torch.arange(seq_len, device=device).float()
        sinusoid_inp = torch.einsum("i,j->ij", pos_seq, inv_freq)
        sin_emb = torch.sin(sinusoid_inp)
        cos_emb = torch.cos(sinusoid_inp)
        return sin_emb, cos_emb
    
    def apply_rotary_pos_emb(self, x, pos_emb):
        # x: [batch_size, seq_len, dim]
        # cos, sin: [seq_len, dim//2]
        cos, sin = pos_emb
        x1 = x[..., ::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices
        x_rotated = torch.cat(
            [x1 * cos + x2 * sin, x2 * cos - x1 * sin], dim=-1
        )  # Rotate pairs
        return x_rotated