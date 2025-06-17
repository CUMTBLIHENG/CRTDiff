# model.py

import torch
import torch.nn as nn
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device) * (-math.log(10000.0) / (half_dim - 1)))
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

class ResDenoiseNet(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(hidden)
        self.fc1 = nn.Linear(input_dim + cond_dim, hidden)
        self.res_block = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden)
        )
        self.out = nn.Linear(hidden, input_dim)
    def forward(self, x_cat, t):
        t_emb = self.time_embed(t)
        h = self.fc1(x_cat) + t_emb
        h = h + self.res_block(h)
        return self.out(h)
