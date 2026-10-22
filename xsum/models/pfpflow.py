"""Masked set flow matching for the PFP set: permutation-equivariant velocity net (transformer encoder over PFP
tokens), conditioned on a per-slice vector concatenated to every token. Adapted from opendata-sim2reco prongflow
(no particle-token cross-attention yet: that comes with the set encoder, M3)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .flow import TimeEmbedding


class PfpVelocity(nn.Module):
    def __init__(self, dim, cond_dim, d_model=192, n_layers=4, n_heads=4, t_dim=64):
        super().__init__()
        self.temb = TimeEmbedding(t_dim)
        self.inp = nn.Linear(dim + t_dim + cond_dim, d_model)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, 0.0, activation="gelu", batch_first=True, norm_first=True) for _ in range(n_layers)])
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, dim))

    def forward(self, x, t, cond, pmask):
        B, N, _ = x.shape
        c = torch.cat([self.temb(t), cond], -1)[:, None].expand(B, N, -1)
        s = self.inp(torch.cat([x, c], -1))
        pad = ~pmask
        for layer in self.layers:
            s = layer(s, src_key_padding_mask=pad)
        return self.out(s)


class PfpFlow(nn.Module):
    def __init__(self, dim, cond_dim, d_model=192, n_layers=4, sigma_min=1e-3):
        super().__init__()
        self.dim, self.sigma_min = dim, sigma_min
        self.v = PfpVelocity(dim, cond_dim, d_model, n_layers)

    def loss(self, x1, cond, pmask):
        x0 = torch.randn_like(x1)
        t = torch.rand(len(x1), device=x1.device); tt = t[:, None, None]
        xt = (1 - (1 - self.sigma_min) * tt) * x0 + tt * x1
        target = x1 - (1 - self.sigma_min) * x0
        err = ((self.v(xt, t, cond, pmask) - target) ** 2).mean(-1)
        return (err * pmask).sum() / pmask.sum().clamp_min(1)

    @torch.no_grad()
    def sample(self, cond, pmask, n_steps=64, bound=20.0):
        B, N = pmask.shape
        x = torch.randn(B, N, self.dim, device=cond.device); dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=cond.device)
            k1 = self.v(x, t, cond, pmask); k2 = self.v(x + 0.5 * dt * k1, t + 0.5 * dt, cond, pmask)
            x = (x + dt * k2).clamp(-bound, bound)
        return torch.nan_to_num(x, nan=0.0, posinf=bound, neginf=-bound)
