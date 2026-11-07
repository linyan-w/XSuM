"""Masked set flow matching for the PFP set: permutation-equivariant velocity net (transformer over PFP tokens),
conditioned on a per-slice vector concatenated to every token. Adapted from opendata-sim2reco prongflow.

With n_part_layers > 0 every PFP layer also cross-attends to the true final-state particle tokens (class embedding +
asinh(p / 0.1 GeV) of (px, py, pz), encoded by a small transformer once per event), as the sim2reco prong flow does:
no PFP-to-particle matching is needed, each PFP learns which particle(s) to read from. A learned null token is always
present so that events without particles are well defined.
With geo=True each particle token also carries its Geant4 truth (xsum.prep.primset.GEO_FIELDS): start and end point
/ 250 cm, asinh(length / 10 cm), contained and crosses_tpc flags, and a has-position flag (positions are missing for
particles of neutrinos outside the active volume; their inputs are then 0).
With geo=2 the token also carries the point where the straight line from the Geant4 start to the Geant4 end leaves
the TPC active volume (/ 250 cm) and an exits flag: for an exiting particle this is where reco tracks end, which
the end point alone (often metres outside) does not give.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .flow import TimeEmbedding
from .setenc import P_SCALE


GEO_DIM = {1: 10, 2: 14}   # start (3), end (3), length, contained, crosses_tpc, has_position [, exit point (3), exits]
AV_LO = (-201.3, -203.7, 0.0)     # TPC active volume used for the exit point, cm (edges of the real reco track-end pile-ups)
AV_HI = (201.3, 203.7, 500.8)


def tpc_exit(s, e):
    """Straight line s -> e (cm, [..., 3]): (exit point [..., 3], exits [...] bool); exits = s inside and e outside the AV."""
    lo = torch.tensor(AV_LO, device=s.device); hi = torch.tensor(AV_HI, device=s.device)
    s_in = ((s > lo) & (s < hi)).all(-1); e_in = ((e > lo) & (e < hi)).all(-1); ex = s_in & ~e_in
    d = e - s; big = torch.full_like(d, 1e9)
    t = torch.where(d > 0, (hi - s) / d.clamp_min(1e-9), torch.where(d < 0, (lo - s) / d.clamp_max(-1e-9), big)).amin(-1).clamp(0, 1)
    p = s + t[..., None] * d
    return torch.where(ex[..., None], p, torch.zeros_like(p)), ex


def geo_features(G, version=1):
    """G [..., 9] int16 (mm, flags; -32768 = missing), numpy or torch -> [..., GEO_DIM[version]] float32 torch tensor."""
    G = torch.as_tensor(G); miss = G == -32768; g = G.float()
    pos = torch.where(miss[..., :6], torch.zeros_like(g[..., :6]), g[..., :6] / 2500.0)              # mm -> units of 250 cm
    ln = torch.where(miss[..., 6], torch.zeros_like(g[..., 6]), torch.asinh(g[..., 6] / 100.0))      # asinh(length / 10 cm)
    fl = torch.where(miss[..., 7:9], torch.zeros_like(g[..., 7:9]), g[..., 7:9])
    has = (~miss[..., 3]).float()
    f = [pos, ln[..., None], fl, has[..., None]]
    if version >= 2:
        p, ex = tpc_exit(g[..., :3] / 10.0, g[..., 3:6] / 10.0); ex = ex & (has > 0)
        f += [torch.where(ex[..., None], p / 250.0, torch.zeros_like(p)), ex.float()[..., None]]
    return torch.cat(f, -1)


class ParticleEncoder(nn.Module):
    def __init__(self, d_model, n_layers=2, n_heads=4, n_classes=12, geo=False):
        super().__init__()
        self.emb = nn.Embedding(n_classes, d_model)
        self.geo = int(geo)
        self.tok = nn.Sequential(nn.Linear(3 + (GEO_DIM[self.geo] if self.geo else 0), d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.null = nn.Parameter(torch.zeros(1, 1, d_model))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, 0.0, activation="gelu", batch_first=True, norm_first=True), n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cls, P, G=None):
        """cls [B,M] long (-1 pad), P [B,M,3] momentum GeV, G [B,M,9] int16 Geant4 geometry (geo models)
        -> memory [B,M+1,d], key padding mask [B,M+1] (True = pad)."""
        valid = cls >= 0
        f = torch.asinh(P[..., :3] / P_SCALE)
        if G is not None: f = torch.cat([f, geo_features(G, self.geo)], -1)
        h = self.emb(cls.clamp_min(0)) + self.tok(f)
        h = torch.cat([self.null.expand(len(h), -1, -1), h], 1)
        pad = torch.cat([torch.zeros_like(valid[:, :1]), ~valid], 1)
        return self.norm(self.enc(h, src_key_padding_mask=pad)), pad


class CrossBlock(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model); self.att = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, s, mem, mpad):
        q = self.norm(s)
        return s + self.att(q, mem, mem, key_padding_mask=mpad, need_weights=False)[0]


class PfpVelocity(nn.Module):
    def __init__(self, dim, cond_dim, d_model=192, n_layers=4, n_heads=4, t_dim=64, n_part_layers=0, geo=False):
        super().__init__()
        self.temb = TimeEmbedding(t_dim)
        self.inp = nn.Linear(dim + t_dim + cond_dim, d_model)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, 0.0, activation="gelu", batch_first=True, norm_first=True) for _ in range(n_layers)])
        self.part = ParticleEncoder(d_model, n_part_layers, n_heads, geo=geo) if n_part_layers > 0 else None
        self.cross = nn.ModuleList([CrossBlock(d_model, n_heads) for _ in range(n_layers)]) if n_part_layers > 0 else None
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, dim))

    def forward(self, x, t, cond, pmask, mem=None):
        B, N, _ = x.shape
        c = torch.cat([self.temb(t), cond], -1)[:, None].expand(B, N, -1)
        s = self.inp(torch.cat([x, c], -1))
        pad = ~pmask
        for i, layer in enumerate(self.layers):
            s = layer(s, src_key_padding_mask=pad)
            if self.cross is not None: s = self.cross[i](s, *mem)
        return self.out(s)


class PfpFlow(nn.Module):
    def __init__(self, dim, cond_dim, d_model=192, n_layers=4, sigma_min=1e-3, n_part_layers=0, geo=False):
        super().__init__()
        self.dim, self.sigma_min = dim, sigma_min
        self.v = PfpVelocity(dim, cond_dim, d_model, n_layers, n_part_layers=n_part_layers, geo=geo)

    def memory(self, part):
        """part = (cls [B,M] long, P [B,M,3][, G [B,M,9] int16]) or None."""
        return None if (part is None or self.v.part is None) else self.v.part(*part)

    def loss(self, x1, cond, pmask, part=None):
        x0 = torch.randn_like(x1)
        t = torch.rand(len(x1), device=x1.device); tt = t[:, None, None]
        xt = (1 - (1 - self.sigma_min) * tt) * x0 + tt * x1
        target = x1 - (1 - self.sigma_min) * x0
        err = ((self.v(xt, t, cond, pmask, self.memory(part)) - target) ** 2).mean(-1)
        return (err * pmask).sum() / pmask.sum().clamp_min(1)

    @torch.no_grad()
    def sample(self, cond, pmask, n_steps=64, bound=20.0, part=None):
        B, N = pmask.shape
        mem = self.memory(part)   # particle encoding once per event, reused at every step
        x = torch.randn(B, N, self.dim, device=cond.device); dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=cond.device)
            k1 = self.v(x, t, cond, pmask, mem); k2 = self.v(x + 0.5 * dt * k1, t + 0.5 * dt, cond, pmask, mem)
            x = (x + dt * k2).clamp(-bound, bound)
        return torch.nan_to_num(x, nan=0.0, posinf=bound, neginf=-bound)
