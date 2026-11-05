"""M3 particle-set encoder: transformer over all post-FSI primaries plus a vertex/context token.

Per token: species embedding + MLP over [unit direction (3), log momentum, log KE, signed path length from the true
vertex along the direction to the cathode plane (x=0) and to the active-volume walls]. The geometry inputs encode
"where this particle goes in the detector", which the fixed summary features could not (no azimuth).
Output: a pooled event embedding (masked mean + context token).
"""
from __future__ import annotations

import torch
import torch.nn as nn

AV = (200.0, 200.0, 0.0, 500.0)   # |x|, |y| half widths, zmin, zmax [cm]


def path_lengths(vtx, d):
    """vtx [B,3], d [B,N,3] unit directions -> [B,N,3]: distance to the cathode plane (inf if moving away),
    to the nearest x/y wall, to the nearest z wall; all clipped to 1000 cm and log-scaled outside."""
    big = torch.full_like(d[..., 0], 1000.0)
    v = vtx[:, None, :].expand_as(d)
    eps = 1e-6
    # cathode at x = 0: crossed if moving toward it
    t_cath = torch.where(v[..., 0] * d[..., 0] < 0, (-v[..., 0] / (d[..., 0] + eps * torch.sign(d[..., 0] + eps))).abs(), big)
    tx = torch.where(d[..., 0].abs() > eps, ((torch.sign(d[..., 0]) * AV[0] - v[..., 0]) / (d[..., 0] + eps * torch.sign(d[..., 0]))).abs(), big)
    ty = torch.where(d[..., 1].abs() > eps, ((torch.sign(d[..., 1]) * AV[1] - v[..., 1]) / (d[..., 1] + eps * torch.sign(d[..., 1]))).abs(), big)
    zw = torch.where(d[..., 2] > 0, torch.full_like(d[..., 2], AV[3]), torch.full_like(d[..., 2], AV[2]))
    tz = torch.where(d[..., 2].abs() > eps, ((zw - v[..., 2]) / (d[..., 2] + eps * torch.sign(d[..., 2]))).abs(), big)
    out = torch.stack([t_cath, torch.minimum(tx, ty), tz], -1).clamp(0, 1000.0)
    return torch.log1p(out)


class SetEncoder(nn.Module):
    def __init__(self, n_classes=12, ctx_dim=80, d_model=128, n_layers=4, n_heads=4, d_out=64):
        super().__init__()
        self.emb = nn.Embedding(n_classes + 1, d_model)              # last index = padding (unused)
        self.tok = nn.Sequential(nn.Linear(3 + 2 + 3, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.ctx = nn.Sequential(nn.Linear(ctx_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.layers = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, 0.0, activation="gelu", batch_first=True, norm_first=True), n_layers)
        self.out = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_out))

    def forward(self, cls, X, vtx, ctx):
        """cls [B,N] long (-1 pad), X [B,N,4] (px,py,pz,ke GeV), vtx [B,3] cm, ctx [B,ctx_dim] standardised features."""
        valid = cls >= 0
        p = X[..., :3]; pm = p.norm(dim=-1, keepdim=True).clamp_min(1e-6); d = p / pm
        feats = torch.cat([d, torch.log(pm + 1e-3), torch.log(X[..., 3:4].clamp_min(0) + 1e-3), path_lengths(vtx, d)], -1)
        h = self.emb(cls.clamp_min(0)) + self.tok(feats)
        c = self.ctx(ctx)[:, None, :]
        s = torch.cat([c, h], 1); mask = torch.cat([torch.ones_like(valid[:, :1]), valid], 1)
        s = self.layers(s, src_key_padding_mask=~mask)
        pooled = (s[:, 1:] * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return self.out(torch.cat([s[:, 0], pooled], -1))
