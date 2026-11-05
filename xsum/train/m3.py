"""M3 stage A: train the particle-set encoder on the matched slices to predict the slice record (topology class,
group presence, containment flags, masked regression of the standardised slice block), then export its 64-d
embedding for every neutrino as extra feature columns (z_0..z_63) so that M2/M4 can be retrained on
[80 summary features + embedding] unchanged.
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as Fn

from ..data.splits import split_by_level
from ..models.setenc import SetEncoder
from .baselines import Standardizer
from .m2 import FLAGS, GROUPS, SPEC, TOPO_COLS, BlockTransform, Topology, mask_from_presence, presence

IDX = ["chunk", "__ntuple", "entry", "rec.mc.nu..index"]
D_OUT = 64


def pack(chunk, ntuple, entry, nu):
    return ((np.asarray(chunk, np.int64) * 100000 + np.asarray(ntuple, np.int64)) * 10000000 + np.asarray(entry, np.int64)) * 16 + np.asarray(nu, np.int64)


class PrimStore:
    """Primary-token arrays addressable by packed neutrino key. Reads the memory-mapped layout written by
    `to_mmap` (keys.npy sorted, cls.npy, X.npy) so that only the requested rows are paged in."""
    def __init__(self, d):
        self.key = np.load(f"{d}/mmap/keys.npy"); self.cls = np.load(f"{d}/mmap/cls.npy", mmap_mode="r"); self.X = np.load(f"{d}/mmap/X.npy", mmap_mode="r")

    def take(self, key):
        pos = np.minimum(np.searchsorted(self.key, key), len(self.key) - 1)
        ok = self.key[pos] == key
        if not ok.all(): raise KeyError(f"{(~ok).sum()} neutrinos missing from the particle sets")
        o = np.argsort(pos, kind="stable"); inv = np.empty_like(o); inv[o] = np.arange(len(o))
        return np.asarray(self.cls[pos[o]])[inv], np.asarray(self.X[pos[o]])[inv]   # sorted reads, original order


def to_mmap(d):
    """Convert group_NN.npz into one key-sorted memory-mapped layout under <d>/mmap (one group in memory at a time)."""
    files = sorted(glob.glob(f"{d}/group_??.npz")); os.makedirs(f"{d}/mmap", exist_ok=True)
    keys = [pack(*np.load(f)["key"].T) for f in files]; sizes = [len(k) for k in keys]; allk = np.concatenate(keys)
    order = np.argsort(allk, kind="stable"); np.save(f"{d}/mmap/keys.npy", allk[order])
    dest = np.empty(len(allk), np.int64); dest[order] = np.arange(len(allk))       # global row -> sorted position
    z0 = np.load(files[0]); N, F = z0["X"].shape[1:]
    cls = np.lib.format.open_memmap(f"{d}/mmap/cls.npy", mode="w+", dtype=np.int8, shape=(len(allk), N))
    X = np.lib.format.open_memmap(f"{d}/mmap/X.npy", mode="w+", dtype=np.float32, shape=(len(allk), N, F))
    off = 0
    for f, n in zip(files, sizes):
        z = np.load(f); tgt = dest[off:off + n]; o = np.argsort(tgt)
        cls[tgt[o]] = z["cls"][o]; X[tgt[o]] = z["X"][o]; off += n; del z
    cls.flush(); X.flush()
    return len(allk)


def read_features(path, columns=None):
    pf = pq.ParquetFile(path); names = [c for c in pf.schema_arrow.names if c not in IDX]
    t = pf.read(columns=IDX + (columns or names)); idx = {c: t.column(c).to_numpy() for c in IDX}
    F = np.stack([t.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in (columns or names)], 1)
    return idx, F, (columns or names)


def run(table_dir, primsets_dir, out_dir, device="cuda", epochs=6, max_rows=3_000_000, seed=0, bs=2048, lr=7e-4):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True); t0 = time.time(); rng = np.random.default_rng(seed)
    idx, F, fnames = read_features(f"{table_dir}/features.parquet")
    T = pd.read_parquet(f"{table_dir}/targets.parquet", columns=[c for c, _, _ in SPEC] + TOPO_COLS + FLAGS)
    sel = np.arange(len(F))
    if max_rows and len(sel) > max_rows: sel = np.sort(rng.choice(sel, max_rows, replace=False))
    F, T = F[sel], T.iloc[sel]; idx = {k: v[sel] for k, v in idx.items()}
    key = pack(idx["chunk"], idx["__ntuple"], idx["entry"], idx["rec.mc.nu..index"])
    store = PrimStore(primsets_dir); cls, X = store.take(key); del store
    file_id = idx["chunk"].astype(np.int64) * 100000 + idx["__ntuple"].astype(np.int64)
    sp = split_by_level(pd.MultiIndex.from_arrays([file_id], names=["file"]), level="file", seed=seed); tr, va, te = sp["train"], sp["val"], sp["test"]
    std = Standardizer(F[tr]); C = std(F); vtx = F[:, [fnames.index("vtx_x"), fnames.index("vtx_y"), fnames.index("vtx_z")]]
    topo = Topology(T, tr); k = topo.encode(T)
    bt = BlockTransform(); Y = bt.matrix(T).astype(np.float32); bt.fit(Y[tr]); Z, Mfin = bt.standardise(Y); del Y
    P = presence(T, bt); Mg = mask_from_presence(P, bt) * Mfin
    Pm = np.stack([P[g] for g in GROUPS], 1).astype(np.float32)
    Fl = np.stack([np.nan_to_num(T[f].to_numpy(), nan=0.0) for f in FLAGS], 1).astype(np.float32); Flm = np.repeat(P["two"][:, None], len(FLAGS), 1).astype(np.float32)
    print(f"rows {len(F)} (train {tr.sum()}), tokens/nu {(cls >= 0).sum(1).mean():.2f}, topology classes {topo.other + 1}  ({time.time()-t0:.0f} s)", flush=True)

    enc = SetEncoder(ctx_dim=C.shape[1], d_out=D_OUT).to(device)
    heads = torch.nn.ModuleDict({"topo": torch.nn.Linear(D_OUT, topo.other + 1), "pres": torch.nn.Linear(D_OUT, len(GROUPS)),
                                 "flag": torch.nn.Linear(D_OUT, len(FLAGS)), "block": torch.nn.Sequential(torch.nn.Linear(D_OUT, 256), torch.nn.SiLU(), torch.nn.Linear(256, Z.shape[1]))}).to(device)
    params = list(enc.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5); sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * (int(tr.sum()) // bs + 1))
    T_ = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)

    def batch_loss(i):
        z = enc(T_(cls[i], torch.long), T_(X[i]), T_(vtx[i]), T_(C[i]))
        l_topo = Fn.cross_entropy(heads["topo"](z), T_(k[i], torch.long))
        l_pres = Fn.binary_cross_entropy_with_logits(heads["pres"](z), T_(Pm[i]))
        fm = T_(Flm[i]); l_flag = (Fn.binary_cross_entropy_with_logits(heads["flag"](z), T_(Fl[i]), reduction="none") * fm).sum() / fm.sum().clamp_min(1)
        mm = T_(Mg[i]); l_blk = (((heads["block"](z) - T_(Z[i])) ** 2) * mm).sum() / mm.sum().clamp_min(1)
        return l_topo + l_pres + l_flag + l_blk, (l_topo.item(), l_pres.item(), l_flag.item(), l_blk.item())

    itr, iva = np.flatnonzero(tr), np.flatnonzero(va); hist = []
    for ep in range(epochs):
        enc.train(); heads.train(); perm = rng.permutation(itr); acc = np.zeros(4); nb = 0
        for b in range(0, len(perm), bs):
            loss, parts = batch_loss(perm[b:b + bs]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); acc += parts; nb += 1
        enc.eval(); heads.eval()
        with torch.no_grad():
            v = np.mean([batch_loss(iva[b:b + 8192])[1] for b in range(0, len(iva), 8192)], 0)
        hist.append({"epoch": ep, "train": (acc / nb).tolist(), "val": v.tolist()})
        print(f"epoch {ep} train topo/pres/flag/block {np.round(acc / nb, 4).tolist()}  val {np.round(v, 4).tolist()}  ({time.time()-t0:.0f} s)", flush=True)
    torch.save({"enc": enc.state_dict(), "std_m": std.m, "std_s": std.s, "features": fnames, "d_out": D_OUT}, out / "encoder.pt")
    json.dump({"rows": int(len(F)), "history": hist, "runtime_s": time.time() - t0}, open(out / "metrics.json", "w"), indent=1)
    print(f"stage A done ({time.time()-t0:.0f} s)", flush=True)


@torch.no_grad()
def export(encoder_dir, primsets_dir, table_dir, out_table_dir, device="cuda", bs=16384):
    """Write <out_table_dir>/features.parquet = original features + z_0..z_63 (row order and row groups preserved);
    labels/targets are symlinked. Streams the features row group by row group."""
    ck = torch.load(f"{encoder_dir}/encoder.pt", map_location=device, weights_only=False)
    enc = SetEncoder(ctx_dim=len(ck["features"]), d_out=ck["d_out"]).to(device); enc.load_state_dict(ck["enc"]); enc.eval()
    store = PrimStore(primsets_dir); os.makedirs(out_table_dir, exist_ok=True)
    pf = pq.ParquetFile(f"{table_dir}/features.parquet"); writer = None; n = 0; t0 = time.time()
    fi = [ck["features"].index(c) for c in ("vtx_x", "vtx_y", "vtx_z")]
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg)
        F = np.stack([t.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in ck["features"]], 1)
        key = pack(*(t.column(c).to_numpy() for c in IDX)); cls, X = store.take(key)
        C = ((np.arcsinh(F) - ck["std_m"]) / ck["std_s"]).astype(np.float32)
        Zs = np.concatenate([enc(torch.tensor(cls[i:i + bs], dtype=torch.long, device=device), torch.tensor(X[i:i + bs], device=device),
                                 torch.tensor(F[i:i + bs][:, fi], device=device), torch.tensor(C[i:i + bs], device=device)).cpu().numpy() for i in range(0, len(F), bs)])
        for j in range(Zs.shape[1]):
            t = t.append_column(f"z_{j}", pa.array(Zs[:, j].astype(np.float32)))
        if writer is None: writer = pq.ParquetWriter(f"{out_table_dir}/features.parquet", t.schema)
        writer.write_table(t); n += len(F)
        print(f"row group {rg + 1}/{pf.num_row_groups}: {n} rows ({time.time()-t0:.0f} s)", flush=True)
    writer.close()
    for name in ("labels.parquet", "targets.parquet"):
        src = os.path.abspath(f"{table_dir}/{name}"); dst = f"{out_table_dir}/{name}"
        if os.path.exists(src) and not os.path.exists(dst): os.symlink(src, dst)
