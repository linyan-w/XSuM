#!/usr/bin/env python3
"""Does the shipped topology head P(class | truth) carry the exit <-> n_pfp=3 link of CC + pion events? Compares, on the M2 test
slices, P(n_pfp=3) real / generated / head probability against the truth-predicted exit and in (p_mu, cos theta_mu) cells."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
from xsum.gen.slc import Surrogate
from xsum.train.baselines import gbdt
from xsum.train.m2 import topo_input
D="data/prod_v10_14_02_03_official/table_m3"
sur=Surrogate("reports/m2_official_m3")
R=pd.read_parquet("reports/m2_official_m3/real_test.parquet"); G=pd.read_parquet("reports/m2_official_m3/generated_test.parquet").loc[R.index]
IDX=["chunk","__ntuple","entry","rec.mc.nu..index"]
def read_rows(path, cols, index):
    """stream the parquet in batches, keep the rows of `index` (memory: one batch at a time)"""
    want=set(zip(*[index.get_level_values(n) for n in IDX])); parts=[]
    for b in pq.ParquetFile(path).iter_batches(batch_size=1_000_000, columns=cols+IDX):
        d=b.to_pandas(); d=d.reset_index() if "chunk" not in d.columns else d; k=pd.MultiIndex.from_frame(d[IDX]); keep=np.fromiter((t in want for t in k), bool, len(d))
        parts.append(d[keep].set_index(IDX))
    return pd.concat(parts)
F=read_rows(f"{D}/features.parquet", sur.features, R.index).loc[R.index]
C=sur.standardise(F.to_numpy(np.float32))
FEAT=[c for c in sur.features if not c.startswith("z_")]; X=F[FEAT].to_numpy(np.float32)
def fv(x,y,z): return (x>-190)&(x<190)&(y>-190)&(y<190)&(z>10)&(z<450)
ex=lambda d: ~fv(*[d[f"vtx_{a}"]+d.long_trk_len*d[f"long_dir{a}"] for a in "xyz"]).to_numpy()
cc=((F.n_mu>=1)&(F.n_pi>=1)).to_numpy(); m=cc&fv(R.vtx_x,R.vtx_y,R.vtx_z).to_numpy()&(R.nu_score>0.45).to_numpy()&R.long_trk_len.notna().to_numpy()
n3r=(R.n_pfp==3).to_numpy(); n3g=(G.n_pfp==3).to_numpy(); exr=ex(R)
rng=np.random.default_rng(0); mm=np.flatnonzero(m); rng.shuffle(mm); tr,te=mm[:len(mm)//2],mm[len(mm)//2:]
pe=gbdt(X[tr],exr[tr].astype(int),max_iter=300).predict_proba(X)[:,1]; pred=pe>0.5
sel=np.zeros(len(R),bool); sel[te]=True; s=sel&m
n3cls=np.array([c[0]==3 for c in sur.topo.classes]+[False])
P=sur.topo_head.predict_proba(C[s])[:,n3cls].sum(1)
pg=gbdt(X[tr],n3r[tr].astype(int),max_iter=300).predict_proba(X[s])[:,1]
ll=lambda p,y: float(-np.mean(y*np.log(np.clip(p,1e-9,1))+(1-y)*np.log(np.clip(1-p,1e-9,1))))
e=pred[s]; y=n3r[s]
print(f"link with truth-predicted exit: real n3 {y[e].mean()-y[~e].mean():+.3f} | gen n3 {n3g[s][e].mean()-n3g[s][~e].mean():+.3f} | shipped MLP head P(n3|truth) {P[e].mean()-P[~e].mean():+.3f} | GBDT P(n3|truth) fit on real {pg[e].mean()-pg[~e].mean():+.3f}")
print(f"binary n3 logloss on real test: MLP head {ll(P,y):.4f}  GBDT {ll(pg,y):.4f}  marginal {ll(np.full(len(y),y.mean()),y):.4f}")
# 2D dependence: P(n3 | p_mu bin x cosz bin) real / gen / MLP
pm=F.lead_p_mu.to_numpy(); cz=F.lead_cosz_mu.to_numpy()
qp=np.quantile(pm[m],[0,.33,.66,1]); qc=np.quantile(cz[m],[0,.33,.66,1])
print("P(n_pfp=3 | p_mu tercile, cosz tercile): real / gen / MLP head   (rows p_mu low->high, cols cosz low->high)")
Ps=np.full(len(R),np.nan); Ps[s]=P
for i in range(3):
    row=[]
    for j in range(3):
        k=s&(pm>=qp[i])&(pm<=qp[i+1])&(cz>=qc[j])&(cz<=qc[j+1])
        row.append(f"{n3r[k].mean():.3f}/{n3g[k].mean():.3f}/{np.nanmean(Ps[k]):.3f}")
    print("   p_mu %.2f-%.2f | "%(qp[i],qp[i+1])+"  ".join(row))
# dependence on the exit predictor itself, in bins
q=np.quantile(pe[s],np.linspace(0,1,6))
print("P(n3 | truth-predicted exit probability bin): real / gen / MLP")
for k in range(5):
    kk=s&(pe>=q[k])&(pe<=q[k+1]); print(f"   P(exit) {q[k]:.2f}-{q[k+1]:.2f}  {n3r[kk].mean():.3f} / {n3g[kk].mean():.3f} / {np.nanmean(Ps[kk]):.3f}   n {kk.sum()}")
