#!/usr/bin/env python3
"""Technote tables for the M3 section: heads (M2 v3 vs M2 with M3 inputs), per-event real-vs-generated correlations,
cohpi signal cut flow, CC1p0pi acceptance test (old vs new surrogate). Writes reports/technote/m3_*.tex and
reports/m3_official/comparison.json."""
import json, os, re
import numpy as np, pandas as pd

OUT = "reports/technote"
A, B = "reports/m2_official_v3", "reports/m2_official_m3"
ma, mb = json.load(open(f"{A}/metrics.json")), json.load(open(f"{B}/metrics.json"))
comp = {}

# heads
names = {"short": "second track", "two": "two-track block", "s1": "hits within 1\\,cm", "s2": "hits within 2\\,cm",
         "is_muon_contained": "muon-candidate contained", "is_cpi_contained": "pion-candidate contained",
         "spike_opening_angle": "back-to-back pair"}
rows = [r"\begin{tabular}{lrrrr}\toprule head & AUC (v3) & AUC (M3) & log loss (v3) & log loss (M3)\\\midrule"]
for k, n in names.items():
    a, b = ma["presence"][k], mb["presence"][k]
    rows.append(f"{n} & {a['auc']:.3f} & {b['auc']:.3f} & {a['logloss']:.3f} & {b['logloss']:.3f}\\\\")
ta, tb = ma["topology"], mb["topology"]
rows.append(r"\midrule topology (121 classes) & \multicolumn{2}{c}{accuracy " + f"{ta['accuracy']:.3f} / {tb['accuracy']:.3f}" + "} & "
            + f"{ta['logloss']:.3f} & {tb['logloss']:.3f}\\\\")
rows.append(f"matched slices per neutrino (0/1/2+) & & & {ma['n_matched_slices']['logloss']:.3f} & {mb['n_matched_slices']['logloss']:.3f}\\\\")
rows.append(r"\bottomrule\end{tabular}")
open(f"{OUT}/m3_heads.tex", "w").write("\n".join(r for r in rows if r) + "\n")
comp["heads"] = {k: [ma["presence"][k]["auc"], mb["presence"][k]["auc"]] for k in names}
comp["flow_val"] = [ma["flow_val_curve"][-1]["val"], mb["flow_val_curve"][-1]["val"]]

# per-event correlations on the same test slices
cols = ["vtx_x", "long_dirx", "long_diry", "long_dirz", "short_dirx", "short_diry", "short_dirz", "muon_end_x", "muon_end_y", "muon_end_z", "cpi_end_x"]
corr = {}
for tag, d in (("v3", A), ("M3", B)):
    R, G = pd.read_parquet(f"{d}/real_test.parquet"), pd.read_parquet(f"{d}/generated_test.parquet")
    j = R.index.intersection(G.index); R, G = R.loc[j], G.loc[j]
    corr[tag] = {}
    for c in cols:
        a, b = R[c].astype(float).to_numpy(), G[c].astype(float).to_numpy(); m = np.isfinite(a) & np.isfinite(b)
        corr[tag][c] = float(np.corrcoef(a[m], b[m])[0, 1])
    corr[tag]["sign_dirx"] = float(np.mean(np.sign(R.long_dirx) == np.sign(G.long_dirx)))
rows = [r"\begin{tabular}{lrr}\toprule column & v3 & M3\\\midrule"]
for c in cols:
    rows.append(f"\\texttt{{{c.replace('_', chr(92) + '_')}}} & {corr['v3'][c]:+.2f} & {corr['M3'][c]:+.2f}\\\\")
rows.append(f"same sign of \\texttt{{long\\_dirx}} & {corr['v3']['sign_dirx']:.2f} & {corr['M3']['sign_dirx']:.2f}\\\\")
rows.append(r"\bottomrule\end{tabular}")
open(f"{OUT}/m3_corr.tex", "w").write("\n".join(rows) + "\n")
comp["corr"] = corr

# cohpi signal cut flow
st = ["pe_bcfm", "2trk", "dir", "chi2", "sphere", "cathode", "sr"]
lab = ["PE, BCFM", "2 tracks", "directions", "chi2", "sphere", "cathode", "signal region"]
rows = [r"\begin{tabular}{l rrr rrr}\toprule & \multicolumn{3}{c}{test files} & \multicolumn{3}{c}{all files}\\ stage & \real{} & v3 & M3 & \real{} & v3 & M3\\\midrule"]
ct, ca = json.load(open("reports/m3_official/cohpi_signal_cutflow.json")), json.load(open("reports/m3_official/cohpi_signal_cutflow_all.json"))
for s, l in zip(st, lab):
    rows.append(f"{l} & {ct['real'][s]} & {ct['v3'][s]} & {ct['m3'][s]} & {ca['real'][s]} & {ca['v3'][s]} & {ca['m3'][s]}\\\\")
rows.append(r"\bottomrule\end{tabular}")
open(f"{OUT}/m3_cohpi.tex", "w").write("\n".join(rows) + "\n")
comp["cohpi_signal"] = {"test": ct, "all": ca}


# acceptance test
def parse(f):
    out, head = {}, ""
    for l in open(f):
        if l.startswith("test neutrinos"): head = l.strip(); continue
        p = l.replace("|", " ").split()
        if len(p) == 8 and p[1].isdigit(): out[p[0]] = list(map(float, p[1:]))
    return head, out


fa, fb = "reports/cc1p_containment/accept_v3.txt", "reports/cc1p_containment/accept_m3.txt"
if os.path.exists(fb):
    ha, a = parse(fa); hb, b = parse(fb)
    rows = [r"\begin{tabular}{l rrr rrr rrr}\toprule & \multicolumn{3}{c}{\real{}} & \multicolumn{3}{c}{XSuM v3 (summary features)} & \multicolumn{3}{c}{XSuM M3 (particle set)}\\",
            r"stage & sel. & eff. & pur. & eff. & pur. & overlap & eff. & pur. & overlap\\\midrule"]
    for k in a:
        r, s = a[k], b.get(k)
        rows.append(f"{k.replace(chr(95), chr(92) + chr(95))} & {int(r[0])} & {r[1]:.3f} & {r[2]:.3f} & {r[4]:.3f} & {r[5]:.3f} & {r[6]:.2f} & {s[4]:.3f} & {s[5]:.3f} & {s[6]:.2f}\\\\")
    rows.append(r"\bottomrule\end{tabular}")
    open(f"{OUT}/m3_accept.tex", "w").write("\n".join(rows) + "\n")
    comp["accept"] = {"v3": a, "m3": b, "head": hb}
def w1med(path, key):
    m = json.load(open(path))[key]
    return float(np.median([v["w1"] for v in m.values() if "w1" in v]))


comp["w1_median"] = {"m2": [w1med(f"{A}/metrics.json", "per_column"), w1med(f"{B}/metrics.json", "per_column")],
                     "m4": [w1med("reports/m4_official_v2/metrics.json", "per_field"), w1med("reports/m4_official_m3/metrics.json", "per_field")]}
mac = {"mThreeFlowA": f"{comp['flow_val'][0]:.3f}", "mThreeFlowB": f"{comp['flow_val'][1]:.3f}",
       "mThreeWtwoA": f"{comp['w1_median']['m2'][0]:.4f}", "mThreeWtwoB": f"{comp['w1_median']['m2'][1]:.4f}",
       "mThreeWfourA": f"{comp['w1_median']['m4'][0]:.4f}", "mThreeWfourB": f"{comp['w1_median']['m4'][1]:.4f}",
       "mThreeAcceptHead": comp.get("accept", {}).get("head", "").replace("test neutrinos", "test neutrinos:").replace("true CC1p0pi signal", "true CC1p0pi signal")}
open(f"{OUT}/m3_numbers.tex", "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
json.dump(comp, open("reports/m3_official/comparison.json", "w"), indent=1)
print(json.dumps(comp)[:1500])
