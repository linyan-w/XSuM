#!/usr/bin/env python
"""Turn the metrics.json files of the report directories into LaTeX macros and tables for the technical note.

usage: python scripts/technote_numbers.py reports/technote [--m1 reports/m1_scratch1633] [--m2 reports/m2_scratch1633]
Missing inputs produce placeholder macros so the note always compiles.
"""
import argparse
import json
import os


def tex(s):
    return str(s).replace("_", "\\_")


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def fmt(x, nd=3):
    return "--" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else f"{x:,}" if isinstance(x, int) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--m1", default=None); ap.add_argument("--m2", default=None); ap.add_argument("--m4", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mac = ""
    m1 = json.load(open(f"{a.m1}/metrics.json")) if a.m1 and os.path.exists(f"{a.m1}/metrics.json") else None
    m2 = json.load(open(f"{a.m2}/metrics.json")) if a.m2 and os.path.exists(f"{a.m2}/metrics.json") else None
    mac += macro("mOneDir", a.m1.replace("_", "\\_") if a.m1 else "--") + macro("mOneDirBase", os.path.basename(a.m1) if a.m1 else "none")
    if m1:
        mac += macro("nNu", fmt(m1["n"])) + macro("nTrain", fmt(m1["n_train"])) + macro("nTest", fmt(m1["n_test"])) + macro("nFeat", fmt(len(m1["features"])))
        mac += macro("fracMatched", fmt(100 * m1["frac_matched"], 1)) + macro("fracSelected", fmt(100 * m1["frac_selected"], 3))
        t = m1["tier0"]["matched"]
        mac += macro("matchedLL", fmt(t["logloss"], 4)) + macro("matchedLLm", fmt(t["logloss_marginal"], 4)) + macro("matchedAUC", fmt(t["auc"], 3))
        s = m1["tier0"]["selected_cohpi"]
        mac += macro("selLL", fmt(s.get("logloss"), 5)) + macro("selLLm", fmt(s.get("logloss_marginal"), 5)) + macro("selAUC", fmt(s.get("auc"), 3))
        mm = m1["multiplicity"]
        mac += macro("npfpLL", fmt(mm["logloss"], 3)) + macro("npfpLLm", fmt(mm["logloss_marginal"], 3)) + macro("npfpAcc", fmt(100 * mm["accuracy"], 1)) + macro("npfpAccm", fmt(100 * mm["accuracy_marginal"], 1))
        rows = []
        for tag, label in (("vertex", "vertex residual (3-d)"), ("leadtrack", "leading track: $\\Delta\\cos\\theta$, $\\log$ length"), ("twotrack", "two-track: $\\log$ range-$p$ ratios, $\\Delta\\cos\\theta_\\mu$")):
            if tag in m1["mdn"]:
                r = m1["mdn"][tag]
                rows.append(f"{label} & {r['mdn8']['n_train']:,} & {r['gauss']['test_nll']:.3f} & {r['mdn8']['test_nll']:.3f} \\\\")
        open(f"{a.out}/m1_mdn.tex", "w").write("\\begin{tabular}{lccc}\n\\toprule\nTarget & training rows & Gaussian NLL & MDN ($K=8$) NLL \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        rows = []
        for tag in ("vertex", "leadtrack", "twotrack"):
            if tag in m1["mdn"]:
                for k, v in m1["mdn"][tag]["mdn8"].items():
                    if isinstance(v, dict) and "w1" in v:
                        rq, fq = v["real_q16_50_84"], v["fake_q16_50_84"]
                        rows.append(f"{tex(k)} & {rq[1]:.3f} [{rq[0]:.3f}, {rq[2]:.3f}] & {fq[1]:.3f} [{fq[0]:.3f}, {fq[2]:.3f}] & {v['w1']:.4f} \\\\")
        open(f"{a.out}/m1_marginals.tex", "w").write("\\begin{tabular}{lccc}\n\\toprule\nVariable & Pandora median [16\\%, 84\\%] & MDN median [16\\%, 84\\%] & $W_1$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        cal = m1["tier0"]["selected_cohpi"].get("calibration", {}).get("nuint_categ", [])
        names = {-1: "out of FV", 0: "NC", 1: "signal", 2: "CC COH (non-signal)", 3: "CCQE", 4: "2p2h", 5: "RES", 6: "DIS", 8: "other"}
        rows = [f"{names.get(r['categ'], r['categ'])} & {r['n']:,} & {100*r['observed']:.3f} & {100*r['predicted']:.3f} \\\\" for r in cal]
        open(f"{a.out}/m1_selected_categ.tex", "w").write("\\begin{tabular}{lccc}\n\\toprule\nTrue category & test neutrinos & selected [\\%] & predicted [\\%] \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    else:
        for k in ("nNu", "nTrain", "nTest", "nFeat", "fracMatched", "fracSelected", "matchedLL", "matchedLLm", "matchedAUC", "selLL", "selLLm", "selAUC", "npfpLL", "npfpLLm", "npfpAcc", "npfpAccm"):
            mac += macro(k, "--")
        for f in ("m1_mdn.tex", "m1_marginals.tex", "m1_selected_categ.tex"):
            open(f"{a.out}/{f}", "w").write("(not yet available)\n")
    if m2:
        mac += macro("mTwoMatched", fmt(m2["n_matched"])) + macro("mTwoEpochs", fmt(m2["epochs"]))
        pc = m2["per_column"]
        rows = [f"{tex(c)} & {100*v['frac_present_real']:.1f} & {100*v['frac_present_gen']:.1f} & {v['w1']:.4f} \\\\" for c, v in pc.items()]
        open(f"{a.out}/m2_columns.tex", "w").write("\\begin{longtable}{lccc}\n\\toprule\nColumn & present, real [\\%] & present, generated [\\%] & $W_1$ (transformed units) \\\\\n\\midrule\n\\endhead\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}\n")
        cf = m2["cohpi_cutflow_test"]; cs = m2.get("cohpi_cutflow_test_signal", {})
        rows = [f"{tex(k)} & {cf['real'][k]:,} & {cf['generated'][k]:,} & {cs.get('real', {}).get(k, '--')} & {cs.get('generated', {}).get(k, '--')} \\\\" for k in cf["real"]]
        open(f"{a.out}/m2_cutflow.tex", "w").write("\\begin{tabular}{lcccc}\n\\toprule\nStage & all, Pandora & all, generated & signal, Pandora & signal, generated \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        pr = m2["presence"]
        rows = [f"{tex(k)} & {100*v['rate']:.1f} & {v['logloss_marginal']:.4f} & {v['logloss']:.4f} & {v['auc']:.3f} \\\\" for k, v in pr.items()]
        open(f"{a.out}/m2_presence.tex", "w").write("\\begin{tabular}{lcccc}\n\\toprule\nGroup / flag & rate [\\%] & log loss (marginal) & log loss (GBDT) & AUC \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        w1 = [v["w1"] for v in pc.values()]
        mac += macro("mTwoWmax", fmt(max(w1), 3)) + macro("mTwoWmed", fmt(sorted(w1)[len(w1) // 2], 3))
    else:
        for k in ("mTwoMatched", "mTwoEpochs", "mTwoWmax", "mTwoWmed"):
            mac += macro(k, "--")
        for f in ("m2_columns.tex", "m2_cutflow.tex", "m2_presence.tex"):
            open(f"{a.out}/{f}", "w").write("(not yet available)\n")
    m4 = json.load(open(f"{a.m4}/metrics.json")) if a.m4 and os.path.exists(f"{a.m4}/metrics.json") else None
    if m4:
        mac += macro("mFourN", fmt(m4["n"])) + macro("mFourEpochs", fmt(m4["epochs"])) + macro("mFourCond", fmt(m4["cond_dim"]))
        mac += macro("mFourNfitAcc", fmt(100 * m4["n_fit"]["accuracy"], 1)) + macro("mFourNfitLL", fmt(m4["n_fit"]["logloss"], 3)) + macro("mFourNfitLLm", fmt(m4["n_fit"]["logloss_marginal"], 3))
        cc = m4["cohpi_counts_test"]; tw = cc["two_track_rate"]
        mac += macro("mFourTwoReal", fmt(100 * tw["real"], 2)) + macro("mFourTwoGen", fmt(100 * tw["gen"], 2)) + macro("mFourTwoSlc", fmt(100 * tw["slc_table"], 2)) + macro("mFourNtest", fmt(tw["n_test"]))
        pf = m4["per_field"]; w1 = sorted(v["w1"] for v in pf.values() if "w1" in v and v["w1"] == v["w1"])
        mac += macro("mFourWmed", fmt(w1[len(w1) // 2], 3)) + macro("mFourWmax", fmt(w1[-1], 3))
        rows = [f"{tex(k)} & {v['mean_real']:.3f} & {v['mean_gen']:.3f} & -- \\\\" if "mean_real" in v else f"{tex(k)} & {v['real_q16_50_84'][1]:.3f} & {v['fake_q16_50_84'][1]:.3f} & {v['w1']:.4f} \\\\" for k, v in pf.items()]
        open(f"{a.out}/m4_fields.tex", "w").write("\\begin{longtable}{lccc}\n\\toprule\nField & Pandora median (flags: mean) & generated & $W_1$ (transformed units) \\\\\n\\midrule\n\\endhead\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}\n")
        rows = [f"{tex(k)} & {cc[k]['mean_real']:.3f} & {cc[k]['mean_gen']:.3f} & " + " / ".join(str(x) for x in cc[k]["dist_real"][:5]) + " & " + " / ".join(str(x) for x in cc[k]["dist_gen"][:5]) + " \\\\" for k in ("n_prong", "n_trk", "n_proton", "n_shower")]
        open(f"{a.out}/m4_counts.tex", "w").write("\\begin{tabular}{lcccc}\n\\toprule\nCount & mean, Pandora & mean, generated & 0/1/2/3/4, Pandora & 0/1/2/3/4, generated \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    else:
        for k in ("mFourN", "mFourEpochs", "mFourCond", "mFourNfitAcc", "mFourNfitLL", "mFourNfitLLm", "mFourTwoReal", "mFourTwoGen", "mFourTwoSlc", "mFourNtest", "mFourWmed", "mFourWmax"):
            mac += macro(k, "--")
        for f in ("m4_fields.tex", "m4_counts.tex"):
            open(f"{a.out}/{f}", "w").write("(not yet available)\n")
    open(f"{a.out}/numbers.tex", "w").write(mac)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
