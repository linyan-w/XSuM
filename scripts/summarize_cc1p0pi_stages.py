#!/usr/bin/env python3
"""Sum the per-group output of check_cc1p0pi_stages.py -> tables of eff, purity, overlap per stage, for the primary
signal definition (topology + momentum caps, no truth containment) and the notebook's (with truth containment).
Old 6-field output (contained definition only) is still read."""
import re
import sys

rows, nsig, ntopo, nnu, order = {}, 0, 0, 0, []
for l in open(sys.argv[1]):
    m = re.match(r"#jobs .* n_nu (\d+) n_signal (\d+)(?: n_signal_topo (\d+))?", l)
    if m: nnu += int(m.group(1)); nsig += int(m.group(2)); ntopo += int(m.group(3) or 0); continue
    p = l.strip().split(",")
    if len(p) not in (6, 9): continue
    k = p[0]; v = list(map(int, p[1:])) + [0] * (9 - len(p))
    if k not in rows: rows[k] = [0] * 8; order.append(k)
    rows[k] = [a + b for a, b in zip(rows[k], v)]


def table(title, n, ir, is_, iboth):
    print(f"\n{title}: {n} true signal neutrinos")
    print(f"{'stage':20s} {'real sel':>8s} {'eff':>6s} {'pur':>6s} | {'XSuM sel':>8s} {'eff':>6s} {'pur':>6s} | {'XSuM/real sel':>13s} {'overlap/real':>12s} {'sig overlap':>11s}")
    for k in order:
        r = rows[k]; rs, ss, both = r[0], r[2], r[4]; rsg, ssg = r[ir], r[is_]
        so = f"{r[iboth] / max(rsg, 1):11.3f}" if iboth is not None else f"{'':11s}"
        print(f"{k:20s} {rs:8d} {rsg / n:6.3f} {rsg / max(rs, 1):6.3f} | {ss:8d} {ssg / n:6.3f} {ssg / max(ss, 1):6.3f} | {ss / max(rs, 1):13.3f} {both / max(rs, 1):12.3f} {so}")


print(f"test neutrinos {nnu}")
if ntopo: table("PRIMARY signal: numu CC 1mu1p0pi, p_mu, p_p < 1 GeV, true vertex in FV; no end-point containment", ntopo, 5, 6, 7)
table("Notebook signal: the same + true mu and p end points in the FV of the vertex's TPC", nsig, 1, 3, None)
