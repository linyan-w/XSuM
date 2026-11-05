#!/usr/bin/env python3
"""Sum the per-group output of check_cc1p0pi_stages.py -> table of eff, purity, overlap per stage."""
import re
import sys

rows, nsig, nnu, order = {}, 0, 0, []
for l in open(sys.argv[1]):
    m = re.match(r"#jobs .* n_nu (\d+) n_signal (\d+)", l)
    if m: nnu += int(m.group(1)); nsig += int(m.group(2)); continue
    p = l.strip().split(",")
    if len(p) != 6: continue
    k = p[0]; v = list(map(int, p[1:]))
    if k not in rows: rows[k] = [0] * 5; order.append(k)
    rows[k] = [a + b for a, b in zip(rows[k], v)]
print(f"test neutrinos {nnu}, true CC1p0pi signal {nsig}")
print(f"{'stage':20s} {'real sel':>8s} {'eff':>6s} {'pur':>6s} | {'XSuM sel':>8s} {'eff':>6s} {'pur':>6s} | overlap/real")
for k in order:
    rs, rsg, ss, ssg, both = rows[k]
    print(f"{k:20s} {rs:8d} {rsg / nsig:6.3f} {rsg / max(rs, 1):6.3f} | {ss:8d} {ssg / nsig:6.3f} {ssg / max(ss, 1):6.3f} | {both / max(rs, 1):.3f}")
