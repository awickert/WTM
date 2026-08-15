#!/usr/bin/env python3
"""Organize the single-node multi-core cc-vs-adaptive scaling data.

Reads results/scaling/scaling_ncore.csv (cores,tiles,cells,method,rc,wall_s,snes_its,stop_cycle) plus the
per-run logs (nc<cores>_<tiles>_<method>.log) and emits, per grid size, a cores-sweep table:

  cores | cc its/wall | adapt its/wall | adapt its@iso-prec | iso-prec iter ratio | strong-scaling (wall)

Iterations are the node-independent signal; WALL on a shared node is noise-limited (the --exclusive pass
replaces it). iso-prec = the cycle at which adaptive's per-cycle RMS(dwtd) first drops to cc's FINAL value.

    scaling_report.py [results/scaling]
"""
import sys, os, re, csv
from collections import defaultdict
R = sys.argv[1] if len(sys.argv) > 1 else "results/scaling"

rows = []
with open(os.path.join(R, "scaling_ncore.csv")) as f:
    for r in csv.DictReader(f):
        r["cores"] = int(r["cores"]); r["cells"] = int(r["cells"])
        r["wall_s"] = float(r["wall_s"]); r["snes_its"] = int(r["snes_its"])
        rows.append(r)

def per_cycle_rms(logp):
    cum, out = 0, []
    if not os.path.exists(logp): return out
    for line in open(logp, errors="ignore"):
        m = re.search(r"Number of nonlinear iterations = (\d+)", line)
        if m: cum += int(m.group(1)); continue
        c = re.search(r"cycle (\d+): per-cycle \|.*?rms=([0-9.eE+-]+)", line)
        if c: out.append((int(c.group(1)), cum, float(c.group(2))))
    return out

def iso_prec_iters(cores, tiles):
    cc = per_cycle_rms(os.path.join(R, f"nc{cores}_{tiles}_cc.log"))
    ad = per_cycle_rms(os.path.join(R, f"nc{cores}_{tiles}_adapt.log"))
    # cores=16 rows came from the earlier run with a different prefix
    if not cc: cc = per_cycle_rms(os.path.join(R, f"{tiles}_cc.log"))
    if not ad: ad = per_cycle_rms(os.path.join(R, f"{tiles}_adapt.log"))
    if not cc or not ad: return (None, None)
    target = cc[-1][2]
    hit = next((r for r in ad if r[2] <= target), None)
    return (hit[1] if hit else None, cc[-1][1])

by_size = defaultdict(dict)   # (tiles,cells) -> {(cores,method): row}
for r in rows:
    by_size[(r["tiles"], r["cells"])][(r["cores"], r["method"])] = r

for (tiles, cells) in sorted(by_size, key=lambda k: k[1]):
    d = by_size[(tiles, cells)]
    cores_list = sorted({c for (c, m) in d})
    print(f"\n### {tiles}  ({cells:,} cells) ###")
    print(f"{'cores':>5} {'cc_its':>8} {'cc_wall':>9} {'adapt_its':>10} {'adapt_wall':>11} "
          f"{'adapt_iso':>10} {'isoP_iters':>10} {'cc/adaptIso':>12} {'ccWall_spd':>10} {'adWall_spd':>10}")
    base_cc = base_ad = None
    for c in cores_list:
        cc = d.get((c, "cc")); ad = d.get((c, "adapt"))
        if not cc or not ad: continue
        iso, cc_it = iso_prec_iters(c, tiles)
        if base_cc is None: base_cc, base_ad, base_c = cc["wall_s"], ad["wall_s"], c
        cc_sp = base_cc / cc["wall_s"] if cc["wall_s"] else float("nan")
        ad_sp = base_ad / ad["wall_s"] if ad["wall_s"] else float("nan")
        isor = (cc["snes_its"] / iso) if iso else float("nan")
        print(f"{c:>5} {cc['snes_its']:>8} {cc['wall_s']:>9.1f} {ad['snes_its']:>10} {ad['wall_s']:>11.1f} "
              f"{str(iso):>10} {(cc_it or ''):>10} {isor:>11.2f}x {cc_sp:>9.2f}x {ad_sp:>9.2f}x")
    print(f"  (strong-scaling wall speedups are vs cores={base_c}; wall shared-node noisy -- see --exclusive)")
