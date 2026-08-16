#!/usr/bin/env python3
"""Organize the cc / fixed_tr / adaptive scaling data (single-node core sweep + multi-node).

Reads results/scaling/scaling_ncore.csv (single-node core sweep) and, if present,
results/scaling/scaling_multinode.csv (multi-node), plus the per-run logs
(nc<cores>_<tiles>_<method>.log) and emits, per grid size:

  SINGLE-NODE  cores | cc its/wall | fixed_tr its/wall | adapt its/wall | adapt its@iso-prec |
               cc/adapt-iso iter ratio | strong-scaling wall speedup (cc / fixed_tr / adapt)
  MULTI-NODE   nodes x ranks | cc | fixed_tr | adapt  (its / wall / stop_cycle each)

Iterations are the node-independent signal but NOT a fair cross-method cost (a TR-BDF2 iteration is two
staged solves, not one backward-Euler solve). WALL on a shared node is noise-limited (the --exclusive pass
replaces it). iso-prec = the cycle at which adaptive's per-cycle RMS(dwtd) first drops to cc's FINAL value.

    scaling_report.py [results/scaling]
"""
import sys, os, re, csv
from collections import defaultdict
R = sys.argv[1] if len(sys.argv) > 1 else "results/scaling"

METHODS = ("cc", "fixed_tr", "adapt")

def load(path):
    out = []
    if not os.path.exists(path): return out
    with open(path) as f:
        for r in csv.DictReader(f):
            r["cells"] = int(r["cells"]); r["wall_s"] = float(r["wall_s"])
            r["snes_its"] = int(r["snes_its"])
            r["stop_cycle"] = int(r["stop_cycle"]) if r.get("stop_cycle") not in (None, "") else 0
            out.append(r)
    return out

def cell(v, key, fmt="{}"):
    """Format field `key` of row `v`, or '-' if the row is missing."""
    return fmt.format(v[key]) if v else "-"

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
    if not cc or not ad: return None
    target = cc[-1][2]
    hit = next((r for r in ad if r[2] <= target), None)
    return hit[1] if hit else None

# ---------------------------------------------------------------- single-node
rows = load(os.path.join(R, "scaling_ncore.csv"))
for r in rows: r["cores"] = int(r["cores"])
by_size = defaultdict(dict)   # (tiles,cells) -> {(cores,method): row}
for r in rows:
    by_size[(r["tiles"], r["cells"])][(r["cores"], r["method"])] = r

print("=" * 104)
print("SINGLE-NODE core sweep (agsmall, shared -> wall noise-limited; iterations are the clean signal)")
print("=" * 104)
hdr = (f"{'cores':>5} {'cc_its':>7} {'cc_wall':>8} {'ftr_its':>8} {'ftr_wall':>9} "
       f"{'ad_its':>7} {'ad_wall':>8} {'ad_iso':>7} {'cc/adIso':>9} {'ccSpd':>7} {'ftrSpd':>7} {'adSpd':>7}")
for (tiles, cells) in sorted(by_size, key=lambda k: k[1]):
    d = by_size[(tiles, cells)]
    cores_list = sorted({c for (c, m) in d})
    print(f"\n### {tiles}  ({cells:,} cells) ###")
    print(hdr)
    base = {}   # method -> (cores, wall) at the smallest core count where it appears
    for c in cores_list:
        got = {m: d.get((c, m)) for m in METHODS}
        for m in METHODS:
            if got[m] and m not in base: base[m] = (c, got[m]["wall_s"])
        iso = iso_prec_iters(c, tiles)
        isor = f"{got['cc']['snes_its'] / iso:.2f}x" if (got["cc"] and iso) else "-"
        def spd(m):
            v = got[m]
            return f"{base[m][1] / v['wall_s']:.2f}x" if (v and v["wall_s"] and m in base) else "-"
        print(f"{c:>5} "
              f"{cell(got['cc'],'snes_its'):>7} {cell(got['cc'],'wall_s','{:.0f}'):>8} "
              f"{cell(got['fixed_tr'],'snes_its'):>8} {cell(got['fixed_tr'],'wall_s','{:.0f}'):>9} "
              f"{cell(got['adapt'],'snes_its'):>7} {cell(got['adapt'],'wall_s','{:.0f}'):>8} "
              f"{(str(iso) if iso else '-'):>7} {isor:>9} "
              f"{spd('cc'):>7} {spd('fixed_tr'):>7} {spd('adapt'):>7}")
    bstr = ", ".join(f"{m}@{base[m][0]}c" for m in METHODS if m in base)
    print(f"  (strong-scaling wall speedups vs smallest core count present [{bstr}]; wall shared-node noisy)")

# ---------------------------------------------------------------- multi-node
mn = load(os.path.join(R, "scaling_multinode.csv"))
if mn:
    for r in mn: r["nodes"] = int(r["nodes"]); r["ranks"] = int(r["ranks"])
    mby = defaultdict(dict)  # (tiles,cells) -> {(nodes,ranks,method): row}
    for r in mn:
        mby[(r["tiles"], r["cells"])][(r["nodes"], r["ranks"], r["method"])] = r
    print("\n" + "=" * 104)
    print("MULTI-NODE (msilarge, shared+cross-node -> wall directional; de-risks the rank-0 FSM gather at scale)")
    print("=" * 104)
    mhdr = (f"{'layout':>10} {'cc_its':>7} {'cc_wall':>8} {'cc_cyc':>6} "
            f"{'ftr_its':>8} {'ftr_wall':>9} {'ftr_cyc':>7} {'ad_its':>7} {'ad_wall':>8} {'ad_cyc':>6}")
    for (tiles, cells) in sorted(mby, key=lambda k: k[1]):
        d = mby[(tiles, cells)]
        layouts = sorted({(nd, rk) for (nd, rk, m) in d})
        print(f"\n### {tiles}  ({cells:,} cells) ###")
        print(mhdr)
        for (nd, rk) in layouts:
            cc = d.get((nd, rk, "cc")); ft = d.get((nd, rk, "fixed_tr")); ad = d.get((nd, rk, "adapt"))
            print(f"{f'{nd}x{rk}':>10} "
                  f"{cell(cc,'snes_its'):>7} {cell(cc,'wall_s','{:.0f}'):>8} {cell(cc,'stop_cycle'):>6} "
                  f"{cell(ft,'snes_its'):>8} {cell(ft,'wall_s','{:.0f}'):>9} {cell(ft,'stop_cycle'):>7} "
                  f"{cell(ad,'snes_its'):>7} {cell(ad,'wall_s','{:.0f}'):>8} {cell(ad,'stop_cycle'):>6}")
    print("\n  layout = nodes x total-ranks.  cyc = stop_cycle (eq_metric frac).  NB over-decomposition")
    print("  (small domain, many ranks) jitters the per-cycle stop metric -> inflated cc/adapt cycles;")
    print("  fixed_tr's cycle count stays decomposition-invariant. See SCALING.md.")
