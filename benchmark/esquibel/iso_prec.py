#!/usr/bin/env python3
# Iso-precision comparison from the scaling logs: for each size, how many iterations does adaptive need to
# reach cc's FINAL per-cycle precision (RMS of per-cycle |dwtd|)? cc's own iso-precision cost = its full run.
import re, os, sys, glob
R = sys.argv[1] if len(sys.argv) > 1 else "."

def parse(p):
    cum, rows = 0, []
    for line in open(p, errors="ignore"):
        m = re.search(r"Number of nonlinear iterations = (\d+)", line)
        if m:
            cum += int(m.group(1)); continue
        c = re.search(r"cycle (\d+): per-cycle \|.*?rms=([0-9.eE+-]+)", line)
        if c:
            rows.append((int(c.group(1)), cum, float(c.group(2))))
    return rows

# wall from the summary CSV (total), to scale iso-precision wall by iters fraction
wall = {}
csv = os.path.join(R, "scaling.csv")
if os.path.exists(csv):
    for ln in open(csv).readlines()[1:]:
        f = ln.strip().split(",")
        if len(f) >= 7:
            wall[(f[0], f[2])] = (float(f[4]), int(f[5]))  # (tiles,method)->(wall_s,its)

hdr = ("size", "cells", "cc_prec_rms", "cc_iters", "cc_wall_s",
       "adapt@ccPrec_cyc", "adapt@ccPrec_iters", "adapt@ccPrec_wall_s~",
       "adapt_full_iters", "isoPrec_iter_spdup", "isoPrec_wall_spdup~")
print("  ".join(f"{h:>18}" for h in hdr))
for t in (1, 2, 3, 4):
    tiles = f"{t}x{t}"; cells = (t*451)*(t*853)
    ccp, adp = f"{R}/{tiles}_cc.log", f"{R}/{tiles}_adapt.log"
    if not (os.path.exists(ccp) and os.path.exists(adp)):
        continue
    ccr, adr = parse(ccp), parse(adp)
    if not ccr or not adr:
        continue
    cc_it, cc_rms = ccr[-1][1], ccr[-1][2]
    iso = next((r for r in adr if r[2] <= cc_rms), None)
    if iso is None:
        print(f"{tiles:>18}  {cells:>18}  {cc_rms:>18.5f}  {cc_it:>18}  (adapt never as coarse as cc)")
        continue
    ad_full = adr[-1][1]
    cc_w, _ = wall.get((tiles, "cc"), (float("nan"), cc_it))
    ad_w, ad_w_it = wall.get((tiles, "adapt"), (float("nan"), ad_full))
    ad_iso_w = ad_w * iso[1] / ad_w_it if ad_w_it else float("nan")   # ~ scale wall by iter fraction
    it_sp = cc_it / iso[1] if iso[1] else float("nan")
    w_sp = cc_w / ad_iso_w if ad_iso_w else float("nan")
    vals = (tiles, cells, f"{cc_rms:.5f}", cc_it, f"{cc_w:.1f}",
            iso[0], iso[1], f"{ad_iso_w:.1f}", ad_full, f"{it_sp:.2f}x", f"{w_sp:.2f}x")
    print("  ".join(f"{str(v):>18}" for v in vals))
