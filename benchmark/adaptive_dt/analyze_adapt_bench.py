#!/usr/bin/env python3
"""Analyze the adaptive-dt benchmark (see README.md).

For a regime (transient|spinup), pair each run's cost (SNES iterations + fractional wall) with its accuracy
(mean/max |wtd error| vs the finest constant-dt run, the dt->0 reference). The question: do adaptive runs
sit ON or BELOW the constant-dt cost-accuracy curve?

Usage: analyze_adapt_bench.py <transient|spinup> [results_dir]   (results_dir defaults to ./results)
"""
import sys, os, numpy as np, rasterio

here = os.path.dirname(os.path.abspath(__file__))
reg  = sys.argv[1] if len(sys.argv) > 1 else "transient"
OUT  = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "results")

rows = [l.strip().split(",") for l in open(os.path.join(OUT, "adapt_bench_%s.csv" % reg))][1:]
consts = [r for r in rows if r[1] == "const" and r[7] != "MISSING" and os.path.exists(r[7])]
if not consts:
    print("[%s] no converged const reference tif in %s" % (reg, OUT)); sys.exit(0)
ref_row = min(consts, key=lambda r: float(r[2]))          # finest constant dt = dt->0 reference
ref = rasterio.open(ref_row[7]).read(1).astype(float); fin0 = np.isfinite(ref)
print("[%s] reference = const dt=%s (%d land cells)" % (reg, ref_row[2], int(fin0.sum())))
print("%-10s%7s%4s%9s%8s%8s%11s%10s" % ("run", "param", "rc", "wall_s", "its", "nsteps", "mean_err", "max_err"))
for r in rows:
    run, par, rc, wall, its, ns, tif = r[1], r[2], r[3], r[4], r[5], r[6], r[7]
    if tif == "MISSING" or not os.path.exists(tif):
        print("%-10s%7s%4s%9s%8s%8s   (no tif; rc=%s)" % (run, par, rc, wall, its, ns, rc)); continue
    a = rasterio.open(tif).read(1).astype(float); fin = fin0 & np.isfinite(a)
    me = float(np.mean(np.abs((a - ref)[fin]))); mx = float(np.max(np.abs((a - ref)[fin])))
    print("%-10s%7s%4s%9s%8s%8s%11.4f%10.3f" % (run, par, rc, wall, its, ns, me, mx))
