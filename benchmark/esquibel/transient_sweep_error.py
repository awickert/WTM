#!/usr/bin/env python3
"""Post-process the transient dt-sweep (#93): trajectory error vs dt, paired with wall time.

For each forcing (dry/wet) the reference "truth" is the TR-BDF2 dt=0.25wk final water table.
Each test run's final wtd is compared to that reference over LAND cells (mask==1, edges dropped):
  L_inf = max|wtd - wtd_ref|,  RMS = sqrt(mean (wtd-wtd_ref)^2).
Emits a table (order-of-accuracy slope should be ~1 for cc, ~2 for tr) and pairs each dt with
its wall time so we can read off wall-time-at-fixed-accuracy.

Usage: transient_sweep_error.py <sweep_dir> <domain_dir>
"""
import os, sys, csv, glob
import numpy as np, rasterio

sweep, dom = sys.argv[1], sys.argv[2]

# land mask (edges already 0 in the model; drop them here too for a clean comparison)
with rasterio.open(os.path.join(dom, "Esquibel_010000_mask.tif")) as m:
    land = m.read(1) > 0.5
land[0, :] = land[-1, :] = land[:, 0] = land[:, -1] = False

def read(tif):
    with rasterio.open(tif) as r:
        return r.read(1).astype("float64")

# gather run rows
runs = []
for csvf in sorted(glob.glob(os.path.join(sweep, "runs_*.csv"))):
    with open(csvf) as f:
        for row in csv.DictReader(f):
            row["dt_wk"] = float(row["dt_wk"]); row["wall_s"] = float(row["wall_s"])
            row["rc"] = int(row["rc"]); runs.append(row)

# resolve final_tif relative to esquibel dir (paths in CSV are results/algo/... relative to ESQ)
ESQ = os.path.abspath(os.path.join(sweep, os.pardir, os.pardir, os.pardir, os.pardir))
def resolve(p): return p if os.path.isabs(p) else os.path.join(ESQ, p)

# reference per forcing = tr, dt=0.25
ref = {}
for r in runs:
    if r["method"] == "tr" and abs(r["dt_wk"] - 0.25) < 1e-9 and r["final_tif"] != "MISSING":
        ref[r["forcing"]] = read(resolve(r["final_tif"]))

out = []
for r in runs:
    if abs(r["dt_wk"] - 0.25) < 1e-9:   # reference itself: skip as a test point
        continue
    fo = r["forcing"]
    if fo not in ref or r["final_tif"] == "MISSING" or r["rc"] != 0:
        out.append({**r, "Linf": np.nan, "RMS": np.nan}); continue
    w = read(resolve(r["final_tif"]))
    d = (w - ref[fo])[land]
    out.append({**r, "Linf": float(np.max(np.abs(d))), "RMS": float(np.sqrt(np.mean(d**2)))})

# write + print
os.makedirs(sweep, exist_ok=True)
outcsv = os.path.join(sweep, "sweep_errors.csv")
cols = ["forcing", "method", "dt_wk", "steps", "rc", "wall_s", "Linf", "RMS"]
with open(outcsv, "w", newline="") as f:
    wcsv = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); wcsv.writeheader()
    for r in sorted(out, key=lambda x: (x["forcing"], x["method"], x["dt_wk"])):
        wcsv.writerow(r)

print(f"reference = TR-BDF2 dt=0.25wk (per forcing); land cells: {int(land.sum())}")
for fo in sorted(set(r["forcing"] for r in out)):
    print(f"\n=== {fo} ===")
    print(f"{'method':6} {'dt_wk':>6} {'steps':>6} {'wall_s':>7} {'L_inf(m)':>11} {'RMS(m)':>11}  {'order*':>7}")
    for meth in ("cc", "tr"):
        rows = sorted([r for r in out if r["forcing"] == fo and r["method"] == meth], key=lambda x: x["dt_wk"])
        prev = None
        for r in rows:
            order = ""
            if prev and np.isfinite(r["RMS"]) and np.isfinite(prev["RMS"]) and prev["RMS"] > 0 and r["RMS"] > 0:
                order = f"{np.log(r['RMS']/prev['RMS'])/np.log(r['dt_wk']/prev['dt_wk']):.2f}"
            li = f"{r['Linf']:.4e}" if np.isfinite(r["Linf"]) else "diverge"
            rm = f"{r['RMS']:.4e}" if np.isfinite(r["RMS"]) else "diverge"
            print(f"{meth:6} {r['dt_wk']:>6} {r['steps']:>6} {r['wall_s']:>7.0f} {li:>11} {rm:>11}  {order:>7}")
            if np.isfinite(r["RMS"]): prev = r
print(f"\nwrote {outcsv}")
print("*order = local convergence slope in RMS between successive dt (expect ~1 for cc, ~2 for tr)")
