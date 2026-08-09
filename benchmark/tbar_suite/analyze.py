#!/usr/bin/env python3
"""Turn suite JSONs into human-readable tables + an accuracy comparison of final equilibria.
Accuracy: compare each run's final water-table raster to a reference (Anderson at the smallest dt),
over land cells, reporting max / mean / 99th-percentile |diff|.

Usage: analyze.py <json1> [<json2> ...]"""
import sys, json, glob, os
import numpy as np
import rasterio

def load(j):
    return json.load(open(j))

def land_diff(a_path, b_path):
    with rasterio.open(a_path) as ra, rasterio.open(b_path) as rb:
        a = ra.read(1).astype(float); b = rb.read(1).astype(float)
    if a.shape != b.shape:
        return None
    d = np.abs(a - b)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    return dict(max=float(d.max()), mean=float(d.mean()), p99=float(np.percentile(d, 99)))

def cold_table(res):
    # group by weeks, columns = solvers
    weeks = sorted({r["weeks"] for r in res})
    solvers = []
    for r in res:
        if r["solver"] not in solvers:
            solvers.append(r["solver"])
    print("\n=== COLD START: status / wall(s) / total nonlinear iters, by dt and solver ===")
    hdr = "dt".ljust(6) + "".join(s.ljust(20) for s in solvers)
    print(hdr)
    for wk in weeks:
        row = f"{wk}wk".ljust(6)
        for s in solvers:
            r = next((x for x in res if x["weeks"] == wk and x["solver"] == s), None)
            if r:
                cell = f"{r['status']},{r['wall']:.0f}s,{r['tot_iters']}it"
            else:
                cell = "-"
            row += cell.ljust(20)
        print(row)
    # max stable dt per solver
    print("\nmax converged dt (weeks) per solver:")
    for s in solvers:
        oks = [r["weeks"] for r in res if r["solver"] == s and r["status"] == "OK"]
        print(f"  {s:16s}: {max(oks) if oks else 'none'}")

def warm_table(res):
    solvers = []
    for r in res:
        if r["solver"] not in solvers:
            solvers.append(r["solver"])
    print("\n=== WARM PERTURBATION: step ceiling (largest converged dt) + cost ===")
    for s in solvers:
        rows = [r for r in res if r["solver"] == s]
        oks = [r for r in rows if r["status"] == "OK"]
        ceil = max((r["weeks"] for r in oks), default=None)
        detail = " ".join(f"{r['weeks']}wk:{r['status'][0]}({r['tot_iters']})" for r in rows)
        print(f"  {s:16s} ceiling={ceil}wk | {detail}")

def accuracy(res, ref_solver="anderson"):
    # reference: chosen solver at smallest dt with an OK status and a raster
    cand = sorted([r for r in res if r["solver"] == ref_solver and r["status"] == "OK" and r.get("raster")],
                  key=lambda r: r["weeks"])
    if not cand:
        print("\n(no accuracy reference available)"); return
    ref = cand[0]
    print(f"\n=== ACCURACY: final equilibrium |diff| vs {ref_solver}@{ref['weeks']}wk (land cells, m) ===")
    for r in res:
        if not r.get("raster") or r["status"] != "OK":
            continue
        d = land_diff(ref["raster"], r["raster"])
        if d:
            print(f"  {r['solver']:16s} {r['weeks']:3d}wk: max={d['max']:.3f} mean={d['mean']:.4f} p99={d['p99']:.3f}")

if __name__ == "__main__":
    for j in sys.argv[1:]:
        res = load(j)
        print("\n" + "#" * 70 + f"\n# {os.path.basename(j)}\n" + "#" * 70)
        if "cold" in j:
            cold_table(res); accuracy(res, "anderson")
        elif "warm" in j:
            warm_table(res)
        else:
            cold_table(res); accuracy(res, "anderson")
