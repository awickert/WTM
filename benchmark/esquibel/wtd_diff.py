#!/usr/bin/env python3
"""Absolute correctness gate for the algorithmic-speedup sweep.

Compares a run's final water-table raster against a trusted reference-equilibrium
raster and prints  max|w-w_ref|  and  RMS(w-w_ref)  over valid (finite, both-present)
cells. This is the dt-INDEPENDENT check that the model's per-cycle max|Δwtd| auto-stop
alone cannot give: it tells a genuinely-converged run apart from a premature stop or a
limit cycle. See benchmark/esquibel/README (algo_speedup) for how it is used.

    wtd_diff.py <run_final.tif> <reference_eq.tif>
    -> prints:  maxabs=<m>  rms=<m>  ncells=<n>
"""
import sys
import numpy as np
import rasterio

def load(p):
    with rasterio.open(p) as ds:
        a = ds.read(1).astype(np.float64)
        nod = ds.nodata
    if nod is not None:
        a = np.where(a == nod, np.nan, a)
    return a

def main():
    run, ref = sys.argv[1], sys.argv[2]
    a, b = load(run), load(ref)
    if a.shape != b.shape:
        print(f"maxabs=nan  rms=nan  ncells=0  (shape {a.shape} != {b.shape})")
        return
    d = a - b
    m = np.isfinite(d)
    d = d[m]
    if d.size == 0:
        print("maxabs=nan  rms=nan  ncells=0")
        return
    print(f"maxabs={np.max(np.abs(d)):.6g}  rms={np.sqrt(np.mean(d**2)):.6g}  ncells={d.size}")

if __name__ == "__main__":
    main()
