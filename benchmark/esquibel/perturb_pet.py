#!/usr/bin/env python3
"""Build a perturbed Esquibel domain for the transient algorithmic-speedup experiment.

A FACTOR-f change in the net atmospheric forcing P-ET, applied by perturbing precip only
(evaporation left untouched):

    precip' = f*P + (1-f)*E   ==>   (P' - E) = f*(P - E)

  f=0.8  -> -20% P-ET (drying; deepens the water table toward the steep exp-T = the telling case)
  f=1.2  -> +20% P-ET (wetting control)
  f=1.1  -> the earlier island +10% test

Mirrors every input raster into <out> (symlinks), overwrites precipitation with the perturbed
one, and drops the supplied equilibrium water table in as the warm-start (Esquibel_010000_wtd.tif).

Usage: perturb_pet.py <domain_dir> <out_dir> <equilibrium_wtd.tif> <factor>
"""
import os, sys, glob, shutil
import numpy as np, rasterio

dom, out, eqwtd, factor = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
os.makedirs(out, exist_ok=True)

# 1. mirror all inputs as symlinks
for f in glob.glob(os.path.join(dom, "Esquibel_*.tif")):
    dst = os.path.join(out, os.path.basename(f))
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(f), dst)

# 2. perturbed precip = f*P + (1-f)*E (a real file, replacing the symlink)
P = os.path.join(dom, "Esquibel_010000_precipitation.tif")
E = os.path.join(dom, "Esquibel_010000_evaporation.tif")
with rasterio.open(P) as rp, rasterio.open(E) as re_:
    p = rp.read(1).astype("float64"); e = re_.read(1).astype("float64"); prof = rp.profile
newp = factor * p + (1.0 - factor) * e
dstP = os.path.join(out, "Esquibel_010000_precipitation.tif")
if os.path.lexists(dstP):
    os.remove(dstP)
with rasterio.open(dstP, "w", **prof) as d:
    d.write(newp.astype(prof["dtype"]), 1)
print(f"perturbed precip written (f={factor}: {factor}P+{1-factor:.3g}E); "
      f"mean P {p.mean():.4g} -> {newp.mean():.4g};  mean(P-E) {(p-e).mean():.4g} -> {(newp-e).mean():.4g}")

# 3. warm-start water table = the equilibrium output.
# WTM's supplied_wt path reads {region}_{time}_starting_wt.tif (src/irf.cpp:139) -- NOT _wtd.tif.
dstW = os.path.join(out, "Esquibel_010000_starting_wt.tif")
if os.path.exists(eqwtd):
    if os.path.lexists(dstW):
        os.remove(dstW)
    shutil.copy(eqwtd, dstW)
    print(f"warm-start wtd <- {os.path.basename(eqwtd)}")
else:
    print(f"WARNING: eq wtd '{eqwtd}' not found -> transient warm-start unavailable")
print(f"perturbed domain ready: {out}")
