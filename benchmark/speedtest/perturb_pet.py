#!/usr/bin/env python3
"""Build a perturbed island domain for the transient experiment.

    precip' = 1.1*P - 0.1*E   ==>   (P' - E) = 1.1 * (P - E)      [an exact P-ET x1.1]

Mirrors every input raster into <out> (symlinks), overwrites precipitation with the perturbed one,
and drops the supplied equilibrium water table in as the warm-start (Esquibel_010000_wtd.tif).

Usage: perturb_pet.py <domain_dir> <out_dir> <equilibrium_wtd.tif>
"""
import os, sys, glob, shutil
import numpy as np, rasterio

dom, out, eqwtd = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)

# 1. mirror all inputs as symlinks
for f in glob.glob(os.path.join(dom, "Esquibel_*.tif")):
    dst = os.path.join(out, os.path.basename(f))
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(f), dst)

# 2. perturbed precip = 1.1P - 0.1E (a real file, replacing the symlink)
P = os.path.join(dom, "Esquibel_010000_precipitation.tif")
E = os.path.join(dom, "Esquibel_010000_evaporation.tif")
with rasterio.open(P) as rp, rasterio.open(E) as re_:
    p = rp.read(1).astype("float64"); e = re_.read(1).astype("float64"); prof = rp.profile
newp = 1.1 * p - 0.1 * e
dstP = os.path.join(out, "Esquibel_010000_precipitation.tif")
if os.path.lexists(dstP):
    os.remove(dstP)
with rasterio.open(dstP, "w", **prof) as d:
    d.write(newp.astype(prof["dtype"]), 1)
print(f"perturbed precip written (1.1P-0.1E); mean {p.mean():.4g} -> {newp.mean():.4g}")

# 3. warm-start water table = the equilibrium output (Esquibel_010000_wtd.tif)
dstW = os.path.join(out, "Esquibel_010000_wtd.tif")
if os.path.exists(eqwtd):
    if os.path.lexists(dstW):
        os.remove(dstW)
    shutil.copy(eqwtd, dstW)
    print(f"warm-start wtd <- {os.path.basename(eqwtd)}")
else:
    print(f"WARNING: eq wtd '{eqwtd}' not found -> transient warm-start unavailable")
print(f"perturbed domain ready: {out}")
