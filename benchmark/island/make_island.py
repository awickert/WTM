#!/usr/bin/env python3
"""Regenerate the island test domain by cropping the full Esquibel rasters.

The committed rasters in ./domain/ ARE the source of truth for the fixture; this
script documents their provenance and lets them be regenerated if lost.

The island is a 117x75 window (col_off=500, row_off=335) cut from the full
Esquibel domain (853x451, 4"/900-cells-per-degree). That window reproduces the
committed rasters byte-for-byte (verified). It is an ocean-ringed sub-island --
a small, cheap, but physically real cold-start-to-equilibrium test.

Usage: make_island.py <full_esquibel_dir> [out_dir]
  <full_esquibel_dir> holds the full-resolution Esquibel_*.tif rasters.
  [out_dir] defaults to ./domain
"""
import os, sys, glob, shutil
import rasterio
from rasterio.windows import Window

COL_OFF, ROW_OFF, WIDTH, HEIGHT = 500, 335, 117, 75  # the island window in the full grid

full = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "domain")
os.makedirs(out, exist_ok=True)
win = Window(COL_OFF, ROW_OFF, WIDTH, HEIGHT)

for src in sorted(glob.glob(os.path.join(full, "Esquibel_*.tif"))):
    # skip timestamped model outputs (…_000000000.tif); keep the static/forcing inputs
    base = os.path.basename(src)
    stem = base[:-4]
    if stem[-9:].isdigit():
        continue
    with rasterio.open(src) as r:
        data = r.read(window=win)
        prof = r.profile
        prof.update(width=WIDTH, height=HEIGHT, transform=r.window_transform(win))
        with rasterio.open(os.path.join(out, base), "w", **prof) as d:
            d.write(data)
    print(f"cropped {base}")

print(f"island domain written to {out}")
