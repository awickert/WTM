#!/usr/bin/env python3
"""Stage the full Esquibel domain (384,703 cells) for the reproducible speed test.

Esquibel is the full domain -- no crop -- so this copies the ten input rasters from
the source dataset into ./domain/ for portability and provenance. domain/ is 16 MB and
gitignored; this script regenerates it. The committed configs/runner/README are the
durable part; the rasters are Andy's source data, referenced here by path.

Usage: make_esquibel.py [source_dir] [out_dir]
  source_dir defaults to the known download location; out_dir defaults to ./domain
"""
import os, sys, glob, shutil

DEFAULT_SRC = "/home/awickert/Downloads/Esquibel_Data-20260801T205621Z-1-001/Esquibel_Data"
src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "domain")
os.makedirs(out, exist_ok=True)

n = 0
for f in sorted(glob.glob(os.path.join(src, "Esquibel_*.tif"))):
    base = os.path.basename(f)
    if base[:-4][-9:].isdigit():  # skip timestamped model outputs (…_000000000.tif)
        continue
    shutil.copy2(f, os.path.join(out, base))
    print("staged", base)
    n += 1
if n == 0:
    sys.exit(f"no Esquibel input rasters found in {src} -- pass the correct source_dir")
print(f"{n} rasters staged to {out}")
