#!/usr/bin/env python3
"""Tile the real Esquibel input stack into a larger domain for FSM/DH cost scaling (#78).

WTM takes cells_per_degree + southern_edge from the CFG and the grid DIMENSIONS from the
raster bands (irf.cpp: ncells_x = topo.width()); latitude of row j is j/cells_per_degree +
southern_edge (run_dephier.cpp). So replicating each raster ty x tx times (numpy.tile) grows
the grid into valid latitudes/longitudes with REAL topography and depressions in every tile --
a legitimate compute-cost domain (caveat: repeated topo + tile seams; not a science domain).

Usage: make_tiled.py <ty> <tx> [src_domain] [out_root]
  ty, tx      vertical / horizontal tile counts (grid becomes ty*451 x tx*853)
  src_domain  default ./domain  (the 384,703-cell Esquibel stack)
  out_root    default ./tiled   (writes <out_root>/<ny>x<nx>/ + a ready cfg)

Emits an eq_tiled.cfg pointing at the new domain with total_cycles 6 (a short cost pass).
"""
import os, sys, glob
import numpy as np
import rasterio

ty = int(sys.argv[1]); tx = int(sys.argv[2])
src = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "domain")
out_root = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(__file__), "tiled")

rasters = sorted(glob.glob(os.path.join(src, "Esquibel_*.tif")))
if not rasters:
    sys.exit(f"no Esquibel_*.tif in {src}")

# probe dimensions from topography
with rasterio.open(os.path.join(src, "Esquibel_010000_topography.tif")) as d0:
    h0, w0 = d0.height, d0.width
ny, nx = ty * h0, tx * w0
outdir = os.path.join(out_root, f"{ny}x{nx}")
os.makedirs(outdir, exist_ok=True)
print(f"tiling {ty}x{tx}: {h0}x{w0} -> {ny}x{nx} = {ny*nx:,} cells  ->  {outdir}")

for f in rasters:
    base = os.path.basename(f)
    if base[:-4][-9:].isdigit():  # skip timestamped model outputs
        continue
    with rasterio.open(f) as d:
        arr = d.read(1)
        prof = d.profile.copy()
        big = np.tile(arr, (ty, tx))
        # keep the same origin + pixel size; just enlarge the raster window
        t = d.transform
        prof.update(height=ny, width=nx, transform=t)
        with rasterio.open(os.path.join(outdir, base), "w", **prof) as o:
            o.write(big, 1)
    print("  tiled", base)

# ready-to-run cfg: same physics as eq_awickert.cfg, pointed at the tiled domain, short run
cfg = os.path.join(outdir, "eq_tiled.cfg")
with open(cfg, "w") as c:
    c.write(f"""run_type           equilibrium
fsm_on             1
evap_mode          1
infiltration_on    0
runoff_ratio_on    1
cells_per_degree   900
southern_edge      55.338391020555555
deltat             604800
total_cycles       6
cycles_to_save     6
maxiter            50
fdepth_a           100
fdepth_b           150
fdepth_fmin        2.5
time_start         010000
time_end           010000
surfdatadir        {outdir}/
region             Esquibel
supplied_wt        0
textfilename       {outdir}/eq_tiled.txt
outfile_prefix     {outdir}/eq_tiled_
""")
print("wrote", cfg)
