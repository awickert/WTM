#!/usr/bin/env python3
"""Fixture for the adaptive-dt / water-depth-metric regression test.

A small coastal wedge (high plateau draining to an ocean strip on the west) that reaches a smooth
cold-start steady state. Every time-integration scheme + every equilibrium-stop metric must converge to the
SAME steady water table, so it is a clean cross-check that:
  - the adaptive-dt controller (`-wtm_tr_bdf2 -wtm_dt_adaptive`) reaches the correct equilibrium, and
  - the pure-water-depth stop metric (`-wtm_eq_metric water-rms`, |S*Δwtd|) reaches the same equilibrium.
Regenerate with:  python3 make_inputs.py
"""
import numpy as np, os, rasterio
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wtm_testgrid import make_transform  # noqa: E402

NX, NY = 12, 8
REGION = "adwater"
OUT = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUT, exist_ok=True)
# CELL SIZE IS PHYSICS, NOT BOOKKEEPING (#34). This was `from_bounds(0, 0, NX, NY, NX, NY)`, the
# arbitrary placeholder from before WTM derived cell geometry from the geotransform (#124) -- it means
# ONE DEGREE per cell, ~111 km. At that size, lateral drainage to the ocean strip is negligible against
# the recharge, so this plateau simply filled up and every compared field went identically constant.
# tests/nonvacuous.py caught it: 3 of 3 final fields identically 0, while the suite reported
# "max|dV| = 0.0000 m water" twice and PASSED.
#
# 64 cells/degree = ~1.74 km, and it is NOT the ~111 m that the other converted fixtures use. This one
# has a window, measured by sweeping the cc arm:
#     cpd     4  -> constant field (still vacuous)
#     cpd    16  -> 17 distinct values, wtd -41.18 .. 0.00
#     cpd    64  -> 81 distinct values, wtd -81.78 .. 0.00     <- chosen
#     cpd   256  -> 88 distinct values, wtd -98.71 .. -90.84   (whole plateau drained to depth)
#     cpd  1000  -> DIVERGED_MAX_IT at 10000 nonlinear iterations
# 64 is the deepest structure that still spans surface to depth: the water table meets the ocean strip
# at 0 and reaches -81.8 m in the interior, so the three arms must agree across the whole range this
# suite's stop metrics care about. 256 pushes every cell into the exponential-T dead zone and 1000 does
# not converge at all, so "as fine as possible" is the wrong instinct here.
CELLS_PER_DEGREE, SOUTHERN_EDGE = 64, 0
tr = make_transform(CELLS_PER_DEGREE, SOUTHERN_EDGE, NY)

def w(name, data, dt="float32"):
    with rasterio.open(os.path.join(OUT, name), "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dt, crs="EPSG:4326", transform=tr) as d:
        d.write(data.astype(dt), 1)

topo = np.full((NY, NX), 100.0, np.float32)  # high plateau
topo[:, 0] = 0.0                             # ocean strip (west) at sea level -> drainage gradient
mask = np.ones((NY, NX), np.float32); mask[:, 0] = 0.0
zero = np.zeros((NY, NX), np.float32)
# Moderate recharge so the cold (wtd=0) table rises to a subsurface steady state (below the 100 m plateau),
# i.e. a clean interior equilibrium, not a surface-crossing flicker case.
for lay, a in {"topography":topo,"slope":zero,"mask":mask,"precipitation":np.full((NY,NX),0.3,np.float32),
               "evaporation":zero,"open_water_evaporation":zero,"winter_temperature":zero}.items():
    w(f"{REGION}_ta_{lay}.tif", a); w(f"{REGION}_tb_{lay}.tif", a)
w(f"{REGION}_horizontal_ksat.tif", np.full((NY, NX), 1e-4, np.float32))
w(f"{REGION}_porosity.tif", np.full((NY, NX), 0.25, np.float32))
w(f"{REGION}_ta_wtd.tif", np.zeros((NY, NX), np.float64), "float64")  # cold start (supplied_wt 0 ignores it)
print("wrote", OUT, "region", REGION)
