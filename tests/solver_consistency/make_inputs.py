#!/usr/bin/env python3
"""Fixture for the solver-consistency (differential-oracle) regression test.

The matrix-free Anderson solver has no independent Jacobian to check it against, so this fixture lets the
two matrix-based solvers -- Picard (frozen-coefficient) and Newton (analytic Jacobian) -- serve as
independent oracles: on a domain where all three are valid they must converge to the SAME water table.

The regime matters. Picard and Newton DIVERGE at a pinned free surface (recharge exceeding lateral drainage
so the table wants to sit above the land -> BE-Picard secant-storativity collapse, Newton line-search
failure; see finding_picard_anderson_fixedpoint_gap.md / issue #97). So this fixture is deliberately a
GENTLE, purely SUBSURFACE equilibrium: a small physical domain (cells_per_degree 100 -> ~1 km cells) with
low recharge over fast ksat, so the steady Dupuit mound rises only a few metres above sea level and stays
well below the 100 m plateau everywhere. There all three solvers land on the same interior fixed point.

Regenerate with:  python3 make_inputs.py
"""
import numpy as np, os, rasterio
from rasterio.transform import from_bounds

NX, NY = 12, 8
REGION = "sconsist"
OUT = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUT, exist_ok=True)
tr = from_bounds(0, 0, NX, NY, NX, NY)

def w(name, data, dt="float32"):
    with rasterio.open(os.path.join(OUT, name), "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dt, crs="EPSG:4326", transform=tr) as d:
        d.write(data.astype(dt), 1)

topo = np.full((NY, NX), 100.0, np.float32)  # high plateau
topo[:, 0] = 0.0                             # ocean strip (west) at sea level -> lateral drainage gradient
mask = np.ones((NY, NX), np.float32); mask[:, 0] = 0.0
zero = np.zeros((NY, NX), np.float32)
# Recharge 0.3 m/yr over fast ksat (1e-3 m/s) on ~1 km cells: the steady water table forms a Dupuit mound
# only ~6 m above sea level at the interior -- far below the 100 m surface, so it never pins at the free
# surface. This is the gentle interior-equilibrium regime where Picard and Newton both remain valid.
for lay, a in {"topography":topo,"slope":zero,"mask":mask,"precipitation":np.full((NY,NX),0.3,np.float32),
               "evaporation":zero,"open_water_evaporation":zero,"winter_temperature":zero}.items():
    w(f"{REGION}_ta_{lay}.tif", a); w(f"{REGION}_tb_{lay}.tif", a)
w(f"{REGION}_horizontal_ksat.tif", np.full((NY, NX), 1e-3, np.float32))  # fast -> low mound, subsurface
w(f"{REGION}_porosity.tif", np.full((NY, NX), 0.25, np.float32))
w(f"{REGION}_ta_wtd.tif", np.zeros((NY, NX), np.float64), "float64")     # cold start (supplied_wt 0)
print("wrote", OUT, "region", REGION)
