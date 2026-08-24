#!/usr/bin/env python3
"""Fixture for the adaptive-restart robustness regression test.

A gentle, purely-subsurface coastal-wedge equilibrium (small physical domain via cells_per_degree 100 -> ~1 km
cells, low recharge over fast ksat -> a ~6 m Dupuit mound below the 100 m plateau). This is the regime where
the ρ-triggered adaptive-restart controller (-wtm_adaptive_restart) used to ABORT: near equilibrium the
Anderson step floors just above the relative step tolerance, so the controller never formally declares true
convergence, exhausts its restart budget, and (before the robust-finish fix) threw "SNES has not converged"
instead of returning the tracked best iterate. Here the controller must instead settle to the SAME water table
as a plain Anderson solve. Regenerate with:  python3 make_inputs.py
"""
import numpy as np, os, rasterio
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wtm_testgrid import make_transform  # noqa: E402
# Intended grid: WTM derives geometry from the geotransform (#124); encode the test's cpd/south.
CELLS_PER_DEGREE, SOUTHERN_EDGE = 100, 0

NX, NY = 12, 8
REGION = "arestart"
OUT = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUT, exist_ok=True)
tr = make_transform(CELLS_PER_DEGREE, SOUTHERN_EDGE, NY)

def w(name, data, dt="float32"):
    with rasterio.open(os.path.join(OUT, name), "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dt, crs="EPSG:4326", transform=tr) as d:
        d.write(data.astype(dt), 1)

topo = np.full((NY, NX), 100.0, np.float32)  # high plateau
topo[:, 0] = 0.0                             # ocean strip (west) at sea level -> lateral drainage gradient
mask = np.ones((NY, NX), np.float32); mask[:, 0] = 0.0
zero = np.zeros((NY, NX), np.float32)
for lay, a in {"topography":topo,"slope":zero,"mask":mask,"precipitation":np.full((NY,NX),0.3,np.float32),
               "evaporation":zero,"open_water_evaporation":zero,"winter_temperature":zero}.items():
    w(f"{REGION}_ta_{lay}.tif", a); w(f"{REGION}_tb_{lay}.tif", a)
w(f"{REGION}_horizontal_ksat.tif", np.full((NY, NX), 1e-3, np.float32))  # fast -> low subsurface mound
w(f"{REGION}_porosity.tif", np.full((NY, NX), 0.25, np.float32))
w(f"{REGION}_ta_wtd.tif", np.zeros((NY, NX), np.float64), "float64")     # cold start
print("wrote", OUT, "region", REGION)
