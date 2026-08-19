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
from rasterio.transform import from_bounds

NX, NY = 12, 8
REGION = "adwater"
OUT = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUT, exist_ok=True)
tr = from_bounds(0, 0, NX, NY, NX, NY)

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
