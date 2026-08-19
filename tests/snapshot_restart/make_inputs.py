#!/usr/bin/env python3
"""Fixture for the snapshot-filename + restart-from-snapshot regression test.

A small coastal wedge with slow drainage (low ksat) so a cold-start equilibrium takes ~10 cycles -- enough
that a mid-run snapshot is clearly pre-equilibrium, so a warm restart from it must (a) reach the SAME
equilibrium and (b) get there in FEWER cycles than a cold start (proving supplied_wt actually loaded the
snapshot rather than silently restarting cold). Regenerate with:  python3 make_inputs.py
"""
import numpy as np, os, rasterio
from rasterio.transform import from_bounds

NX, NY = 12, 8
REGION = "snaptest"
OUT = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUT, exist_ok=True)
tr = from_bounds(0, 0, NX, NY, NX, NY)

def w(name, data, dt="float32"):
    with rasterio.open(os.path.join(OUT, name), "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dt, crs="EPSG:4326", transform=tr) as d:
        d.write(data.astype(dt), 1)

topo = np.full((NY, NX), 100.0, np.float32)  # high plateau
topo[:, 0] = 0.0                             # ocean strip (west) -> drainage gradient
mask = np.ones((NY, NX), np.float32); mask[:, 0] = 0.0
zero = np.zeros((NY, NX), np.float32)
for lay, a in {"topography":topo,"slope":zero,"mask":mask,"precipitation":np.full((NY,NX),0.05,np.float32),
               "evaporation":zero,"open_water_evaporation":zero,"winter_temperature":zero}.items():
    w(f"{REGION}_ta_{lay}.tif", a); w(f"{REGION}_tb_{lay}.tif", a)
w(f"{REGION}_horizontal_ksat.tif", np.full((NY, NX), 1e-6, np.float32))  # slow -> many cycles to equilibrate
w(f"{REGION}_porosity.tif", np.full((NY, NX), 0.25, np.float32))
w(f"{REGION}_ta_wtd.tif", np.zeros((NY, NX), np.float64), "float64")
print("wrote", OUT, "region", REGION)
