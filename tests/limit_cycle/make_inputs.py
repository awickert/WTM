#!/usr/bin/env python3
"""Fixture for the LIMIT-CYCLE (lakeshore-flicker) diagnostic test.

A high plateau ringed by ocean, water table starting below the surface with strong recharge, so the
interior mound rises ACROSS the land surface and sits at the free boundary (wtd=0). There, backward Euler
+ Anderson OVERSHOOT the surface each step (storativity jump porosity->~1 and the seepage/removal kink),
producing a period-2 LIMIT CYCLE -- the water table bounces above/below the surface and never settles.

This is a deliberately BAD (unmanaged) configuration: run bare, it flickers, and different time-integration
schemes land on different flickering states -> multiple methods act as an OVERSHOOT DIAGNOSTIC. Add the
post-solve clamp (-wtm_surface_exfiltration_to_runoff) and the cycle is suppressed and the schemes agree.
The test asserts BOTH: the flicker/disagreement bare, and the suppression/agreement with the clamp.

Regenerate with: python3 make_inputs.py
"""
import numpy as np, os, rasterio
from rasterio.transform import from_bounds
NX = NY = 12
REGION = "limitcyc"
OUT = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUT, exist_ok=True)
tr = from_bounds(0, 0, NX, NY, NX, NY)
def w(name, data, dt="float32"):
    with rasterio.open(os.path.join(OUT, name), "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dt, crs="EPSG:4326", transform=tr) as d:
        d.write(data.astype(dt), 1)
topo = np.full((NY, NX), 100.0, np.float32)
mask = np.ones((NY, NX), np.float32); mask[0,:]=mask[-1,:]=mask[:,0]=mask[:,-1]=0  # ocean ring (drainage)
topo[mask == 0] = 0.0
zero = np.zeros((NY, NX), np.float32)
for lay, a in {"topography":topo,"slope":zero,"mask":mask,"precipitation":np.full((NY,NX),1.5,np.float32),
               "evaporation":zero,"open_water_evaporation":zero,"winter_temperature":zero}.items():
    w(f"{REGION}_ta_{lay}.tif", a); w(f"{REGION}_tb_{lay}.tif", a)
w(f"{REGION}_horizontal_ksat.tif", np.full((NY, NX), 1e-4, np.float32))
w(f"{REGION}_porosity.tif", np.full((NY, NX), 0.25, np.float32))
w(f"{REGION}_ta_wtd.tif", np.full((NY, NX), -2.0, np.float64), "float64")
print("wrote", OUT, "region", REGION)
