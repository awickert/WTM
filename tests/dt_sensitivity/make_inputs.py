#!/usr/bin/env python3
"""Fixture for the dt-SENSITIVITY test.

An ocean-ringed plateau with moderate recharge and weak drainage, so the interior water table equilibrates
just BELOW the land surface -- inside the depth range where the legacy taper-1 sub-surface sink acts. Because
that sink's band width scales with the timestep (2*qmax*dt), a table sitting in the band equilibrates at a
dt-DEPENDENT depth: the whole reason the exact in-residual seepage face (runoff_collector=implicit, the
default) replaced it. This fixture makes that contrast sharp: implicit is dt-independent to ~machine
precision, the legacy band sink is dt-dependent by ~0.8 m across a 4x dt change.

Regenerate with: python3 make_inputs.py
"""
import numpy as np, os, rasterio
from rasterio.transform import from_bounds
NX = NY = 12
REGION = "dtsens"
OUT = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUT, exist_ok=True)
tr = from_bounds(0, 0, NX, NY, NX, NY)
def w(name, data, dt="float32"):
    with rasterio.open(os.path.join(OUT, name), "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dt, crs="EPSG:4326", transform=tr) as d:
        d.write(data.astype(dt), 1)
topo = np.full((NY, NX), 10.0, np.float32)
mask = np.ones((NY, NX), np.float32); mask[0,:]=mask[-1,:]=mask[:,0]=mask[:,-1]=0  # ocean ring (drainage)
topo[mask == 0] = 0.0
zero = np.zeros((NY, NX), np.float32)
for lay, a in {"topography":topo, "slope":zero, "mask":mask,
               "precipitation":np.full((NY,NX),0.3,np.float32),  # moderate recharge -> table just below surface
               "evaporation":zero, "open_water_evaporation":zero, "winter_temperature":zero}.items():
    w(f"{REGION}_ta_{lay}.tif", a); w(f"{REGION}_tb_{lay}.tif", a)
w(f"{REGION}_horizontal_ksat.tif", np.full((NY, NX), 1e-5, np.float32))  # weak drainage -> table sits in the sink band
w(f"{REGION}_porosity.tif", np.full((NY, NX), 0.25, np.float32))
w(f"{REGION}_ta_starting_wt.tif", np.full((NY, NX), -1.0, np.float64), "float64")
print("wrote", OUT, "region", REGION)
