#!/usr/bin/env python3
"""Minimal fixture for the RECHARGE / storativity cross-scheme consistency test.

A flat land plateau ringed by ocean (Dirichlet h=0). The interior water table starts just BELOW the
surface (wtd0 < 0); strong recharge drives the centre cells UP ACROSS the surface within a step, while
the ocean edges drain them -- so there is both a surface crossing and lateral flux, the regime where the
recharge/storativity treatment diverges between time-integration schemes. Every scheme integrates the
SAME parabolic problem, so all must converge to the SAME water table as dt -> 0. They do not today
(recharge is applied as a storativity-scaled head; BE uses secant, TR-BDF2/BDF2-on-V use tangent).
"""
import numpy as np, os, rasterio
from rasterio.transform import from_bounds

NX = NY = 16
REGION = "rech_test"
OUT = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUT, exist_ok=True)
tr = from_bounds(0, 0, NX, NY, NX, NY)

def w(name, data, dt="float32"):
    with rasterio.open(os.path.join(OUT, name), "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dt, crs="EPSG:4326", transform=tr) as d:
        d.write(data.astype(dt), 1)

topo = np.full((NY, NX), 100.0, np.float32)
mask = np.ones((NY, NX), np.float32); mask[0,:]=mask[-1,:]=mask[:,0]=mask[:,-1]=0
zero = np.zeros((NY, NX), np.float32)
precip = np.full((NY, NX), 1.5, np.float32)     # recharge tuned so the steady mound STRADDLES (m/yr) -> centre crosses the surface
ksat   = np.full((NY, NX), 1e-4, np.float32)
poro   = np.full((NY, NX), 0.25, np.float32)
wtd0   = np.full((NY, NX), -2.0, np.float64)     # start BELOW surface -> crosses up within a step

for lay, a in {"topography":topo,"slope":zero,"mask":mask,"precipitation":precip,
               "evaporation":zero,"open_water_evaporation":zero,"winter_temperature":zero}.items():
    w(f"{REGION}_ta_{lay}.tif", a); w(f"{REGION}_tb_{lay}.tif", a)  # steady in time
w(f"{REGION}_horizontal_ksat.tif", ksat)
w(f"{REGION}_porosity.tif", poro)
w(f"{REGION}_ta_wtd.tif", wtd0, "float64")
print("wrote", OUT)
