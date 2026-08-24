#!/usr/bin/env python3
"""Fixture for the runoff_collector selector test.

A low ocean-ringed plateau under strong recharge and weak drainage, so the interior is driven up to the
land surface and must shed its excess -- a PARTIAL seepage face (interior cells pinned at wtd=0, cells near
the ocean ring below). This is the regime that distinguishes the three surface-water routing modes:
  implicit (in-residual seepage)  -> table pinned ~0 (seepage face), water NOT piled,
  explicit (post-solve clamp)     -> table clamped to exactly 0,
  off      (no collection)        -> water piles far above the surface (nonphysical),
and the unset default (legacy band sink) holds the table just BELOW the surface.

Regenerate with: python3 make_inputs.py
"""
import numpy as np, os, rasterio
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wtm_testgrid import make_transform  # noqa: E402
# Intended grid: WTM derives geometry from the geotransform (#124); encode the test's cpd/south.
CELLS_PER_DEGREE, SOUTHERN_EDGE = 120, 0
NX = NY = 20
REGION = "rcoll"
OUT = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUT, exist_ok=True)
tr = make_transform(CELLS_PER_DEGREE, SOUTHERN_EDGE, NY)
def w(name, data, dt="float32"):
    with rasterio.open(os.path.join(OUT, name), "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dt, crs="EPSG:4326", transform=tr) as d:
        d.write(data.astype(dt), 1)
topo = np.full((NY, NX), 20.0, np.float32)
mask = np.ones((NY, NX), np.float32); mask[0,:]=mask[-1,:]=mask[:,0]=mask[:,-1]=0  # ocean ring (drainage)
topo[mask == 0] = 0.0
zero = np.zeros((NY, NX), np.float32)
for lay, a in {"topography":topo, "slope":zero, "mask":mask,
               "precipitation":np.full((NY,NX),0.5,np.float32),   # strong recharge -> interior reaches the surface
               "evaporation":zero, "open_water_evaporation":zero, "winter_temperature":zero}.items():
    w(f"{REGION}_ta_{lay}.tif", a); w(f"{REGION}_tb_{lay}.tif", a)
w(f"{REGION}_horizontal_ksat.tif", np.full((NY, NX), 1e-5, np.float32))  # weak drainage -> partial seepage face
w(f"{REGION}_porosity.tif", np.full((NY, NX), 0.25, np.float32))
w(f"{REGION}_ta_starting_wt.tif", np.full((NY, NX), -1.0, np.float64), "float64")
print("wrote", OUT, "region", REGION)
