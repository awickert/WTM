#!/usr/bin/env python3
"""Fixture for the DIRECT-TO-RUNOFF gathering test (free-surface flicker, the routing-success view).

See benchmark/FREE_SURFACE_FLICKER.md. The free-surface flicker is not a numerical problem when the
above-surface water has somewhere to GO: routed into the runoff array, the water table is held at the surface
(wtd = 0) instead of piling up and sloshing. This fixture demonstrates that success for the IN-RESIDUAL
exfiltration route (-wtm_direct_to_runoff): a low ocean-ringed plateau (with a slight interior low as a
concentrator) under strong recharge and weak drainage, so the interior is driven to the surface and must shed
its excess. With direct-to-runoff the excess is gathered to the runoff array and the table sits exactly at
wtd = 0 (the exfiltration complementarity: removal active <=> table pinned at the surface); the water budget
closes (recharge = surface_removed + ocean_outflow). Without any gathering (ponding allowed) the same water
piles hundreds of metres above the surface -- the failure the routing prevents.

Regenerate with: python3 make_inputs.py
"""
import numpy as np, os, rasterio
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wtm_testgrid import make_transform  # noqa: E402
# Intended grid: WTM derives geometry from the geotransform (#124); encode the test's cpd/south.
CELLS_PER_DEGREE, SOUTHERN_EDGE = 120, 0
NX = NY = 12
REGION = "runoffgather"
OUT = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUT, exist_ok=True)
tr = make_transform(CELLS_PER_DEGREE, SOUTHERN_EDGE, NY)
def w(name, data, dt="float32"):
    with rasterio.open(os.path.join(OUT, name), "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dt, crs="EPSG:4326", transform=tr) as d:
        d.write(data.astype(dt), 1)
topo = np.full((NY, NX), 20.0, np.float32)
topo[6, 6] = 18.0                                       # a slight interior low: a concentrator cell
mask = np.ones((NY, NX), np.float32); mask[0,:]=mask[-1,:]=mask[:,0]=mask[:,-1]=0  # ocean ring (drainage)
topo[mask == 0] = 0.0
zero = np.zeros((NY, NX), np.float32)
for lay, a in {"topography":topo, "slope":zero, "mask":mask,
               "precipitation":np.full((NY,NX),3.0,np.float32),  # strong recharge -> interior driven to surface
               "evaporation":zero, "open_water_evaporation":zero, "winter_temperature":zero}.items():
    w(f"{REGION}_ta_{lay}.tif", a); w(f"{REGION}_tb_{lay}.tif", a)
w(f"{REGION}_horizontal_ksat.tif", np.full((NY, NX), 1e-6, np.float32))  # weak drainage -> table pins at surface
w(f"{REGION}_porosity.tif", np.full((NY, NX), 0.25, np.float32))
w(f"{REGION}_ta_starting_wt.tif", np.full((NY, NX), -1.0, np.float64), "float64")
print("wrote", OUT, "region", REGION)
