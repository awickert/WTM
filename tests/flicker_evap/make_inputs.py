#!/usr/bin/env python3
"""Fixture for the EVAPORATION-DISCONTINUITY flicker (free-surface flicker mechanism 2).

See benchmark/FREE_SURFACE_FLICKER.md. A low plateau ringed by ocean with strong net supply below the
surface (P - ET > 0, the table is driven UP to the surface) but a net DEFICIT above it (P - owe < 0, open
water evaporates faster than it is supplied). The two sides of wtd = 0 therefore push in OPPOSITE
directions: below the surface the cell fills toward the surface; above it, open-water evaporation drains it
back down. Under the LEGACY hard evap_mode-1 switch the recharge forcing jumps discontinuously at wtd = 0
(P - owe vs max(0, P - ET); WTM.cpp), so the per-cycle iteration overshoots the surface each cycle and
settles into a period-2 LIMIT CYCLE. The smooth evaporation taper (taper 2, -wtm_evap_taper, default on)
replaces that jump with a single C1 transition, so the same fixture SETTLES at the surface instead.

To exercise the discontinuity with FSM off, above-surface water must be allowed to persist so the owe branch
can fire -- hence the diagnostic runs with -wtm_dev_allow_aboveground_water_columns (the surface clamp off).
The taper is then the ONLY thing that can settle the surface crossing, which is exactly what we test.

Regenerate with: python3 make_inputs.py
"""
import numpy as np, os, rasterio
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wtm_testgrid import make_transform  # noqa: E402
# Intended grid: WTM derives geometry from the geotransform (#124); encode the test's cpd/south.
CELLS_PER_DEGREE, SOUTHERN_EDGE = 120, 0
NX = NY = 12
REGION = "flickevap"
OUT = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUT, exist_ok=True)
tr = make_transform(CELLS_PER_DEGREE, SOUTHERN_EDGE, NY)
def w(name, data, dt="float32"):
    with rasterio.open(os.path.join(OUT, name), "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dt, crs="EPSG:4326", transform=tr) as d:
        d.write(data.astype(dt), 1)
P, ET, OWE = 2.0, 0.1, 8.0        # m/yr: ET < P < owe -> below-surface fills, above-surface drains
topo = np.full((NY, NX), 10.0, np.float32)
mask = np.ones((NY, NX), np.float32); mask[0,:]=mask[-1,:]=mask[:,0]=mask[:,-1]=0  # ocean ring (drainage)
topo[mask == 0] = 0.0
zero = np.zeros((NY, NX), np.float32)
for lay, a in {"topography":topo, "slope":zero, "mask":mask,
               "precipitation":np.full((NY,NX),P,np.float32),
               "evaporation":np.full((NY,NX),ET,np.float32),
               "open_water_evaporation":np.full((NY,NX),OWE,np.float32),
               "winter_temperature":zero}.items():
    w(f"{REGION}_ta_{lay}.tif", a); w(f"{REGION}_tb_{lay}.tif", a)
w(f"{REGION}_horizontal_ksat.tif", np.full((NY, NX), 1e-7, np.float32))  # low ksat: weak drainage -> table pins at surface
w(f"{REGION}_porosity.tif", np.full((NY, NX), 0.25, np.float32))
w(f"{REGION}_ta_starting_wt.tif", np.full((NY, NX), -1.0, np.float64), "float64")
print("wrote", OUT, "region", REGION, "P", P, "ET", ET, "owe", OWE)
