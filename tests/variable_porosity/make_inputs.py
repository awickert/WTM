#!/usr/bin/env python3
"""Fixture with SPATIALLY VARYING POROSITY -- the one thing no other fixture has.

WHY IT EXISTS. Every other fixture in this suite is uniform phi = 0.25. On a uniform-porosity,
purely subsurface domain, V(wtd) has slope exactly phi everywhere, so converting a comparison from
metres of HEAD to metres of WATER VOLUME is an exact x0.25 rescale: the cell RANKING is unchanged and
nothing that passes can start failing. The distinction the whole #61/#65 arc is about is therefore
INVISIBLE to the suite by construction, and a bug that bites only where porosity varies -- exactly
the class that motivated judging convergence in water volume -- could not be caught here at all.
Production data has spatially variable porosity, so a tolerance calibrated on phi = 0.25 does not
transfer either.

THE DESIGN IS DELIBERATE, not just "make phi a gradient". For head and volume norms to disagree
about WHICH CELL DOMINATES, the largest head differences must fall where the porosity is LOW (a big
head swing moving little water) and vice versa. So phi runs 0.05 in the EAST, far from the ocean
strip where the Dupuit mound is highest and scheme-to-scheme head differences are largest, up to 0.40
in the WEST near the drain. A head norm then weights the east; a volume norm weights the west.

Otherwise this mirrors solver_consistency's fixture: a gentle, purely SUBSURFACE equilibrium, so
Picard and Newton stay valid (they diverge at a pinned free surface, issue #97) and several schemes
can be compared on it. Purely subsurface also means dV/dwtd is exactly phi, so any head-vs-volume
disagreement here is caused by POROSITY alone and not by the surface blend -- which is what makes the
demonstration clean.

Regenerate with:  python3 make_inputs.py
"""
import numpy as np, os, rasterio
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wtm_testgrid import make_transform  # noqa: E402
CELLS_PER_DEGREE, SOUTHERN_EDGE = 100, 0

NX, NY = 12, 8
REGION = "varphi"
PHI_WEST, PHI_EAST = 0.40, 0.05        # the range real data covers; Andy's call
OUT = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUT, exist_ok=True)
tr = make_transform(CELLS_PER_DEGREE, SOUTHERN_EDGE, NY)

def w(name, data, dt="float32"):
    with rasterio.open(os.path.join(OUT, name), "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dt, crs="EPSG:4326", transform=tr) as d:
        d.write(data.astype(dt), 1)

topo = np.full((NY, NX), 100.0, np.float32)   # high plateau
topo[:, 0] = 0.0                              # ocean strip (west) -> lateral drainage gradient
mask = np.ones((NY, NX), np.float32); mask[:, 0] = 0.0
zero = np.zeros((NY, NX), np.float32)

# phi: high beside the drain, low far from it. Column 0 is ocean and its value is inert.
phi = np.tile(np.linspace(PHI_WEST, PHI_EAST, NX, dtype=np.float32), (NY, 1))

for lay, a in {"topography":topo,"slope":zero,"mask":mask,"precipitation":np.full((NY,NX),0.3,np.float32),
               "evaporation":zero,"open_water_evaporation":zero,"winter_temperature":zero}.items():
    w(f"{REGION}_ta_{lay}.tif", a); w(f"{REGION}_tb_{lay}.tif", a)
w(f"{REGION}_horizontal_ksat.tif", np.full((NY, NX), 1e-3, np.float32))
w(f"{REGION}_porosity.tif", phi)
w(f"{REGION}_ta_wtd.tif", np.zeros((NY, NX), np.float64), "float64")
print("wrote", OUT, "region", REGION, "porosity %.2f..%.2f" % (phi.min(), phi.max()))
