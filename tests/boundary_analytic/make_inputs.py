#!/usr/bin/env python3
"""Fixtures for the ANALYTICAL boundary tests (closed-form, not snapshot goldens).

Steady groundwater with CONSTANT transmissivity T and uniform recharge R obeys  T h'' = -R  -> the head is a
parabola. WTM's T is exponential in general, but for wtd > 0 it is clamped to the constant surface value
T = ksat*(1.5 + fdepth) (see depthIntegratedTransmissivity). So on a FLAT sea-level domain (topo = 0) with the
water table mounded ABOVE the surface by recharge (ponding allowed, all removals off), the whole domain sits in
the constant-T regime and the exact steady solution is a parabola that the discrete scheme reproduces to solver
tolerance. Two fixtures, both flat and uniform, thin & uniform in y (the land y-edges are no-flow, so the
problem is 1-D in x):

  anbcD : ocean (Dirichlet h=0) at BOTH x-ends            -> symmetric parabola  h(x) = A x (L-x)   (h=0 at both)
  anbcN : ocean (Dirichlet h=0) at the LEFT, LAND at the  -> half-parabola,  zero gradient (vertex) exactly at
          right x-end (no-flow / neumann_toposlope)          the no-flow face -- the Neumann BC, analytically.

Regenerate with:  python3 make_inputs.py
"""
import numpy as np, os, rasterio
from rasterio.transform import from_bounds

NX, NY = 22, 3
OUT = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUT, exist_ok=True)

def write(region, mask):
    tr = from_bounds(0, 0, NX, NY, NX, NY)
    def wr(name, a, dt="float32"):
        with rasterio.open(os.path.join(OUT, name), "w", driver="GTiff", height=NY, width=NX, count=1,
                           dtype=dt, crs="EPSG:4326", transform=tr) as d:
            d.write(a.astype(dt), 1)
    zero = np.zeros((NY, NX), np.float32)
    topo = np.zeros((NY, NX), np.float32)                     # flat, at sea level
    for lay, a in {"topography":topo,"slope":zero,"mask":mask,"precipitation":np.full((NY,NX),0.05,np.float32),
                   "evaporation":zero,"open_water_evaporation":zero,"winter_temperature":zero}.items():
        wr(f"{region}_ta_{lay}.tif", a); wr(f"{region}_tb_{lay}.tif", a)
    wr(f"{region}_horizontal_ksat.tif", np.full((NY, NX), 1e-4, np.float32))
    wr(f"{region}_porosity.tif", np.full((NY, NX), 0.25, np.float32))
    wr(f"{region}_ta_wtd.tif", np.zeros((NY, NX), np.float64), "float64")

# Dirichlet: ocean at both x-ends (h=0). Neumann: ocean at the left only; the right x-end stays land (no-flow).
mD = np.ones((NY, NX), np.float32); mD[:, 0] = 0; mD[:, -1] = 0
mN = np.ones((NY, NX), np.float32); mN[:, 0] = 0
write("anbcD", mD)
write("anbcN", mN)
print("wrote", OUT, "-- anbcD (ocean both ends), anbcN (ocean-left, land no-flow right)")
