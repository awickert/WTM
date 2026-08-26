#!/usr/bin/env python3
"""Fixtures for the LOCAL-IN-SPACE water ledger (tests/local_ledger).

Two regions, each isolating one half of "water went to the wrong PLACE" -- the failure mode every
existing budget test is blind to, because a global sum is unchanged when water is moved from one cell
to another. tests/fsm_conservation says so in its own header: "conservation catches water
CREATED/DESTROYED but not MISPLACED."

ledgerA -- PLACEMENT. Lateral conductivity is essentially zero, so each cell is an isolated column and
    the water table must rise by exactly the local forcing. Precipitation varies differently along x
    and along y, on purpose: a pattern symmetric under transpose would let an axis swap pass. WTM has
    had two real bugs of exactly this shape (the lat/lon swap, and the E-W/N-S cell-size swap).

ledgerB -- REDISTRIBUTION. A closed domain (all land, no-flow edges) with ZERO forcing and a mound in
    the starting water table. Nothing enters or leaves, so the total stored volume must be constant
    while the table visibly moves. This is a sharp probe of the lateral flux operator: any face whose
    flux is not exactly antisymmetric between the two cells sharing it creates or destroys water.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wtm_testgrid import write_tif as _write_tif  # noqa: E402

NY, NX = 24, 24
CELLS_PER_DEGREE = 10
SOUTHERN_EDGE = -45
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs")
os.makedirs(OUT, exist_ok=True)


def write_tif(path, data, dtype="float32"):
    _write_tif(os.path.join(OUT, path), data, CELLS_PER_DEGREE, SOUTHERN_EDGE, dtype=dtype)


yy, xx = np.mgrid[0:NY, 0:NX]

# ---- shared: flat topography, all land (closed domain), no slope/temperature structure ----
topo = np.full((NY, NX), 100.0, dtype=np.float32)
mask = np.ones((NY, NX), dtype=np.float32)          # no ocean: nothing can leave through a boundary
slope = np.zeros((NY, NX), dtype=np.float32)
winter_temp = np.zeros((NY, NX), dtype=np.float32)
porosity = np.full((NY, NX), 0.25, dtype=np.float32)
zero = np.zeros((NY, NX), dtype=np.float32)

# ---- ledgerA: placement ----
# Precipitation varies as a*x + b*y with a != b, so the field is NOT symmetric under transpose and a
# swapped axis cannot pass. Kept strictly positive so every cell is forced.
precip_a = (0.05 + 0.0300 * xx + 0.0070 * yy).astype(np.float32)
# Essentially no lateral conduction, so each column stands alone and the local balance is exact.
ksat_a = np.full((NY, NX), 1e-12, dtype=np.float32)
# Start well below the surface: storedVolume(wtd) = wtd*porosity there (the smoothed |wtd| term is
# within 1e-8 of |wtd| by |wtd| ~ 1 m), so the expected rise is analytic and needs no model internals.
start_a = np.full((NY, NX), -20.0, dtype=np.float64)

# ---- ledgerB: redistribution ----
precip_b = zero                                       # nothing in
ksat_b = np.full((NY, NX), 1e-4, dtype=np.float32)    # ordinary conductivity: water really moves
r2 = ((xx - (NX - 1) / 2.0) ** 2 + (yy - (NY - 1) / 2.0) ** 2) / (2.0 * 5.0 ** 2)
start_b = (-20.0 + 12.0 * np.exp(-r2)).astype(np.float64)  # a mound, entirely below the surface

FILES = {}
for region, precip, ksat, start in (("ledgerA", precip_a, ksat_a, start_a),
                                    ("ledgerB", precip_b, ksat_b, start_b)):
    FILES.update({
        f"{region}_t0_topography.tif":             (topo, "float32"),
        f"{region}_t0_slope.tif":                  (slope, "float32"),
        f"{region}_t0_mask.tif":                   (mask, "float32"),
        f"{region}_t0_precipitation.tif":          (precip, "float32"),
        f"{region}_t0_evaporation.tif":            (zero, "float32"),
        f"{region}_t0_open_water_evaporation.tif": (zero, "float32"),
        f"{region}_t0_winter_temperature.tif":     (winter_temp, "float32"),
        f"{region}_horizontal_ksat.tif":           (ksat, "float32"),
        f"{region}_porosity.tif":                  (porosity, "float32"),
        f"{region}_t0_starting_wt.tif":            (start, "float64"),
    })

for path, (arr, dt) in FILES.items():
    write_tif(path, arr, dt)
print(f"wrote {len(FILES)} rasters to {OUT}")
