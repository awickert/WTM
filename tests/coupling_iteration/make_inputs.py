#!/usr/bin/env python3
"""Single-pit fixture for the FSM<->recharge coupling iteration (#112).

ONE depression that fills and then spills off-map, chosen deliberately over a chain:

  rim 100 m, ocean ring on the edges
  pit    : floor 92 m, outlet sill 96 m -> spills OFF-MAP to the ocean

WHY ONE PIT. The subject here is the COUPLING, not the routing topology -- tests/fsm_cascade
already covers a spill chain. A single basin gives a clean two-regime run on one fixture: the
lake FILLS (transient, where the lagged coupling is the only place it can be wrong) and then
SITS at its sill (equilibrium, where the lag is identically zero by construction and iterating
must therefore change nothing). Both assertions this suite makes need one of those regimes.

Slow ksat, so water ponds and routes through FillSpillMerge rather than draining away -- with
fast drainage there would be no surface water and nothing to couple.
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wtm_testgrid import write_tif as _write_tif  # noqa: E402

NX, NY = 24, 24
REGION, TIME = "coupling_iteration", "t0"
OUTDIR = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUTDIR, exist_ok=True)

# Intended grid; WTM derives geometry from the geotransform the shared writer encodes (#124).
CELLS_PER_DEGREE = 10.0
SOUTHERN_EDGE    = -45.0


def write_tif(path, data, dtype="float32"):
    _write_tif(path, data, CELLS_PER_DEGREE, SOUTHERN_EDGE, dtype=dtype)


topo = np.full((NY, NX), 100.0, dtype=np.float32)     # rim
topo[6:18, 5:19] = 92.0                               # the pit floor
topo[18:24, 10:14] = 96.0                             # outlet notch to the bottom ocean edge (sill 96)

slope = np.zeros((NY, NX), dtype=np.float32)

mask = np.ones((NY, NX), dtype=np.float32)
mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0

precip          = np.full((NY, NX), 3.0, dtype=np.float32)
evap            = np.full((NY, NX), 0.2, dtype=np.float32)
open_water_evap = np.full((NY, NX), 0.5, dtype=np.float32)
winter_temp     = np.zeros((NY, NX), dtype=np.float32)
ksat            = np.full((NY, NX), 1e-6, dtype=np.float32)   # slow: water ponds and routes via FSM
porosity        = np.full((NY, NX), 0.25, dtype=np.float32)
starting_wt     = np.zeros((NY, NX), dtype=np.float64)

files = {
    f"{REGION}_{TIME}_topography.tif":             (topo, "float32"),
    f"{REGION}_{TIME}_slope.tif":                  (slope, "float32"),
    f"{REGION}_{TIME}_mask.tif":                   (mask, "float32"),
    f"{REGION}_{TIME}_precipitation.tif":          (precip, "float32"),
    f"{REGION}_{TIME}_evaporation.tif":            (evap, "float32"),
    f"{REGION}_{TIME}_open_water_evaporation.tif": (open_water_evap, "float32"),
    f"{REGION}_{TIME}_winter_temperature.tif":     (winter_temp, "float32"),
    f"{REGION}_horizontal_ksat.tif":               (ksat, "float32"),
    f"{REGION}_porosity.tif":                      (porosity, "float32"),
    f"{REGION}_{TIME}_starting_wt.tif":            (starting_wt, "float64"),
}
for fname, (arr, dt) in files.items():
    write_tif(os.path.join(OUTDIR, fname), arr, dt)
print("Done: single-pit coupling fixture (floor 92, sill 96 -> ocean).")
