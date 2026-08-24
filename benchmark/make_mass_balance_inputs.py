#!/usr/bin/env python3
"""Small synthetic fixture for the mass-balance MPI-consistency test.

The test only needs `run_type test` inputs (topography + slope); WTM synthesizes the rest (edge-ocean mask,
uniform precip/evap/ksat/porosity) internally. The invariant it checks -- the cumulative water-budget
diagnostics (total_added_recharge, total_loss_to_ocean) agree between n=1 and n=N ranks -- is independent of
grid size, so a small dome (radial drainage to the auto edge-ocean, exercising BOTH the GW and FSM ocean-loss
paths) tests exactly the same accounting/reduce as the old 1000x1000 global DEM, in seconds instead of minutes.
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
from wtm_testgrid import write_tif  # noqa: E402

NX = NY = 40                       # splits cleanly across 1..8 ranks; big enough that a 2-D decomposition is real
REGION = "mb"
OUTDIR = os.path.join(os.path.dirname(__file__), "mb_inputs")
os.makedirs(OUTDIR, exist_ok=True)
CELLS_PER_DEGREE, SOUTHERN_EDGE = 10.0, -45.0

# Dome: high centre sloping to low edges, so groundwater drains radially to the ocean ring and surface excess
# near the rim ponds and routes to the ocean via FillSpillMerge -- both budget-loss paths active.
yy, xx = np.mgrid[0:NY, 0:NX]
r = np.sqrt((xx - (NX - 1) / 2) ** 2 + (yy - (NY - 1) / 2) ** 2)
topo = (5.0 + 45.0 * (1.0 - r / r.max())).astype(np.float32)   # ~50 m centre -> ~5 m rim
slope = np.zeros((NY, NX), dtype=np.float32)

write_tif(os.path.join(OUTDIR, f"{REGION}_topography.tif"), topo, CELLS_PER_DEGREE, SOUTHERN_EDGE)
write_tif(os.path.join(OUTDIR, f"{REGION}_slope.tif"), slope, CELLS_PER_DEGREE, SOUTHERN_EDGE)
print(f"wrote {OUTDIR}/{REGION}_topography.tif + slope ({NX}x{NY} dome, topo {topo.min():.1f}..{topo.max():.1f} m)")
