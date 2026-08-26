#!/usr/bin/env python3
"""Synthetic inputs for the FillSpillMerge (FSM) MPI-consistency test.

Unlike the ghost_cell fixture (whose water table stays below ground, so FSM is a
no-op), this fixture supplies an initial water table ABOVE the surface over a
plateau containing an off-centre depression. FSM must route that surface water
downhill and pond it in the depression -- so FSM genuinely modifies wtd, and the
result must be identical whether the run is on 1 or N MPI ranks. The depression
is placed off-centre so that a 2-D domain decomposition splits both it and the
surrounding flow field across ranks.
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wtm_testgrid import write_tif as _write_tif  # noqa: E402

NX, NY = 16, 16          # 14x14 interior after the ocean edge ring
REGION, TIME = "fsm_test", "t0"
OUTDIR = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUTDIR, exist_ok=True)

# Intended grid: WTM derives geometry from the geotransform (#124), which the shared writer encodes.
CELLS_PER_DEGREE = 10.0
SOUTHERN_EDGE    = -45.0


def write_tif(path, data, dtype="float32"):
    _write_tif(path, data, CELLS_PER_DEGREE, SOUTHERN_EDGE, dtype=dtype)


# Plateau at 100 m with a square pit (depression) at 90 m, placed off-centre.
topo = np.full((NY, NX), 100.0, dtype=np.float32)
topo[9:13, 9:13] = 90.0        # off-centre pit -> crosses a 2-D rank split

slope = np.zeros((NY, NX), dtype=np.float32)

mask = np.ones((NY, NX), dtype=np.float32)
mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0

# Modest forcing; the point is FSM moving the SUPPLIED surface water, not recharge.
precip          = np.full((NY, NX), 0.1, dtype=np.float32)
evap            = np.zeros((NY, NX), dtype=np.float32)
open_water_evap = np.full((NY, NX), 0.2, dtype=np.float32)
winter_temp     = np.zeros((NY, NX), dtype=np.float32)
ksat            = np.full((NY, NX), 1e-4, dtype=np.float32)
# Vertical (infiltration) conductivity. Only read when infiltration_during_flow is on, which is
# what selects the SERIAL rank-0 recharge path (distribute_recharge = !fsm_on || !infiltration_on).
# Added so that path can be tested at all -- see tests/serial_recharge.
vert_ksat       = np.full((NY, NX), 1e-6, dtype=np.float32)
porosity        = np.full((NY, NX), 0.25, dtype=np.float32)

# Initial water table 5 m ABOVE the surface everywhere -> abundant surface water
# for FSM to redistribute into the pit.
starting_wt = np.full((NY, NX), 5.0, dtype=np.float64)

files = {
    f"{REGION}_{TIME}_topography.tif":            (topo, "float32"),
    f"{REGION}_{TIME}_slope.tif":                 (slope, "float32"),
    f"{REGION}_{TIME}_mask.tif":                  (mask, "float32"),
    f"{REGION}_{TIME}_precipitation.tif":         (precip, "float32"),
    f"{REGION}_{TIME}_evaporation.tif":           (evap, "float32"),
    f"{REGION}_{TIME}_open_water_evaporation.tif": (open_water_evap, "float32"),
    f"{REGION}_{TIME}_winter_temperature.tif":    (winter_temp, "float32"),
    f"{REGION}_horizontal_ksat.tif":              (ksat, "float32"),
    f"{REGION}_porosity.tif":                     (porosity, "float32"),
    f"{REGION}_vertical_ksat.tif":                (vert_ksat, "float32"),
    f"{REGION}_{TIME}_starting_wt.tif":           (starting_wt, "float64"),
}
for fname, (arr, dt) in files.items():
    path = os.path.join(OUTDIR, fname)
    write_tif(path, arr, dt)
    print(f"  wrote {path}")
print("Done.")
