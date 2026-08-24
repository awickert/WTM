#!/usr/bin/env python3
"""Synthetic inputs for the TRANSIENT golden/consistency test.

Transient runs interpolate the time-varying fields from time 'ta' to 'tb' by
cycles_done/total_reports, and rebuild the depression hierarchy every cycle (the
per-cycle dephier path that Phase 2d moves to rank 0). To exercise that path the
topography must CHANGE between ta and tb, so the depression moves and dephier's
result differs each cycle. Surface water is supplied (initial wtd above ground)
so FillSpillMerge is active too.
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wtm_testgrid import write_tif as _write_tif  # noqa: E402

NX, NY = 16, 16
REGION = "transient_test"
OUTDIR = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUTDIR, exist_ok=True)

# Intended grid: WTM derives geometry from the geotransform (#124), which the shared writer encodes.
CELLS_PER_DEGREE = 10.0
SOUTHERN_EDGE    = -45.0


def write_tif(path, data, dtype="float32"):
    _write_tif(path, data, CELLS_PER_DEGREE, SOUTHERN_EDGE, dtype=dtype)


def plateau_with_pit(y0, y1, x0, x1):
    t = np.full((NY, NX), 100.0, dtype=np.float32)
    t[y0:y1, x0:x1] = 90.0
    return t


# Topography moves the pit between ta and tb -> dephier differs each cycle.
topo_ta = plateau_with_pit(9, 13, 9, 13)
topo_tb = plateau_with_pit(5, 9, 5, 9)

slope = np.zeros((NY, NX), dtype=np.float32)
mask = np.ones((NY, NX), dtype=np.float32)
mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0

precip          = np.full((NY, NX), 0.1, dtype=np.float32)
evap            = np.zeros((NY, NX), dtype=np.float32)
open_water_evap = np.full((NY, NX), 0.2, dtype=np.float32)
winter_temp     = np.zeros((NY, NX), dtype=np.float32)
ksat            = np.full((NY, NX), 1e-4, dtype=np.float32)
porosity        = np.full((NY, NX), 0.25, dtype=np.float32)
wtd_ta          = np.full((NY, NX), 5.0, dtype=np.float64)   # surface water at start

per_time = {
    "topography": (topo_ta, topo_tb),
    "slope": (slope, slope),
    "mask": (mask, mask),
    "precipitation": (precip, precip),
    "evaporation": (evap, evap),
    "open_water_evaporation": (open_water_evap, open_water_evap),
    "winter_temperature": (winter_temp, winter_temp),
}
for layer, (a, b) in per_time.items():
    write_tif(os.path.join(OUTDIR, f"{REGION}_ta_{layer}.tif"), a)
    write_tif(os.path.join(OUTDIR, f"{REGION}_tb_{layer}.tif"), b)

# Static fields and the initial water table (loaded at time_start).
write_tif(os.path.join(OUTDIR, f"{REGION}_horizontal_ksat.tif"), ksat)
write_tif(os.path.join(OUTDIR, f"{REGION}_porosity.tif"), porosity)
write_tif(os.path.join(OUTDIR, f"{REGION}_ta_wtd.tif"), wtd_ta, "float64")

print(f"  wrote transient inputs to {OUTDIR}")
