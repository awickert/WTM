#!/usr/bin/env python3
"""Inputs for the lake-evaporation-equals-ET test.

TWO regions that are IDENTICAL in every field except the evaporation pair, so the
control below is a same-method comparison rather than a different experiment:

  eq   open_water_evaporation == evaporation   (0.2 / 0.2)
  neq  open_water_evaporation != evaporation   (0.5 / 0.2)

Everything else -- topography, mask, precipitation, ksat, porosity, the supplied
initial water table -- is byte-for-byte the same, written from the same arrays.

WHY THE PAIR EXISTS. WTM does not model soil ET and open-water (lake) evaporation as
separate processes with a switch between them; it BLENDS them as a logistic in water-table
depth (`evaporation.et_sigmoid`), so a cell evaporates at the ET rate when deep, at the
open-water rate when ponded, and somewhere between across the transition:

    E_eff(wtd) = ET + (owe - ET) * sigma((wtd - wtd_center) / logistic_width)

Setting owe == ET makes the `(owe - ET)` factor identically zero, so E_eff collapses to
ET at every depth and the sigmoid has nothing left to interpolate. The transition
parameters then cannot influence the answer AT ALL -- not the residual, and not the
Jacobian, since the tangent carries the same factor. That is an EXACT invariant, which is
what `run.sh` asserts, with the `neq` region as the control that proves the parameters do
matter when the two rates differ.

Forcing is chosen so a lake persists at equilibrium: precip (1.0 m/yr) exceeds ET
(0.2 m/yr), and the off-centre pit collects the surplus FillSpillMerge routes into it.
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wtm_testgrid import write_tif as _write_tif  # noqa: E402

NX, NY = 16, 16          # 14x14 interior after the ocean edge ring
TIME   = "t0"
OUTDIR = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUTDIR, exist_ok=True)

CELLS_PER_DEGREE = 10.0
SOUTHERN_EDGE    = -45.0


def write_tif(path, data, dtype="float32"):
    _write_tif(path, data, CELLS_PER_DEGREE, SOUTHERN_EDGE, dtype=dtype)


# Plateau at 100 m with a square pit at 90 m, off-centre so a 2-D rank split crosses it.
topo = np.full((NY, NX), 100.0, dtype=np.float32)
topo[9:13, 9:13] = 90.0

slope = np.zeros((NY, NX), dtype=np.float32)

mask = np.ones((NY, NX), dtype=np.float32)
mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0

precip      = np.full((NY, NX), 1.0, dtype=np.float32)
evap        = np.full((NY, NX), 0.2, dtype=np.float32)
winter_temp = np.zeros((NY, NX), dtype=np.float32)
ksat        = np.full((NY, NX), 1e-4, dtype=np.float32)
vert_ksat   = np.full((NY, NX), 1e-6, dtype=np.float32)
porosity    = np.full((NY, NX), 0.25, dtype=np.float32)

# Initial table 5 m ABOVE the surface: abundant surface water for FSM to redistribute.
starting_wt = np.full((NY, NX), 5.0, dtype=np.float64)

# The ONLY difference between the two regions.
OPEN_WATER_EVAP = {
    "eq":  np.full((NY, NX), 0.2, dtype=np.float32),   # == evaporation
    "neq": np.full((NY, NX), 0.5, dtype=np.float32),   # != evaporation (control)
}

for region, owe in OPEN_WATER_EVAP.items():
    files = {
        f"{region}_{TIME}_topography.tif":             (topo, "float32"),
        f"{region}_{TIME}_slope.tif":                  (slope, "float32"),
        f"{region}_{TIME}_mask.tif":                   (mask, "float32"),
        f"{region}_{TIME}_precipitation.tif":          (precip, "float32"),
        f"{region}_{TIME}_evaporation.tif":            (evap, "float32"),
        f"{region}_{TIME}_open_water_evaporation.tif": (owe, "float32"),
        f"{region}_{TIME}_winter_temperature.tif":     (winter_temp, "float32"),
        f"{region}_horizontal_ksat.tif":               (ksat, "float32"),
        f"{region}_porosity.tif":                      (porosity, "float32"),
        f"{region}_vertical_ksat.tif":                 (vert_ksat, "float32"),
        f"{region}_{TIME}_starting_wt.tif":            (starting_wt, "float64"),
    }
    for fname, (arr, dt) in files.items():
        path = os.path.join(OUTDIR, fname)
        write_tif(path, arr, dt)
        print(f"  wrote {path}")
print("Done.")
