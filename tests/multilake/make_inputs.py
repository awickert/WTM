#!/usr/bin/env python3
"""Synthetic inputs for the MULTI-LAKE fixture: several genuine lakes at DIFFERENT stages.

WHY THIS EXISTS. The lake-aware active-set exfiltration constraint pins each cell's head at
`topo + surface_water_depth`, which is meant to be ONE flat free-surface elevation per lake.
Every fixture we had exercised that on a single depression -- on the island benchmark the entire
claim rested on one 4-cell lake, with the other ponded cells being single-cell pits or
below-sea-level fill. A one-lake fixture cannot distinguish "the pin reproduces a flat lake" from
"the pin holds whatever level it was handed", and cannot test that DIFFERENT lakes hold DIFFERENT
stages simultaneously.

THE DESIGN. A plateau ringed by ocean, containing four depressions whose floors are at four
different elevations, so their equilibrium stages must differ:

    A  a 4x4 pit at 90 m               -- plain lake, several cells wide
    B  a 6x3 pit at 95 m               -- shallower and differently shaped, so it holds a
                                          different stage and a different depth-to-width ratio
    C  a NESTED pair: an 8x8 basin at 92 m with a 3x3 inner pit at 85 m
                                       -- a metadepression: the inner pit fills first, then the
                                          outer basin, so FSM's hierarchy is genuinely walked and
                                          the two sub-lakes MERGE into one free surface once the
                                          inner one fills
    D  a 3x3 pit at 88 m behind a LOW SILL (97 m) near the coast
                                       -- fills, spills over the sill, and drains to the ocean, so
                                          at least one lake is capacity-limited rather than
                                          supply-limited

Each depression is several cells across, so `surface_water_depth` VARIES across it while
`topo + surface_water_depth` must not -- that difference is the whole point of the test.

The floors are deliberately non-uniform inside A and C so that a flat free surface implies a
per-cell depth that varies -- a fixture with a flat floor would pass the flatness assertion
trivially, since a constant depth on a constant floor is flat for the wrong reason.
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wtm_testgrid import write_tif as _write_tif  # noqa: E402

NX, NY = 40, 40
REGION, TIME = "multilake", "t0"
OUTDIR = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUTDIR, exist_ok=True)

CELLS_PER_DEGREE = 10.0
SOUTHERN_EDGE = -45.0


def write_tif(path, data, dtype="float32"):
    _write_tif(path, data, CELLS_PER_DEGREE, SOUTHERN_EDGE, dtype=dtype)


# The plateau is gently TILTED (100 m in the north-west down to 97 m in the south-east) rather than
# flat. On a perfectly flat plateau the flow directions are degenerate and every depression ends up
# sharing one catchment, which is what made the first version of this fixture fill all four lakes to
# a single common level -- a fixture that cannot show different stages cannot test for them.
PLATEAU = 100.0
ramp_y = np.linspace(0.0, 1.5, NY, dtype=np.float32)[:, None]
ramp_x = np.linspace(0.0, 1.5, NX, dtype=np.float32)[None, :]
topo = (PLATEAU - ramp_y - ramp_x).astype(np.float32)

# --- A: plain 4x4 pit at 90 m, with a tilted floor so a flat lake implies varying depth ---
topo[5:9, 5:9] = 90.0
topo[5:9, 5:9] += np.linspace(0.0, 1.5, 4, dtype=np.float32)[None, :]   # floor 90.0 .. 91.5

# --- B: 6x3 pit at 95 m (shallower, different aspect ratio) ---
topo[6:9, 20:26] = 95.0

# --- C: nested metadepression -- 8x8 basin at 92 m containing a 3x3 inner pit at 85 m ---
topo[22:30, 6:14] = 92.0
topo[22:30, 6:14] += np.linspace(0.0, 0.8, 8, dtype=np.float32)[:, None]  # basin floor 92.0 .. 92.8
topo[25:28, 9:12] = 85.0                                                  # inner pit

# --- D: 3x3 pit at 88 m behind a low sill, near the coast, so it spills to the ocean ---
topo[24:27, 30:33] = 88.0
topo[23:28, 33] = 97.0     # the sill: lower than the 100 m plateau, so D overflows over it
topo[23:28, 34:38] = 96.0  # a shallow ramp from the sill down toward the ocean edge

# Ocean ring: mask 0 and topography at sea level, so spilled water leaves the domain.
mask = np.ones((NY, NX), dtype=np.float32)
mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
topo[mask == 0] = 0.0

slope = np.zeros((NY, NX), dtype=np.float32)

# Forcing. The crux: OPEN-WATER EVAPORATION EXCEEDS PRECIPITATION (2.0 vs 1.0 m/yr), so a lake loses
# water from its own surface and can only be sustained by runoff from its catchment. Each lake then
# equilibrates where (catchment inflow) balances (lake area x net open-water loss) -- i.e. at a stage
# set by its own catchment-to-area ratio, which differs per depression. That is what makes the four
# lakes hold four DIFFERENT levels instead of all filling to the brim.
precip = np.full((NY, NX), 1.0, dtype=np.float32)
evap = np.full((NY, NX), 0.2, dtype=np.float32)
open_water_evap = np.full((NY, NX), 2.0, dtype=np.float32)
winter_temp = np.zeros((NY, NX), dtype=np.float32)

# Low conductivity so the lakes are not drained laterally through the aquifer faster than they
# fill -- the point is to hold standing water long enough to test the free surface.
ksat = np.full((NY, NX), 1e-5, dtype=np.float32)
porosity = np.full((NY, NX), 0.25, dtype=np.float32)

# Start DRY (10 m below the surface) so the lakes must actually be filled by routed water rather
# than inherited from the initial condition -- otherwise the pin could be reproducing a level it
# was handed at t=0 rather than one the coupled model produced.
starting_wt = np.full((NY, NX), -10.0, dtype=np.float64)
starting_wt[mask == 0] = 0.0

files = {
    f"{REGION}_{TIME}_topography.tif": (topo, "float32"),
    f"{REGION}_{TIME}_slope.tif": (slope, "float32"),
    f"{REGION}_{TIME}_mask.tif": (mask, "float32"),
    f"{REGION}_{TIME}_precipitation.tif": (precip, "float32"),
    f"{REGION}_{TIME}_evaporation.tif": (evap, "float32"),
    f"{REGION}_{TIME}_open_water_evaporation.tif": (open_water_evap, "float32"),
    f"{REGION}_{TIME}_winter_temperature.tif": (winter_temp, "float32"),
    f"{REGION}_horizontal_ksat.tif": (ksat, "float32"),
    f"{REGION}_porosity.tif": (porosity, "float32"),
    f"{REGION}_{TIME}_starting_wt.tif": (starting_wt, "float64"),
}
for fname, (arr, dt) in files.items():
    path = os.path.join(OUTDIR, fname)
    write_tif(path, arr, dt)
    print(f"  wrote {path}")
print(f"Done. {NX}x{NY}, four depressions with floors at 85/88/90/92/95 m.")
