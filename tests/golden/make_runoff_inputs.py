#!/usr/bin/env python3
"""Synthetic inputs for the runoff-distribution golden case (runoff_ratio_on + FSM).

Topography is the 2D sinusoid Kerry & Andy use for WTM tests: one wavelength in each
direction over the interior, giving two hills and two closed depressions. The smooth
gradient makes FSM flow routing unambiguous (cross-rank reproducible), and the
depressions are closed basins that pond surface water.

The point of THIS fixture is the runoff: recharge is split by a large runoff_ratio so
most of it becomes runoff (arp.runoff), which FSM routes into the depressions. The
water table starts deep and ksat is low, so groundwater stays well below the surface
over the run -- the ONLY surface water is the runoff-fed ponds. Run with evap_mode 1
(keeps surface water). Skipping the runoff gather then leaves the depressions dry: a
large, deterministic change, so the case bites the runoff path.
"""
import numpy as np
import os
import rasterio
from rasterio.transform import from_bounds

NX, NY = 16, 16
REGION, TIME = "runoff_test", "t0"
OUTDIR = os.path.join(os.path.dirname(__file__), "inputs_runoff")
os.makedirs(OUTDIR, exist_ok=True)
transform = from_bounds(0, 0, NX, NY, NX, NY)
CRS = "EPSG:4326"


def write_tif(path, data, dtype="float32"):
    with rasterio.open(path, "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dtype, crs=CRS, transform=transform) as dst:
        dst.write(data.astype(dtype), 1)


# 2D sinusoid over the interior (indices 1..NX-2): one wavelength each way -> two hills
# (sin*sin > 0) and two closed depressions (sin*sin < 0). Base 50 m, amplitude 10 m, so
# topo runs 40 m (depression floors) to 60 m (hill tops), with the interior rim at 50 m.
BASE, AMP = 50.0, 10.0
NIN = NX - 2                      # interior width (one wavelength spans the interior)
yy, xx = np.mgrid[0:NY, 0:NX]
phase_x = 2.0 * np.pi * (xx - 1) / NIN
phase_y = 2.0 * np.pi * (yy - 1) / NIN
topo = (BASE + AMP * np.sin(phase_x) * np.sin(phase_y)).astype(np.float32)

slope = np.zeros((NY, NX), dtype=np.float32)

mask = np.ones((NY, NX), dtype=np.float32)
mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0

# Split most of the recharge into runoff (runoff_ratio 0.8): infiltrating recharge is
# then tiny, so the deep water table barely moves and stays below ground; the runoff
# is the surface-water signal FSM ponds in the depressions.
precip          = np.full((NY, NX), 0.2, dtype=np.float32)
evap            = np.zeros((NY, NX),      dtype=np.float32)
open_water_evap = np.full((NY, NX), 0.1, dtype=np.float32)
winter_temp     = np.zeros((NY, NX),      dtype=np.float32)
ksat            = np.full((NY, NX), 1e-6, dtype=np.float32)   # low -> GW stays deep
porosity        = np.full((NY, NX), 0.25, dtype=np.float32)
runoff_ratio    = np.full((NY, NX), 0.8,  dtype=np.float32)

# Water table 40 m below the surface -> no supplied surface water; stays below ground.
# starting_wt is the water-table DEPTH relative to the surface (positive = above), so a
# 40 m below-ground table is -40 everywhere (cf. the fsm fixture's +5 = 5 m above).
starting_wt = np.full((NY, NX), -40.0, dtype=np.float64)

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
    f"{REGION}_{TIME}_runoff_ratio.tif":          (runoff_ratio, "float32"),
    f"{REGION}_{TIME}_starting_wt.tif":           (starting_wt, "float64"),
}
for fname, (arr, dt) in files.items():
    path = os.path.join(OUTDIR, fname)
    write_tif(path, arr, dt)
    print(f"  wrote {path}")
print("Done.")
