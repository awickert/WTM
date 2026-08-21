#!/usr/bin/env python3
"""Cascade + heterogeneous-fill fixture.

A CHAIN of depressions, so surface water flows depression -> depression -> off map,
and (with wetter forcing over the upper pit) one depression fills while the other lags:

  rim 100 m, ocean ring on the edges
  pit A  : floor 94 m, outlet sill 97 m  -> spills DOWN into basin B
  basin B: floor 88 m, outlet sill 95 m  -> spills OFF-MAP to the ocean

Precip is higher over the upper half (pit A) than the lower half (basin B), so A fills
to its 97 m sill and spills into B while B is still filling -- exercising A->B->ocean
routing AND a state where A is full/spilling and B is not (heterogeneous fullness).
"""
import numpy as np
import os
import rasterio
from rasterio.transform import from_bounds

NX, NY = 30, 30
REGION, TIME = "fsm_cascade", "t0"
OUTDIR = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUTDIR, exist_ok=True)
transform = from_bounds(0, 0, NX, NY, NX, NY)
CRS = "EPSG:4326"


def write_tif(path, data, dtype="float32"):
    with rasterio.open(path, "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dtype, crs=CRS, transform=transform) as dst:
        dst.write(data.astype(dtype), 1)


topo = np.full((NY, NX), 100.0, dtype=np.float32)     # rim
# Basin B (lower), rows 15-26, with a 95 m outlet notch down to the bottom ocean edge.
topo[15:27, 4:26] = 88.0
topo[26:30, 13:16] = 95.0                             # B -> ocean notch (sill 95)
# Pit A (upper), rows 4-11, connected to B by a 97 m channel.
topo[4:12, 11:20] = 94.0
topo[11:15, 13:16] = 97.0                             # A -> B channel (sill 97)

slope = np.zeros((NY, NX), dtype=np.float32)

mask = np.ones((NY, NX), dtype=np.float32)
mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0

# Wetter over the upper half (pit A) than the lower half (basin B) -> A fills first.
precip = np.full((NY, NX), 1.0, dtype=np.float32)
precip[:15, :] = 4.0
evap            = np.full((NY, NX), 0.2, dtype=np.float32)
open_water_evap = np.full((NY, NX), 0.5, dtype=np.float32)
winter_temp     = np.zeros((NY, NX), dtype=np.float32)
ksat            = np.full((NY, NX), 1e-6, dtype=np.float32)   # slow drainage -> water ponds and routes via FSM
porosity        = np.full((NY, NX), 0.25, dtype=np.float32)
starting_wt     = np.zeros((NY, NX), dtype=np.float64)

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
    f"{REGION}_{TIME}_starting_wt.tif":           (starting_wt, "float64"),
}
for fname, (arr, dt) in files.items():
    write_tif(os.path.join(OUTDIR, fname), arr, dt)
print("Done: cascade fixture (pit A@94 sill97 -> basin B@88 sill95 -> ocean; wetter over A).")
