#!/usr/bin/env python3
"""Synthetic inputs for a NESTED-depression fixture (metadepression hierarchy).

Unlike fsm_test (a single leaf pit), this builds a higher-order depression set so
the FSM fullness walk traverses a real hierarchy AND so the #119 lake-aware skim
has a known spill elevation to be checked against at equilibrium:

  plateau 100 m  (rim; ocean ring on the edges)
    basin 95 m   (a broad bowl -> becomes the METADEPRESSION when both pits merge)
      pit A 90 m ) two leaf depressions, separated by the 95 m basin floor (their
      pit B 90 m ) mutual spill level); each spills into the basin at 95 m
    outlet notch 97 m  (the basin/metadepression spills to the ocean here)

So the depression hierarchy is: 2 leaf pits (A, B) + 1 metadepression (the basin),
with known sill elevations -- pit->basin at 95 m, basin->ocean at 97 m.
"""
import numpy as np
import os
import rasterio
from rasterio.transform import from_bounds

NX, NY = 24, 24
REGION, TIME = "fsm_fullness", "t0"
OUTDIR = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUTDIR, exist_ok=True)
transform = from_bounds(0, 0, NX, NY, NX, NY)
CRS = "EPSG:4326"


def write_tif(path, data, dtype="float32"):
    with rasterio.open(path, "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dtype, crs=CRS, transform=transform) as dst:
        dst.write(data.astype(dtype), 1)


topo = np.full((NY, NX), 100.0, dtype=np.float32)   # plateau / rim
topo[6:18, 4:20] = 95.0                              # basin floor  -> metadepression
topo[10:14, 6:10] = 90.0                             # pit A (leaf)
topo[10:14, 14:18] = 90.0                            # pit B (leaf)
# Outlet notch: cut a 97 m channel from the basin (row 6) up to the ocean edge (row 0),
# so the metadepression's spill point to the ocean is a well-defined 97 m.
topo[1:7, 11:13] = 97.0

slope = np.zeros((NY, NX), dtype=np.float32)

mask = np.ones((NY, NX), dtype=np.float32)
mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0   # ocean ring

# Net-positive forcing over the interior so groundwater + FSM fill the pits to
# spilling at steady state (precip > evap); open-water evap on ponded cells.
precip          = np.full((NY, NX), 3.0, dtype=np.float32)
evap            = np.full((NY, NX), 0.2, dtype=np.float32)
open_water_evap = np.full((NY, NX), 0.5, dtype=np.float32)
winter_temp     = np.zeros((NY, NX), dtype=np.float32)
ksat            = np.full((NY, NX), 1e-6, dtype=np.float32)
porosity        = np.full((NY, NX), 0.25, dtype=np.float32)

# Start with a water table AT the surface (saturated), so the depressions can
# actually reach "full" (wtd_vol is small when wtd_only ~ 0) and reach their
# spill stage without an enormous drawdown transient.
starting_wt = np.zeros((NY, NX), dtype=np.float64)

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
print("Done: nested-depression fixture (2 leaf pits @90 + basin metadepression @95, outlet @97).")
