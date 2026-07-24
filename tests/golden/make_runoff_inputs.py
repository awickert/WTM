#!/usr/bin/env python3
"""Synthetic inputs for the runoff-distribution golden case (runoff_ratio_on + FSM).

Topography comes from the shared spectral (Fourier-mode) terrain generator: the
fundamental (1,1) mode gives two hills and two closed depressions. It is band-limited
(smooth gradients -> deterministic FillSpillMerge routing -> cross-rank reproducible).
The recharge is split by a large runoff_ratio so most of it becomes runoff, which FSM
routes into the depressions; the water table starts deep and ksat is low, so groundwater
stays below the surface and the runoff is the dominant signal. Run with evap_mode 1
(keeps surface water). See tests/spectral_terrain.py and the golden fsm_runoff case.
"""
import numpy as np
import os
import sys
import rasterio
from rasterio.transform import from_bounds

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from spectral_terrain import spectral_terrain

NX, NY = 16, 16
TIME = "t0"
OUTDIR = os.path.join(os.path.dirname(__file__), "inputs_runoff")
os.makedirs(OUTDIR, exist_ok=True)
transform = from_bounds(0, 0, NX, NY, NX, NY)
CRS = "EPSG:4326"

# region name -> Fourier mode spectrum (kx, ky, amplitude_m). Base 50 m; kept band-limited
# (max wavenumber 3 << Nyquist 7) so gradients stay smooth and FSM routing is deterministic.
REGIONS = {
    "runoff_test": [(1, 1, 10.0)],
}
BASE = 50.0


def write_tif(path, data, dtype="float32"):
    with rasterio.open(path, "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dtype, crs=CRS, transform=transform) as dst:
        dst.write(data.astype(dtype), 1)


# Forcing shared by both regions. Large runoff_ratio (0.8) -> infiltrating recharge is
# tiny, so the deep water table stays below ground and the runoff is the surface signal.
precip          = np.full((NY, NX), 0.2, dtype=np.float32)
evap            = np.zeros((NY, NX),      dtype=np.float32)
open_water_evap = np.full((NY, NX), 0.1, dtype=np.float32)
winter_temp     = np.zeros((NY, NX),      dtype=np.float32)
ksat            = np.full((NY, NX), 1e-6, dtype=np.float32)   # low -> GW stays deep
porosity        = np.full((NY, NX), 0.25, dtype=np.float32)
runoff_ratio    = np.full((NY, NX), 0.8,  dtype=np.float32)
slope           = np.zeros((NY, NX), dtype=np.float32)

for region, modes in REGIONS.items():
    topo, mask = spectral_terrain((NY, NX), modes, base=BASE)
    # Water table 40 m below the surface -> no supplied surface water; stays below ground.
    # starting_wt is the water-table DEPTH relative to the surface (positive = above).
    starting_wt = np.full((NY, NX), -40.0, dtype=np.float64)

    files = {
        f"{region}_{TIME}_topography.tif":             (topo, "float32"),
        f"{region}_{TIME}_slope.tif":                  (slope, "float32"),
        f"{region}_{TIME}_mask.tif":                   (mask, "float32"),
        f"{region}_{TIME}_precipitation.tif":          (precip, "float32"),
        f"{region}_{TIME}_evaporation.tif":            (evap, "float32"),
        f"{region}_{TIME}_open_water_evaporation.tif": (open_water_evap, "float32"),
        f"{region}_{TIME}_winter_temperature.tif":     (winter_temp, "float32"),
        f"{region}_horizontal_ksat.tif":               (ksat, "float32"),
        f"{region}_porosity.tif":                      (porosity, "float32"),
        f"{region}_{TIME}_runoff_ratio.tif":           (runoff_ratio, "float32"),
        f"{region}_{TIME}_starting_wt.tif":            (starting_wt, "float64"),
    }
    for fname, (arr, dt) in files.items():
        write_tif(os.path.join(OUTDIR, fname), arr, dt)
    print(f"  wrote region {region} ({len(modes)} mode(s))")
print("Done.")
