#!/usr/bin/env python3
"""128x128 sub-surface-equilibrium fixture for the Picard timestep experiments.

Flat land at 100 m ringed by ocean (Dirichlet h=0), homogeneous ksat, ZERO
forcing -- so the interior (starting at the surface, h = topo = 100 m; wtd = 0)
drains monotonically to the ocean ring, a clean groundwater-mound decay to steady
state with no surface-water toggling. The diffusion timescale (~2e4 yr with
ksat 1e-3) spans "many steps" to "a few steps" across the dt sweep. Writes the 9
GeoTIFFs run_type equilibrium expects.

Usage:  python3 make_equil128.py        # writes to $WTM_WORK/equil128_inputs
"""
import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds

NX = NY = 128
REGION, TIME = "equil128", "t0"
WORK = os.environ.get("WTM_WORK", "/tmp/wtm_picard_bench")
OUTDIR = os.path.join(WORK, "equil128_inputs")
os.makedirs(OUTDIR, exist_ok=True)
transform = from_bounds(0, 0, NX, NY, NX, NY)


def w(path, data, dtype="float32"):
    with rasterio.open(path, "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dtype, crs="EPSG:4326", transform=transform) as dst:
        dst.write(data.astype(dtype), 1)


topo  = np.full((NY, NX), 100.0, np.float32)
slope = np.zeros((NY, NX), np.float32)                # -> fdepth = fdepth_a
mask  = np.ones((NY, NX), np.float32)
mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0   # ocean ring (Dirichlet h=0)
precip          = np.zeros((NY, NX), np.float32)      # zero forcing -> monotonic drainage
evap            = np.zeros((NY, NX), np.float32)
open_water_evap = np.zeros((NY, NX), np.float32)
winter_temp     = np.zeros((NY, NX), np.float32)      # > -5 -> no permafrost
ksat            = np.full((NY, NX), 1e-3, np.float32)
porosity        = np.full((NY, NX), 0.25, np.float32)

files = {
    f"{REGION}_{TIME}_topography.tif": topo,
    f"{REGION}_{TIME}_slope.tif": slope,
    f"{REGION}_{TIME}_mask.tif": mask,
    f"{REGION}_{TIME}_precipitation.tif": precip,
    f"{REGION}_{TIME}_evaporation.tif": evap,
    f"{REGION}_{TIME}_open_water_evaporation.tif": open_water_evap,
    f"{REGION}_{TIME}_winter_temperature.tif": winter_temp,
    f"{REGION}_horizontal_ksat.tif": ksat,
    f"{REGION}_porosity.tif": porosity,
}
for fn, arr in files.items():
    w(os.path.join(OUTDIR, fn), arr)
print(f"wrote {len(files)} files to {OUTDIR}")
