#!/usr/bin/env python3
"""Sub-surface-equilibrium fixture for the Picard timestep/adaptive experiments.

Flat land at 100 m ringed by ocean (Dirichlet h=0), homogeneous ksat, ZERO
forcing -- so the interior (starting at the surface, h = topo = 100 m; wtd = 0)
drains monotonically to the ocean ring, a clean groundwater-mound decay to steady
state with no surface-water toggling. cells_per_degree = grid/12.8 keeps the
PHYSICAL domain fixed at 12.8 degrees for every grid, so the diffusion timescale
(~8e4 yr) -- and hence the number of adaptive steps -- is grid-independent
(mesh-refinement, not a different problem). grid=128 reproduces the original
128^2 fixture (cpd=10). Writes the 9 GeoTIFFs run_type equilibrium expects.

Usage:  python3 make_equil.py [grid]      # default 128; writes $WTM_WORK/equil<grid>_inputs
"""
import os, sys
import numpy as np
import rasterio
from rasterio.transform import from_bounds

GRID = int(sys.argv[1]) if len(sys.argv) > 1 else 128
NX = NY = GRID
REGION, TIME = f"equil{GRID}", "t0"
WORK = os.environ.get("WTM_WORK", "/tmp/wtm_picard_bench")
OUTDIR = os.path.join(WORK, f"equil{GRID}_inputs")
os.makedirs(OUTDIR, exist_ok=True)
transform = from_bounds(0, 0, NX, NY, NX, NY)

# cells_per_degree for this grid (fixed 12.8-degree domain). Callers must set the
# same value in the config's cells_per_degree.
CPD = GRID / 12.8


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
print(f"wrote {len(files)} files to {OUTDIR}  (grid={GRID}, cells_per_degree={CPD:.4f})")
