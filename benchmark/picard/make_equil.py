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
# cells_per_degree for this grid (fixed 12.8-degree domain). Callers must set the
# same value in the config's cells_per_degree.
CPD = GRID / 12.8

# GEOREFERENCING. This used to be `from_bounds(0, 0, NX, NY, NX, NY)` -- a pixel-space
# placeholder -- and that was harmless while geometry came from the config's grid: block
# (cells_per_degree + southern_edge) and the geotransform was decorative. #124 made the
# geotransform AUTHORITATIVE (derive_grid_geometry in src/irf.cpp warns that grid: is now
# deprecated and IGNORED whenever the raster carries one), which turned the placeholder into
# the thing the model believes: 1-degree cells running from latitude 0 to 128, so every row
# above 90 degrees got cos(lat) < 0 and the model aborted with "Cell with a negative area was
# found!". Every script in this directory died there.
#
# So this is not a change of geometry -- it is writing down the geometry the fixture always
# had. SOUTH = -45 and a 12.8-degree domain are exactly what the callers' configs specified
# via southern_edge/cells_per_degree, and src/test_geometry.cpp pins that the geotransform
# path reproduces the old cells_per_degree path. Keep SOUTH in step with the callers'
# southern_edge; they describe one grid and must not drift apart.
SOUTH  = -45.0                 # matches southern_edge in every caller's config
DOMAIN = 12.8                  # degrees, fixed for all grids (see CPD above)
transform = from_bounds(0.0, SOUTH, DOMAIN, SOUTH + DOMAIN, NX, NY)


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
