#!/usr/bin/env python3
"""Land-edge fixture for the mask-aware ghost boundary (now the default; task #96).

A coastal wedge: an ocean strip on the WEST (i=0, mask=0 -> Dirichlet h=0), LAND everywhere else, so the
NORTH, SOUTH, and EAST domain edges are all LAND edges. Topography rises eastward, so the boundary
exercises every ghost case:
  - WEST edge  -> Dirichlet ocean (h=0): the drainage outlet.
  - EAST edge  -> land-slope Neumann with terrain RISING to the edge -> inflow from off-map upslope.
  - N/S edges  -> land-slope Neumann; topo is flat in j -> zero slope -> no-flow divide.
This is the boundary the padded convention could not represent (it forced every edge to sea-level ocean).
With an ocean outlet the problem has a finite steady state. Every solver integrates the SAME steady
residual, so cc / TR-BDF2 / BDF2-on-V / Newton must all agree under the ghost boundary; run.sh checks that,
MPI determinism, and the Newton Jacobian against finite differences.

Regenerate the committed .tif inputs with:  python3 make_inputs.py
"""
import numpy as np, os, rasterio
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wtm_testgrid import make_transform  # noqa: E402

NX, NY = 24, 20
REGION = "ghostbc"

# CELL SIZE IS PHYSICS HERE, NOT BOOKKEEPING (#34). This generator used
# `from_bounds(0, 0, NX, NY, NX, NY)`, the arbitrary placeholder from when WTM ignored georeferencing.
# WTM derives cell geometry from the geotransform (#124), and that placeholder means ONE DEGREE per
# cell -- 111 km. With ksat 1e-4 and a 24-cell domain, lateral drainage to the ocean outlet is then
# utterly negligible against 0.2 m/yr of recharge, so the whole wedge simply filled up: measured, the
# steady state was wtd == 0 in all 460 land cells, and `collection.method: active_set` pinned them
# there. Every comparison in run.sh was then between identically-zero fields, and the suite passed on
# 0.00e+00 for 460 simulated years.
#
# 1000 cells/degree = ~111 m cells, the value tests/boundary_analytic already uses. Measured across a
# sweep (cells/degree -> land cells still pinned at wtd == 0, out of 460):
#     1 -> 460    4 -> 460    16 -> 460    64 -> 440    256 -> 320    1000 -> 0
# At 1000 the designed coastal wedge actually forms: head rises monotonically 0 -> 93.25 m from the
# ocean outlet to the east land edge, the east-edge head rise is 4.0039 m per column (the terrain slope,
# which IS the neumann_toposlope off-map ghost signature), and the N/S edge rows equal the interior row
# (the zero-slope no-flow divide). That is the boundary this suite exists to validate, now visible in
# the answer instead of buried under a saturated field.
CELLS_PER_DEGREE, SOUTHERN_EDGE = 1000, 0

OUT = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUT, exist_ok=True)
tr = make_transform(CELLS_PER_DEGREE, SOUTHERN_EDGE, NY)

def w(name, data, dt="float32"):
    with rasterio.open(os.path.join(OUT, name), "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dt, crs="EPSG:4326", transform=tr) as d:
        d.write(data.astype(dt), 1)

ii = np.arange(NX)[None, :] * np.ones((NY, 1))       # column index broadcast
topo = (2.0 + 4.0 * ii).astype(np.float32)           # rises eastward: 2 m (i=1) -> 94 m (i=23)
topo[:, 0] = 0.0                                      # ocean column at sea level
mask = np.ones((NY, NX), np.float32); mask[:, 0] = 0.0   # ONLY the west column is ocean
zero = np.zeros((NY, NX), np.float32)
precip = np.full((NY, NX), 0.2, np.float32)          # recharge (m/yr)
ksat   = np.full((NY, NX), 1e-4, np.float32)
poro   = np.full((NY, NX), 0.25, np.float32)
wtd0   = np.full((NY, NX), -5.0, np.float64)         # start below the surface

for lay, a in {"topography":topo,"slope":zero,"mask":mask,"precipitation":precip,
               "evaporation":zero,"open_water_evaporation":zero,"winter_temperature":zero}.items():
    w(f"{REGION}_ta_{lay}.tif", a); w(f"{REGION}_tb_{lay}.tif", a)  # steady in time
w(f"{REGION}_horizontal_ksat.tif", ksat)
w(f"{REGION}_porosity.tif", poro)
w(f"{REGION}_ta_wtd.tif", wtd0, "float64")
print("wrote", OUT, "region", REGION, f"({NX}x{NY}, ocean west, land N/S/E)")
