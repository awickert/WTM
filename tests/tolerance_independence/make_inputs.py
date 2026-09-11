#!/usr/bin/env python3
"""Fixture for the CONVERGENCE-TOLERANCE INDEPENDENCE test.

THE PROPERTY UNDER TEST NEEDS NO REFERENCE FILE. A converged answer cannot depend on the tolerance you
stopped at. Run the same configuration at the shipped tolerance and at a much tighter one; if the two
disagree, the looser one had not converged -- whatever reason code it printed. That is self-contained:
there is no stored golden to regenerate, so the check cannot be quietly regolded into agreement.

THE FIXTURE IS DELIBERATELY THE SAME CONSTRUCTION as tests/variable_porosity's, with its own region
name. That is where #104 was found and measured, and a test for a defect should run the geometry the
defect was characterised on rather than a fresh one whose behaviour nobody has mapped.

WHAT MAKES IT BITE: a plateau whose water table starts AT the land surface (run.initial_water_table:
saturated), draining west to an ocean strip, under the shipped `active_set` collector. Every cell
begins on the constraint, so the first step is the hardest semismooth solve the model ever does -- and
that is exactly where the relative-water-step criterion mistakes a stall for convergence.

Regenerate with:  python3 make_inputs.py
"""
import numpy as np, os, rasterio
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wtm_testgrid import make_transform  # noqa: E402

CELLS_PER_DEGREE, SOUTHERN_EDGE = 100, 0   # ~1.1 km cells, as variable_porosity uses
NX, NY = 12, 8
REGION = "tolind"
PHI_WEST, PHI_EAST = 0.40, 0.05            # variable phi, so the disagreement is judged in real water
OUT = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUT, exist_ok=True)
tr = make_transform(CELLS_PER_DEGREE, SOUTHERN_EDGE, NY)

def w(name, data, dt="float32"):
    with rasterio.open(os.path.join(OUT, name), "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dt, crs="EPSG:4326", transform=tr) as d:
        d.write(data.astype(dt), 1)

topo = np.full((NY, NX), 100.0, np.float32)   # high plateau
topo[:, 0] = 0.0                              # ocean strip (west) -> lateral drainage gradient
mask = np.ones((NY, NX), np.float32); mask[:, 0] = 0.0
zero = np.zeros((NY, NX), np.float32)
phi  = np.tile(np.linspace(PHI_WEST, PHI_EAST, NX, dtype=np.float32), (NY, 1))

for lay, a in {"topography":topo,"slope":zero,"mask":mask,"precipitation":np.full((NY,NX),0.3,np.float32),
               "evaporation":zero,"open_water_evaporation":zero,"winter_temperature":zero}.items():
    w(f"{REGION}_ta_{lay}.tif", a); w(f"{REGION}_tb_{lay}.tif", a)
w(f"{REGION}_horizontal_ksat.tif", np.full((NY, NX), 1e-3, np.float32))
w(f"{REGION}_porosity.tif", phi)
w(f"{REGION}_ta_wtd.tif", np.zeros((NY, NX), np.float64), "float64")
print("wrote", OUT, "region", REGION, "porosity %.2f..%.2f" % (phi.min(), phi.max()))
