#!/usr/bin/env python3
"""Three geometries that differ ONLY in where a lake's water has to go to reach the ocean.

WHY THIS FIXTURE EXISTS. On tests/golden's `transient` fixture the time-stepping error against a
dt-refined reference is not spread over the domain at all: the MEDIAN error across land is exactly
0.0, and everything above 10 cm sits in THREE cells. Those three are not "coastal cells" in general
-- the other ~50 land cells with an ocean neighbour are exact. They are the cells between the
depression and the NEAREST ocean, i.e. the path the lake's water takes to leave.

That was found on a fixture whose pit happens to sit off-centre (2 land cells from the right coast,
8 from the left), so the finding and the fixture's lopsidedness were confounded. These three
geometries separate them. The pit is IDENTICAL in size and depth in all three and STATIC in time
(golden's version migrates between ta and tb, which would add a second moving part); only its
horizontal position changes:

    exit_right   pit 2 land cells from the RIGHT coast, 8 from the left
    exit_left    the mirror image
    centre       5 land cells from BOTH -- no nearest coast

WHAT EACH ONE IS FOR. exit_right and exit_left ask whether the error FOLLOWS the water: mirror the
geometry and the error should mirror with it. `centre` is the one that catches a DIRECTIONAL BUG --
with no nearest coast, a correct model has no reason to prefer a side, so an asymmetric answer there
means the asymmetry is in the code and not in the terrain.
"""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wtm_testgrid import write_tif as _write_tif  # noqa: E402

NX, NY = 16, 16
CELLS_PER_DEGREE, SOUTHERN_EDGE = 10.0, -45.0
HERE = os.path.dirname(os.path.abspath(__file__))

# Pit rows are the same everywhere; only the COLUMNS move. Land is columns 1..14 (0 and 15 are ocean),
# so col 12 leaves 2 land cells to the right coast and col 3 leaves 2 to the left. 6..9 is centred.
PIT_ROWS = (6, 10)
GEOM = {"exit_right": (9, 13), "exit_left": (3, 7), "centre": (6, 10)}


def plateau_with_pit(c0, c1):
    t = np.full((NY, NX), 100.0, dtype=np.float32)
    t[PIT_ROWS[0]:PIT_ROWS[1], c0:c1] = 90.0
    return t


mask = np.ones((NY, NX), dtype=np.float32)
mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
zeros = np.zeros((NY, NX), dtype=np.float32)

for name, (c0, c1) in GEOM.items():
    out = os.path.join(HERE, "inputs_" + name)
    os.makedirs(out, exist_ok=True)
    topo = plateau_with_pit(c0, c1)
    fields = {
        "topography": topo,
        "slope": zeros,
        "mask": mask,
        "precipitation": np.full((NY, NX), 0.1, dtype=np.float32),
        "evaporation": zeros,
        "open_water_evaporation": np.full((NY, NX), 0.2, dtype=np.float32),
        "winter_temperature": zeros,
    }
    for t in ("ta", "tb"):                       # STATIC: ta and tb identical, unlike golden's
        for f, a in fields.items():
            _write_tif(os.path.join(out, f"exitpath_{t}_{f}.tif"), a, CELLS_PER_DEGREE, SOUTHERN_EDGE)
    _write_tif(os.path.join(out, "exitpath_ta_wtd.tif"),
               np.full((NY, NX), 5.0, dtype=np.float64), CELLS_PER_DEGREE, SOUTHERN_EDGE, dtype="float64")
    for f, a in (("horizontal_ksat", np.full((NY, NX), 1e-4, dtype=np.float32)),
                 ("porosity", np.full((NY, NX), 0.25, dtype=np.float32))):
        _write_tif(os.path.join(out, f"exitpath_{f}.tif"), a, CELLS_PER_DEGREE, SOUTHERN_EDGE)
    left = c0 - 1
    right = 14 - (c1 - 1)
    print(f"  {name:<11} pit cols {c0}..{c1-1}   {left} land cells to the LEFT coast, {right} to the RIGHT")
