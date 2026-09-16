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

# THREE MORE, all with the pit centred, addressing two limits of the first three.
#
# (a) EQUATOR-CENTRED GRID. WTM reads geographic grids only -- src/grid_geometry.cpp: "Phase 1 supports
#     geographic (lat-lon) grids only: a projected CRS puts dx/dy in metres and makes the cos(lat)
#     treatment wrong" -- so a uniform-cellsize rectangular grid is not an option. Centring the domain on
#     the equator gets the same thing for symmetry purposes: cos(lat) is EVEN in latitude, so rows
#     mirrored about the equator have identical cell widths and up-down becomes a fair test too. The
#     first three sit at -45 deg, where it is not.
#
# (b) BROKEN TIES. `centre` cannot distinguish "the depression hierarchy picks arbitrarily among TIED
#     outlets" -- which src/dephier.hpp documents three times, e.g. "If a depression has more than one
#     outlet at the same level one of them is arbitrarily chosen" -- from "something prefers a direction
#     regardless". Self-symmetry cannot separate them, because ANY left-right symmetric terrain has tied
#     mirror-pair outlets by construction. So tilt_e and tilt_w carry a tiny monotonic tilt that makes
#     ONE outlet strictly lowest, and they are exact mirror images OF EACH OTHER. A model with no
#     directional preference must answer one as the mirror of the other. That test survives broken ties,
#     which the self-symmetry test cannot.
EQUATOR_SOUTHERN_EDGE = -NY / (2.0 * CELLS_PER_DEGREE)   # domain straddles lat 0 symmetrically
# A ONE-AXIS tilt does NOT break the ties: tilting only east-west leaves the whole west perimeter column
# at one elevation, 14 cells tied. The gradient has to run on both axes, and the N-S component must be
# incommensurate with the E-W one or the diagonals tie instead. 0.381966 = 1 - 1/phi, the usual
# low-discrepancy choice. Total relief over the grid is ~0.2 m against a 10 m pit, so it orders the
# outlets and changes nothing else; at 1e-2 m/cell each increment is ~500 float32 ulps near 100 m, far
# above rounding.
TILT    = 1.0e-2        # metres per cell, east-west
TILT_NS = 0.381966      # north-south component, as a fraction of TILT


def plateau_with_pit(c0, c1):
    t = np.full((NY, NX), 100.0, dtype=np.float32)
    t[PIT_ROWS[0]:PIT_ROWS[1], c0:c1] = 90.0
    return t


mask = np.ones((NY, NX), dtype=np.float32)
mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
zeros = np.zeros((NY, NX), dtype=np.float32)

EXTRA = {"centre_eq": (6, 10, 0.0), "tilt_e": (6, 10, +TILT), "tilt_w": (6, 10, -TILT)}
ALL = {k: (c0, c1, None) for k, (c0, c1) in GEOM.items()}
ALL.update(EXTRA)

for name, (c0, c1, tilt) in ALL.items():
    out = os.path.join(HERE, "inputs_" + name)
    os.makedirs(out, exist_ok=True)
    topo = plateau_with_pit(c0, c1)
    south = SOUTHERN_EDGE if tilt is None else EQUATOR_SOUTHERN_EDGE
    if tilt:
        # tilt_w is built as the EXACT MIRROR of tilt_e rather than as its own formula. Tilting by -TILT
        # would leave a constant offset (100 + TILT*i vs 100 - TILT*i differ by 15*TILT) plus float32
        # rounding, and the test asserts an exact mirror -- so construct it as one.
        ii = np.arange(NX, dtype=np.float32)[None, :]
        jj = np.arange(NY, dtype=np.float32)[:, None]
        topo = topo + TILT * (ii + TILT_NS * jj)
        if tilt < 0:
            topo = np.ascontiguousarray(topo[:, ::-1])
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
            _write_tif(os.path.join(out, f"exitpath_{t}_{f}.tif"), a, CELLS_PER_DEGREE, south)
    _write_tif(os.path.join(out, "exitpath_ta_wtd.tif"),
               np.full((NY, NX), 5.0, dtype=np.float64), CELLS_PER_DEGREE, south, dtype="float64")
    for f, a in (("horizontal_ksat", np.full((NY, NX), 1e-4, dtype=np.float32)),
                 ("porosity", np.full((NY, NX), 0.25, dtype=np.float32))):
        _write_tif(os.path.join(out, f"exitpath_{f}.tif"), a, CELLS_PER_DEGREE, south)
    left = c0 - 1
    right = 14 - (c1 - 1)
    where = "lat -45" if tilt is None else "EQUATOR-centred"
    t = "flat" if not tilt else f"tilt {tilt:+.0e} m/cell"
    print(f"  {name:<11} pit cols {c0}..{c1-1}  L{left} R{right}  {where:<16} {t}")
