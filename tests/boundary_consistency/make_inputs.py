#!/usr/bin/env python3
"""Fixtures for the boundary-consistency regression: land-edge Dirichlet ghost == old sea-level padding.

Two fixtures built from ONE interior terrain so their shared cells are physically identical:
  bcons    (NY x NX)         : a land-edge domain (all four edges are LAND) with a single interior ocean cell
                              (DepressionHierarchy needs one ocean cell). Run with -wtm_land_boundary dirichlet,
                              its LAND edges become Dirichlet h=0 via ghost nodes.
  bconspad ((NY+2)x(NX+2))  : the SAME interior, zero-padded with a one-cell OCEAN ring at sea level. This IS
                              the old method (1-cell sea-level padding); run with the default mask-aware
                              boundary the ocean ring imposes Dirichlet h=0 all around. The ring cells INHERIT
                              the adjacent edge's ksat/porosity so the boundary transmissivity matches the
                              ghost's surface-T exactly (the two agree to machine precision only then).

The padded grid's southern_edge is set one cell further south (in run.sh) so the shared interior cells sit at
identical latitudes (identical cell geometry). Regenerate with:  python3 make_inputs.py
"""
import numpy as np, os, sys, rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wtm_testgrid import make_transform  # noqa: E402

NX, NY = 10, 8
CPD = 100.0                    # cells_per_degree; WTM reads geometry from the geotransform (#124), not the config
OUT = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUT, exist_ok=True)

def write(region, topo, mask, ksat, poro, precip, southern_edge):
    # Geometry from the geotransform: bconspad is shifted one cell SOUTH so its interior rows sit at the SAME
    # latitudes as bcons (cos-latitude sets the cell geometry; longitude is translation-invariant). See #124.
    h, w = topo.shape
    tr = make_transform(CPD, southern_edge, h)
    def wr(name, a, dt="float32"):
        with rasterio.open(os.path.join(OUT, name), "w", driver="GTiff", height=h, width=w, count=1,
                           dtype=dt, crs="EPSG:4326", transform=tr) as d:
            d.write(a.astype(dt), 1)
    zero = np.zeros((h, w), np.float32)
    for lay, a in {"topography":topo,"slope":zero,"mask":mask,"precipitation":precip,
                   "evaporation":zero,"open_water_evaporation":zero,"winter_temperature":zero}.items():
        wr(f"{region}_ta_{lay}.tif", a); wr(f"{region}_tb_{lay}.tif", a)
    wr(f"{region}_horizontal_ksat.tif", ksat)
    wr(f"{region}_porosity.tif", poro)
    wr(f"{region}_ta_wtd.tif", np.zeros((h, w), np.float64), "float64")

# interior terrain: 50 m plateau, all land, uniform ksat/porosity/precip, with ONE interior ocean cell for DH.
topo = np.full((NY, NX), 50.0, np.float32)
mask = np.ones((NY, NX), np.float32)
oj, oi = NY // 2, NX // 2
topo[oj, oi] = 0.0; mask[oj, oi] = 0.0          # interior ocean cell (identical in both fixtures)
ksat = np.full((NY, NX), 1e-3, np.float32)
poro = np.full((NY, NX), 0.25, np.float32)
precip = np.full((NY, NX), 0.2, np.float32)
write("bcons", topo, mask, ksat, poro, precip, 0.0)

# padded: interior = the terrain above; ring = ocean (mask 0) at sea level (topo 0); ring INHERITS edge
# ksat/porosity so the padded boundary transmissivity equals the ghost's surface-T.
NYp, NXp = NY + 2, NX + 2
def pad(a, fill):  # interior=a; ring=scalar `fill`, or edge-replicated when fill=="inherit"
    b = np.empty((NYp, NXp), a.dtype); b[1:-1, 1:-1] = a
    if fill == "inherit":
        b[0, 1:-1] = a[0, :]; b[-1, 1:-1] = a[-1, :]; b[1:-1, 0] = a[:, 0]; b[1:-1, -1] = a[:, -1]
        b[0, 0] = a[0, 0]; b[0, -1] = a[0, -1]; b[-1, 0] = a[-1, 0]; b[-1, -1] = a[-1, -1]
    else:
        b[0, :] = fill; b[-1, :] = fill; b[:, 0] = fill; b[:, -1] = fill
    return b
write("bconspad", pad(topo, 0.0), pad(mask, 0.0), pad(ksat, "inherit"), pad(poro, "inherit"), pad(precip, "inherit"), -1.0 / CPD)
print("wrote", OUT, "-- bcons", (NY, NX), "bconspad", (NYp, NXp))
