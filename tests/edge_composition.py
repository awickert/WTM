#!/usr/bin/env python3
"""WHICH BOUNDARY CONDITION does this fixture actually exercise?

THE FAILURE THIS PREVENTS. #105 -- a mass leak of 45x the domain's recharge, in the SHIPPED DEFAULT land
boundary -- lived because that question had no answer anywhere in the suite. Finding it took reading the
ghost-node construction, three fixtures and a three-point experiment, and the decisive fact was a
property of the FIXTURES nobody had written down: every budget fixture is ocean-ringed, so its land
boundary is never in play, so no budget test could ever see a land-boundary defect.

A fixture's mask decides this, not its config. `boundaries.land: neumann_toposlope` in a config means
nothing if every domain-edge cell is ocean -- the ghost node is never consulted. So the answer has to be
read off the mask, and printed, by every suite that runs.

WHAT IT REPORTS, per fixture:
  ocean edge cells   domain-edge cells with mask == 0 -> ALWAYS Dirichlet h=0, not selectable
  land  edge cells   domain-edge cells with mask != 0 -> the off-map ghost, boundaries.land applies
  sloping land edge  land edge cells whose inward neighbour differs in topography. THE ONE THAT MATTERS:
                     neumann_toposlope sets h_ghost = h_edge + (topo_edge - topo_inland), so its flux is
                     ZERO on a flat edge and non-zero on a sloping one. A fixture with land edges that
                     are all flat exercises the Neumann BC without exercising its flux.
"""
import sys, os, glob


def compose(mask_path, topo_path=None):
    import numpy as np, rasterio
    m = rasterio.open(mask_path).read(1)
    t = rasterio.open(topo_path).read(1).astype(float) if topo_path and os.path.exists(topo_path) else None
    ny, nx = m.shape
    edge, sloping = [], 0
    for j in range(ny):
        for i in range(nx):
            on_edge = (j == 0 or j == ny - 1 or i == 0 or i == nx - 1)
            if not on_edge:
                continue
            edge.append((j, i, m[j, i]))
            if m[j, i] != 0 and t is not None:
                # inward neighbour, for the edge this cell sits on
                jj = j + (1 if j == 0 else -1 if j == ny - 1 else 0)
                ii = i + (1 if i == 0 else -1 if i == nx - 1 else 0)
                if 0 <= jj < ny and 0 <= ii < nx and abs(float(t[j, i]) - float(t[jj, ii])) > 0.0:
                    sloping += 1
    ocean = sum(1 for _, _, v in edge if v == 0)
    land = len(edge) - ocean
    return len(edge), ocean, land, sloping


def main():
    if len(sys.argv) < 2:
        print("usage: edge_composition.py <inputs-dir>...", file=sys.stderr)
        return 2
    for d in sys.argv[1:]:
        masks = sorted(glob.glob(os.path.join(d, "*_ta_mask.tif")))
        if not masks:
            continue
        for mk in masks:
            topo = mk.replace("_mask.tif", "_topography.tif")
            try:
                n, ocean, land, sloping = compose(mk, topo)
            except Exception as e:
                print(f"  note  EDGES     {os.path.basename(mk)}: could not read ({e})")
                continue
            region = os.path.basename(mk).split("_ta_")[0]
            if land == 0:
                verdict = ("OCEAN-RINGED: every domain-edge cell is ocean, so boundaries.land is NEVER "
                           "exercised here -- a land-boundary defect cannot show up in this fixture")
            elif sloping == 0:
                verdict = (f"{land} land edge cells, ALL FLAT: the Neumann ghost is consulted but its flux "
                           "is zero by construction (h_ghost - h_edge = 0), so its FLUX is not exercised")
            else:
                verdict = (f"{land} land edge cells, {sloping} of them SLOPING: the Neumann off-map flux "
                           "IS exercised here (h_ghost - h_edge = the terrain rise)")
            print(f"  note  EDGES     {region}: {n} edge cells, {ocean} ocean / {land} land. {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
