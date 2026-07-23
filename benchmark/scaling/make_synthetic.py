#!/usr/bin/env python3
"""Generate a synthetic WTM `run_type test` input pair at an arbitrary grid size.

`run_type test` (src/irf.cpp: InitialiseTest) reads exactly two rasters --
topography and slope -- and synthesizes every other field (ksat, porosity,
precip, mask, ...) in code. So a scaling study needs only these two files at the
desired dimensions; no production data is required.

The topography is a smooth, deterministic multi-scale field (dome + sinusoids)
so the groundwater solve sees real head gradients and does representative work,
rather than the trivial solve a flat surface would give. It is well out of cache
at the sizes we sweep, so the strong-scaling timings are not skewed by an
artificially cache-resident grid. Slope is the true gradient magnitude of that
topography (units 0..1), which feeds fdepth.

Usage:
    make_synthetic.py NX [NY] [--outdir DIR] [--region NAME]

NY defaults to NX (square grid). Writes {outdir}/{region}_topography.tif and
{outdir}/{region}_slope.tif (float32).
"""
import argparse
import os

import numpy as np
import rasterio
from rasterio.transform import from_bounds


def build_topography(nx, ny):
    """Smooth multi-scale relief, ~20..~600 m, deterministic (no RNG)."""
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys)  # shape (ny, nx)
    dome = 120.0 * (1.0 - ((X - 0.5) ** 2 + (Y - 0.5) ** 2))
    topo = (
        250.0
        + 150.0 * np.sin(3.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
        + 80.0 * np.sin(7.0 * np.pi * X + 1.0) * np.sin(5.0 * np.pi * Y)
        + dome
    )
    # Keep everything positive land; the model zeros the ocean edge itself.
    topo = np.clip(topo, 20.0, None)
    return topo.astype(np.float32)


def slope_from_topo(topo):
    """Gradient magnitude of the topography as a dimensionless slope (0..1)."""
    gy, gx = np.gradient(topo.astype(np.float64))
    slope = np.hypot(gx, gy)
    # Normalize into a physically reasonable 0..~0.1 band for fdepth.
    smax = slope.max()
    if smax > 0:
        slope = 0.1 * slope / smax
    return slope.astype(np.float32)


def write_tif(path, data):
    ny, nx = data.shape
    transform = from_bounds(0, 0, nx, ny, nx, ny)
    with rasterio.open(
        path, "w", driver="GTiff", height=ny, width=nx, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nx", type=int, help="grid width in cells")
    ap.add_argument("ny", type=int, nargs="?", default=None,
                    help="grid height in cells (default: = nx)")
    ap.add_argument("--outdir", default=".", help="output directory (default: .)")
    ap.add_argument("--region", default="synth", help="region name (default: synth)")
    args = ap.parse_args()

    nx = args.nx
    ny = args.ny if args.ny is not None else nx
    os.makedirs(args.outdir, exist_ok=True)

    topo = build_topography(nx, ny)
    slope = slope_from_topo(topo)

    topo_path = os.path.join(args.outdir, f"{args.region}_topography.tif")
    slope_path = os.path.join(args.outdir, f"{args.region}_slope.tif")
    write_tif(topo_path, topo)
    write_tif(slope_path, slope)

    cells = nx * ny
    mb = cells * 4 / 1e6
    print(f"  {nx} x {ny} = {cells:,} cells  (~{mb:.1f} MB per float32 raster)")
    print(f"  wrote {topo_path}")
    print(f"  wrote {slope_path}")


if __name__ == "__main__":
    main()
