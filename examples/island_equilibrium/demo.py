#!/usr/bin/env python3
"""Island equilibrium demo: implicit (2nd-order Picard) groundwater, serial == parallel.

Runs WTM to equilibrium on an island (ocean along every side) with the 2nd-order-in-time
implicit solver (`-wtm_bdf2_on_V`), in serial and on N MPI ranks, and shows that the result is
cross-rank consistent (identical to floating-point-reduction noise) while producing the expected
surface hydrology: lakes ponded in closed depressions, rivers draining to the coast, and the ocean
boundary.

Two topographies:
  * `spectral` -- a synthetic island (radial dome + Fourier roughness + two carved basins).
    Deterministic, self-contained.
  * `corsica`  -- a real DEM: a 240x156 window of GEBCO_08 over Corsica (bundled as
    `corsica_gebco.tif`, so no GEBCO download is needed). Steep real terrain -- note the default
    Anderson solver fails to converge here, while the implicit Picard solver does.

Usage:
    demo.py spectral [--ranks 4 8] [--map]
    demo.py corsica  [--ranks 4 8] [--map]

Requires ../../build/wtm.x, rasterio, numpy (matplotlib only for --map).
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import rasterio
from rasterio.transform import from_bounds

HERE = os.path.dirname(os.path.abspath(__file__))
WTM = os.path.abspath(os.path.join(HERE, "..", "..", "build", "wtm.x"))
# 2nd-order implicit method. -wtm_bdf2_on_V drives the semi-implicit Picard/BDF2-on-V path.
SOLVER_FLAGS = ["-wtm_bdf2_on_V", "-snes_stol", "1e-8"]


def _fields(H, W, topo, mask):
    """Uniform forcing that yields a surplus (lakes + rivers) without saturating the whole island."""
    return {
        "topography": (topo, "float32"),
        "slope": (np.zeros((H, W)), "float32"),
        "mask": (mask, "float32"),
        "precipitation": (np.full((H, W), 0.22), "float32"),        # m/yr, > evap+... : net surplus
        "evaporation": (np.full((H, W), 0.10), "float32"),
        "open_water_evaporation": (np.full((H, W), 0.30), "float32"),  # caps lakes
        "winter_temperature": (np.zeros((H, W)), "float32"),
        "horizontal_ksat": (np.full((H, W), 1e-4), "float32"),
        "porosity": (np.full((H, W), 0.25), "float32"),
        "runoff_ratio": (np.full((H, W), 0.5), "float32"),          # overland flow -> rivers
        "starting_wt": (np.full((H, W), -20.0), "float64"),
    }


def _write(d, region, H, W, topo, mask):
    tr = from_bounds(0, 0, W, H, W, H)
    for name, (arr, dt) in _fields(H, W, topo, mask).items():
        # ksat/porosity are time-independent (no _t0); the rest carry the time tag.
        fn = f"{region}_{name}.tif" if name in ("horizontal_ksat", "porosity") else f"{region}_t0_{name}.tif"
        with rasterio.open(os.path.join(d, fn), "w", driver="GTiff", height=H, width=W, count=1,
                           dtype=dt, crs="EPSG:4326", transform=tr) as o:
            o.write(np.asarray(arr).astype(dt), 1)


def make_spectral(d, N=96):
    y, x = np.mgrid[0:N, 0:N]
    cx = cy = (N - 1) / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    R = 0.40 * N
    dome = 280.0 * (1 - (r / R) ** 2)                                   # crosses sea level -> coast
    rough = np.zeros((N, N))
    for kx, ky, amp, px, py in [(3, 2, 28, 0.3, 0.7), (5, 4, 16, 1.1, 0.2), (7, 3, 11, 2.0, 1.5),
                                (2, 6, 18, 0.5, 2.3), (9, 7, 6, 1.7, 0.9)]:
        rough += amp * np.sin(2 * np.pi * kx * x / N + px) * np.sin(2 * np.pi * ky * y / N + py)
    topo = dome + rough
    for bx, by, br, dep in [(0.36, 0.44, 0.09, 70), (0.60, 0.60, 0.08, 55)]:  # carved basins -> lakes
        dd = np.sqrt((x - bx * N) ** 2 + (y - by * N) ** 2)
        topo -= dep * np.exp(-(dd / (br * N)) ** 2)
    mask = (topo > 0.0).astype("float32")
    mask[0] = mask[-1] = mask[:, 0] = mask[:, -1] = 0                    # ocean edge ring
    topo = np.where(mask > 0, np.maximum(topo, 0.5), 0.0).astype("float32")
    _write(d, "spectral", N, N, topo, mask)
    return "spectral", 10, -30.0                                        # region, cells/deg, southern_edge


def make_corsica(d):
    with rasterio.open(os.path.join(HERE, "corsica_gebco.tif")) as s:
        dem = s.read(1).astype("float32")
    H, W = dem.shape
    mask = (dem > 0).astype("float32")
    mask[0] = mask[-1] = mask[:, 0] = mask[:, -1] = 0                    # force ocean boundary
    topo = np.where(mask > 0, np.maximum(dem, 0.5), 0.0).astype("float32")
    _write(d, "corsica", H, W, topo, mask)
    return "corsica", 120, 41.2                                         # GEBCO 30" ; Corsica latitude


def cfg(d, region, cpd, south, txt, pfx):
    return (f"run_type equilibrium\nfsm_on 1\nevap_mode 1\ninfiltration_on 0\nrunoff_ratio_on 1\n"
            f"cells_per_degree {cpd}\nsouthern_edge {south}\ndeltat 31536000\ntotal_cycles 15\nmaxiter 3\n"
            f"fdepth_a 200\nfdepth_b 150\nfdepth_fmin 2\ntime_start t0\ntime_end t0\n"
            f"surfdatadir {d}\nregion {region}\nsupplied_wt 0\ncycles_to_save 9999\n"
            f"textfilename {txt}\noutfile_prefix {pfx}\n")


def run(d, region, cpd, south, n):
    c = os.path.join(d, f"cfg_n{n}")
    with open(c, "w") as f:
        f.write(cfg(d, region, cpd, south, os.path.join(d, f"out_n{n}.txt"), os.path.join(d, f"n{n}_")))
    subprocess.run(["mpirun", "-n", str(n), WTM, c] + SOLVER_FLAGS, cwd=d,
                   env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with rasterio.open(sorted(glob.glob(os.path.join(d, f"n{n}_*.tif")))[-1]) as s:
        return s.read(1)


def render_map(d, region, w_serial, w_par, npar, out):
    """Left: the feature map (ocean / rivers / lakes). Right: |serial - parallel| -- the visual
    serial==parallel proof (machine-zero)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with rasterio.open(os.path.join(d, f"{region}_t0_topography.tif")) as s:
        topo = s.read(1)
    with rasterio.open(os.path.join(d, f"{region}_t0_mask.tif")) as s:
        land = s.read(1) > 0
    NY, NX = topo.shape
    acc = np.ones_like(topo)                                            # D8 flow accumulation -> rivers
    for i in np.argsort(-np.where(land, topo, -1e9).ravel()):
        yy, xx = i // NX, i % NX
        if not land[yy, xx]:
            continue
        bz, best = topo[yy, xx], None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = yy + dy, xx + dx
                if 0 <= ny < NY and 0 <= nx < NX and (dy or dx) and topo[ny, nx] < bz:
                    bz, best = topo[ny, nx], (ny, nx)
        if best:
            acc[best] += acc[yy, xx]
    rivers = (acc > 0.008 * land.sum()) & land

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 7))
    # -- left: features --
    ax.imshow(np.where(land, topo, np.nan), cmap="terrain", vmin=0, vmax=topo.max())
    ax.imshow(np.where(~land, 0, np.nan), cmap="Blues", vmin=-1, vmax=1)     # ocean (blue)
    ax.imshow(np.where(rivers, 1, np.nan), cmap="winter", vmin=0, vmax=1)    # rivers (cyan)
    lk = np.where(land & (w_serial > 0.05), w_serial, np.nan)               # lakes (magenta)
    if np.isfinite(np.nanmax(lk)):
        ax.imshow(lk, cmap="cool", vmin=0, vmax=np.nanmax(lk))
    ax.set_title(f"{region}: equilibrium (2nd-order Picard, serial)\nterrain, ocean (blue), "
                 "rivers (cyan), lakes (magenta)")
    ax.axis("off")
    # -- right: serial vs parallel, in nanometres --
    diff = np.where(land, np.abs(w_serial - w_par) * 1e9, np.nan)           # m -> nm
    im = ax2.imshow(diff, cmap="magma")
    ax2.set_title(f"|serial - parallel(n={npar})|   (nanometres)\n"
                  f"max = {np.nanmax(diff):.2f} nm  =>  serial == parallel")
    ax2.axis("off")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="nm")
    plt.tight_layout()
    plt.savefig(out, dpi=95)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("region", choices=["spectral", "corsica"])
    ap.add_argument("--ranks", type=int, nargs="*", default=[4, 8])
    ap.add_argument("--map", action="store_true")
    a = ap.parse_args()
    if not os.access(WTM, os.X_OK):
        sys.exit(f"build wtm.x first (looked for {WTM})")
    d = os.path.join(HERE, f"_work_{a.region}")
    os.makedirs(d, exist_ok=True)
    if a.region == "spectral":
        region, cpd, south = make_spectral(d)
    else:
        region, cpd, south = make_corsica(d)
    w1 = run(d, region, cpd, south, 1)
    with rasterio.open(os.path.join(d, f"{region}_t0_mask.tif")) as s:
        land = s.read(1) > 0
    lakes = int((land & (w1 > 0.05)).sum())
    print(f"{region}: serial (n=1) done -- {lakes} lake cells (max {w1[land].max():.1f} m), "
          f"{int((~land).sum())} ocean cells")
    fail = 0
    w_par, npar = None, None
    for n in a.ranks:
        wn = run(d, region, cpd, south, n)
        md = float(np.abs(w1 - wn)[land].max())
        ok = md < 1e-6
        fail += not ok
        w_par, npar = wn, n                                             # keep the last (largest) for the map
        print(f"  serial vs n={n}: max|dwtd| = {md:.3e} m  {'CONSISTENT' if ok else 'MISMATCH'}")
    if a.map and w_par is not None:
        render_map(d, region, w1, w_par, npar, os.path.join(HERE, f"{region}_map.png"))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
