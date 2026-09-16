#!/usr/bin/env python3
"""Island equilibrium demo: implicit (2nd-order Picard) groundwater, serial == parallel.

Runs WTM to equilibrium on an island (ocean along every side) with the 2nd-order-in-time
implicit solver (`solver.method: picard` + `solver.time_integration: bdf2`, the BDF2-on-V path),
in serial and on N MPI ranks, and shows that the result is cross-rank consistent (identical to
floating-point-reduction noise) while producing the expected surface hydrology: lakes ponded in
closed depressions, rivers draining to the coast, and the ocean boundary.

MEASURED 2026-09-17, on the repaired demo:
    spectral   n=4: max|dwtd| = 5.684e-13 m      n=8: 7.958e-13 m    (37 lake cells, max 8.3 m)
    corsica    n=4: max|dwtd| = 1.206e-09 m      n=8: 1.720e-09 m    (218 lake cells, max 79.0 m)
both against a 1e-6 m threshold.

Two topographies:
  * `spectral` -- a synthetic island (radial dome + Fourier roughness + two carved basins).
    Deterministic, self-contained.
  * `corsica`  -- a real DEM: a 240x156 window of GEBCO_08 over Corsica (bundled as
    `corsica_gebco.tif`, so no GEBCO download is needed). Steep real terrain -- note the default
    Anderson solver fails to converge here, while the implicit Picard solver does. (That observation
    predates the current default set -- anderson / tr-bdf2 / active_set / adaptive -- and has not
    been re-measured against it.)

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

HERE = os.path.dirname(os.path.abspath(__file__))
WTM = os.path.abspath(os.path.join(HERE, "..", "..", "build", "wtm.x"))

# The ONE shared geotransform writer (tests/wtm_testgrid.py). WTM derives its grid geometry -- degree
# spacing, southern-edge latitude, and the cos-latitude cell-size scaling -- from each input raster's
# GDAL geotransform (#124). This demo used to stamp `from_bounds(0, 0, W, H, W, H)`, i.e. ONE DEGREE
# PER CELL starting at the equator, from the era when WTM ignored georeferencing entirely. That is
# 111 km cells against the 11 km (spectral) and 0.9 km (corsica) the demo intends, so the run was
# solving a different problem than the one it describes. Routed through the shared helper rather than
# hand-rolled here, for the reason that file gives: a hand-rolled transform drifts from its grid.
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "tests")))
from wtm_testgrid import write_tif  # noqa: E402


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


def _write(d, region, H, W, topo, mask, cpd, south):
    """Write the input rasters with the geotransform the demo actually intends (cpd, south)."""
    for name, (arr, dt) in _fields(H, W, topo, mask).items():
        # ksat/porosity are time-independent (no _t0); the rest carry the time tag.
        fn = f"{region}_{name}.tif" if name in ("horizontal_ksat", "porosity") else f"{region}_t0_{name}.tif"
        write_tif(os.path.join(d, fn), np.asarray(arr), cpd, south, dtype=dt)


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
    _write(d, "spectral", N, N, topo, mask, 10, -30.0)
    return "spectral", 10, -30.0                                        # region, cells/deg, southern_edge


def make_corsica(d):
    with rasterio.open(os.path.join(HERE, "corsica_gebco.tif")) as s:
        dem = s.read(1).astype("float32")
    H, W = dem.shape
    mask = (dem > 0).astype("float32")
    mask[0] = mask[-1] = mask[:, 0] = mask[:, -1] = 0                    # force ocean boundary
    topo = np.where(mask > 0, np.maximum(dem, 0.5), 0.0).astype("float32")
    _write(d, "corsica", H, W, topo, mask, 120, 41.2)
    return "corsica", 120, 41.2                                         # GEBCO 30" ; Corsica latitude


def cfg(d, region, txt, pfx, outdir):
    """The run's configuration, in the nested-YAML schema (see config.yaml at the repo root).

    EVERY setting this run resolves to is stated here. That is the house rule for configs in this
    repo, and it is what makes a run reproducible from the file alone rather than from knowing which
    defaults applied on the day. Four of them are LOAD-BEARING for what this demo claims:

      solver.method: picard + time_integration: bdf2
          The 2nd-order-in-time BDF2-on-V path. This is what the demo has always run (it asked for it
          as the retired flag `-wtm_bdf2_on_V`), and it is the cross-rank-deterministic solver, which
          is the property the demo exists to show.
      time_step.mode: fixed
          The legacy behaviour: dt is exactly `dt` below. Picard would otherwise resolve to
          `adaptive`. Held fixed so the serial and parallel runs take the SAME steps, and any
          difference between them is the thing being measured rather than a different step sequence.
      equilibrium_stop.tol: 0
          Run the clock out; never stop early. A determinism comparison that is free to auto-stop can
          have its two runs halt at different cycles and then differ for a reason that is not a
          defect -- exactly the trap #95 caught in the taper study.
      surface_water.routing: impulse
          FSM's routed table replaces the step baseline -- what the model has always done, and what
          v2.0.1 does. `continuous` is REFUSED with the `explicit` collector that Picard resolves to.
    """
    return f"""run:
  type: equilibrium
  initial_water_table: saturated   # was `supplied_wt 0`
  equilibrium_stop:
    tol: 0            # LOAD-BEARING: never auto-stop -- see the docstring
    metric: frac      # INERT with tol 0
    frac: 0.001       # INERT, as above
time:
  total: "15yr"       # was `total_cycles 15` at a 1 yr step
  report_interval: 1
  save_every_n_reports: 1
io:
  source: '{d}'
  region: '{region}'
  time_start: 't0'
  time_end: 't0'
output:
  directory: '{outdir}'
  outfile_prefix: '{pfx}'
  run_log: '{txt}'
  verbosity: normal
  if_exists: overwrite
boundaries:
  land: neumann_toposlope
transmissivity:
  fdepth:
    a: 200
    b: 150
    fmin: 2
  additive_background_transmissivity: 0
evaporation:
  et_sigmoid:
    wtd_center: 0.05
    logistic_width: 0.1
  extinction_depth: 8
  tapers:
    surface_transition: true
    depth_extinction: true
surface_water:
  routing: impulse    # LOAD-BEARING -- see the docstring
  runoff_ratio: raster   # was `runoff_ratio_on 1`: use the runoff_ratio layer this demo writes
  infiltration_during_flow: false
  collection:
    method: explicit  # what Picard resolves to; active_set is refused on the Picard path
solver:
  method: picard
  time_integration: bdf2   # BDF2-on-V, the 2nd-order path. Was `-wtm_bdf2_on_V`
  convergence:
    metric: volume         # the shipped default (#61): the per-solve step judged in WATER
    water_volume_tol: 1.0e-8
  smoothing:
    ksat_surface: 0
    ksat_soilbottom: 0
    storativity_surface: 0.01
  time_step:
    mode: fixed       # LOAD-BEARING -- see the docstring
    dt: 31536000      # 1 year; was `deltat`
parallel:
  threads_per_rank: 1
"""


def run(d, region, cpd, south, n):
    c = os.path.join(d, f"cfg_n{n}.yaml")
    with open(c, "w") as f:
        # Absolute paths, as the test suites do: `directory` is the run's provenance dir, while the
        # raster prefix and the log are written where this script then looks for them.
        f.write(cfg(d, region, os.path.join(d, f"out_n{n}.txt"),
                    os.path.join(d, f"n{n}_"), os.path.join(d, f"prov_n{n}")))
    subprocess.run(["mpirun", "-n", str(n), WTM, c], cwd=d,
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
