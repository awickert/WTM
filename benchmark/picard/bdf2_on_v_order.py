#!/usr/bin/env python3
"""Temporal order of BDF2-on-V (-wtm_bdf2_on_V): TRUE 2nd order in time.

Measures the real time-discretization error of the genuine 2nd-order scheme on
the 128^2 drainage fixture, at Δt = 1..100 yr against a fine (Δt=0.25 yr)
reference at MATCHED physical time, over land cells. Runs TWO initial conditions:

  * COLD START  -- every land cell begins exactly at the surface (wtd=0) with a
    discontinuous head jump to the ocean ring: a t=0 parabolic singularity. It
    injects an O(Δt) STARTUP error that dominates the O(Δt^2) truncation once
    Δt <~ 5 yr, so the *measured* order crosses over from ~2 (coarse) to ~1
    (fine). This is a property of the singular start, NOT of the scheme -- ruled
    out (each verified): solver tolerance (snes_atol 1e-6 vs 1e-10, identical),
    the C0 transmissivity kinks (smoothing them 100x changes nothing), and
    float32 output (the field is float64).

  * SMOOTH START -- restart (supplied_wt) from a resolved state (the Δt=0.25 run
    at T=1000 yr), so there is no t=0 singularity. The order is then CLEAN ~2
    across the whole 1..100 yr range: BDF2-on-V is genuinely 2nd order in time.

Practical read: production transients that continue from a spun-up equilibrium
start smooth -> genuine order 2. A cold flat start pays a one-time O(Δt) startup
penalty capping fine-Δt convergence at ~order 1, but only at sub-0.02 mm error.

Prereq:  python3 make_equil.py            # writes $WTM_WORK/equil128_inputs
Usage:   python3 bdf2_on_v_order.py
"""
import glob, os, subprocess
import numpy as np, rasterio
import paths  # noqa: F401
from paths import WTM, WORK

INP = os.path.join(WORK, "equil128_inputs")
YEAR = 31536000
REF_DT = 0.25                          # yr; fine reference (finer than every test Δt)
DTS = [1, 2, 5, 10, 100]               # yr; test steps (all divide T)
T = 1000                               # yr; matched physical window
env = dict(os.environ, OMP_NUM_THREADS="1")
mask = rasterio.open(os.path.join(INP, "equil128_t0_mask.tif")).read(1) == 1


def run(dt_yr, T_yr, tag, supplied_wt):
    steps = int(round(T_yr / dt_yr))
    cfg = os.path.join(WORK, f"{tag}.cfg")
    open(cfg, "w").write(
        f"run_type equilibrium\nfsm_on 0\nevap_mode 0\ninfiltration_on 0\nrunoff_ratio_on 0\n"
        f"cells_per_degree 10\nsouthern_edge -45\ndeltat {int(dt_yr*YEAR)}\n"
        f"total_cycles {steps}\nreport_interval 1\nfdepth_a 200\nfdepth_b 150\nfdepth_fmin 2\n"
        f"time_start t0\ntime_end t0\nsurfdatadir {INP}\nregion equil128\nsupplied_wt {supplied_wt}\n"
        f"textfilename {WORK}/{tag}_log.txt\noutfile_prefix {WORK}/{tag}_out_\nsave_nreport_interval 9999999\n")
    subprocess.run(["mpiexec", "-n", "1", WTM, cfg, "-wtm_bdf2_on_V"],
                   capture_output=True, text=True, env=env)
    tifs = sorted(glob.glob(os.path.join(WORK, f"{tag}_out_*.tif")))  # output name now carries a _<yr>yr suffix; take the final
    return rasterio.open(tifs[-1]).read(1) if tifs else None


def order_table(label, supplied_wt):
    ref = run(REF_DT, T, f"ov_ref_{label}", supplied_wt)
    print(f"\n=== BDF2-on-V order, {label} start (window {T} yr, ref Δt={REF_DT} yr) ===")
    print(f"{'dt(yr)':>7}{'mean|err|_mm':>14}{'max|err|_mm':>13}{'order':>8}")
    prev = prevdt = None
    for dt in DTS:
        f = run(dt, T, f"ov_{label}_{dt}", supplied_wt)
        e = np.abs(f - ref)[mask]
        m = e.mean()
        o = f"{np.log(m/prev)/np.log(dt/prevdt):.2f}" if prev else ""
        print(f"{dt:>7}{m*1000:>14.4g}{e.max()*1000:>13.4g}{o:>8}")
        prev, prevdt = m, dt


# COLD START (singular IC): the measured order crosses over ~2 -> ~1 at fine Δt.
order_table("cold", 0)

# Build a smooth resolved IC (Δt=0.25 at T=1000 from the cold start) and write it as starting_wt.
smooth = run(REF_DT, T, "ov_genIC", 0)
with rasterio.open(os.path.join(INP, "equil128_t0_topography.tif")) as t:
    prof = t.profile
prof.update(dtype="float64")
with rasterio.open(os.path.join(INP, "equil128_t0_starting_wt.tif"), "w", **prof) as d:
    d.write(smooth.astype("float64"), 1)

# SMOOTH START (supplied_wt): clean ~2nd order across the whole range.
order_table("smooth", 1)
