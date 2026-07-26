#!/usr/bin/env python3
"""BDF2-on-V temporal order WITH recharge: it drops from 2 to 1 (source-limited).

BDF2-on-V is 2nd order in time for the homogeneous (drainage) problem, but the
recharge SOURCE is integrated at only 1st order, so any run with recharge -- i.e.
every production run -- is 1st order in time. This script reproduces that and the
diagnosis (see benchmark/BDF2_RECHARGE_ORDER.md):

  * A/B order study, identical deep smooth initial condition (supplied_wt), only
    recharge differs: P=0 -> order ~2, P=0.01 m/yr -> order ~1.
  * BDF2-on-V vs plain backward Euler with recharge: BDF2-on-V is a constant ~2x
    more accurate, but BOTH are order 1 (the storage is BDF2, the source is not).

Diagnostics that pinned the cause (not reproduced here; see the design note):
D1 mass conserved across dt (not a scaling bug); D2 error is on deep smooth cells;
D3 smoothing the C0 kinks does NOT help (not kink-crossing). => the recharge
source's temporal integration is 1st-order. Fix: Richardson-in-time (guaranteed)
or a manufactured-solution dig to pin & fix the source term.

Prereq:  python3 make_equil.py
Usage:   python3 recharge_order.py
"""
import os, subprocess
import numpy as np, rasterio
import paths  # noqa: F401
from paths import WTM, WORK

INP = os.path.join(WORK, "equil128_inputs")
YEAR = 31536000
T = 1000                         # yr window
DTS = [1, 2, 5, 10, 100]         # yr
env = dict(os.environ, OMP_NUM_THREADS="1")
mask = rasterio.open(os.path.join(INP, "equil128_t0_mask.tif")).read(1) == 1
_pp = os.path.join(INP, "equil128_t0_precipitation.tif")
with rasterio.open(_pp) as s:
    _pprof = s.profile


def set_precip(P):
    with rasterio.open(_pp, "w", **_pprof) as d:
        d.write(np.full((128, 128), P, np.float32), 1)


def run(dt_yr, T_yr, tag, supplied_wt, extra):
    steps = int(round(T_yr / dt_yr))
    cfg = os.path.join(WORK, f"{tag}.cfg")
    open(cfg, "w").write(
        f"run_type equilibrium\nfsm_on 0\nevap_mode 0\ninfiltration_on 0\nrunoff_ratio_on 0\n"
        f"cells_per_degree 10\nsouthern_edge -45\ndeltat {int(dt_yr*YEAR)}\n"
        f"total_cycles {steps}\nmaxiter 1\nfdepth_a 200\nfdepth_b 150\nfdepth_fmin 2\n"
        f"time_start t0\ntime_end t0\nsurfdatadir {INP}\nregion equil128\nsupplied_wt {supplied_wt}\n"
        f"textfilename {WORK}/{tag}_log.txt\noutfile_prefix {WORK}/{tag}_out_\ncycles_to_save 9999999\n")
    subprocess.run(["mpiexec", "-n", "1", WTM, cfg, *extra], capture_output=True, text=True, env=env)
    tif = os.path.join(WORK, f"{tag}_out_{steps:09d}.tif")
    return rasterio.open(tif).read(1) if os.path.exists(tif) else None


# Common deep smooth IC: drainage (P=0), dt=0.25, T yr, from the cold start.
set_precip(0.0)
smooth = run(0.25, T, "ro_genIC", 0, ["-wtm_bdf2_on_V"])
with rasterio.open(os.path.join(INP, "equil128_t0_topography.tif")) as t:
    prof = t.profile
prof.update(dtype="float64")
with rasterio.open(os.path.join(INP, "equil128_t0_starting_wt.tif"), "w", **prof) as d:
    d.write(smooth.astype("float64"), 1)

# A/B: order vs recharge, same deep smooth IC, vs a dt=0.25 reference.
for P in [0.0, 0.01]:
    set_precip(P)
    ref = run(0.25, T, f"ro_ref_{P}", 1, ["-wtm_bdf2_on_V"])
    print(f"\n=== BDF2-on-V order, recharge P={P} m/yr (deep smooth IC, ref dt=0.25) ===")
    print(f"{'dt(yr)':>7}{'mean|err|_mm':>14}{'order':>8}")
    prev = prevdt = None
    for dt in DTS:
        f = run(dt, T, f"ro_{P}_{dt}", 1, ["-wtm_bdf2_on_V"])
        m = np.abs(f - ref)[mask].mean()
        o = f"{np.log(m/prev)/np.log(dt/prevdt):.2f}" if prev else ""
        print(f"{dt:>7}{m*1000:>14.4g}{o:>8}")
        prev, prevdt = m, dt

# BDF2-on-V vs backward Euler, recharge on, vs the dt=0.25 BDF2-on-V reference.
set_precip(0.01)
ref = run(0.25, T, "ro_cmp_ref", 1, ["-wtm_bdf2_on_V"])
print(f"\n=== recharge on (P=0.01): BDF2-on-V vs backward Euler, vs dt=0.25 ref ===")
print(f"{'dt(yr)':>7}{'BE (mm)':>12}{'BDF2-on-V (mm)':>16}{'V/BE':>7}")
for dt in [1, 10, 100]:
    be = run(dt, T, f"ro_be_{dt}", 1, ["-wtm_picard"])
    v = run(dt, T, f"ro_v_{dt}", 1, ["-wtm_bdf2_on_V"])
    ebe = np.abs(be - ref)[mask].mean() * 1000
    ev = np.abs(v - ref)[mask].mean() * 1000
    print(f"{dt:>7}{ebe:>12.4g}{ev:>16.4g}{ev/ebe:>7.2f}")

set_precip(0.0)  # restore the fixture to zero forcing
