#!/usr/bin/env python3
"""Recharge temporal order, broken down by water-table depth: is the SUBSURFACE 2nd order?

BDF2-on-V is 2nd order in time for the homogeneous problem but drops to 1st order once
recharge pushes cells across the LAND SURFACE (wtd -> 0), where the storativity jumps
~4x (porosity below, 1 above) -- a moving-free-boundary nonlinearity. Production discards
surface water (Fan-style full evaporation, or FSM gathers it into lakes), so the question
that matters is: is the *subsurface* water table (wtd < 0, what we keep) 2nd order, or does
the free-boundary error diffuse in and spoil it?

Finding (clean dirs, 2026-07-27): it spoils it. Even deep subsurface cells (wtd < -10 m)
are order ~1.15 -- the error shrinks with depth but the ORDER stays ~1 domain-wide as long
as anything crosses the surface. See BDF2_RECHARGE_ORDER.md.

NOTE: this is a benchmark diagnostic, NOT a unit test -- it runs many full model solves
(minutes). It measures order by self-convergence vs a fine (dt=0.25 yr) reference.

Prereq:  python3 make_equil.py            # writes $WTM_WORK/equil128_inputs
Usage:   python3 recharge_order_by_depth.py
"""
import os, subprocess, math
import numpy as np, rasterio
import paths  # noqa: F401
from paths import WTM, WORK

INP = os.path.join(WORK, "equil128_inputs")
YEAR = 31536000
T = 1000                       # yr window
DTS = [1, 2, 5, 10, 100]       # yr; dt=0.25 is the reference
P_RECH = 0.01                  # m/yr uniform recharge (drives cells across the surface)
env = dict(os.environ, OMP_NUM_THREADS="1")
mask = rasterio.open(os.path.join(INP, "equil128_t0_mask.tif")).read(1) == 1
_pf = os.path.join(INP, "equil128_t0_precipitation.tif")
with rasterio.open(_pf) as s:
    _pprof = s.profile


def set_precip(P):
    with rasterio.open(_pf, "w", **_pprof) as d:
        d.write(np.full((128, 128), P, np.float32), 1)


def run(dt_yr, T_yr, tag, supplied_wt):
    steps = int(round(T_yr / dt_yr))
    cfg = os.path.join(WORK, f"{tag}.cfg")
    open(cfg, "w").write(
        f"run_type equilibrium\nfsm_on 0\nevap_mode 0\ninfiltration_on 0\nrunoff_ratio_on 0\n"
        f"cells_per_degree 10\nsouthern_edge -45\ndeltat {int(dt_yr*YEAR)}\n"
        f"total_cycles {steps}\nmaxiter 1\nfdepth_a 200\nfdepth_b 150\nfdepth_fmin 2\n"
        f"time_start t0\ntime_end t0\nsurfdatadir {INP}\nregion equil128\nsupplied_wt {supplied_wt}\n"
        f"textfilename {WORK}/{tag}_log.txt\noutfile_prefix {WORK}/{tag}_out_\ncycles_to_save 9999999\n")
    subprocess.run(["mpiexec", "-n", "1", WTM, cfg, "-wtm_bdf2_on_V"],
                   capture_output=True, text=True, env=env)
    tif = os.path.join(WORK, f"{tag}_out_{steps:09d}.tif")
    return rasterio.open(tif).read(1) if os.path.exists(tif) else None


# Deep smooth IC (drainage), then uniform recharge -- some cells rise back across the surface.
set_precip(0.0)
ic = run(0.25, T, "rd_gen", 0)
with rasterio.open(os.path.join(INP, "equil128_t0_topography.tif")) as t:
    prof = t.profile
prof.update(dtype="float64")
with rasterio.open(os.path.join(INP, "equil128_t0_starting_wt.tif"), "w", **prof) as d:
    d.write(ic.astype("float64"), 1)

set_precip(P_RECH)
ref = run(0.25, T, "rd_ref", 1)
fields = {dt: run(dt, T, f"rd_{dt}", 1) for dt in DTS}
set_precip(0.0)  # restore fixture

print(f"Recharge P={P_RECH} m/yr, T={T} yr; land wtd range [{ref[mask].min():.1f}, {ref[mask].max():.2f}] m")
print("Order of mean|err| by reference depth (does the subsurface stay 2nd order?):\n")
print(f"{'subset':<24}{'%cells':>7}{'err@1':>9}{'err@10':>9}{'err@100':>10}{'order':>8}")
subsets = [("all land", mask),
           ("subsurface wtd<-2 m", mask & (ref < -2)),
           ("subsurface wtd<-5 m", mask & (ref < -5)),
           ("subsurface wtd<-10 m", mask & (ref < -10)),
           ("near-surface wtd>=-2 m", mask & (ref >= -2))]
for label, sel in subsets:
    n = int(sel.sum())
    if n == 0:
        continue
    e = {dt: np.abs(fields[dt] - ref)[sel].mean() * 1000 for dt in DTS}
    order = math.log(e[100] / e[10]) / math.log(10.0)   # coarse-dt order (unfloored)
    print(f"{label:<24}{100*n/mask.sum():>6.1f}%{e[1]:>9.3g}{e[10]:>9.3g}{e[100]:>10.4g}{order:>8.2f}")
