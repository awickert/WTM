#!/usr/bin/env python3
"""Verify the temporal order of the BDF2 path (-wtm_bdf2) by self-convergence.

At a fixed physical time T, run the 128^2 drainage fixture at a sequence of
halving time steps and measure the change between successive resolutions
(Richardson). For a p-th order method error ~ dt^p, so halving dt cuts the error
by 2^p: the ratio err(2dt)/err(dt) -> 2 for backward Euler (p=1) and -> 4 for
BDF2 (p=2). Reference = the finest run; no external ground truth needed.

Prereq:  python3 make_equil128.py
Usage:   python3 bdf2_order.py
"""
import os, subprocess
import numpy as np, rasterio
import paths  # noqa: F401
from paths import WTM, WORK

INP = os.path.join(WORK, "equil128_inputs")
YEAR = 31536000
T = 8000                              # yr; divisible by all dts below
DTS = [2000, 1000, 500, 250, 125]     # finest (125) is the reference
env = dict(os.environ, OMP_NUM_THREADS="1")


def field(dt_yr, bdf2):
    steps = T // dt_yr
    tag = f"ord_{'bdf2' if bdf2 else 'be'}_{dt_yr}"
    cfg = os.path.join(WORK, f"{tag}.cfg")
    open(cfg, "w").write(
        f"run_type equilibrium\nfsm_on 0\nevap_mode 0\ninfiltration_on 0\nrunoff_ratio_on 0\n"
        f"cells_per_degree 10\nsouthern_edge -45\ndeltat {dt_yr*YEAR}\n"
        f"total_cycles {steps}\nmaxiter 1\nfdepth_a 200\nfdepth_b 150\nfdepth_fmin 2\n"
        f"time_start t0\ntime_end t0\nsurfdatadir {INP}\nregion equil128\nsupplied_wt 0\n"
        f"textfilename {WORK}/{tag}_log.txt\noutfile_prefix {WORK}/{tag}_out_\ncycles_to_save 9999999\n")
    extra = ["-wtm_picard"] + (["-wtm_bdf2"] if bdf2 else [])
    subprocess.run(["mpiexec", "-n", "1", WTM, cfg, *extra], capture_output=True, text=True, env=env)
    tif = os.path.join(WORK, f"{tag}_out_{steps:09d}.tif")
    return rasterio.open(tif).read(1) if os.path.exists(tif) else None


for label, bdf2 in [("backward Euler", False), ("BDF2", True)]:
    fields = {dt: field(dt, bdf2) for dt in DTS}
    ref = fields[DTS[-1]]
    print(f"\n=== {label} (self-convergence vs dt={DTS[-1]}yr reference, T={T}yr) ===")
    print(f"{'dt(yr)':>7} {'steps':>6} {'mean|err|':>11} {'ratio':>7} {'-> order':>9}")
    prev = None
    for dt in DTS[:-1]:
        e = float(np.abs(fields[dt] - ref).mean())
        ratio = (prev / e) if prev else None
        order = (np.log(prev / e) / np.log(2.0)) if prev else None
        print(f"{dt:>7} {T//dt:>6} {e:>11.3e} "
              f"{('%.2f' % ratio) if ratio else '-':>7} {('%.2f' % order) if order else '-':>9}")
        prev = e
