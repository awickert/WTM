#!/usr/bin/env python3
"""Transient accuracy vs time step, against Anderson dt=1yr as ground truth.

The equilibrium is dt-independent, but backward Euler is only 1st-order in time,
so at LARGE dt the transient PATH is distorted even while the solve stays stable.
Compares the water-table field at MATCHED physical times T:
  ground truth = Anderson, dt=1yr, T steps
  test         = Picard, dt in {10,100,1000,10000}, T/dt steps
for T in {1000, 10000, 40000} yr. Deviation should scale ~O(dt) and shrink toward
equilibrium (large T) -- stability without free accuracy, first-order-controllable.

Prereq:  python3 make_equil128.py
Usage:   python3 transient_accuracy.py
"""
import os, subprocess
import numpy as np, rasterio
import paths  # noqa: F401
from paths import WTM, WORK

INP = os.path.join(WORK, "equil128_inputs")
YEAR = 31536000
env = dict(os.environ, OMP_NUM_THREADS="1")
TIMES = [1000, 10000, 40000]        # yr
PIC_DTS = [10, 100, 1000, 10000]    # yr


def run(dt_yr, cycles, picard):
    tag = f"tra_{'p' if picard else 'a'}_{dt_yr}_{cycles}"
    cfg = os.path.join(WORK, f"{tag}.cfg")
    open(cfg, "w").write(
        f"run_type equilibrium\nfsm_on 0\nevap_mode 0\ninfiltration_on 0\nrunoff_ratio_on 0\n"
        f"cells_per_degree 10\nsouthern_edge -45\ndeltat {dt_yr*YEAR}\n"
        f"total_cycles {cycles}\nmaxiter 1\nfdepth_a 200\nfdepth_b 150\nfdepth_fmin 2\n"
        f"time_start t0\ntime_end t0\nsurfdatadir {INP}\nregion equil128\nsupplied_wt 0\n"
        f"textfilename {WORK}/{tag}_log.txt\noutfile_prefix {WORK}/{tag}_out_\ncycles_to_save 9999999\n")
    extra = ["-wtm_picard"] if picard else []
    subprocess.run(["mpiexec", "-n", "1", WTM, cfg, *extra], capture_output=True, text=True, env=env)
    tif = os.path.join(WORK, f"{tag}_out_{cycles:09d}.tif")
    return rasterio.open(tif).read(1) if os.path.exists(tif) else None


gt = {T: run(1, T, False) for T in TIMES}   # Anderson dt=1 ground truth at each time
print(f"{'T(yr)':>7} {'method':>16} {'dt(yr)':>7} {'steps':>6} {'max|dev|':>10} {'mean|dev|':>10}")
for T in TIMES:
    g = gt[T]
    print(f"{T:>7} {'Anderson(truth)':>16} {1:>7} {T:>6} {'0':>10} {'0':>10}")
    for dt in PIC_DTS:
        if T % dt:
            continue
        steps = T // dt
        f = run(dt, steps, True)
        if f is None or g is None:
            print(f"{T:>7} {'Picard':>16} {dt:>7} {steps:>6} {'MISSING':>10}")
            continue
        d = np.abs(f - g)
        print(f"{T:>7} {'Picard':>16} {dt:>7} {steps:>6} {d.max():>10.3e} {d.mean():>10.3e}")
