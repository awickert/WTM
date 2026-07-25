#!/usr/bin/env python3
"""Equilibrium accuracy: does Picard's big-step steady state match Anderson's?

At steady state d h/dt = 0, so the equilibrium solves the SAME elliptic problem
for every dt and every solver -- they should agree. This also serves as the
Picard-vs-Anderson correctness check. On the 128^2 pure-drainage fixture:
  - Picard dt=1000 yr  (~50 steps)     <- reference
  - Picard dt=100000 yr (~4 steps)     <- dt-independence check
  - Anderson dt=1 yr, increasing cycles <- converges to the same field, slowly
Compares final water-table rasters.

Prereq:  python3 make_equil.py
Usage:   python3 equilibrium_accuracy.py
"""
import os, subprocess
import numpy as np, rasterio
import paths  # noqa: F401
from paths import WTM, WORK

INP = os.path.join(WORK, "equil128_inputs")
YEAR = 31536000
env = dict(os.environ, OMP_NUM_THREADS="1")


def run(tag, dt_yr, cycles, picard):
    cfg = os.path.join(WORK, f"ea_{tag}.cfg")
    open(cfg, "w").write(
        f"run_type equilibrium\nfsm_on 0\nevap_mode 0\ninfiltration_on 0\nrunoff_ratio_on 0\n"
        f"cells_per_degree 10\nsouthern_edge -45\ndeltat {dt_yr*YEAR}\n"
        f"total_cycles {cycles}\nmaxiter 1\nfdepth_a 200\nfdepth_b 150\nfdepth_fmin 2\n"
        f"time_start t0\ntime_end t0\nsurfdatadir {INP}\nregion equil128\nsupplied_wt 0\n"
        f"textfilename {WORK}/ea_{tag}_log.txt\noutfile_prefix {WORK}/ea_{tag}_out_\ncycles_to_save 9999999\n")
    extra = ["-wtm_picard"] if picard else []
    subprocess.run(["mpiexec", "-n", "1", WTM, cfg, *extra], capture_output=True, text=True, env=env)
    tif = os.path.join(WORK, f"ea_{tag}_out_{cycles:09d}.tif")
    return rasterio.open(tif).read(1) if os.path.exists(tif) else None


runs = [("picard_dt1000", 1000, 60, True),
        ("picard_dt100000", 100000, 20, True),
        ("anderson_2k", 1, 2000, False),
        ("anderson_10k", 1, 10000, False),
        ("anderson_40k", 1, 40000, False)]
fields = {}
for tag, dt, cyc, pic in runs:
    fields[tag] = run(tag, dt, cyc, pic)
    f = fields[tag]
    print(f"{tag:>18}: " + ("MISSING" if f is None else
          f"wtd range [{f.min():.4f}, {f.max():.4f}]  mean {f.mean():.4f}"))

ref = fields["picard_dt1000"]
print("\nreference = picard_dt1000")
for tag in fields:
    if tag == "picard_dt1000" or fields[tag] is None:
        continue
    d = np.abs(fields[tag] - ref)
    print(f"  {tag:>18} vs ref:  max|diff| = {d.max():.3e}   mean|diff| = {d.mean():.3e}")
