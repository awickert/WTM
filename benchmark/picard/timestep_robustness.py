#!/usr/bin/env python3
"""Approach-to-equilibrium vs time step: Anderson vs semi-implicit Picard.

Each cycle = one backward-Euler step of size dt (report_interval=1) on the 128^2
pure-drainage fixture (make_equil.py). Runs many cycles and reads the
per-cycle abs water-table change from the textfile to find the step at which
equilibrium is reached (mean |dwtd|/cell < TOL). Shows Anderson's stability
ceiling (it diverges above ~1 yr) vs Picard's unconditional stability (reaches
equilibrium in a handful of steps at large dt).

Prereq:  python3 make_equil.py
Usage:   python3 timestep_robustness.py [dt_yr ...]
"""
import os, re, subprocess, sys
import paths  # noqa: F401
from paths import WTM, WORK

INP = os.path.join(WORK, "equil128_inputs")
NCELL = 128 * 128
YEAR = 31536000
TOL = 1e-3          # equilibrium: mean |dwtd| per cell < 1 mm / step
MAXCYC = 400
SNES_RE = re.compile(r"nonlinear iterations =\s*(\d+)")
KSP_RE = re.compile(r"Linear solve converged.*iterations\s+(\d+)")
DTS_YR = [int(x) for x in sys.argv[1:]] or [1, 10, 100, 1000, 10000, 100000]


def run(method, dt_yr):
    tag = f"{method}_{dt_yr}"
    log = os.path.join(WORK, f"tsr_{tag}_log.txt")
    cfg = os.path.join(WORK, f"tsr_{tag}.cfg")
    open(log, "w").close()
    open(cfg, "w").write(
        f"run_type equilibrium\nfsm_on 0\nevap_mode 0\ninfiltration_on 0\nrunoff_ratio_on 0\n"
        f"cells_per_degree 10\nsouthern_edge -45\ndeltat {dt_yr*YEAR}\n"
        f"total_cycles {MAXCYC}\nreport_interval 1\nfdepth_a 200\nfdepth_b 150\nfdepth_fmin 2\n"
        f"time_start t0\ntime_end t0\nsurfdatadir {INP}\nregion equil128\nsupplied_wt 0\n"
        f"textfilename {log}\noutfile_prefix {WORK}/tsr_{tag}_out_\nsave_nreport_interval 9999999\n")
    extra = ["-wtm_picard", "-ksp_converged_reason"] if method == "picard" else []
    env = dict(os.environ, OMP_NUM_THREADS="1")
    p = subprocess.run(["mpiexec", "-n", "1", WTM, cfg, *extra],
                       capture_output=True, text=True, env=env)
    out = p.stdout + p.stderr
    snes = [int(x) for x in SNES_RE.findall(out)]
    ksp = [int(x) for x in KSP_RE.findall(out)]
    steps_eq, last = None, (0, None)
    for ln in open(log):
        f = ln.split()
        if len(f) >= 11:
            try:
                cyc, chg = int(f[0]), float(f[4]) / NCELL
            except ValueError:
                continue
            last = (cyc, chg)
            if steps_eq is None and cyc > 0 and chg < TOL:
                steps_eq = cyc
    diverged = ("not converged" in out) or (p.returncode != 0)
    return {"diverged": diverged, "steps_eq": steps_eq, "last": last,
            "outer_max": max(snes) if snes else None,
            "inner_med": sorted(ksp)[len(ksp)//2] if ksp else None}


print(f"{'method':>9} {'dt(yr)':>8} {'conv':>5} {'steps_eq':>9} {'outer_max':>9} {'inner_med':>9} {'final_chg':>11}")
for dt in DTS_YR:
    for method in ("anderson", "picard"):
        r = run(method, dt)
        se = r["steps_eq"] if r["steps_eq"] is not None else f">{r['last'][0]}"
        fc = f"{r['last'][1]:.2e}" if r["last"][1] is not None else "-"
        print(f"{method:>9} {dt:>8} {'no' if r['diverged'] else 'yes':>5} {str(se):>9} "
              f"{str(r['outer_max']):>9} {str(r['inner_med']):>9} {fc:>11}")
    sys.stdout.flush()
