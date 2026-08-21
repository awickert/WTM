#!/usr/bin/env python3
"""Core + grid sweep: matrix-free Anderson vs semi-implicit Picard.

For grids 64..1024 and rank counts 1..8, times the GW solve (run_type test, one
cycle, `report_interval` GW solves) and records outer (SNES) and inner (CG+GAMG)
iteration counts. Reveals: inner CG flat vs grid (GAMG works), Anderson outer
count growing with grid, Picard's higher per-solve cost but better strong
scaling. Uses the synthetic topography from benchmark/scaling.

Usage:  python3 core_grid_sweep.py                 # default grids/ranks
        python3 core_grid_sweep.py 128 256 512     # custom grids
"""
import os, re, subprocess, sys, time
import paths  # noqa: F401  (sets sys.path, WORK, WTM)
from paths import WTM, SCALING, WORK
import scaling_study as ss

GRIDS = [int(x) for x in sys.argv[1:]] or [64, 128, 256, 512, 1024]
RANKS = [1, 2, 4, 8]
MAXITER = 10
GW_RE   = re.compile(r"GW time =\s*(\d+(?:\.\d+)?)(?=\s|$)")
SNES_RE = re.compile(r"nonlinear iterations =\s*(\d+)")
KSP_RE  = re.compile(r"Linear solve converged.*iterations\s+(\d+)")
SYNTH = os.path.join(SCALING, "synth")
os.makedirs(SYNTH, exist_ok=True)


def cfg(grid, prefix):
    sdir = ss.ensure_inputs(grid, SYNTH)
    p = os.path.join(WORK, f"sweep_{prefix}.cfg")
    open(p, "w").write(
        f"run_type test\nfsm_on 0\nevap_mode 0\ninfiltration_on 0\nrunoff_ratio_on 0\n"
        f"cells_per_degree {grid/120.0:.6f}\nsouthern_edge -45\ndeltat 31536000\n"
        f"total_cycles 1\nreport_interval {MAXITER}\nfdepth_a 200\nfdepth_b 150\nfdepth_fmin 2\n"
        f"time_start t0\ntime_end t0\nsurfdatadir {sdir}\nregion synth\nsupplied_wt 0\n"
        f"textfilename {WORK}/sweep_{prefix}_log.txt\noutfile_prefix {WORK}/sweep_out_{prefix}_\n"
        f"save_nreport_interval 9999\n")
    return p


def run(grid, n, picard):
    prefix = f"{'pic' if picard else 'and'}_{grid}_{n}"
    c = cfg(grid, prefix)
    extra = ["-wtm_picard", "-ksp_converged_reason"] if picard else []
    env = dict(os.environ, OMP_NUM_THREADS="1")
    best = None
    for _ in range(3):  # best-of-3 (shared-machine noise)
        t0 = time.time()
        p = subprocess.run(["mpiexec", "-n", str(n), WTM, c, *extra],
                           capture_output=True, text=True, env=env)
        wall = time.time() - t0
        if best is None or wall < best[0]:
            best = (wall, p.stdout + p.stderr, p.returncode)
    wall, out, rc = best
    gw = max((float(x) for x in GW_RE.findall(out)), default=None)
    snes = [int(x) for x in SNES_RE.findall(out)]
    ksp = [int(x) for x in KSP_RE.findall(out)]
    return {"rc": rc, "gw": gw,
            "outer": (sum(snes)/len(snes)) if snes else None,
            "inner": (sum(ksp)/len(ksp)) if ksp else None}


print(f"{'grid':>5} {'method':>8} {'n':>2} {'GWtime':>8} {'outer_avg':>9} {'inner_avg':>9} rc")
res = {}
for grid in GRIDS:
    for picard in (False, True):
        for n in RANKS:
            r = run(grid, n, picard)
            res[(grid, "picard" if picard else "anderson", n)] = r
            g = f"{r['gw']:.3f}" if r["gw"] is not None else "-"
            oa = f"{r['outer']:.1f}" if r["outer"] is not None else "-"
            ia = f"{r['inner']:.1f}" if r["inner"] is not None else "-"
            print(f"{grid:>5} {'picard' if picard else 'anderson':>8} {n:>2} {g:>8} {oa:>9} {ia:>9} {r['rc']}")
        sys.stdout.flush()

print("\n=== strong scaling (GW-time speedup vs n=1) ===")
print(f"{'grid':>5} {'method':>8} " + " ".join(f"n={n:<5}" for n in RANKS))
for grid in GRIDS:
    for m in ("anderson", "picard"):
        base = res[(grid, m, 1)]["gw"]
        row = [f"{base/res[(grid,m,n)]['gw']:5.2f}x" if (base and res[(grid,m,n)]['gw']) else "  -  "
               for n in RANKS]
        print(f"{grid:>5} {m:>8} " + " ".join(f"{c:<7}" for c in row))
