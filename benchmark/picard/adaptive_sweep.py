#!/usr/bin/env python3
"""Adaptive BDF2 across grid size, core count, and tolerance.

On the fixed-domain drainage fixture (make_equil.py, mesh-refined so the transient
timescale -- and thus the ideal step count -- is grid-independent), runs
-wtm_dt_adaptive over a target window (report_interval*deltat) and records:
  * steps taken (vs the fixed report_interval) and GW wall time, over grids x cores;
  * whether the step count is INVARIANT to core count (the controller reduces the
    error estimate with MPI_Allreduce, so decisions must be rank-consistent);
  * step count vs tolerance and vs grid (should be ~grid-independent);
  * accuracy vs a fixed-BDF2 reference (report_interval steps), per grid at n=1.

Prereq: build wtm.x; rasterio available. Fixtures are generated here.
Usage:  python3 adaptive_sweep.py
"""
import glob, os, re, subprocess, sys, time
import numpy as np, rasterio
import paths  # noqa: F401
from paths import WTM, WORK

GRIDS = [64, 128, 256, 512, 1024]
RANKS = [1, 2, 4, 8]
TOLS = [1.0, 10.0]
YEAR = 31536000
BASE_DT_YR = 100
MAXITER = 100            # target window = 100 * 100 yr = 10000 yr
STEP_RE = re.compile(r"adaptive dt:\s*(\d+)\s*steps")
GW_RE = re.compile(r"GW time =\s*(\d+(?:\.\d+)?)(?=\s|$)")
env = dict(os.environ, OMP_NUM_THREADS="1")


def fixture(grid):
    inp = os.path.join(WORK, f"equil{grid}_inputs")
    if not os.path.exists(os.path.join(inp, f"equil{grid}_t0_topography.tif")):
        subprocess.run([sys.executable, os.path.join(paths.HERE, "make_equil.py"), str(grid)],
                       capture_output=True, env=env)
    return inp, grid / 12.8


def cfg(grid, tag):
    inp, cpd = fixture(grid)
    p = os.path.join(WORK, f"asw_{tag}.cfg")
    open(p, "w").write(
        f"run_type equilibrium\nfsm_on 0\nevap_mode 0\ninfiltration_on 0\nrunoff_ratio_on 0\n"
        f"cells_per_degree {cpd:.6f}\nsouthern_edge -45\ndeltat {BASE_DT_YR*YEAR}\n"
        f"total_cycles 1\nreport_interval {MAXITER}\nfdepth_a 200\nfdepth_b 150\nfdepth_fmin 2\n"
        f"time_start t0\ntime_end t0\nsurfdatadir {inp}\nregion equil{grid}\nsupplied_wt 0\n"
        f"textfilename {WORK}/asw_{tag}_log.txt\noutfile_prefix {WORK}/asw_{tag}_out_\nsave_nreport_interval 9999999\n")
    return p


def run(grid, n, tol, fixed=False):
    tag = f"{'fix' if fixed else f't{tol}'}_{grid}_{n}"
    c = cfg(grid, tag)
    extra = ["-wtm_bdf2"] if fixed else ["-wtm_dt_adaptive", "-wtm_dt_tol", str(tol)]
    t0 = time.time()
    p = subprocess.run(["mpiexec", "-n", str(n), WTM, c, *extra], capture_output=True, text=True, env=env)
    wall = time.time() - t0
    out = p.stdout + p.stderr
    steps = int(STEP_RE.search(out).group(1)) if STEP_RE.search(out) else (MAXITER if fixed else None)
    gw = max((float(x) for x in GW_RE.findall(out)), default=None)
    # total_cycles is 1 for both paths (the report_interval/adaptive loop is inside one cycle), so the
    # final field is saved at cycles_done = 1 -> out_000000001.tif (NOT the step count).
    tifs = sorted(glob.glob(os.path.join(WORK, f"asw_{tag}_out_*.tif")))  # output name now carries a _<yr>yr suffix; take the final
    field = rasterio.open(tifs[-1]).read(1) if tifs else None
    return {"rc": p.returncode, "wall": wall, "gw": gw, "steps": steps, "field": field}


# --- grids x cores at tol=1.0 ---
print("=== adaptive (tol=1.0 m): grids x cores ===")
print(f"{'grid':>5} {'n':>2} {'steps':>6} {'GWtime':>8} rc")
main = {}
for grid in GRIDS:
    for n in RANKS:
        r = run(grid, n, 1.0)
        main[(grid, n)] = r
        print(f"{grid:>5} {n:>2} {str(r['steps']):>6} {(f'{r['gw']:.3f}' if r['gw'] else '-'):>8} {r['rc']}")
    sys.stdout.flush()

print("\n=== step count vs cores (must be identical per grid: MPI-consistent controller) ===")
for grid in GRIDS:
    s = [main[(grid, n)]['steps'] for n in RANKS]
    print(f"{grid:>5}: " + " ".join(f"n{n}={main[(grid,n)]['steps']}" for n in RANKS) +
          ("   OK" if len(set(s)) == 1 else "   <<< MISMATCH"))

print("\n=== strong scaling (GW-time speedup vs n=1) ===")
for grid in GRIDS:
    b = main[(grid, 1)]['gw']
    print(f"{grid:>5}: " + " ".join(f"n{n}={(b/main[(grid,n)]['gw'] if b and main[(grid,n)]['gw'] else 0):.2f}x" for n in RANKS))

# --- step count vs tolerance vs grid (n=1) + accuracy vs fixed reference ---
# The fixed-BDF2 reference (MAXITER steps) is expensive at large grids and accuracy is
# already validated finely at 128^2, so only compute it for grid <= 256.
print("\n=== tol x grid (n=1): adaptive steps  [fixed = %d] ; accuracy vs fixed-BDF2 ref (grid<=256) ===" % MAXITER)
print(f"{'grid':>5} " + " ".join(f"tol={t}".ljust(11) for t in TOLS) + " maxErr(tol=1 vs fixed)")
for grid in GRIDS:
    cells = {1.0: main[(grid, 1)]}
    for t in TOLS:
        if t != 1.0:
            cells[t] = run(grid, 1, t)
    row = " ".join(f"{cells[t]['steps']} steps".ljust(11) for t in TOLS)
    err = "- (skipped)"
    if grid <= 256:
        ref = run(grid, 1, 0.0, fixed=True)["field"]
        a1 = cells[1.0]["field"]
        err = (f"{np.abs(a1-ref).max():.3f} m" if (a1 is not None and ref is not None) else "-")
    print(f"{grid:>5} {row} {err}")
