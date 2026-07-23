#!/usr/bin/env python3
"""Single-file scaling + memory study for the WTM groundwater solve.

Runs a grid-size x MPI-rank sweep across up to three sibling builds and writes a
CSV plus a printed analysis. It expects the three build trees laid out as
siblings of this repo (the layout on MSI)::

    ~/models/WTM             (this repo -- the "after"/distributed build)
    ~/models/WTM-before      (commit e5aab70: ghost-fix + Anderson, PRE-flip)
    ~/models/WTM-kcallaghan  (KCallaghan/master: the published version)

with each built to <folder>/build/wtm.x. Missing builds are skipped with a note,
so it also runs with only WTM present (local smoke test).

What it measures, per (build, grid, ranks):
  * wall_s   -- total end-to-end wall time (the honest, unambiguous metric).
  * gw_s     -- max over ranks of the model's "t GW time =" (solve phase;
                indicative only -- inflated by non-root ranks waiting on rank-0
                serial init, and it is NOT summed across ranks).
  * iters    -- SNES nonlinear iteration count (should be invariant to ranks).
  * mem total/max/min (GB) -- PETSc -memory_view peak process memory: summed over
                ranks / largest rank (rank 0 holds the full grid) / smallest rank
                (a subdomain -- this is the flip's payoff).
  * rc       -- exit code (nonzero usually = OOM at that grid/ranks; itself data).

Why the comparisons matter:
  before -> after       isolates the distributed-ArrayPack flip (memory + timing).
  kcallaghan -> before  isolates the ghost-fix + Anderson solver change.
  kcallaghan -> after   the TOTAL speedup vs. the published model.

`run_type test` synthesizes every field except topography and slope, which this
script generates (via make_synthetic) at each grid size. cells_per_degree is set
to grid/120 so the domain always spans the same -45..+75 latitude band and never
runs a cell past the pole (which is a hard error in the area calculation).

Usage (from anywhere; paths are resolved relative to this file):
    source ~/models/WTM/msi_env.sh          # load mpiexec / petsc / gdal
    python3 scaling_study.py                 # defaults below
    python3 scaling_study.py --grids 1000 2000 4000 --ranks 1 2 4 8 --maxiter 5
    python3 scaling_study.py --builds after before kcallaghan
    python3 scaling_study.py --strong 4000 --ranks 1 2 4 8 16   # one grid, many ranks

Comparisons are only meaningful with the sibling builds present; kcallaghan is
physically correct only at n=1 (its ghost-cell bug mis-runs at >1 rank), so its
n>1 rows are labeled INDICATIVE and excluded from the memory/flip comparison.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import make_synthetic  # noqa: E402  (same directory)

# models root = parent of the WTM repo (this file is WTM/benchmark/scaling/...).
MODELS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

# label -> repo folder name (built to <folder>/build/wtm.x)
BUILD_FOLDERS = {
    "after": "WTM",
    "before": "WTM-before",
    "kcallaghan": "WTM-kcallaghan",
}

SNES_ARGS = ["-snes_type", "anderson", "-snes_stol", "1e-6"]  # no -snes_mf (deadlocks under MPICH)

# A single well-formed float. Bounded so it stops at the next number even when
# concurrent MPI ranks interleave their output with no separator (e.g. two ranks
# printing "t GW time = 16.8993" and "16.9009" can arrive as "...16.899316.9009").
_NUM = r"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
MEM_RE = re.compile(r"process memory:.*?total\s+" + _NUM + r"\s+max\s+" + _NUM + r"\s+min\s+" + _NUM, re.I)
GW_RE = re.compile(r"GW time =\s*" + _NUM)
ITER_RE = re.compile(r"nonlinear iterations =\s*(\d+)")


def build_path(label):
    return os.path.join(MODELS_ROOT, BUILD_FOLDERS[label], "build", "wtm.x")


def ensure_inputs(grid, root):
    """Generate the topography+slope pair for a square grid if not already present."""
    sdir = os.path.join(root, f"{grid}x{grid}")
    topo = os.path.join(sdir, "synth_topography.tif")
    if not os.path.exists(topo):
        os.makedirs(sdir, exist_ok=True)
        t = make_synthetic.build_topography(grid, grid)
        s = make_synthetic.slope_from_topo(t)
        make_synthetic.write_tif(topo, t)
        make_synthetic.write_tif(os.path.join(sdir, "synth_slope.tif"), s)
    return sdir


def write_cfg(path, sdir, grid, maxiter, total_cycles):
    # cells_per_degree = grid/120 keeps the -45..+75 band on the globe at any size.
    cpd = grid / 120.0
    with open(path, "w") as f:
        f.write(
            f"run_type           test\n"
            f"fsm_on             0\n"
            f"evap_mode          0\n"
            f"infiltration_on    0\n"
            f"runoff_ratio_on    0\n"
            f"cells_per_degree   {cpd:.6f}\n"
            f"southern_edge      -45\n"
            f"deltat             31536000\n"
            f"total_cycles       {total_cycles}\n"
            f"maxiter            {maxiter}\n"
            f"fdepth_a           200\n"
            f"fdepth_b           150\n"
            f"fdepth_fmin        2\n"
            f"time_start         t0\n"
            f"time_end           t0\n"
            f"surfdatadir        {sdir}\n"
            f"region             synth\n"
            f"supplied_wt        0\n"
            f"textfilename       /tmp/scaling_run.txt\n"
            f"outfile_prefix     /tmp/scaling_out_\n"
            f"cycles_to_save     9999\n"
        )


def parse(out):
    """Extract (gw_max, iters, mem_total, mem_max, mem_min) in GB from run output."""
    gws = [float(x) for x in GW_RE.findall(out)]
    iters = [int(x) for x in ITER_RE.findall(out)]
    # PETSc -memory_view prints a "Current" and a "Maximum (over computational
    # time)" process-memory line; we want the Maximum, else the first available.
    mem_lines = [ln for ln in out.splitlines() if re.search(r"process memory", ln, re.I)]
    chosen = next((ln for ln in mem_lines if re.search(r"maximum", ln, re.I)),
                  mem_lines[0] if mem_lines else None)
    mem = None
    if chosen:
        m = MEM_RE.search(chosen)
        if m:
            mem = tuple(float(v) / 1e9 for v in m.groups())
    gw = max(gws) if gws else None
    it = max(iters) if iters else None
    return gw, it, mem


def run_one(binary, mpiexec, nranks, cfg):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    cmd = [mpiexec, "-n", str(nranks), binary, cfg, *SNES_ARGS, "-memory_view"]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall = time.time() - t0
    out = proc.stdout + proc.stderr
    gw, it, mem = parse(out)
    return {
        "rc": proc.returncode, "wall": wall, "gw": gw, "iters": it,
        "mem_total": mem[0] if mem else None,
        "mem_max": mem[1] if mem else None,
        "mem_min": mem[2] if mem else None,
        "raw": out,
    }


def fmt(x, spec=".2f"):
    return format(x, spec) if isinstance(x, (int, float)) else "-"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grids", type=int, nargs="+", default=[1000, 2000, 4000],
                    help="square grid side lengths (cells per side)")
    ap.add_argument("--strong", type=int, default=None,
                    help="shortcut: sweep a single grid over --ranks (overrides --grids)")
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8],
                    help="MPI rank counts")
    ap.add_argument("--builds", nargs="+", default=list(BUILD_FOLDERS),
                    choices=list(BUILD_FOLDERS), help="which builds to run")
    ap.add_argument("--maxiter", type=int, default=5,
                    help="GW solves per cycle (higher amortizes one-time serial init)")
    ap.add_argument("--reps", type=int, default=1,
                    help="repeat each run N times and keep the best (min-wall); "
                         "on a shared node single-shot timing is noisy -- use 3 for timing runs")
    ap.add_argument("--total-cycles", type=int, default=1)
    ap.add_argument("--mpiexec", default="mpiexec")
    ap.add_argument("--outcsv", default=os.path.join(SCRIPT_DIR, "results.csv"))
    ap.add_argument("--keep-failed-logs", action="store_true",
                    help="write the full output of failed runs next to the CSV")
    args = ap.parse_args()

    grids = [args.strong] if args.strong else args.grids
    synth_root = os.path.join(SCRIPT_DIR, "synth")
    os.makedirs(synth_root, exist_ok=True)

    # Resolve the requested builds; skip (with a note) any that are absent.
    builds = []
    for label in args.builds:
        p = build_path(label)
        if os.access(p, os.X_OK):
            builds.append((label, p))
        else:
            print(f"  note: {label} build not found at {p} -- skipping")
    if not builds:
        sys.exit("No usable builds found. Build at least WTM/build/wtm.x.")

    print(f"\nmodels root : {MODELS_ROOT}")
    print(f"builds      : {', '.join(l for l, _ in builds)}")
    print(f"grids       : {grids}")
    print(f"ranks       : {args.ranks}")
    print(f"maxiter     : {args.maxiter}   total_cycles: {args.total_cycles}")
    print(f"solver args : {' '.join(SNES_ARGS)}\n")

    rows = []
    header = f"{'build':<11}{'grid':>6}{'n':>4}{'rc':>4}{'iters':>6}{'wall_s':>9}{'gw_s':>9}   mem GB total/max/min"
    print(header)
    print("-" * len(header))

    # Write the CSV incrementally and flush after every run, so a crash or an
    # aborted session never loses the rows already gathered.
    fields = ["build", "binary", "grid", "cells", "nranks", "rc", "iters",
              "wall_s", "gw_s", "mem_total", "mem_max", "mem_min"]
    csvf = open(args.outcsv, "w", newline="")
    writer = csv.DictWriter(csvf, fieldnames=fields)
    writer.writeheader()
    csvf.flush()

    for grid in grids:
        sdir = ensure_inputs(grid, synth_root)
        cfg = os.path.join(sdir, f"synth_m{args.maxiter}_c{args.total_cycles}.cfg")
        write_cfg(cfg, sdir, grid, args.maxiter, args.total_cycles)
        for label, binary in builds:
            for n in args.ranks:
                # Never let a single run (or a parse hiccup) abort the sweep.
                # Repeat --reps times; keep the best (min-wall) successful run to
                # suppress shared-node timing noise. Fall back to the last result
                # if every rep failed.
                r = None
                for _ in range(max(1, args.reps)):
                    try:
                        cur = run_one(binary, args.mpiexec, n, cfg)
                    except Exception as e:  # noqa: BLE001 -- record and keep going
                        cur = {"rc": -99, "wall": None, "gw": None, "iters": None,
                               "mem_total": None, "mem_max": None, "mem_min": None,
                               "raw": f"scaling_study exception: {e!r}"}
                    if r is None:
                        r = cur
                    elif cur["rc"] == 0 and (r["rc"] != 0 or
                                             (cur["wall"] or 1e9) < (r["wall"] or 1e9)):
                        r = cur
                tag = "  [INDICATIVE: ghost bug at n>1]" if (label == "kcallaghan" and n > 1) else ""
                if r["rc"] == -99:
                    tag += "  [run/parse error -- see log]"
                mem = f"{fmt(r['mem_total'])} / {fmt(r['mem_max'])} / {fmt(r['mem_min'])}"
                print(f"{label:<11}{grid:>6}{n:>4}{r['rc']:>4}"
                      f"{(fmt(r['iters'],'d') if r['iters'] is not None else '?'):>6}"
                      f"{fmt(r['wall'],'.1f'):>9}{fmt(r['gw'],'.2f'):>9}   {mem}{tag}")
                row = dict(build=label, binary=binary, grid=grid, cells=grid * grid,
                           nranks=n, rc=r["rc"], iters=r["iters"], wall_s=r["wall"],
                           gw_s=r["gw"], mem_total=r["mem_total"], mem_max=r["mem_max"],
                           mem_min=r["mem_min"])
                rows.append(row)
                writer.writerow(row)
                csvf.flush()
                if r["rc"] != 0 and (args.keep_failed_logs or r["rc"] == -99):
                    lp = os.path.join(SCRIPT_DIR, f"fail_{label}_{grid}_n{n}.log")
                    with open(lp, "w") as f:
                        f.write(r.get("raw", ""))
                    print(f"      (rc={r['rc']}; log: {lp})")
        print()

    csvf.close()
    print(f"CSV: {args.outcsv}\n")

    analyze(rows, grids, args.ranks, [l for l, _ in builds])


def analyze(rows, grids, ranks, labels):
    def get(build, grid, n, key):
        for r in rows:
            if r["build"] == build and r["grid"] == grid and r["nranks"] == n:
                return r[key]
        return None

    print("=== Strong scaling (wall speedup vs n=1, per build/grid) ===")
    for label in labels:
        for grid in grids:
            base = get(label, grid, 1, "wall_s")
            if not base:
                continue
            parts = []
            for n in ranks:
                w = get(label, grid, n, "wall_s")
                parts.append(f"n={n}:{base / w:.2f}x" if w else f"n={n}:-")
            print(f"  {label:<11} grid={grid:<6} {'  '.join(parts)}")

    print("\n=== Memory: min per-rank (GB) -- the flip's payoff ===")
    print("  (before/kcallaghan replicate the full grid on every rank; after does not)")
    for grid in grids:
        for n in ranks:
            cells = []
            for label in labels:
                mn = get(label, grid, n, "mem_min")
                if mn is not None and not (label == "kcallaghan" and n > 1):
                    cells.append(f"{label}={mn:.2f}")
            if cells:
                print(f"  grid={grid:<6} n={n:<3} {'  '.join(cells)}")

    if len(labels) > 1:
        print("\n=== Speedup decomposition at n=1 (wall_s ratios) ===")
        for grid in grids:
            k = get("kcallaghan", grid, 1, "wall_s")
            b = get("before", grid, 1, "wall_s")
            a = get("after", grid, 1, "wall_s")
            msgs = []
            if k and b:
                msgs.append(f"solver+ghostfix (kcallaghan/before)={k / b:.2f}x")
            if b and a:
                msgs.append(f"flip (before/after)={b / a:.2f}x")
            if k and a:
                msgs.append(f"TOTAL (kcallaghan/after)={k / a:.2f}x")
            if msgs:
                print(f"  grid={grid:<6} " + "  ".join(msgs))
    print()


if __name__ == "__main__":
    main()
