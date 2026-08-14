#!/usr/bin/env python3
"""Algorithmic-speedup analysis for the transient dt-sweep (#92/#93).

For each time-integration scheme, pair the ACCURACY (error vs a common fine-dt reference at the shared
mid-relaxation horizon T_end) with the COST (wall time / cycles from the harness CSV). A higher-order
scheme that holds accuracy at a coarser dt reaches a target error in fewer cycles / less wall -> the
honest speedup. The reference is the finest-dt run of a 2nd-order scheme (its dt->0 answer).

Usage:  dtsweep_speed.py <results_dir> [ftag=dry] [tend_wk=2] [ref_method=tr] [ref_dt=0.0625]
  results_dir e.g. benchmark/esquibel/results/algo/transient/bench1
"""
import sys, os, glob, numpy as np, rasterio

d       = sys.argv[1]
ftag    = sys.argv[2] if len(sys.argv) > 2 else "dry"
tend_wk = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
ref_m   = sys.argv[4] if len(sys.argv) > 4 else "tr"
ref_dt  = sys.argv[5] if len(sys.argv) > 5 else "0.0625"
DTS     = os.environ.get("WTM_DTS", "0.0625 0.125 0.25 0.5 1 2").split()  # fine -> coarse
METHODS = os.environ.get("WTM_METHODS", "cc tr bdf2v").split()

def cyc(dt):  # cycles to T_end at this dt (matches the harness: round(tend/dt))
    return int(tend_wk / float(dt) + 0.5)

def field(m, dt):
    p = os.path.join(d, f"{ftag}_{m}_dt{dt}wk_{cyc(dt):09d}.tif")
    return rasterio.open(p).read(1).astype(float) if os.path.exists(p) else None

def wall_csv(m):  # {dt_wk: (steps, rc, wall_s)} from runs_<ftag>_<m>.csv (forcing,method,dt_wk,steps,rc,wall_s,tif)
    out = {}
    p = os.path.join(d, f"runs_{ftag}_{m}.csv")
    if os.path.exists(p):
        for line in open(p):
            c = line.strip().split(",")
            if len(c) >= 6 and c[0] == ftag:
                out[c[2]] = (c[3], c[4], c[5])
    return out

def iters(m, dt):  # total SNES (Anderson) iterations over the run = residual evals -- the node-INDEPENDENT
    p = os.path.join(d, f"{ftag}_{m}_dt{dt}wk.log")  # cost metric (wall on shared nodes is contention-noisy)
    if not os.path.exists(p):
        return None
    tot = 0
    for line in open(p, errors="ignore"):
        k = line.find("Number of nonlinear iterations = ")
        if k >= 0:
            try: tot += int(line[k + 33:].split()[0])
            except Exception: pass
    return tot

ref = field(ref_m, ref_dt)
if ref is None:
    print(f"reference {ref_m}@{ref_dt}wk missing in {d}"); sys.exit(1)
fin0 = np.isfinite(ref)
print(f"reference = {ref_m} @ dt={ref_dt}wk (2nd-order dt->0), T_end={tend_wk}wk, forcing={ftag}, {int(fin0.sum())} land cells")

rows = {}  # (method, dt) -> (steps, wall_s, emax, emean, its, its_per_cyc)
for m in METHODS:
    W = wall_csv(m)
    print(f"\n{m}:")
    print(f"  {'dt(wk)':>8} {'steps':>6} {'rc':>3} {'wall_s':>7} {'SNES_its':>9} {'its/cyc':>8} {'max|err|(m)':>12} {'mean|err|(m)':>13}")
    for dt in DTS:
        steps, rc, ws = W.get(dt, (str(cyc(dt)), "?", "?"))
        ncyc = int(steps) if str(steps).isdigit() else cyc(dt)
        it = iters(m, dt)
        ipc = (it / ncyc) if (it and ncyc) else None
        a = field(m, dt)
        if a is None:
            rows[(m, dt)] = (steps, ws, None, None, it, ipc)
            print(f"  {dt:>8} {steps:>6} {rc:>3} {ws:>7} {str(it):>9} {(f'{ipc:.0f}' if ipc else '?'):>8}  (no field: rc={rc})")
            continue
        fin = fin0 & np.isfinite(a)
        emax = float(np.max(np.abs((a - ref)[fin])))
        emean = float(np.mean(np.abs((a - ref)[fin])))
        rows[(m, dt)] = (steps, ws, emax, emean, it, ipc)
        print(f"  {dt:>8} {steps:>6} {rc:>3} {ws:>7} {str(it):>9} {(f'{ipc:.0f}' if ipc else '?'):>8} {emax:>12.3e} {emean:>13.3e}")

# TIMING comparison: least wall to reach a MEAN-error target (mean is representative; MAX is dominated by
# the deep exp-T outliers, the accuracy floor). Targets span cc-reachable (loose) to 2nd-order-only (tight),
# so we get a timing comparison where all reach it and a success comparison below cc's floor.
def wall_num(ws):
    try: return float(ws)
    except Exception: return float("inf")
print("\nSpeed-to-accuracy (least WALL to reach mean|err| <= target; '--' = unreachable at any swept dt):")
print(f"  {'target(m)':>10} " + "".join(f"{m:>22}" for m in METHODS))
for target in (2e-1, 1.5e-1, 1.2e-1, 1e-1, 5e-2, 1e-2):
    cells = []
    for m in METHODS:
        cand = [(wall_num(rows[(m, dt)][1]), dt, rows[(m, dt)][0]) for dt in DTS
                if (m, dt) in rows and rows[(m, dt)][3] is not None and rows[(m, dt)][3] <= target]
        if not cand:
            cells.append(f"{'--':>22}")
        else:
            w, dt, st = min(cand)  # least wall meeting the target
            cells.append(f"{f'{w:.0f}s @dt{dt}({st}cyc)':>22}")
    print(f"  {target:>10.3g} " + "".join(cells))
