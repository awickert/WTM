#!/usr/bin/env python3
"""Precision-matched report for the scheme benchmark (see run.sh for the design rationale).

Reads each arm's log, reconstructs the per-cycle trajectory (cumulative SNES iterations vs the
water-depth rms in mm-water), and emits three tables:

  1. ISO-PRECISION   cost to first REACH a target rms -- the only fair cross-scheme comparison.
  2. FLOOR           the finest rms each scheme reaches at all.
  3. NATIVE STOP     what each scheme's own auto-stop would have cost, and the rms it stopped at.
                     Explicitly NOT precision-matched; shown so the two are never confused.

Usage: report.py <results_dir>
"""
import re, os, sys, math

R = sys.argv[1] if len(sys.argv) > 1 else "results"
TARGETS = [100.0, 10.0, 1.0]          # mm-water
NATIVE_TOL = 1e-2   # shipped default -wtm_eq_tol -- applied to the HEAD rms in METRES (index 3),
                    # NOT to the mm-water precision axis. Mixing them makes table 3 nonsense.

ITER_RE = re.compile(r"Number of nonlinear iterations = (\d+)")
# cycle N: per-cycle |dwtd| max=.. rms=<HEAD_RMS> frac>tol=.. m  |S*dwtd| max=.. rms=<WATER_RMS> mm-water
# Two different metrics on one line and they are NOT interchangeable: the auto-stop (-wtm_eq_tol) is
# applied to the HEAD rms in metres, while the precision axis for cross-scheme comparison is the
# water-depth rms in mm-water. Capture both so table 3 cannot silently compare against the wrong one.
CYCLE_RE = re.compile(
    r"cycle (\d+): per-cycle \|.wtd\| max=[0-9.eE+-]+ rms=([0-9.eE+-]+) .*?rms=([0-9.eE+-]+) mm-water")


def trajectory(path):
    """[(cycle, cumulative_iters, water_rms_mm, head_rms_m)] in cycle order."""
    cum, rows = 0, []
    for line in open(path, errors="ignore"):
        m = ITER_RE.search(line)
        if m:
            cum += int(m.group(1))
            continue
        c = CYCLE_RE.search(line)
        if c:
            rows.append((int(c.group(1)), cum, float(c.group(3)), float(c.group(2))))
    return rows


def load_summary():
    p = os.path.join(R, "summary.csv")
    out = []
    if not os.path.exists(p):
        return out
    for ln in open(p).readlines()[1:]:
        f = ln.rstrip("\n").split(",")
        if len(f) >= 6:
            out.append(dict(stem=f[0], label=f[1], rc=int(f[2]), wall=float(f[3]),
                            iters=int(f[4]), cycles=int(f[5])))
    return out


def first_at_or_below(traj, target):
    """First row whose WATER rms (mm) is at or below target."""
    for row in traj:
        if row[2] <= target:
            return row[0], row[1], row[2]
    return None


arms = load_summary()
if not arms:
    sys.exit(f"no summary.csv in {R} -- run run.sh first")

data = {}
for a in arms:
    a["traj"] = trajectory(os.path.join(R, a["stem"] + ".log"))
    data[a["stem"]] = a

print("Fixture: island 117x75 (8775 cells), cold start (saturated), dt = 1 week, FSM on.")
print("Precision axis: per-cycle water-depth rms [mm-water] (volume-consistent, not a head diff).")
print("Wall is measured per ARM; iso-precision wall is apportioned by iteration share and marked '~'.")
print("Iterations are the clean algorithmic metric; wall carries ~10% machine noise.\n")

# ---------------------------------------------------------------- 1. iso-precision
print("=" * 100)
print("1. ISO-PRECISION COST  (cycles / SNES iters / ~wall to FIRST reach the target rms)")
print("=" * 100)
hdr = f"{'scheme':<30}"
for t in TARGETS:
    hdr += f"{('rms<=' + str(t) + 'mm'):>22}"
print(hdr)
print(f"{'':<30}" + "".join(f"{'cyc / iters / ~s':>22}" for _ in TARGETS))
print("-" * 100)
for a in arms:
    row = f"{a['label']:<30}"
    if a["rc"] != 0 and not a["traj"]:
        print(row + f"{'DID NOT RUN (rc=' + str(a['rc']) + ')':>22}")
        continue
    for t in TARGETS:
        hit = first_at_or_below(a["traj"], t)
        if hit is None:
            row += f"{'never':>22}"
        else:
            cyc, it, _ = hit
            frac = it / a["iters"] if a["iters"] else 0.0
            cell = "{} / {} / {:.1f}".format(cyc, it, a["wall"] * frac)
            row += f"{cell:>22}"
    print(row)

# ---------------------------------------------------------------- 2. floor
print()
print("=" * 100)
print("2. PRECISION FLOOR  (finest rms reached, and whether it was still improving at the budget end)")
print("=" * 100)
print(f"{'scheme':<30}{'best rms [mm]':>16}{'at cycle':>10}{'final rms':>14}{'still improving?':>20}")
print("-" * 100)
for a in arms:
    if not a["traj"]:
        print(f"{a['label']:<30}{'-- did not run --':>16}")
        continue
    best = min(a["traj"], key=lambda r: r[2])
    final = a["traj"][-1]
    k = max(1, len(a["traj"]) // 10)
    if final[2] > best[2] * 1.5:
        state = "REGRESSED"          # went past its best and got worse -- not a floor at all
    elif a["traj"][-k][2] > final[2] * 1.05:
        state = "still improving"    # had not converged within the budget
    else:
        state = "PLATEAU"            # genuinely floored
    print(f"{a['label']:<30}{best[2]:>16.4g}{best[0]:>10}{final[2]:>14.4g}{state:>20}")

# ---------------------------------------------------------------- 3. native stop
print()
print("=" * 100)
print(f"3. NATIVE STOP  (where each scheme's own auto-stop -wtm_eq_tol {NATIVE_TOL:g} would have fired)")
print("   *** NOT precision-matched -- each row stops at a DIFFERENT rms. Never compare these times")
print("       across schemes; that is exactly the error table 1 exists to prevent. ***")
print("=" * 100)
print(f"{'scheme':<30}{'stop cycle':>12}{'iters':>10}{'~wall_s':>10}{'rms AT STOP [mm]':>20}")
print("-" * 100)
for a in arms:
    if not a["traj"]:
        print(f"{a['label']:<30}{'-- did not run --':>12}")
        continue
    hit = next((r for r in a["traj"] if r[3] <= NATIVE_TOL), None)
    if hit is None:
        print(f"{a['label']:<30}{'never reached':>12}{'':>10}{'':>10}"
              f"{a['traj'][-1][2]:>20.4g}")
        continue
    cyc, it, rms = hit[0], hit[1], hit[2]
    frac = it / a["iters"] if a["iters"] else 0.0
    print(f"{a['label']:<30}{cyc:>12}{it:>10}{a['wall'] * frac:>10.1f}{rms:>20.4g}")

# ---------------------------------------------------------------- totals
print()
print("Full-budget totals (all arms ran the same cycle budget):")
print(f"{'scheme':<30}{'rc':>5}{'wall_s':>10}{'iters':>10}{'cycles':>9}{'iters/cycle':>13}")
print("-" * 100)
for a in arms:
    ipc = a["iters"] / a["cycles"] if a["cycles"] else float("nan")
    print(f"{a['label']:<30}{a['rc']:>5}{a['wall']:>10.2f}{a['iters']:>10}{a['cycles']:>9}{ipc:>13.2f}")
