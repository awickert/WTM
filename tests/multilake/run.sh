#!/usr/bin/env bash
# MULTI-LAKE test: several genuine lakes at DIFFERENT stages, and what the exfiltration constraint
# does to them as the time step changes.
#
# WHY THIS FIXTURE EXISTS. The lake-aware active-set constraint pins each cell's head at
# `topo + surface_water_depth`, which is meant to be ONE flat free-surface elevation per lake. Every
# earlier fixture tested that on a SINGLE depression -- on the island benchmark the whole claim rested
# on one 4-cell lake. One lake cannot show that different lakes hold different stages simultaneously,
# nor that the lake TOPOLOGY is stable. This fixture has four multi-cell lakes (including a nested
# metadepression) whose floors differ, and whose stages are set by an inflow/evaporation rate balance
# rather than by filling to a sill -- see make_inputs.py.
#
# WHAT IT ASSERTS
#   1. NON-TRIVIAL   the fixture really does produce >=4 multi-cell lakes at >=3 distinct stages, with
#                    the water depth varying by >=1 m WITHIN a lake. Without that last part a flat
#                    free surface would be trivially satisfied by a flat floor.
#   2. FLAT          every lake's free-surface elevation (topo + wtd) is constant to <1e-9 m. This is
#                    an invariant of any correct configuration, checked in every run.
#   3. CONVERGENT    under active-set, the lake COUNT is identical at three time steps and each stage
#                    converges at ~first order as dt halves (difference ratio in [0.3, 0.7]). This is
#                    the real claim: a supply-limited lake stage is a rate balance, so it MUST carry
#                    ordinary truncation error -- what matters is that it converges to a dt->0 limit
#                    rather than wandering. (A capacity-limited lake filled to its sill is set by
#                    topography and is exactly dt-independent; that case is covered by dt_sensitivity.)
#   4. BITES         under the default `implicit` collector the same comparison FAILS structurally --
#                    the lake COUNT itself changes with dt (6 lakes at dt=0.5yr, 5 at dt=0.25yr),
#                    because the in-residual siphon leaves a head ~ linear in dt which FSM then routes
#                    into a different set of lakes. This is what makes the test a regression test and
#                    not a tautology: it fails without the active-set constraint.
#
# Usage:  tests/multilake/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/multilake_t0_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
WORK=$(mktemp -d /tmp/multilake_XXXX); trap 'rm -rf "$WORK"' EXIT
PY="${PY:-python3}"
export OMP_NUM_THREADS=1
RANKS="${RANKS:-4}"

# Every arm names its collector EXPLICITLY. Inheriting the default is a trap: when the default flipped
# from `implicit` to `active_set` the two "implicit" arms silently became active-set arms and the BITES
# check went from [6,5] to [4,4] -- the test correctly reported that it had stopped discriminating.
#
# 150 yr is ample: the lakes reach a per-cycle rms of ~1e-4 mm-water and their stages are unchanged
# at 400 yr. dt is the only thing that varies between arms; report_interval scales with it so the
# reporting cadence (and therefore the FSM/coupling cadence per report) is held fixed.
mkcfg() { # $1 = stem, $2 = deltat seconds, $3 = report_interval steps, $4 = runoff_collector
    ../emit_config.sh > "$WORK/$1.yaml" <<EOF
run_type equilibrium
total_time 150yr
supplied_wt 1
deltat $2
report_interval $3
save_nreport_interval 9999
cells_per_degree 10
southern_edge -45
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
infiltration_on 0
fsm_on 1
runoff_collector $4
surfdatadir $INP
region multilake
time_start t0
time_end t0
eq_tol 0
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}

run() { # $1 = stem, $2 = deltat, $3 = report_interval, $4 = collector, $5.. = solver flags
    local stem="$1" dt="$2" ri="$3" coll="$4"; shift 4
    mkcfg "$stem" "$dt" "$ri" "$coll"
    mpirun -n "$RANKS" "$WTM" "$WORK/$stem.yaml" "$@" -snes_stol 1e-8 \
        > "$WORK/$stem.log" 2>&1 || { echo "  RUN FAILED: $stem"; tail -3 "$WORK/$stem.log"; return 1; }
}

echo "=== multi-lake: four lakes at different stages, vs the time step ==="
fail=0
run A1 15768000 10 active_set -wtm_anderson || fail=1   # dt = 0.5   yr
run A2  7884000 20 active_set -wtm_anderson || fail=1   # dt = 0.25  yr
run A4  3942000 40 active_set -wtm_anderson || fail=1   # dt = 0.125 yr
run I1 15768000 10 implicit   -wtm_anderson || fail=1   # implicit, dt = 0.5  yr
run I2  7884000 20 implicit   -wtm_anderson || fail=1   # implicit, dt = 0.25 yr
[[ $fail -eq 0 ]] || { echo "MULTI-LAKE: FAILED (a run did not complete)"; exit 1; }

WORK="$WORK" INP="$INP" "$PY" - <<'PY'
import os, sys, glob
import numpy as np, rasterio
from collections import deque

W, INP = os.environ["WORK"], os.environ["INP"]
topo = rasterio.open(f"{INP}/multilake_t0_topography.tif").read(1).astype(float)
mask = rasterio.open(f"{INP}/multilake_t0_mask.tif").read(1).astype(float) > 0
FLAT_TOL, MIN_SPREAD = 1e-9, 1.0

def lakes(stem):
    """[(ncells, stage, sigma, depth_spread)] for connected ponded clusters of >=3 cells."""
    fs = sorted(glob.glob(f"{W}/{stem}_[0-9]" + "[0-9]"*8 + "_*yr.tif"))
    if not fs: return None
    w = rasterio.open(fs[-1]).read(1).astype(float)
    P = {(y, x) for y, x in zip(*np.where((w > 1e-3) & mask))}
    seen, out = set(), []
    for c in list(P):
        if c in seen: continue
        q, comp = deque([c]), []
        seen.add(c)
        while q:
            y, x = q.popleft(); comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    n = (y+dy, x+dx)
                    if n in P and n not in seen: seen.add(n); q.append(n)
        if len(comp) >= 3:
            t = np.array([topo[y, x] for y, x in comp]); d = np.array([w[y, x] for y, x in comp])
            h = t + d
            out.append((len(comp), h.mean(), h.std(), d.max() - d.min()))
    return sorted(out, key=lambda r: r[1])

R = {s: lakes(s) for s in ("A1", "A2", "A4", "I1", "I2")}
if any(v is None for v in R.values()):
    print("  FAIL  missing output rasters"); sys.exit(1)
fail = 0

# 1. the fixture is non-trivial
n, stages = len(R["A1"]), sorted({round(r[1], 3) for r in R["A1"]})
spread = max(r[3] for r in R["A1"])
ok = n >= 4 and len(stages) >= 3 and spread >= MIN_SPREAD
fail |= not ok
print(f"  {'PASS' if ok else 'FAIL'}  NON-TRIVIAL   {n} multi-cell lakes, {len(stages)} distinct stages, "
      f"max within-lake depth spread {spread:.2f} m (need >=4, >=3, >={MIN_SPREAD})")

# 2. every lake has a flat free surface, in every run
worst = max((r[2] for v in R.values() for r in v), default=0.0)
ok = worst < FLAT_TOL
fail |= not ok
print(f"  {'PASS' if ok else 'FAIL'}  FLAT          worst free-surface sigma over all runs "
      f"{worst:.2e} m (tol {FLAT_TOL:.0e})")

# 3. active-set: stable topology + first-order convergence toward a dt->0 limit
counts = [len(R[s]) for s in ("A1", "A2", "A4")]
ok = len(set(counts)) == 1
fail |= not ok
print(f"  {'PASS' if ok else 'FAIL'}  CONVERGENT/a  active-set lake count stable across dt: {counts}")
if ok:
    d1s = [l1[1] - l2[1] for l1, l2 in zip(R["A1"], R["A2"])]
    if all(abs(d) < 1e-9 for d in d1s):
        # Stages IDENTICAL across dt. That looks like perfect dt-independence but is the signature of
        # the aquifer not participating at all: if the pin loses its lake stage (surface_water_depth
        # forced to 0) the level becomes pure topographic fill by FSM, independent of the solve. A
        # supply-limited lake IS a rate balance and must carry some truncation error.
        fail |= 1
        print("  FAIL  CONVERGENT/b  stages do not respond to dt AT ALL (max change "
              f"{max(abs(d) for d in d1s):.2e} m) -- the lake level is not coupled to the aquifer; "
              "check that the pin still reads a lake stage")
    else:
        ratios = [(l2[1] - l4[1]) / d1 if abs(d1) > 1e-9 else float("nan")
                  for d1, l2, l4 in zip(d1s, R["A2"], R["A4"])]
        good = all(0.3 <= r <= 0.7 for r in ratios)
        fail |= not good
        print(f"  {'PASS' if good else 'FAIL'}  CONVERGENT/b  stage difference ratios on dt halving "
              f"{[round(r, 3) for r in ratios]} (first order => ~0.5, need 0.3..0.7)")

# 4. BITES: the default implicit collector is NOT topologically stable in dt
ci = [len(R["I1"]), len(R["I2"])]
bites = ci[0] != ci[1]
fail |= not bites
print(f"  {'PASS' if bites else 'FAIL'}  BITES         implicit lake count CHANGES with dt: {ci} "
      f"(if these ever match, this test no longer proves active-set is what fixes it)")

print("\nMULTI-LAKE: " + ("ALL PASSED" if not fail else "FAILED"))
sys.exit(1 if fail else 0)
PY
