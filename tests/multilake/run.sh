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
#   3. CONVERGENT    under active-set, the lake COUNT is identical at three time steps under BOTH
#                    coupling schemes (a), and then three clauses about the dt->0 limit. A
#                    supply-limited lake stage is a rate balance, so it MUST carry ordinary
#                    truncation error -- what matters is that it converges rather than wanders.
#                    (A capacity-limited lake filled to its sill is set by topography and is exactly
#                    dt-independent; that case is covered by dt_sensitivity.)
#                      b  the error against the finest rung SHRINKS as dt shrinks, both schemes.
#                      c  iterating the FSM coupling is never WORSE than the lagged scheme.
#                      d  the two schemes CONVERGE ON EACH OTHER as dt falls -- because the lag is
#                         one step's worth of FSM output, so the need for the iteration vanishes
#                         with dt (Andy, 2026-09-23). Measured: lake A's cross-scheme gap goes
#                         5.06e-02 m -> machine noise between dt and dt/2.
#                    THIS SUITE RUNS BOTH SCHEMES ITSELF (arms A* lagged, B* iterated) because (c)
#                    and (d) are COMPARISONS, which one invocation cannot make. It is therefore not
#                    in run_all.sh's converged parallel pass.
#                    b REPLACED a first-order rate band, [0.3, 0.7], which the iteration broke BY
#                    BEING MORE ACCURATE: lake A's coarse-dt error fell 24% while its dt/2 error was
#                    unchanged, which pushes the ratio UP (0.462 -> 0.712). A rate band punishes an
#                    accuracy gain at the coarse end; these clauses assert the stated intent instead.
#   4. BITES         under the default `implicit` collector the same comparison FAILS structurally --
#                    the lake COUNT itself changes with dt (6 lakes at dt=0.5yr, 5 at dt=0.25yr),
#                    because the in-residual siphon leaves a head ~ linear in dt which FSM then routes
#                    into a different set of lakes. This is what makes the test a regression test and
#                    not a tautology: it fails without the active-set constraint.
#
# Usage:  tests/multilake/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/multilake_t0_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
make_work multilake
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
# THE CONFIG IS A FILE NOW (#83): tests/multilake/config.yaml. Every setting the run resolves to is
# stated there, and tests/config_identity.py enforces it for every suite (#79 Phase 5).
#
# THREE KEYS MOVE TOGETHER PER ARM and all three are supplied here, none inherited: dt,
# report_interval (scaled inversely so every arm covers the SAME simulated time) and the collector.
# Holding total simulated time fixed while dt changes is what makes the comparison about dt rather
# than about how long the runs went.
mkcfg() { # $1 stem, $2 dt, $3 report_interval, $4 collection.method, $5 coupling.iterations (ALL REQUIRED)
    local dt="${2:?mkcfg needs a dt -- it is the subject, never inherit it}"
    local ri="${3:?mkcfg needs a report_interval: it scales inversely with dt}"
    local cm="${4:?mkcfg needs a collection.method: active_set or the implicit control}"
    # THE COUPLING SCHEME IS PER-ARM AND EXPLICIT (#112). This suite runs BOTH -- the lagged
    # 1-pass scheme and the iterating default -- because its convergence claim is a COMPARISON
    # between them, and a comparison cannot be made from one invocation. It is therefore NOT in
    # run_all.sh's converged parallel pass, and does NOT call apply_test_iterations: an override
    # would rewrite the lagged arms too and leave the suite comparing iterated against iterated.
    local it="${5:?mkcfg needs coupling.iterations: this suite compares 1 against the default}"
    sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
        -e "s|@DT@|$dt|g" -e "s|@REPORT@|$ri|g" \
        -e "s|^    iterations: 1.*|    iterations: $it   # PER-ARM: the coupling scheme is the subject|" \
        -e "s|^    method: active_set|    method: $cm|" config.yaml > "$WORK/$1.yaml"
}

run() { # $1 stem, $2 deltat, $3 report_interval, $4 collector, $5 iterations, $6.. solver flags
    local stem="$1" dt="$2" ri="$3" coll="$4" it="$5"; shift 5
    mkcfg "$stem" "$dt" "$ri" "$coll" "$it"
    mpirun -n "$RANKS" "$WTM" "$WORK/$stem.yaml" "$@" \
        > "$WORK/$stem.log" 2>&1 || { echo "  RUN FAILED: $stem"; tail -3 "$WORK/$stem.log"; return 1; }
}

echo "=== multi-lake: four lakes at different stages, vs the time step ==="
fail=0
run A1 15768000 10 active_set 1 || fail=1   # LAGGED,   dt = 0.5   yr
run A2  7884000 20 active_set 1 || fail=1   # LAGGED,   dt = 0.25  yr
run A4  3942000 40 active_set 1 || fail=1   # LAGGED,   dt = 0.125 yr
run B1 15768000 10 active_set 4 || fail=1   # ITERATED, dt = 0.5   yr
run B2  7884000 20 active_set 4 || fail=1   # ITERATED, dt = 0.25  yr
run B4  3942000 40 active_set 4 || fail=1   # ITERATED, dt = 0.125 yr
run I1 15768000 10 implicit   1 || fail=1   # implicit, dt = 0.5  yr
run I2  7884000 20 implicit   1 || fail=1   # implicit, dt = 0.25 yr
[[ $fail -eq 0 ]] || { echo "MULTI-LAKE: FAILED (a run did not complete)"; exit 1; }

# PROMOTED 2026-09-22: both were Python locals inside the heredoc, invisible to #121's sweep.
# SPREAD: 0   measured 2026-09-22 by repeat run after promotion.
# DERIVED, ONE-SIDED: a filled lake's free surface must be FLAT, so the within-lake sigma is zero
#   in exact arithmetic. Measured 2.84e-14 m over all runs -- roundoff on an O(100 m) field, i.e.
#   ~1e-16 relative. The bound at 1e-9 m sits 5 orders above that floor and far below any real
#   tilt, which would be centimetres.
FLAT_TOL="${FLAT_TOL:-1e-9}"
# SPREAD: 0   measured 2026-09-22 by repeat run after promotion.
# DERIVED, ONE-SIDED bite guard: the lakes must actually have DEPTH, or a flat surface is trivially
#   flat and the FLAT check proves nothing. The degenerate value is 0 m.
SPREAD_MIN="${SPREAD_MIN:-1.0}"
# SPREAD: 0   measured 2026-09-23 -- repeat runs of both coupling schemes reproduce every stage
#             bit-identically, so headroom here measures SENSITIVITY, not flake risk.
# DERIVED 2026-09-23, SEPARATING: the error against the finest rung must SHRINK as dt shrinks --
#   that is the "converges rather than wanders" claim, and a WANDERING stage gives a ratio >= 1.
#   Measured worst over 4 lakes x 2 coupling schemes: 0.416. 0.8 sits between: 1.9x above the
#   measurement and 20% below the degenerate value. Deliberately NOT a first-order band -- the old
#   [0.3, 0.7] encoded a RATE, and the coupling iteration broke it by being more accurate at coarse
#   dt (lake A: 2.108e-01 -> 1.602e-01 m, ratio 0.462 -> 0.712) with the dt/2 error unchanged.
CONV_MAX="${CONV_MAX:-0.8}"   # |dt/2 - dt/4| / |dt - dt/4|, per lake, per scheme
# SPREAD: 0   measured 2026-09-23; see the note at CONV_MAX.
# DERIVED 2026-09-23, ONE-SIDED and the multiplier is NOT a convention: the bound IS 1, because the
#   claim is "iterating never makes a lake worse" and 1 is where worse begins. The 1e-6 is float
#   slack for the two lakes that tie EXACTLY (ratio 1.0000: B and C are unchanged to five figures).
#   Measured across 4 lakes: min 0.7601 (lake A, 24% better), max 1.0000.
NOWORSE_MAX="${NOWORSE_MAX:-1.000001}"   # err(iterated) / err(lagged) at the coarsest dt
# SPREAD: 0   measured 2026-09-23; see the note at CONV_MAX.
# DERIVED 2026-09-23, SEPARATING, and it encodes Andy's point that AS dt -> 0 THE NEED FOR THE
#   ITERATION VANISHES: the lag is one step's worth of FSM output, so the two schemes must agree in
#   the limit. Measured per-lake |iterated - lagged|: lake A 5.06e-02 -> 1.42e-14 -> 1.85e-13 m,
#   lake D 6.39e-03 -> 5.70e-04 -> 1.08e-04 m. The gap collapses to MACHINE NOISE, where a strict
#   monotone test would fail on 1e-14 wobble. The floor separates the two edges it sits between:
#   5400x ABOVE the largest observed noise gap (1.85e-13 m) and 5e7x BELOW the physical signal it
#   must not mask (5.06e-02 m).
GAP_FLOOR="${GAP_FLOOR:-1e-9}"   # |iterated - lagged| below this counts as zero
CONV_MAX="$CONV_MAX" NOWORSE_MAX="$NOWORSE_MAX" GAP_FLOOR="$GAP_FLOOR" \
FLAT_TOL="$FLAT_TOL" SPREAD_MIN="$SPREAD_MIN" \
  WORK="$WORK" INP="$INP" TESTS="$(readlink -f ..)" "$PY" - <<'PY'
import os, sys, glob
import numpy as np, rasterio
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL              # latest_output: refuses a match from a DIFFERENT stem
from collections import deque

W, INP = os.environ["WORK"], os.environ["INP"]
topo = rasterio.open(f"{INP}/multilake_t0_topography.tif").read(1).astype(float)
mask = rasterio.open(f"{INP}/multilake_t0_mask.tif").read(1).astype(float) > 0
FLAT_TOL   = float(os.environ["FLAT_TOL"])    # derivations beside the shell defaults
CONV_MAX    = float(os.environ["CONV_MAX"])
NOWORSE_MAX = float(os.environ["NOWORSE_MAX"])
GAP_FLOOR   = float(os.environ["GAP_FLOOR"])
MIN_SPREAD = float(os.environ["SPREAD_MIN"])

def lakes(stem):
    """[(ncells, stage, sigma, depth_spread)] for connected ponded clusters of >=3 cells."""
    fs = [VOL.latest_output(f"{W}/{stem}_")]   # guards against a stem that is a prefix of another
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

R = {s: lakes(s) for s in ("A1", "A2", "A4", "B1", "B2", "B4", "I1", "I2")}
if any(v is None for v in R.values()):
    print("  FAIL  missing output rasters"); sys.exit(1)
fail = 0

# 1. the fixture is non-trivial
n, stages = len(R["A1"]), sorted({round(r[1], 3) for r in R["A1"]})
spread = max(r[3] for r in R["A1"])
ok = n >= 4 and len(stages) >= 3 and spread >= MIN_SPREAD
fail |= not ok
print(f"  {'PASS' if ok else 'FAIL'}  NON-TRIVIAL   {n} multi-cell lakes, {len(stages)} distinct stages, "
      f"max within-lake depth spread {spread:.2f} m (min SPREAD_MIN={MIN_SPREAD}); lakes >=4, stages >=3")

# 2. every lake has a flat free surface, in every run
worst = max((r[2] for v in R.values() for r in v), default=0.0)
ok = worst < FLAT_TOL
fail |= not ok
print(f"  {'PASS' if ok else 'FAIL'}  FLAT          worst free-surface sigma over all runs "
      f"{worst:.2e} m (tol FLAT_TOL={FLAT_TOL:.0e})")

# 3. active-set: stable topology + convergence toward a dt->0 limit, under BOTH coupling schemes
counts = [len(R[s]) for s in ("A1", "A2", "A4", "B1", "B2", "B4")]
ok = len(set(counts)) == 1
fail |= not ok
print(f"  {'PASS' if ok else 'FAIL'}  CONVERGENT/a  active-set lake count stable across dt AND "
      f"coupling scheme: {counts}")
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
        # WHAT THIS ASSERTS, AND WHY IT IS NOT A RATE BAND ANY MORE (#112, 2026-09-23).
        # It used to require the halving ratio in [0.3, 0.7] -- i.e. FIRST ORDER. That is a stronger
        # claim than the intent stated at the top of this file ("converges to a dt->0 limit rather
        # than wandering"), and the coupling iteration violated it BY BEING MORE ACCURATE: lake A's
        # coarse-dt error fell 24% (2.108e-01 -> 1.602e-01 m) while its dt/2 error was unchanged to
        # four figures, which necessarily pushes the ratio UP (0.462 -> 0.712). A rate band punishes
        # an accuracy gain at the coarse end. These three clauses assert the intent directly.
        def err(scheme, lake):      # |stage(dt) - stage(dt/4)| and |stage(dt/2) - stage(dt/4)|
            c, m, f = (R[scheme + k][lake][1] for k in ("1", "2", "4"))
            return abs(c - f), abs(m - f)
        nlake = len(R["A1"])
        conv, noworse, shrink = [], [], []
        for i in range(nlake):
            ec_l, em_l = err("A", i)      # lagged
            ec_i, em_i = err("B", i)      # iterated
            conv.append(max(em_l / ec_l if ec_l else 9.9, em_i / ec_i if ec_i else 9.9))
            noworse.append(ec_i / ec_l if ec_l else 9.9)
            g = [abs(R["B" + k][i][1] - R["A" + k][i][1]) for k in ("1", "2", "4")]
            g = [0.0 if x < GAP_FLOOR else x for x in g]      # below the floor IS zero; see its note
            shrink.append(g[0] >= g[1] >= g[2])
        good_c = all(c < CONV_MAX for c in conv)
        good_n = all(n <= NOWORSE_MAX for n in noworse)
        good_s = all(shrink)
        fail |= not (good_c and good_n and good_s)
        print(f"  {'PASS' if good_c else 'FAIL'}  CONVERGENT/b  error SHRINKS with dt, both schemes: "
              f"worst |dt/2-dt/4| / |dt-dt/4| = {max(conv):.3f} (tol CONV_MAX={CONV_MAX})")
        print(f"  {'PASS' if good_n else 'FAIL'}  CONVERGENT/c  iterating is never WORSE: worst "
              f"err(iterated)/err(lagged) at coarse dt = {max(noworse):.4f} (tol NOWORSE_MAX={NOWORSE_MAX})"
              f" -- best {min(noworse):.4f}")
        print(f"  {'PASS' if good_s else 'FAIL'}  CONVERGENT/d  the two schemes CONVERGE ON EACH OTHER "
              f"as dt falls: per-lake |iterated-lagged| at dt, dt/2, dt/4 monotone non-increasing "
              f"(floor GAP_FLOOR={GAP_FLOOR:.0e} m) -- {['ok' if x else 'NO' for x in shrink]}")

# 4. BITES: the default implicit collector is NOT topologically stable in dt
ci = [len(R["I1"]), len(R["I2"])]
bites = ci[0] != ci[1]
fail |= not bites
print(f"  {'PASS' if bites else 'FAIL'}  BITES         implicit lake count CHANGES with dt: {ci} "
      f"(if these ever match, this test no longer proves active-set is what fixes it)")

print("\nMULTI-LAKE: " + ("ALL PASSED" if not fail else "FAILED"))
sys.exit(1 if fail else 0)
PY
