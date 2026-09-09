#!/usr/bin/env bash
# CROSS-RANK DETERMINISM UNDER ADAPTIVE dt (task #56).
#
# The adaptive controller chooses the step from an error estimate that is a GLOBAL REDUCTION, so it
# is the one part of the model whose decisions can differ between MPI decompositions without any
# per-cell state differing. When that happens the run takes DIFFERENT STEPS on different core counts
# and therefore lands on a different -- and equally valid -- answer, which no comparison tolerance
# can reconcile. This test exists because that is exactly what happened:
#
#   The active-set pin committed cells a few ULPs ABOVE their constraint (the residual IS the water
#   table there, while the SNES variable is the HEAD of order topo, and snes_stol is a RELATIVE STEP
#   tolerance -- so the constraint could only ever be met to within ULP(topo), and WHICH ulp depended
#   on the arithmetic path). FSM then correctly read wtd > 0 as surface water and rewrote the cell,
#   making its FSM delta nonzero; the estimator excludes delta-carrying cells by an exact `!= 0.0`
#   test, so its RMS DIVISOR became decomposition-dependent (99 / 164 / 100 of 196 land cells) while
#   the NUMERATOR was bit-identical. est moved 22%, the growth factor moved, the step sizes moved.
#   Fixed by projecting pinned cells onto their constraint (99a8eee).
#
# ASSERTS THE ESTIMATE ITSELF, not just the answer: the DTTRACE sequence must be identical across
# rank counts. That is the quantity that was wrong, and asserting it directly means a regression is
# reported as "the controller disagreed" rather than as an unexplained field difference.
#
# Everything here is left at its DEFAULT (adaptive_dt auto -> true, fsm_coupling continuous,
# collection.method active_set) because the defaults are what production runs, and the defect lived
# only in that combination.
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM=$(readlink -f "${1:-../../build/wtm.x}")
RANKS="${*:2}"; RANKS="${RANKS:-1 2 4 6}"
[[ -x "$WTM" ]] || { echo "ERROR: WTM binary not found at $WTM" >&2; exit 1; }
[[ -f ../golden/inputs/transient_test_ta_topography.tif ]] || ( cd ../golden && python3 make_transient_inputs.py >/dev/null )
TRANS=$(readlink -f ../golden/inputs)
make_work xrank_adaptive

echo "=== cross-rank determinism under adaptive dt (defaults: adaptive, continuous, active_set) ==="
echo "WTM binary: $WTM"
echo

fail=0
for n in $RANKS; do
    { cat <<EOF
solver_method anderson
run_type           transient
fsm_on             1
evap_mode          0
infiltration_on    0
runoff_ratio_on    0
deltat             252288000
report_interval    1
total_time         32yr
fdepth_a           200
fdepth_b           150
fdepth_fmin        2
time_start         ta
time_end           tb
surfdatadir        $TRANS
region             transient_test
supplied_wt        0
save_nreport_interval 9999
eq_tol 0
textfilename   $WORK/n$n.txt
outfile_prefix $WORK/n${n}_
EOF
      echo "snes_stol 1e-12"
      echo "trace dt"
    } | ../emit_config.sh > "$WORK/n$n.yaml"
    # Long cycles (8 yr) so the controller is FREE to choose the step. With short cycles the step is
    # quantised by the report interval and the controller never binds -- the arm would pass vacuously.
    ( cd "$WORK" && OMP_NUM_THREADS=1 mpirun -n "$n" "$WTM" "n$n.yaml" \
        > "$WORK/n$n.log" 2>&1 ) || { echo "  n=$n: RUN FAILED"; tail -5 "$WORK/n$n.log" | sed 's/^/      /'; fail=1; continue; }
    grep -oE 'dt=[0-9.e+-]+ est=[0-9.e+-]+' "$WORK/n$n.log" > "$WORK/est_n$n.txt"
done

PY=${PYTHON:-python3}
TESTS="$(readlink -f ..)" "$PY" - "$WORK" $RANKS <<'PYEOF' || fail=1
import sys, glob, os
import numpy as np, rasterio
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL              # latest_output: refuses a match from a DIFFERENT stem
work, ranks = sys.argv[1], [int(r) for r in sys.argv[2:]]
ok = True
def check(label, cond, msg):
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}: {msg}")
    ok = ok and cond

def last(n):
    # latest_output, not a bare glob: `n1_` would also match `n1x_...` if a stem were ever added
    # whose name extends this one, and the last match would silently be the wrong run.
    with rasterio.open(VOL.latest_output(os.path.join(work, f"n{n}_"))) as s:
        a = s.read(1).astype(float); nod = s.nodata
    return np.where(a == nod, np.nan, a) if nod is not None else a

def parse(n):
    dts, ests = [], []
    for l in open(os.path.join(work, f"est_n{n}.txt")):
        a, b = l.split()
        dts.append(float(a.split('=')[1])); ests.append(float(b.split('=')[1]))
    return dts, ests
seq = {n: parse(n) for n in ranks}
est = {n: seq[n][1] for n in ranks}
ref = ranks[0]

# A global reduction cannot be bit-reproducible across decompositions, so compare RELATIVELY. The
# margin is MEASURED, not guessed. Reduction noise on this fixture, accumulated over the 25 steps:
# 2.8e-08 (n=2), 4.4e-08 (n=4), 8.4e-08 (n=6). The defect this test exists for was 2.2e-01 relative
# (est 4.008e-02 at n=1 against 3.114e-02 at n=2). Those are 6.4 orders apart, and 1e-4 is the
# GEOMETRIC MIDDLE of the two -- ~1000x above the noise floor, ~2000x below the defect. Do not tighten
# it toward the noise: the noise is real and will grow with the cell count.
REL = 1e-4
def relmax(a, b):
    return max(abs(x - y) / max(abs(x), abs(y), 1e-300) for x, y in zip(a, b))

# PRECONDITION -- the controller must actually be choosing steps, or every assertion below is vacuous.
nsteps = len(est[ref])
ncycles = 4
check("PRECONDITION controller binds", nsteps > ncycles,
      f"{nsteps} adaptive steps over {ncycles} cycles (must exceed 1/cycle, else the step is quantised "
      f"by the report interval and nothing is being tested)")
nonzero = sum(1 for e in est[ref] if e > 0.0)
check("PRECONDITION estimate is live", nonzero >= nsteps - 1,
      f"{nonzero}/{nsteps} steps have a nonzero error estimate")

# THE ASSERTIONS: the controller's own decisions must not depend on the decomposition.
for n in ranks[1:]:
    # (a) the number of steps -- a discrete decision, so this must match exactly
    check(f"step COUNT n={ref} vs n={n}", len(est[n]) == len(est[ref]),
          f"{len(est[ref])} vs {len(est[n])} steps")
    if len(est[n]) != len(est[ref]):
        continue
    # (b) the step SIZES -- what actually determines the trajectory
    d = relmax(seq[ref][0], seq[n][0])
    check(f"step SIZE sequence n={ref} vs n={n}", d < REL, f"max relative difference {d:.3e} (tol {REL:.0e})")
    # (c) the estimate itself -- the quantity that was decomposition-dependent
    d = relmax(est[ref], est[n])
    check(f"error ESTIMATE sequence n={ref} vs n={n}", d < REL, f"max relative difference {d:.3e} (tol {REL:.0e})")

# And the answer itself, which follows from the above but is what a user actually sees.
r = last(ref)
for n in ranks[1:]:
    d = float(np.nanmax(np.abs(last(n) - r)))
    check(f"final water table n={ref} vs n={n}", d < 1e-9, f"max|delta| = {d:.3e} m (tol 1e-09)")

sys.exit(0 if ok else 1)
PYEOF

echo
[[ $fail -eq 0 ]] && echo "CROSS-RANK ADAPTIVE DETERMINISM PASSED" || echo "CROSS-RANK ADAPTIVE DETERMINISM FAILED" >&2
exit $fail
