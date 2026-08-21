#!/usr/bin/env bash
# Active-set / semismooth seepage-face regression (-wtm_dev_active_set).
#
# The active-set pin enforces the wtd<=0 seepage complementarity INSIDE the matrix-free Anderson residual
# (min-NCP f = max(w_c, f)), so it supersedes the runoff_collector enforcement entirely: every above-surface
# cell is pinned to wtd=0 EXACTLY and its water is handed to FillSpillMerge. The payoff (see
# benchmark/FSM_EVERY_STEP_DESIGN.md and finding_active_set_seepage_spike) is that the FSM-on equilibrium
# becomes INDEPENDENT of the runoff_collector choice -- the collector x FSM coupling ambiguity is dissolved.
#
# On the fsm_test fixture (a plateau with an off-centre depression, surface water supplied) this test asserts,
# on the Anderson path with FSM on:
#   PIN      : with active-set, max wtd = 0 exactly (< 1e-6 m) for implicit/explicit/off -- a pin, not a pile.
#   COLLECTOR-INDEPENDENT : with active-set, implicit == explicit == off to machine zero (< 1e-9 m spread).
#   BITE     : WITHOUT active-set, the collector choice moves the equilibrium (implicit vs explicit spread
#              > 0.1 m) -- proving the independence above is the active-set doing work, not a vacuous fixture.
#
# active-set is EXPERIMENTAL and OFF BY DEFAULT (-wtm_dev_active_set); it is NON-PHYSICAL for lakes until it is
# lake-aware (pins the lake column to wtd=0, removing lake pressure -- see task #119). This test guards the
# mechanism as it stands, not a production default.
#
# Usage:  tests/active_set/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }

# Reuse the fsm_consistency fixture (the fsm_test region), as the golden suite does.
FSMDIR=$(readlink -f ../fsm_consistency)
[[ -f "$FSMDIR/inputs/fsm_test_t0_topography.tif" ]] || ( cd "$FSMDIR" && python3 make_inputs.py >/dev/null )
INP="$FSMDIR/inputs"
WORK=$(mktemp -d /tmp/as_XXXX); trap 'rm -rf "$WORK"' EXIT
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

emit() { # stem  collector
  cat > "$WORK/$1.cfg" <<EOF
run_type equilibrium
fsm_on 1
evap_mode 0
infiltration_on 0
runoff_ratio_on 0
runoff_collector $2
cells_per_degree 10
southern_edge -45
deltat 31536000
total_time 6yr
report_interval 2
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
time_start t0
time_end t0
surfdatadir $INP
region fsm_test
supplied_wt 1
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
save_nreport_interval 9999
EOF
}
run() { # stem  collector  extra-flags
  emit "$1" "$2"
  "$WTM" "$WORK/$1.cfg" -wtm_anderson $3 -wtm_eq_tol 0 > "$WORK/$1.log" 2>&1 \
    || { echo "RUN FAILED: $1"; tail -3 "$WORK/$1.log"; exit 2; }
}
# Without active-set: the collector choice is a live variable (the BITE).
run imp_plain implicit ""
run exp_plain explicit ""
# With active-set: it supersedes the collector, so all three must agree exactly.
run imp_as implicit "-wtm_dev_active_set"
run exp_as explicit "-wtm_dev_active_set"
run off_as off      "-wtm_dev_active_set"

IP=$(ls "$WORK"/imp_plain_*.tif | tail -1); EP=$(ls "$WORK"/exp_plain_*.tif | tail -1)
IA=$(ls "$WORK"/imp_as_*.tif | tail -1);    EA=$(ls "$WORK"/exp_as_*.tif | tail -1)
OA=$(ls "$WORK"/off_as_*.tif | tail -1)
"$PY" - "$IP" "$EP" "$IA" "$EA" "$OA" <<'PY'
import sys, numpy as np, rasterio
ip, ep, ia, ea, oa = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:6]]
def interior(a): return a[1:-1, 1:-1]
ip, ep, ia, ea, oa = map(interior, (ip, ep, ia, ea, oa))
pin        = max(float(ia.max()), float(ea.max()), float(oa.max()))
indep      = max(float(np.max(np.abs(ia - ea))), float(np.max(np.abs(ia - oa))))
bite       = float(np.max(np.abs(ip - ep)))
ok = True
def check(name, cond, detail):
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'} {name}: {detail}"); ok = ok and cond
check("PIN (active-set pins wtd=0 exactly)",        pin < 1e-6,
      f"max wtd over {{implicit,explicit,off}} = {pin:.3e} m")
check("COLLECTOR-INDEPENDENT (active-set)",         indep < 1e-9,
      f"max spread implicit/explicit/off = {indep:.3e} m")
check("BITE (collectors diverge without active-set)", bite > 0.1,
      f"max|implicit - explicit| (no active-set) = {bite:.4f} m")
print("PASS: active-set pins the seepage face and dissolves the collector x FSM ambiguity"
      if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
