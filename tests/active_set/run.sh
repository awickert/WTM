#!/usr/bin/env bash
# Lake-aware active-set / semismooth exfiltration regression (-wtm_active_set).
#
# The active-set pin enforces the exfiltration complementarity INSIDE the matrix-free Anderson residual, pinned to
# the FSM FREE SURFACE (wtd <= d_pond, d_pond = lagged ponded depth; 0 off lakes) via the min-NCP
# f = max(w_c - d_pond, f). It supersedes the runoff_collector enforcement, so the FSM-on equilibrium is
# INDEPENDENT of the collector choice -- the collector x FSM coupling ambiguity is dissolved -- WHILE keeping
# lakes: a ponded cell holds water up to its stage (its head is felt during the solve), and only the overflow
# above the stage is skimmed to runoff. (See benchmark/FSM_EVERY_STEP_DESIGN.md, project_lake_head_boundary_design.)
#
# On the fsm_test fixture (a plateau with an off-centre depression, surface water supplied), on the Anderson
# path with FSM on, this test asserts:
#   LAKE PERSISTS         : with active-set the lake keeps its head (max wtd well above 0) -- it is NOT
#                           flattened to the land surface (the pre-lake-aware pin gave max wtd = 0).
#   COLLECTOR-INDEPENDENT  : with active-set, implicit == explicit == off to machine zero (< 1e-9 m spread).
#   BITE                   : WITHOUT active-set the collector choice moves the equilibrium (implicit vs
#                            explicit spread > 0.05 m) -- proving the independence is the pin doing work.
#
# active-set is EXPERIMENTAL and OFF BY DEFAULT (-wtm_active_set). Anderson residual only for now.
#
# Usage:  tests/active_set/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }

# Reuse the fsm_consistency fixture (the fsm_test region), as the golden suite does.
FSMDIR=$(readlink -f ../fsm_consistency)
[[ -f "$FSMDIR/inputs/fsm_test_t0_topography.tif" ]] || ( cd "$FSMDIR" && python3 make_inputs.py >/dev/null )
INP="$FSMDIR/inputs"
make_work as
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

emit() { # stem  collector
  ../emit_config.sh > "$WORK/$1.yaml" <<EOF
solver_method anderson
run_type equilibrium
total_time 6yr
supplied_wt 1
deltat 31536000
report_interval 2
save_nreport_interval 9999
cells_per_degree 10
southern_edge -45
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
infiltration_on 0
fsm_on 1
runoff_collector $2
surfdatadir $INP
region fsm_test
time_start t0
time_end t0
eq_tol 0
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}
run() { # stem  collector  extra-flags
  emit "$1" "$2"
  "$WTM" "$WORK/$1.yaml" $3 > "$WORK/$1.log" 2>&1 \
    || { echo "RUN FAILED: $1"; tail -3 "$WORK/$1.log"; exit 2; }
}
# Without active-set: the collector choice is a live variable (the BITE).
run imp_plain implicit ""
run exp_plain explicit ""
# Lake-aware active-set, now selected as a MODE (collection.method: active_set) rather than by a flag
# that superseded whatever collector was configured.
#
# THE COLLECTOR-INDEPENDENCE ARM IS GONE, and deliberately, not by oversight. It ran implicit/explicit/off
# each with -wtm_active_set on top and asserted the three agreed to 1e-9: the flag was an ORTHOGONAL
# switch, so "which collector did you ask for" was a live variable that active-set had to dissolve. As a
# member of the collection.method enumeration, active_set is mutually exclusive with the other five -- the
# three configs would now be textually identical and the assertion could not fail. That is a genuine loss
# of a property, not a rename: the supersession it tested no longer exists to be tested.
run as active_set ""

IP=$(ls "$WORK"/imp_plain_*.tif | tail -1); EP=$(ls "$WORK"/exp_plain_*.tif | tail -1)
IA=$(ls "$WORK"/as_*.tif | tail -1)
"$PY" - "$IP" "$EP" "$IA" <<'PY'
import sys, numpy as np, rasterio
ip, ep, ia = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:4]]
def interior(a): return a[1:-1, 1:-1]
ip, ep, ia = map(interior, (ip, ep, ia))
lake_head = float(ia.max())
bite      = float(np.max(np.abs(ip - ep)))
# active_set must also DIFFER from both plain collectors -- otherwise this arm is measuring nothing.
differs   = min(float(np.max(np.abs(ia - ip))), float(np.max(np.abs(ia - ep))))
ok = True
def check(name, cond, detail):
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'} {name}: {detail}"); ok = ok and cond
check("LAKE PERSISTS (head kept, not flattened)", lake_head > 1.0,
      f"max wtd with active-set = {lake_head:.4f} m (lake stage; the pre-lake-aware pin gave 0)")
check("DISTINCT (active-set is not either plain collector)", differs > 1e-6,
      f"min|active_set - {{implicit,explicit}}| = {differs:.3e} m")
check("BITE (collectors diverge without active-set)", bite > 0.05,
      f"max|implicit - explicit| (no active-set) = {bite:.4f} m")
print("PASS: lake-aware active-set keeps the lake's head and differs from both plain collectors"
      if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
