#!/usr/bin/env bash
# LIMIT-CYCLE ("weird golden") diagnostic test -- the useful BAD result, remembered and turned into a guard.
#
# ██ THE BARE FLICKER IS AN EXPECTED *WRONG* SOLUTION -- IT IS NOT A TARGET AND MUST NOT BE "FIXED" HERE. ██
# It is a non-physical, unconverged limit cycle that we DELIBERATELY PRESERVE because it is a useful
# DIAGNOSTIC: when the surface OVERSHOOT is unmanaged, different volume-resolving time-integration schemes
# (backward-Euler cc vs BDF2-on-V, etc.) settle into DIFFERENT flickering wrong states, so their
# disagreement flags the bad free boundary. The real FIX is the post-solve clamp
# (-wtm_surface_exfiltration_to_runoff), which this test also verifies. If you make the BARE flicker go
# away, you have changed the overshoot handling -- update this test on purpose, do not silence it.
#
# On a plateau whose interior mound rises to the free boundary (wtd=0), backward Euler + Anderson OVERSHOOT
# the surface each step (storativity jump + seepage kink), producing a period-2 LIMIT CYCLE. This is the
# lakeshore flicker. Two facts we want to REMEMBER and monitor:
#   (1) BARE (no surface clamp) the water table FLICKERS -- per-cycle |Δwtd| never decays -- and different
#       time-integration schemes (cc vs BDF2-on-V) land on DIFFERENT flickering states. So running multiple
#       methods is itself an OVERSHOOT DIAGNOSTIC: their disagreement flags an unmanaged free boundary.
#   (2) The post-solve clamp (-wtm_surface_exfiltration_to_runoff) pins wtd<=0 and the cycle is SUPPRESSED
#       (per-cycle Δ -> 0, steady state) and the schemes RECONCILE (agree to machine precision).
#
# The test asserts all four -- flicker + disagreement bare, quiet + agreement clamped -- so a future change
# to the overshoot handling (good OR bad) is caught. See finding_lakeshore_flicker / _bounce_activeset.
# NOTE: the clamp is now DEFAULT ON (physical runs never flicker), so "bare" here explicitly requests the
# nonphysical regime via the developer switch -wtm_allow_surface_ponding to reach the flicker.
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/limitcyc_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
WORK=$(mktemp -d /tmp/limitcyc_XXXX); trap 'rm -rf "$WORK"' EXIT
PY="${PY:-python3}"
export OMP_NUM_THREADS=1
# thresholds (observed: flicker 1.92, disagree 8.4e-3, clamped 0/0) with wide margins
FLICK="${FLICK:-0.5}"       # bare per-cycle Δ must EXCEED this (limit cycle present)
DIVERGE="${DIVERGE:-1e-3}"  # bare cc-vs-bdf2v must EXCEED this (methods disagree)
QUIET="${QUIET:-1e-4}"      # clamped per-cycle Δ must be BELOW this (suppressed)
RECON="${RECON:-1e-4}"      # clamped cc-vs-bdf2v must be BELOW this (reconciled)

emit() { cat > "$WORK/$1.cfg" <<EOF
run_type transient
fsm_on 0
evap_mode 0
infiltration_on 0
runoff_ratio_on 0
cells_per_degree 1
southern_edge 0
deltat 604800
total_cycles 60
cycles_to_save 60
maxiter 200
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
time_start ta
time_end tb
surfdatadir $INP
region limitcyc
supplied_wt 0
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}
run() { emit "$1"; "$WTM" "$WORK/$1.cfg" $2 -snes_stol 1e-8 > "$WORK/$1.log" 2>&1 \
        || { echo "RUN FAILED: $1"; tail -3 "$WORK/$1.log"; exit 2; }; }
# BARE = the nonphysical unmanaged-surface regime. The clamp is now DEFAULT ON, so we must explicitly
# request ponding with the developer switch to reach the flicker (otherwise "bare" would be clamped).
PONDING="-wtm_allow_surface_ponding"
CLAMP="-wtm_surface_exfiltration_to_runoff"   # now the default; kept explicit for clarity
run cc_bare  "-wtm_anderson $PONDING"
run bd_bare  "-wtm_anderson -wtm_bdf2_on_V $PONDING"
run cc_clamp "-wtm_anderson $CLAMP"
run bd_clamp "-wtm_anderson -wtm_bdf2_on_V $CLAMP"

FLICK="$FLICK" DIVERGE="$DIVERGE" QUIET="$QUIET" RECON="$RECON" "$PY" - "$WORK" <<'PY'
import sys, os, glob, re, numpy as np, rasterio
work = sys.argv[1]
def pcd(s): return float(re.findall(r"per-cycle max\|Δwtd\| = ([0-9.eE+-]+)", open(f"{work}/{s}.log").read())[-1])
def fld(s): return rasterio.open(sorted(glob.glob(f"{work}/{s}_*.tif"))[-1]).read(1).astype(float)
cc_b, bd_b, cc_c, bd_c = fld("cc_bare"), fld("bd_bare"), fld("cc_clamp"), fld("bd_clamp")
m = np.ones_like(cc_b, bool); m[0,:]=m[-1,:]=m[:,0]=m[:,-1]=False
mx = lambda a,b: float(np.max(np.abs((a-b)[m])))
flick, diverge = pcd("cc_bare"), mx(cc_b, bd_b)
quiet, recon   = pcd("cc_clamp"), mx(cc_c, bd_c)
FLICK,DIVERGE,QUIET,RECON = (float(os.environ[k]) for k in ("FLICK","DIVERGE","QUIET","RECON"))
print("  NOTE: the BARE flicker below is an EXPECTED *WRONG* (non-physical) solution, kept as a diagnostic.")
print(f"  BARE (known-WRONG): per-cycle Δ = {flick:.3f} (> {FLICK}?)   cc-vs-bdf2v = {diverge:.3e} m (> {DIVERGE}?)")
print(f"  CLAMPED (the fix):  per-cycle Δ = {quiet:.3e} (< {QUIET}?)   cc-vs-bdf2v = {recon:.3e} m (< {RECON}?)")
ok = (flick > FLICK) and (diverge > DIVERGE) and (quiet < QUIET) and (recon < RECON)
if ok:
    print("PASS: the wrong flicker is present bare (and volume-resolving methods disagree, diagnosing the")
    print("      surface overshoot); the clamp suppresses the cycle and reconciles the schemes.")
    sys.exit(0)
print("FAIL: the surface-overshoot limit-cycle signature CHANGED. This is not a solution regression to")
print("      silence -- the overshoot handling moved (flicker suppressed, or clamp broken). Update on purpose.")
sys.exit(1)
PY
