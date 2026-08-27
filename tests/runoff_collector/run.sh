#!/usr/bin/env bash
# runoff_collector selector: the input-file choice of how the wtd<=0 exfiltration constraint is enforced. One boundary
# condition, three enforcements (see benchmark/SURFACE_WATER_ROUTING.md):
#   implicit : in-residual exfiltration (direct_to_runoff) -- pins wtd=0, dt-independent, exact (Anderson today).
#   explicit : post-solve clamp -- robust on every solver, a dt-lagged form of the same face.
#   off      : no collection -- above-surface water piles up (NONPHYSICAL; warns).
#   extended_soil : also piles, but continues the AQUIFER above the surface (storativity stays porosity,
#              T never clamps), so it removes the wtd=0 free boundary instead of leaving it unenforced.
#              NONPHYSICAL/[WIP]. `-wtm_extended_soil` is the legacy alias that selects it.
# On a partial-exfiltration fixture (interior driven to the surface) these are distinguishable by the peak water
# table. Asserts, on the matrix-free Anderson path unless noted:
#   IMPLICIT : table pinned at the surface (0 <= max wtd < 0.5 m: a exfiltration constraint, not a pile) with exfiltrating cells.
#   EXPLICIT : table clamped to exactly the surface (|max wtd| < 1e-4 m) with exfiltrating cells.
#   OFF      : water piles far above the surface (max wtd > 5 m) AND the nonphysical warning is printed.
#   UNSET    : no collector set -> the DEFAULT applies. That default is now active_set (was implicit),
#              so UNSET is compared against the ACTIVE_SET run -- this is the check that catches a flip.
#              (The legacy band sink is no longer the default; it is covered as an explicit mode in taper / dt_sensitivity.)
#   AGREE    : implicit and explicit land within a few cm (same face, converging as dt->0).
#   SOLVER   : explicit also converges on the default Picard path (it needs no tangent).
# This test asserts the modes via the config KEY, so it also fences off the -wtm_<flag> 0 CLI mis-parse hazard.
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/rcoll_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
WORK=$(mktemp -d /tmp/rc_XXXX); trap 'rm -rf "$WORK"' EXIT
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

emit() { # stem  collector-line
  ../emit_config.sh > "$WORK/$1.yaml" <<EOF
run_type equilibrium
fsm_on 0
evap_mode 0
infiltration_on 0
runoff_ratio_on 0
$2
cells_per_degree 120
southern_edge 0
deltat 2419200
total_time 14515200000s
save_nreport_interval 120
report_interval 50
fdepth_a 100
fdepth_b 150
fdepth_fmin 2
time_start ta
time_end tb
surfdatadir $INP
region rcoll
supplied_wt 1
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}
run() { # stem  collector-line  extra-flags
  emit "$1" "$2"
  "$WTM" "$WORK/$1.yaml" -wtm_anderson $3 -wtm_eq_tol 0 > "$WORK/$1.log" 2>&1 \
    || { echo "RUN FAILED: $1"; tail -3 "$WORK/$1.log"; exit 2; }
}
run implicit "runoff_collector implicit" ""
run explicit "runoff_collector explicit" ""
run off      "runoff_collector off"      ""
run aset     "runoff_collector active_set" ""
run unset    ""                          ""
# extended_soil: the mode, the legacy alias that selects it, and the supersession when a config names
# a different method. All three are needed -- the alias and the supersession are the halves that used
# to fail SILENTLY, with -wtm_extended_soil printing its banner while a collector quietly overrode it.
run xsoil_mode "runoff_collector extended_soil" ""
run xsoil_flag ""                               "-wtm_extended_soil"
run xsoil_sup  "runoff_collector explicit"      "-wtm_extended_soil"
# explicit on the DEFAULT Picard path (no -wtm_anderson): must converge (no tangent needed)
emit picard "runoff_collector explicit"
"$WTM" "$WORK/picard.yaml" -wtm_eq_tol 0 > "$WORK/picard.log" 2>&1 \
  || { echo "RUN FAILED: explicit on Picard"; tail -3 "$WORK/picard.log"; exit 2; }
OFFWARN=$(grep -c "WARNING \[runoff_collector=off\]" "$WORK/off.log" || true)
# The extended-soil mode must announce itself, and the superseded run must say so rather than silently
# dropping the request. Both are asserted: a silent override is the exact defect these arms exist for.
XSBANNER=$(grep -c "runoff_collector=extended_soil\]: NONPHYSICAL" "$WORK/xsoil_mode.log" || true)
XSUPWARN=$(grep -c "SUPERSEDES it" "$WORK/xsoil_sup.log" || true)

IM=$(ls "$WORK"/implicit_*.tif | tail -1); EX=$(ls "$WORK"/explicit_*.tif | tail -1)
OF=$(ls "$WORK"/off_*.tif | tail -1);      UN=$(ls "$WORK"/unset_*.tif | tail -1)
AS=$(ls "$WORK"/aset_*.tif | tail -1)
XS=$(ls "$WORK"/xsoil_mode_*.tif | tail -1); XF=$(ls "$WORK"/xsoil_flag_*.tif | tail -1)
XP=$(ls "$WORK"/xsoil_sup_*.tif | tail -1)
OFFWARN="$OFFWARN" XSBANNER="$XSBANNER" XSUPWARN="$XSUPWARN" \
  "$PY" - "$IM" "$EX" "$OF" "$UN" "$AS" "$XS" "$XF" "$XP" <<'PY'
import sys, os, numpy as np, rasterio
im, ex, of, un, aset, xs, xf, xp = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:9]]
def interior(a): return a[1:-1, 1:-1]
im_mx, ex_mx, of_mx, un_mx, as_mx = (float(interior(a).max()) for a in (im, ex, of, un, aset))
im_seep = int((interior(im) > -1e-3).sum()); ex_seep = int((interior(ex) > -1e-3).sum())
agree = float(np.max(np.abs(im - ex)))
offwarn = int(os.environ["OFFWARN"]) > 0
ok = True
def check(name, cond, detail):
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'} {name}: {detail}")
    ok = ok and cond
check("IMPLICIT (exfiltration constraint, not piled)", (0.0 - 1e-3 <= im_mx < 0.5) and im_seep > 0,
      f"max wtd = {im_mx:.4f} m, exfiltrating cells = {im_seep}")
check("EXPLICIT (clamped to surface)",      abs(ex_mx) < 1e-4 and ex_seep > 0,
      f"max wtd = {ex_mx:.4e} m, exfiltrating cells = {ex_seep}")
check("OFF (piles + warns)",                of_mx > 5.0 and offwarn,
      f"max wtd = {of_mx:.2f} m, warning printed = {offwarn}")
# UNSET must track the CURRENT default, which is active_set (it was implicit until 2026-08-25). This
# check is the one that catches a default flip, so it compares against the active_set run rather than
# hard-coding a number.
check("UNSET (defaults to active_set)", abs(un_mx - as_mx) < 1e-6,
      f"max wtd = {un_mx:.4f} m (== active_set {as_mx:.4f} m; implicit would be {im_mx:.4f} m)")
check("AGREE implicit vs explicit",         agree < 0.1,
      f"max|implicit - explicit| = {agree:.3e} m")

# --- extended_soil: mode, alias, supersession -----------------------------------------------------
xs_mx, xf_mx, xp_mx = (float(interior(a).max()) for a in (xs, xf, xp))
xs_banner = int(os.environ["XSBANNER"]) > 0
xsup_warn = int(os.environ["XSUPWARN"]) > 0
# It piles like `off` -- but NOT identically to it, and that difference is the point. Both leave wtd<=0
# unenforced; extended soil additionally continues the aquifer upward, so above-surface water fills
# pore space at porosity instead of standing as free surface water, and the same water makes a
# DIFFERENT pile. Asserting "differs from off" keeps this arm from passing on a build where
# extended_soil silently degrades to plain `off`.
check("EXT_SOIL mode (piles, announces, and is NOT `off`)",
      xs_mx > 5.0 and xs_banner and abs(xs_mx - of_mx) > 1e-6,
      f"max wtd = {xs_mx:.2f} m (off = {of_mx:.2f} m), banner printed = {xs_banner}")
# The legacy flag must SELECT the mode when no method is configured. Before extended soil joined the
# enumeration this silently did nothing: the unset config resolved to the default collector, whose pin
# held wtd<=0, and the flag printed its banner while being overridden.
check("EXT_SOIL alias (-wtm_extended_soil alone == the mode)",
      abs(xf_mx - xs_mx) < 1e-6,
      f"flag-alone max wtd = {xf_mx:.4f} m (== mode {xs_mx:.4f} m; default would be {un_mx:.4f} m)")
# A configured method wins over the flag, and SAYS so. The pair is contradictory -- one disposes of
# surface water, the other declares there is none -- so the run must match plain `explicit` exactly and
# must not resolve it in silence.
check("EXT_SOIL supersession (explicit wins, and warns)",
      abs(xp_mx - ex_mx) < 1e-6 and xsup_warn,
      f"max wtd = {xp_mx:.4e} m (== explicit {ex_mx:.4e} m), warning printed = {xsup_warn}")
print("PASS: runoff_collector modes behave as specified" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
checks_rc=$?
echo "  SOLVER: explicit converged on the default Picard path (no tangent needed)"
# Propagate the checks' status. This was previously dropped: the trailing echo returned 0, so a FAILING
# assertion still exited 0 and run_all.sh reported the suite PASS while printing "FAIL" in its output.
exit $checks_rc
