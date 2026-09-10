#!/usr/bin/env bash
# SECANT ≡ VOLUME backward-Euler storage equivalence (unit/regression test).
#
# WTM's default backward Euler forms the storage term as S·Δh with the EXACT secant effective storativity
# S = (V(wⁿ⁺¹) − V(wⁿ))/(wⁿ⁺¹ − wⁿ); -wtm_volume_storage uses the stored-volume change ΔV directly. Since S
# is the exact secant, S·Δh ≡ ΔV identically (even across the surface where dV/dh jumps porosity→~1). So on
# a well-behaved (non-oscillating) domain the two must agree to MACHINE PRECISION.
#
# This guards that identity: it bites if updateEffectiveStorativity ever stops being the exact secant (e.g.
# a tangent or endpoint storativity), which would make the default BE storage inconsistent with the volume
# schemes. See finding on the (retracted) "secant storage inconsistency" -- there is none, and this proves it.
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
# .tif inputs are gitignored -> generate them if absent (needs rasterio, like the other suites)
[[ -f inputs/storeq_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
make_work storeq
# metres OF WATER VOLUME (|V(wtd_a)-V(wtd_b)|, tests/wtm_volume.py), not head: the model conserves water
# and judges every stopping criterion in water volume (#61/#65). Uniform phi = 0.25 on this fixture, so this
# is the old 1e-6 head bound x0.25 exactly -- the same strictness, correctly labelled.
TOL="${TOL:-2.5e-7}"      # machine-precision agreement expected (observed ~1e-12 head)
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

# Both pins below are load-bearing, and neither was here before 26ad949 -- which made this test VACUOUS.
#   time_integration: with auto now resolving anderson -> tr-bdf2, the storage branch of the residual is
#     never reached at all (tr_stage and bdf2v both take precedence), so neither arm would exercise the
#     thing under test.
#   runoff_collector: the default is active_set, which REQUIRES the b=0 volume path, so
#     dev.storage_form: secant with it now aborts by name. explicit is the collector this identity can
#     be measured under.
emit() { # $1 stem  [env: STORAGE=volume|secant]
  ../emit_config.sh > "$WORK/$1.yaml" <<EOF
snes_stol 1e-10
solver_method anderson
time_integration backward-euler
runoff_collector explicit
run_type transient
${STORAGE:+storage $STORAGE}
fsm_on 0
infiltration_on 0
runoff_ratio_on 0
deltat 2419200
total_time 1451520000s
save_nreport_interval 3
report_interval 200
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
time_start ta
time_end tb
surfdatadir $INP
region storeq
supplied_wt 0
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}

# mask-aware ghost boundary (now the default) so edge land cells are not forced against a
# hard-draining topo-0 ocean pad. Under the old padding, boundary-adjacent SURFACE cells drain so hard that
# the ÷S (secant) vs ÷Sy (tangent) residual scaling leaves a ~1e-4 convergence-region difference there --
# a scaling/conditioning artifact, NOT the identity failing. The ghost boundary removes that edge stress so
# the S·Δh ≡ ΔV identity shows at machine precision (observed ~1e-15) and the test is a clean invariant check.
# STORAGE=secant is now EXPLICIT. It used to be left unset, relying on secant being the default -- and
# when dev.storage_form's default became volume (879a188), that made this test compare volume against
# volume, i.e. a config against ITSELF. It still reported max|dwtd| = 0.000e+00 and still PASSED, which
# is exactly what a vacuous arm looks like.
STORAGE=secant emit secant; STORAGE=volume emit volume
"$WTM" "$WORK/secant.yaml"                     > "$WORK/secant.log" 2>&1 \
  || { echo "RUN FAILED: secant"; tail -3 "$WORK/secant.log"; exit 2; }
"$WTM" "$WORK/volume.yaml" > "$WORK/volume.log" 2>&1 \
  || { echo "RUN FAILED: volume"; tail -3 "$WORK/volume.log"; exit 2; }

# NON-VACUITY GATE. This test compares two runs and asserts they AGREE, so it passes trivially if the
# two arms are the same run -- and that is exactly what happened when dev.storage_form's default became
# volume (879a188) while the secant arm relied on the default: it compared volume against volume,
# reported max|dwtd| = 0.000e+00, and PASSED. An agreement test must prove its arms differ before its
# agreement means anything. Checked on the emitted CONFIGS, which is where the failure actually was.
SEC_FORM=$(grep -oE "storage_form: [a-z]+" "$WORK/secant.yaml" | head -1)
VOL_FORM=$(grep -oE "storage_form: [a-z]+" "$WORK/volume.yaml" | head -1)
echo "  arms: secant -> ${SEC_FORM:-<unset, i.e. the DEFAULT>}   volume -> ${VOL_FORM:-<unset, i.e. the DEFAULT>}"
if [ "$SEC_FORM" != "storage_form: secant" ] || [ "$VOL_FORM" != "storage_form: volume" ]; then
    echo "FAIL: the two arms are not the two forms -- this test would be VACUOUS." >&2
    echo "      Each arm must set dev.storage_form EXPLICITLY; relying on the default is what made it" >&2
    echo "      compare a configuration against itself once that default changed." >&2
    exit 1
fi

SEC=$(ls "$WORK"/secant_*.tif | tail -1)
VOL=$(ls "$WORK"/volume_*.tif | tail -1)
TOL="$TOL" PHI="$INP/storeq_porosity.tif" TESTS="$(readlink -f ..)" "$PY" - "$SEC" "$VOL" <<'PY'
import sys, os, numpy as np, rasterio
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL                      # ONE verified V(wtd); see tests/verify_wtm_volume.sh

sec, vol = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:3]]
m = np.ones_like(sec, bool); m[:, 0] = False   # interior + land edges (exclude ocean column)
phi = VOL.read_band(os.environ["PHI"])
d = float(VOL.volume_diff(sec, vol, phi)[m].max())
tol = float(os.environ["TOL"])
print(f"  secant-BE vs volume-BE (S·Δh ≡ ΔV): max|ΔV| = {d:.3e} m water volume  (tol {tol})")
if d <= tol:
    print("PASS: exact-secant storativity makes the two forms identical (machine precision)")
    sys.exit(0)
print(f"FAIL: {d:.3e} m > {tol} m -> the BE secant storage is NOT the exact volume secant")
sys.exit(1)
PY
