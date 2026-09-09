#!/usr/bin/env bash
# Adaptive-dt + pure-water-depth-metric regression. On a small coastal wedge, a cold-start equilibrium must
# reach the SAME steady water table regardless of time-integration scheme or equilibrium-stop metric. Guards:
#   (1) -wtm_tr_bdf2 -wtm_dt_adaptive converges AND matches backward-Euler (cc) -> adaptive reaches the
#       correct equilibrium (not just "runs").
#   (2) -wtm_eq_metric water-rms stops on the pure-water-depth metric (|S*Δwtd|) AND reaches the same table.
# Bites if the adaptive controller or the water-depth metric ever produces a wrong field or fails to stop.
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
# .tif inputs are gitignored -> generate them if absent (needs rasterio, like the other suites)
[[ -f inputs/adwater_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
make_work adw
# metres OF WATER VOLUME (tests/wtm_volume.py), not head -- see #61/#65. Uniform phi = 0.25 here, so this is
# the old 0.05 m head bound x0.25 exactly.
TOL="${TOL:-0.0125}"     # cross-scheme steady-state agreement, in water
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

emit() { # $1 stem  [env: INTEG= ADAPT_MODE= EQ_TOL=]
  # BOTH keys are emitted UNCONDITIONALLY, with the control arm's values as the defaults. They used to be
  # emitted only when the caller set them (`${INTEG:+...}`), which left the key ABSENT -- and absent means
  # `auto`, which resolves to tr-bdf2 on the Anderson path and to adaptive under any non-implicit
  # collector. So all three arms ran tr_bdf2 + adaptive + active_set: the `cc` arm named as the
  # backward-Euler fixed-step CONTROL was a second copy of `adapt`, and guard (1) -- "adaptive matches
  # backward-Euler" -- compared adaptive with itself. See #24, #37.
  ../emit_config.sh > "$WORK/$1.yaml" <<EOF
solver_method anderson
time_integration ${INTEG:-backward-euler}
time_step_mode ${ADAPT_MODE:-fixed}
run_type equilibrium
fsm_on 0
evap_mode 0
infiltration_on 0
runoff_ratio_on 0
deltat 2419200
total_time 24192000000s
save_nreport_interval 200
report_interval 50
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
time_start ta
time_end tb
surfdatadir $INP
region adwater
supplied_wt 0
eq_tol ${EQ_TOL:-0.001}
eq_metric rms
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}

BB=""
emit cc; ADAPT_MODE=adaptive INTEG=tr-bdf2 emit adapt; EQ_TOL=0.0005 emit water
export WTM_COVERAGE_LOG="${WTM_COVERAGE_LOG:-$WORK/coverage.txt}"
# The whole point of this suite is that DIFFERENT schemes reach the SAME equilibrium, so each arm has to
# prove it ran the scheme it names. Checked against the fingerprint the model writes, not the config we
# think we wrote -- that is what caught all three arms silently sharing one configuration.
go() { # $1 stem  $2.. expected resolutions
    local stem="$1"; shift
    WTM_COVERAGE_TAG="adaptive_water/$stem" "$WTM" "$WORK/$stem.yaml" $BB > "$WORK/$stem.log" 2>&1 \
      || { echo "RUN FAILED: $stem"; tail -3 "$WORK/$stem.log"; exit 2; }
    expect_resolved "$WTM_COVERAGE_LOG" "$@" >/dev/null || exit 3
}
go cc    integrator=be_volume dtctl=fixed
go adapt integrator=tr_bdf2   dtctl=adaptive
go water integrator=be_volume dtctl=fixed

# (1) adaptive must have actually reached equilibrium (not hit the total_time cap)
grep -q "equilibrium reached" "$WORK/adapt.log" || { echo "FAIL: adaptive did not reach equilibrium"; exit 1; }
# (2) the water arm must stop on the (water-based) rms metric at the tighter 0.5 mm tolerance -- every metric
#     judges water moved |S*Δwtd| now, so this exercises a tighter pure-water-depth stop than cc's 1 mm.
grep -q "equilibrium reached (rms metric)" "$WORK/water.log" \
  || { echo "FAIL: the rms (water-depth) metric did not drive the stop"; grep -i "equilibrium reached" "$WORK/water.log"; exit 1; }

CC=$(ls "$WORK"/cc_*.tif | tail -1); AD=$(ls "$WORK"/adapt_*.tif | tail -1); WA=$(ls "$WORK"/water_*.tif | tail -1)
TOL="$TOL" PHI="$(readlink -f inputs/adwater_porosity.tif)" TESTS="$(readlink -f ..)" \
  "$PY" - "$CC" "$AD" "$WA" <<'PY'
import sys, os, numpy as np, rasterio
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL                      # ONE verified V(wtd); see tests/verify_wtm_volume.sh
cc, ad, wa = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:4]]
phi = VOL.read_band(os.environ["PHI"])
m = np.ones_like(cc, bool); m[:, 0] = False   # exclude the ocean column
d_ad = float(VOL.volume_diff(ad, cc, phi)[m].max()); d_wa = float(VOL.volume_diff(wa, cc, phi)[m].max())
tol = float(os.environ["TOL"])
print(f"  adaptive (tr-bdf2+dt_adaptive) vs cc: max|ΔV| = {d_ad:.4f} m water")
print(f"  water-depth metric vs cc:            max|ΔV| = {d_wa:.4f} m water volume  (tol {tol})")
if d_ad <= tol and d_wa <= tol:
    print("PASS: adaptive dt and the pure-water-depth stop metric both reach cc's equilibrium")
    sys.exit(0)
print(f"FAIL: adaptive={d_ad:.4f}, water={d_wa:.4f} m water volume exceed tol {tol} m water")
sys.exit(1)
PY
