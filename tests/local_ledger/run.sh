#!/usr/bin/env bash
# LOCAL-IN-SPACE water ledger: did the water go to the right PLACE, not just in the right amount?
#
# THE GAP THIS CLOSES. Every budget check in this suite is a GLOBAL sum, or global-per-cycle. A sum is
# unchanged when water is moved from one cell to another, so a whole class of error is invisible to all
# of them: forcing applied to the wrong cell, a scatter/gather that transposes, an axis swap, a
# depression filled in the wrong basin. tests/fsm_conservation states the limitation in its own header
# -- "conservation catches water CREATED/DESTROYED but not MISPLACED" -- and nothing had closed it.
# WTM has had at least two real bugs of exactly this shape: the lat/lon swap, and the E-W/N-S
# cell-size swap.
#
# TWO ARMS, each isolating one half of "misplaced".
#
#   A. PLACEMENT. Lateral conductivity is ~zero (ksat 1e-12), so every cell is an isolated column and
#      the water table must rise by EXACTLY the local forcing. Checked per cell against an INDEPENDENT
#      analytic expectation computed from the input rasters -- porosity * dwtd == precip/yr * years --
#      with no reference to any model internal. The precipitation field varies as 0.03*x + 0.007*y, so
#      it is not symmetric under transpose and an axis swap cannot pass.
#
#   B. REDISTRIBUTION. A closed domain (all land, no-flow edges) with ZERO forcing and a mound in the
#      starting table. Nothing can enter or leave, so total stored volume must hold constant while the
#      table visibly moves. This is the sharp probe of the lateral flux operator: any face whose flux
#      is not exactly antisymmetric between the two cells sharing it creates or destroys water. The
#      non-triviality check (the mound must actually spread) is what stops it passing vacuously.
#
# Usage:  tests/local_ledger/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/ledgerA_t0_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
WORK=$(mktemp -d /tmp/ledger_XXXX); trap 'rm -rf "$WORK"' EXIT
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

mkcfg() { # $1 stem, $2 region, $3 total_time, $4 deltat, $5 collector, $6 report_interval, $7 runoff_ratio
    { cat <<EOF
run_type equilibrium
total_time $3
supplied_wt 1
deltat $4
report_interval ${6:-1}
runoff_ratio ${7:-0}
save_nreport_interval 1
cells_per_degree 10
southern_edge -45
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
infiltration_on 0
fsm_on 0
surfdatadir $INP
region $2
time_start t0
time_end t0
eq_tol 0
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
      [ -n "${5:-}" ] && echo "runoff_collector $5"; } | ../emit_config.sh > "$WORK/$1.yaml"
}

run() { # $1 stem, $2 region, $3 total_time, $4 deltat, $5 collector, $6 ri, $7 rr, $8.. flags
    local stem="$1" region="$2" tt="$3" dt="$4" coll="$5" ri="$6" rr="$7"; shift 7
    mkcfg "$stem" "$region" "$tt" "$dt" "$coll" "$ri" "$rr"; rm -f "$WORK/$stem.txt" "$WORK/${stem}_"*.tif
    if ! "$WTM" "$WORK/$stem.yaml" "$@" -snes_stol 1e-10 > "$WORK/$stem.log" 2>&1; then
        echo "  RUN FAILED: $stem"; grep -m2 -iE "what\(\)|ERROR" "$WORK/$stem.log" | sed 's/^/        /'
        return 1
    fi
}

echo "=== local-in-space water ledger ==="
echo "WTM binary: $WTM"
echo
fail=0
run place  ledgerA "2yr"  15768000 ""    1 0 -wtm_anderson || fail=1   # dt = 0.5 yr, 4 steps
# Arm B pins runoff_collector=off ON PURPOSE, to isolate the lateral flux operator. Under the default
# active_set the multiplier max(0, -f*Sy) is captured for EVERY cell, not only pinned ones, so
# freely-solving cells contribute the RECTIFIED part of their converged residual noise -- a small,
# systematically positive total_surface_removed even here, where the table sits 8-20 m below the
# surface and nothing can exfiltrate (measured 31.27 m^3 against 2.16e11 m^3 stored, i.e. 1.4e-10).
# It does not move any water -- stored_volume drift is 0.000e+00 either way -- but it would make the
# CLOSED precondition below meaningless. See task #17.
run redist ledgerB "20yr" 31536000 "off" 1 0 -wtm_anderson || fail=1   # dt = 1 yr, 20 steps

# C. CADENCE -- and be precise about WHOSE cadence, because it is easy to get wrong.
#
# FillSpillMerge is NOT on a configurable cadence: with fsm_on it runs once per ACCEPTED STEP, always
# (WTM.cpp -- "tight coupling: FillSpillMerge runs EVERY step"; three of the four
# couple_surface_and_recharge call sites sit inside the step loops under `if (params.fsm_on)`). So
# there is no FSM handoff frequency to vary.
#
# The FOURTH call site, `if (!params.fsm_on)`, fires once per REPORT -- and in that branch FSM is
# skipped entirely. That is the path this arm exercises: with no surface-water model, the
# recharge/runoff bookkeeping happens per report rather than per step, and report_interval sets how
# often. The routed input channel (col 20) must depend on ELAPSED TIME, not on how many times that
# bookkeeping call fired. ledgerA is the fixture that makes this exact: precipitation is strictly
# positive and the table is deep, so recharge does not depend on the state and the split is exactly
# runoff_ratio. On a fixture where recharge CAN go negative the split only fires where rech > 0, so
# col 19 carries negatives col 20 never sees and no clean expectation exists -- that mistake cost an
# hour, so the fixture choice here is load-bearing.
#
# It bites: the routed share used to be booked (N-1)/N -- 3/4, 1/2 and 0/1 of its true value at 4, 2
# and 1 cycles -- because the FIRST interval's split was deposited in arp.runoff by irf.cpp's
# initialisation and zeroed by couple_surface_and_recharge before the handoff read it.
for RI in 1 2 4; do
    run "cad0_$RI" ledgerA "4yr" 31536000 "off" "$RI" 0   -wtm_anderson || fail=1
    run "cad3_$RI" ledgerA "4yr" 31536000 "off" "$RI" 0.3 -wtm_anderson || fail=1
done
[[ $fail -eq 0 ]] || { echo "LOCAL LEDGER: FAILED (a run did not complete)"; exit 1; }

WORK="$WORK" INP="$INP" "$PY" - <<'PY'
import glob, os, sys
import numpy as np
import rasterio

W, INP = os.environ["WORK"], os.environ["INP"]
SPY = 31536000.0
fail = 0

def rd(p):
    with rasterio.open(p) as s:
        return s.read(1).astype(np.float64)

def final_wtd(stem):
    fs = sorted(glob.glob(f"{W}/{stem}_[0-9]" + "[0-9]" * 8 + "_*yr.tif"))
    return rd(fs[-1]) if fs else None

# ---------------------------------------------------------------- A. PLACEMENT
print("-- A. PLACEMENT: with ~no lateral flow, each column must rise by its OWN forcing --")
precip = rd(f"{INP}/ledgerA_t0_precipitation.tif")
poro   = rd(f"{INP}/ledgerA_porosity.tif")
w0     = rd(f"{INP}/ledgerA_t0_starting_wt.tif")
w1     = final_wtd("place")
if w1 is None:
    print("  FAIL  PLACEMENT -- no output raster"); fail = 1
else:
    YEARS = 2.0
    # Independent expectation. Below the surface storedVolume(wtd) = wtd*porosity to ~1e-8, so the rise
    # is (precip * years) / porosity. Nothing here reads a model internal.
    expect = precip * YEARS / poro
    got    = w1 - w0
    rel    = np.abs(got - expect) / np.maximum(np.abs(expect), 1e-30)

    # PRECONDITIONS. Without these the comparison is vacuous or invalid.
    ok = w1.max() < -1.0
    fail |= not ok
    print(f"  {'PASS' if ok else 'FAIL'}  PRECONDITION  table stays below the surface "
          f"(max wtd {w1.max():.3f} m) -- the analytic form needs it")
    spread = expect.max() / expect.min()
    ok = spread > 5.0
    fail |= not ok
    print(f"  {'PASS' if ok else 'FAIL'}  PRECONDITION  forcing varies {spread:.1f}x across the grid "
          f"(a uniform field could not detect misplacement)")

    ok = rel.max() < 1e-6
    fail |= not ok
    j, i = np.unravel_index(np.argmax(rel), rel.shape)
    print(f"  {'PASS' if ok else 'FAIL'}  PER-CELL   max rel error {rel.max():.3e} at (row {j}, col {i}) "
          f"expected {expect[j, i]:.6f} m, got {got[j, i]:.6f} m  (tol 1e-6)")

    # Axis-swap probe: state it explicitly rather than trusting the per-cell check to imply it.
    swapped = np.abs(got - expect.T) / np.maximum(np.abs(expect.T), 1e-30)
    ok = swapped.max() > 1e-3
    fail |= not ok
    print(f"  {'PASS' if ok else 'FAIL'}  AXIS-SWAP  the transposed forcing does NOT fit "
          f"(max rel {swapped.max():.3e}) -- so this arm can tell the axes apart")

# ---------------------------------------------------------------- B. REDISTRIBUTION
print("\n-- B. REDISTRIBUTION: closed domain, zero forcing; volume must hold while the table moves --")
b0 = rd(f"{INP}/ledgerB_t0_starting_wt.tif")
b1 = final_wtd("redist")
rows = [[float(x) for x in l.split()] for l in open(f"{W}/redist.txt")
        if l.split() and l.split()[0].isdigit() and len(l.split()) >= 23]
if b1 is None or len(rows) < 3:
    print("  FAIL  REDISTRIBUTION -- missing output"); fail = 1
else:
    moved = np.abs(b1 - b0)
    ok = moved.max() > 0.5
    fail |= not ok
    print(f"  {'PASS' if ok else 'FAIL'}  PRECONDITION  the mound actually spreads "
          f"(max |dwtd| {moved.max():.3f} m, rms {np.sqrt((moved**2).mean()):.3f} m)")

    # Nothing enters or leaves, so every input/loss channel must be identically zero. If any is not,
    # the volume check below would be comparing against a moving target.
    r = rows[-1]
    channels = {"9 recharge": r[8], "10 loss_to_ocean": r[9], "12 surface_removed": r[11],
                "13 ocean_outflow": r[12], "18 evap": r[17], "20 routed": r[19]}
    nz = {k: v for k, v in channels.items() if abs(v) > 1e-6}
    ok = not nz
    fail |= not ok
    print(f"  {'PASS' if ok else 'FAIL'}  CLOSED     every input/loss channel is zero"
          + ("" if ok else f" -- NONZERO: {nz}"))

    # The conservation statement. stored_volume is the model's exact sum V(wtd)*area; with the domain
    # closed it must not drift, and a face flux that is not antisymmetric between its two cells would
    # show up here immediately.
    vol = [x[13] for x in rows]
    drift = max(abs(v - vol[0]) for v in vol) / (abs(vol[0]) or 1.0)
    ok = drift < 1e-12
    fail |= not ok
    print(f"  {'PASS' if ok else 'FAIL'}  CONSERVED  stored_volume drift over {len(vol)} cycles "
          f"{drift:.3e} (tol 1e-12); volume {vol[0]:.9e} m^3")

# ---------------------------------------------------------------- C. CADENCE
print("\n-- C. CADENCE: the routed input channel must track elapsed time, not coupling frequency --")
def lastrow(stem):
    rows = [[float(x) for x in l.split()] for l in open(f"{W}/{stem}.txt")
            if l.split() and l.split()[0].isdigit() and len(l.split()) >= 23]
    return rows[-1] if rows else None
for RI in (1, 2, 4):
    a, b = lastrow(f"cad0_{RI}"), lastrow(f"cad3_{RI}")
    if a is None or b is None:
        print(f"  FAIL  CADENCE    report_interval {RI}: missing output"); fail = 1; continue
    split = b[19] / b[18] if b[18] else float("nan")   # col20 / col19
    same_in = b[8] / a[8] if a[8] else float("nan")    # col9(rr=0.3) / col9(rr=0)
    ok = abs(split - 3.0 / 7.0) < 1e-6 and abs(same_in - 1.0) < 1e-6
    fail |= not ok
    print(f"  {'PASS' if ok else 'FAIL'}  CADENCE    report_interval {RI}: col20/col19 = {split:.6f} "
          f"(want 3/7 = 0.428571), total input vs runoff_ratio=0 = {same_in:.6f} (want 1.000000)")

print("\nLOCAL LEDGER: " + ("ALL PASSED" if not fail else "FAILED"))
sys.exit(1 if fail else 0)
PY
