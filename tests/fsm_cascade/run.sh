#!/usr/bin/env bash
# CASCADE fixture: surface water flows depression -> depression -> off map, with the lake-aware skim.
#
# pit A (floor 94, outlet sill 97) spills DOWN into basin B (floor 88, outlet sill 95), which spills OFF-MAP
# to the ocean. At steady state both fill to their sills, so this checks that the skim + FSM route a spill
# CHAIN correctly (A->B->ocean) and reach the known sill elevations, MPI-consistently and conserving mass.
# (The separate "one full, one not" heterogeneous state uses its own fixture; see #123.)
#
# Asserts, Anderson path + FSM every step + active-set skim:
#   CHAIN LEVELS : pit A fills to 97.0 m (its A->B sill) and basin B fills to 95.0 m (its off-map sill).
#   CONSERVATION : per-cycle water balance closes (|Δbudget_residual|/Δrecharge < 1e-4).
#   MPI CONSISTENT: n=1 == n=4 (byte-identical wtd).
#
# Usage:  tests/fsm_cascade/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/fsm_cascade_t0_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
WORK=$(mktemp -d /tmp/casc_XXXX); trap 'rm -rf "$WORK"' EXIT
PY="${PY:-python3}"; export OMP_NUM_THREADS=1

emit() { ../emit_config.sh > "$WORK/$1.yaml" <<EOF
run_type equilibrium
total_time 120yr
supplied_wt 1
deltat 31536000
report_interval 5
save_nreport_interval 9999
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
infiltration_on 0
fsm_on 1
runoff_collector implicit
surfdatadir $INP
region fsm_cascade
time_start t0
time_end t0
eq_tol 0
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}
emit skim
"$WTM" "$WORK/skim.yaml" -wtm_anderson -wtm_active_set > "$WORK/skim.err" 2>&1 \
  || { echo "RUN FAILED: skim"; tail -3 "$WORK/skim.err"; exit 2; }
emit skim4
mpirun -n 4 "$WTM" "$WORK/skim4.yaml" -wtm_anderson -wtm_active_set \
    -da_processors_x 2 -da_processors_y 2 > "$WORK/skim4.err" 2>&1 \
  || { echo "RUN FAILED: skim4"; tail -3 "$WORK/skim4.err"; exit 2; }

SK=$(ls "$WORK"/skim_*.tif | tail -1); SK4=$(ls "$WORK"/skim4_*.tif | tail -1)
"$PY" - "$INP/fsm_cascade_t0_topography.tif" "$SK" "$SK4" "$WORK/skim.txt" <<'PY'
import sys, numpy as np, rasterio
topo, wk, wk4 = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:4]]
txt = sys.argv[4]
def surf(region):
    w = region_of(wk, region); p = w > 1e-6
    return float((region_of(topo, region) + w)[p].max()) if p.any() else -999.0
def region_of(a, r):
    return a[r]
A = (slice(4, 12), slice(11, 20))    # pit A
B = (slice(15, 27), slice(4, 26))    # basin B
sA, sB = surf(A), surf(B)
mpi = float(np.abs(wk - wk4).max())
rows = [l.split() for l in open(txt) if l and l[0].isdigit()]
R = np.array([float(r[8]) for r in rows]); res = np.array([float(r[15]) for r in rows])
rel = (np.abs(np.diff(res))[-4:] / np.where(np.abs(np.diff(R))[-4:] > 0, np.abs(np.diff(R))[-4:], 1)).max()
ok = True
def check(name, cond, detail):
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'} {name}: {detail}"); ok = ok and cond
check("CHAIN LEVELS (A->97 sill, B->95 sill)", abs(sA - 97.0) < 0.2 and abs(sB - 95.0) < 0.2,
      f"pit A surface = {sA:.3f} m (sill 97), basin B surface = {sB:.3f} m (sill 95)")
check("CONSERVATION (per-cycle balance closes)", rel < 1e-4,
      f"max |Δbudget_residual|/Δrecharge = {rel:.3e}")
check("MPI CONSISTENT (n=1 == n=4)", mpi < 1e-9, f"max|Δwtd| = {mpi:.3e} m")
print("PASS: cascade A->B->ocean routes to the correct sills, conserving and MPI-consistent"
      if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
