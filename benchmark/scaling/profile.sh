#!/usr/bin/env bash
# profile.sh -- PETSc -log_view profile of the WTM groundwater solve.
#
# Tier-1 of the speedup work: profile BEFORE optimizing. Attributes (a) the
# per-core cost -- the ~2x "after" vs kcallaghan regression -- and (b) the
# parallel halo/reduction overhead, so we optimize measured hot spots rather
# than hypotheses.
#
# Usage:
#   source ~/models/WTM/msi_env.sh test        # toolchain + rasterio (for the generator)
#   cd ~/models/WTM/benchmark/scaling
#   ./profile.sh [GRID] [PAR_RANKS]            # defaults: 4000 16
#
# Also configurable via env: GRID, PAR_RANKS, MAXITER, MPIEXEC, SNES.
#   GRID=8000 ./profile.sh                      # 8000^2 shows the full ~2.35x regression (~23 min at n=1)
#
# Expects the three sibling builds (as on MSI):
#   ~/models/WTM  (after)   ~/models/WTM-before  (before)   ~/models/WTM-kcallaghan
# Missing builds are skipped with a note.
#
# Writes profiles/prof_<label>_n<N>.txt (PETSc event tables; gitignored). Read the
# "Event" table:
#   * SNESFunctionEval  = FormFunctionLocal (residual + T/S). Compare Time/Count
#                         after@n1 vs kcallaghan@n1 -> localizes the per-core cost.
#   * VecScatterBegin/End + MPI reductions (in the n=PAR run) -> communication/Amdahl.
set -u
cd "$(dirname "$0")"
HERE=$(pwd)
MODELS_ROOT=$(readlink -f ../../..)

GRID=${1:-${GRID:-4000}}
PAR_RANKS=${2:-${PAR_RANKS:-16}}
MAXITER=${MAXITER:-5}
MPIEXEC=${MPIEXEC:-mpiexec}
SNES=${SNES:-"-snes_type anderson -snes_stol 1e-6"}   # no -snes_mf (deadlocks under MPICH)

# Pin one OpenMP thread/rank so the profile is not muddied by thread oversubscription.
export OMP_NUM_THREADS=1

OUT=$HERE/profiles
mkdir -p "$OUT"

# 1) synthetic grid (run_type test needs only topography + slope)
SDIR=$HERE/synth/$GRID
if [ ! -f "$SDIR/synth_topography.tif" ]; then
    # Pick a Python that actually has rasterio. On MSI the module `python3` on
    # PATH often shadows the active conda env (e.g. wtmtest), so prefer the env's
    # own interpreter ($CONDA_PREFIX/bin/python) when present. Override: PYTHON=...
    PYTHON=${PYTHON:-}
    if [ -z "$PYTHON" ]; then
        if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
            PYTHON="$CONDA_PREFIX/bin/python"
        else
            PYTHON=python3
        fi
    fi
    if ! "$PYTHON" -c "import rasterio" 2>/dev/null; then
        echo "ERROR: '$PYTHON' cannot import rasterio (needed only to generate the grid)." >&2
        echo "  Fix: activate the env with rasterio (e.g. conda activate wtmtest), or run" >&2
        echo "       PYTHON=\$CONDA_PREFIX/bin/python ./profile.sh" >&2
        exit 1
    fi
    echo "generating ${GRID}^2 grid with $PYTHON"
    "$PYTHON" make_synthetic.py "$GRID" --outdir "$SDIR" --region synth
fi

# 2) matching config (cells_per_degree = grid/120 keeps the domain on the globe)
CPD=$(awk "BEGIN{printf \"%.6f\", $GRID/120.0}")
CFG=$OUT/prof_${GRID}.cfg
cat > "$CFG" <<EOF
run_type           test
fsm_on             0
evap_mode          0
infiltration_on    0
runoff_ratio_on    0
cells_per_degree   $CPD
southern_edge      -45
deltat             31536000
total_cycles       1
report_interval            $MAXITER
fdepth_a           200
fdepth_b           150
fdepth_fmin        2
time_start         t0
time_end           t0
surfdatadir        $SDIR
region             synth
supplied_wt        0
textfilename       $OUT/prof_run.txt
outfile_prefix     $OUT/prof_out_
save_nreport_interval     9999
EOF

run() {  # label  binary  nranks
    local label=$1 bin=$2 n=$3
    local log=$OUT/prof_${label}_n${n}.txt
    if [ ! -x "$bin" ]; then
        echo "  skip $label (n=$n): binary not found at $bin" >&2
        return
    fi
    echo "--- profiling $label at n=$n  (${GRID}^2, report_interval=$MAXITER) -> $log"
    # shellcheck disable=SC2086  # $SNES is intentionally word-split into flags
    $MPIEXEC -n "$n" "$bin" "$CFG" $SNES -log_view :"$log"
}

echo "=== WTM solve profile: grid ${GRID}^2, parallel ranks ${PAR_RANKS} ==="
run after      "$MODELS_ROOT/WTM/build/wtm.x"            1           # per-core cost (the regression target)
run before     "$MODELS_ROOT/WTM-before/build/wtm.x"     1           # before vs after @ n=1 -> the flip's per-core cost
run kcallaghan "$MODELS_ROOT/WTM-kcallaghan/build/wtm.x" 1           # kcallaghan vs after @ n=1 -> the full ~2x
run after      "$MODELS_ROOT/WTM/build/wtm.x"            "$PAR_RANKS" # parallel overhead (VecScatter + reductions)

echo
echo "Profiles written to $OUT/  (gitignored)."
echo "Key comparison: SNESFunctionEval time-per-call, prof_after_n1.txt vs prof_kcallaghan_n1.txt."
