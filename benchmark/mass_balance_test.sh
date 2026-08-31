#!/usr/bin/env bash
# Mass-balance MPI-consistency test.
#
# The cumulative water-budget diagnostics (total_added_recharge and
# total_loss_to_ocean, columns 9 and 10 of the text output) must be identical
# whether the model runs on 1 or N MPI ranks. This exercises:
#   - set_starting_values + solve copy-back: owned-cell partials, scalar-reduced
#   - FillSpillMerge: accumulated on the full replicated grid (already global)
# A mismatch means the owned-only accounting or the reduce is wrong.
#
# Usage:
#   cd benchmark
#   ./mass_balance_test.sh [path/to/wtm.x] [N_ranks]
#
# Default binary: ../build/wtm.x   Default N: 8
set -euo pipefail
cd "$(dirname "$0")"

WTM=${1:-../build/wtm.x}
NRANKS=${2:-8}

if [[ ! -x "$WTM" ]]; then
    echo "ERROR: WTM binary not found at $WTM" >&2
    exit 1
fi

# Small synthetic dome fixture (run_type test synthesizes precip/evap/mask internally). The old fixture was a
# 1000x1000 global DEM -- ~1000x more cells than this MPI-accounting invariant needs. .tif inputs are gitignored.
[[ -f mb_inputs/mb_topography.tif ]] || python3 make_mass_balance_inputs.py >/dev/null
INP=$(readlink -f mb_inputs)

# FSM on so that both accumulation paths (GW ocean loss and FSM ocean loss) are exercised.
run() { # $1 = nranks -> echoes "recharge loss" from the last data line
    local n="$1" tag="mbtest_n${1}"
    local cfg tf
    cfg=$(mktemp /tmp/${tag}_XXXX.yaml)
    tf="/tmp/${tag}.txt"
    rm -f "$tf" "/tmp/${tag}_"*.tif
    # run_type test: only topography + slope are read; geometry comes from the geotransform (#124).
    ../tests/emit_config.sh > "$cfg" <<EOF
run_type test
fsm_on 1
deltat 31536000
total_time 3yr
report_interval 1
save_nreport_interval 9999
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
surfdatadir $INP
region mb
time_start t0
time_end t0
supplied_wt 0
eq_tol 0
textfilename $tf
outfile_prefix /tmp/${tag}_
EOF
    # -wtm_eq_tol 0: run the full fixed cycle count (do not let the equilibrium auto-stop default fire).
    OMP_NUM_THREADS=1 mpirun -n "$n" "$WTM" "$cfg" -snes_stol 1e-6 >/dev/null 2>&1
    rm -f "$cfg"
    awk 'NF>=11 && $1 ~ /^[0-9]+$/ {r=$9; o=$10} END{print r, o}' "$tf"
}

echo "=== Mass-balance MPI-consistency test (small dome fixture, run_type test, fsm_on=1) ==="
echo "WTM binary: $WTM   comparing n=1 vs n=$NRANKS"
echo

read -r R1 O1 < <(run 1)
read -r RN ON < <(run "$NRANKS")

printf "  n=1        recharge_added=%.6e  loss_to_ocean=%.6e\n" "$R1" "$O1"
printf "  n=%-2s       recharge_added=%.6e  loss_to_ocean=%.6e\n" "$NRANKS" "$RN" "$ON"
echo

# Relative comparison (tolerance 1e-9); guard against divide-by-zero.
ok=$(awk -v r1="$R1" -v o1="$O1" -v rn="$RN" -v on="$ON" 'BEGIN{
    tol=1e-9;
    dr = (r1==0)? (rn==0?0:1) : (rn-r1)/r1; if(dr<0)dr=-dr;
    do_ = (o1==0)? (on==0?0:1) : (on-o1)/o1; if(do_<0)do_=-do_;
    print (dr<tol && do_<tol) ? "PASS" : "FAIL";
}')

if [[ "$ok" == "PASS" ]]; then
    echo "PASS: water-budget diagnostics agree between n=1 and n=$NRANKS."
    exit 0
else
    echo "FAIL: diagnostics differ between n=1 and n=$NRANKS (mass-balance accounting is MPI-inconsistent)." >&2
    exit 1
fi
