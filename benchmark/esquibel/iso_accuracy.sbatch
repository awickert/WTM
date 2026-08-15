#!/bin/bash -l
#SBATCH --job-name=iso-accuracy
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=48gb
#SBATCH --time=4:00:00
#SBATCH --output=iso_accuracy-%j.out
#
# ISO-ACCURACY equilibrium comparison (Esquibel 384k, cold->eq): how many SNES iterations does each of
# cc (KCallaghan fixed 1-wk) and the one-knob adaptive need to reach a given accuracy vs a common
# tightly-converged "truth"? The eq-stop is only a proxy; this is the honest accuracy-vs-cost curve.
#   truth      = adaptive, eq_tol 0.001 (very converged) -- the shared steady state (order-independent)
#   cc_traj    = cc fixed 1-wk,  eq_tol 0.01, save EVERY cycle
#   adapt_traj = one-knob adaptive, eq_tol 0.01, save EVERY cycle
# Post: for each saved cycle, RMS(field - truth) and CUMULATIVE SNES iters -> CSV -> iters@accuracy.
set -uo pipefail
N="${N:-16}"
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"; BIN="$WTM_ROOT/build/wtm.x"; ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1 PROJ_DATA="${PROJ_DATA:-/usr/share/proj}" PROJ_LIB="${PROJ_LIB:-/usr/share/proj}"
cd "$ESQ"
grep -qa "derived from eq_tol" "$BIN" || { echo "FATAL: stale binary."; exit 1; }
R=results/iso_accuracy; mkdir -p "$R"
BASE="-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6 -wtm_surface_exfiltration_to_runoff -wtm_eq_metric frac"

run() { # label eq_tol save_every extra_flags...
  local lbl="$1" eqt="$2" save="$3"; shift 3
  local stem="$R/$lbl"
  sed "s#^deltat.*#deltat 604800#;s#^total_cycles.*#total_cycles 200#;s#^cycles_to_save.*#cycles_to_save $save#;s#^supplied_wt.*#supplied_wt 0#;s#^textfilename.*#textfilename ${stem}.txt#;s#^outfile_prefix.*#outfile_prefix ${stem}_#" eq_awickert.cfg > "${stem}.cfg"
  rm -f "${stem}.txt" "${stem}"_*.tif
  echo "=== $lbl (eq_tol=$eqt, save_every=$save): $* ==="
  mpiexec -ppn "$N" -n "$N" "$BIN" "${stem}.cfg" $BASE -wtm_eq_tol $eqt "$@" > "${stem}.log" 2>&1
  echo "  rc=$? cycles=$(grep -c 'per-cycle |' ${stem}.log)"
}

# 1) truth (tight), final only.   2) cc trajectory.   3) adaptive trajectory.
run truth      0.001 200 -wtm_tr_bdf2 -wtm_dt_adaptive
run cc_traj    0.01    1
run adapt_traj 0.01    1 -wtm_tr_bdf2 -wtm_dt_adaptive
TRUTH=$(ls -1 "$R"/truth_*.tif | tail -1)
echo "truth field: $TRUTH"

# --- accuracy-vs-cost: per cycle, RMS(field-truth) and cumulative SNES iters ---
python3 - "$R" "$TRUTH" <<'PY'
import sys, glob, os, re
import numpy as np, rasterio
R, truthp = sys.argv[1], sys.argv[2]
with rasterio.open(truthp) as ds:
    truth = ds.read(1).astype(np.float64); nod = ds.nodata
def rms_vs_truth(p):
    with rasterio.open(p) as ds:
        a = ds.read(1).astype(np.float64)
    m = np.isfinite(a) & np.isfinite(truth)
    if nod is not None: m &= (a != nod) & (truth != nod)
    d = a[m] - truth[m]
    return float(np.sqrt(np.mean(d*d))), float(np.max(np.abs(d)))
# cumulative iters per cycle from the log: sum "Number of nonlinear iterations = N" between "cycle K:" lines
def cum_iters(logp):
    cum, out = 0, {}
    with open(logp) as f:
        for line in f:
            m = re.search(r'Number of nonlinear iterations = (\d+)', line)
            if m: cum += int(m.group(1)); continue
            c = re.match(r'cycle (\d+):', line)
            if c: out[int(c.group(1))] = cum
    return out
for meth in ("cc_traj", "adapt_traj"):
    ci = cum_iters(f"{R}/{meth}.log")
    rows = []
    for tif in sorted(glob.glob(f"{R}/{meth}_0*.tif")):
        cyc = int(re.search(r'_(\d+)\.tif$', tif).group(1))
        if cyc == 0 or cyc not in ci: continue
        rms, mx = rms_vs_truth(tif)
        rows.append((cyc, ci[cyc], rms, mx))
    with open(f"{R}/{meth}_accuracy.csv", "w") as o:
        o.write("cycle,cum_iters,rms_vs_truth,maxabs_vs_truth\n")
        for r in rows: o.write(f"{r[0]},{r[1]},{r[2]:.6g},{r[3]:.6g}\n")
    print(f"\n### {meth}: cycle  cum_iters  rms_vs_truth  maxabs")
    for r in rows: print(f"  {r[0]:3d}  {r[1]:8d}  {r[2]:10.5f}  {r[3]:8.3f}")
# iters@accuracy: first cycle each method reaches RMS <= target
print("\n### ITERS TO REACH RMS-vs-truth TARGET ###")
def iters_at(meth, target):
    for line in open(f"{R}/{meth}_accuracy.csv").readlines()[1:]:
        cyc, it, rms, mx = line.split(",");
        if float(rms) <= target: return int(it)
    return None
for target in (0.20, 0.10, 0.05, 0.02):
    a = iters_at("cc_traj", target); b = iters_at("adapt_traj", target)
    sp = (a/b) if (a and b) else float('nan')
    print(f"  RMS<={target:4.2f} m:  cc={a}  adaptive={b}  speedup={sp:.2f}x")
PY
echo "### DONE ###"
