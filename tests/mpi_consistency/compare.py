#!/usr/bin/env python3
"""Compare two WTM runs (given output prefixes) for MPI consistency.

Usage: compare.py <prefix_a> <prefix_b>

Checks that the final water-table TIF and the cumulative water-budget
diagnostics agree between the two runs (e.g. an n=1 run vs an n=N run). Exits 0
if consistent, 1 otherwise. Raster values are compared exactly (the model is
deterministic; MPI decomposition must not change the result).
"""
import sys
import glob
import numpy as np

try:
    import rasterio
except ImportError:
    sys.stderr.write("rasterio required for compare.py\n")
    sys.exit(2)

# Cross-rank-count comparison is NOT bitwise reproducible: PETSc/Anderson global
# reductions (dot products, norms, MPI_Allreduce sums) accumulate in an order
# that depends on the domain decomposition, so the converged water table differs
# at the floating-point-reduction level. This was verified to be pre-existing:
# the same ~1e-11 m n=1-vs-n=4 difference appears in the baseline before any of
# the ArrayPack-distribution work. The tolerance below sits ~5 orders of
# magnitude above that noise floor and far below any physically or numerically
# meaningful error (a real MPI bug -- e.g. a ghost-cell fault -- perturbs the
# field by >=0.1 m at boundaries, and the accounting bugs seen during this work
# were O(1)-O(1e12)). Same-rank-count refactor regressions are checked
# separately and must be bit-identical.
WTD_TOL = 1e-6         # metres; above FP reduction noise, below any real error

# EVERY budget column is compared, not just recharge and ocean loss. Three tolerance CLASSES, each
# measured rather than assumed, because they fail for genuinely different reasons and collapsing them
# would either hide a real defect or manufacture a spurious one.
#
#   EXACT   the INPUT channels. Driven by the forcing and elapsed time, not by the state, so they come
#           out bit-identical across decompositions (measured 0.000e+00). This is the sharp check --
#           the runoff channel in particular is accumulated on rank 0 alone on the serial recharge
#           path, and is correct only because every other rank contributes exactly zero. Never loosen
#           this class; it is the only exact check here.
#   STATE   sums over the converged water table. That field is NOT bit-identical across
#           decompositions (PETSc reductions accumulate in decomposition-dependent order), so sums
#           over it inherit ~1e-10 relative noise. Measured worst 1.974e-10.
#   DISCRETE  the two GROSS-FLUX counters, total_surface_removed and total_loss_to_ocean. Both are
#           driven by DISCRETE decisions -- which cells the semismooth active set pins, and which
#           depression FillSpillMerge fills or spills -- so a sub-nanometre difference in the field can
#           flip a decision and move the total by far more than the field moved. (On this fixture the
#           two columns are numerically IDENTICAL -- measured 3.09375623024 for both -- because the
#           exfiltrated water is routed straight to the ocean on a small ocean-ringed grid.) Measured:
#           1.112e-6 at n=2 and 2.746e-6 at n=4 with runoff_ratio 0, rising to 2.549e-3 and 3.549e-3
#           with runoff_ratio 0.3, which sends far more water through the routing path. 1e-2 carries
#           ~3x margin over the worst.
#
#           This was INVISIBLE until the run log went from 6 to 12 significant digits: 24.5833330834
#           and 24.5833057561 both print as "24.5833". Pre-existing and not introduced -- the digits
#           prove it. The converged water table itself stays within WTD_TOL, so the ANSWER is
#           consistent across decompositions; it is these gross-flux diagnostics that are sensitive.
DIAG_RTOL_EXACT    = 1e-12
DIAG_RTOL_STATE    = 1e-9
DIAG_RTOL_DISCRETE = 1e-2

# (0-indexed column, name, tolerance). See benchmark/WATER_BUDGET.md for the column list.
DIAG_COLS = [
    (8,  "total_recharge_added",  DIAG_RTOL_EXACT),
    (18, "recharge_direct",       DIAG_RTOL_EXACT),
    (19, "runoff_to_surface",     DIAG_RTOL_EXACT),
    (12, "total_ocean_outflow",   DIAG_RTOL_STATE),
    (13, "stored_volume",         DIAG_RTOL_STATE),
    (17, "total_evap_removed",    DIAG_RTOL_STATE),
    (11, "total_surface_removed", DIAG_RTOL_DISCRETE),
    (9,  "total_loss_to_ocean",   DIAG_RTOL_DISCRETE),
]


def last_tif(prefix):
    tifs = sorted(glob.glob(prefix + "*.tif"))
    if not tifs:
        raise FileNotFoundError(f"no output TIF for prefix {prefix}")
    return tifs[-1]


def read(path):
    with rasterio.open(path) as s:
        a = s.read(1).astype(np.float64)
        nod = s.nodata
    return np.where(a == nod, np.nan, a) if nod is not None else a


def last_diag_line(txt):
    """Return the last data row of the run log as a list of floats, or None."""
    row = None
    try:
        with open(txt) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 23 and parts[0].isdigit():
                    row = [float(x) for x in parts]
    except FileNotFoundError:
        pass
    return row


def approx(a, b, rtol):
    if a is None or b is None:
        return a == b
    if a == 0.0 and b == 0.0:
        return True
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom <= rtol


def main():
    pa, pb = sys.argv[1], sys.argv[2]
    ok = True

    # 1) water table field, exact
    a, b = read(last_tif(pa)), read(last_tif(pb))
    if a.shape != b.shape:
        print(f"  shape mismatch {a.shape} vs {b.shape}", file=sys.stderr)
        return 1
    d = np.abs(a - b)
    d = d[~np.isnan(d)]
    maxd = float(d.max()) if d.size else 0.0
    if maxd > WTD_TOL:
        print(f"  wtd differs: max|delta|={maxd:.3e}", file=sys.stderr)
        ok = False

    # 2) diagnostics, relative
    ra = last_diag_line(pa + ".txt" if not pa.endswith(".txt") else pa)
    rb = last_diag_line(pb + ".txt" if not pb.endswith(".txt") else pb)
    # The text file path is "<prefix without trailing _>.txt" as written by run.sh.
    # run.sh names it "<tag>.txt"; prefixes here are "<work>/<tag>_" so strip the trailing underscore.
    if ra is None:
        ra = last_diag_line(pa.rstrip("_") + ".txt")
    if rb is None:
        rb = last_diag_line(pb.rstrip("_") + ".txt")
    if ra is None or rb is None:
        print("  missing run-log data row (need >= 23 columns)", file=sys.stderr)
        return 1
    for idx, name, rtol in DIAG_COLS:
        if not approx(ra[idx], rb[idx], rtol):
            rel = abs(ra[idx] - rb[idx]) / max(abs(ra[idx]), abs(rb[idx]), 1e-300)
            print(f"  {name} differs: {ra[idx]!r} vs {rb[idx]!r}  (rel {rel:.3e} > {rtol:.0e})",
                  file=sys.stderr)
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
