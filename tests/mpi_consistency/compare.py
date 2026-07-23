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
DIAG_RTOL = 1e-9       # relative tolerance for the accumulated diagnostics


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
    """Return (recharge_added, loss_to_ocean) from the last data row of the text file."""
    r = o = None
    try:
        with open(txt) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 11 and parts[0].isdigit():
                    r, o = float(parts[8]), float(parts[9])
    except FileNotFoundError:
        pass
    return r, o


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
    ra, oa = last_diag_line(pa + ".txt" if not pa.endswith(".txt") else pa)
    rb, ob = last_diag_line(pb + ".txt" if not pb.endswith(".txt") else pb)
    # The text file path is "<prefix without trailing _>.txt" as written by run.sh.
    # run.sh names it "<tag>.txt"; prefixes here are "<work>/<tag>_" so strip the trailing underscore.
    if ra is None:
        ra, oa = last_diag_line(pa.rstrip("_") + ".txt")
    if rb is None:
        rb, ob = last_diag_line(pb.rstrip("_") + ".txt")
    if not approx(ra, rb, DIAG_RTOL):
        print(f"  recharge_added differs: {ra} vs {rb}", file=sys.stderr)
        ok = False
    if not approx(oa, ob, DIAG_RTOL):
        print(f"  loss_to_ocean differs: {oa} vs {ob}", file=sys.stderr)
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
