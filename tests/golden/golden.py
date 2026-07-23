#!/usr/bin/env python3
"""Generate or check a golden (expected-results) reference for a WTM run.

A golden reference pins the water table produced by a run we believe correct
(captured at n=1, which is deterministic -- no cross-rank reduction noise).
Unlike the n=1-vs-n=N consistency tests, this catches regressions that perturb
*every* rank count equally, e.g. a change to the physics or the solve. It is a
CHANGE DETECTOR, not a proof of physical correctness: if the model's behavior
changes on purpose, regenerate the references (run.sh --generate) and review the
diff.

Usage:
  golden.py generate <tif_prefix> <ref.txt>
  golden.py check    <tif_prefix> <ref.txt> [tol]

The reference is a plain-text full-precision dump of the final-output raster
(git-diffable). Comparison treats nodata as NaN and NaN==NaN as equal.
"""
import sys
import glob
import numpy as np

try:
    import rasterio
except ImportError:
    sys.stderr.write("rasterio required for golden.py\n")
    sys.exit(2)

DEFAULT_TOL = 1e-6  # metres; above FP-reduction noise, below any real change


def last_tif(prefix):
    tifs = sorted(glob.glob(prefix + "*.tif"))
    if not tifs:
        raise FileNotFoundError(f"no output TIF for prefix {prefix}")
    return tifs[-1]


def read_field(prefix):
    with rasterio.open(last_tif(prefix)) as s:
        a = s.read(1).astype(np.float64)
        nod = s.nodata
    return np.where(a == nod, np.nan, a) if nod is not None else a


def generate(prefix, ref):
    a = read_field(prefix)
    with open(ref, "w") as f:
        f.write(f"# WTM golden reference; shape={a.shape[0]}x{a.shape[1]} (rows x cols); values row-major, %.17g\n")
        for row in a:
            f.write(" ".join("nan" if np.isnan(v) else repr(float(v)) for v in row) + "\n")
    print(f"  wrote {ref} ({a.shape[0]}x{a.shape[1]})")


def load_ref(ref):
    rows = []
    with open(ref) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            rows.append([np.nan if t == "nan" else float(t) for t in line.split()])
    return np.array(rows, dtype=np.float64)


def check(prefix, ref, tol):
    a = read_field(prefix)
    b = load_ref(ref)
    if a.shape != b.shape:
        print(f"  shape mismatch: run {a.shape} vs ref {b.shape}", file=sys.stderr)
        return 1
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.abs(np.where(both_nan, 0.0, a - b))
    # A NaN in exactly one of the two is a mismatch.
    one_nan = np.isnan(a) ^ np.isnan(b)
    d = np.where(one_nan, np.inf, d)
    maxd = float(np.nanmax(d)) if d.size else 0.0
    if maxd > tol:
        print(f"  golden mismatch: max|delta|={maxd:.3e} > tol={tol:.1e}", file=sys.stderr)
        return 1
    return 0


def main():
    if len(sys.argv) < 4:
        sys.stderr.write(__doc__)
        return 2
    mode, prefix, ref = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode == "generate":
        generate(prefix, ref)
        return 0
    if mode == "check":
        tol = float(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_TOL
        return check(prefix, ref, tol)
    sys.stderr.write(f"unknown mode {mode}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
