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
import os
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
    # Delegates to the shared guard: many matches are expected (one per cycle) and the last is the
    # one wanted, but ONLY if every match belongs to this stem. `fsm_runoff_` would also match
    # `fsm_runoff_hi_...`, and since 'h' sorts after a digit the last match would be the WRONG
    # fixture. Safe today only because PREFIX carries an `_n<ranks>_` infix that separates them --
    # a property of the naming, not of this function, so assert it rather than rely on it.
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import wtm_volume as VOL
    return VOL.latest_output(prefix)


def read_field(prefix):
    with rasterio.open(last_tif(prefix)) as s:
        a = s.read(1).astype(np.float64)
        nod = s.nodata
    return np.where(a == nod, np.nan, a) if nod is not None else a


def provenance(cfg, reason, commit):
    """The header lines that say WHERE THIS REFERENCE CAME FROM (#85).

    A reference IS the assertion -- every golden comparison is "the model still agrees with this
    file" -- so a reference that cannot say what produced it is a number without provenance sitting at
    the centre of a test. This suite has already been burned by exactly that gap: the `transient`
    reference was once captured from a run that NEVER FINISHED (bit-exactly the masked initial water
    table), so the case asserted output == input and passed vacuously for its whole life, until
    309dbf0 required exit 0 AND an output at the configured total_time.

    THE CONFIG HASH IS THE PART A MACHINE CAN CHECK. Commit, date and reason are prose a reader
    evaluates; the sha256 of the config that produced the reference can be compared against the config
    a later check is running, which turns "was this regolded under different settings" from a question
    into a measurement. check() reports it on FAILURE, where it is needed.
    """
    import hashlib, datetime
    h = "unknown"
    if cfg and os.path.exists(cfg):
        h = hashlib.sha256(open(cfg, "rb").read()).hexdigest()[:16]
    out = ["# provenance (#85) -- what produced this reference:",
           f"#   generated : {datetime.date.today().isoformat()}",
           f"#   commit    : {commit or 'unknown'}",
           f"#   config    : {cfg or 'unknown'}  sha256:{h}",
           f"#   reason    : {reason}"]
    return "\n".join(out) + "\n"


def read_provenance(ref):
    """Return the reference's provenance comment block, or a line saying it has none."""
    out = []
    with open(ref) as f:
        for line in f:
            if not line.startswith("#"):
                break
            out.append(line.rstrip("\n"))
    body = [l for l in out if "provenance" in l or l.startswith("#   ")]
    return body or ["# (this reference records no provenance)"]


def generate(prefix, ref, cfg=None, reason=None, commit=None):
    # A REGOLD WITHOUT A STATED REASON IS REFUSED. #55 regolded five references each with a reason, but
    # the reasons live only in commit messages -- so the files themselves could not say why they
    # changed. Requiring it here puts the reason where the number is, and makes a silent regold
    # impossible rather than merely discouraged.
    if not reason:
        sys.stderr.write("golden.py generate: a REASON is required -- say why this reference is being\n"
                         "  (re)generated. It is written into the file, because a reference IS an\n"
                         "  assertion and an assertion that cannot say where it came from is a number\n"
                         "  without provenance. Pass it as argv[4] or set GOLDEN_REASON.\n")
        sys.exit(2)
    a = read_field(prefix)
    with open(ref, "w") as f:
        f.write(f"# WTM golden reference; shape={a.shape[0]}x{a.shape[1]} (rows x cols); values row-major, %.17g\n")
        f.write(provenance(cfg, reason, commit))
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
        # A FAILING GOLDEN IS THE MOMENT THE PROVENANCE IS WORTH HAVING. The question a reader has is
        # "is this reference still the right thing to compare against", and that is decided by what
        # produced it -- which commit, under which config, and why it was last written. Printing it
        # here saves the git archaeology that was the ONLY route before #85.
        for line in read_provenance(ref):
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


def main():
    if len(sys.argv) < 4:
        sys.stderr.write(__doc__)
        return 2
    mode, prefix, ref = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode == "generate":
        #  argv: generate <prefix> <ref> [config] [reason] [commit]
        #  reason also accepted via GOLDEN_REASON so run.sh can require it once for a whole regold.
        cfg    = sys.argv[4] if len(sys.argv) > 4 else None
        reason = sys.argv[5] if len(sys.argv) > 5 else os.environ.get("GOLDEN_REASON")
        commit = sys.argv[6] if len(sys.argv) > 6 else os.environ.get("GOLDEN_COMMIT")
        generate(prefix, ref, cfg, reason, commit)
        return 0
    if mode == "check":
        tol = float(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_TOL
        return check(prefix, ref, tol)
    sys.stderr.write(f"unknown mode {mode}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
