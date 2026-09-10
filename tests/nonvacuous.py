#!/usr/bin/env python3
"""A test's compared fields must HAVE STRUCTURE. A constant field asserts nothing.

WHY. This is the sixth vacuity mechanism, and the five general guards already in place cannot
see it. The run completes. Every key resolves to exactly what the config declared, so
config_identity.py passes. Every arm proves it ran what it configured, so expect_resolved
passes. make_work keeps its evidence of success. And the field being compared is identically
zero, so every difference is 0.00e+00 and the suite reports PASS.

Three suites were in that state simultaneously, for different reasons, and all three had been
green for months (#34):

  boundary_analytic     0 of 66  cells nonzero -- the surface-transition taper was off, so the
                                 model removed all surface water every step and the closed-form
                                 mound the suite fits could never exist. Its parabola fit fit a
                                 zero field and reported residual 0.000e+00.
  ghost_boundary        0 of 480 -- the fixture's placeholder geotransform made cells 111 km, so
                                 the coastal wedge saturated and active_set pinned it at wtd = 0.
  recharge_consistency  0 of 256 -- the run was 400 weeks against an intended 20, so the domain
                                 had saturated and gone still 380 weeks before the comparison.

Each cause was different; the SIGNATURE was identical. That is what makes it worth a general
guard rather than three fixes: a fourth suite will drift into it for a fourth reason.

WHAT IS CHECKED. For every output stem in the work directory, the LAST snapshot it wrote -- the
one an assertion is most likely to read. A field is vacuous if it is identically constant over
the whole raster. Zero is the common case but not the only one: a field pinned everywhere at any
single value has no spatial information left to compare.

The FIRST snapshot is deliberately not checked. An initial condition is often uniform by design,
and that is not a defect; what matters is whether the model produced structure by the end.

EXEMPTIONS are named in lib.sh (WTM_VACUITY_EXEMPT), never inferred, and each one must say why
in the code that would otherwise fail it -- the same rule as WTM_DECLARED_EXEMPT. A suite whose
SUBJECT is that the water table is held flat (direct_to_runoff, flicker_evap) is a legitimate
exemption; a suite that merely happens to be flat today is not.
"""
import sys, os, glob, re


def _last_snapshot_per_stem(workdir):
    """{stem: path} for the highest-numbered snapshot each stem wrote.

    Snapshot names are '<prefix><9-digit report>_<years>yr.tif' (src/WTM.cpp snapshot_filename),
    so the stem is everything before the report number and the number orders them.
    """
    pat = re.compile(r"^(?P<stem>.*?)(?P<n>\d{9})_[-0-9.]+yr\.tif$")
    best = {}
    for path in glob.glob(os.path.join(workdir, "*.tif")):
        m = pat.match(os.path.basename(path))
        if not m:
            continue
        stem, n = m.group("stem"), int(m.group("n"))
        if stem not in best or n > best[stem][0]:
            best[stem] = (n, path)
    return {s: p for s, (n, p) in best.items()}


def scan(workdir):
    """[(stem, path, n_distinct, value_if_constant)] for every stem's last snapshot."""
    try:
        import numpy as np, rasterio
    except ImportError:
        return None
    out = []
    for stem, path in sorted(_last_snapshot_per_stem(workdir).items()):
        try:
            a = rasterio.open(path).read(1).astype(float)
        except Exception:
            continue
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            out.append((stem, path, 0, float("nan")))
            continue
        n_distinct = int(np.unique(finite).size)
        out.append((stem, path, n_distinct, float(finite.flat[0]) if n_distinct == 1 else None))
    return out


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("--summary", "--report"):
        print("usage: nonvacuous.py --summary|--report <workdir>", file=sys.stderr)
        return 2
    rows = scan(sys.argv[2])
    if rows is None:          # no numpy/rasterio: stay silent rather than fail a suite on tooling
        return 0
    if not rows:
        return 0
    bad = [r for r in rows if r[2] <= 1]
    if sys.argv[1] == "--summary":
        print(f"{len(rows)} final fields, {len(bad)} with no structure"
              if bad else f"{len(rows)} final fields, all have structure")
        return 1 if bad else 0
    for stem, path, n_distinct, const in rows:
        if n_distinct <= 1:
            what = "ALL-NaN" if n_distinct == 0 else f"identically {const:g}"
            print(f"  VACUOUS  {stem}  final snapshot is {what}: {os.path.basename(path)}")
    print("  A comparison against a field with no structure asserts nothing. Either the arm is not")
    print("  reaching the regime it was written for, or the run has gone past it into a dead state.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
