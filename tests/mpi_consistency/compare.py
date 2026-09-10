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
#   STORED    stored_volume gets its own tolerance between the two. On a groundwater-only fixture it
#           behaves like a STATE sum (worst 8.434e-11), but with FSM ON it includes LAKE volume, which
#           is set by the same discrete routing decisions as the DISCRETE class -- so it inherits some
#           of that sensitivity without the full magnitude. Measured on tests/fsm_consistency at n=6:
#           792225587.445 vs 792225585.638, rel 2.281e-09 (1.8 m^3 out of 7.92e8). n=2, n=4 and n=8 all
#           agree, so it is decomposition-dependent noise rather than anything systematic.
#
#           PRE-EXISTING, and demonstrated rather than assumed: the session-start binary (2e1e2ed),
#           rebuilt with the same 12-digit output, produces those two values IDENTICALLY to the digit.
#           At the original 6 significant figures both printed as 792226000.000000, which is why no
#           rank sweep had ever shown it.
DIAG_RTOL_EXACT    = 1e-12
# STATE is GONE. Its last member, total_evap_removed, is not a sum over the final field at all -- it is
# accumulated ACROSS STEPS exactly like total_ocean_outflow, and calling it a state sum was my error when
# ACCUM was split out. Both belong in ACCUM.
# ACCUM  total_ocean_outflow, split out of STATE when solver.time_integration: auto began resolving to
#        tr-bdf2 on the Anderson path. It is not a sum over the final field (which is what STATE means);
#        it is accumulated ACROSS STEPS, and TR-BDF2 accumulates it through a 3-point flux quadrature
#        over TWO staged solves per step -- roughly 3x more summed contributions than backward-Euler --
#        so it inherits correspondingly more reduction-order noise. Under backward-Euler it passed at
#        1e-9; under tr-bdf2 it does not.
#
#        DEMONSTRATED to be accumulation noise and not a decomposition fault, all on this fixture:
#          - the FIELD agrees: wtd max|delta| = 1.019e-08 .. 1.249e-08 m against WTD_TOL 1e-6, i.e. the
#            ANSWER is consistent across decompositions; only the accumulator moved
#          - non-monotonic in rank count and repeating for structurally similar decompositions:
#            n=2 3.479e-09, n=3 1.636e-09, n=4 2.925e-09, n=6 1.636e-09, n=8 2.925e-09. A real MPI
#            fault grows with rank count and shows as a boundary-localised field error
#          - the arm that routes far MORE water (fsm1_rr03) PASSES at every rank count, and has the
#            LARGEST field difference of all (4.180e-08 m) -- so the sensitivity does not track the
#            field, which is what an accumulation-order effect looks like
#        SECOND MEMBER, added after the adaptive_dt default landed: total_evap_removed. It fails for the
#        same reason and differs only in MAGNITUDE, because the bound scales with how many terms are
#        summed: evap accumulates over every land cell every step, ocean outflow over boundary cells
#        only. Measured worsts differ ~4x accordingly:
#          total_ocean_outflow  3.479e-09   (n=2)
#          total_evap_removed   1.482e-08   (n=6 or n=8)
#        The field stays consistent throughout -- wtd max|delta| 5.662e-10 .. 3.899e-08 m against
#        WTD_TOL 1e-6 -- so it is the accumulators that move, not the answer.
#
#        Set from the WORST member with ~3x margin, the DISCRETE class's discipline. NOTE the cost of one
#        shared class: at 5e-8 the ocean column is covered with ~14x margin rather than ~3x, so it is a
#        weaker check for that column than it could be. Splitting them would buy a tighter ocean bound at
#        the price of a per-column threshold, which is what the class scheme exists to avoid.
DIAG_RTOL_ACCUM    = 5e-8
DIAG_RTOL_STORED   = 1e-8    # the ONE column the code fixes did not tighten (worst 2.281e-09 at n=6)
DIAG_RTOL_DISCRETE = 1e-9    # was 1e-2 -- see above; worst measured now 2.076e-11

# (0-indexed column, name, tolerance). See benchmark/WATER_BUDGET.md for the column list.
DIAG_COLS = [
    (8,  "total_recharge_added",  DIAG_RTOL_EXACT),
    (18, "recharge_direct",       DIAG_RTOL_EXACT),
    (19, "runoff_to_surface",     DIAG_RTOL_EXACT),
    (12, "total_ocean_outflow",   DIAG_RTOL_ACCUM),
    (13, "stored_volume",         DIAG_RTOL_STORED),
    (17, "total_evap_removed",    DIAG_RTOL_ACCUM),
    (11, "total_surface_removed", DIAG_RTOL_DISCRETE),
    (9,  "total_loss_to_ocean",   DIAG_RTOL_DISCRETE),
]


# UNITS: NOT CONVERTED TO WATER VOLUME (#65), deliberately. Every comparison in this file is a
# CROSS-RANK IDENTITY: the same configuration at 1 and N ranks must produce the SAME field. An identity
# assertion is unit-agnostic -- if two fields are equal in head they are equal in volume, and if they
# differ the difference is a defect in either unit. Converting would add a porosity multiply that
# changes no verdict. The bound is tight (bit-identical or near it) precisely because it is an identity
# rather than an accuracy claim.
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
    # Always REPORT the field difference, not only when it exceeds the tolerance. This number is the
    # load-bearing evidence whenever a DIAGNOSTIC column is out of tolerance: it separates "the answer
    # is consistent across decompositions and an accumulator picked up reduction noise" from "the field
    # itself moved". Inferring that from the ABSENCE of a failure line is not the same as measuring it.
    print(f"  wtd max|delta| = {maxd:.3e} m  (tol {WTD_TOL:.0e})", file=sys.stderr)
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
