"""Compare water tables in WATER VOLUME, not head.

WHY THIS EXISTS. WTM conserves WATER. Its budget is in cubic metres, its equilibrium stop
(`eq_tol`), its adaptive step target (`error_tol`) and, since #61, its per-solve convergence test
are all judged as |S*dwtd|. A tolerance written in metres of HEAD is therefore in the wrong units
for almost everything a test wants to assert, and the error is not cosmetic: below the surface
dV/dwtd is the porosity, so on a phi = 0.25 fixture a head norm over-weights deep cells 4x, and a
cell 40 m down can swing its head a long way while moving almost no water. Every stopping criterion
in the model was converted to water for exactly this reason; this module is how the TESTS follow.

USE THIS RATHER THAN SUBTRACTING RASTERS. `volume_diff` mirrors src/update_effective_storativity.cpp
storedVolume() exactly, including the surface smoothing width, and is verified against the C++ by
tests/verify_wtm_volume.sh. Hand-rolled `abs(a - b)` in a test is a head norm wearing no label.

UNITS. storedVolume returns stored water volume PER UNIT AREA -- a DEPTH in metres, not m^3. So
`volume_diff` is in "metres of water volume": at phi = 0.25 and wtd well below the surface it is one quarter
of the head difference, and above the surface it approaches the head difference (slope -> 1).
"""
import os
import numpy as np

DEFAULT_SMOOTHING = 0.01   # g_storativity_surface_smoothing_width, src/update_effective_storativity.cpp


def stored_volume(wtd, porosity, smoothing=DEFAULT_SMOOTHING, extended_soil=False):
    """Stored water per unit area, V(wtd). Mirrors storedVolume() in the C++."""
    wtd = np.asarray(wtd, dtype=float)
    porosity = np.asarray(porosity, dtype=float)
    if extended_soil:
        return porosity * wtd
    return 0.5 * (wtd * (1.0 + porosity)
                  + np.sqrt(wtd * wtd + smoothing * smoothing) * (1.0 - porosity))


def volume_diff(wtd_a, wtd_b, porosity, smoothing=DEFAULT_SMOOTHING, extended_soil=False):
    """|V(a) - V(b)| per cell, in metres of water volume."""
    return np.abs(stored_volume(wtd_a, porosity, smoothing, extended_soil)
                  - stored_volume(wtd_b, porosity, smoothing, extended_soil))


def read_band(path, band=1):
    """Read one band, keeping the dataset alive until after the read.

    The obvious one-liner -- gdal.Open(p).GetRasterBand(1).ReadAsArray() -- lets the dataset be
    collected before the read and dies with a confusing TypeError deep inside gdal_array.
    """
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(path)
    arr = ds.GetRasterBand(band).ReadAsArray().astype(float)
    ds = None
    return arr


def only_match(pattern):
    """Glob for EXACTLY one file, and refuse ambiguity.

    A prefix glob is a silent-wrong-answer machine: 'runoff_*' also matches 'runoff_hi_*', and two
    different fixtures then report identical statistics with nothing to show anything went wrong.
    """
    import glob
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise AssertionError("pattern %r matched %d files, expected exactly 1:\n  %s"
                             % (pattern, len(hits), "\n  ".join(hits) or "(none)"))
    return hits[0]


def latest_output(prefix, suffix=".tif"):
    """The newest model output for `prefix`, refusing any match that belongs to a DIFFERENT stem.

    Output names are `<prefix><9-digit cycle>_<years>yr.tif`, so the correct file is the
    lexicographically last -- but only if every match really is this stem. `only_match` is the wrong
    guard here: these globs are SUPPOSED to match many files, one per cycle.

    The hazard is a stem that is a prefix of another. `fsm_runoff_` matches `fsm_runoff_hi_...` and,
    because 'h' sorts after a digit, the LAST match is then the wrong fixture's output entirely. That
    is not hypothetical -- it produced two fixtures reporting identical statistics in an
    investigation, caught only because the numbers were suspiciously equal to five figures.

    So: require what follows the prefix to be a DIGIT. `fsm_runoff_hi_...` continues with 'h' and is
    rejected by name instead of silently winning the sort.
    """
    import glob as _glob
    hits = sorted(_glob.glob(prefix + "*" + suffix))
    if not hits:
        raise FileNotFoundError("no output matching %r" % (prefix + "*" + suffix))
    strays = [h for h in hits if not h[len(prefix):len(prefix) + 1].isdigit()]
    if strays:
        raise AssertionError(
            "prefix %r also matched output from another stem -- the last of these would be the WRONG "
            "fixture:\n  %s" % (prefix, "\n  ".join(strays)))
    return hits[-1]


# ---------------------------------------------------------------------------------------------------
# WHICH ERROR IS THIS? Two different quantities, and mixing them manufactures a difference that is not
# there. The names are the model's (see output.extra_rasters.post_groundwater in config.yaml):
#
#   WTM error              measured against the ordinary snapshot -- the MODEL'S answer, post-FSM.
#                          This is what an accuracy claim means unless it says otherwise.
#   post-groundwater error measured against <prefix>postgw_* -- the SOLVE'S answer, before
#                          FillSpillMerge places surface water.
#
# The kind is INFERRED FROM THE FILENAME rather than passed in, because a caller who has to declare it
# is a caller who can declare it wrongly. `error()` then refuses a cross-kind comparison outright: it is
# not a warning, because a plausible number is exactly what makes this mistake survive review.
WTM_ERROR              = "WTM error"
POST_GROUNDWATER_ERROR = "post-groundwater error"


def error_kind(path):
    """Which error a raster measures, read off its name. See the note above."""
    return POST_GROUNDWATER_ERROR if "postgw_" in os.path.basename(path) else WTM_ERROR


def error(run_path, ref_path, porosity, land=None, smoothing=DEFAULT_SMOOTHING,
          extended_soil=False, cell_threshold=0.01, label=None, quiet=False):
    """Error of `run` against `ref`, in metres of water, reported as MEDIAN and MAX.

    MEDIAN AND MAX TOGETHER, always (Andy, 2026-09-16: "median feels more useful"). Max alone is one
    cell and it misleads: on tests/golden's transient fixture the median error is exactly 0.0 at every
    step size while the max is ~1 m, because the whole discrepancy sits in three cells on a lake's
    discharge path. Reporting only the max made a three-cell artefact read as a model-wide inaccuracy.

    `land` is a boolean mask. Pass it. Ocean cells are pinned, contribute exactly 0, and only dilute a
    norm -- they left every rms 14% low before anyone noticed.

    RAISES on a cross-kind comparison rather than returning a number.
    """
    kr, kf = error_kind(run_path), error_kind(ref_path)
    if kr != kf:
        raise AssertionError(
            f"refusing to compare a {kr} raster against a {kf} one:\n"
            f"    run = {run_path}\n    ref = {ref_path}\n"
            "These measure different quantities -- the model's answer and the solve's answer before\n"
            "FillSpillMerge -- so their difference is not an error, it is the two being different\n"
            "things. Compare like with like, and say in the test which kind you meant.")
    d = volume_diff(read_band(run_path), read_band(ref_path), porosity, smoothing, extended_soil)
    if land is not None:
        d = d[land]
    d = d[np.isfinite(d)]
    out = {"kind": kr, "median": float(np.median(d)), "max": float(d.max()),
           "rms": float(np.sqrt((d ** 2).mean())),
           "n_over": int((d > cell_threshold).sum()), "n_total": int(d.size)}
    if not quiet:
        print("  %s%s  median %.4e m   max %.4e m   cells over %g: %d of %d"
              % (kr, (" [%s]" % label) if label else "",
                 out["median"], out["max"], cell_threshold, out["n_over"], out["n_total"]))
    return out


def compare_series(pairs, reference, porosity, label="run",
                   smoothing=DEFAULT_SMOOTHING, extended_soil=False, cell_threshold=0.01, quiet=False):
    """Compare a run against a reference at EVERY time, not just the last one.

    `pairs` is [(time, path), ...] for the run; `reference` is {time: path}. Only times present in
    BOTH are compared -- comparing at mismatched model times is not a comparison at all.

    WHY THIS IS THE DEFAULT SHAPE. An endpoint number cannot distinguish an error that ACCUMULATED
    from one that is an artefact of where you happened to stop, and that distinction changed the
    reading of a real result this week: the endpoint said 1.60 m, and the trajectory said
    0.32 -> 1.27 -> 1.60, i.e. accumulating. Answering it took a second investigation; it should not
    have to. Returns the per-time rows and a `growing` flag.
    """
    rows, prev = [], None
    growing = True
    for t, path in sorted(pairs):
        if t not in reference:
            continue
        d = volume_diff(read_band(path), read_band(reference[t]), porosity, smoothing, extended_soil)
        d = d[np.isfinite(d)]
        row = {"t": t, "max": float(d.max()), "rms": float(np.sqrt((d ** 2).mean())),
               "n_over": int((d > cell_threshold).sum()), "n_total": int(d.size)}
        if prev is not None and row["max"] <= prev:
            growing = False
        prev = row["max"]
        rows.append(row)
    if not rows:
        raise AssertionError("no matched times between the run and the reference -- a comparison at "
                             "mismatched model times is not a comparison")
    if not quiet:
        print("  WATER-VOLUME series  %s vs reference  [metres of water volume, at matched model time]" % label)
        print("    %-14s %-13s %-13s %s" % ("t", "max", "rms", "cells over %g" % cell_threshold))
        for r in rows:
            print("    %-14.6g %-13.4e %-13.4e %d of %d" % (r["t"], r["max"], r["rms"], r["n_over"], r["n_total"]))
        print("    error is %s" % ("ACCUMULATING (monotone in t)" if growing and len(rows) > 1
                                   else "not monotone -- an endpoint number would misrepresent it"))
    return {"rows": rows, "growing": growing}


def compare(wtd_a, wtd_b, porosity, label_a="a", label_b="b",
            smoothing=DEFAULT_SMOOTHING, extended_soil=False, cell_threshold=0.01, quiet=False):
    """Compare two water tables in water volume and PRINT what was compared, in what units.

    The label on a number is written by the code that made the number, so this prints the norms it
    computed, the units, the porosity range it used and how many cells carry the difference -- a max
    alone cannot distinguish one bad cell from a field-wide bias.

    Returns a dict: max, rms, n_over, n_total, threshold.
    """
    d = volume_diff(wtd_a, wtd_b, porosity, smoothing, extended_soil)
    finite = d[np.isfinite(d)]
    out = {"max": float(finite.max()), "rms": float(np.sqrt((finite ** 2).mean())),
           "n_over": int((finite > cell_threshold).sum()), "n_total": int(finite.size),
           "threshold": cell_threshold}
    if not quiet:
        phi = np.asarray(porosity, dtype=float)
        print("  WATER-VOLUME comparison  %s vs %s  [metres of water volume, V(wtd) per unit area;"
              " porosity %.3f-%.3f, smoothing %g m%s]"
              % (label_a, label_b, float(np.nanmin(phi)), float(np.nanmax(phi)), smoothing,
                 ", extended_soil" if extended_soil else ""))
        print("    max %.4e   rms %.4e   cells over %g: %d of %d"
              % (out["max"], out["rms"], cell_threshold, out["n_over"], out["n_total"]))
    return out
