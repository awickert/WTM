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
