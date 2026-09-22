#!/usr/bin/env python3
"""Tests for the tools that judge every other test. Runs in seconds; no model, no fixtures.

WHY THIS EXISTS. assertion_health.py and assertion_probe.py decide what #113, #114 and #115 see, and
both were written in a single day and debugged ONLY by running them. Five bugs, every one found by
use rather than by test:

  1  the derivation scan read the comment above the VARIABLE, when this repo documents a per-arm
     bound above the ARM -- so budget_closure's best-derived arm, carrying a swept snes_stol table,
     reported UNDERIVED
  2  the label regex mis-paired quotes on any line holding an empty argument (COLL="") and captured
     shell words instead of the label, so no derivation was ever found for that arm
  3  the bite detector demanded a parenthesised bound on the FAIL line itself, and read
     adaptive_water's real bite as INCONCLUSIVE
  4  a stale unpack after the parser gained a third field killed every probe instantly
  5  the bite tightened FLOORS the wrong way -- lowering, which LOOSENS them

NUMBER 5 IS WHY THIS FILE EXISTS RATHER THAN A PROMISE TO BE CAREFUL. It does not crash. It produces
a plausible verdict. And it lands its accusation -- "THE KNOB IS NOT CONNECTED" -- on the non-vacuity
guards specifically, the checks whose whole job is to stop a suite passing while comparing nothing.
It was caught by reading the code while fixing bug 4, because bug 4's crash had stopped every probe
before a floor was ever reached. Nothing would have caught it otherwise.

Every bug above has a case here, so each stays fixed. The ground truth comes from the hand audit of
2026-09-19, which established for six real assertions which were derived and which were not.
"""
import os, sys, tempfile, textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assertion_health as H
import assertion_probe as P

fails = []


def close(a, b):
    """Float comparison. A test file about tolerances compared 1.0e-3 * 0.9 against 0.0009 with `==`
    and failed on 0.0009000000000000001 -- so the first bug this file caught was its own, and it was
    the very mistake it exists to police."""
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-12 * max(abs(a), abs(b), 1.0)
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return all(close(x, y) for x, y in zip(a, b))
    return a == b


def check(name, got, want):
    ok = close(got, want)
    print(f"  {'OK  ' if ok else 'FAIL'} {name}")
    if not ok:
        print(f"         got  {got!r}\n         want {want!r}")
        fails.append(name)


# --- the parser: value, bound, shape ------------------------------------------------------------
check("ceiling parses value and bound",
      H._value_and_tol("  OK  CONSERVATION: max = 1.2e-07 (tol 1e-04)"), (1.2e-07, 1e-04, False, None))
check("floor parses, and is marked a floor",
      H._value_and_tol("  OK  LAKE PERSISTS: max wtd = 9.9212 m (min 1.0)"), (9.9212, 1.0, True, None))
# BUG 4: the third field. A consumer unpacking two died on every line.
# BUG 4 was a consumer unpacking the wrong number of fields. The count changed AGAIN when the
# bound learned to name itself, and this case caught it immediately -- which is the point.
check("parser returns FOUR fields: value, bound, shape, name (bug 4)",
      len(H._value_and_tol("  x = 0.5 (tol 1.0)")), 4)
# A floor's value is normally ABOVE 1, which the ceiling rule (magnitude <= 1) discards outright.
check("a floor may exceed 1 -- the ceiling rule would drop it",
      H._value_and_tol("  max wtd = 65.7 m (min 1.0)"), (65.7, 1.0, True, None))
check("a bare integer is never the compared value",
      H._value_and_tol("  ran 1yr: rel = 4.0e-09 (tol 1e-06)"), (4.0e-09, 1e-06, False, None))
check("no bound on the line -> not an assertion",
      H._value_and_tol("  cc steady wtd: min -3.221 max 0.000 m"), None)

# --- the bound naming itself: exact linkage, no inference -----------------------------------------
check("a named ceiling yields its NAME",
      H._value_and_tol("  max|dV| = 1.98e-03 m (tol TOL=0.0065)"), (1.98e-03, 0.0065, False, "TOL"))
check("a named floor yields its NAME",
      H._value_and_tol("  max wtd = 9.92 m (min LAKE_MIN=1.0)"), (9.92, 1.0, True, "LAKE_MIN"))
check("two bounds sharing a value are told apart by name",
      (H._value_and_tol("  a = 0.5 (tol FLAT_MAX=10.0)")[3],
       H._value_and_tol("  b = 2.0e4 (min DISTINCT_MIN=10.0)")[3]), ("FLAT_MAX", "DISTINCT_MIN"))
check("the bare form still parses -- nothing breaks mid-migration",
      H._value_and_tol("  max|dV| = 1.98e-03 m (tol 0.0065)")[3], None)

# --- #123: a line that claims a bound but yields no value must not vanish -----------------------
check("a line with a bound claims to be an assertion",
      H.claims_bound("  SETTLING: |dwtd| = 0 m (tol TOL=0.0001)"), True)
check("a line with no bound claims nothing",
      H.claims_bound("  cc steady wtd: min -3.221 max 0.000 m"), False)
# The exact case that prompted it: a PERFECT result printed as a bare integer.
check("a bare-integer value yields nothing, so the line is reportable (#123)",
      (H.claims_bound("  |dwtd| = 0 m (tol TOL=0.0001)"),
       H._value_and_tol("  |dwtd| = 0 m (tol TOL=0.0001)")), (True, None))
check("a nan value yields nothing too",
      H._value_and_tol("  RATIO: value = nan (tol T=1e-06)"), None)
check("a floor marker also counts as claiming a bound",
      H.claims_bound("  max wtd = 9.9 m (min LAKE_MIN=1.0)"), True)

# --- #122: assumptions that only held because ceilings were the only shape ---------------------
# A CEILING MAY EXCEED 1. The magnitude rule keeps absolute quantities out of a ratio comparison,
# but xrank_growth's "last/first" ratio is 2747 and the line parsed to NOTHING.
check("a ceiling above 1 still parses (falls back to nearest)",
      H._value_and_tol("  last/first = 2747.06 (tol FLAT_MAX=10.0)"), (2747.06, 10.0, False, "FLAT_MAX"))
check("the <=1 rule still wins when a candidate exists",
      H._value_and_tol("  idx 23: rel = 4.0e-09 (tol T=1e-06)")[0], 4.0e-09)
# FLOORS ARE NOT EXEMPT FROM THE AMBIGUITY LINT. I asserted they were and never tested it.
# A TRAILING COUNT IS NOT AMBIGUOUS -- it is EXCLUDED. The bare-integer rule ("a count, an index or
# a duration is never the compared value") was applied only to ceilings, because floors were added
# later; making it uniform fixes this case at the root rather than flagging it.
check("a floor with a trailing count parses the measurement, not the count (#122)",
      H._value_and_tol("  max wtd = 9.9212 m over 5 cells (min LAKE_MIN=1.0)")[0], 9.9212)
check("...and is therefore not ambiguous",
      H.ambiguous("  max wtd = 9.9212 m over 5 cells (min LAKE_MIN=1.0)"), False)
check("a bare integer in a floor's label cannot outrank the value (#122)",
      H.ambiguous("  max|ΔV(1yr) - ΔV(quarter-yr)| = 3.4e-02 m (min BITE_MIN=0.001)"), False)
check("a floor with a trailing decimal IS ambiguous (#122)",
      H.ambiguous("  spread = 18.0x across the grid, 0.5 threshold (min SPREAD_MIN=5.0)"), True)
check("a clean floor is still not ambiguous",
      H.ambiguous("  max wtd = 9.9212 m (min LAKE_MIN=1.0)"), False)

# --- shapes that are not a pass and only one of which is a failure -------------------------------
check("xfail is `pinned`, not a failure", H.classify(0.5, True, "0", True, False), "pinned")
check("unverified is a COVERAGE GAP, not a failure", H.classify(0.5, True, "0", True, True), "NOT ASKED")
check("derived + spread 0 + low headroom is SHARP", H.classify(2.0, True, "0", False, False), "sharp")
check("low headroom with spread unknown is unreadable, not damning",
      H.classify(2.0, True, None, False, False), "spread?")
check("no derivation is the strongest signal, whatever the headroom",
      H.classify(5000.0, False, "0", False, False), "UNDERIVED")

# --- the ambiguity lint (#118): more than one plausible answer on the line --------------------------
# Every case below is a REAL line from this tree, five of them lines that actually misparsed.
check("clean line is not ambiguous",
      H.ambiguous("  max|dV| = 1.984e-03 m water volume (tol 0.0125)"), False)
check("a decimal in the label is ambiguous -- dt_sensitivity",
      H.ambiguous("  max|ΔV(1yr) - ΔV(0.25yr)| = 1.2e-14 m water  (tol 0.00025)"), True)
check("a decimal in the label is ambiguous -- boundary_analytic",
      H.ambiguous("  NEUMANN slope (topo=0.05/cell): residual = 3.479e-08 m (tol 1e-06)"), True)
check("a table column is ambiguous -- tolerance_independence",
      H.ambiguous("    1.0000   fixed   max|dV| = 3.1000e-06   agree (tol 0.0001)"), True)
check("a multi-term breakdown is ambiguous -- limit_cycle's mass balance",
      H.ambiguous("  dRech=4.4e-04 dSurf=2.1e-04 residual=1.1e-05 (tol 1e-3)"), True)
check("the BOUND itself in the prose is ambiguous -- direct_to_runoff",
      H.ambiguous("  gathered max wtd = 4.2e-03 m (<= 0.5, at surface) (tol 0.5)"), True)
# A floor takes the number NEAREST the marker, so an earlier candidate cannot outrank it.
check("floors are exempt -- their rule is nearest, not largest",
      H.ambiguous("  0.25 yr arm: max wtd = 9.9212 m (min 1.0)"), False)
check("a bare integer is not a candidate, so it cannot make a line ambiguous",
      H.ambiguous("  ran 1yr over 8 cycles: rel = 4.0e-09 (tol 1e-06)"), False)

# --- the bite: direction, and what counts ---------------------------------------------------------
# BUG 5. The one that fails quietly.
check("tighten a CEILING downward", P.tighten(1.0e-3, False), 0.9e-3)
check("tighten a FLOOR upward (bug 5)", P.tighten(1.0, True), 1.1)
check("a floor is never loosened by tightening (bug 5)", P.tighten(2.0, True) > 2.0, True)
# BUG 3. A real bite whose FAIL line carries no parenthesised bound.
check("a FAIL line need not parenthesise its bound (bug 3)",
      P.is_bite("FAIL: adaptive=0.0001, water=0.0018 m water volume exceed tol 0.0016 m water"), True)
check("a parenthesised FAIL line counts too",
      P.is_bite("  FAIL AGREEMENT  cc vs tr: max|dV| = 1.98e-03 (tol 0.001)"), True)
check("a FLOOR's failure is a bite too -- it says (min X), never tol (#121)",
      P.is_bite("  FAIL BITE (collectors diverge without active-set): max|dV| = 1.0e-04 m (min 0.0125)"), True)
check("a crash is NOT a bite", P.is_bite("ERROR: could not open file with GDAL!"), False)
check("a missing fixture is NOT a bite", P.is_bite("FAIL: inputs/topography.tif not found"), False)

# An inverted assertion must not be reported as a dead knob (found on storage_equivalence).
check("an xfail line is recognisable as inverted",
      "xfail" in "  xfail  KNOWN: 2.4e-03 m (tol TOL=2.5e-07)".lower(), True)
check("a NOT ASKED line is recognisable as inverted",
      "not asked" in "  unverified  NOT ASKED: 2.386e-03 m (tol TOL=2.5e-07)".lower(), True)

# --- the derivation scan: where a derivation actually lives ---------------------------------------
with tempfile.TemporaryDirectory() as d:
    suite = os.path.join(d, "fake"); os.makedirs(suite)
    # BUG 1: the derivation sits above the ARM, not above the variable.
    # BUG 2: COLL="" on the arm line mis-paired the quotes and hid the label.
    open(os.path.join(suite, "run.sh"), "w").write(textwrap.dedent('''\
        # no derivation here, just units
        TOL="${TOL:-1e-6}"
        BITE_MIN="${BITE_MIN:-0.0125}"
        # MEASURED 2026-09-04 by scaling the solve: 5.733e-08 at snes_stol 1e-8, floors at 1e-10.
        ROUTING=continuous COLL="" METHOD=newton ARM_TOL=1e-4 check "Newton, unset -> active_set [tight solve]" d_ntu
        '''))
    bounds, labels = H.source_evidence(d)
    check("a FLOOR named *_MIN is found, not only *_TOL (#121)",
          sorted(n for n, _, _, _ in bounds.get("fake", [])), ["BITE_MIN", "TOL"])
    check("a bound with no derivation above it reads UNDERIVED",
          [b[2] for b in bounds.get("fake", [])], [False, False])
    check("the ARM's derivation is found (bug 1)",
          [l[1] for l in labels.get("fake", [])], [True])
    check("the arm LABEL survives an empty quoted argument (bug 2)",
          [l[0] for l in labels.get("fake", [])], ["Newton, unset -> active_set [tight solve]"])

# --- the SPREAD note must not be read as a DERIVATION (they answer different questions) -----------
# THE BUG THIS PINS: _EVIDENCE matches MEASURED case-insensitively, and the note answering Q6 says
# "measured ... by assertion_probe.py". Writing the notes therefore marked every bound derived, and
# the UNDERIVED worklist collapsed from 49 to ZERO -- the tool reporting no work left because of its
# own annotation. Q6 and Q4 are separate questions and must be read separately.
with tempfile.TemporaryDirectory() as d:
    suite = os.path.join(d, "fake2"); os.makedirs(suite)
    open(os.path.join(suite, "run.sh"), "w").write(textwrap.dedent('''\
        # SPREAD: 0   measured 2026-09-22 by assertion_probe.py -- bit-identical across repeat runs.
        #             Headroom here therefore measures SENSITIVITY, never flake risk; see
        #             tests/ASSERTION_HEALTH.md sec 3 for why that inverts how a low headroom reads.
        TOL="${TOL:-1e-6}"
        # SPREAD: 0   measured 2026-09-22 by assertion_probe.py; see the note at this file\'s first bound.
        # DERIVED 2026-09-19: swept the solve and the residual floors at 5.7e-08; 3x that.
        REAL_TOL="${REAL_TOL:-1.7e-7}"
        '''))
    bounds, _ = H.source_evidence(d)
    got = {n: der for n, _, der, _ in bounds.get("fake2", [])}
    check("a SPREAD note alone does NOT count as a derivation", got.get("TOL"), False)
    check("a real derivation beside a SPREAD note still counts", got.get("REAL_TOL"), True)
    got_sp = {n: sp for n, _, _, sp in bounds.get("fake2", [])}
    check("the SPREAD value is still read from the stripped note", got_sp.get("TOL"), "0")
    check("a terse one-line SPREAD note is read too", got_sp.get("REAL_TOL"), "0")


# --- linkage precedence: an EXACT name beats a SUBSTRING arm match ---------------------------------
# THE BUG THIS PINS: the code resolved the name, then let any arm label that appeared as a substring
# of the line OVERWRITE it -- so an arm's derivation credited a bound it never derived. Measured on
# one sweep: 26 rows clobbered, 5 verdicts flipped, 3 of them in the dangerous direction (an
# UNDERIVED bound reported as derived, i.e. Q4 work that looks finished and is not).
B = {"s": [("AGREE_TOL", 0.1, False, "0"), ("OTHER_TOL", 7.0, True, "0")]}
A = {"s": [("AGREE implicit vs explicit", True, None)]}
line = "  OK   AGREE implicit vs explicit: max|dV| = 3.509e-02 m (tol AGREE_TOL=0.1)"
check("an exact bound name outranks an arm label that merely appears in the line",
      H.resolve("s", line, "AGREE_TOL", 0.1, B, A), (False, "0"))
check("the arm label is still used when the line names no bound",
      H.resolve("s", line, None, 999.0, B, A), (True, None))


# --- a misconfigured SCAN must not read as a healthy report --------------------------------------
# THE BUG THIS PINS: suite keys come from the .out FILENAME, bounds are keyed by DIRECTORY. When
# run_all.sh named its files after the human-readable display label, every lookup missed and all 149
# rows of the full run read `unlinked`. The tool said "derived is unknown for them" -- a footnote,
# not an alarm -- and a genuinely UNDERIVED bound would have been invisible among them.
#
# USES A REAL SUITE AND A REAL BOUND: assertion_health always scans the actual tests/ directory
# (from __file__), so a fabricated suite can never link and would fail this for the wrong reason.
# That is how the first version of this test was wrong.
import subprocess as _sp
_HERE = os.path.dirname(os.path.abspath(__file__))
_LINE = "  PASS  PER-CELL   max rel error 2.963e-07 (tol CELL_TOL=1e-06) at (row 23, col 23)\n"
def _scan(fname):
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, fname), "w").write(_LINE)
        return _sp.run([sys.executable, os.path.join(_HERE, "assertion_health.py"), d],
                       capture_output=True, text=True).stdout
check("an .out named for the suite DIRECTORY links, and raises no alarm",
      "SCAN LOOKS MISCONFIGURED" in _scan("local_ledger.out"), False)
check("an .out named for a DISPLAY LABEL raises the misconfiguration alarm",
      "SCAN LOOKS MISCONFIGURED" in _scan("per_cell_column_ledger.out"), True)


print()
if fails:
    print(f"ASSERTION TOOLS: {len(fails)} FAILED -- {', '.join(fails)}")
    sys.exit(1)
print("ASSERTION TOOLS: ALL PASSED")
