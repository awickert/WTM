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
      H._value_and_tol("  OK  CONSERVATION: max = 1.2e-07 (tol 1e-04)"), (1.2e-07, 1e-04, False))
check("floor parses, and is marked a floor",
      H._value_and_tol("  OK  LAKE PERSISTS: max wtd = 9.9212 m (min 1.0)"), (9.9212, 1.0, True))
# BUG 4: the third field. A consumer unpacking two died on every line.
check("parser returns THREE fields (bug 4)",
      len(H._value_and_tol("  x = 0.5 (tol 1.0)")), 3)
# A floor's value is normally ABOVE 1, which the ceiling rule (magnitude <= 1) discards outright.
check("a floor may exceed 1 -- the ceiling rule would drop it",
      H._value_and_tol("  max wtd = 65.7 m (min 1.0)"), (65.7, 1.0, True))
check("a bare integer is never the compared value",
      H._value_and_tol("  ran 1yr: rel = 4.0e-09 (tol 1e-06)"), (4.0e-09, 1e-06, False))
check("no bound on the line -> not an assertion",
      H._value_and_tol("  cc steady wtd: min -3.221 max 0.000 m"), None)

# --- shapes that are not a pass and only one of which is a failure -------------------------------
check("xfail is `pinned`, not a failure", H.classify(0.5, True, "0", True, False), "pinned")
check("unverified is a COVERAGE GAP, not a failure", H.classify(0.5, True, "0", True, True), "NOT ASKED")
check("derived + spread 0 + low headroom is SHARP", H.classify(2.0, True, "0", False, False), "sharp")
check("low headroom with spread unknown is unreadable, not damning",
      H.classify(2.0, True, None, False, False), "spread?")
check("no derivation is the strongest signal, whatever the headroom",
      H.classify(5000.0, False, "0", False, False), "UNDERIVED")

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
check("a crash is NOT a bite", P.is_bite("ERROR: could not open file with GDAL!"), False)
check("a missing fixture is NOT a bite", P.is_bite("FAIL: inputs/topography.tif not found"), False)

# --- the derivation scan: where a derivation actually lives ---------------------------------------
with tempfile.TemporaryDirectory() as d:
    suite = os.path.join(d, "fake"); os.makedirs(suite)
    # BUG 1: the derivation sits above the ARM, not above the variable.
    # BUG 2: COLL="" on the arm line mis-paired the quotes and hid the label.
    open(os.path.join(suite, "run.sh"), "w").write(textwrap.dedent('''\
        # no derivation here, just units
        TOL="${TOL:-1e-6}"
        # MEASURED 2026-09-04 by scaling the solve: 5.733e-08 at snes_stol 1e-8, floors at 1e-10.
        ROUTING=continuous COLL="" METHOD=newton ARM_TOL=1e-4 check "Newton, unset -> active_set [tight solve]" d_ntu
        '''))
    bounds, labels = H.source_evidence(d)
    check("a bound with no derivation above it reads UNDERIVED",
          [b[2] for b in bounds.get("fake", [])], [False])
    check("the ARM's derivation is found (bug 1)",
          [l[1] for l in labels.get("fake", [])], [True])
    check("the arm LABEL survives an empty quoted argument (bug 2)",
          [l[0] for l in labels.get("fake", [])], ["Newton, unset -> active_set [tight solve]"])

print()
if fails:
    print(f"ASSERTION TOOLS: {len(fails)} FAILED -- {', '.join(fails)}")
    sys.exit(1)
print("ASSERTION TOOLS: ALL PASSED")
