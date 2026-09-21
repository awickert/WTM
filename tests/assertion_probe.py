#!/usr/bin/env python3
"""Run a suite under perturbation to answer two of the six questions in ASSERTION_HEALTH.md.

    Q6  does the value WOBBLE?     run the suite twice, unchanged, and diff.
    Q5  can the assertion FAIL?    re-run with the bound tightened below the measured value.

ONE TOOL BECAUSE IT IS ONE ACTION. Both questions are answered by "run this suite again with
something changed", and the Q6 run is what tells Q5 how far to tighten. Building them separately
would run every suite twice as often for no extra information.

WHY Q6 COMES FIRST, restated here because it is the whole reason this exists: a bound sitting 6x
above its measured value means opposite things depending on whether that value moves. If it never
moves, 6x is LOOSE -- a regression must be large to be caught. If it moves by 3x between runs, 6x is
TIGHT and the test will flake. The same number, read two ways, and nothing in the tree currently
measures which case applies.

WHY Q5 IS THE STRONGEST CHECK. A bound no regression can reach is a test that reports success
forever. The mechanism is already in every suite: each tolerance is an overridable shell default
(`NAME="${NAME:-VALUE}"`), so tightening it from outside needs no edit to the suite.

    A BITE MUST FAIL FOR THE RIGHT REASON. A suite exits non-zero for many reasons -- a missing
    fixture, a build mismatch, a crash. This requires the SPECIFIC assertion governed by that
    tolerance to print FAIL. A non-zero exit with no FAIL line is reported as INCONCLUSIVE, not as a
    pass, because "it broke somehow" is not evidence the assertion is live.

    A TOLERANCE THAT CANNOT BE MADE TO BITE IS ITSELF THE FINDING. It means the assertion is
    structurally vacuous rather than merely loose -- the knob is not connected to anything.

COST, and why this is not in run_all.sh: one suite costs 2 runs for Q6 plus one per tolerance for
Q5. That is an occasional audit, invoked deliberately on named suites, not a per-commit check.

USAGE
    ./assertion_probe.py <suite> [<suite>...]        both questions
    ./assertion_probe.py --spread-only <suite>...    just the repeat-run diff (cheap: 2 runs)
    ./assertion_probe.py --bite-only <suite>...      just the tightening (needs 1 baseline + N)

Prints, for each tolerance, the annotation to paste beside it in run.sh. Writes nothing itself: a
measurement that edits the file it measures is one I would not trust either.
"""
import os, re, subprocess, sys, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assertion_health import _value_and_tol, source_evidence   # ONE definition of "an assertion"

_NUMS = re.compile(r"[-+]?(?:[0-9]+\.?[0-9]*[eE][+-]?[0-9]+|[0-9]*\.[0-9]+|[0-9]+)")
TESTS = os.path.dirname(os.path.abspath(__file__))


def run_suite(suite, env_extra=None, timeout=7200):
    """(exit code, stdout+stderr). Runs the suite in its own directory, as a person would."""
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    if env_extra:
        env.update(env_extra)
    d = os.path.join(TESTS, suite)
    try:
        p = subprocess.run(["./run.sh"], cwd=d, env=env, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        return 124, "(timed out)"
    except OSError as e:
        return 127, f"(could not run: {e})"


def tighten(worst, is_floor):
    """The bound to retry with, so the assertion is forced to fail.

    EXTRACTED TO BE TESTABLE. This was three characters inside bite(), reachable only by running a
    real suite, and it was WRONG for floors: a ceiling bites when the bound drops BELOW the measured
    value, a floor when it is RAISED ABOVE it. Lowering both would loosen every floor and then report
    "THE KNOB IS NOT CONNECTED" about assertions that are perfectly live -- and in this tree the
    floors are the non-vacuity guards, so the accusation lands in the worst possible place.
    """
    return worst * 1.1 if is_floor else worst * 0.9


def is_bite(line):
    """Does this output line show an ASSERTION failing, as opposed to the suite breaking?

    EXTRACTED TO BE TESTABLE. A non-zero exit proves nothing: a missing fixture, a build mismatch or
    a crash all produce one. Requiring the word `tol` keeps that discrimination without demanding a
    single print format from 39 independently written suites -- an earlier version required a
    parenthesised bound on the FAIL line itself and read a real bite as INCONCLUSIVE.
    """
    # EITHER MARKER. Requiring "tol" made every FLOOR's failure invisible -- a lower bound prints
    # "(min 1.0)", never "tol" -- so all four of active_set's non-vacuity guards reported
    # INCONCLUSIVE after being made probeable. The bite guards are precisely the ones whose
    # liveness we most need, and they were the ones this could not see.
    low = line.lower()
    return "FAIL" in line and ("tol" in low or "(min " in low)


def assertions(text):
    """{stable key: (value, tol, line)} for every assertion line in a suite's output.

    The key is the line with its NUMBERS BLANKED, so the same assertion can be matched across two
    runs even when its value differs -- which is exactly the case Q6 is looking for.
    """
    out = {}
    for line in text.splitlines():
        vt = _value_and_tol(line)
        if not vt:
            continue
        # THREE fields, not two: the shared parser gained a `floor` flag when (min X) was added, and
        # this consumer was not updated with it. Sharing the parser is still right -- "what counts as
        # an assertion" must have ONE definition -- but a shared return shape has to be changed in
        # both places at once, and it was not.
        val, tol, floor, bname = vt
        if tol <= 0:
            continue
        out[_NUMS.sub("#", line.strip())] = (val, tol, line.strip(), floor, bname)
    return out


def spread(suite):
    """Run twice unchanged; report how far each value moved. Zero is the expected, useful answer."""
    print(f"\n=== {suite}: Q6, does the value wobble? (2 runs, nothing changed) ===")
    rc1, a = run_suite(suite)
    rc2, b = run_suite(suite)
    if rc1 != 0 or rc2 != 0:
        print(f"  SUITE DID NOT PASS (rc {rc1}, {rc2}) -- spread is not meaningful until it does.")
        return None
    A, B = assertions(a), assertions(b)
    common = sorted(set(A) & set(B))
    if not common:
        print("  no assertion lines matched between the two runs.")
        return None
    worst, moved = 0.0, 0
    for k in common:
        va, vb = A[k][0], B[k][0]
        d = abs(va - vb)
        rel = d / max(abs(va), 1e-300)
        if d:
            moved += 1
            worst = max(worst, rel)
            print(f"  MOVES  {rel:9.2e} rel   {A[k][2][:76]}")
    only = (set(A) ^ set(B))
    for k in sorted(only):
        print(f"  APPEARS IN ONE RUN ONLY: {(A.get(k) or B.get(k))[2][:70]}")
    print(f"  {len(common)} assertions compared, {moved} moved, {len(only)} unmatched.")
    if moved == 0 and not only:
        today = datetime.date.today().isoformat()
        print(f"  SPREAD IS ZERO. Paste beside each bound in {suite}/run.sh:")
        print(f"      # SPREAD: 0   verified {today} -- bit-identical across repeat runs")
    return A


def bite(suite, baseline):
    """Tighten each tolerance below its measured value; the assertion it governs must print FAIL."""
    print(f"\n=== {suite}: Q5, can each assertion fail? (tighten the bound and re-run) ===")
    bounds, _ = source_evidence(TESTS)
    mine = bounds.get(suite, [])
    if not mine:
        print(f"  no `NAME=\"${{NAME:-VALUE}}\"` bounds found in {suite}/run.sh -- nothing to tighten.")
        return
    for name, val, _derived, _sp in mine:
        # MATCH BY VALUE *AND* SHAPE. Value alone conflates two different bounds that happen to share
        # a default: xrank_growth has FLAT_MAX=10.0 (a ceiling) and DISTINCT_MIN=10.0 (a floor), so
        # the probe grabbed both, decided the pair was a floor, and RAISED the ceiling -- loosening it
        # and then reporting "THE KNOB IS NOT CONNECTED" about a live assertion.
        # The name carries the shape: *_MIN is a floor, everything else a ceiling. That is the same
        # convention the suites already follow, so it is read rather than invented.
        # BY NAME WHEN THE LINE GIVES ONE -- exact, and it ends the guessing that caused six
        # separate linkage defects. Shape-and-value matching stays as the fallback for a line that
        # has not been converted yet, so nothing breaks mid-migration.
        governed = {k: v for k, v in baseline.items() if v[4] == name}
        if not governed:
            want_floor = name.endswith("_MIN") or "_MIN_" in name
            governed = {k: v for k, v in baseline.items() if v[1] == val and v[3] == want_floor}
        if not governed:
            governed = {k: v for k, v in baseline.items() if v[1] == val}
            if governed:
                print(f"  {name:<14} matches {len(governed)} assertion(s) by value {val:g} but none of "
                      f"the expected shape ({'floor' if want_floor else 'ceiling'}) -- CANNOT PROBE "
                      f"without the bound's NAME on the line.")
                continue
        if not governed:
            print(f"  {name:<14} governs no assertion that printed -- CANNOT PROBE (tol {val:g} "
                  f"appears on no line). That is itself worth knowing.")
            continue
        is_floor = any(v[3] for v in governed.values())
        # TIGHTEN IN THE DIRECTION THAT BITES. A ceiling bites when the bound drops BELOW the measured
        # value; a floor bites when it is raised ABOVE it. Using one direction for both would have
        # loosened every floor and reported "DID NOT BITE" on assertions that are perfectly live.
        worst = min(v[0] for v in governed.values()) if is_floor else max(v[0] for v in governed.values())
        if worst <= 0:
            print(f"  {name:<14} its assertions all measured exactly 0 -- no tighter bound exists. "
                  f"Vacuity here is nonvacuous.py's question, not this one.")
            continue
        tight = tighten(worst, is_floor)
        rc, out = run_suite(suite, {name: repr(tight)})
        # A FAIL LINE NEED NOT CARRY "(tol X)". Requiring that was too strict and reported
        # adaptive_water as INCONCLUSIVE when its bound had in fact bitten: that suite prints the
        # value and tolerance on one line and its verdict on ANOTHER --
        #     FAIL: adaptive=0.0001, water=0.0018 m water volume exceed tol 0.0016 m water
        # no parentheses, different line. Requiring the word `tol` on the FAIL line keeps the
        # discrimination that matters -- a crash or a missing fixture does not mention a tolerance --
        # without also requiring one print format across 39 independently written suites.
        failed = [l for l in out.splitlines() if is_bite(l)]
        if failed:
            print(f"  {name:<14} BITES   at {tight:.3e} (was {val:g}): {failed[0].strip()[:62]}")
        elif rc != 0:
            print(f"  {name:<14} INCONCLUSIVE  rc={rc} but no assertion printed FAIL -- the suite "
                  f"broke for some other reason, which is not evidence the bound is live.")
        else:
            print(f"  {name:<14} DID NOT BITE  suite still PASSED with the bound at {tight:.3e}, "
                  f"below its own measured {worst:.3e}. THE KNOB IS NOT CONNECTED.")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if not args:
        print(__doc__.split("USAGE")[1].strip(), file=sys.stderr)
        return 2
    for suite in args:
        if not os.path.isdir(os.path.join(TESTS, suite)):
            print(f"  no such suite: {suite}", file=sys.stderr)
            continue
        base = None
        if "--bite-only" not in flags:
            base = spread(suite)
        if "--spread-only" not in flags:
            if base is None:
                rc, out = run_suite(suite)
                if rc != 0:
                    print(f"  {suite}: baseline run failed (rc {rc}); cannot choose a tighter bound.")
                    continue
                base = assertions(out)
            bite(suite, base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
