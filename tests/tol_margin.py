#!/usr/bin/env python3
"""How close does each assertion run to its own tolerance?

THE FAILURE THIS SURFACES. A test can fail for two reasons: the model changed, or the number it was
compared against was never far enough from the measurement to mean anything. The second kind costs
whoever reads it real time and teaches nothing -- it is noise wearing a defect's clothes. Andy,
2026-09-11: "I do not have time to fix poorly set-up tests: arbitrary thresholds, arbitrary iter counts,
and other ways in which an error could show up that does not relate to the code."

A margin is tol/value. A margin of 1.2 means the assertion passes by twenty percent, so a different rank
count, compiler, or BLAS will eventually flip it. A margin of 1e6 means the assertion is about the model.

WHAT IT DOES NOT DO. It does not fail a run, and it does not carry a threshold of its own -- that would
just be another invented number, one level up. It sorts every assertion by margin and prints the thin
end of the list, because the fix is never "loosen it": it is to ask what the number should have been
DERIVED from. The two that prompted this are worth keeping as examples:

  STATE == ACC   compared two independent floating-point sums of the same storage and demanded they
                 agree to 1e-12. That is the arithmetic noise floor; it failed at 2.29e-12, a couple of
                 ulps. The derived bound is the precision of the sums, not a round number.
  SAME ROOT      compared Anderson and Newton at equilibrium against 0.05. Each solver converged to the
                 run's own water tolerance, so THAT is what bounds their disagreement -- and it moves
                 automatically if someone tightens the solver.

Reads whatever a suite printed, so it needs no cooperation from the suites themselves.
"""
import re, sys, os, glob

# "... <value> ... (tol <tol>)". Take the number NEAREST the tolerance, not the first on the line: the
# first is usually a magnitude or a cell index. Getting that wrong reported eight perfectly-agreeing
# MPI checks ("rel diff 0.000e+00") as failing, because it had matched their absolute quantity instead.
_TOL = re.compile(r"\(tol\s*([0-9.eE+-]+)\)")
_NUM = re.compile(r"[-+]?(?:[0-9]+\.?[0-9]*[eE][+-]?[0-9]+|[0-9]*\.[0-9]+|[0-9]+)")


def _value_and_tol(line):
    """(worst-case value, tol) for a line stating both, else None.

    WHICH number on the line is the one being compared is genuinely ambiguous -- a line may also carry a
    cell index, an absolute magnitude, or a second statistic. Two rules settle it without guessing:

      1. A quantity compared against a tolerance is a RATIO or a small difference, so candidates are
         numbers of magnitude <= 1. That drops "(row 23, col 23)" and "1.896865911e+10" and keeps
         "rel diff 0.000e+00", "2.963e-07", "4.271e-02".
      2. Where several candidates remain -- "max|dV| = 4.271e-02 ... rms = 1.079e-02" -- take the one
         with the SMALLEST margin. Over-reporting a line costs a glance; under-reporting it is the
         whole failure this script exists to catch.

    A line whose compared value genuinely exceeds 1 yields no candidate and is skipped rather than
    guessed at; those are counted separately so the skip is visible.
    """
    t = _TOL.search(line)
    if not t:
        return None
    try:
        tol = float(t.group(1))
    except ValueError:
        return None
    cands = []
    for n in _NUM.findall(line[: t.start()]):
        # A BARE INTEGER IS NEVER THE COMPARED VALUE. "1yr" contributed a 1 that then dominated every
        # candidate and reported three passing conservation checks as failing. A measured quantity
        # carries a decimal point or an exponent; a count, an index or a duration does not.
        if "." not in n and "e" not in n and "E" not in n:
            continue
        try:
            v = abs(float(n))
        except ValueError:
            continue
        if v <= 1.0:
            cands.append(v)
    if not cands:
        return None
    return max(cands), tol          # max value == smallest margin


def scan(paths):
    out = []
    for p in paths:
        try:
            text = open(p, errors="ignore").read()
        except OSError:
            continue
        suite = os.path.basename(p).rsplit(".", 1)[0]
        for line in text.splitlines():
            vt = _value_and_tol(line)
            if not vt:
                continue
            val, tol = vt
            if tol <= 0:
                continue
            margin = float("inf") if val == 0 else tol / val
            out.append((margin, suite, line.strip()))
    out.sort(key=lambda r: r[0])
    return out


def main():
    args = sys.argv[1:]
    if not args:
        print("usage: tol_margin.py <suite-output-file>...", file=sys.stderr)
        return 2
    paths = []
    for a in args:
        paths.extend(sorted(glob.glob(os.path.join(a, "*.out"))) if os.path.isdir(a) else [a])
    rows = scan(paths)
    if not rows:
        return 0
    thin = [r for r in rows if r[0] < 10.0]
    print(f"  {len(rows)} assertions printed a value beside a tolerance.")
    if not thin:
        print("  every one of them clears its tolerance by more than 10x.")
        return 0
    print(f"  {len(thin)} run within 10x of failing -- those are the ones that can go red without the")
    print("  model changing. The fix is to DERIVE the bound (from the run's own solver tolerance, from")
    print("  the precision of the quantity, from a measured spread), not to loosen it:")
    for margin, suite, line in thin:
        m = "  0.00x (FAILING)" if margin < 1 else f"{margin:7.2f}x"
        print(f"      {m}  {suite:<22s} {line[:88]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
