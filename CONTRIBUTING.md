Setup
==================================

Install pre-commit linters with:

    pip3 install pre-commit
    pre-commit install

## Scripted edits to source and test files

Most of this tree's `run.sh` files and several `src/` files are edited programmatically. Two
hazards have each cost real time here, and both are silent:

**1. Substitute the CODE first, before writing any comment that mentions it.** A replacement like
`s.replace(old, new, 1)` matches the FIRST occurrence — and if you have already inserted an
explanatory comment that quotes the original line verbatim, that is the first occurrence. The
replacement lands in your own prose and the code is untouched. Seen 2026-09-23 in
`tests/coupling_convergence/run.sh`, where the result did not even parse as bash.

**2. Never insert a line after one ending in a backslash.** It severs the continuation, and
`bash -n` PASSES on the wreckage — the file is still syntactically valid, it just means something
else. Seen 2026-09-22 in `tests/solver_consistency/run.sh`. Any inserter must walk back over
continuations and refuse that position rather than trusting the syntax check.

**And prefer explicit line numbers to searching** when editing a large file. Search-then-insert
shifts the target for the next insertion; if you must do both, insert the LATER position first so
the earlier one cannot move it. Getting this wrong put two brackets in different loop branches of
`src/WTM.cpp` twice on 2026-09-23.

**After any scripted edit, diff and confirm ONLY the intended change landed** — a text-mode
round-trip can renormalise line endings and turn a two-line change into whole-file churn.
