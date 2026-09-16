# RESUME FRAME — read this first, then verify it against git before acting

**HEAD:** see `git log --oneline -1`. Branch `bdf2-adaptive-dt`, LOCAL, unpushed.
**Tree:** clean except nothing. `README.md` lives in `git stash` (see below), deliberately.

Treat every structural claim here as a HYPOTHESIS and check it (branch, HEAD, file, stash) before
acting on it. That rule exists because a remembered role once justified an hour of rework.

## WHAT THIS WORK IS FOR

Finish the WTM model code and hand it off. **At-scale validation (Esquibel, North America) is OUT OF
SCOPE** — do not reintroduce it as an open item.

## STATE: the model list is EMPTY

Every model-code item closed. What remains is parked, set aside, or hygiene:

| # | state |
|---|---|
| 60 | **CLOSED** 2026-09-16, not parked. Andy: "I do not care to prove that adaptive time stepping is more robust." Its only unblocking condition WAS that proof. Patch stays in `git stash` — find it by MESSAGE (`git stash list \| grep '#60 v2'`), never by index. |
| 77 | a measurement on the record, no action claimed |
| 58, 59, 84, 85, 90, 98 | harness, set aside by Andy. None changes an answer. |

**THE ONE REAL OPEN ITEM:** `README.md`'s coverage table is in `git stash` (find by message: "README
coverage table from PARTIAL suite runs"). It was regenerated from partial runs, so its counts are wrong
and the new `fsm_exit_path` suite is missing. It needs a full `tests/run_all.sh` — which doubles as the
end-to-end green check nobody has run since today's changes.

## THE RULES THAT GOVERN HOW TO WORK HERE

- **Do not run the full suite at will.** Andy has said so repeatedly. Run the affected suites.
- **Commit each logical change as it completes.** A local commit is the rollback mechanism. Pushing,
  tagging, releasing, version bumps and closing issues ALWAYS need explicit current-message
  authorisation; committing never does.
- **Report the DELTA on the list**, not the process. Open with the score.
- **Lead with the concrete artifact** — the equation, the code line, the measured number — then one
  sentence of what it buys. An abstract label carries nothing Andy can evaluate.
- **Don't generate work off the goal path.** An item being open, blocked, or newly unblockable is not a
  reason to do it.
- **Never give a theoretical concern the force of a measured one.**

## THE MEASUREMENT RULES (learned the hard way today)

- **Error = `|V(a) - V(b)|` from `tests/wtm_volume.py`**, in metres of WATER, not head.
- **Ground truth is the `dt` = 1/1000 yr run.** NEVER the previous refinement: under adaptive stepping
  two runs at different `dt0` are different step PATTERNS, not a refinement of one.
- **LAND cells only**, and **report MEDIAN AND MAX together**. Max alone turned a three-cell artefact
  into an apparent model-wide inaccuracy and several wrong conclusions were built on it.
- **`tests/wtm_volume.py` now enforces this**: `error()` infers the kind from the filename and REFUSES a
  cross-kind comparison. `WTM error` = against the ordinary snapshot (the model's answer);
  `post-groundwater error` = against `<prefix>postgw_*` (the solve's answer, pre-FSM), enabled by
  `output.extra_rasters.post_groundwater`.
- **The tell for a bad comparison is a number that repeats where it has no business repeating.** Two
  metric bugs today were caught that way and nothing else.

## THE PHYSICS RULE ANDY STATED, which several things now rest on

**"The only valid states are immediately after FSM is run; within-cycle motion is computational but not
physical."** This is why the per-cycle equilibrium metric is correct (#103) and why the serial path was
wrong to read pre-FSM states (#106, fixed). It is written into `src/WTM.cpp:906`.

## WHAT CLOSED TODAY, and the one-line reason

- **#105** the default land boundary leaked mass (44.5x recharge unaccounted) → booked; column 26 reports it
- **#104** the per-solve step verdict is judged against what the run has DEMONSTRATED it can reach
- **#64** NOT a time-stepping defect: FSM picks arbitrarily among TIED outlets, as `src/dephier.hpp`
  documents three times. On flat ground every outlet ties. `tests/fsm_exit_path` pins it.
- **#103** the stopping test was right; within-cycle motion is not physical
- **#106** the equilibrium stop read PRE-FSM state on the serial path — fixed
- **#102** the `secant x active_set` refusal is correct and stays; three unearned citations removed
- **#6** scheme_bench re-run; two harness bugs fixed (relative `OUT`; producer/consumer name mismatch)
- **#107** Newton+`implicit` was NEVER converging — it exited on the head relative-step stagnation test.
  #104's gate now refuses that false verdict. The 2026-08-25 benchmark table is the misleading one.

## THE NEWTON RESULT, stated with its limits

Under `active_set`, Newton + continuation converges GENUINELY: 120/120 solves `CONVERGED_FNORM_ABS`,
8-9 iterations each. Under `implicit` it never converges — the siphon `max(0,wtd)/dt` is non-smooth at
`wtd=0` and stiff as `1/dt`. Same property that makes `implicit` refused under adaptive stepping.

**PLAIN Newton still fails everywhere** (rc=1, 2-3 iterations). It needs the ramp. Andy's open question,
and it is NOT answered: whether the Jacobian stays tractable when transmissivity varies as sharply as
real terrain and head make it. `finding_operator_singularity` records exp-`T` spanning 69 orders →
rank-deficient operator; `finding_picard_root_cause` names exp-`T` steepness as Picard's real limiter.
Nothing today tested that, and at-scale is out of scope.

## NEW THIS SESSION, so a reader knows it exists

- `tests/fsm_exit_path` — six geometries; error follows the water, vanishes with routing off, and a
  TIE-BROKEN mirror pair agrees to 1.7578e-08 m (so there is no directional bias, only the tie-choice)
- `solver.time_step.dt_min` — a floor that CLAMPS on the controller path and ABORTS on the failure path,
  plus an unconditional `t + dt == t` roundoff guard
- `output.trace` and `output.extra_rasters` are now MAPS of name → bool, not lists
- `benchmark/n64_adaptive_yardstick/`, `benchmark/lc103_knob_sweep/` — reproduction records
