# RESUME FRAME — read this first, then verify it against git before acting

**HEAD:** `git log --oneline -1`. Branch `bdf2-adaptive-dt`, LOCAL, 1105 commits unpushed vs `origin`.
**Tree:** clean. `README.md`'s coverage table is in `git stash`, deliberately (see below).

Treat every structural claim here as a HYPOTHESIS and check it (branch, HEAD, file, stash) before
acting on it. That rule exists because a remembered role once justified an hour of rework.

## WHAT THIS WORK IS FOR

Finish the WTM model code and hand it off. **At-scale validation (Esquibel, North America) is OUT OF
SCOPE** — do not reintroduce it as an open item.

## THE NUMBERED LIST — nothing is open that Andy has not parked

| # | state |
|---|---|
| 77 | a measurement on the record, no action claimed. Not open. |
| **84** | **ADVANCED, not complete.** `tol_margin.py` is now wired into `run_all.sh` and ranks for the first time; the thinnest bound is derived. FIVE thin margins remain, and most of the ~30 tolerances are unexamined. A ranked worklist, not an unknown. |
| **109** | **PARKED BY ANDY.** `solver.time_step.dt_min` is missing from every adaptive-path suite config, so those suites exit 3 on the declared-config rule. **This is the only thing between us and a green `run_all.sh`.** Value is computable: `1e-5 × dt` (`CreateSNES.cpp:352`), verified against the observed `2522.88 s` and `315.36 s`. |
| **111** | The corsica oscillation: fully characterised, mechanism NOT known. See `examples/island_equilibrium/OSCILLATION.md`. Not a work item unless the last candidate is to be tested. |
| **112** | **PARKED PLAN** by Andy: iterate the FSM→recharge coupling within a step. Full design written, including the enumerated rollback state and the byte-identical guard to build first. |

Closed 2026-09-17/18: **#60** (its missing case FOUND), **#85**, **#90**, **#98**, **#108**, **#110**.

## THE RULES THAT GOVERN HOW TO WORK HERE

- **Do not run the full suite at will.** Run the affected suites.
- **Do not flatter Andy.** Memory `feedback_no_flattery_tell_it_straight.md`, added 2026-09-18 with
  measured counts. Attribute the METHOD, never credit the PERSON. Test: would I write the line if the
  result had gone the other way?
- **Commit each logical change as it completes.** Pushing, tagging, releasing, version bumps and
  closing issues ALWAYS need explicit current-message authorisation; committing never does.
- **Report the DELTA on the list**, not the process. Open with the score.
- **Lead with the concrete artifact** — the equation, the code line, the measured number.
- **Don't generate work off the goal path.** Andy caught me doing this twice today and was right both
  times.

## THE MEASUREMENT RULES

- **Error = `|V(a) − V(b)|`** from `tests/wtm_volume.py`, in metres of WATER. Ground truth is the
  `dt` = 1/1000 yr run, never the previous refinement. **LAND cells only, MEDIAN AND MAX together.**
- **The tell for a bad comparison is a number that repeats where it has no business repeating.**
- **Sweep the tolerance to tell noise from a defect: noise scales, a defect plateaus.** Used twice
  today, decisively both times.
- **A conclusion pre-written into a script's `print` is not a measurement.** I did this once
  (`"THE 4x BUFFER NEVER ENGAGES — REFUTED"` printed as literal text before the number existed; the
  number said the opposite).
- **The logs APPEND.** Re-running a configuration adds a second block; a naive line count double-counts.
  `examples/island_equilibrium/summarize.py` reads the last block for this reason.
- **`elapsed_time_s` is SIMULATED seconds, not wall clock** (`transient_groundwater.cpp:2466`). The run
  log carries no wall time at all.

## THE PHYSICS RULE ANDY STATED

**"The only valid states are immediately after FSM is run; within-cycle motion is computational but not
physical."** Written into `src/WTM.cpp`, and now ENFORCED — see the defect below.

## WHAT CHANGED 2026-09-17/18 — 36 commits

### The real model defect: the equilibrium metric read PRE-FSM state on the PRODUCTION path (`92051e4`)

`#106` fixed this for the SERIAL path and left a comment asserting the distributed path was fine. The
comment was false, and the distributed path is the production one (`distribute_recharge = !fsm_on ||
!infiltration_on`). Measured: the logged metric matched the pre-FSM raster pair **digit for digit**,
differing from the physical state by **67× at cycle 0**. Every equilibrium stop on the shipped default
was being decided on a state that does not physically exist.

**Fixed by DELETING the branch**, not correcting it: the pre-FSM loop and `prev_cycle_wtd` are gone, the
post-FSM recomputation runs unconditionally. Net −44 lines. **Pinned by the new suite
`tests/postfsm_metric`**, which was shown to fail on exactly the path `#106` missed and pass on the one
it fixed.

**The transferable lesson:** the WTM-error / post-groundwater-error naming was enforced in
`tests/wtm_volume.py` — Python, the analysis layer. The defect was in C++, the model's control flow,
which had no such guard. **The enforcement went where we were looking, not where the state lives.**

### `#110`: snapshot filenames carried the wrong year under `mode: ramp` (`8df81df`)

Derived as `cycles_done × report_seconds`, which `ramp` does not honour. Said 1140 yr where the clock
said 37670.6. Now named from `params.elapsed_time_s`. Verified: `fixed` names are byte-identical
(the real risk — `tests/golden`'s globs), `ramp` now correct, `adaptive` was already correct.

### `examples/island_equilibrium` — was broken, now the richest fixture we have

Filed as one retired flag; it had **three** breakages, the flag being the last it would hit: a dead
legacy config format, and a placeholder **1-degree-per-cell** geotransform (111 km cells) left from
`#124`, which converted 10 of 15 generators. Now runs `--solver picard|anderson|newton`,
`--mode fixed|adaptive|ramp`, `--dt-weeks N`, `--equilibrium`, and reports a non-converging solver as
an OUTCOME rather than a traceback.

### `#60`'s missing case EXISTS — measured, on real terrain

Corsica, Anderson, serial, step walked up in powers of two: **at 512 wk and 1024 wk `fixed` DIES and
`adaptive` completes.** The task stays closed (Andy does not want the robustness proof), but the case
is no longer hypothetical. Documented at `solver.time_step.mode` in `config.yaml`, alongside the COST:
**adaptive is not cross-rank reproducible on a long run** (108 vs 104 steps at n=1 vs n=4; NOT the
`#56` defect, which is fixed — `nest` and `ncpl` are identical).

### The corsica oscillation — `examples/island_equilibrium/OSCILLATION.md`

Real, `dt`-invariant, a true 210 yr limit cycle in 32 of 14064 cells. **SEVEN mechanisms excluded by
measurement; FIVE explanations of mine refuted.** The mechanism is NOT known, and the document says so.
Untested candidate: FSM's overland TRANSPORT of half the water budget — absent from every reduction
that settled. **An 11×11 patch of the real terrain, with the model's own piecewise `T`, converges
monotonically.**

## THE ONE REAL OPEN ITEM (unchanged)

`README.md`'s coverage table is in `git stash` — find by MESSAGE, `git stash list | grep 'README
coverage table'`, never by index. It needs a full `tests/run_all.sh`, which doubles as the end-to-end
green check. **Blocked on `#109` first**, or it will simply report the `dt_min` failures.

## READINESS

If this file, git history and the task list survive, the state is reconstructible and verifiable
without the conversation. Every number above is in a commit message or a committed document, with the
command that produced it.
