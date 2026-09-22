# RESUME FRAME — read this first, then verify it against git before acting

**HEAD:** `git log --oneline -1`. Branch `bdf2-adaptive-dt`, LOCAL, **544 commits unpushed vs
`fork/bdf2-adaptive-dt`** (the branch's own tracking ref, and the only comparison that means
anything here). The remote is named `fork`, not `origin`; a stale `origin/master` ref also exists and
is 1187 behind, which is why an earlier version of this line said "1105 vs origin" and was measuring
the wrong thing.
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
| **84** | **SUPERSEDED 2026-09-20 by #113-#116.** It was question 4 of six. The framework is now `tests/ASSERTION_HEALTH.md`. |
| **113** | **DONE 2026-09-22.** Spread measured across 32 suites and it is ZERO everywhere; 67 of 69 bounds declare it. The 2 that do not are outside the framework by nature (see 5e). |
| **114** | **DONE 2026-09-22.** 0 of 74 bounds underived. Five MORE bounds were found and promoted en route – they were Python locals inside heredocs, so #121's sweep could not see them. |
| **115** | **DONE 2026-09-22.** 32 suites probed; all 14 bite guards proven live. Every CANNOT PROBE turned out to be a missing marker, not a defect, and all were fixed. |
| **116** | **DISSOLVED.** 1 of 22, not 6, and that 1 was stale scan data. |
| **109** | **DONE 2026-09-21.** No default: an explicit `mode: adaptive` must STATE `dt_min`, following MODFLOW 6 and ParFlow, which require it the same way. 27 configs migrated at `1e-5 × dt`, behaviour-preserving by construction and proven by `golden` 35/35 unmoved. |
| **124** | **PARKED BY ANDY** ("when I have time to take the decisions"). `dt_min` is the wrong SHAPE: the ERROR TARGET, not the step size, sets how small `dt` must go. **`tests/boundary_consistency/config.yaml` is on a TEMPORARY `dt_min: "0s"` and a green suite must not launder that into permanence.** Options were A (per-suite small floor), B (`"0s"` here – APPLIED, temporarily) and C (revisit the ratio generally – DONE, and it REFUTED the tidy fix: `boundary_analytic` carries the same `error_tol: 1e-08` and the same `24.192 s` floor and never engages it, so stiffness decides, not the setting). There is no option D; an earlier note saying "four options" was wrong. |
| **111** | The corsica oscillation: fully characterised, mechanism NOT known. See `examples/island_equilibrium/OSCILLATION.md`. Not a work item unless the last candidate is to be tested. |
| **112** | **PARKED PLAN** by Andy: iterate the FSM→recharge coupling within a step. Full design written, including the enumerated rollback state and the byte-identical guard to build first. |

Closed 2026-09-17/18: **#60** (its missing case FOUND), **#85**, **#90**, **#98**, **#108**, **#110**.
Closed 2026-09-21/22: **#109**, **#113**, **#114**, **#115**, **#117**–**#123** (the framework's own
tooling), and **#116** dissolved.

## THE RULES THAT GOVERN HOW TO WORK HERE

- **Do not run the full suite at will.** Run the affected suites. **The precondition Andy set for
  the one full run is now MET** – "I think that we should make the new suites before testing
  anything", and #113/#114/#115 are closed – so the next full `run_all.sh` is sanctioned and
  expected. His concern was never the time: it was that a big failure list would pull focus off the
  items still open. That is the reason to run it only when the list is otherwise clear.
- **Do not flatter Andy.** Now in `~/.claude/CLAUDE.md` (GLOBAL, not project memory — Andy moved it
  there 2026-09-18 because it governs all work, not WTM). Attribute the METHOD, never credit the
  PERSON. Test: would I write the line if the result had gone the other way?
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

## WHAT CHANGED 2026-09-21/22 — 38 commits

### `#109`: `dt_min` has NO default, and an adaptive run must state it

MODFLOW 6's `dtmin` was mis-cited here as recommending `1e-5 × dt`. It does not: it is an ABSOLUTE
length in model time units, recommended for REPORTING ALIGNMENT, and the quote is now in
`benchmark/BDF2_ADAPTIVE_DESIGN.md` verbatim with its sha256. Andy chose to follow MODFLOW and
ParFlow – **require it, do not guess it** – so `src/parameters.cpp` refuses an explicit
`mode: adaptive` that omits `dt_min`, and names the value that would preserve today's behaviour.

**It broke one suite, and that is the useful part.** `boundary_consistency` runs at
`error_tol: 1e-08` and needs steps BELOW the `24.192 s` floor; measured `24.192s → rc=2`,
`0.001s → rc=0`. It was one of ~20 configs edited and never run. **Base rate: 1 break in 20 unrun
edits** – the reason the next full run matters.

### `tests/ASSERTION_HEALTH.md`: every bound now answers three questions

**THE INVERSION RULE, and it is the thing to carry:** because spread is zero everywhere, headroom
cannot express flake risk. It measures SENSITIVITY alone, so **a LOW headroom is a SHARP test, not a
fragile one.** Reading it the other way is what started this arc.

74 bounds: all declare their measured spread, all say where their number came from. Two derivation
shapes, defined in sec 5d-bis – a SEPARATING bound cites both edges of a measured gap (many suites
here run a healthy arm AND a broken-by-construction one, so no convention is needed); a ONE-SIDED
bound must say plainly that its multiplier is a CONVENTION.

**Five bugs, every one found by disbelieving a NUMBER rather than by reading code:**

1. The spread note says "measured …", and `MEASURED` is a derivation keyword – writing the notes
   marked all 49 underived bounds as derived. **The tool would have reported no work left because of
   its own annotation.** Caught because 49 bounds do not acquire derivations from a pasted comment.
2. A substring arm-label overwrote the EXACT bound name: 26 rows, 5 verdicts flipped, 3 of them in
   the dangerous direction. Caught because one row printed a verdict that requires `derived=True`
   while the source said otherwise.
3. Five bounds were Python locals inside heredocs, invisible to `#121`'s promotion sweep, printing
   bare literals no override could reach.
4. Seven assertions ran, passed, and printed no bound at all – one printed only on FAILURE.
5. `recharge_consistency` was already derived; the note sat BELOW the definition, where the tool
   does not look. **My own scan for this missed it**, because the evidence regex needs e-notation or
   a keyword and that note is plain decimals and prose.

**The transferable lesson is the same one as 2026-09-17/18, one level up:** the enforcement went
where we were looking. Every one of these was caught by a count that could not be true, never by
inspection.

### Two bounds worth knowing about before a run

- **`local_ledger`'s `MOVED_MIN` is the sharpest in the tree**: 0.511 m measured against a 0.5 floor,
  2% of headroom. It passes, and spread is zero so it is sharp rather than fragile – but it is the
  one most likely to fire on an unrelated fixture change. **If it fails, re-measure the mound before
  hunting a regression.**
- **`0.0125` is INHERITED by three suites** (`adaptive_water`, `snapshot_restart`,
  `recharge_consistency`). `variable_porosity` measured the same-shaped bound on ITS fixture, set
  `0.0065` at 3× the worst case, and recorded that `0.0125` there was "a label with no derivation
  behind it". **That refutation does not transfer across fixtures** – which is why each is flagged
  in its derivation rather than silently re-based. Re-basing is Andy's call, not a doc fix.

### Four bounds are BLUNT without a structural excuse

`fsm_cascade` `CONS_TOL` (3.3e8), `fsm_conservation` `TOL` (7500), and the two `MB_TOL`s (~4000).
Each says so in its own derivation. Tightening them changes what the suites accept, so they were
recorded, not changed. Several OTHER blunt bounds are blunt CORRECTLY – the collapse guards
(`XS_DIFF_MIN`, `LF_DISTINCT_MIN`, `DISTINCT_MIN`) ask a yes/no question, not a size question, and
their derivations say so to stop someone "fixing" them.

## THE ONE REAL OPEN ITEM

`README.md`'s coverage table is in `git stash` — find by MESSAGE, `git stash list | grep 'README
coverage table'`, never by index. It needs a full `tests/run_all.sh`, which doubles as the end-to-end
green check. **NO LONGER BLOCKED:** `#109` is done and the assertion framework is built, so the next
full run both regenerates this table and settles the coverage question.

## NEXT-SESSION GOALS, in order

**Agreed with Andy 2026-09-22.** The frame update below was moved AHEAD of the full run for a
reason: a full `run_all.sh` plus fixing what it finds is the longest operation on the list and
therefore the likeliest moment to lose context. Updating the continuity artifact afterwards is
backwards.

1. **A full `tests/run_all.sh` — 46 suites.** The precondition is met and nothing else gates it.
   It closes three things at once: the end-to-end green check nothing has had since these changes;
   the **11 adaptive suites not exercised in the 2026-09-22 sweep** (`adaptive_restart`,
   `fsm_consistency`, `ghost_boundary`, `golden`, `lake_evap_equals_et`, `limit_cycle`,
   `log_schema`, `mpi_consistency`, `storage_equivalence`, `taper`, `variable_porosity` — 16 of 27
   adaptive suites DID run green); and the regeneration of the stashed `README.md` table.
   **Expect failures.** Andy expects them too: "I expect errors to come up on our full-suite run
   because we have not run it. But I expect that the net effort will still be less than running the
   full suite more frequently."
2. **Fix what it finds, ONE AT A TIME**, committing each. Do not batch, and do not hand the whole
   failure list back for disposal — surface it, keep the order, let Andy pull the next item.
3. **Regenerate the stashed `README.md` coverage table** from that run. Find the stash by MESSAGE.
4. **`#124`** with real data: the 11 suites above are the evidence for whether any floor other than
   `boundary_consistency`'s binds. **Its `dt_min: "0s"` is TEMPORARY** and a green run does not make
   it permanent.
5. **`#112`** if wanted — the plan is complete, including the byte-identical guard to build FIRST.
6. **`#111`'s last candidate** if wanted — FSM's overland TRANSPORT of half the water budget, the
   one pathway absent from every reduction that settled.

**What is NOT a goal:** at-scale validation; re-running corsica/newton to the cap; proposing a sixth
mechanism for the oscillation without testing one of the named candidates first; retightening the
four blunt bounds without Andy deciding what the suites should accept.

## READINESS

If this file, git history and the task list survive, the state is reconstructible and verifiable
without the conversation. Every number above is in a commit message or a committed document, with the
command that produced it.
