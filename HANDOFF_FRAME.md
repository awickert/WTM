# RESUME FRAME — read this first, then VERIFY it against git before acting

> This file has been WRONG twice. Every structural claim below is checkable in seconds; check it.

## CURRENT FRAME (2026-09-24) — supersedes everything under "SUPERSEDED" further down

**State.** Branch `bdf2-adaptive-dt`, HEAD was `7dc5000` when this was written. **RECOMPUTE, never
quote:** `git rev-parse --short HEAD`, `git rev-list --count fork/bdf2-adaptive-dt..HEAD` (2 then),
`git status --porcelain` (clean then). Remote is `fork` (awickert); `origin` (KCallaghan) has push
DISABLED. Pushing needs explicit current-message authorization AND `CLAUDE_ALLOW=push`.

**Why this work exists.** Finish the WTM model and hand it off. At-scale validation (Esquibel, North
America) is OUT OF SCOPE.

### ARC 1 — #112, the FSM coupling iteration: **DONE**

`surface_water.coupling.iterations` defaults to **4**. Two rules govern it, both structural:

1. **`iterations` FOLLOWS `routing`, per arm** — the key is inert under `impulse`/`off` and an explicit
   `>1` is refused there by name. `coupling_iters_for` in `tests/lib.sh`. Eleven suite arms aborted
   before this was understood.
2. **Iterate only once a LAG EXISTS** (`64b5e59`) — the carrier is zeroed at startup, so step 0
   disposes of the initial condition's surface water and step 1 consumes that disposal. Neither is a
   lag. Guard: iterate once the PREVIOUS step consumed a delta. No threshold, no step count.

Why rule 2 matters: the coupling map's multiplier is ~-1, so it ALTERNATES rather than converging,
with amplitude equal to whatever FillSpillMerge must move — **828 on a ponded cold start against
0.09-0.38 in the lagged regime**. Iterating there commits an arbitrary point on an 828-wide
oscillation and `solver.method: newton` aborts on the following step.

Suites **47/47 green** (54 -> 47 when the duplicate converged pass was removed; no coverage lost —
zero-cell count 40 before and after). Record: `benchmark/FSM_COUPLING_ITERATION.md` AMENDMENT 17,
and **17b, which WITHDRAWS** the claim that the iteration repaired FSM's L-R asymmetry (it was the
far side of the cold-start orbit; post-fix the schemes agree to 2.5%).

**Nothing remains on #112.** Phase 4 (taper's converged arm) is moot — taper runs the default directly.

### ARC 2 — #111, the corsica oscillator: **IN PROGRESS, plan written**

**What it is (measured, annual sampling):** 13 cells, rows 118-123 x cols 60-65, driver `(120,63)`.
**Period 220 yr** (not the 210 from decadal sampling), driver swing **24.52 m**, fall limb **99 yr
three times running** while the rise varies (122, 123). Smooth at weekly resolution; the A->B transfer
has no lag at one week; ~80% of the driver's loss cascades to the three cells downslope.

**NEXT ACTION = step A of "THE PLAN FROM HERE" in `examples/island_equilibrium/OSCILLATION.md`:**
sweep `FREE_MIN` in `benchmark/chain_numerics/chain.py` from 0.1 downward, growing the free set from
13 cells toward the island, and watch the driver's span. Rationale: the reduction freezes 19 cells at
their time-means because they move <0.1 m, but **small amplitude is not small influence** — only
amplitude was ever verified. Then steps B (subsystem ablation, ~6 min per arm), C (is removal a
feedback or a sink), D (spatial refinement — the only axis never tested).

### OPEN, PARKED ON ANDY — pending, NOT declined

- **#124** — `dt_min` is the wrong SHAPE (an error target, not a step size, sets how small dt goes).
  `boundary_consistency` is running on a TEMPORARY `dt_min: "0s"`, and a green suite must not make
  that permanent. Options A/B/C (not four): B applied temporarily; C done, and it refuted deriving
  the floor from `error_tol`.
- **#127** — the ramp path RECORDS a `dt_min` it never reads: a provenance value that looks consulted
  and is not.
- **#128** — a BACKUP POLICY for `~/.claude`, which Andy asked for AFTER this compaction. The
  directory is not version-controlled, so the global `debugging` skill and every project memory are
  unbacked; `~/misc/claude-config-backup-2026-09-24/` is a snapshot, not a policy. Scope, mechanism
  and whether anything leaves the machine are Andy's calls — transcripts and credentials must be
  decided separately before any `git init`. See the task for the full framing.

### ARTIFACT ROLES

| artifact | role |
|---|---|
| `examples/island_equilibrium/OSCILLATION.md` | THE #111 record: behaviour, the excluded-by-measurement table (rows 1-7), refuted explanations R1-R5, the 2026-09-24 resampling results, and the plan |
| `benchmark/chain_numerics/chain.py` | the REBUILT reduction — measured 13-cell free set, every cell free, 4 outlets each. **SETTLES** (0.005 m vs 24.52 m) |
| `benchmark/twocell_numerics/twocell.py` | SUPERSEDED as a reduction (3 defects), but KEPT: its lagged-T sweep is still the oracle for "can the discretisation do this?" |
| `examples/island_equilibrium/_work_corsica/` | **GITIGNORED but PERSISTS on disk.** Holds the 20 000 yr run and the restart raster `anderson_fixed_dt31536000_eq_n1_000002000_20000yr.tif`, plus `cfg_REPRO_*.yaml` |

### RAW DATA — kept, untracked, at `examples/island_equilibrium/_work_111/`

The run output the #111 numbers came from was MOVED out of the dead /tmp path and now lives in
`examples/island_equilibrium/_work_111/` (895 MB; gitignored by `_work_*/`, so it persists on disk but
never enters git): `annual_restart_630yr/` (631 rasters, the cycle) and `weekly_zoom_yr131_166/`
(1821 rasters, the fall resolved), plus both repro configs and **`READINESS.md`**, the handoff
readiness test worked through item by item. So the measurements can be RE-DERIVED rather than trusted.

### REPRODUCE (if the raw data is ever lost)

- **Annual restart, 6 min, 630 yr, shows the cycle:** `build/wtm.x` on
  `_work_corsica/cfg_REPRO_annual_restart.yaml`. Restarts from the 20 000 yr raster,
  `report_interval: 1`, `equilibrium_stop.tol: 0`. **Edit `initial_water_table`, `outfile_prefix` and
  `run_log` first — they point into a dead /tmp path.**
- **Weekly zoom, 10 min, yr 131-166:** `cfg_REPRO_weekly_zoom.yaml`, `dt: 604800`, restarts from the
  annual run's yr-131 snapshot.

### TRAPS HIT ON 2026-09-24 — each cost real time

- **`report_interval` changes the EQUILIBRIUM VERDICT.** The `frac` stop counts cells moving >1 mm
  PER REPORT; at annual reporting that is a tenth of a decade's movement, so the run declares
  equilibrium at cycle 26 of 630. **Always set `equilibrium_stop.tol: 0` for these diagnostics.**
  Corsica's "never settles" headline is partly a statement about DECADAL reporting.
- **`pgrep -f "wtm.x <cfg>"` matches the waiting shell's OWN command line** -> false "RUNNING", and a
  launcher that waits on that pattern deadlocks against itself. Use a pidfile + `kill -0`.
- **`cp -p` restores an OLD mtime** -> make skips the rebuild -> you run a stale binary. `touch` after
  restoring, and check with `strings build/wtm.x | grep <new-symbol>`.
- **DMDA arrays are checked out for the whole run.** `VecNorm` returns a stale cached value and
  `VecScale` silently corrupts. Read/write `dmdapack.<field>_dist`. See
  [[finding_dmda_array_vs_vec_trap]].
- **A knob whose readings agree to six figures across every setting is NOT CONNECTED.**

### #111 NEGATIVE RESULTS — do not re-walk

- Time discretisation: 4x refinement (<=0.5%) and 52x (weekly vs annual, 0.3%).
- Solve convergence: 10 000x tighter leaves the amplitude at ratio 1.0000. Not a lagged coefficient.
- Collector choice: all three agree to 0.09 m. FSM outlet switching: lake volume bit-identical.
- The reduction's SIZE: three sizes (2 cells, 7-cell chain, 13-cell measured set) all settle.
- **R1 (open-water evaporation) and R3 (ponding allowance) are INCONCLUSIVE, not refuted** — they
  ablate surface water the oscillating cells never have (they turn at -0.062 m). Their nulls were
  guaranteed.
- The pinned-sink hypothesis (mine, 2026-09-24) is **WITHDRAWN**: the reduction's baseline already
  contains a pinned neighbour and it is STABLE. Pinning goes with stability here, not oscillation.

---

# SUPERSEDED (2026-09-22 and earlier) — kept for history, not for action

**HEAD:** `git log --oneline -1`. Branch `bdf2-adaptive-dt`, LOCAL, **commits unpushed vs `fork/bdf2-adaptive-dt`:
recompute, do not trust a written number** — `git rev-list --count fork/bdf2-adaptive-dt..HEAD` (568 as of 42c205e; the act of writing this figure changes it, which is how the previous version of this line went stale)** (the branch's own tracking ref, and the only comparison that means anything
here; the remote is named `fork`, and a stale `origin/master` also exists). `git log --since='2026-09-22 00:00' --oneline | wc -l` for the day's count (53 as of 42c205e).
**Tree:** clean. `README.md`'s coverage table is COMMITTED and current as of the 46/46 run (2026-09-22); the older stashed version is SUPERSEDED but deliberately not dropped (see below).

Treat every structural claim here as a HYPOTHESIS and check it (branch, HEAD, file, stash) before
acting on it. That rule exists because a remembered role once justified an hour of rework.

## WHAT THIS WORK IS FOR

Finish the WTM model code and hand it off. **At-scale validation (Esquibel, North America) is OUT OF
SCOPE** — do not reintroduce it as an open item.

## THE NUMBERED LIST — nothing is open that Andy has not parked

| # | state |
|---|---|
| 77 | **CLOSED 2026-09-23.** It was a record that said MEASURED and carried no measurement. `tests/CONFIG_BASELINE.md` now has it: iterations flat 14735 -> 14728, answer identical to machine precision under `active_set`, while `explicit` moves both — **and the SCOPE caveat that had been lost**: one 18x18 equilibrium fixture, NOT the cold-start-at-scale regime the smoothing was introduced for. The width is inert WHERE MEASURED; that is not the same as the smoothing being removable. |
| **84** | **SUPERSEDED 2026-09-20 by #113-#116.** It was question 4 of six. The framework is now `tests/ASSERTION_HEALTH.md`. |
| **113** | **DONE 2026-09-22.** Spread measured and it is ZERO everywhere. **Counts move as suites are onboarded — recompute, do not quote: `cd tests && python3 -c "from assertion_health import source_evidence; b,_=source_evidence('.'); print(sum(len(a) for a in b.values()))"`** (86 across 37 suites as of d4a7bcc). The 2 that do not are outside the framework by nature (see 5e). |
| **114** | **DONE 2026-09-22.** 0 bounds underived (86 as of d4a7bcc; see #113's row for the recompute). Five MORE bounds were found and promoted en route – they were Python locals inside heredocs, so #121's sweep could not see them. |
| **115** | **DONE 2026-09-22.** 32 suites probed at the time; all 14 bite guards then existing proven live. **#126 later onboarded 5 more suites, so 25 floors now exist and three of them postdate the bite sweep** (`coupling_convergence/GAP_MIN`, `RATE_MIN`, `multilake/SPREAD_MIN`) — each was individually proven to bite when added, but no single sweep covers all 25. Every CANNOT PROBE turned out to be a missing marker, not a defect, and all were fixed. |
| **116** | **RE-OPENED then CLOSED 2026-09-22.** Its own note said "confirm at the finale run" — the finale run said **21 of 149 unlinked, not 1 of 22**, so the dissolution had been measured on a subset. The note that scheduled the check is what caught it. Resolved by #126. |
| **109** | **DONE 2026-09-21.** No default: an explicit `mode: adaptive` must STATE `dt_min`, following MODFLOW 6 and ParFlow, which require it the same way. 27 configs migrated at `1e-5 × dt`, behaviour-preserving by construction and proven by `golden` 35/35 unmoved. |
| **124** | **PARKED BY ANDY** ("when I have time to take the decisions"). `dt_min` is the wrong SHAPE: the ERROR TARGET, not the step size, sets how small `dt` must go. **`tests/boundary_consistency/config.yaml` is on a TEMPORARY `dt_min: "0s"` and a green suite must not launder that into permanence.** Options were A (per-suite small floor), B (`"0s"` here – APPLIED, temporarily) and C (revisit the ratio generally – DONE, and it REFUTED the tidy fix: `boundary_analytic` carries the same `error_tol: 1e-08` and the same `24.192 s` floor and never engages it, so stiffness decides, not the setting). There is no option D; an earlier note saying "four options" was wrong. |
| **111** | The corsica oscillation: fully characterised, mechanism NOT known. See `examples/island_equilibrium/OSCILLATION.md`. Not a work item unless the last candidate is to be tested. |
| **112** | **IN PROGRESS 2026-09-23** — steps 1, 2 and **2b** committed and INERT. Andy: **iteration is the DEFAULT, 1 pass the opt-out**; **plain Picard**; config key unnamed (his). Design + FIVE amendments at `benchmark/FSM_COUPLING_ITERATION.md`. **The full step is MEASURED: 18 Vecs, not 3, and the set MOVES WITH THE CONFIG (17/16/11/10/8 by arm); n=1 and n=4 give the identical set.** The previous version of this row said the in-loop probe PERTURBED the run and that step 2b needed an out-of-process instrument. **BOTH WRONG — it was a SEGV** (`changed()` matched by position; a step CREATES Vecs), fixed at 3c61334, and the in-loop and deferred instruments then agree step-for-step. **NEXT NEEDS ANDY: which capture shape** — measured set / every non-null Vec / measured set + a runtime check. The guards have now caught the design's own lists wrong three times: a TENTH accumulator, a Vec list that was 1-of-3, and a full-step list that was 3-of-18. |
| **125** | **DONE 2026-09-23.** The MODFLOW mis-citation survived in the two places a user reads — the refusal message and the shipped `config.yaml`. Value unchanged; the false attribution dropped. |
| **126** | **DONE 2026-09-22.** FIVE suites were never onboarded into the assertion framework. 21 unlinked rows: 2 never a defect, 3 COMPUTED-from-config and must stay so, 16 real. |
| **127** | **OPEN, needs Andy.** The ramp RECORDS a `dt_min` it never reads, so `full_config.yaml` asserts a floor that did not act (#27/#35 class). Needs per-arm rendering across 4 suites + a run, and a refuse-vs-stop-emitting decision that is his. |

Closed 2026-09-17/18: **#60** (its missing case FOUND), **#85**, **#90**, **#98**, **#108**, **#110**.
Closed 2026-09-21/22: **#109**, **#113**, **#114**, **#115**, **#117**–**#123** (the framework's own
tooling). **#116 was dissolved here and that dissolution was REFUTED — see its row above.**
Closed 2026-09-22/23: **#77**, **#116** (properly, via #126), **#125**, **#126**.

**TWO NUMBERING SYSTEMS COLLIDE — check which one a `#N` means.** The numbered list above is the
TASK list. The source tree also cites FORK ISSUE numbers in the same `#N` form, and they are
unrelated: `#127` here is the ramp-path `dt_min` item, while `src/CreateSNES.hpp:109`,
`src/CreateSNES.cpp:93` and `src/transient_groundwater.cpp:983` use `#127` for volume-weighted
per-solve convergence. Grepping a task number out of this file can land you in unrelated solver
code. When in doubt, the task list is authoritative for task numbers and `git log` for the rest.

## THE RULES THAT GOVERN HOW TO WORK HERE

- **Do not run the full suite at will.** Run the affected suites. **The precondition Andy set for
  the one full run is now MET** – "I think that we should make the new suites before testing
  anything", and #113/#114/#115 are closed – so the full `run_all.sh` was sanctioned and RAN on 2026-09-22 (46/46). A further run is sanctioned whenever a
  fresh baseline is wanted. His concern was never the time: it was that a big failure list would pull focus off the
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
- **`elapsed_time_s` is SIMULATED seconds, not wall clock** (`transient_groundwater.cpp:2475`). The run
  log carries no wall time at all.

## THE PHYSICS RULE ANDY STATED

**"The only valid states are immediately after FSM is run; within-cycle motion is computational but not
physical."** Written into `src/WTM.cpp`, and now ENFORCED — see the defect below.

## WHAT CHANGED 2026-09-17/18 — the post-FSM metric defect arc

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

## WHAT CHANGED 2026-09-21/22 — #109 and the assertion framework

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

Every bound declares its measured spread and says where its number came from (74 when this was written; 86 as of d4a7bcc). Two derivation
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

## WHAT CHANGED 2026-09-22/23 — the full run, the framework's completion, and #112 starting

*(53 commits since 2026-09-22 00:00 as of 42c205e. The section above overlaps on 09-22: it covers the #109 +
dt_min arc, this one covers everything after.)*

### The assertion framework is COMPLETE: 86 bounds, 0 underived, 0 unlinked

**THE INVERSION RULE is the idea to carry.** Spread was measured at ZERO on every assertion across
32 suites — nothing in this tree flakes. So headroom CANNOT express flake risk; it measures
SENSITIVITY alone, and **a LOW headroom is a SHARP test, not a fragile one.** Misreading that is
what started the whole arc.

Every bound now declares its measured spread and says where its number came from. `#113`, `#114`,
`#115`, `#116`, `#117`-`#123`, `#125`, `#126` all closed.

**Five tooling bugs, every one found by disbelieving a NUMBER rather than by reading code:**
the spread note marked all 49 underived bounds *derived* (it says "measured", and `MEASURED` is a
derivation keyword — the tool would have reported no work left because of its own annotation); a
substring arm-label overwrote the exact bound name (26 rows, 5 verdicts flipped); five bounds were
Python locals inside heredocs, invisible to `#121`'s sweep; seven assertions ran, passed and printed
no bound; and `recharge_consistency` was already derived with the note sitting BELOW the definition.

**`#126` found the framework's coverage was a SUBSET of the tree and nothing said so.** Of 21
unlinked rows: 2 were never a defect (a correct bound the TOOL could not see, because its name
lacked TOL/MIN/MAX/FLOOR/BAR — now documented in sec 5d-ter), 3 are COMPUTED from the run's own
config and must STAY that way (sec 5c-bis says so explicitly), 16 were real.

### The full suite ran GREEN — and is now ONE RUN BEHIND

`tests/run_all.sh`: **46/46, exit 0**, 2026-09-22 (~547 s wall, from the run's own timing and recorded nowhere else). It regenerated `README.md` and
`tests/COVERAGE.md` (375 runs, 192 tests, 36 combinations, 18 uncovered crossings), which is what
the stashed table had been waiting for since 2026-09-16. **`#126` then touched 7 suites and `#125`
touched `src/` and `config.yaml`, so that green result and the table are ONE RUN BEHIND.** Andy
declined a re-run. Do not describe the tree as currently green without saying this.

The ONE failure was `benchmark/mass_balance_config.yaml` still using `trace: []`, retired
2026-09-16 — **`#101`'s class, third occurrence**: a vocabulary migration that sweeps `tests/` and
leaves `benchmark/` behind. The harness had also sent the model's explanation to `/dev/null`.

**The assertion tool was BLIND under `run_all.sh`** — it keys bounds by suite DIRECTORY and looked
them up by the `.out` FILENAME, which `run_all` built from the display label. All 149 rows read
`unlinked`. Correct standalone, useless in the mode that gates a release.

### Two claims I propagated between documents, both refuted by files already in `benchmark/`

- **FSM is NOT the serial ceiling.** `benchmark/esquibel/FSM_COST.md` measured it: GW solve ~99.7%,
  FSM **0.142%** mean and 0.00007% at cold start. Parallelizing FSM has no speed case; the driver
  is memory. I asserted the opposite twice and committed it.
- **MODFLOW's `1e-5` is an ABSOLUTE length**, not a ratio to the step. The ~35-file correction had
  reached every derived surface and missed the two a user actually reads: the refusal message and
  the shipped `config.yaml`.

**CHECK THE REPO BEFORE REPEATING THE REPO.**

### `#112` STARTED — steps 1 and 2 committed, and both found the design wrong

Andy took `#112` off the parked list on 2026-09-23. The design lives at
**`benchmark/FSM_COUPLING_ITERATION.md`** — verbatim from the task store, plus FOUR amendments. It
was never lost: it had been in the task's DESCRIPTION all along and I read only the SUBJECT.

**HIS DECISIONS, which changed the design:**
1. **ITERATION IS THE DEFAULT; 1 pass is the opt-out.** Not an option bolted on. The transient is
   where spin-up — WTM's actual job — spends all its time.
2. **Plain Picard.** Backed by the ~3x-per-step transient decay, implying contraction ~0.3.
3. **The config key is unnamed. That is his.**

**HIS TWO CORRECTIONS, both sharper than my analysis:**
- **The settled-regime number is NOT the coupling error.** At a fixed point `w_{n+1} = w_n`, so
  `FSM(w_n)` and `FSM(w_{n+1})` are the SAME ARRAY — the lag is IDENTICALLY ZERO by construction.
  The design's headline `1.58e-06 settled` measures DISTANCE FROM EQUILIBRIUM. **The lag is
  DEFINITIONALLY transient-only.** It also means both schemes share a fixed point, so equilibrium
  goldens should NOT move physically — only by ~the stop tolerance.
- **"Arbitrary but consistent?"** Yes — deterministic tie-break, reproducing bit-identically at
  n=1 vs n=4. That buys **EXACT bitwise cycle detection**: `w^{k+2} == w^k` proves a period-2 orbit.
  He has asked this more than once, which means the DOCS were wrong; `CHANGELOG.md` said "if you
  need a deterministic outlet, add a gradient", implying non-determinism. Corrected.

**THE GUARDS EARNED THEMSELVES TWICE — the design's "enumerated from source, not assumed" lists
were wrong BOTH times:**
- It said NINE accumulators. There are **TEN**: `total_storage_change` reads as derived but is
  accumulated independently (`transient_groundwater.cpp:883` `+=`, `:2442` `-=`). Caught by
  `tests/lint_norms.sh` on its FIRST run, before anything depended on the list.
- It named `starting_wtd, lake_stage, rech_vec` for the Vecs. Measured across the coupling call,
  the three that change are **`fsm_delta_vec  lake_stage  wtd_global`** — one right, two missed.

**THE NEGATIVE RESULT THAT SETS THE NEXT ACTION: the in-loop probe PERTURBS the run.**
Instrumenting solve+couple took `tests/fsm_cascade` from rc=0 to rc=2; reverting restored it.
Almost certainly `VecEqual` being a COLLECTIVE, 39 of them inside the step loop. **A measurement
instrument that changes what it measures is useless — do not just move the brackets.**

**NEXT ACTION: measure the full step OUT-OF-PROCESS** — capture during, compare after the loop
exits, or dump Vecs and diff externally. Precedent: the 2026-09-18 lag numbers came from two raster
series with NO code change.

**Committed and INERT** (nothing in the model calls it): `src/coupling_snapshot.hpp`,
`src/test_coupling_snapshot.cpp` (in `test_dmda.x`), the `lint_norms.sh` set-equality guard.
Verified: `strings build/wtm.x | grep -c "FULLSTEP\|PROBE step"` is **0**.

**STILL UNMEASURED, and both gate the design:** the COLD-START lag (needs NO code — two raster
series — and sizes both the cap and the convergence tolerance), and what a warm-started pass 2 costs
in inner iterations (decides whether iterating is ~2x the cycle or a few percent, since **each outer
pass is a full GW RE-SOLVE**, not a cheap FSM call).

## THE README COVERAGE TABLE — DONE 2026-09-22, and the stash is SUPERSEDED

`run_all.sh` regenerates `README.md` and `tests/COVERAGE.md` itself, so the 46/46 green run wrote
them: **375 runs across 192 tests, 36 combinations, 18 uncovered crossings** (was 364/199).
Verified there was NO coverage regression — same 36 combinations, none removed, and not one cell
went from tested to zero, checked cell by cell against HEAD rather than by reading the diff.

The delta is fully accounted for: three test labels vanished and they are exactly `#98`'s deleted
redundant arms (`budget_closure/a_as`, `c_im`, `tr_as`); two appeared — `budget_closure/c_rof`
(`#98`'s gap closure) and **`mass-balance_MPI`, which shows up only now because it was FAILING and
therefore recorded no runs.** Worth knowing about this document: a suite that cannot run is
invisible to it — it reads as absent, not as broken.

**The stash is SUPERSEDED, NOT APPLIED, and NOT DROPPED.** `git stash list | grep 'README coverage
table'` — find by MESSAGE, never by index. It is labelled WRONG in its own message (built from
PARTIAL runs) but it is the only copy of that state, so discarding it is Andy's call, not mine.

## NEXT-SESSION GOALS, in order

**Everything below is parked by Andy or waiting on his decision. Nothing is blocked on me.**

1. **`#112` — step 2b is DONE (8ef9c52); the next move is ANDY'S DECISION on the capture shape.**
   AMENDMENT 5 states the three options with their costs: capture the measured set (smaller, breaks
   SILENTLY when a new integrator appears and there is no syntactic signature to lint it), capture
   every non-null Vec (correct by construction, ~120 MB at 384,703 cells and linear in the grid), or
   the measured set plus a runtime check that nothing outside it moved.
   Then: restore for the chosen set -> step 0's refactor (factor `solve + couple` into ONE
   function; 4 call sites, 3 loops; prove answer-neutral with `golden`) -> config key -> the Picard
   loop with a cap and the EXACT bitwise cycle detector -> the end-to-end byte-identical guard
   (k>1 with unchanged source must equal k=1) -> `config.yaml` + CHANGELOG.

2. **The two measurements that gate `#112`'s design, and the first needs NO code:**
   the COLD-START lag (two raster series, as on 2026-09-18) and the warm-started pass-2 cost.

3. **`#124`** — `dt_min`'s SHAPE. Andy's decision. `tests/boundary_consistency/config.yaml` sits on
   a TEMPORARY `dt_min: "0s"`; a green suite must not launder that into permanence.

4. **`#127`** — the ramp RECORDS a `dt_min` it never reads. Per-arm config rendering across 4
   suites plus a run, and a refuse-vs-stop-emitting decision that is his.

5. **`#111`'s last candidate** if wanted — FSM's overland TRANSPORT of half the water budget. Its
   metric-artifact hypothesis is ALREADY EXCLUDED by measurement (row 1 of `OSCILLATION.md`: pre-
   and post-FSM metrics agree to 1e-12). Do not re-raise it.

6. **A confirming `run_all.sh`** whenever a green baseline is wanted again — the 46/46 of
   2026-09-22 predates `#125` and `#126`.

**What is NOT a goal:** at-scale validation; re-running corsica/newton to the cap; proposing a sixth
mechanism for the oscillation without testing a named candidate; retightening the four
blunt-without-excuse bounds (`fsm_cascade` CONS_TOL 3.3e8, `fsm_conservation` TOL 7500, the two
MB_TOLs ~4000) without Andy deciding what the suites should accept.

## READINESS — checked 2026-09-23, not assumed

**The test: if the conversation were deleted right now, could the next session reconstruct AND
VERIFY the whole state from git + this file + the task list alone?** Yes, and here is the check
rather than the assertion.

| what a resumption needs | where it lives | how to verify it |
|---|---|---|
| **The next action** | `#112` row above, and goal 1 | `benchmark/FSM_COUPLING_ITERATION.md` |
| **Why this work exists** | "WHAT THIS WORK IS FOR" | — |
| **Andy's live decisions** | `#112` row (iteration is the default, plain Picard, key unnamed) | quoted verbatim in the amendments |
| **Branch / HEAD / cleanliness** | header | `git status`, `git log -1` |
| **What is committed vs inert** | `#112` row | `strings build/wtm.x \| grep -c "FULLSTEP\|PROBE step"` must be **0** |
| **Suite state, with its caveat** | this section, below | `tests/run_all.sh` |
| **Results WITH their method** | the "WHAT CHANGED" sections | each cites the file or command |
| **NEGATIVE results** | `#112` row (the probe perturbs), `#124` (C refuted the tidy fix), `#111` (7 mechanisms excluded) | — |
| **Reproduction** | commands inline throughout | — |

**THE SUITE IS NOT CURRENTLY KNOWN-GREEN, and this is the thing most likely to be mis-stated.**
`run_all.sh` was **46/46, exit 0, 547 s on 2026-09-22**. `#125` then changed `src/` and
`config.yaml`, and `#126` changed 7 suites. Andy declined a re-run. So: *last known green, one run
behind.* What IS verified at HEAD: build clean, 30 unit tests at n=1..8, `lint_norms.sh`,
`test_assertion_tools.py`, 86 bounds with 0 underived, `fsm_cascade` green after the probe revert.

**Binary freshness**: no source file is newer than `build/wtm.x` (`find src/ CMakeLists.txt -newer build/wtm.x`). It may PREDATE HEAD when the latest commits are docs-only — that is fine; what matters is that no SOURCE is newer. That check
matters here — a one-commit-stale binary nearly invalidated a whole run on 2026-09-22, caught only
because the commit timestamp was 27 seconds after the build.

**Memory** carries the same frame in `~/.claude/projects/-home-awickert-models-WTM/memory/`:
`project-frame-handoff-readiness` (READ FIRST), `project-112-coupling-iteration-state`,
`finding-fsm-not-the-bottleneck`, `finding-tied-outlets-arbitrary-but-consistent`.

**WHAT THIS FILE HAS BEEN WRONG ABOUT BEFORE** — three times, so distrust it on exactly these:
a commit count measured against the wrong remote; a claim that `#112`'s design was written here when
it was in the task store; and the header count going stale the moment the frame was committed. Every
volatile number now ships with the command that recomputes it.
