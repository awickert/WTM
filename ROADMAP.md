# WTM: the road to a finalized, usable model

The **running list** — the single source of truth for what is open, in what order, and why.
`HANDOFF_READINESS.md` describes the *state* of the handoff; this file describes the *work remaining*.

**Rewritten 2026-09-11 after a staleness sweep over all 25 open items**, four agents verifying each
claim against the tree rather than against task prose. **Twelve closed as already-done or obsolete.**
**Updated later the same day:** `#34` closed, taking `#91` and `#96` with it; `#102` opened by the guard
that closing it produced. Fourteen closed, one new. 13 open → 12.

## The goal

A model someone else can run, trust and reproduce without the person who built it.

## Why this list keeps going stale — read before adding to it

Andy, 2026-09-11: *"I open one, you create a huge task list to bring our code base in line with it…
and in the end, that request is because the code is now far past the numbered to-do item."*

The fixes here have been **mechanism-level**, and each dissolved a whole class of numbered items at
once. An item can be obsolete without its claim ever having been wrong: the mechanism it depended on
stopped existing, or a general guard made its class unreachable.

| general mechanism | class it dissolved |
|---|---|
| declared-config rule + `full_config.yaml` | every "this test does not state X" item |
| `tests/nonvacuous.py` (structure guard) | every "this suite compares identical/empty fields" item (#34, #91, #96) |
| `src/resolve_defaults.cpp` | every per-key "what does an absent key mean" question |
| `tests/emit_config.sh` **deleted** | the shim's entire failure surface (#99) |
| `tests/wtm_volume.py` + fatal `lint_norms.sh` | head-vs-volume items, **and prevents recurrence** |
| `tests/wtm_log.py` (columns by name) | hardcoded-column-index items (#53's class) |
| `make_work`, `expect_resolved` | lost-evidence and did-this-arm-run-what-it-configured items |
| `-wtm_` namespace closed to **0** call sites | every flag-named item |

**So: re-read this list after each general fix, not before each individual one.** Before working an
item, verify its claim against the tree first.

## Closed by the sweep (verified, not assumed)

`#37` 37 of 37 configs state `time_integration` (claimed 8 of 39) · `#48` all four metric items
shipped · `#53` both prescriptions implemented, 456 step-checks · `#97` sweep performed, negative ·
`#99` moot, shim deleted · `#66` the doc it targets **does not exist in the repo** · `#76` every
Phase-A item closed · `#80` all four levels closed · `#39` `xrank_growth` ships and asserts the growth
rate · plus `#42`, `#65`, `#73` closed earlier today · and `#34` / `#91` / `#96` closed by the work in
section 1 below.

## THE OPEN LIST

Andy, 2026-09-11: *"go through your numbered steps. All those that have to do with issues around the
tests rather than the code itself: set aside. I am interested only in improving the code."* The list is
split on that line. Section A is work on the model; section B is real but is about the harness, and is
parked until the model work is done.

### A – THE CODE: **ONE ITEM**, reopened 2026-09-17

This section was empty for a day. It is not any more.

| # | item | state |
|---|---|---|
| **110** | Snapshot filenames carry the **wrong simulated year** under `mode: ramp` | **FIXED AND VERIFIED (`8df81df`).** All three checks pass on a rebuilt binary: `fixed` filenames are byte-identical to before (so `tests/golden`'s globs are safe — this was the real risk, since `elapsed_time_s` is summed where the old form multiplied); `ramp` now names from the clock (index 15 → `2yr` against a logged 1.8 yr, where the old code said `15yr`); and `adaptive` was already consistent, checked rather than inherited. `snapshot_filename` derived the year as `cycles_done × report_seconds`, which assumes every report spans the nominal duration — true under `fixed` and under `adaptive` (clamped to the report span), false under `ramp` (explicitly unclamped). Measured: the filename said **1140 yr** where the model's own clock said **37670.6 yr**. Both filename builders held identical copies of the wrong expression; they now share one helper reading `params.elapsed_time_s`, which the model already sums. **Three checks outstanding**, first of them a real risk: does a `fixed`-mode filename move? `elapsed_time_s` is summed where the old form multiplied, so `{:.0f}` could round a year differently and break `tests/golden`'s globs. |

**How it was found is the transferable part.** Nothing was looking for it. `examples/island_equilibrium/summarize.py`
cross-checks the year in a raster's NAME against the year in the run LOG, and warned when they
disagreed. A label computed by anything other than the code that did the work can drift from it — and
this one had, silently, for as long as `ramp` has existed.

**It also invalidates a comparison made earlier the same day.** Reading `<prefix>_15yr.tif` from three
solvers, the `newton/ramp` file was not at 15 years. The lake-count disagreement reported then (165 vs
254) carried that error on top of the separate fact that none of the runs had converged.

#### Closed on 2026-09-16, listed rather than deleted

The three that stood here on 2026-09-11 closed the next day, each with its reason:

| # | closed as | commit |
|---|---|---|
| **106** | The equilibrium stop read **pre-FSM** state on the serial path. Size measured before proposing anything, as the item demanded: the two metrics disagree by 52% / 9%, and the disagreement changes the stop decision for `eq_tol` in [4.548, 4.986]. Fixed – the serial path now recomputes from post-FSM `arp.wtd` against post-FSM `arp.wtd_old`. | `4c1e3b4` |
| **102** | The `secant × active_set` refusal is CORRECT and stays: an assembly constraint, since `secant` puts `b = h^n` in the RHS and the active-set pin would not be enforced. The open part was only that it cited `tests/storage_equivalence` as authority for something that suite never checked. Three unearned citations removed; the message now says the identity is true BY CONSTRUCTION. | `af8c6aa` |
| **6** | `scheme_bench` re-run. It was never blocked by #102: the refusal it cited applies to `secant`, and the benchmark does not run `secant`. `active_set` Newton went 1780 → 44 iterations. Two harness bugs fixed on the way (a relative `OUT` that doubled the path, and a producer/consumer directory-name mismatch). The `implicit` arm got 7× worse and was filed as **#107**. | `a5e7247` |

**#107**, filed out of #6, also closed: Newton + `implicit` was **never** converging. It had been
exiting on the head relative-step stagnation test, and #104's residual gate now refuses that false
verdict. Not a regression – the 2026-08-25 benchmark table is the misleading artefact. Confirmed by one
controlled run: `residual_gate 1e+30` restores the old 14 iterations and `CONVERGED_SNORM_RELATIVE`.

**#64** and **#103** closed the same day; both have their own sections below. **#60** closed on Andy's
decision, also below. The remaining open work is section B, plus the one unnumbered item at the end of
this file.

**`#104` — the per-solve water-step test declared convergence after 4-7 iterations on the shipped
`active_set` path, committing a first step tens of metres from the answer.** FIXED (`db54072`,
`2cc272a`, `5101991`): the verdict is judged against what the run has DEMONSTRATED it can reach,
not a fixed reduction. 10 of 20 sweep arms disagreeing → 0. Three conditions, each proven
load-bearing by ablation. Pinned by `tests/tolerance_independence`.

**`#60` — order-aware retry: CLOSED, not parked (Andy, 2026-09-16).** *"I do not care to prove that
adaptive time stepping is more robust."* Its only unblocking condition was a case that would otherwise
abort, i.e. a robustness demonstration. That demonstration is explicitly not wanted, so the item has no
route to being worth doing. Consistent with the same day's decision that adaptive's robustness is
recorded as DESIGN INTENT in `config.yaml` rather than measured. The v2 patch stays in `git stash` —
find it by message, `git stash list | grep '#60 v2'` — because the idea is sound and only its
justification is withdrawn.

### #103 RESOLVED: the stopping test was right, and the reason is physical

Opened on the belief that a run halting with within-cycle motion still present was converging early.
It was not. Andy, 2026-09-16: **"The only valid states are immediately after FSM is run; within-cycle
motion is computational but not physical."** So the per-cycle metric is not a convenience that happens
to dodge the flicker — it is the only correct quantity, and the per-sub-step `max|Δw|` is a flicker
DIAGNOSTIC, correctly located inside `update()` where flicker lives. That argument is now in the design
note at `src/WTM.cpp:906`, which previously stated the choice without the reason.

Its other two questions closed too: `explicit` + adaptive is a property rather than a defect (and is
**not** an FSM interaction — the fixture that shows it runs with `fsm_on = 0`; the clamp is
dt-dependent and adaptive keeps changing `dt`), with a warning shipped; and the `secant × active_set`
refusal is correct as an assembly constraint and stays.

One real defect came out of it and is **not** buried in the closure: `#106`.

### #64 RESOLVED, and it was not a time-stepping problem at all

The task recorded an accuracy defect: adaptive stepping "worse than fixed at equal cost", `error_tol`
inert, error at 27-42% of cells. Re-measured against the corrected metric — error versus the `dt` =
1/1000 yr run, **land cells only, median and max** — none of that survives in the form it was written.

**The median error is ZERO**, at every step size, in both modes. More than half the land matches the
reference bit-for-bit. The "27-42% of cells" counted anything above 1 cm; above 10 cm there are
**three cells**, and they sit in one column against the ocean on the side nearest the depression. The
other ~50 land cells with an ocean neighbour are exact.

**And the headline reverses.** At matched accuracy adaptive needs 13-15x FEWER steps than fixed (17
against 256 for ~1.0 m). It does saturate — refining `dt0` 128x moves it only 1.25 → 0.92 m — but that
is adaptive doing its job. It targets `error_tol` and stops; it is not a convergence ladder.

**The cause is FillSpillMerge's outlet choice, and it is documented behaviour.** `src/dephier.hpp`
says it three times: *"If a depression has more than one outlet at the same level one of them is
arbitrarily chosen; hopefully this happens only rarely in natural environments."* On a flat plateau
every perimeter cell is a tied outlet, so the tie-break decides everything rather than nothing.

`tests/fsm_exit_path` pins all of it: the error follows the water when the geometry is mirrored, it
vanishes with the routing off, and a tie-broken mirror pair agrees to **1.7578e-08 m** — so given a
unique lowest outlet the model has **no** directional preference. Nothing to fix.

NO CODE CHANGE FOLLOWS. Options A (continuous PI on the coupling error) and C (sub-cycling) were both
aimed at an accuracy problem that is really a documented tie-choice on degenerate terrain. C was ruled
out first — there is no sub-step machinery, so it collapses to "use a smaller dt". A is a different
step-size POLICY rather than a refinement, its patch is in `git stash`, and it should be judged on
robustness if it is ever judged at all, not on an accuracy comparison that was measuring the wrong
thing.

NOT AN OPEN ITEM, by decision (Andy, 2026-09-16): adaptive stepping is **intended** to complete runs
that a fixed step cannot, and in principle it should. It shrinks where the problem is hard and grows
where it is easy, so a stiff stretch that would stall at a constant `dt` should be survivable. That is
what it is for. It has not been measured head-to-head on a run that fixed stepping cannot finish, and
direct tests are deferred to later or to users. Recorded as design intent in `config.yaml`, where
someone choosing a mode will read it, rather than carried as a to-do.

### Found 2026-09-17, outside both sections

| # | item | state |
|---|---|---|
| **108** | `examples/island_equilibrium/demo.py` could not run | **FIXED**, four commits. It had **three** independent breakages, and the retired flag it was filed for was the *last* one it would have hit: a legacy flat `key value` config the parser refuses outright, and a placeholder **1-degree-per-cell** geotransform (111 km cells where the demo intends 11 km and 0.9 km) left over from #124, which converted 10 of 15 generators. Now runs both topographies on three solvers, two step modes and a `--dt-weeks` axis. |
| **109** | `solver.time_step.dt_min` is missing from **30 suite configs** | **OPEN.** A regression from this session: the `dt_min` floor landed, `3956ad0` made the model *write* it to `full_config.yaml`, but no suite config *states* it — so the declared-config rule fails them. Measured: 0 configs declare it, 30 resolve it. **This is the rule working**, and it blocks `run_all.sh` being green. |

### B — THE HARNESS, set aside

**RE-VERIFIED against the tree 2026-09-17, every item, and the task text updated with what was found.**
Two of these were mis-filed here, one has grown, and three are confirmed exactly as written.

| # | verified state |
|---|---|
| **58** | **CLOSED 2026-09-17 on a MEASUREMENT.** See the run below. Its rate-vs-exposure finding stands as a permanent record. |
| **59** | **CLOSED 2026-09-17.** Item (b) is measured away by the same run. Item (a) – FSM activity cannot predict the error, because exposure is a property of the trajectory *ahead* – stands permanently and must not be rebuilt. |
| **77** | Confirmed, and its premise is now the shipped default: `active_set` is the default collector (`resolve_defaults.cpp:31`) and `CONFIG_BASELINE.md:29` already cites this measurement. A record, no action claimed. |
| **84** | **ADVANCED 2026-09-18 (`16b5a5f`), not closed.** The count stands: **30 tolerances across 20 suites** (`grep -rnoE '\$\{[A-Z_]*TOL[A-Z_]*:-[^}]*\}' tests/*/run.sh`). `tol_margin.py` is now WIRED IN – `run_all.sh` tees each suite to a scan directory and ranks at the end – so the ranking it was written to produce has been seen for the first time. From a PARTIAL 10-suite scan: 22 assertions print a value beside a tolerance and **6 run within 10× of failing**. One (`newton_solver`, 2.01×) has had its bound DERIVED from a measured sweep; **five remain**, named below. The ranking is only as complete as the scan – a full `run_all.sh` (blocked on `#109`) ranks all 30. |
| **85** | **CLOSED 2026-09-18 (`fc2529b`).** Verified first as written – all 7 references carried `shape=` and nothing else. Now each reference carries provenance, a regold must STATE A REASON (`generate()` exits 2 without one), and `check()` prints that provenance on failure. |
| **90** | **CLOSED 2026-09-18**, all three parts. (a) resolved by MEASUREMENT, not rewording: `impulse` + `fixed` is the one pairing all three collectors can take (the two refusals bite on different axes), so three like-for-like arms were added with the collector as the ONLY variable – the pin's effect survives isolation at 2.672e-01 m. The task had guessed this was unreachable. (b) the header advertised a DELETED assertion; (c) it called active-set "experimental and off by default (`-wtm_active_set`)", wrong three ways. Suite 5 of 5 green. |
| **98** | **CLOSED 2026-09-18.** Confirmed as written (`a_as`/`c_as` resolved identically at `ARM_TOL=1e-5`), then **three** redundant arms were deleted, not two – `tr_as` was missed by the original filing. The coverage gap left behind (`#16`'s configuration) is closed by a new arm, and a model defect surfaced in the process. |

`#50`'s doc half also sits here.

#### The five thin margins that remain — `#84`'s worklist

Verbatim from `tests/tol_margin.py` over a **partial, 10-suite** scan (`active_set`, `budget_closure`,
`dt_sensitivity`, `ghost_boundary`, `golden`, `newton_solver`, `postfsm_metric`, `runoff_collector`,
`storage_equivalence`, `variable_porosity`). Recorded here because the scan directory was `/tmp` and
would not have survived a reboot.

```
   2.01x  newton_solver          PASS  SAME ROOT  Anderson vs Newton at equilibrium: max|dV| = 4.271e-02 m
   2.84x  budget_closure         PASS  Newton + continuation x implicit (impulse) cumulative=2.49e-07
   6.30x  variable_porosity      OK    AGREEMENT  cc vs tr-bdf2: max|dV| = 1.984e-03 m water volume (tol 0.0125)
   6.61x  variable_porosity      OK    BUDGET tr  |exact residual|/recharge = 1.513e-06 (tol 1e-05)
   7.81x  budget_closure         PASS  Newton, unset -> active_set [tight solve] cumulative=6.00e-09
   9.42x  variable_porosity      OK    BUDGET cc  |exact residual|/recharge = 1.062e-06 (tol 1e-05)
```

**AUDITED 2026-09-19, all six lines, by re-running both suites.** Every value reproduced its scan
figure EXACTLY, so these quantities are deterministic and no margin here carries flake risk. The
classification that matters:

| line | verdict |
|---|---|
| `newton_solver` 2.01× | **DERIVED** — `AGREE_TOL` 0.05 → 0.086 from a measured sweep (`16b5a5f`) |
| `budget_closure` 2.84× | **ALREADY DERIVED** — table at `run.sh:471-478`, taken at `snes_stol` 1e-10 "past the solver floor, so this is the model and not the solve". Reproduces `2.491e-07`/`3.521e-07` exactly |
| `budget_closure` 7.81× | **ALREADY DERIVED** — `snes_stol` swept 1e-8/1e-10/1e-12 at `run.sh:455-466`, shown to FLOOR at 1e-10, with the reason the arm was broken at 1e-8 (solver noise 1.780e-04 exceeded the 1e-4 closure being asserted). Reproduces exactly |
| `variable_porosity` 6.61× / 9.42× | **ALREADY DERIVED** — `BUDGET_TOL`'s derivation sits beside it. Its `tr` figure had drifted (1.57e-06 recorded, 1.513e-06 actual) and is refreshed |
| `variable_porosity` 6.30× | **NOT DERIVED — the only real one.** `TOL=0.0125` carries a label, not a derivation |

**THE LESSON, and it changes how the rest of `#84` should be read: a tolerance ranks thin PRECISELY
WHEN IT WAS DERIVED TIGHTLY.** Five of six top-ranked margins were already done properly; they rank
high *because* someone did the work. `tol_margin` ranks by margin alone, so it will keep surfacing the
well-derived bounds first and burying genuinely loose ones further down. The defect `#84` exists to
catch is a bound set too **loose** — a vacuous test reporting success — and margin alone does not
point at it. A better signal would be **thin AND undocumented**: cross-reference whether a derivation
appears near the tolerance. Not built; proposed.

**Still open regardless:** the scan covered 10 suites and ranked 22 of 30 tolerances. Eight, in the
unscanned suites, have never been ranked at all.

The first is DONE: `AGREE_TOL` in `tests/newton_solver/run.sh` went 0.05 → 0.086, derived from a measured
sweep — and the derivation matters more than the number, because **the hypothesis for where that bound
should come from was WRONG.** Sweep; do not assume. The other five are the worklist.

**The rule for all five: DERIVE the bound** — from the run's own solver tolerance, from the precision of
the quantity, or from a measured spread — **never loosen it to buy margin.** A tolerance is what decides
pass/fail, so one set too loose is a vacuous test reporting success.

One line was reported UNPARSED (`budget_closure`, the routing-off arm): `0.3` in the *label* matched the
tool's magnitude rule better than the `1.42e-08` actually compared. The label has since been fixed
(`run.sh:425` now reads "runoff share still booked"), so a re-scan should rank it. The tool no longer
calls a line FAILING when its suite printed PASS — that disagreement is the parse being wrong, and a
false alarm in the one output whose job is ranking real risk is worse than over-reporting.

#### The decisive run that closed `#58` and `#59`

Clean binary at `1507e28`; `tests/golden` transient_test, `continuous` / `active_set` / `tr-bdf2`,
`mode: adaptive`, `error_tol 0.1`, `trace.dt true`, 8 yr. `gn = nest - ncpl` is the interior count that
informs `est_int` — the old `n_in`.

```
 #     dt(yr)          est         eint         ecpl   factor   nest   ncpl     gn  acc
 1     0.9993   1.8255e-02   1.8255e-02   0.0000e+00    1.499    196      0    196    1
 2     0.9993   2.8272e-01   0.0000e+00   2.8272e-01    1.000    196    196      0    1
 3     0.9993   5.1343e-02   5.1343e-02   9.0904e-04    1.099    196     16    180    1
 9     0.1249   1.7401e-03   1.7401e-03   1.3347e-05    1.500    196     32    164    1
```

**Step 2 is the blind step** — all 196 land cells FSM-touched, so `est_int` has no data. It used to
report `est = 0.0` and grow maximally. It now reports `est_cpl = 2.83e-01` against `tol = 0.1` and holds
at `factor = 1.000`, the only step in the run that does not grow. No step reports `est == 0` while the
estimate is valid. The step is still ACCEPTED, correctly: the coupling part withholds growth and never
forces a shrink (`#63`).

### The shipped example is BROKEN — `#108`, found 2026-09-17

Section A stays empty: `src/` is fine. But `examples/island_equilibrium/demo.py:36` passes
`-wtm_bdf2_on_V` to `build/wtm.x`, and the retired-namespace guard (`WTM.cpp:1251`) aborts the run.
**Verified by execution: `rc=1`.** This is the demonstration a new user runs first, and the goal of this
document is a model someone else can run without us.

`#101` closed having found "two benchmark scripts" of this kind. A scan for lines that BUILD A COMMAND
LINE containing `-wtm_` returns **29 files**, four of which are legitimate (the guard itself, two stored
records, and the fork-targeted `scaling_study.py`, where `-wtm_anderson` is correct for v2.0.1). So
~24 candidates, **one verified**. Each needs its target binary identified before being touched: "fixing"
a fork-targeted script would break a working baseline.

### Closed today beyond the sweep

`#34` (with `#91`, `#96`) · `#78` → root-caused into `#104` · **`#52`** — both encoded reproductions now
close, at 319x and ~5800x margin, and are promoted to plain checks. · **`#50`** — measured, and the
answer is that the ramp IS still needed: plain Newton fails at 5 of 6 `dt`, and the single success at
1.0 wk is the same single-`dt` trap. The claim in the docs stands, unchanged.

**`#105` — the default land boundary leaked mass.** Under `boundaries.land: neumann_toposlope` the
ghost head is `h_edge + (topo_edge − topo_inland)`, which is zero flux relative to the *land surface*,
not zero Darcy flux; wherever terrain rises away from the edge it drives water **in**. The solve used
that flux and the budget ignored it. Unaccounted inflow ran to **44.5× recharge**; the exact residual
went `4.4521e+01` → `8.3964e-10`, with `dirichlet_sea_level` and flat terrain both bit-unchanged. It
hid for the suite's whole life because **every other fixture is ocean-ringed**, so the term was a
structural zero and every budget check was true and empty. Now reported as run-log column 26,
`boundary_inflow_gw`, and pinned by `tests/ghost_boundary` — which also asserts the term is nonzero, so
the check cannot go quietly vacuous the way it did before.

**`#54` — CLOSED, 3 of 5 items done, 2 declined with reasons.** Done: the boundary *budget* is asserted,
not just the boundary solution (item 3); the collector × boundary matrix was swept and is clean for
everything that runs, with one standing arm kept for the constraint-on-a-boundary-face crossing that
both `#52` and `#105` needed (item 4); and the off-map flux is now recomputed by a second independent
route — the pre-fix budget *gap* and the term the model *books* agree to five significant figures,
`4.4521e+01` both ways (item 5).

Declined, with the reasoning recorded so it is not re-opened blind:

- *Item 1's remainder, a region-partitioned budget.* It would evaluate
  `storage_change = recharge + boundary_inflow − ocean_outflow − surface_removed − evap` over a subset
  of cells rather than the whole domain. **It does not diagnose boundary conditions.** The budget asks
  whether all the water was accounted for, not whether the right water was moved: on one fixture,
  `neumann_toposlope` and `dirichlet_sea_level` differ by **93.2 m** at 460 of 480 cells and *both*
  close, at 8.4e-10 and 5.6e-09. A wrong ghost formula keeps the solve and the books in agreement and
  the budget stays silent. BC correctness is guarded instead by the Jacobian-vs-finite-difference check
  on the off-map tangent, serial/MPI agreement, and the analytic expectation. For mass-accounting
  defects the coarse discrimination already exists in column 26 — a failure with a large
  `boundary_inflow_gw` points at the edge, one with `boundary_inflow_gw = 0` points at the interior.
  What a per-cell map adds is attribution of the book-vs-solve mismatch, which is a debugging
  convenience, not a guard, and the guard is already at 1e-10.

- *Item 2, a perimeter-scaling property test.* Needs a new fixture whose land-edge extent varies with
  local conditions held fixed. It would measure the missed-edge and double-counted-corner cases; both
  were read in source and look right, but reading is not measuring. Declined as speculative against its
  fixture cost.

## Known repo-hygiene items found by the sweep

- ~~`tests/lib.sh` **contradicts itself**: line 171 says three suites are exempt and names
  `runoff_collector`; line 184 lists two.~~ **HANDLED `3956ad0`.** The prose now says TWO, matching
  `WTM_DECLARED_EXEMPT`, and says in the same breath that `runoff_collector` used to be the third and
  why it stopped being one (its `unset` arm became declarable via the OPTIONAL marker, #92). Written
  down rather than silently corrected, because a list and its description drifting apart is the exact
  defect class that file exists to prevent.
- ~~`tests/lib.sh:155-164` still carries the **pre-#79 ratchet paragraph** ("494 of 494 runs currently
  leave at least one key implicit") directly above the paragraph saying the rule is unconditional.~~
  **HANDLED `3956ad0`.** Removed; `grep -n 494 tests/lib.sh` now returns nothing. What stands in its
  place states the rule unconditionally and explains that the opt-in list was migration scaffolding,
  retired once all 39 suites were materialised (#83).
- ~~`#52`, `#58`, `#59` cite `scratchpad/*.patch` reproduction routes.~~ **HANDLED 2026-09-16.** The
  directory is not in the repo and 3 of 4 patches no longer apply, so the routes were never usable.
  Rather than delete the references, each task now says so at the TOP and names what IS reproducible
  (for `#58`: `BDF2_ADAPTIVE_DESIGN.md` §3.5 at commit `5520e67`, plus the N-sweep from the fixture named
  there), with the patch list demoted to a historical record explicitly marked do-not-use. Deleting them
  would have hidden that the work happened; leaving them as instructions was worse.

## Standing

Nothing has left this machine. Push to `origin` (KCallaghan) is disabled; pushing, tagging and
releasing each need their own explicit, current-message go-ahead.
