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

### A – THE CODE: **EMPTY** as of 2026-09-17

Every model-code item is closed. The three that stood here on 2026-09-11 closed on 2026-09-16, each
with its reason, and they are listed rather than deleted so a reader can see what the section held:

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

### B — THE HARNESS, set aside

Real, and none of it changes an answer. `#84` tolerance provenance (now partly mechanised by
`tests/tol_margin.py`) · `#85` goldens carry no in-file provenance · `#98` `budget_closure`'s `a_as` ≡
`c_as` · `#90` `active_set`'s header still calls it experimental and off-by-default · `#50`'s doc half ·
`#58 / 59` strike the "not implemented" sections · `#77` a record, no action claimed.

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
