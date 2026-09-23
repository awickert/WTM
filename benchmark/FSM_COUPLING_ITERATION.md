# #112 — iterate the FSM→recharge coupling within a step

**Status: PARKED by Andy, 2026-09-18.** To be done after the numbered list is finished.

**Provenance.** This is the design verbatim from the task store
(`~/.claude/tasks/<session>/112.json`, created 2026-09-18T17:00Z), copied into the repo on
2026-09-22 and not re-typed. It had never been written to a file here, so
`HANDOFF_FRAME.md` claimed a design that a reader of this repository could not find — the
task store is not the repository, and a handoff does not carry it.

Andy's instruction that created it:

> Before that, I think we should modify the code in general to permit the iteration. We can
> default to skipping the iterations (so just 1 or 0 iterations, depending on how you define
> it). But having that machinery in place will be helpful both for testing and for basic
> correctness.

and, parking it:

> Hm. Let's write this plan and hold it, because I realize that now I am the one wandering
> from our mission. Let's add this as an item to do after we finish going through the
> numbered list.

**Two things the design does NOT contain**, so nobody goes looking: it names no config key
(only "config key + `resolve_defaults` entry"), and it has no under-relaxation or Anderson
acceleration of the outer cycle. Its convergence criterion is a tolerance on the source plus a
hard cap on passes, with a cheaper variant noted below.

---

## AMENDMENT 2026-09-22 — Andy: ITERATION IS THE DEFAULT, not an option

> "The differences during transience are big enough that this should be a standard method for
> running WTM instead of just an option — the option becomes setting just 1 iteration (the current
> mode)."

This INVERTS the default in the plan below, which says "Default to 1 pass (today's behaviour,
byte-identical)". It no longer does. Iterating is the method; `1` is the opt-out.

**The reasoning, and why it beats the plan's own framing.** The plan measured the settled regime
(1.58e-06) and treated the restart transient (17.6% → 5.5% → 1.7%) as secondary. That is backwards
for this model: WTM's production job is SPIN-UP, cold start to equilibrium, so the transient is
where essentially all compute goes and where the answer is a trajectory rather than a fixed point.
The settled number describes the destination, not the journey.

**What this changes in the plan below.**

1. The scope list's item 2 (config key) now defaults to iterating, with `1` as the documented
   escape hatch. The KEY IS STILL UNNAMED — that is Andy's to choose.
2. The "cheaper variant" at the end stops being a variant and becomes the likely default shape:
   iterate UNTIL THE SOURCE STOPS MOVING, with a hard cap. A fixed k multiplies the FSM serial
   ceiling by k on every run; convergence-based iteration pays ~2 passes in transient and 1 when
   settled.
3. The byte-identical guard (build FIRST) does not change, but its ROLE does. It no longer proves
   the default is inert — it proves the OPT-OUT reproduces today's answers. That is still exactly
   the test to write first, and it is now the thing that keeps every pre-2026-09-22 result
   reproducible.

**AMENDMENT 2, same day — Andy, on what the settled number actually measures.**

> "The settled regime of course should have no difference: each step must necessarily be like those
> before when the model is at equilibrium. Think about it numerically."

He is right, and it invalidates the plan's headline evidence rather than merely re-weighting it.

At a fixed point `w_{n+1} = w_n`, so `FSM(w_n)` and `FSM(w_{n+1})` are the SAME ARRAY. The lagged
scheme feeds step n+1 with step n's FSM output, which at equilibrium IS step n+1's own output. **The
lag is not small at equilibrium; it is identically zero, by construction.**

So `settled regime 1.58e-06 relative` is NOT a measurement of the coupling. It is the run's DISTANCE
FROM EQUILIBRIUM. The reasoning is circular: the lag is small because the state is barely changing,
and the lag IS the change. The plan states its own refutation one line below the number -- "FSM's
output stops moving because the lakes do (volume 4751.58 -> 4751.71 over 54 steps)" -- it had the
mechanism and drew the wrong inference from it.

**Consequence 1 — the lag is DEFINITIONALLY transient-only.** Not mostly. There is no regime where
the lagged coupling is both wrong and safely ignored: the only place it can be nonzero is the
transient, which is where all the compute goes.

**Consequence 2 — THE BLAST RADIUS BELOW IS OVERSTATED, and this corrects it.** The lagged scheme's
fixed point satisfies `w* = G(w*, FSM(w*))`; the iterated scheme solves that same equation every
step. SAME FIXED POINT. Iterating changes the PATH, not the DESTINATION.

  - Equilibrium-run goldens should NOT move on physical grounds. One bookkeeping caveat: the
    equilibrium stop fires on per-cycle change, so a changed trajectory can change WHEN it stops and
    therefore which state is written. Expect movement of order the stop tolerance, not of order the
    coupling error.
  - Transient goldens move properly, and should.

**Consequence 3 — a free correctness test, orthogonal to the byte-identical guard.** Iterating must
not move a CONVERGED equilibrium answer by more than the equilibrium tolerance. If it does, either
the run was not converged or the iteration is wrong. The byte-identical guard checks the opt-out
path; this checks the default's physics.

**What still needs measuring is unchanged and now better posed.** The cold-start lag is the only
quantity that sizes the cap and the convergence tolerance, and "how far from equilibrium is this
run" is the correct reading of any settled-regime number taken alongside it.

**AMENDMENT 3 — Andy: is the tied-outlet choice arbitrary but CONSISTENT? Yes, and it changes the
iteration design.**

Deterministic by construction, in three layers: `src/dephier.hpp:449` states the rule ("If two or
more cells are of equal elevation then the one added last"); `FSM_SERIAL_DESIGN.md` records the
explicit tie-break ordering and the DH's Phase-C outlet sort, and notes that serial-on-rank-0
PREVENTS rank-dependence, which the serial≡parallel equilibrium result depends on. Measured:
`fsm_cascade` and `fsm_fullness` both give max|dwtd| n=1 vs n=4 = 0.000e+00, and the 2026-09-22
sweep found spread = 0 on every assertion across 32 suites.

So "arbitrary" means PHYSICALLY UNMOTIVATED, not unpredictable — the mirror-image island shows a
stable, reproducible 8.43 m preference, not noise.

**Consequences for the iteration, and the third one is a design improvement.**

1. `F` is a genuine deterministic function of `w`, so the fixed-point framing is valid and nothing
   hangs on randomness. A DETERMINISTIC period-2 orbit remains possible where the fixed point
   straddles a spill threshold — which is not a new hazard, it is the flicker
   (`FREE_SURFACE_FLICKER.md`: non-contraction of this same outer operator).
2. The byte-identical guard is meaningful ONLY because of this. Against a nondeterministic `F` it
   could not be written at all. Anderson on the outer loop likewise needs a deterministic operator.
3. **CYCLE DETECTION CAN BE EXACT.** If `w^{k+2} == w^k` BITWISE, the iteration is provably in a
   period-2 orbit: stop at once, report it, keep the better of the two states. No tolerance, no
   heuristic, and no spending the whole cap to discover it. A few lines, and it turns the worst
   failure mode from "silently burns k full GW solves going nowhere" into "says so on the first
   repeat." This is only available because the tie-break is consistent.

**AMENDMENT 4 — the Vec rollback surface, MEASURED (2026-09-23), and the design's list is wrong
in both directions.**

The scalars could be enumerated mechanically (`double total_*` is a syntactic signature), and that
lint immediately caught a TENTH accumulator the design had missed. THE VECS CANNOT BE: they are
written through dmdapack's array views (`dmdapack.rech_vec[j][i] = ...`,
transient_groundwater.cpp:1291), so no grep lists them. AppCtx holds 39.

So it was measured instead of judged: capture all 39, run, report which differ. Temporary
instrumentation, since reverted; the list is the deliverable.

```
PROBE step 0 changed: fsm_delta_vec lake_stage wtd_global
PROBE step 1 changed: fsm_delta_vec lake_stage wtd_global
PROBE step 2 changed: fsm_delta_vec lake_stage wtd_global
```

The design named `starting_wtd, lake_stage, rech_vec`. It got ONE right. **`fsm_delta_vec` and
`wtd_global` are missed entirely**, and two of its three do not change here.

**SCOPE, and it is narrower than the rollback needs — do not treat this as the final list.**
  - Measured across the COUPLING CALL only, not across solve+couple. The rollback must restore
    whatever the WHOLE step touches, so the next measurement brackets the solve too.
  - One fixture (fsm_cascade), n=1, three steps, `routing: continuous`, `mode: adaptive`.
  - `rech_vec` not appearing is a result to be careful with: the coupling DOES write it
    (WTM.cpp:346 "set the recharge for the NEXT step"), so on this fixture it is most likely
    written to the SAME value under steady forcing. That is a fixture property, not a general one.

**METHOD NOTE worth keeping.** The probe was first written as a function-local `static` object. Its
destructor runs after MPI_FINALIZE, where VecDestroy is disallowed, and the run aborted with "This
is disallowed by the MPI standard". Any PETSc-owning object with static lifetime has this bug --
including the real snapshot if it is ever held that way.

`src/CreateSNES.hpp` had NO include guard, so any header including it collided. Fixed with
`#pragma once` -- correct on its own merits and unrelated to #112.

**AMENDMENT 5 — the FULL STEP measured (2026-09-23), and the instrument was CRASHING, not perturbing.**

Amendment 4 ends "the next measurement brackets the solve". This is that measurement, and it first
had to correct the record: the note carried into the handoff said the in-loop probe PERTURBED the
run. It did not. It SEGV-ed it. `changed()` walked `saved[k]` by POSITION, which assumes the set of
non-null Vecs is invariant across a step, and **A STEP CREATES VECS**:

    PROBECOUNT captured=33 live=37 appeared: tr_exfil_stage1 tr_fwork tr_head_old vol_prev_x

Those four are allocated lazily on first use (transient_groundwater.cpp:732, :1004, :1876, :1877),
so the compare ran off the end of a 33-entry vector. fsm_cascade reported that only as "RUN FAILED",
exit 2 — which is how a crash spent a day mislabelled as a perturbation. Fixed at 3c61334 (match by
name; [CREATED] and [DESTROYED] reported separately, because undoing a creation means DESTROYING the
Vec, not copying a value back). **No out-of-process instrument was needed.** Running the in-loop and
a deferred-comparison instrument side by side gives step-for-step identical lists, which is the
direct check that the in-loop VecEqual does not disturb the run.

**THE STEP CHANGES 18 VECS, not 3.** Union over every arm below. The coupling call alone changed 3;
the design named 3 and, for the coupling call, got one right.

    exfiltration_vec  fsm_delta_vec  lake_stage  picard_r  rech_vec  sink_removed_dist_vec
    starting_wtd  starting_wtd_local  starting_wtd_prev  T_local  tr_exfil_stage1  tr_expl
    tr_fwork  tr_head_old  tr_ygamma  vol_prev_x  wtd_global  x

**AND IT IS CONFIGURATION-DEPENDENT**, which is the finding that shapes the rollback. Every arm run,
including the two the model refused and the one abandoned:

| arm | step mode | solver | integrator | routing | collector | n | Vecs changed |
|---|---|---|---|---|---|---|---|
| fsm_cascade as shipped | adaptive | anderson | tr-bdf2 | continuous | active_set | 1 | **17** |
| same, decomposed | adaptive | anderson | tr-bdf2 | continuous | active_set | 4 | **17, IDENTICAL SET** |
| step-mode flip | fixed | anderson | tr-bdf2 | continuous | active_set | 1 | 16 (no `starting_wtd_prev`) |
| integrator flip | fixed | anderson | bdf2 | continuous | active_set | 1 | 11 |
| Newton ramp | ramp | newton | backward-euler | continuous | active_set | 1 | 10 (no `tr_*`) |
| Anderson restart ON | adaptive | anderson | tr-bdf2 | continuous | active_set | 1 | 16 (no `vol_prev_x`) |
| Picard | fixed | picard | backward-euler | impulse | explicit | 1 | 8 (the only `picard_r` sighting) |
| Picard x continuous | — | picard | — | continuous | active_set | — | **REFUSED, twice over** |
| multilake | fixed | anderson | — | continuous | active_set | 1 | **ABANDONED at 10 min wall** |

Two of those rows are results rather than gaps. **Picard can never need this rollback under
`continuous`**: `active_set` is refused on Picard (the pin is absent from its operator) and
`explicit` is refused with `continuous` (they fight and never settle). The pair is unreachable, so
`picard_r` only appears under `impulse`. And the multilake arm was abandoned rather than tuned —
flipping one key on the fsm_cascade config is the better comparison anyway, because the fixture is
then held fixed and only the subject moves.

**21 of the 39 AppCtx Vecs changed on NO arm**: `ar_best_x`, `b`, `cellsize_EW_squared`, `evap_vec`,
`fdepth_local`, `fdepth_vec`, `geom_ew_vec`, `geom_n_vec`, `geom_s_vec`, `ksat_local`, `ksat_vec`,
`mask`, `mask_local`, `open_water_evap_vec`, `porosity_vec`, `precip_vec`, `rech_source`,
`runoff_dist_vec`, `runoff_ratio_vec`, `topo_local`, `topo_vec`. Most are static inputs and that is
expected. **NOT OBSERVED IS NOT NEVER**, and `ar_best_x` is the live example: the restart arm enabled
Anderson restarting and `ar_best_x` still never moved, which means restarting did not fire on this
fixture rather than that the Vec is inert. Treat this column as a lower bound on the surface.

**THE DECISION THIS HANDS TO ANDY, and it is a decision, not an implementation detail.** The captured
set is configuration-dependent, so there are two shapes and they trade correctness against memory:

  - **Capture the measured set.** Smaller, but correctness rests on the enumeration staying right. A
    new integrator, or an arm nobody measured, breaks it SILENTLY — which is the exact failure the
    scalar lint exists to prevent, and there is no syntactic signature to lint here.
  - **Capture every non-null Vec.** Correct by construction; the enumeration problem disappears. The
    cost is memory: 39 grid vectors held for one step, which is ~120 MB at Esquibel's 384,703 cells
    and scales linearly with the grid.

A third shape exists and is worth costing: capture the measured set and keep the probe as a
RUNTIME CHECK that nothing outside it moved, so the enumeration is verified by the run rather than
by me. That converts a silent breakage into a loud one at the price of the comparison.

SCOPE, stated as narrowly as it deserves: ONE fixture (fsm_cascade, 30x30-class), n=1 and n=4,
`routing: continuous` except where the table says otherwise. The step-mode and integrator flips are
single-key edits of that one config, so the FIXTURE is controlled and only the subject moves.

**AMENDMENT 6 — the rollback SHAPE chosen, and the three decisions the loop still contains.**

Andy chose (2026-09-23) the third shape of Amendment 5: **carry the measured set as copies, and let
the RUN verify the enumeration**. Shipped at dec5266 as `CouplingVecSnapshot`, still inert.

One consequence is mine, not his, and is stated rather than buried: **the check uses FINGERPRINTS,
not copies.** Holding copies of the other 21 Vecs to compare against would cost exactly the memory
the choice was made to avoid, so the outside-the-set check stores three norms (1, 2, infinity) per
Vec — 3 doubles instead of a grid. It therefore cannot see a mutation that preserves all three
norms; it is a tripwire for "this Vec is not inert after all", not a proof of equality.

**DECISION 1 — THE EXCLUSION SET, and it is exactly one Vec.** A rollback that restored everything
would restore the very quantity the iteration is solving for, and the loop would never move. So the
iteration variable must be EXCLUDED, and under `routing: continuous` it is identifiable in one
place: FSM's per-cell volume change is scattered into its OWN carrier, `fsm_delta_dist`
(WTM.cpp:578), deliberately kept out of `rech_dist` because the delta is internal redistribution
rather than external input. That carrier is `fsm_delta_vec`, and it is the only thing pass k+1 must
inherit from pass k.

  Φ(w) = G(w_n, F(w)) — restore everything, keep F(w). Exclusion set = { `fsm_delta_vec` }.

`rech_vec` is NOT excluded, and that is the part worth stating because the opposite is tempting.
Its post-step value is "the recharge for step n+1", computed from the POST-step water table. A
re-solve of step n must use step n's own recharge — the one the previous step's coupling set — so
`rech_vec` is restored, and the final accepted pass recomputes the next step's value anyway.
Restoring it loses nothing; keeping it would silently advance the forcing by one step inside the
iteration.

**DECISION 2 — does an inner pass count as a solve?** `params.solves_done`, the budget trace and
`nsteps` are all per-step counters today. My proposal: they count the ACCEPTED pass only, so every
existing per-step diagnostic keeps meaning what it means, and the iteration reports its pass count
SEPARATELY. The alternative — counting every pass — would make the run log comparable on cost but
would silently change the meaning of numbers several suites assert on.

**DECISION 3 — THE CONFIG KEY IS UNNAMED AND IS ANDY'S.** Iterating is the default (Amendment 1);
`1` is the opt-out. Nothing below picks a name.

**The blast radius, stated so the decision carries its full cost.**

- **Every golden reference moves,** and every benchmark number in `benchmark/` describes a mode that
  is no longer the default. Those results are not wrong; they are measurements of the 1-pass method
  and must be relabelled as such rather than silently re-baselined.
- **The outer convergence criterion becomes load-bearing.** As an option it was a nicety; as the
  default it decides both the answer and the cost of every production run. It needs a DERIVED
  tolerance under `tests/ASSERTION_HEALTH.md` sec 5d-bis, not a chosen one.
- **Cost: FAR SMALLER THAN THIS PLAN AND I BOTH ASSERTED, and the measurement was already in the
  tree.** `benchmark/esquibel/FSM_COST.md` (task #77, 384,703 cells, 8 ranks) measures the split:
  GW solve 3.84 s/cycle (**~99.7%**), FSM 5.5e-03 s/cycle — **0.142% mean, 0.00007% at COLD START**,
  the case with the most water to route. The median cycle routes ~nothing and FSM early-exits in
  ~2.4 microseconds; PETSc `-log_view` independently puts `SNESSolve` at 92.6% of total.
  So "FSM is the serial ceiling" — asserted in the plan below (see its cheaper-variant section),
  in `FREE_SURFACE_FLICKER.md`, and by me on 2026-09-22 — is NOT SUPPORTED at measured scale.
  **Doubling 0.142% costs 0.142%.** `PORT_TO_UPSTREAM.md` had already drawn the conclusion for the
  related question: FSM parallelization is "DECIDED-park (memory-only driver, no speed case)" — the
  reason to parallelize FSM would be MEMORY, since rank 0 holds the full grid, not time.
  **TWO CAVEATS KEPT, neither measured.** (1) The fraction grows with RANK COUNT by Amdahl: FSM is
  fixed while the GW solve divides, and NA 30" runs on far more than 8 ranks. (2) The mean is pulled
  up by rare routing spikes of up to ~4 s, and iterating multiplies those spikes — so for THIS
  feature the TAIL matters more than the mean. From 0.142% there is real headroom before either
  bites, but neither is dismissed.

**The cheap measurement to take FIRST, and it needs none of this machinery.** The 2026-09-18 numbers
were taken from two raster series with NO CODE CHANGE. The cold-start lag — the one quantity the
plan says to get before judging the benefit, and now the one that sizes the default's cost — can be
measured the same way today. It answers "how many passes does a cold start actually need", which is
what the cap and the convergence tolerance both have to be set from.

---

PARKED 2026-09-18 by Andy, to be done AFTER the numbered list is finished. He named it correctly as a
wander from the mission. The plan is written out so nothing has to be re-derived.

=== WHAT AND WHY ===

Under `surface_water.routing: continuous`, FSM's per-cell volume change from step n feeds step n+1's
recharge source (WTM.cpp:346 -- "set the recharge for the NEXT step"). So the coupling is LAGGED by one
step. Andy: iterate instead, so a step uses its OWN runoff. "It is costly. But it is also exact."
Default to 1 pass (today's behaviour, byte-identical); the machinery is wanted for TESTING and for
BASIC CORRECTNESS, independent of the measured size of the lag.

MEASURED SIZE OF THE LAG (2026-09-18, corsica, from the two raster series, no code change):
    settled regime      1.58e-06 relative (median), max 4.32e-06
    restart transient   17.6% -> 5.5% -> 1.7% -> 0.7% ...  ~3x decay per step, <1e-4 by step 10
FSM's output stops moving because the lakes do (volume 4751.58 -> 4751.71 over 54 steps). A true COLD
START is unmeasured and is where the lag would be largest -- measure that before judging the benefit.

=== THE STATE A RE-SOLVE MUST RESTORE (enumerated from source, not assumed) ===

    params.elapsed_time_s                        1 scalar   (transient_groundwater.cpp:2466)
    dmdapack.starting_wtd, lake_stage, rech_vec  3 arrays   (:2321, :2336, and the recharge write)
    arp.total_*                                  NINE accumulators:
        boundary_inflow_gw, evap_removed, loss_to_ocean, loss_to_ocean_gw, ocean_outflow_gw,
        recharge_direct, runoff_to_surface, solver_recharge, surface_removed
    user_context.step.{dt,elapsed_from,elapsed_to} and user_context.deltat
    BDF2 / TR-BDF2 history vectors

NOT the two of task #41. #41 is right that the REJECT path needs only two -- because it returns ABOVE
the commit block, so most accumulation has not happened. A POST-COMMIT rollback is a much larger
surface, and it is exactly where #13, #14, #15, #17 and #41 all lived.

=== TWO STRUCTURES, AND THE RECOMMENDATION ===

A. REJECT BEFORE COMMIT. Run FSM on the trial solution inside update(), before the commit block, and
   signal a "coupling reject" reusing the existing non-commit path (without shrinking dt). Rollback is
   then nearly free and uses machinery five defect fixes have already proven.
   COST: pulls FSM -- a rank-0 SERIAL operation -- inside the distributed solver step. A permanent
   architectural inversion of the FSM/solver separation, for a feature that defaults OFF.

B. SNAPSHOT AND RESTORE AT THE WTM.cpp LEVEL  <-- RECOMMENDED
   Keep FSM where it is. Before the solve, save the 9 doubles + elapsed_time_s + the step record +
   Vec copies of starting_wtd and lake_stage; restore on iterate.
   COST: the rollback surface is ours to get right, and a missed item is a SILENT MASS ERROR.

B is recommended because it leaves the FSM/solver separation intact and the snapshot is mechanical and
auditable, where A changes the architecture permanently.

=== THE GUARD THAT MAKES B SAFE -- WRITE IT FIRST ===

A test that runs with iteration ENABLED but the source UNCHANGED must be BYTE-IDENTICAL to the
non-iterating run. Any state that fails to roll back breaks it immediately. Write it BEFORE the
feature, so the safety net exists before the machinery.

=== SCOPE ===

 1. the byte-identical guard (first)
 2. config key + resolve_defaults entry + declared-config entries in the affected suites
 3. snapshot/restore primitive at WTM.cpp level
 4. the iteration loop: convergence tolerance on the source + a hard cap on passes
 5. config.yaml documentation and CHANGELOG
Several granular commits. Default 1 pass, byte-identical, verified by (1).

=== A CHEAPER VARIANT WORTH COSTING WHEN THIS IS PICKED UP ===

Iterate UNTIL THE SOURCE STOPS MOVING rather than a fixed count. Given the 3x-per-step decay that is
one extra FSM pass during a transient and a single pass thereafter -- exact where it matters, free
where it does not. FSM is the serial ceiling, so a fixed k multiplies that ceiling by k.

=== NOT A FIX FOR THE CORSICA OSCILLATION ===

The coupling lag is EXCLUDED as its cause (row 7 of examples/island_equilibrium/OSCILLATION.md): the
source is constant to 1e-06 while the groundwater swings 24 m. This item is about correctness and
testability, not about that.
