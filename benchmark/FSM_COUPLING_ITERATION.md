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
