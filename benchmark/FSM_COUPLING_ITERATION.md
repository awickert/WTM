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
