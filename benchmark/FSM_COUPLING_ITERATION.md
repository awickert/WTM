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

**AMENDMENT 7 — IT RUNS (2026-09-23). `surface_water.coupling.iterations`, and what the first runs
measured.**

Andy named the key and chose accepted-pass counting. The loop lives in one place -- a `take_step`
lambda that all three step loops call -- so the iteration is written once rather than three times.

**THE DEFAULT IS STILL 1, AND THAT IS DELIBERATE FOR NOW.** Amendment 1 says iterating becomes the
method; flipping the default moves every equilibrium golden and re-labels every benchmark number in
`benchmark/`, so it is its own change with its own decision. What landed here is the machinery, with
the lagged scheme byte-identical: `golden` 35/35 unmoved.

**MEASURED on tests/fsm_cascade, n=1, iterations 1 / 2 / 3:**

| iterations | solver calls | steps (rejected) | final water table |
|---|---|---|---|
| 1 | 70  | 4 (0)  | reference |
| 2 | 184 | 15 (1) | max\|dwtd\| = **0.000e+00 m**, 0 of 900 cells moved |
| 3 | 212 | 7 (1)  | max\|dwtd\| = **0.000e+00 m**, 0 of 900 cells moved |

Read it the right way round. The passes DO run -- 70 to 184 to 212 solver calls, and a step pattern
that changes completely -- and the answer is nevertheless bit-identical. That is Amendment 2's
consequence, arriving as a measurement rather than an argument: at equilibrium `w_{n+1} = w_n`, so
`FSM(w_n)` and `FSM(w_{n+1})` are the same array and the lag is identically zero. **Same fixed
point, three different paths.** It is also Consequence 3's free correctness test, passed at the
strongest possible value: iterating moved a converged equilibrium answer by exactly nothing.

Mass is conserved on all three. The EXACT budget residual -- the conservation check -- is
-1.29762625e+05, -4.8019875e+04 and -1.246915e+05 against 2.83595444906e+15 of recharge, i.e. ~1e-11
relative. The PHYSICAL residual (col 16, `ocean_loss_closing - ocean_outflow - loss_to_ocean`) does
move, 1.44e10 to 2.38e13 to -7.52e12; that column is documented as carrying the BDF2-startup gap and
is a path integral, so three different step patterns give three different values. Recorded rather
than waved away, because it is the one number here that a reader might mistake for a leak.

**WHAT THE FIRST RUN FOUND, and it was my bug.** `iterations: 2` aborted with "adaptive dt: step
failed after max retries" at every dt the controller tried -- because it never tried a different
one. A rejected later pass restored the FULL scalar snapshot, which put back the pre-step `deltat`
and undid the shrink `update()` had just applied for the retry. The reject path now preserves the
shrunken step size and restores everything else. A rollback that is too COMPLETE is a failure mode
too, and this design had only ever worried about the opposite one.

**THE RUNTIME CHECK FIRES, and says what Amendment 5 predicted.** Four notices per run:
`tr_exfil_stage1`, `tr_fwork`, `tr_head_old` and `vol_prev_x` were CREATED during the step, so there
is nothing to restore into them. All four are scratch -- each fully overwritten before it is read --
so this is a notice, not a defect. It is printed once per message per run rather than per step,
which is a readability choice with a cost: a second distinct cause on the same Vec would hide behind
the first.

**A THIRD CATEGORY THE ROLLBACK DOES NOT COVER, and the runtime check cannot see it.** The snapshot
carries PETSc Vecs and scalars. It carries NO rank-0 `ArrayPack` arrays, and the coupling writes five:
`arp.wtd`, `arp.runoff`, `arp.rech`, `arp.wtd_mid` and `arp.runoff_nominal` (the rest of what it
touches -- `cell_area`, `topo`, `precip`, `porosity`, `evap`, `land_mask`, `runoff_ratio`,
`open_water_evap` -- are inputs it only reads).

The argument that this is safe is that each is REDERIVED from state that IS restored: `arp.wtd` is
re-gathered from `starting_wtd`, `arp.wtd_mid` is rewritten as the pre-FSM table, `arp.runoff` is
zeroed and re-armed every coupling, `arp.rech` is recomputed from precip/evap, and
`arp.runoff_nominal` is set once at cycle 0. **That argument is reasoning, not measurement, and this
file's whole history says which of those to trust.** The evidence that nothing is leaking is
indirect: iterations 1/2/3 agree bit-for-bit and the exact budget residual stays at ~1e-11 relative,
which a gross leak would disturb. Indirect is not the same as checked.

So it is recorded as an OPEN GAP rather than closed by argument. The fix is the same shape that
already works for the Vecs -- fingerprint the five before the step and compare after -- and it is
cheap, since these are rank-0 arrays and the comparison is serial. Until then, the runtime check's
scope is exactly "PETSc Vecs", and it must not be read as "the step's state".

**REFUSALS, all four verified by running them:** `iterations: 0`; `> 1` with `routing: impulse`;
`> 1` with `routing: off`; and the typo `iteration` caught by the schema with a did-you-mean.

**40 suite configs migrated** to declare `iterations: 1`, because `full_config.yaml` now prints the
key for every run and the declared-config rule (#83) makes an undeclared resolved key a failure.
Same blast radius #109 had, and behaviour-preserving by construction.

**AMENDMENT 8 — the rank-0 gap CLOSED, and the check I built for it was worse than useless.**

Amendment 7 left five rank-0 `ArrayPack` arrays outside the rollback -- `arp.wtd`, `wtd_mid`,
`runoff`, `rech`, `runoff_nominal` -- safe by the ARGUMENT that each is rederived from state that is
restored. Closing it produced three results, and the middle one is the uncomfortable one.

**1. THE FINGERPRINT CHECK WAS BUILT, RAN, AND WAS REMOVED.** Comparing each array before and after
the step reported `arp.wtd` and `arp.wtd_mid` on every single iterated run, and stayed SILENT on
`arp.runoff`. Both reports were false alarms; the silence was the real defect. A step naturally
changes the water table, so "it moved" is not evidence of anything -- and a step can leave
`arp.runoff` with the same sum, sum of squares and max as it started with while its CONTENTS matter.
Two false alarms per run plus a miss on the one that counted is worse than no check at all: it
trains the reader to scroll past the line that matters. Deleted rather than tuned.

**2. THE QUESTION IS DECIDABLE, AND POISONING DECIDES IT.** The property that matters is not "did it
move" but "is it REDERIVED BEFORE IT IS READ". Fill the array with NaN immediately after the
rollback and compare the run against a clean one: if every read is preceded by a write, the answer
is untouched; if not, the NaN propagates. Five runs on `tests/coupling_iteration`, k=3:

| poisoned after the rollback | max\|Δwtd\| vs clean | verdict |
|---|---|---|
| `arp.wtd` | 0.000e+00 | rederived before read |
| `arp.wtd_mid` | 0.000e+00 | rederived before read |
| `arp.rech` | 0.000e+00 | rederived before read |
| `arp.runoff_nominal` | 0.000e+00 | rederived before read |
| **`arp.runoff`** | **4.000e+00 m** | **READ FIRST -- it is step state** |

4.000 m is the entire depth of the fixture's lake, so this is not a marginal reading.

**3. THE MECHANISM, and it is an ordering.** The coupling ACCUMULATES into `arp.runoff`
(`arp.runoff(i,j) += routed`, in the exfiltration gather) BEFORE it zeroes the array and re-arms it
for the next step. Its pre-step contents are therefore read, and a second pass that inherits the
first pass's re-armed value is reading the NEXT step's carrier instead of this one's. It is now
restored; the other four are not, and that exemption is earned by the table above rather than by the
paragraph that used to stand in its place.

**WHAT THE FIX CHANGES ON THIS FIXTURE: NOTHING, 0.000e+00 m** -- and the reason is the same
attractor that blunts INVARIANT. At equilibrium the pre-step and post-pass-1 runoff carriers hold
the same steady value, so restoring makes no difference; in a transient they differ. The fix is
correct and necessary, and its effect is below the resolution of the only fixture that currently
exercises it. Said plainly rather than dressed up as a visible repair.

**KEPT FOR NEXT TIME:** the poison experiment is the instrument that settles "rederived before read"
for any rank-0 array, and it is five runs. Re-run it before trusting the four-way exemption against
changed code.

**AMENDMENT 9 — THE ITERATION DOES NOT CONVERGE. It oscillates, on every topography tried.**

Andy decided (2026-09-23) that iterating becomes the default, and asked the right question of the
plan for a convergence-based stop: *"by 'bitwise', do you really mean that no value changes? It would
take a long time for an iterator to get there."* Measuring the answer produced something neither of
us expected, and it bears on the default itself rather than on the stopping rule.

**WHAT WAS MEASURED.** A temporary probe recorded, for every pass of every step,
`dmax = max|F(w^k) - F(w^{k-1})|`, the source's own magnitude, the number of cells that moved, and
whether `F^k == F^{k-2}` bitwise. Cap of 8 passes, on `coupling_iteration`, `fsm_cascade` and
`fsm_fullness` -- three independent topographies (single pit; a spill CHAIN; heterogeneous fullness).

**RESULT: `dmax` IS FLAT.** On every step that routes water, the per-pass change stops decreasing
after pass 2 and then holds constant to three significant figures for the remaining six passes:

    fsm_cascade  step 6    2.50e+00 2.50e+00 2.50e+00 2.50e+00 2.50e+00 2.50e+00 2.50e+00
    fsm_cascade  step 9    1.38e+00 1.38e+00 1.38e+00 1.38e+00 1.38e+00 1.38e+00 1.38e+00
    fsm_fullness step 4    3.86e+00 3.86e+00 3.86e+00 3.86e+00 3.86e+00 3.86e+00 3.86e+00
    coupling_it. step 2    1.47e-01 1.47e-01 1.47e-01 1.47e-01 1.47e-01 1.47e-01 1.47e-01

and the source's own magnitude ALTERNATES between a large value and ~0 -- for `coupling_iteration`
step 2: `1.47e-01, 1.65e-05, 1.47e-01, 0.00e+00, 1.47e-01, 0.00e+00, 1.47e-01`, with the moved-cell
count flipping 184/168. **That is a period-2 orbit**, entered immediately and never left.

Counting only steps with a REAL source -- the majority of steps route no surface water at all, and
those say nothing about convergence -- roughly a QUARTER settle and three quarters oscillate.

**IT IS THE OPERATOR, NOT THE ROLLBACK, and that was checked rather than assumed.** This is the
design's own byte-identical guard, finally run: restore the source TOO, so every pass re-solves the
identical problem, and a k=8 run must reproduce k=1 bit for bit. It does, on all three fixtures --
`max|Δwtd| = 0.000e+00 m, 0 cells moved`. The rollback is complete; the oscillation is `Φ`.

**WHAT THIS MEANS FOR THE DEFAULT, stated plainly because it cuts against the decision just taken.**
There is no converged answer for the iteration to reach on these cases. `iterations: k` returns
whichever state the orbit is in at pass k, so **the answer depends on the PARITY of the cap**, and
the two states are far apart (a source of metres versus ~zero). A default that iterates would make
the model's answer a function of a tuning knob's parity, which is worse than the lagged scheme --
lagged is at least a single definite rule.

**AND NO STOPPING RULE RESCUES IT.** A tolerance cannot fire on a sequence that does not decay.
Bitwise convergence cannot either. Exact period-2 detection fired on only 3 of 16 moving steps, and
two of those three were false positives of my own detector: `F^k == F^{k-2}` is trivially true when
`F` is CONSTANT, so a real detector has to require `F^k == F^{k-2}` AND `F^k != F^{k-1}`.

**THIS IS PROBABLY NOT NEW.** `FREE_SURFACE_FLICKER.md` describes non-contraction of this same outer
operator, and task #111's corsica oscillator is measured, real and unexplained. This measurement
gives that phenomenon a clean, cheap reproduction: three small fixtures, eight passes, no at-scale
run required.

**OPTIONS, with their costs -- none of them chosen here.**
  1. **Keep the iteration opt-in** and do not flip the default. Costs nothing already built; the
     machinery has just proved its worth as an INSTRUMENT, which is what the original instruction
     asked for ("helpful both for testing and for basic correctness").
  2. **Damp the outer loop** -- under-relaxation, or Anderson on the outer iterate. The design
     explicitly has neither. It could turn the orbit into convergence, and it introduces a
     parameter that would need deriving.
  3. **Define the answer on the orbit** -- average the two states, or keep the one with the smaller
     source. Needs a definition of "better" that is physical rather than convenient.
  4. **Chase the mechanism first**, since it is very likely #111's. That is a research question, not
     a config change.

**AMENDMENT 10 — WHY IT OSCILLATES: THE DELTA IS MEASURED AGAINST A MOVING BASELINE.**

Andy: *"Analyze the source of the orbit... We will act with information rather than hopeful
guesswork."* Here is the mechanism, derived, and then confirmed by three predictions that were made
before they were tested.

**THE DERIVATION, in four lines.**

1. `fsm_delta = V(post-FSM) - V(pre-FSM)`, and **pre-FSM is the CURRENT pass's solve output**
   (WTM.cpp, `wm = arp.wtd_mid`), not the step's starting state.
2. Pass k+1 receives `F_k` as a source, so its solve output ALREADY contains that water:
   `V(pre-FSM_{k+1}) = V(pre-FSM_1) + F_k`.
3. FillSpillMerge then levels to the same place regardless -- **measured: the post-FSM table is
   INVARIANT across the orbit** (fsm_cascade, both halves: `post = 2034.0000` while `pre` alternates
   1804.416 / 1270.506).
4. Therefore `F_{k+1} = V(sill) - V(pre-FSM_1) - F_k`, i.e.

        F_{k+1} = C - F_k      -- a linear map with multiplier EXACTLY -1.

Multiplier -1 is marginally unstable: no decay, no growth, period 2 forever, and
`|F_{k+1} - F_k| = C` constant. That is why `dmax` held to seven figures instead of drifting.

**THREE PREDICTIONS, EACH TESTED.**

  - `F_k + F_{k+1}` must be CONSTANT within a step. Measured on fsm_cascade: **818.6077, 818.6076,
    818.6077, 818.6076...** on one step and **2132.968, 2132.968** on another.
  - Under-relaxation with weight w gives multiplier `(1 - 2w)`, so **w = 0.5 must converge in ONE
    pass**, to the midpoint `C/2`. Measured: `405.7527, 409.3013, 409.3032, 409.3036, 409.3038,
    409.3039, 409.3039` against the undamped `405.77 / 412.84` alternation -- and 409.3039 is the
    midpoint of those two.
  - Freezing the BASELINE at the step's starting state must remove the oscillation outright.
    Measured, on both fixtures: fsm_cascade `1475.2060, 1559.0260, 1559.0260, 1559.0260...` and
    coupling_iteration `0.0000` from the first pass on.

**WHAT THIS MEANS, and neither escape is a fix.**

*Damping is not a fix.* It converges, to `C/2` -- the state where the water is credited HALF to the
source channel and half to FillSpillMerge's own levelling. That is a well-defined fixed point with no
physical claim to being right.

*The fixed baseline is not a fix either*, and this is the trap to avoid: `V(post-FSM) - V(w_n)`
conflates FSM's redistribution with **the groundwater solve's own change over the step**. On
fsm_cascade it is ~1559 against the moving baseline's ~409, and the ~1150 difference IS the solve's
contribution, which would then be fed back as a source on top of the solve that already produced it.
It is a diagnostic that localises the -1 to the baseline, not a formulation to adopt.

**THE UNDERLYING PROBLEM, stated once and plainly.** Within a single step, "the source term
delivered this water" and "FillSpillMerge delivered this water" are THE SAME EVENT. The lagged
scheme keeps them distinguishable by separating them in TIME -- FSM acts at step n, its delta is a
source at step n+1, and each is counted once. Iterating within the step collapses that separation,
and the map's only way to express "count it once" is to alternate which channel gets the credit.
**The lag is not an error term the iteration can remove; it is what makes the two channels
distinguishable.**

**SO THE DIRECTION, IF THE ITERATION IS STILL WANTED**, is not a stopping rule and not damping: it is
to reformulate the delta as something the solve does not itself duplicate -- FSM's redistribution as
a lateral FLUX applied during the step, rather than a volume jump measured after it. That is a
scheme change, not a config change, and it is Andy's call.

**THE INSTRUMENTS, worth keeping since they were each decisive:** the `WHY` probe (pre- and post-FSM
surface water plus the signed delta, per pass), the under-relaxation switch, and the frozen baseline.
All three were temporary and are reverted; the method is recorded here so it need not be re-derived.

**AMENDMENT 11 — THE SEPARATE ARRAY IS THE RIGHT SHAPE, and three wrong versions of it were
measured out of the way first.**

Andy, on the flux idea: *"How do we apply a lateral flux? How do we do it during the step before we
have run FSM? The continuous application seems more principled."* That objection is correct and the
flux suggestion is withdrawn -- an a-priori flux estimate would be a worse approximation than the
thing it replaces. And: *"Is the easier answer to have a separate array so we can see the FSM
contribution independently?"* Yes. Here is what the measurements say it has to hold.

**THREE CANDIDATE FIXES, EACH TESTED AND EACH REFUTED.** Recorded because each looked right.

| candidate | oscillation | what it actually did |
|---|---|---|
| under-relax at w = 0.5 | removed | converges to `C/2` -- the water credited HALF to the source and half to FSM. Well defined, no physical claim. |
| freeze the baseline at the step's start | removed | subtracts the GROUNDWATER SOLVE'S own change too (~1559 vs ~409; the ~1150 gap is the solve). Converges to the wrong quantity. |
| ACCUMULATE the carrier within the step | removed on 69 of 70 steps | **banks the same redistribution once per pass.** Carrier magnitude measured at 1.3427e+03 / 2.6855e+03 / 5.3710e+03 for k = 2 / 4 / 8 -- exactly linear in k. |

**HOW THE THIRD ONE WAS CAUGHT is the part worth keeping.** Under accumulate the water budget CLOSED
(exact residual 3.27e-13 on fsm_cascade, 273x BETTER than assign), total recharge was unchanged to
0.000e+00 relative, stored volume was unchanged, and the final water table was BIT-IDENTICAL. Every
check that usually catches a mass error said fine. The only column that moved was LOSS TO OCEAN:
+278% on coupling_iteration and +545% on fsm_cascade. The phantom water enters as source and leaves
to the sea, so the exact residual -- which is defined on the FULL source -- moves both sides together
and closes by construction. **A closing budget is not evidence against double counting when the
double count is in the source term.**

Two explanations for that inflation were proposed and both REFUTED by measurement rather than
argument: it is not `arp.runoff` being re-delivered (this fixture runs `runoff_ratio: 0`, so that
carrier is re-armed to zero -- forcing it to zero changed nothing), and the carrier is not growing
ACROSS steps (measured constant at 5.3710e+03 at every step start). The real cause is that each pass
re-solves the step with the SAME recharge, that water reaches the surface again, and FSM routes it
again -- so the increment never falls to zero and accumulation counts it k times.

**SO THE TWO REQUIREMENTS COLLIDE, and that is the whole problem in one line.** ASSIGN gives the
right magnitude but measures against a moving baseline, which is the -1 multiplier. ACCUMULATE fixes
the baseline but counts the redistribution once per pass.

**WHAT THE SEPARATE ARRAY MUST HOLD.** To measure FSM's delta correctly you need to know HOW MUCH OF
THE CURRENT STATE CAME FROM THE FSM CARRIER, so it can be excluded from the baseline:

    d = V(post-FSM) - V(w_solve - S)     instead of    d = V(post-FSM) - V(w_solve)

and then ASSIGN as today, keeping the self-clearing property across steps. That `d` is the step's
true total redistribution `T`, independent of the pass, so the iteration converges at pass 2 with no
accumulation and no double count. The frozen-baseline experiment was a crude version of exactly this
-- it subtracted the whole step's change rather than only the source's share.

**THE OPEN DIFFICULTY, stated rather than glossed.** The solve is NONLINEAR: the state change the
source produced is not exactly `S`. Some of it drains laterally, some evaporates, some leaves to the
ocean. `w_solve - S` is therefore an approximation, and how to attribute the source's share of the
state properly is a MODELLING decision, not a coding one. That is the question to answer before
this is built.

**AMENDMENT 12 — ANDY WAS RIGHT: THE FSM INPUTS DO CONVERGE OVER TIME, AND AMENDMENT 9 OVERSTATED.**

Andy: *"I expect that we will see the FSM inputs converge over time. This is the point of a separate
array here. We will not be able to separate out the FSM input effects easily. So we can use overall
convergence here."* Both halves check out, and the first one corrects this document.

**THE OSCILLATION IS CONFINED TO THE FILLING TRANSIENT.** Amendment 9 said "the iteration does not
converge" and generalised a PER-STEP behaviour into a property of the scheme. Laid out per step, with
`O` = still oscillating at pass 8, `c` = settled, `.` = FSM idle:

    fsm_cascade          OOOOOOOOOOc........................   (70 steps)
    coupling_iteration   OO.................................   (126 steps)

Ten and two oscillating steps respectively, ALL at the start, and ZERO in the second half of either
run. Once the lakes reach their sills the coupling is inert. The orbit is a property of FILLING, not
of the scheme -- which is exactly what "the FSM inputs converge over time" predicts.

**AND THE ORBIT NEVER REACHES THE ANSWER.** Under the shipped `assign` scheme, every reported
quantity is pass-count-invariant across k = 1, 2, 3, 4 on coupling_iteration:

    recharge        relative spread  0.000e+00
    stored volume   relative spread  0.000e+00
    evaporation     relative spread  0.000e+00
    loss to ocean   relative spread  1.952e-04      (0.02%)
    exact residual  ~1e-13 relative at every k

So the decision Andy took -- iterating by default -- is SAFE on this evidence. The honest other half:
these fixtures cannot DEMONSTRATE the benefit either, because they equilibrate inside one report
interval, where the lag is definitionally zero. The cold-start measurement remains the one that would.

**THE OUTER STOP, BUILT AS ANDY SPECIFIED: overall convergence, no decomposition.** Separating the
FSM contribution from the solve's own share turned out to need an attribution the nonlinear solve
does not give (Amendment 11), so the stop does not attempt it. It compares successive passes on the
WHOLE state instead:

  - QUANTITY: the L1 water-volume change between passes -- summed per cell, NOT the change in the
    domain total. The iteration REDISTRIBUTES water, so a pass can move a great deal while the total
    is unchanged; a scalar total would call that converged.
  - TOLERANCE: `solver.convergence.water_volume_tol`, THE INNER SOLVE'S OWN BAR, reused rather than
    invented. The outer iteration cannot resolve a change smaller than the solve producing each pass.
    No new config key.
  - CAP: `surface_water.coupling.iterations`, reached only where the coupling is still moving.

**MEASURED COST, and this is what makes iterating affordable as a default** (coupling_iteration,
solver calls):

    cap = 1     131      the lagged scheme
    cap = 2     252
    cap = 4     262      (was 393 before the stop)
    cap = 8     258      (would have been ~1000)

Raising the cap from 2 to 8 costs 6 solver calls, because the stop fires as soon as the state stops
moving. The cost of iterating is ~2x the lagged scheme, NOT kx. `golden` 35/35 unmoved at the default.

**ONE DEFECT THIS INTRODUCED AND FIXED.** With an early exit, "the accepted pass" is no longer known
before the coupling runs, and the per-step bookkeeping was keyed on `pass == passes`. It silently
stopped firing: the run-log solve count collapsed from 131 to **1** at caps 4 and 8. The budget trace
is therefore SPLIT into a sample (taken every pass at the same point as before, post-solve and
pre-coupling) and an emit (once, after the loop). At `iterations: 1` it is arithmetically identical
to the single call it replaces -- same sample point, same order, and the running previous-values
advance exactly once per step.

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
