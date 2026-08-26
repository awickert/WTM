# Design Note: higher-order + adaptive time stepping for transient accuracy

**Date:** 2026-07-25
**Branch:** `bdf2-adaptive-dt` (off `picard-mg` — builds on the semi-implicit Picard solve)
**Status:** DESIGN / prototype sketch. For a bounded experiment, not a commitment.
**Authors:** Andy Wickert + Claude
**Companions:** `PICARD_MG_DESIGN.md`, `PICARD_MATH.md` (the implicit solver this builds on).

---

## TL;DR

- **Goal:** cheap 2nd-order-in-time accuracy for *transient* WTM runs (equilibrium is already
  Δt-independent after the Picard work, so it only wants the biggest stable step).
- **BDF2 alone did not deliver order 2** — the measured temporal order was **~1**. We chased
  this to ground:
  - *Not* the C0 transmissivity kinks at −1.5 m / 0 m (smoothing T — **disproven**).
  - *Not* the storativity surface corner at 0 m (smoothing V — **disproven**).
  - **Cause: the backward-Euler *secant* effective storativity** — a 2-level construction
    `S_eff = (V(hⁿ⁺¹)−V(hⁿ))/Δh` sitting under BDF2's 3-level time derivative. The
    mismatch drops the achieved order to 1. Forcing constant S restored order 2, which
    isolated the storativity as the culprit.
- **Fix — BDF2-on-V:** apply the 3-level BDF2 difference to the *volume* `V(h)` directly
  (`(3Vⁿ⁺¹−4Vⁿ+Vⁿ⁻¹)/(2Δt) = flux`), Picard-linearized with the **tangent** `dV/dh`
  (`specificYield`) as the operator diagonal. Physics-preserving (no fixed-point shift).
  **2nd order in time for the HOMOGENEOUS (relaxation/drainage) problem — order ≈ 2.0 across
  Δt = 1–100 yr** once the startup transient is smooth (§2 addendum). (An earlier apparent fine-Δt
  order→1 was a cold-start O(Δt) artifact from the fixture's singular IC, not the scheme — ruled
  out solver tolerance, coefficient kinks, and float32 output; a smooth resolved start restores
  clean order 2.) **BUT with recharge active — i.e. every production run — it drops to 1st order:**
  the recharge *source* is integrated at only 1st order (diagnosed 2026-07-27 — not scaling, not
  kinks; see `BDF2_RECHARGE_ORDER.md`). Still ~2× more accurate than backward Euler at the same
  step and unconditionally stable, but not 2nd-order in production. Restoring 2nd order needs a
  source fix (being instrumented) or Richardson-in-time.
- **Adaptive Δt (Picard/BDF2, forward) — shelved (kept in code, default off).** The forward per-step
  history-extrapolation estimator over-refines: one fast near-ocean cell pins the step, so at matched
  accuracy it ran ~2× *more* steps than a well-chosen uniform Δt. For the Picard/BDF2 path the uniform
  BDF2-on-V step is the recommendation; the forward controller stays available behind `-wtm_dt_adaptive`.
- **Adaptive Δt for TR-BDF2 — IMPLEMENTED (2026-08-15, `-wtm_tr_bdf2 -wtm_dt_adaptive`, default off).**
  This is the "fundamentally different error estimator" §2/§6 called for, and it resolves the near-ocean
  pinning that shelved the forward version. Two coupled mechanisms (Andy's framing: (2) principled accuracy
  governor on top of (1) the necessary convergence floor):
  1. **Reject/retry feasibility floor.** A non-converged TR-BDF2 stage or step returns the `-1` reject
     sentinel (shrink `deltat` via `dtc_shrink`, retry from the *uncommitted* state); the `use_dt_adaptive`
     loop in `WTM.cpp` rolls back the step's recharge/ocean accumulators and retries, capped by
     `dtc_max_retries`. Reuses the dt-continuation plumbing + the recharge-rescale-to-actual-Δt fix.
  2. **Embedded error estimator from the two stages** (no history, valid on step 1): the linear
     extrapolation through (tₙ, hⁿ) and (tₙ+γΔt, Y_γ) to tₙ+Δt is `h_pred = [Y_γ − (1−γ)hⁿ]/γ` — EXACT for
     linear-in-time, `O(Δt²)` for curvature. **Measured in WATER (volume) (2026-08-24):**
     `est = |storedVolume(wtdⁿ⁺¹) − storedVolume(wtd_pred)|` = the local truncation error expressed as water
     moved `|S·Δwtd|` (slope 1 above the surface, porosity below; reuses the storage `V(wtd)`), with no ground
     truth needed. This puts `est`/`dt_tol` in the SAME units as `eq_tol`, so the per-step accuracy and the
     equilibrium stop are directly comparable — a time-marching scheme cannot resolve a steady state finer than
     its own per-step error (symmetry through convergence). `est > dt_tol` ⇒ accuracy reject (shrink, retry);
     else accept and grow toward `dt_tol`, capped by convergence headroom (`dtc_easy_iters`) and `dtc_dt_max`.
  - **Surface-inclusive norm (corrected 2026-08-15).** The error norm covers ALL land cells INCLUDING the
    free surface. An earlier version EXCLUDED surface cells (`wtd ≥ −band`) to keep the clamp's
    non-smoothness from spiking the estimate — but **stability is SET at the free surface**, so excluding it
    blinded the controller to the surface overshoot and it grew Δt into a limit cycle that never settled
    (island cold start: rang forever at 14843 iters excluded vs a monotone settle in 1547 including it). A
    *settled* clamped cell has `h_pred ≈ hⁿ⁺¹` ⇒ deviation ≈ 0, so inclusion costs nothing on a warm
    transient (byte-identical iteration counts to the excluded norm at `dt_tol` 0.5/5/20); only a
    *transitioning / ringing* surface cell spikes, which correctly shrinks Δt. **RMS norm (default)** or MAX
    (`-wtm_dt_norm_max`). RMS is the default because, under the water (volume) step-error, the MAX worst-cell
    norm is hostage to a handful of surface-kink cells (the `storedVolume` slope changes 1→φ at `wtd=0`) whose
    spike is **dt-INDEPENDENT** — on a cold start MAX shrinks Δt to the floor and aborts (max retries) at any
    `dt_tol`; RMS averages those cells over the domain and cold-starts robustly. See GH #13.
  - **Validated** (Esquibel, dry −20 %, `tests/run_all.sh` green): `-wtm_dt_tol 1/5/20 m` → 33/12/8 steps
    (6/3/1 rejected), all converged — monotone in the tolerance, both mechanisms live. This is the general
    stability route: the controller keeps Δt as large as the *local* terrain/conditions allow and backs off
    automatically where they don't. See the implementation in `transient_groundwater.cpp` (estimate +
    controller split, reject returns), `WTM.cpp` (reject/retry loop), and `CreateSNES.cpp` (flag wiring).
  - **Detached from the integrator (2026-08-15).** The controller no longer forces TR-BDF2 (or the BDF2
    residual): the *estimate* is the only method-specific piece — TR-BDF2 uses its embedded two-stage
    estimate; every other integrator uses the generic linear-history predictor `h_pred = hⁿ + ω(hⁿ − hⁿ⁻¹)`
    (needs the last two accepted states) — and both feed one **method-agnostic controller**
    (grow/shrink/reject). So `-wtm_dt_adaptive` now composes with any integrator:
    `-wtm_anderson` → 1st-order backward-Euler (cc, ring-proof), `-wtm_tr_bdf2` → 2nd-order TR-BDF2,
    `-wtm_bdf2_on_V` → 2nd-order BDF2-on-V. History (`wⁿ⁻¹`) is tracked whenever adaptive is on. Island
    cold→eq, all settle: cc 13459 its (1st-order, robust fallback), TR-BDF2 1547, BDF2-on-V 7776.
  - **PI controller + one-knob `eq_tol` coupling (2026-08-15).** The plain I-controller (hard reject on any
    overshoot, grow on any undershoot) HUNTS near the tolerance and locks into Δt limit cycles at certain
    `dt_tol` — a measured resonance dead-band at ~0.2–0.25 m that never settles (84–99 cycles), flanked by
    fine values at 0.15 and 0.3. Fix (standard stiff-ODE practice): a **PI step-size controller** that damps
    the oscillation with the previous accepted error, and a PI-damped reject shrink instead of a hard slam —
    the 0.2–0.25 band drops to 11–21 cycles, no catastrophe anywhere in the 0.1–0.5 operating range. Cost:
    the nominal point is ~28 % slower than the (hunting-prone) I-controller (island 1547 → 1977) — robustness
    over speed, still ~2× better than fixed-1-wk cc.
  - **`dt_tol` ↔ `eq_tol`: coupled → decoupled → unit-matched (2026-08-24).** The original one-knob coupling
    `dt_tol = min(k·eq_tol, ring_cap)` (k = 50, ring_cap = 0.5 m) placed the step tol at the sweet spot for the
    default `eq_tol` and made `eq_tol` the single knob. It was then **decoupled** (5a71e5c) because `eq_tol`
    became a WATER depth while `dt_tol` was a HEAD error — deriving one from the other mixed units. The
    volume-norm above **removes that mismatch**: `dt_tol` is now water too, so the two are directly comparable
    and the default re-tracks `eq_tol` — this time unit-correctly and at **k = 1**: on an equilibrium run
    `dt_tol = min(eq_tol, ring_cap)` unless `-wtm_dt_tol` is set (integrate to the accuracy you detect; the
    ring cap still keeps Δt below the free-surface overshoot for a loose `eq_tol`). The coherence is required,
    not cosmetic: a looser step tol than `eq_tol` cannot converge (the run jitters at the step-error amplitude
    — the adaptive_water failure that surfaced this). An explicit `-wtm_dt_tol` looser than `eq_tol` on an
    equilibrium run now warns. **A transient run has no convergence target**, so `dt_tol` is a pure accuracy
    knob (0.1 m water default, or `-wtm_dt_tol`) — independent of `eq_tol`, which is not a stop criterion there.
    The config key for this tolerance is `solver.water_volume_timestep_error_tol` (→ `-wtm_dt_tol`).
  - **Operational home + later work (single source of truth): `benchmark/adaptive_dt/`** (README + tests).
    It carries the MSI benchmark verdict (adaptive = **robustness / spin-up tool**: ties well-chosen
    constant dt on smooth transients, decisively *wins* spin-up — bounded worst-cell error where fixed dt
    blows up); the **equilibrium auto-stop metric** (`-wtm_eq_metric`, default `frac` = "<0.1 % of cells
    still exceed `eq_tol`"; the MAX-metric "deep oscillation" was diagnosed a *metric artifact*, not physics —
    the bulk converges monotonically); and **Tbar as an opt-in stiff hammer** (composes, not auto-engaged).
- **Bottom line:** for transient accuracy, use BDF2-on-V at a fixed, generous Δt (see §1–§3
  for the measured order and the max-Δt-for-a-target-error table). The smoothing knobs
  (`-wtm_ksat_soilbottom_smoothing_width`, `-wtm_ksat_surface_smoothing_width`,
  `-wtm_storativity_surface_smoothing_width`) remain as physical sub-grid options but are *not*
  the accuracy lever.

*(Findings and order-verification narrative below are kept verbatim as the record of how this
was established. Note: the smoothing flags were later renamed to a `{quantity}_{location}` scheme.
The narrative's `-wtm_smooth_T` / `-wtm_smooth_eps` (transmissivity, both boundaries) are now the
per-boundary `-wtm_ksat_soilbottom_smoothing_width` (−1.5 m) and `-wtm_ksat_surface_smoothing_width`
(0 m), each width 0 = off; `-wtm_storativity_eps` is now `-wtm_storativity_surface_smoothing_width`.)*

---

Semi-implicit Picard made the backward-Euler step **unconditionally stable**, which
already reframes WTM's two run types:

- **Equilibrium** — we only want the steady endpoint, and the steady state is
  *Δt-independent* (measured: Picard reaches the same field in 4 steps at Δt=10⁵yr as
  Anderson does in ~40,000 steps at Δt=1yr, agreeing to ~mm). So the path is irrelevant;
  take the biggest stable step. **Already blazing** via the Picard work; the only lever
  left is automating the step growth (pseudo-transient continuation, §4).
- **Transient** — the *path through time* is the answer, so it needs temporal **accuracy**,
  not just stability. This is the target of this branch.

The distinction is now numerical, not just a config flag: equilibrium is a root-find
(drive residual → 0, any stable path), transient is an initial-value problem (resolve the
trajectory). This note designs the transient side: **2nd-order time (BDF2)** for cheap
accuracy, and **adaptive Δt** to control it — together.

---

## 1. The measured problem: backward Euler is first-order in time

On the 128² pure-drainage fixture, comparing the water-table field at *matched physical
times* against an Anderson Δt=1yr ground truth (mean |deviation|, m; field range 100 m):

| T (yr) | Picard Δt=10 | Δt=100 | Δt=1000 | Δt=10000 |
|---|---|---|---|---|
| 1,000  | 0.077 | 0.83 | 6.8 (1 step) | — |
| 10,000 | 0.016 | 0.18 | 1.76 | 15.1 (1 step) |
| 40,000 (≈eq) | 1.2e-4 | 1.3e-3 | 0.017 | 0.69 (4 steps) |

Two facts drive the design: **(a) the transient error is first-order in Δt** (10× Δt →
~10× error — clean backward-Euler O(Δt)); **(b) it washes out as T → equilibrium** (the
distortion is in the path, not the destination — consistent with the Δt-independent steady
state). So for transients we want higher temporal order, and a way to size Δt to a target
path accuracy.

---

## 1.5 WTM's timeline (Kerry's production structure) — resolves the "what is a step" question

Kerry's production cadence pins the structure down:

    deltat = 1 day    maxiter = 7    total_cycles = 4    ->  run = 4 weeks,  FSM once per week

- **A time step = `deltat` = 1 day** — one GW backward-Euler step. Each `maxiter`
  iteration *advances one day* (arithmetic: 4 cycles x 7 x 1 day = 28 days; confirmed a
  time sub-step, not a convergence re-solve).
- **`maxiter` = daily GW sub-steps between FSM applications** (7 = one week).
- **A cycle = the FSM interval = 1 week.**
- **GW <-> FSM is operator splitting:** GW diffuses daily; FSM routes surface water weekly.

Note the name `maxiter` is a fossil: it originates from the **equilibrium** solutions, where
each pass genuinely *iterates* toward steady state. On a **transient** run the same loop
*advances time* (one day per pass), so "iteration" is a misnomer there -- the root of much
confusion about what a "step" is. The two regimes this note formalizes (PTC for equilibrium,
BDF2+adaptive for transient) finally give that overloaded loop distinct meaning per run type.

**Consequences for BDF2:** history `h^n, h^{n-1}` is uniform *within* a week (clean BDF2);
the weekly FSM kick perturbs `h`, so the history is invalid across a cycle boundary ->
**bootstrap one backward-Euler step at the first day of each week** (1 BE + 6 BDF2 / week).

**Why the step is daily (resolved with Andy):** it is a **stability workaround**, not a
science requirement, for the common case — Anderson diverges at large dt (we measured a
ceiling ~1 yr on the drainage test; the daily production step is the same phenomenon at
production stiffness). The forcing is weekly and the *science of interest is changes from
seasonal to 100s–1000s of years*, so a daily step over-resolves by 2–5 orders of magnitude
purely to keep Anderson stable. The rare cases that genuinely need daily resolution can keep
using Anderson. **So the transient win is real:** Picard's unconditional stability lets us
step at the *timescale of interest* (weeks → years → decades) instead of daily — orders of
magnitude fewer GW solves — and BDF2 keeps the coarser steps accurate.

**The new cap this exposes — FSM cadence.** A GW step cannot exceed the FSM (surface-water
routing) interval without either sub-stepping GW inside it or coarsening FSM too. Today FSM
is weekly, so the *immediate, safe* win is `maxiter` 7 → 1 (one weekly Picard GW step vs 7
daily), ~7x fewer GW solves, FSM unchanged. Reaching the seasonal-to-millennial step sizes
the science wants requires asking **how often FSM must actually run** — the next design
question, and now the governing constraint on achievable step size.

**BDF2 interacts with the FSM cadence.** BDF2 needs >=2 consecutive GW steps between FSM
kicks (for the `h^n, h^{n-1}` history); with `maxiter`=1 (one GW step per FSM cycle) there is
no in-cycle history, so that regime is backward-Euler unless history is carried across the
FSM boundary. That may be viable: FSM only moves **surface** water (wtd>0), so for the
**sub-surface** groundwater (wtd<0, the bulk) the history is continuous across an FSM kick —
BDF2 could apply there while surface cells bootstrap. Worth exploring, but adds complexity.

**Accuracy floor:** the GW<->FSM operator split is itself ~first-order at the FSM cadence, so
BDF2 on the GW steps only helps down to that floor; if the split dominates, Strang splitting
(2nd order) is a separate lever. Measure which error dominates first.

---

## 2. BDF2 — cheap second order

Backward Euler (BDF1) approximates the time derivative with a straight line through two
levels; its leading error is `-(Δt/2) h''` → **first order**. BDF2 fits a **quadratic**
through the last three levels and differentiates it, cancelling the `h''` term:

$$
\frac{3h^{n+1} - 4h^{n} + h^{n-1}}{2\,\Delta t} \;=\; \mathcal{L}(h^{n+1})
\qquad\Rightarrow\qquad \text{local } O(\Delta t^{3}),\ \text{global } O(\Delta t^{2}).
$$

So the transient deviations above become second-order: 10× Δt → **100×** error, i.e. for a
fixed path tolerance you take a Δt that is `~1/√tol` larger — far fewer, bigger steps.

> **MEASURED (2026-07-26) — achieved order is ~1, and it is NOT the transmissivity.**
> Self-convergence on the 128² drainage fixture: the *achieved* temporal order is ~2 at coarse
> Δt (250–2000 yr) but **degrades to ~1 at fine Δt** (10–100 yr, order ≈ 1.05) — verified *not*
> a solver-tolerance artifact (identical with `-snes_rtol 1e-12`). Consequence: **BDF2 beats
> backward Euler by a constant factor at equal Δt, but not by an *order*** at practical (sub-mm)
> tolerances (0.1 mm needs Δt ≈ 10 yr mean / 5–6 yr max here).
>
> **Hypothesis TESTED and DISPROVEN:** I attributed this to the piecewise (C⁰) Fan
> transmissivity kinks. Swapping in the smooth (C∞) `T` (`-wtm_smooth_T`) and *widening* its
> smoothing band (`-wtm_smooth_eps` = 0.01 → 1.0 m, 10× past the ~0.1 m physical limit) leaves
> the order at ~0.9 and the errors unchanged — **only the physics shifts**. So transmissivity
> smoothing is a dead end for the order.
>
> **Storativity smoothing ALSO DISPROVEN.** Widening the storativity surface-transition
> (`-wtm_storativity_eps` 0.01 → 1.0 m) does not restore order either — order dips to ~0.7 and
> errors grow 8–45×, while distorting the physics badly (13 mm shift already at 10 cm). So the
> order-1 is **not from coefficient non-smoothness — neither T nor S.** (Two hypotheses refuted.)
>
> **CAUSE FOUND (constant-storativity test):** with `S ≡ porosity` (`-wtm_const_storativity`),
> **BDF2 recovers order ~2** (error ratio → 4: 0.0068 / 0.0427 / 0.172 mm at Δt=10/20/40) and is
> 15× more accurate than the effective-`S` case. So the order-1 was the **2-level backward-Euler
> *secant* effective storativity** `(V(hⁿ⁺¹)−V(hⁿ))/Δh` sitting on BDF2's 3-level time
> derivative — *not* coefficient smoothness (smoothing T or S changed nothing; removing the
> secant fixes it).
>
> **The fix is `BDF2-on-V`** — apply the 3-level BDF2 difference to the stored *volume*:
> `(3Vⁿ⁺¹ − 4Vⁿ + Vⁿ⁻¹)/(2Δt) = flux`, instead of `S_eff·BDF2(h)`. Crucially this is
> **physics-preserving**: `V` is the true stored-water function, so **no fixed-point shift**
> (unlike smoothing, which cost 13 mm at 10 cm), and equilibrium is unchanged
> (`Vⁿ⁺¹=Vⁿ=Vⁿ⁻¹`). A targeted operator/RHS change to the storage term.
>
> **IMPLEMENTED + VALIDATED (`-wtm_bdf2_on_V`, 31e802c).** The operator diagonal carries the
> tangent `dV/dh`, the RHS the volume history `b·V(hⁿ)−c·V(hⁿ⁻¹)` plus a Picard-linearization
> consistency term; SPD structure and ocean Dirichlet unchanged. Measured with the **real
> nonlinear storativity**: self-convergence order **~2** (0.0046 / 0.0347 / 0.151 mm at
> Δt=10/20/40, ratio→4), equilibrium matches the secant scheme to **6×10⁻⁸ m** (physics-
> preserving), and **~23× more accurate than secant BDF2 at Δt=10**. So WTM now has a genuine
> 2nd-order transient scheme, gated/default-off. (Recharge enters as a volume `~Sy·rech`,
> exercised only when `rech≠0`; the order test is zero-forcing, so verify recharge before
> production use.)
>
> **Practical verdict:** the 2nd-order advantage is real but largest at *fine* Δt, where both
> schemes are already far below any meaningful threshold (0.005 vs 0.1 mm at Δt=10). At the
> ~10 yr FSM cadence the secant BDF2 is already ~0.1 mm-accurate, so `BDF2-on-V` is **formal
> completeness, not a practical necessity** — turn it on only if a study needs sub-0.01 mm
> transient paths. The stability win (step ~10 yr not daily) stands independently.

> **ADAPTIVE CONTROLLER — RESOLVED (does not pay off).** Re-tested with the order fixed:
> BDF2-on-V gives the **same step count** as the secant (~52k at tol=1 mm) — it fixed the error's
> non-monotonic U-shape (an order-2 payoff) but **not** the over-refinement. Smoothing **both**
> coefficient interfaces (T at −1.5 m & 0 and storativity at 0, over 10 cm) *also* leaves the step
> count unchanged (~52k). So the over-refinement is **neither the time-order nor the coefficient
> kinks** — it is the controller's **max-over-cells, per-step linear-extrapolation estimator**: one
> fast (near-ocean) cell pins Δt small for the whole domain. At accuracy matched to fixed-1yr
> (~0.05 mm) adaptive needs ~2× the steps of uniform (a modest loss, not the 60× the tight-tol
> numbers suggest). **Conclusion: fixed-step BDF2 at the FSM cadence is the recommendation; adaptive
> would need a fundamentally different error estimator and likely still wouldn't beat uniform on a
> problem this smooth/dissipative.**

> **TRUE 2ND ORDER — RESOLVED (2026-07-26): BDF2-on-V is 2nd order across the full 1–100 yr Δt
> range; the earlier fine-Δt "order → 1" was a COLD-START ARTIFACT, not the scheme.** The order
> tests above start every land cell exactly at the surface (wtd = 0) with a discontinuous head jump
> to the ocean ring — a t = 0 parabolic singularity that injects an **O(Δt) startup error**. That
> term dominates the tiny O(Δt²) truncation once Δt ≲ 5 yr, which is exactly the observed crossover
> (order ~2 coarse, ~1 fine). Ruled out in turn, each by experiment: solver tolerance (identical at
> `-snes_atol` 1e-6 vs 1e-10), the C0 transmissivity kinks (smoothing them 100× changes nothing —
> re-confirmed now on BDF2-on-V), and float32 output (the field is float64). **Decisive test:** rerun
> the convergence study from a SMOOTH resolved state (`supplied_wt`, the Δt = 0.25 yr field at
> T = 1000 yr) so there is no t = 0 singularity. Order is then **clean ~2 everywhere** (further
> 1000 yr window, vs Δt = 0.25 ref, mean |err| over land):
>
> | Δt (yr) | 1 | 2 | 5 | 10 | 100 |
> |---|---|---|---|---|---|
> | mean \|err\| | 0.0022 mm | 0.0093 mm | 0.061 mm | 0.24 mm | 25 mm |
> | order        | — | 2.07 | 2.05 | 2.00 | 2.01 |
>
> (Cold start, same fixture/ref: 0.013 / 0.024 / 0.082 / 0.44 / 59 mm, order 0.9 → 1.3 → 2.4 → 2.1 —
> the startup term inflates fine Δt.) **Practical read:** production transients that *continue from a
> spun-up equilibrium* start smooth → genuine order 2 (~0.002 mm at 1 yr, 0.24 mm at 10 yr, 25 mm at
> 100 yr). A cold flat start pays a one-time O(Δt) startup penalty that caps fine-Δt convergence at
> ~order 1, but only at sub-0.02 mm error. Reproduce: `benchmark/picard/bdf2_on_v_order.py`.
>
> **IMPORTANT scope (2026-07-27):** the order-2 result above is for the **homogeneous** (zero-forcing)
> problem. With **recharge** active the order drops to **1** — the recharge *source* is integrated at
> 1st order (not scaling, not kinks; diagnosed in `BDF2_RECHARGE_ORDER.md`). So production runs
> (recharge always on) are 1st-order in time, ~2× better than backward Euler but not 2nd order.

**It composes with the Picard solve** — each step is the *same* SPD elliptic Picard problem
(`PICARD_MATH.md`), with only:

- the diagonal storage term scaled by 3/2 (`S_c·3/(2Δt)` replaces `S_c/Δt` after the
  row-scaling), and
- the RHS carrying two history levels: `b = S_c·(4h^{n} - h^{n-1})/(2Δt)·Δt + ...`
  (i.e. the frozen-time source uses `(4h^n - h^{n-1})/3` in place of `h^n` for the
  standard BDF2 rearrangement).

The SPD / CG+GAMG structure and unconditional stability are preserved.

**Why BDF2 and not Crank–Nicolson** (the other 2nd-order option): CN is A-stable but **not
L-stable** — its stiff-mode amplification factor → −1, so it *rings* on sharp fronts and
stiff modes, exactly WTM's regime (discontinuous T, sharp water tables). BDF2 damps stiff
modes toward zero (L-stable). For this problem BDF2 is the correct 2nd-order method.

**Costs:** one extra stored field (`h^{n-1}`); a one-step backward-Euler **bootstrap** (no
`h^{n-1}` at t₀); and, under adaptive Δt, the **variable-step BDF2** coefficients (the
3/2, −2, 1/2 weights change when consecutive steps differ — standard, but must be used or
the order drops).

---

## 3. Adaptive Δt without a ground truth

The key idea: **never compare to truth — compare two approximations of different order at
the same step** to estimate the *local* truncation error, then size Δt to hold it under a
tolerance.

- **Embedded estimate (preferred, ~free):** BDF2 already carries a BDF1 (or explicit
  predictor) companion sharing the same solve; `‖h_BDF2 − h_predictor‖` estimates the local
  error. This is how stiff ODE solvers (SUNDIALS CVODE, `ode15s`) do variable-order/step.
- **Step-doubling (fallback, method-agnostic):** one step of Δt vs two of Δt/2; their
  difference estimates the error. ~3 solves/step.

**Controller:** accept the step if `err < tol`; set the next `Δt_new = Δt·(tol/err)^{1/(p+1)}`
(p = order), clamped by a safety factor and max growth ratio (a standard PI controller is
smoother). Reject and retry with smaller Δt if `err > tol`.

**Why local control is trustworthy here:** the problem is **dissipative** — diffusion damps
perturbations, so old truncation errors *decay* rather than accumulate. Unlike a chaotic or
hyperbolic system, local-error control therefore yields reliable global accuracy. (This is
also why the transient error washed out toward equilibrium in §1.)

The user sets one knob — a **path tolerance** (e.g. "keep the transient water table within
X m") — and Δt is chosen automatically, small early (strong nonlinearity, fast transient)
and growing as the system settles.

### The step's Δt and the next step's Δt are different numbers

Worth stating explicitly, because conflating them was a real bug (fixed 2026-08-26). The
controller's output is the **next** step's Δt. Everything that accounts the step just *taken* —
the BDF2 history ratio `ω = Δt/Δt_{n-1}`, the taper-1 and taper-2/3 removal depths, the
land→ocean flux accumulation, TR-BDF2's step-flux quadrature — needs **that step's** Δt. Writing
the controller's output into `user_context.deltat` at the point it is computed handed all five of
them the wrong number.

It went unnoticed because nothing checked the water budget under adaptive Δt. Once TR-BDF2 gained
an exact budget it was immediate: exact residual **−1.603 of recharge** for TR-BDF2 + adaptive and
**−0.417** for BDF2-on-V + adaptive, against ~2e-07 for the same schemes at fixed Δt. Sizing is now
the last thing `update()` does, and `tests/budget_closure` carries an adaptive arm per integrator.

The reject path was never affected: it returns before any accounting runs, so it still writes
`deltat` directly.

### `implicit` + adaptive Δt cannot complete

The `implicit` collector's retained head is ~linear in Δt, so **shrinking Δt moves the solution
instead of converging it**. The local-error estimate therefore never settles and the controller
exhausts its retries, aborting with *"adaptive dt: step failed after max retries"* under both
TR-BDF2 and BDF2-on-V. This is not a controller fault — it is the same dt-dependence that made
`active_set` the default enforcement, showing up as a hard failure rather than as a quietly
dt-dependent lake. Use the default collector with adaptive Δt.

---

## 4. The equilibrium side: pseudo-transient continuation (automate the big step)

For `run_type equilibrium` we don't want path accuracy at all — just the steady endpoint.
The natural automation is **pseudo-transient continuation / switched evolution relaxation
(SER)**: grow Δt as the residual falls (e.g. `Δt ∝ 1/‖F‖`), ramping from small (far from
steady state, nonlinearity strong) to enormous (near steady state) automatically. This
drives to the Δt-independent equilibrium in a handful of effective steps with **no tolerance
on the path and no ground truth** — it is a root-find, not an IVP. Same solver, opposite
Δt policy from the transient controller. Low priority (Picard + a large fixed Δt already
gets most of this), but it removes the "pick a good Δt" burden from equilibrium runs.

---

## 5. The unified design

| run type | goal | Δt policy | order |
|---|---|---|---|
| equilibrium | steady endpoint (root-find) | grow with 1/‖residual‖ (PTC) | irrelevant (BDF1 fine) |
| transient | trajectory (IVP) | adaptive, local-error-controlled | BDF2 |

Both share the semi-implicit Picard solve; only the time term and the Δt policy differ.
This is the numerical content of WTM's existing equilibrium-vs-transient distinction.

---

## 6. Implementation sketch

1. **BDF2 time term (fixed step first).** Add a `-wtm_bdf2` path: store `h^{n-1}`, form the
   BDF2 diagonal (3/2 scaling) and two-level RHS in `FormPicardOperator`/`FormPicardRHS`,
   bootstrap step 0 with backward Euler. Verify order-2 by the §1 matched-time comparison
   (deviations should scale as Δt², not Δt).
2. **Adaptive controller.** Add the embedded BDF1/BDF2 error estimate + PI step controller +
   accept/reject, behind `-wtm_dt_adaptive -wtm_dt_tol <X>`. Reuse the §1 harness to confirm
   the achieved path error tracks the tolerance.
3. **PTC for equilibrium** (optional, later): Δt ∝ 1/‖F‖ growth for `run_type equilibrium`.
4. Keep fixed-Δt backward Euler the default; gate all of the above behind flags.

---

## 7. Risks / open questions

- **What is a "time step" in WTM's cycle/maxiter/FSM structure?** Today each `maxiter`
  iteration is a full backward-Euler step of `deltat` with recharge applied, and FSM runs
  once per cycle. BDF2 needs a *consistent history* (`h^n, h^{n-1}` at known Δt spacing),
  so the refactor must define the timeline cleanly: is a step a maxiter iteration or a
  cycle? How do the per-cycle FSM and recharge updates interleave with a multi-step history?
  **This is the central design question and must be settled before coding BDF2.**
- **Is a given transient integrator-limited or forcing-limited?** WTM transients are driven by
  time-sliced paleoclimate forcing; where the usable Δt is set by the *forcing cadence*,
  BDF2/adaptivity buys less. This is no longer a *gate* (per the scope decision we build the GW
  capacity regardless — the daily step is a stability workaround, and the science of interest
  is seasonal-to-millennial); it's a **per-study usage question** — the adaptive controller
  should simply not step past the forcing cadence.
- **Variable-step BDF2** must use the correct non-uniform coefficients, or it silently drops
  to first order; and large step-ratio jumps can dent stability — cap the growth ratio.
- **Storativity/transmissivity nonlinearity in the time term.** S and T are frozen at the
  Picard iterate; with two history levels confirm the freezing still converges to the right
  fixed point each step (the §1 matched-time check catches this).
- **Interaction with FSM.** As with the Picard tests, `fsm_off` isolates the GW integrator;
  a production-like `fsm_on` transient may behave differently (surface water routed by FSM
  each cycle) — test both.

---

## 8. Phasing

**Scope decision (Andy, 2026-07-25): improve the groundwater-modeling capacity first, then
consider the coupling.** FSM frequency *varies* across studies, so we do **not** hard-couple
the time-integration design to a fixed FSM cadence. Build and validate the GW time integrator
as a standalone capability **with `fsm_off`** (exactly what `benchmark/picard/` already
exercises), then layer FSM coupling on top as a later phase.

*Phase A — GW time-stepping capacity (fsm_off):*
1. Timeline is settled (§1.5); go straight to the integrator.
2. **BDF2 fixed-step**, verify order 2 on the matched-time harness (`transient_accuracy.py`).
3. **Adaptive Δt**, verify achieved path error tracks the tolerance.
4. **PTC for equilibrium** — automate the big-step drive-to-steady-state.
5. Decide per-run-type defaults; keep backward-Euler fixed-step as the fallback.

*Phase B — FSM coupling (later, once A is solid):*
6. Interleave the GW integrator with FSM for a range of FSM frequencies (it varies): how large
   a GW step is safe/accurate between surface-routing events; carry BDF2 history across FSM
   kicks (FSM only moves surface water, so sub-surface history is continuous); Strang-split if
   the operator-split error dominates. Test `fsm_on`.
