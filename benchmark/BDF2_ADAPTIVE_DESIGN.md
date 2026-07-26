# Design Note: higher-order + adaptive time stepping for transient accuracy

**Date:** 2026-07-25
**Branch:** `bdf2-adaptive-dt` (off `picard-mg` — builds on the semi-implicit Picard solve)
**Status:** DESIGN / prototype sketch. For a bounded experiment, not a commitment.
**Authors:** Andy Wickert + Claude
**Companions:** `PICARD_MG_DESIGN.md`, `PICARD_MATH.md` (the implicit solver this builds on).

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
> **Updated takeaway:** genuine 2nd order *is* achievable on WTM via `BDF2-on-V` (no physics
> compromise) — so the transient-accuracy path is real. Whether it also rehabilitates the
> adaptive controller (its step-explosion may be downstream of the order-1) is an open follow-up.
> The stability win (step at ~10 yr not daily) stands independently.

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
