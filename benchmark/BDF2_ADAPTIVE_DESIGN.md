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

**But this also sharpens the payoff question.** At `deltat` = 1 day the backward-Euler
temporal error is already small, so BDF2 buys little *at that step size*. The value hinges
on **why the step is daily**:

- If daily is an **Anderson-stability workaround** (Anderson diverges at large dt — we
  measured a ceiling ~1 yr on the drainage test; the daily production step may be the same
  phenomenon at production stiffness), then Picard's unconditional stability lets us take
  **far fewer, bigger GW sub-steps per week** (perhaps 1 weekly step vs 7 daily), and BDF2
  keeps those coarser steps accurate. *This is the transient win.*
- If daily is **physically required** (a fast event the science needs resolved) or set by
  the forcing, the step can't be coarsened and BDF2 mainly buys accuracy on the resolved
  daily transient.

Either way, **the GW<->FSM operator split is itself ~first-order at the weekly cadence**, a
floor on overall transient accuracy: BDF2 on the daily sub-steps only helps down to that
floor. Whether the daily-diffusion error or the weekly-splitting error dominates is the
first thing to measure. (Strang splitting would lift the split to 2nd order if it becomes
the bottleneck — a separate lever.)

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
- **Is the transient even integrator-limited?** WTM transients are driven by time-sliced
  paleoclimate forcing; if the usable Δt is set by the *forcing cadence*, BDF2/adaptivity
  buys little. **Check how transient runs are actually forced before investing** — this
  gates the whole branch's value.
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

1. **Settle the timeline** (risk #1) + confirm transients aren't forcing-limited (risk #2).
   If forcing-limited, stop here and record.
2. **BDF2 fixed-step**, verify order 2 on the matched-time harness.
3. **Adaptive Δt**, verify achieved error tracks tolerance.
4. **PTC for equilibrium** (optional).
5. Decide defaults per run type; keep backward-Euler fixed-step as the fallback.
