# Transient recharge is storativity-weighted inconsistently across time-integration paths

**Date:** 2026-08-13 · **Branch:** `bdf2-adaptive-dt` · **Status:** diagnosed; fix proposed, not yet applied.
**Scope:** correctness of the **transient** groundwater step at cells whose water table crosses the land
surface within a step. Equilibrium runs, below-surface transients, and the golden tests are **unaffected**
(and stay byte-identical), which is why this stayed hidden.

## One-sentence statement

Recharge is applied as a *head increment* whose value assumes storativity = porosity, and the several
time-integration residuals then re-scale it by **different** storativities (backward-Euler and Picard by
the **secant** `S`; TR-BDF2 and BDF2-on-V by the **tangent** `Sy`), so the schemes converge to **different
water tables as dt → 0** for any cell that crosses `wtd = 0` within a step.

## How it was found (Esquibel, warm −20% P−ET step, task #93)

The transient benchmark warm-starts from the corrected equilibrium and applies a −20% step to net forcing
`P − E`, then compares corrected-Callaghan (cc: matrix-free Anderson, backward-Euler, 1st-order) against
TR-BDF2 (`-wtm_tr_bdf2`, 2nd-order). The intended deliverable was "how much runtime does 2nd order buy."
The chase instead ran through three layers:

1. **The endpoint metric was measuring the wrong thing.** A relaxation smoke showed the system relaxes to
   its floor by cycle ~16 (e-folding ~1.7 wk dry, ~1.1 wk wet), so an endpoint at 16 wk sits at
   steady state where time-order is invisible. Moving the measurement to t = 2 wk (mid-relaxation, per-method
   self-convergence) fixed the metric.

2. **Order is capped at 1, for both schemes.** Self-convergence at t = 2 wk gives order → 1.0 for cc **and**
   for TR-BDF2 — the non-smooth seepage face (active-set switching at `wtd = 0`) destroys TR-BDF2's formal
   2nd order. TR-BDF2 keeps a ~2× smaller error constant and a **4× larger stable step** (converges to
   dt = 8 wk warm; cc diverges `DIVERGED_MAX_IT` at dt ≥ 4 wk), but it does not buy an order.

3. **The schemes disagree in the dt → 0 limit.** Cell (449,833) (topography 260 m, initial `wtd` −82.9 m)
   converges — **dt-independently across a 32× range, 0.0625 → 2 wk** — to cc = −32.9 m and TR-BDF2 = −77.4 m
   at t = 2 wk. Two consistent discretizations of the same parabolic problem must share the dt → 0 limit;
   that they do not means one path is **inconsistent**.

**The clamp is not the cause.** With the surface clamp turned off entirely (`CLAMP=0`, pure mass-conserving
groundwater), the same cell still converges to cc = **+4.92 m** (above the surface) versus TR-BDF2 = −74.96 m,
dt-independent. Field-wide at t = 4 wk: **10,506 land cells** differ by >1 m (RMS 0.82 m), ~23 by >10 m. So
the divergence lives in the residual, not in the post-solve exfiltration projection.

## Root cause

`my_rech` (the per-cell recharge stored in `rech_vec`) is `add_recharge() = rate·dt / porosity` for a
below-surface cell (`add_recharge.hpp`). That is a **head-rise** — correct only where storativity equals
porosity, i.e. below the surface. The physically correct recharge is a fixed **volume**,
`V_r = rate·dt·A = poro·my_rech·A`, independent of storativity. The residual paths, however, re-apply
storativity, and they do not agree on which one (`transient_groundwater.cpp`):

| path | flag | recharge enters as | effective recharge volume | storativity |
|---|---|---|---|---|
| Backward-Euler (**default**) | — | `(x − my_rech) + dt·N/(A·S)`, b = hⁿ | `A·S·my_rech` | **secant** `S = updateEffectiveStorativity(start, w_c)` |
| Picard | `-wtm_picard` | RHS `A·S_c·(hⁿ + my_rech)` | `A·S_c·my_rech` | **secant** `S_c` |
| TR-BDF2 | `-wtm_tr_bdf2` | `… − TR_G·my_rech`, storage ÷ Sy | `TR_G·Sy·my_rech` | **tangent** `Sy = specificYield(w_c)` |
| BDF2-on-V | `-wtm_bdf2_on_V` | `storage − my_rech`, storage ÷ Sy | `Sy·my_rech` | **tangent** `Sy` |

Below the surface `Sy = S = poro`, so every path applies `poro·my_rech·A = V_r` — correct, and identical.
The paths diverge **only** for a cell that crosses `wtd = 0` within a step, where secant ≠ tangent ≠ porosity:

- Backward-Euler applies `(S/poro)·V_r`. As the cell rises toward the surface `S → 1`, so it **over-recharges**,
  which lifts the cell further, which raises `S` again — a positive feedback that drives the cell to +5 m
  **above** the surface on a *drying* step. This is the default path.
- TR-BDF2 and BDF2-on-V use the endpoint tangent `Sy`; because their cells stay below the surface they get
  `Sy = poro` and the correct volume — but a cell that genuinely must cross would make their endpoint-`Sy`
  wrong too.

The tangent group does not even agree with itself: at the same cell, no clamp, dt → 0, TR-BDF2 = −74.96 m
while BDF2-on-V = −84.80 m — 10 m apart and on the **opposite sign** of change relative to the −82.9 m start.
**Three schemes, three dt → 0 limits.** The correct statement is therefore not "backward-Euler is buggy and
TR-BDF2 is right"; it is that the near-surface recharge/storativity coupling is formulated inconsistently
across every time-integration path, and none of them uses the correct fixed-volume recharge. Backward-Euler —
the default — is the clearest outlier, landing a drying cell on the wrong side of the surface.

## This is an upstream (v2.0.1) defect, not new

`src/add_recharge.hpp` is **byte-identical** to `kcallaghan-wtm` (v2.0.1) apart from our added
`-wtm_extended_soil` branch and comments — the `/porosity` head-conversion is inherited from the Callaghan
lineage. And v2.0.1's transient residual (`kcallaghan-wtm/src/transient_groundwater.cpp:277`,
`f = (uxx+uyy)*deltat/my_storativity + this_x - my_rech`, `my_storativity` = the secant) **is** the
backward-Euler `cc` path structurally. So the recharge-volume-at-crossing error lives in upstream too. It
stays invisible there because v2.0.1 has only one scheme (nothing to disagree with) and is run to
equilibrium (the within-step crossing vanishes at the stationary fixed point). Our TR-BDF2 / BDF2-on-V
paths **revealed** the defect; they did not introduce it. The fix is therefore a genuine upstream
contribution (bearing on the #83 PR), not merely a branch fix.

## Why equilibria never showed it

At equilibrium the water table is stationary, so within the converged step there is no secant/tangent gap:
`S`, `Sy`, and `poro` coincide at the fixed point for any cell not sitting exactly on the surface. The
mass-conservation program, the golden tests, and every below-surface transient are consequently correct and
remain byte-identical. The defect is specific to **cells in transit across the surface during a transient
step** — the regime this −20% forcing experiment was the first to exercise directly.

## Proposed fix (stewardship — touches the default path and the goldens)

Recharge should enter the volume balance as a fixed **volume** `poro·my_rech` (= `rate·dt`), independent of
storativity, in **every** path, with the surface-crossing partition (pore-fill below, surface water above)
resolved **inside** the solve against the current iterate rather than baked from `starting_wtd`. Concretely:

1. Add the recharge volume to the discrete balance directly (`V(wⁿ⁺¹) − V(wⁿ) = V_r − dt·N`, using the
   `storedVolume` nonlinearity that already partitions pore-fill vs surface water at the iterate), rather
   than as a storativity-scaled head increment.
2. Make the three residual branches and the Picard RHS share that one recharge term, so secant/tangent no
   longer enters the source.
3. Regression test (must bite before the fix): a single surface-crossing cell must converge to the **same
   dt → 0 limit** under cc, TR-BDF2, and BDF2-on-V — the cross-scheme agreement that fails today.

The fix is expected to change the goldens for any transient with surface-crossing cells (equilibria and
below-surface cases should stay byte-identical); it needs a careful regold and Andy's sign-off.

## Validation of the fix (volume-based recharge)

Four independent checks, all consistent:
1. **Regression test** `tests/recharge_consistency/` (16x16 surface-crossing plateau): cc-vs-TR-BDF2 gap
   **3.66 m -> 0.034 m** (bites without the fix; passes with it). Confirmed on both the laptop and MSI binaries.
2. **Goldens**: below-surface-strict cells and below-surface equilibrium fixed points are byte-identical;
   the surface-touching cases move because the old values under-counted recharge at crossing cells (they
   were the buggy values). Regold required.
3. **FSM before/after** (golden `fsm_*` cases, pre-fix reference vs fixed binary): the fix materially moves
   FSM-coupled fields -- `fsm_evap1` **+5.0 m wetter** (restored recharge leak), `fsm_runoff` drains four
   lake cells from +0.38 m to -8.9 m (max change 36 m). Direction is domain-dependent. This is a strong
   candidate for previously-observed "odd water tables", and FSM masks the water-table symptom by routing
   the spurious surface water into the lake field.
4. **Esquibel field-wide** (cc, no-clamp, t=4 wk, fixed vs pre-fix binary): 18,544 land cells shift, max
   0.32 m, mean **+0.004 m (slightly wetter)** -- the restored recharge, broadly and modestly.

**Scope correction.** The fix does **not** change the Esquibel cell (449,833) 80 m cc-vs-TR-BDF2
disagreement that first surfaced this investigation (cc +4.92 / tr -75 / bdf2v -85, byte-identical before
and after). That cell rises 87 m in four weeks -- a numerical artifact of the deep, steep exponential-T
tail (`finding_operator_singularity`), **not** a recharge effect. The recharge/storativity crossing bug and
that deep-cell disagreement are two separate things; only the former is fixed here. The deep-cell exp-T
disagreement is a distinct, still-open anomaly.

## Bottom line for #93

The benchmark question has an answer — TR-BDF2 buys **~4× stability and a ~2× smaller error constant, with
no order gain** (the seepage face caps both schemes at 1st order) — but the finding that matters is the
recharge-consistency bug it surfaced in the default transient path. See
`benchmark/FREE_SURFACE_RUNOFF.md` (the sibling free-surface treatment) and the memory note
`finding_recharge_storativity_inconsistency`.
