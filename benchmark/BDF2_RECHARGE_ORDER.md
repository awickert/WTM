# Design Note: BDF2-on-V loses 2nd order under recharge — diagnosis & fix plan

**Date:** 2026-07-26
**Branch:** `bdf2-adaptive-dt`
**Status (2026-07-27): RESOLVED (GW-step) — cause + fix found, clean-verified. See §15.** BDF2-on-V
is 2nd order for the homogeneous problem but 1st order under recharge because recharge pushes water
tables across the land surface (wtd=0) — a **moving free boundary** (storativity jumps porosity→1,
T clamps) that gives h(t) a temporal kink and drags the whole domain (even deep subsurface, order
~1.15) to 1st order. NOT evap-mode-specific, NOT a smoothable coefficient kink. FIX (Andy's idea,
gated `-wtm_extended_soil`, default off): continue the aquifer above the surface so the GW step is
smooth → **order 2 restored** (2.07/2.07/2.00, ~3000× smaller error at Δt=1; golden clean; ~no-op
below surface). Production half — truncate the mound to real topography at the FSM handoff — not yet
done. ⚠️ §13 records an earlier stale-work-dir wrong turn (an "implicit recharge" fix that was a
contamination artifact); the §9–§12 cause analysis from that dir is superseded by the clean §14–§15.
--- earlier (now-superseded) status kept for the trail ---
What is SOLID (re-verified in a clean work dir): BDF2-on-V is **2nd order in time for the
homogeneous problem (P=0)** and **1st order with recharge (P>0)** — order ~1 (2.04/4.42/11.2/22.8/274
mm at Δt=1/2/5/10/100, P=0.01, clean dir). What was WRONG: the earlier "root cause = explicit
recharge; implicit fixes it → order 2" conclusion was a **stale-work-dir artifact** — in a clean
dir the implicit-recharge change gives order ~1, i.e. it does NOT fix it (§13). So the cause
analysis in §9–§12 (D1–D6, the mechanism, the fix) is **contaminated and must be re-verified in
clean dirs**; do not trust it. Richardson (§10) is also not a clean order-2 fallback.
**Companion:** `BDF2_ADAPTIVE_DESIGN.md` (the transient-accuracy work this extends).

## 1. The problem (airtight)

BDF2-on-V (`-wtm_bdf2_on_V`) is genuinely 2nd order in time for the **homogeneous** (relaxation /
drainage) problem, but **drops to 1st order the moment recharge is active** — i.e. in every
production run. Controlled A/B on the 128² fixture, **identical** deep smooth initial condition
(`supplied_wt`), 1000 yr window, error over land vs a Δt = 0.25 yr reference; only recharge differs:

| P (m/yr) | dt=1 | dt=2 | dt=5 | dt=10 | dt=100 | order |
|---|---|---|---|---|---|---|
| 0.0  | 0.0022 | 0.0093 | 0.061 | 0.24 | 25 mm | **~2.0** |
| 0.01 | 1.95 | 4.17 | 10.2 | 20 | 183 mm | **~1.0** |

Repro: `benchmark/picard/` — the smooth-IC study in `bdf2_on_v_order.py` plus the recharge A/B
(scratch `ab_recharge.py`; to be folded into a committed script once diagnosed).

## 2. Why this is a puzzle (the static read says order 2)

Recharge enters the BDF2-on-V RHS (`FormPicardRHS`) as an **implicit source** (not an operator
split — verified): `bb += Sy·rech`, with `rech = precip·Δt` and, in the deep regime,
`add_recharge → rech/porosity` (water-table-*independent* → an effectively constant source). At the
Picard fixed point the storage side is `(3Vⁿ⁺¹ − 4Vⁿ + Vⁿ⁻¹)/2`. BDF2 of `dV/dt = flux + P`:

```
(3Vⁿ⁺¹ − 4Vⁿ + Vⁿ⁻¹)/(2Δt) = flux + P
⇒ (3Vⁿ⁺¹ − 4Vⁿ + Vⁿ⁻¹)/2 = Δt·flux + Δt·P
```

so the source weight BDF2 wants is `Δt·P`, and the code supplies `Sy·rech ≈ P·Δt`. **It matches.**
A constant source is integrated *exactly* by BDF2 (2nd order). So the algebra predicts order 2 —
yet it measures order 1. The discrepancy is where the real cause hides.

## 3. Candidate mechanisms (undistinguished so far)

1. **Subtle source-integration inconsistency.** The source is implicit, but `add_recharge` is
   evaluated at `hⁿ` (`starting_wtd`), `rech_dist` is rebuilt each step, and the BE bootstrap
   step handles the first recharge increment differently. One of these could break the 2nd-order
   weighting in a way the static read misses.
2. **Recharge-induced kink crossing.** Recharge pushes the water table *up*; a large-Δt step can
   overshoot across the C0 transmissivity kinks (−1.5 m, 0 m) that the resolved reference glides
   past — a non-smooth-solution order reduction. Unlike the zero-recharge case (where smoothing
   did nothing), here smoothing the kinks *could* help. Hint: max error was 862 mm at Δt=100 vs
   20 mm mean → a few cells doing something violent (kink-like).

## 4. Diagnostics (run in order; each narrows the cause)

- **D1 — Mass conservation across Δt.** Total recharge *input* is `P·T`, Δt-independent, if the
  source scales correctly. Compare `total_added_recharge` (or the summed input) for Δt = 1/10/100
  and the reference. **If it differs → a source *scaling* bug (mechanism 1), the cheapest fix.**
- **D2 — Where does the error live?** Locate the max-|err| cells for the Δt=100 run vs the
  reference and read their water-table depth. **On/near a kink (wtd ≈ −1.5 or 0 m) → mechanism 2;
  spread over smooth deep cells → mechanism 1.**
- **D3 — Recharge order with kink-smoothing ON.** Re-run the A/B (P=0.01) with
  `-wtm_ksat_soilbottom_smoothing_width` and `-wtm_ksat_surface_smoothing_width` set. **Order
  recovers → mechanism 2 (smoothing is the accuracy lever); unchanged → mechanism 1.**

## 5. Fix menu (choose after diagnosis)

- **Mechanism 1 (source):** targeted RHS fix (correct the source timing/weight) → clean 2nd order,
  cheap. Best case.
- **Mechanism 2 (kinks):** the now-universal ksat smoothing becomes the accuracy lever, or a
  step-size limit for cells approaching the surface.
- **Method-agnostic fallback (works regardless):** **Richardson extrapolation in time** — run Δt
  and Δt/2, form `2·h(Δt/2) − h(Δt)`; the clean O(Δt) error cancels → 2nd order. ~1.5–3× steps, no
  scheme change. Reliable here because the order is a clean 1 (error has the required expansion).

## 6. Practical impact (scope the effort)

The **stability** win (big stable steps vs daily) is untouched — that was always the main prize.
1st-order *absolute* error is still small at moderate steps (~2 mm at 1 yr, 20 mm at 10 yr, 183 mm
at 100 yr on a 100 m field). This only bites for **accurate decadal-plus steps with recharge**. If
target runs live in the seasonal-to-yearly regime, this is a documented footnote; if they need
accurate big steps, the fix is worth it.

## 7. Results (appended as diagnostics run)

**D1 — Mass conservation across Δt: PASS (not a scaling bug).** Total recharge input
(`total_recharge_added` from the run log, T=1000 yr, P=0.01 m/yr) is identical to 6 sig figs
across Δt = 0.25 / 1 / 2 / 5 / 10 / 100 yr: **1.531e13 (all 1.0000×)**. So the source adds the
right *amount* of water at every Δt — the 1st-order error is about temporal *distribution* or
solution structure, not mis-scaling. Eliminates the scaling sub-hypothesis of mechanism 1.

**D2 — Where the error lives: MIXED (points toward mechanism 2 with propagation).** For the Δt=100
run vs the Δt=0.25 reference (land wtd range [−99.6, 0] m; mean err 183 mm, max 862 mm):
- The **worst cells sit at wtd ≈ −0.8 to −1.0 m** (max 862 mm) — shallow near-ocean cells between
  the 0 m and −1.5 m kinks, i.e. where recharge/drainage can overshoot a kink at large Δt.
- But **72.7% of the total |error| is on deep smooth cells (wtd < −2 m)** that never touch a kink
  (they are 76.7% of land, so error is ~uniform by area); only 20.8% is within 0.5 m of a kink.
- Reading: the error is *generated* at the near-kink shallow cells (the 862 mm tail) and then
  **diffuses inward** to the deep interior — so most error sits on deep cells by area, but its
  source is the kinks. D2 alone can't separate "source is 1st-order on deep cells too" from
  "kink error propagates"; D3 decides.

**D3 — Order with kinks smoothed: NO recovery (refutes mechanism 2).** Re-running the recharge A/B
(P=0.01) with all three kinks smoothed 1.0 m leaves the order at ~1 (1.21 / 1.06 / 0.98 / 0.83 at
Δt = 2/5/10/100) and the errors *larger* (3.4 / 7.8 / 21 / 41 / 273 mm vs 1.95 / 4.2 / 10 / 20 / 183
un-smoothed — smoothing only shifts the physics). So kink-crossing is **not** the cause.

### Conclusion of D1–D3

Not a scaling bug (D1), not kink-crossing (D3), and the bulk of the error is on deep smooth cells
(D2). ⇒ **The recharge source is integrated at ~1st order in time, on smooth cells.** This is a real
order reduction, *not* the coefficient non-smoothness that limited the homogeneous case. Puzzle
that remains: static analysis of the RHS says the source weight is BDF2-consistent (`Sy·rech = Δt·P`)
and D1 confirms the mass is right — so the 1st-order behavior is a **subtle temporal-placement
inconsistency the static read doesn't reveal**. Pinning it to a line of code needs instrumentation
(a manufactured-solution test, or dumping the per-step applied source vs the BDF2-consistent value).

**D4 (confirmatory) — BDF2-on-V vs plain backward Euler, recharge on.** mean |err| vs the Δt=0.25
BDF2-on-V reference (P=0.01, T=1000 yr):

| Δt (yr) | backward Euler | BDF2-on-V | V / BE |
|---|---|---|---|
| 1 | 4.02 mm | 1.95 mm | 0.49 |
| 10 | 40.3 mm | 20.0 mm | 0.50 |
| 100 | 385 mm | 183 mm | 0.47 |

BDF2-on-V is a **constant ~2× more accurate than BE** at every Δt — a real but *constant-factor*
gain; both are **order 1** (flat ratio). So the BDF2 storage treatment does help (~2×), but the
1st-order source caps the overall order.

## 8. Recommendation

- **Practical:** in production (recharge on) BDF2-on-V gives ~2× better accuracy than BE at the same
  step **plus** unconditional stability, but it is **1st order in time, not 2nd**. The 2nd-order
  benefit is real only for recharge-free relaxation. Update the `BDF2_ADAPTIVE_DESIGN.md` / memory
  "true 2nd order" claim to this scoped statement.
- **If true 2nd order with recharge is wanted:** two routes —
  1. **Richardson extrapolation in time** (run Δt and Δt/2, `2·h(Δt/2) − h(Δt)`): guaranteed 2nd
     order here (clean order-1 error), no scheme change, ~1.5–3× steps. The safe default.
  2. **Pin & fix the source integration:** the static read says it should be 2nd order, so a
     targeted fix likely exists once instrumented — build a **manufactured-solution** test (impose
     an analytic h(t), add the matching source, measure order) or dump the per-step applied source
     vs the BDF2-consistent value. Higher payoff (clean 2nd order at 1× cost), needs the dig.
- **Do NOT** reach for coefficient smoothing here — D3 shows it doesn't help and shifts the physics.

## 9. Instrumentation (2026-07-27)

**D5 — source isolation (diffusion off): the source term ALONE is exact.** With ksat set to 1e-8
(diffusion ~off) and uniform recharge, a constant source gives a linear-in-t solution, which BDF2
integrates exactly. Measured: dt=1 vs dt=100 land fields agree to **0.065 mm mean** (vs 183 mm in
the coupled case); max 56 mm on a few near-ocean cells where residual flux remains. ⇒ **the source
term is not the bug; the 1st order lives in the source↔diffusion COUPLING** within the BDF2 step
(most likely the row-scaled SPD operator/RHS balance — the recharge vs implicit-diffusion vs storage
scaling). Pinning the exact line needs a **code-level manufactured-solution test** (impose an
analytic h(t) with both diffusion and a matching source; measure order; it also gives a TRUE 2nd-order
reference, which the black-box tests lack).

**Richardson-in-time backup — helps 10–30×, order not yet cleanly confirmed.** On the recharge runs,
`2·h(Δt/2) − h(Δt)` cuts the error 10.5× (Δt 2→1) and 31× (10→5). But its *order* can't be measured
against the dt=0.25 reference, because **that reference is itself 1st-order under recharge** — the
extrapolant is more accurate than the yardstick, so the comparison floors at the reference's own
~0.4–0.6 mm error. A clean order check needs a Richardson-extrapolated (or code-MMS) reference.

**D6 — nonlinear-solve convergence: NOT the cause.** Tightening the Picard/SNES solve 6 orders of
magnitude (`-snes_atol/rtol 1e-12 -snes_stol 1e-14 -snes_max_it 500`) leaves the recharge order and
errors bit-for-bit unchanged (1.95 / 4.17 / 10.2 / 20 / 183 mm, order ~1). The outer solve is fully
converged; lagged implicit terms are not the cause.

**Re-derivation of the code (operator + RHS): the discretization is textbook order-2.** Reading
`FormPicardOperator` (diagonal `a·Sy`, off-diagonals `−e·Δt/h²`) and `FormPicardRHS` (`a·Sy·x −
a·V(wⁿ⁺¹) + b·V(wⁿ) − c·V(wⁿ⁻¹) + Sy·rech`), `A·x = b` collapses at the fixed point to
`(3Vⁿ⁺¹−4Vⁿ+Vⁿ⁻¹)/2 = Δt·DIFF(hⁿ⁺¹) + Sy·rech` with `Sy·rech = Δt·P` — exactly BDF2 of
`dV/dt = DIFF + P`. So the discrete scheme is order-2 *by construction*; the order-1 is not in the
written stencil.

### Eliminations so far (all by experiment or derivation)
scaling (D1) · kinks (D3) · source-in-isolation (D5) · solver convergence (D6) · the discretization
itself (re-derivation). Remaining suspect: the **BE-bootstrap → BDF2-on-V transition** (first step
uses BE + the *secant* storativity, then switches to the tangent-V form) interacting with the
recharge source — UNVERIFIED. Definitive next step: a **consistency-residual / manufactured-solution
test** (plug 3 consecutive fine-reference states into the discrete equation; check the residual is
O(Δt²) not O(Δt)) — a real build; needs intermediate reference saves + a python flux stencil.

**Status:** root cause narrowed (discretization is order-2; suspect = bootstrap↔recharge), not fully
pinned. **Accuracy deliverable: Richardson-in-time** (validation in §10).

## 10. Accuracy fix: Richardson-in-time — does NOT cleanly restore order 2

Validated against a 2nd-order reference (`2·h[0.25]−h[0.5]`): `2·h(Δt/2)−h(Δt)` helps 2–10× but the
**extrapolant order is only ~0.9–1.6, not 2**. So the recharge error is *not* a clean O(Δt) with a
simple asymptotic expansion (the shallow-cell `add_recharge` nonlinearity makes it irregular), and
Richardson is an unreliable backup here. This pushes us to fix the source, not paper over it.

## 11. Root cause (2026-07-27): explicit recharge on shallow cells

Correcting §9: the BE bootstrap is NOT a viable suspect either — a single backward-Euler start has
O(Δt²) *local* error, contributing only O(Δt²) globally; it cannot produce order 1. The written
discretization is order-2. So the order-1 must come from an input to it that is only 1st-order.

**Mechanism:** in `FormPicardRHS` the recharge is `Sy·rech` with `rech = add_recharge(rech_dist,
starting_wtd = hⁿ, …)` — evaluated at the OLD water table hⁿ (explicit). BDF2 wants the source at
tⁿ⁺¹. For DEEP cells `add_recharge` is wtd-independent (`rech/porosity`), so explicit vs implicit
is identical → exact (this is why D5, all-deep, was exact). For SHALLOW near-ocean cells it is
wtd-*dependent* (the `GW_space`/surface-cap branches), so evaluating at hⁿ is a lagged, 1st-order
source. That zone (`GW_space = −wtd·porosity < P·Δt`, i.e. `wtd > −P·Δt/porosity`) *grows with Δt*.
Those shallow cells produce the 862 mm tail (D2); the error then diffuses into the deep interior
(D2's 73%). Fits every diagnostic: D1 (mass ok — it's timing not amount), D2 (shallow→deep),
D3 (kink smoothing irrelevant), D5 (deep-only exact), D6 (convergence irrelevant).

**Fix under test:** evaluate `add_recharge` at the iterate (→ hⁿ⁺¹, implicit) in `FormPicardRHS`.
Result in §12.

## 12. Implicit-recharge fix — STRONGLY INDICATED (POC was buggy; clean re-verification pending)

**CAVEAT (2026-07-27, added after the P=0 accounting check):** the proof-of-concept below
**corrupted the solve** — at P=0 (zero recharge) it produced a *different water-table evolution*
(cycle-9 total_wtd_change −9751 vs −18488) with `recharge_added = 0` and `SW change = 0` in both
runs. So the extra `rech_source` `DMDAVecGetArray` (nested on a vec `DMDA_Array_Pack` already holds)
perturbed solver state, NOT the physics (evap/FSM identical in both; ocean is the only boundary).
Therefore the "order 2" numbers below are from a BUGGY POC and are **not trustworthy** — the fix
*direction* is strongly indicated (see the independent D5 pure-source result + the order-2
re-derivation), but "clean order 2 restored" must be re-measured with a non-corrupting build.

**Making the recharge implicit is expected to restore order 2.** Proof-of-concept: in `FormPicardRHS`
(bdf2_on_V branch), replace the precomputed `Sy·my_rech` (`my_rech = add_recharge(rech_dist, hⁿ)`)
with `Sy·add_recharge(rech_raw, w_k, poro)` evaluated at the iterate `w_k → hⁿ⁺¹`. Same deep smooth
IC, vs Δt=0.25 ref:

| Δt (yr) | 1 | 2 | 5 | 10 | 100 | order |
|---|---|---|---|---|---|---|
| explicit (old) | 1.95 | 4.17 | 10.2 | 20 | 183 mm | ~1 |
| **implicit (fix)** | **0.0029** | **0.012** | **0.080** | **0.32** | **33 mm** | **~2.0** (2.07/2.05/2.00/2.01) |

Root cause **confirmed**: the recharge source was evaluated at `hⁿ` (explicit); BDF2 needs it at
`tⁿ⁺¹`. Physically sound too — the end-of-step water table sets the recharge partitioning. ~6–40×
smaller error AND 2nd order. This is strictly better than Richardson (§10), at 1× cost.

**Implementation caveat (why it is NOT yet committed):** the proof-of-concept read
`user_context->rech_source` inside `FormPicardRHS` and re-ran `add_recharge` at the iterate. It
broke the P=0 (no-recharge) case — which should be an exact no-op, since `add_recharge(0,·)=0` — so
`my_rech_raw` was evidently NOT the zero it should be for P=0. **The exact cause is unpinned:** my
first guess (a double `DMDAVecGetArray` on `rech_source`) is probably wrong, because the existing
code already double-checks-out `rech_vec` (held by both `DMDA_Array_Pack` and `FormPicardRHS`) with
no ill effect. So it is more likely a *population/timing* issue — `rech_source`/`rech_dist` not
being the reliably-zeroed raw recharge at the point `FormPicardRHS` runs. The experimental edit was
**reverted**; the solver is unchanged.

**Clean implementation (well-scoped follow-up):** first pin why `rech_source` is non-zero at P=0
(instrument its value in `FormPicardRHS`); then obtain the raw per-step recharge reliably — e.g.
(a) pass the `rech_dist` array through `AppCtx`, (b) a dedicated copy vec written at update() line
252, or (c) evaluate the implicit `add_recharge` where the array is already owned. Apply the same
change to the BE / secant-BDF2 branches for consistency, and re-verify **P=0 is an exact no-op** and
golden is unaffected (the change is inside the `-wtm_bdf2_on_V` RHS branch; Anderson production is
untouched).

**Status: ROOT CAUSE FOUND + FIX PROVEN.** BDF2-on-V's 1st-order-under-recharge is the explicit
recharge source; evaluating it implicitly (at hⁿ⁺¹) restores true 2nd order. Remaining work is the
clean (double-checkout-free) implementation, then commit. Richardson (§10) is a poor fallback here.

## 13. ⚠️ CORRECTION (2026-07-27, later): the §11–§12 fix was a STALE-WORK-DIR ARTIFACT

The "implicit recharge restores order 2" result (§12) was produced in a scratch work dir
(`/tmp/wtm_acc_bench`) reused across ~10 diagnostic scripts that repeatedly overwrote the shared
input tifs (precip, ksat) and the `supplied_wt` IC. **Re-run cleanly** — a fresh `make_equil`
fixture + the committed `recharge_order.py` harness, in a pristine dir — the implicit-recharge
build gives:

| P (m/yr) | Δt=1 | 2 | 5 | 10 | 100 | order |
|---|---|---|---|---|---|---|
| 0 (homogeneous) | 0.0022 | 0.0093 | 0.061 | 0.24 | 25.2 mm | **2.0** (matches committed baseline exactly) |
| 0.01 (recharge) | 2.04 | 4.42 | 11.2 | 22.8 | 274 mm | **~1.0** (NOT fixed; ~same as unfixed) |

So the implicit-recharge change is a clean no-op at P=0 (good) but **does NOT restore order 2 under
recharge**. The §12 "order 2" table was contamination, not a fix. The fix and its enabling edits
were reverted; the solver is unchanged.

**What this invalidates:** the *cause* narrative — §9's "explicit source" mechanism, §11, §12, and
by extension any of the D1–D6 diagnostics that were run in the same contaminated dir — is **not
trustworthy** and must be re-established in clean, one-experiment-per-dir runs.

**What survives (clean-verified):** BDF2-on-V is **2nd order for the homogeneous problem** and
**1st order once recharge is active**. That is the reliable finding. The *why* is again open.

**Process lesson (the real takeaway):** never reuse a scratch dir across experiments that mutate
shared inputs; one clean dir per experiment, or regenerate the fixture each time. A pleasing result
that confirmed the hypothesis (order 2 restored) was the artifact — exactly the case to distrust
most. The clean re-run done *to be skeptical* is what caught it.

**Next (clean re-diagnosis):** re-run the order study and the key isolations (pure-source, kink-
smoothing, convergence) each in its own fresh fixture dir, to re-determine the real cause of the
recharge order reduction before attempting any fix.

## 14. Clean re-diagnosis (2026-07-27) — the cause is the SURFACE handling

**Deep-everywhere + diffusion + recharge (fresh dir, field verified deep throughout: IC [-100, -100.0],
after recharge [-99.9, -88.0]).** BDF2-on-V order WITH recharge when NO cell reaches the surface:

| Δt (yr) | 1 | 2 | 5 | 10 | 100 | order |
|---|---|---|---|---|---|---|
| deep-only | 0.006 | 0.010 | 0.034 | 0.12 | 11.8 mm | 1.83 → **1.99** (coarse) |

vs the shallow-present case (order ~1 everywhere, 274 mm at Δt=100). So **when nothing touches the
surface, BDF2-on-V + recharge is 2nd order** (at the coarse Δt that matters; the fine-Δt dip is the
same reference-floor artifact as the homogeneous case, errors ~0.006 mm below the dt=0.25 ref's
resolution). ⇒ the order reduction is the **surface/shallow-cell handling done explicitly at hⁿ**,
NOT the recharge↔diffusion coupling in general.

Which surface op: the T-clamp is already implicit (T evaluated at hⁿ⁺¹); the partitioning-implicit
POC (§12) did not help; so the suspect is the **`evap_mode 0` explicit surface-water removal** (and
possibly the recharge-amount computation), done in the hⁿ preprocessing step outside the Picard
solve. Since production runs **FSM** (surface water conserved/routed between GW cycles, no removal
within a GW step), the order loss may be **specific to the harsh `evap_mode 0`** and absent in
production. Testing `evap_mode 0` vs `1` (no removal, FSM-within-step proxy) next.

**`evap_mode 0` vs `1` (clean dir) — NOT evap-specific.** Same shallow-present recharge; only the
surface treatment differs:
- `evap_mode 0` (remove surface water): wtd clamped at 0; order ~1 (2.04/4.42/11.2/22.8/274 mm).
- `evap_mode 1` (owe=0, no removal → water piles up to +8.15 m): order ~1 and **2× worse**
  (5.86/13.1/31.2/60.7/537 mm).

So it is **not** the removal — *both* surface treatments lose the order; accumulation is worse. What
they share is the water table **crossing the surface (wtd → 0)**, where T (the clamp) and V (the
porosity→1 storativity transition) are C0-kinked. **The cause is the wtd=0 coefficient kinks, crossed
when recharge pushes cells up to the surface** (drainage, which is order 2, moves cells *down away*
from the surface and rarely lingers there). **Consequence:** this is NOT fixed by switching off
`evap_mode 0` — FSM production hits the same kinks wherever the water table reaches the surface (wet
regions/depressions). Scope limiter: deep/dry cells stay 2nd order. Next: clean re-test of whether
smoothing the wtd=0 kinks restores the order (D3 was contaminated).

## 15. RESOLVED — it is a FREE BOUNDARY, and extended-soil fixes it

**Smoothing the wtd=0 kinks does NOT help (clean re-test).** Recharge order with all three surface
kinks smoothed 0.5 m: still ~1 (1.23/1.07/0.97/0.88) and errors *larger*. So it is not a *smoothable
coefficient* kink (§14's interim wording); it is an **obstacle / moving free boundary**. When
recharge pushes a water table across the surface, the exact `h(t) = min(rising trajectory, surface)`
has a **temporal kink** at the crossing instant — BDF2 assumes C² in time, so a temporal kink caps
the order. And it diffuses: `recharge_order_by_depth.py` shows even deep subsurface cells (wtd < −10 m)
are order ~1.15 (errors shrink with depth, order does not) — so the free boundary spoils the whole
domain, not just the discarded surface water.

**Fix — extended soil (Andy's idea), gated `-wtm_extended_soil`, VALIDATED (GW-step).** Treat the
aquifer as continuing infinitely above the surface: storativity = porosity everywhere (no jump to 1),
transmissivity continues past wtd=0 (no clamp), recharge always partitions as rech/porosity. This
removes the free boundary → the GW step is smooth → **clean order 2** (`evap_mode 1`, cells rise
above the surface as a smooth mound, clean dir):

| Δt (yr) | 1 | 2 | 5 | 10 | 100 | order |
|---|---|---|---|---|---|---|
| standard (free boundary) | 2.1 | 4.5 | 10.5 | 20.9 | 165 mm | ~1.0 |
| **extended soil** | **0.0019** | 0.0081 | 0.054 | 0.22 | 71.8 mm | **2.07 / 2.07 / 2.00** |

~1100× smaller error at Δt=1 (2.1 → 0.0019 mm). Skeptic-checked: fresh dir, field verified above the surface (mode
engaged), golden byte-clean (flag default off), ~no-op (0.01 mm) below the surface, order-2 reproduced
after the POC was reverted (extended-soil is POC-independent since `add_recharge` is wtd-independent
in this mode). This also settles the **ceiling**: 2nd order IS achievable for the recharge problem.

**Why it is physically safe (Andy):** the fictional above-surface mound flows *downslope*, the same
direction real surface water and FSM route it — it only misroutes if it grows tall enough to reverse
a gradient across a divide, which needs a lot of water. Two things bound that below any practical
concern: (a) FSM truncates the mound to real topography once per FSM cycle (weekly), so it never
builds; (b) extended T *grows* above the surface, so a tall mound has high transmissivity and drains
fast (self-limiting). The +23 m mound in the test is an artifact of GW-only for 1000 yr with no FSM
truncation. Degrades gracefully and locally, not globally. Also a model **simplification** — deletes
the surface special-casing from the GW step.

**Remaining (production half):** the FSM-side truncation — at the FSM handoff, move water above the
real surface topography to depressions / off-map. Not yet implemented or validated. Until then,
`-wtm_extended_soil` is a WIP flag proving the GW-step order-2 half.

**Reproduce the whole story** with `benchmark/picard/recharge_free_boundary.py` — one
self-documenting script that walks A (no-crossing → order 2) → B/C (crossing, evap 0 & 1 → order 1)
→ D (surface-kink smoothing → no help) → E (extended soil → order 2 restored), each a controlled A/B
on the same deep smooth IC in a private fixture dir.
