# Design Note: BDF2-on-V loses 2nd order under recharge — diagnosis & fix plan

**Date:** 2026-07-26
**Branch:** `bdf2-adaptive-dt`
**Status:** DIAGNOSED (D1–D4 run 2026-07-26). Cause = the recharge *source's* temporal
integration is 1st-order (NOT scaling, NOT kinks). Exact code line unresolved by static analysis
(needs a manufactured-solution test). Fix routes in §8; Richardson is the safe default.
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
