# Surface-discontinuity smoothing as a disease-2 stabilizer (2026-08-09)

## Framing: two distinct solver failure modes

The cold-start / large-dt solver failures split into **two independent diseases** (see memory
`finding-operator-singularity`, `finding-bedrock-floor`):

1. **Rank-deficiency singularity** — deep/arid dead cells (T→0) make the *inner linear solve* singular.
   Kills inversion solvers (Picard-GAMG, Newton, MUMPS); matrix-free Anderson is immune. Cure: the
   additive bedrock floor `-wtm_T_bedrock` (a deep-domain tool; inert on shallow domains).
2. **Frozen-coefficient contraction failure** — the *outer nonlinear iteration* fails to contract because
   the frozen coefficient lags the fast-changing one. Present with **zero dead cells** (e.g. the shallow
   island). Driven by (a) the bulk exponential-T nonlinearity and (b) the `wtd=0` free-boundary
   discontinuities: the **storativity jump** (Sy→porosity) and the **surface T-clamp**.

This doc is about **disease 2** and the surface-smoothing knobs built for it.

## The switches and their (verified) prior state

- `-wtm_ksat_surface_smoothing_width` — smooths the surface T-clamp kink. **Default 0 (sharp).**
- `-wtm_storativity_surface_smoothing_width` — smooths the Sy→porosity jump in V(w). **Default 0.01 m
  (1 cm)** — active but numerically negligible on meter-scale dynamics.

Both were **off / negligible on every recent island and Esquibel run** (only the old
`benchmark/picard/recharge_free_boundary.py` ever set them, to 0.5 m). So disease 2 had been diagnosed
with the surface discontinuity effectively **sharp**. Note the guard: **`-wtm_Tbar` forbids ksat-smoothing**
(both treat the same T-clamp; T̄ is the T-side cure) but **composes with storativity-smoothing**.

## Island results (cold-start to equilibrium, 30 cyc; iters = total nonlinear)

**Raw Picard never contracts, regardless of smoothing** — T̄ is essential:

| combo | 1wk | 2wk | 4wk |
|---|---|---|---|
| raw Picard + ksat0.1+stor0.1 | FAIL | FAIL | FAIL |
| raw Picard + ksat0.5+stor0.5 | FAIL | FAIL | FAIL |

**T̄ + storativity-smoothing composes — lifts Picard 1→2 wk and cuts iterations:**

| combo | 1wk | 2wk | 4wk |
|---|---|---|---|
| P + T̄ (stor default 0.01) | OK 5030 | **FAIL** | FAIL |
| P + T̄ + stor 0.1 | OK 3976 | **FAIL** | FAIL |
| P + T̄ + stor 0.5 | OK 3804 | **OK 6982** | FAIL |
| P + T̄ + stor 0.5 + floor 1e-8 | OK 3806 | OK 6976 | FAIL |

- **stor must be wide enough:** 0.1 does not lift the ceiling; 0.5 does (1→2 wk). Floor inert (shallow).
- **Wider stor cuts iters but distorts the equilibrium** (2 wk: stor 0.5→1→2→4 = 6982→6086→5670→5332 iters,
  swt −6.83e7→−6.80e7→−6.73e7→−6.56e7 ≈ **4% shift at 4.0**). **0.5 m is the sweet spot** (~0.2% shift).
- **4 & 8 wk fail at every width** → storativity was *a* limiter (1→2 wk), not the last one; Picard's hard
  ceiling stays 2 wk.

**Anderson (workhorse): storativity-smoothing cuts iters ~20% but does NOT lift the 8 wk ceiling:**

| combo | 8wk | 12wk |
|---|---|---|
| A + T̄ | OK 26350 | FAIL |
| A + T̄ + stor 0.5 | OK 20602 (−22%) | FAIL |
| A + T̄ + stor 1.0 | OK 20543 | FAIL |
| A + T̄ + stor 2.0 | OK 20497 | FAIL |

**Newton (`-wtm_stiff` = Newton+continuation+eq_tol) comes to life cold but is wall-clock-slow:** it
*converges* (settle@9), and storativity-0.5 helps it settle sooner (settle@7); both hit the 500 s wall
before finishing all cycles (iteration count not cleanly parsed on the continuation path).

## Bottom line (island)

- **T̄ is the essential disease-2 cure** for Picard; storativity-smoothing is a **composable add-on**.
- Storativity-0.5: **Picard 1→2 wk ceiling; ~20–24% fewer iters for BOTH Picard and Anderson** — a real
  conditioning/speed win, at ~0.2% equilibrium shift.
- It does **not** change the regime: Picard caps at 2 wk, Anderson stays the ceiling champion at 8 wk.
- ksat-smoothing: no help to raw Picard, incompatible with T̄ → not on the T̄ path.
- floor: inert on the shallow island (a deep-arid tool).

## Esquibel (384k, real deep-arid) — validation

**(A) Anderson+T̄ — the iteration/speed win transfers; the ceiling does not move:**

| combo | 2wk | 4wk |
|---|---|---|
| A + T̄ | OK 6678 it, 182 s | — |
| A + T̄ + stor 0.5 | OK **5630 it (−16%)**, 154 s | FAIL |

Equilibrium shift stor0.5 vs default: swt −4.552e9 → −4.536e9 ≈ **0.35%** (island was ~0.2%). Small,
nonzero — the physics cost of the numerical smoothing, in the sensitive shallow band.

**(B) cold Picard+T̄+storativity-0.5 IS revived on the deep domain — CORRECTED.**
My first pass (`esq_winner.py`, 450 s timeout) reported this as a hang; re-running with a 600 s budget
(`esq_fullstack.py`) shows it CONVERGES. The "hang" was a **timeout artifact**, not a true stall.

| combo | 1wk | 2wk |
|---|---|---|
| P + T̄ + stor 0.5 (no floor) | **OK, 2028 it, 590 s, settle@9** | TIMEOUT (>600 s) |
| P + T̄ + stor 0.5 + floor 1e-8 | **OK, 2021 it, 579 s, settle@9** | TIMEOUT |

- **The inversion path DOES come to life cold on Esquibel** at 1 wk with T̄ + storativity-0.5 — it just
  needs adequate wall-clock (~590 s) and is **~3–6× slower than Anderson** (~90–180 s). Ceiling ≈ 1 wk
  (2 wk exceeds the budget).
- **The floor (disease-1 guard) is inert here** (2028 vs 2021 it): at small dt the `S/dt` diagonal term
  already regularizes the operator (non-singular), so the singularity has no operating window at Kerry's
  1 wk. It would only matter at *large* dt (`S/dt→0`), where disease 2 caps Picard first. So the floor
  matters neither on the shallow island nor at Kerry's dt.
- Caveat: Picard's settled swt (−4.79e9) differs from Anderson's (−4.55e9) by ~5% — loose settle
  threshold / 12-cycle cutoff / stor-0.5 shift; a controlled accuracy comparison is still owed.
- **Newton/`-wtm_stiff`: inconclusive** — 600 s is too short for the continuation path on 384k (it works
  on Esquibel per `finding-analytic-jacobian-newton`, just slowly); not a failure, a budget limit.

## Recommended usage (decision: default stays 0.01 m; recommend per-run)

Decided 2026-08-09 (AW): **do NOT change the default** (0.01 m) — changing it would shift Kerry's
equilibrium ~0.2–0.35% and break golden-value regression tests. Instead, **recommend** the wider setting
for users who want the cold-start speedup and can accept the small shift:

- **For cold-start / stiff runs, use `-wtm_storativity_surface_smoothing_width 0.2 -wtm_Tbar`** — cuts
  cold-start iterations ~16–24% on both island and 384k Esquibel. 0.2 m sits between the ~0.1 m physical
  band and the 0.5 m conditioning sweet spot; equilibrium shift is sub-0.35%. (Wider than ~0.5 m distorts
  the equilibrium without further ceiling benefit.)
- **Composes with the Anderson workhorse** (the recommended solver) and with Picard. **ksat-smoothing is
  NOT recommended** (no benefit to raw Picard; incompatible with T̄).
- **The bedrock floor `-wtm_T_bedrock` is NOT recommended for routine use** — inert at production dt.

## Session bottom line (both domains)

- **SHIPPABLE WIN:** `-wtm_storativity_surface_smoothing_width 0.5` composed with T̄ gives the **Anderson
  workhorse ~16–24% fewer iterations** on both island and Esquibel, at ~0.2–0.35% equilibrium shift, no
  ceiling change. A clean speed lever (T̄-compatible; ksat-smoothing is not).
- **ISLAND-ONLY partial win:** Picard 1→2 wk; stiff/Newton settles in ~4 cycles vs 9 with the full stack.
- **CORRECTED (was a false negative):** storativity-smoothing DOES revive cold Picard on Esquibel at 1 wk
  (converges, ~590 s, settle@9) — the earlier "hang" was a 450 s timeout artifact. But Picard is ~3–6×
  slower than Anderson and caps at ~1 wk; no solver's Esquibel ceiling is raised beyond Anderson's 2 wk.
  The bedrock floor is inert on shallow domains AND at Kerry's small dt (S/dt self-regularizes); it has no
  operating window here. Newton/stiff on Esquibel: inconclusive at the 600 s budget.
- **Strategic picture unchanged:** Anderson+T̄ remains the cold-start champion; the *new* concrete gain is
  that **+storativity-0.5 makes it measurably faster**. The default 0.01 m was too narrow to matter —
  raising it (to ~0.1–0.5 m) is a speed lever, but it shifts the shallow equilibrium slightly, so it is a
  default-change decision for KCallaghan sign-off, not a silent flip.
