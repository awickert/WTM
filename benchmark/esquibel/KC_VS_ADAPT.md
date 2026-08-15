# KCallaghan fixed-1-week vs. our adaptive TR-BDF2 — Esquibel

Two regimes, two different boosts (measured below):
- **Spin-up** (cold `wtd = 0` → equilibrium, KCallaghan's fixed-1-week workflow): the true equilibrium is the
  dt→∞ fixed point (order-independent), but at the practical `eq_tol` stop adaptive is both **faster** (2–3×
  at iso-accuracy) **and lands a more accurate equilibrium** (RMS 0.031 vs cc's 0.081 to truth) — cc's small
  fixed steps make it stop prematurely-in-accuracy. See the iso-accuracy curve below.
- **Transient** (a ±20 % P−ET step over a fixed horizon): 2nd-order TR-BDF2 vs cc's 1st-order backward-Euler
  gives a large **accuracy** boost (~20× lower RMS error) *and* ~2.4× fewer iterations.

The shipped adaptive is **one-knob and robust** (`eq_tol` only; `dt_tol` auto-derived, PI-damped,
ring-capped) — its value is not raw speed on a domain where 1 week happens to be a good fixed step, but that
you need not know the right step in advance.

## Why this comparison

The upstream v2.0.1 fixed-1-week backward-Euler is what a KCallaghan run uses to spin up. Rather than
PR our work into that code, we run **KCallaghan's *method* inside our corrected code** (the `cc` path =
fixed-dt backward-Euler matrix-free Anderson) against **our adaptive TR-BDF2** — so the comparison
isolates the *time-stepping*, with identical (corrected) physics, boundary conditions, and inner
tolerance on both sides.

## Setup

- Domain: Esquibel, 384k land cells, cold (`supplied_wt 0`), `fsm_on 1`, `deltat` base = 1 week.
- Common flags (all arms): `-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6`
  `-wtm_surface_exfiltration_to_runoff -wtm_eq_tol 0.01 -wtm_eq_metric frac`.
- Arms differ only in the integrator / adaptive flags (below).
- MPI ranks: N = 16, one shared `agsmall` node (**wall is a first-look number** — co-tenant noise
  possible; NOT `--exclusive`, which is deferred to finalized code. **SNES iterations are the clean,
  node-independent cost.**).
- Harness: [`kc_vs_adapt.sbatch`](kc_vs_adapt.sbatch). Raw per-run record:
  `results/kc_vs_adapt/summary.csv` (git-ignored, regenerable).

## Results — spin-up cost (N = 16)

| arm | flags added | SNES iters | wall (s) | stop cycle |
|---|---|--:|--:|--:|
| `kcallaghan_cc` (fixed 1-wk BE) | — | 9827 | 36.1 | 14 |
| `fixed_tr` (2nd-order, no adapt) | `-wtm_tr_bdf2` | 5133 | 37.7 | 19 |
| **one-knob adaptive (SHIPPED)** | `-wtm_tr_bdf2 -wtm_dt_adaptive` (dt_tol auto = 0.5) | **6458** | 52.5 | 20 |
| adaptive, *old* I-controller | `… -wtm_dt_tol 0.5` | 4251 | 32.7 | 14 |
| adaptive, `-wtm_dt_tol 2.0` (too loose) | — | never stops | — | 527+ (killed) |

- **The shipped one-knob adaptive beats KCallaghan's fixed-1wk ~1.5×** (9827 → 6458) and settles with no
  `dt_tol` to pick — `eq_tol` is the only knob (`dt_tol` auto-derived = `min(50·eq_tol, 0.5)`, PI-damped,
  ring-capped).
- The `-wtm_dt_tol 0.5` row (4251, ~2.3×) is the **old I-controller**: faster but FRAGILE — a resonance
  dead-band at `dt_tol` 0.2–0.25 never settled (84–99 cycles), and `dt_tol 2.0` ran 527 cycles without the
  eq-stop ever firing. Replaced by a **PI step-size controller** (kills the hunting) + the one-knob coupling;
  the 4251 → 6458 slowdown is the deliberate **price of robustness** on unfamiliar terrain.
- Decomposition: 1st→2nd order (`cc`→`fixed_tr`) is ~1.9× (9827→5133). On *this* domain a hand-picked
  fixed-TR-1wk actually beats adaptive — adaptive's value is that **you don't have to know** 1 week is the
  right step; it auto-finds a stable one anywhere, from one physical knob.

## Spin-up converges to the same equilibrium as cc

`adaptive_eq` vs `cc_eq` (both at `eq_tol=0.01`): **RMS = 0.050 m**, max 3.1 m (384 703 cells). The bulk
agrees to ~5 cm; the 3.1 m max is a few slow deep exp-T cells where *both* stopped short at `eq_tol=0.01`
(a shared stopping-tolerance artifact, not a disagreement). The *true* equilibrium is order-independent, but
at the practical `eq_tol` stop adaptive lands closer to it than cc (RMS 0.031 vs 0.081 to truth) — see the
iso-accuracy curve below; the spin-up boost is faster **and** more accurate, not speed-only.

## Transient (±20 % P−ET step, 100-wk horizon) — where 2nd order pays

The **accuracy** boost is a transient property. Error vs a fine (0.25-wk) TR reference:

| vs fine ref | cc_1wk RMS | adaptive RMS | cc worst | adaptive worst | its cc | its adpt |
|---|--:|--:|--:|--:|--:|--:|
| dry −20 % | 0.702 | **0.035** | 52.5 | 0.98 | 1219 | 497 |
| wet +20 % | 0.664 | **0.035** | 76.3 | 2.19 | 1247 | 542 |

Adaptive-TR is **~20× more accurate in RMS** (and ~50–75× at the worst cell) than cc's fixed-1wk, at
**~2.4× fewer iterations** — cc's 1st-order backward-Euler smears the transient; 2nd-order TR-BDF2 tracks it
to centimetres. Combined boost = *a bit of speed + an order of magnitude of accuracy*, in the regime that
matters for paleo time-evolution.

## Iso-accuracy equilibrium curve (`iso_accuracy.sbatch`, job 15845541)

The `eq_tol` stop is only a proxy — it measures per-cycle *change*, not accuracy-to-truth. Against a
tightly-converged truth (adaptive, `eq_tol=0.001`), cumulative SNES iters to reach an RMS-accuracy:

| RMS-vs-truth (m) | cc iters | adaptive iters | speedup |
|---|--:|--:|--:|
| ≤ 0.20 | 3327 | 1693 | **2.0×** |
| ≤ 0.10 | 5927 | 1986 | **3.0×** |
| ≤ 0.05 | **never** | 2878 | cc can't reach |
| **saturation** | **0.081 @ 9177 its** | **0.0315 @ 6458 its** | — |

- **The equilibrium speedup is 2–3× at iso-accuracy, not 1.5×.** The 1.5× (iso-`eq_tol`) understated it,
  because cc and adaptive reach *different* accuracies at the same `eq_tol=0.01` stop.
- **Adaptive reaches a strictly better accuracy** — RMS 0.0315 vs cc's 0.081 (2.6× closer to truth), in
  fewer iterations. cc *saturates coarse*: its small fixed 1-week steps make the slow deep exp-T cells
  crawl, so their per-cycle change falls below `eq_tol` while they are still ~0.08 m from truth — cc stops
  **prematurely in accuracy**. Adaptive's big steps drive those cells to genuine convergence.
- So on the spin-up the boost is **not** speed-only (an earlier statement, now corrected): it is faster to
  any accuracy *and* delivers a more accurate equilibrium.

## Note on wall time

Wall tracks iterations only loosely here (16-core shared `agsmall` node, co-tenant noise); the iteration
counts are the trustworthy comparison until an `--exclusive` timing pass on finalized code.
