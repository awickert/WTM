# KCallaghan fixed-1-week vs. our adaptive TR-BDF2 — Esquibel spin-up

**Experiment type: SPIN-UP** (cold start `wtd = 0` → equilibrium), *not* a transient run. This is
KCallaghan's fixed-1-week equilibrium workflow, and the regime where the adaptive controller wins
(the island cold start showed the same ~2.6× before this). The metric is **cost to reach
equilibrium**; the equilibrium itself is the dt→∞ fixed point, so all arms converge to the same
water table (each stopped by the same per-cycle `frac` criterion).

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

## Results (N = 16, job 15833623)

| arm | flags added | SNES iters | wall (s) | stop cycle |
|---|---|--:|--:|--:|
| `kcallaghan_cc` (fixed 1-wk BE) | — | 9827 | 36.1 | 14 |
| `fixed_tr` (2nd-order, no adapt) | `-wtm_tr_bdf2` | 5133 | 37.7 | 19 |
| `adaptive_tr_0p5` | `-wtm_tr_bdf2 -wtm_dt_adaptive -wtm_dt_tol 0.5` | 4251 | 32.7 | 14 |
| `adaptive_tr_2`  | `-wtm_tr_bdf2 -wtm_dt_adaptive -wtm_dt_tol 2.0` | never stops | — | 527+ (killed) |

**`dt_tol=2.0` is too loose for a spin-up:** the steps stay healthy (~25/cycle, few rejects), but the
looser per-step tolerance leaves a residual per-cycle wobble that keeps >0.1 % of cells above `eq_tol`, so
the `frac` equilibrium stop **never fires** — it ran 527 cycles (~527 model-years) without settling and was
cancelled. There is a sweet spot: `dt_tol` must be tight enough that the per-cycle change can fall below the
stop threshold. `0.5` is near-optimal here (stops at cycle 14, the 2.3× win); `2.0` never converges. (This
is a *spin-up* constraint — a transient run has a fixed horizon and no eq-stop, so a looser tol is fine
there.)

## Verdict

- **Adaptive TR-BDF2 beats KCallaghan's fixed-1-week by ~2.3×** on iterations (9827 → 4251), matching
  the island's ~2.6×.
- The decomposition shows where it comes from: going 1st-order → 2nd-order (`cc` → `fixed_tr`) already
  cuts iterations ~1.9× (9827 → 5133); adaptivity adds the rest.
- Wall tracks iterations loosely here (shared-node noise); the iteration counts are the trustworthy
  comparison until an `--exclusive` timing pass on finalized code.
