# Anderson-accelerated GAMG-Picard (`-wtm_aa_picard`) — experimental, NEGATIVE result

## Idea

Combine GAMG-preconditioned Picard's large-step power with Anderson's oscillation-damping, on a
well-scaled residual (so we avoid the volume-form residual plain Anderson diverges on). Realized with
PETSc **nonlinear preconditioning**: the OUTER solver is `SNESANDERSON` on the head-form residual
(`FormFunctionLocal`); the GAMG-Picard solve (`FormPicardOperator`/`FormPicardRHS`, `SNESSetPicard`) is
attached as the outer's **nonlinear preconditioner** (`SNESGetNPC`), one sweep per outer step. So each
Anderson iterate is a full multigrid Picard update that Anderson then mixes/damps.

## Wiring

`CreateSNES`: `-wtm_aa_picard` ⇒ force the Anderson main path (head-form residual), allocate the Picard
operator (`picard_A`/`picard_r`) for the NPC, instantiate the NPC (`SNESGetNPC`), and default it to a
defect-correction Picard (`-npc_snes_type newtonls`, `-npc_snes_linesearch_type basic`,
`-npc_snes_max_it 1`) with a **GMRES** inner solve (`-npc_ksp_type gmres`) + unsmoothed GAMG. `update()`
registers the NPC's Picard callbacks each solve. NPC side = **left** (`PC_LEFT`). Off by default.

Two settings were needed just to make it run:
- **GMRES, not CG, for the NPC inner solve.** The GAMG preconditioner comes out slightly *indefinite*
  here, so CG bails with `DIVERGED_INDEFINITE_PC` (even with unsmoothed aggregation). GMRES tolerates it.
- **Left nonlinear preconditioning.** `PC_RIGHT` diverged cold (`DIVERGED_INNER`); `PC_LEFT` converges.

## Result — it does not beat plain Anderson + T̄

Island (warm 2× perturbation and cold-start-to-equilibrium), vs `-wtm_anderson -wtm_Tbar`:

| regime | Anderson + T̄ | AA-Picard + T̄ |
|---|---|---|
| warm 2× step | converges, cheap (matrix-free) | converges but **expensive** (~160 iters, a GAMG solve every iteration) |
| cold → equilibrium (1 wk) | robust (ceiling 8 wk), ~2 s | **FAILS** — diverges mid-drainage (~30 s); NPC sweeps 1/3/5 all fail |

**Why it fails:** the nonlinear preconditioner (Picard) can *hard-fail* (`DIVERGED_INNER` — its inner
GMRES/GAMG diverges at a stiff cold state), which aborts the whole outer solve. Plain matrix-free Anderson
has no such inner solve and so no such failure mode. And every AA-Picard iteration pays the GAMG solve —
the very per-iteration cost that made Picard ~25× slower than Anderson at scale.

## Warm-transience mechanics (the honest, useful part)

Digging into warm-start transience (supplied equilibrium + 2× perturbation, island) resolves *how*
Anderson-accelerated Picard behaves, in three regimes (outer nonlinear iters / wall, per warm ceiling sweep):

| dt | plain Picard+T̄ | AA-Picard+T̄ | plain Anderson+T̄ |
|---|---|---|---|
| 1 wk (Picard easy) | 300, 2 s | 378, 3 s | 549, 1 s |
| 2 wk | 389, 2 s | 705, 5 s | 747, 1 s |
| **4 wk (Picard oscillating)** | **6358, 25 s** | **759, 5 s** | 1110, 1 s |
| 8 wk | FAIL | FAIL (NPC hard-fails, 0 it) | 3193, 1 s |

1. **Where plain Picard is easy (1–2 wk): Anderson slightly HURTS** (378 vs 300) — mixing overhead with no
   oscillation to damp; the acceleration is dead weight when the base iteration already converges in ~2 steps.
2. **Where plain Picard OSCILLATES (4 wk, near its ceiling): Anderson works beautifully** — it damps the
   frozen-coefficient thrashing from **6358 → 759 iters (~8×), 25 s → 5 s (~5×)**. The least-squares over the
   residual history cancels the oscillation. It even beats plain Anderson on *iteration count* there
   (759 < 1110) — GAMG-preconditioning makes each step more productive, so fewer are needed.
3. **Where plain Picard FAILS (8 wk): AA-Picard fails too** — Anderson damps the *oscillation*, but the weak
   link moves to the **GAMG linear solve of the stiff operator**, which hard-fails (`DIVERGED_INNER`);
   Anderson cannot rescue an inner solve that errors out. So the ceiling stays at Picard's 4 wk.

**So AA-Picard is a "damped Picard": ~8× better than raw Picard near the ceiling, but it inherits Picard's
two liabilities** — a GAMG solve every iteration (5 s vs plain Anderson's 1 s at 4 wk) and the GAMG-solve
failure at 8 wk (ceiling 4 wk vs plain Anderson's 8 wk). The deep reason plain Anderson wins: **matrix-free
Anderson never inverts the stiff operator** (it accelerates the residual directly), whereas AA-Picard's GAMG
step does — and inverting `A(x)` is exactly what is expensive per step and what fails at 8 wk. Here
**preconditioning is the liability, not the asset**: it buys a per-step iteration reduction (759 < 1110) that
the per-step cost more than eats, plus a lower ceiling. Anderson's *acceleration* is the load-bearing part
(it damps oscillation superbly, preconditioned or not); for this stiff exponential the *unpreconditioned,
matrix-free* map is both cheaper and more robust, so `-wtm_anderson -wtm_Tbar` is the right realization of
"Anderson-Picard."

**Conclusion:** bolting GAMG-preconditioning onto Anderson yields *GAMG's cost + a new fragility*, not
"large steps + robustness." This is the third independent confirmation that, for the stiff exponential
transmissivity, **matrix-free Anderson + T̄ is the sweet spot** and adding preconditioned linear solves
(Picard as the solver, Newton's exact Jacobian, or Picard as an NPC) does not help. Kept off-by-default as
a documented dead-end (cf. `-wtm_kirchhoff`); do **not** port. A different NPC (e.g. a robust
smoother, or NGMRES outer) might behave differently, but the per-iteration-cost barrier remains.
