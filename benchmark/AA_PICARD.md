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

**Conclusion:** bolting GAMG-preconditioning onto Anderson yields *GAMG's cost + a new fragility*, not
"large steps + robustness." This is the third independent confirmation that, for the stiff exponential
transmissivity, **matrix-free Anderson + T̄ is the sweet spot** and adding preconditioned linear solves
(Picard as the solver, Newton's exact Jacobian, or Picard as an NPC) does not help. Kept off-by-default as
a documented dead-end (cf. `-wtm_kirchhoff`); do **not** port. A different NPC (e.g. a robust
smoother, or NGMRES outer) might behave differently, but the per-iteration-cost barrier remains.
