# Robust convergence to equilibrium from a far initial guess

Reaching the steady-state water table from a *cold* start (`wtd = 0`, water table at the surface →
whole domain saturated) on stiff real terrain is a hard nonlinear problem. Every solver we have —
matrix-free Anderson, defect-correction Picard, even a true (FD) Newton with a `bt` line search —
oscillates or stalls from that far start, and *whether* it converges depends fragilely on `deltat`.
Tuning `dt` is not a robust solution. This note catalogs the standard methods for globalizing
nonlinear solvers / initializing far from the solution, and tracks what we've tried on WTM.

Motivating case: Kerry's Esquibel patch (SE Alaska; 451×853 ≈ 384k cells at `cells_per_degree 900`;
`run_type equilibrium`; `supplied_wt 0` cold start; `deltat` 0.2 yr). The GW solve is an elliptic
diffusion equation with a *coefficient* nonlinearity `T(h)` (transmissivity depends on the water
table), source terms (recharge/evaporation), an FSM surface-water coupling, and a free surface at
`wtd = 0`. See [[finding-kerry-picard-hang]].

## The core idea: two orthogonal levers
When the guess is far from the solution you either **(A) make the solver robust from far away**
(globalization) or **(B) make the guess closer** (better initialization). **Continuation bridges the
two** — it generates a sequence of nearby problems, each a good guess for the next. Robust production
solvers almost always combine A and B.

## I. Globalizing Newton (same Jacobian, converge from far)
- **Line search** (Armijo/backtracking `bt`, L2, critical-point): shrink the step to force residual
  decrease. Cheap, ubiquitous. *Tried (`bt`): stops the oscillation but stalls* — line search fails
  when the Newton *direction* itself is poor far from the solution.
- **Trust region** (dogleg, Steihaug-CG; `SNESNEWTONTR`): restrict the step to a radius where the
  quadratic model is trusted; grow/shrink by predicted-vs-actual decrease. Often succeeds where line
  search stalls. *Untried.*
- **Pseudo-transient continuation (PTC / SER; Kelley–Keyes)**: add a pseudo-time term `(x−x_old)/Δτ`;
  small Δτ far from steady state → diagonally dominant/robust, grow Δτ→∞ as the residual falls →
  recovers Newton. The *principled* "ramp dt." *A crude fixed-factor dt-ramp was tried — Δτ got stuck;
  a real SER controller with the pseudo-time term in the operator is different and untried.*

## II. Continuation & homotopy (deform an easy problem into the hard one)
- **Natural-parameter continuation**: walk a *physical* knob from an easy value to the target,
  re-solving from the previous solution. For us: ramp **recharge** from 0 (trivial baseline) to full,
  or ramp initial **saturation**, or **conductivity**. Physically clean; sidesteps `dt`. *Untried,
  strong candidate.*
- **Artificial-parameter homotopy**: `H(x,λ)=λF(x)+(1−λ)G(x)`, `G` easy; track λ=0→1. General but
  less intuitive.
- **Pseudo-arclength (Keller)**: only if the solution path has turning points/folds (unlikely for a
  monotone drainage steady state) — probably overkill.
- **Mesh / nested iteration (Full Multigrid)**: coarse → interpolate up as the fine guess. *Tried —
  the coarse solve itself diverges cold, so this needs a robust coarse solve (pair with A/B above).*

## III. Better initial guesses — physically-based (Andy's priority)
Shrink the distance directly using the actual physics. Cheapest and most robust in principle.
- **Frozen-coefficient / linearized elliptic solve**: fix `T` at a value (e.g., full saturated
  thickness), solve the *linear* `∇·(T∇h)+R=0` once with multigrid — always convergent, gives a
  physically-shaped, below-surface water table using the real recharge/conductivity/boundaries.
  *Untried; the natural first thing.*
- **Recharge–drainage (Poisson) guess**: constant-`T` limit → `∇²h = −R/T`, `h=0` at the ocean; the
  water table mounds by ~`R·L²/T` above sea level (`L` = drainage length). `wtd = min(h−topo, 0)`.
- **Dupuit–Forchheimer / hillslope analytical**: 1-D unconfined steady state, `h²` parabola between
  drainage boundaries; depth below surface scales with `(R/K)·L²`, `L` = distance to nearest
  drainage (a distance transform from the ocean/boundary).
- **Tóth subdued-replica heuristic**: water table = heavily-smoothed topography; deep under ridges,
  shallow in valleys. *Rough versions tried (uniform −20/−50, `0.1·topo`) — they oscillate: "roughly
  below surface" is not close enough; the guess must be near the true nonlinear equilibrium.*

## IV. Intrinsically robust nonlinear solvers
- **Nonlinear multigrid — FAS (Full Approximation Scheme; Brandt; `SNESFAS`)**: multigrid on the
  nonlinear residual; coarse levels give *global* corrections → robust, mesh-independent for elliptic
  problems. Gold standard; biggest investment.
- **Nonlinear preconditioning / composed solvers (Brune et al., "Composing Scalable Nonlinear
  Solvers")**: NGMRES- or Anderson-*accelerated* Newton, Newton left-preconditioned by Picard/
  nonlinear-GS, ASPIN. Routinely rescue "Newton stalls from far away." Runtime compositions in PETSc
  (`-snes_npc_*`, `SNESNGMRES`, `SNESCOMPOSITE`). *Cheap to test, high leverage.*
- **Anderson acceleration (Walker–Ni)**: WTM's matrix-free solver — undamped it fails on steep
  terrain (hence β=0.5) and cannot take big steps.
- **Quasi-Newton (Broyden, L-BFGS)**: approximate curvature without the Jacobian; cheaper, less
  robust than trust-region Newton.

## V. Regularizing the problem itself
- **Under-relaxation** (`x ← x + ω·Δx`, ω<1): crudest globalization; `bt` is its adaptive cousin.
- **Smoothing the nonlinearity**: *already in WTM* (C∞ tapers, smooth transmissivity) — a prerequisite.
- **Artificial diffusion / pseudo-time**: overlaps with PTC.

## WTM status
| method | tried? | verdict |
|---|---|---|
| Line search (`bt`) | yes | stalls from cold start (even with exact Jacobian + LU: 4.4e4→2.8e4 in 2 steps, then stalls) |
| Fixed dt-ramp (crude PTC) | yes | Δτ stuck; not robust |
| Coarse→fine (nested) | yes | coarse itself diverges cold |
| Rough physics guess (uniform/topo-scaled) | yes | not close enough (oscillates) |
| True Jacobian (FD-colored) | yes | *helps* → the Jacobian is not the blocker |
| **Analytic-Jacobian Newton (`-wtm_newton`)** | **yes** | **BUILT + FD-verified to 6.4e-8; quadratic convergence. From COLD wtd=0: still stalls (globalization, not Jacobian). From a DUPUIT guess: converges at equilibrium dt in 8 Newton iters, GMRES/GAMG scalable.** |
| Trust-region Newton (`-snes_type newtontr`) | yes | smoother decrease than `bt` but hits the SAME ~2.84e4 cold-start plateau; needs the guess too |
| Nonlinear preconditioning (NGMRES/ASPIN) | no | cheap, high-leverage |
| Proper PTC (SER, pseudo-time in operator) | no | principled "ramp Δτ" |
| **Dupuit initial guess + analytic Newton** | **yes** | **THE WORKING RECIPE — see below** |
| Linear / Poisson initial guess | no | cheaper cousins of Dupuit; untried |
| FAS nonlinear multigrid | no | most robust; biggest investment |

## Synthesis
The robust recipe is almost always **a good initial guess (III) + a globalized/composed solver
(I/IV)** — e.g. linear-solve guess → NGMRES-accelerated or trust-region Newton, or a recharge
continuation with Newton per step; FAS for the mesh-independent gold standard. The Jacobian (analytic
vs FD) is an orthogonal *speed* choice, settled last. The FD-Newton experiment already showed the
Jacobian is not the robustness blocker — **globalization + a physical initial guess is.**

## RESULT (2026-08-03): analytic Newton + Dupuit guess — the working recipe
Built the opt-in analytic-Jacobian Newton (`-wtm_newton`, [[finding-analytic-jacobian-newton]]) and
FD-verified it to 6.4e-8 on all terms. Findings, in order:
1. **Cold wtd=0 confirms the diagnosis.** Even with the EXACT Jacobian and an EXACT (LU) linear solve,
   Newton from cold stalls: `bt` line search 4.4e4→2.8e4 in 2 steps then flat; `newtontr` trust region
   decreases smoother but hits the same ~2.84e4 plateau. The plateau = cold start sitting inside every
   near-surface taper band simultaneously. **The Jacobian was never the blocker.**
2. **A physical guess closes it.** From a Dupuit mound guess (`wtd = −sqrt(R/K)·dist_to_ocean`,
   supplied as starting_wt) at the full equilibrium dt (1 yr), Newton converges in **8 iterations** with
   a textbook quadratic tail — where Anderson took 73 and Picard 415 from the same guess.
3. **Scalable + production-ready.** GMRES + GAMG (unsmoothed aggregation) preconditions the NON-symmetric
   Jacobian in 6–12 inner iters/step (mesh-independent-looking) — LU not required. Converges with BOTH
   the smooth T (exact Newton) and the piecewise production T (inexact Newton, same 8/6 iters); the
   smoothing widths are needed only to FD-verify, not to converge.

So the durable answer to cold-start equilibrium is **analytic Newton (`-wtm_newton`) + a Dupuit physical
initial guess + a scalable linear solver.** On SMALL grids that solver is GMRES/GAMG (8 Newton iters).

**Fine grid (Esquibel 384k) — SOLVED; the blocker was large-dt overshoot, not the preconditioner.**
The Jacobian is correct at scale too (quadratic convergence proves it). The initial "inner solve fails
at step 1" was a red herring about the Krylov method: at her dt (0.2 yr) BOTH GAMG *and MUMPS (a direct
solver!)* fail at step 1 — which means the step-1 **Jacobian is singular**, not that the preconditioner
is weak. Mechanism: at large dt the storage term `S/dt` on the diagonal is tiny, so the diagonal loses
dominance and the full Newton step from the far Dupuit guess **overshoots** into a singular-Jacobian
state. At **small dt (0.001 yr) the diagonal is dominant and Newton + GMRES/GAMG converges cleanly on
384k**: 3–5 Newton iters/step, quadratic tails, inner GAMG only **4–5 GMRES iters/step**
(mesh-independent — fully scalable, no MUMPS needed). The overshoot threshold from the Dupuit guess is
≈ dt 0.02–0.05 yr.

**The complete recipe: `-wtm_newton` (GMRES/GAMG) + a Dupuit initial guess + dt-continuation** — start
dt ≈ 0.005–0.01 yr, ramp up as the state warms toward equilibrium (warm + large dt is fine, as Corsica's
warm-start showed). This is pseudo-transient continuation. MUMPS (`-pc_type lu
-pc_factor_mat_solver_type mumps`) is a robust direct fallback where available.

**dt-continuation IMPLEMENTED (`-wtm_dt_continuation`, 2026-08-03).** Opt-in on the Newton path: starts
`deltat` small (default `params.deltat/200`) and grows it 1.5×/converged step (`-wtm_dtc_grow`),
persisting across cycles. On Esquibel 384k from the Dupuit guess it ramps dt 0.001→0.038 yr with **every
step converging** (4–12 Newton iters, GMRES/GAMG 4–5 inner iters — scalable), through the exact
early-phase that a fixed large dt could not survive. GROW-ONLY for now: it eventually overshoots the safe
dt ceiling (dt grew faster than the state warmed) and throws — next is reject+retry (needs a budget-
accumulator rollback, since `set_starting_values` accumulates per call) or a gentler ramp/cap.

### The recharge–dt coupling: a root-cause fix (the important finding)
Enabling *any* variable-dt path exposed a latent bug. Recharge is a per-step **amount** `= rate·dt`, but
`rech_dist` is baked once as `rate·params.deltat` (irf.cpp / WTM.cpp), and the residual scales only the
*flux* by the actual `user_context.deltat`. So when a variable-dt path shrinks dt below `params.deltat`,
the source stays at the full `params.deltat` amount while drainage shrinks → **the table is over-recharged,
faster and faster as the step shrinks** → instability. This is almost certainly why the earlier
adaptive-dt attempts were "problematic" (dt got stuck small / diverged): shrinking dt to fight stiffness
*amplified* the over-recharge. **Fix:** rescale `rech_dist` by `user_context.deltat/params.deltat` so
recharge and drainage scale together; the steady state is then dt-independent (`rate = drainage` at the
fixed point, dt cancels). Exactly 1.0 on every fixed-dt path → those are byte-identical (golden unchanged).
**Demonstrated (Esquibel 384k):** with the rescale the continuation ramps 9 steps; without it the identical
run diverges at step 3, and one 0.001-yr step over-recharges the table 6× more. The `-wtm_dt_adaptive`
Picard path shares this recharge code, so the fix likely rehabilitates adaptive stepping there too —
worth revisiting.

Remaining: (1) continuation reject+retry / gentler ramp so it doesn't overshoot the tail; (2) revisit the
adaptive Picard path now that recharge is dt-correct; (3) optionally auto-generate the Dupuit guess in-WTM.

## Verified formulas & Esquibel findings (2026-08)
**Literature (verified against sources):**
- **Haitjema & Mitchell-Bruker (2005) Water Table Ratio: `WTR = R·L² / (m·K·H·d)`** (`R` recharge,
  `L` distance between surface-water bodies, `K` conductivity, `H` saturated thickness, `d` terrain
  rise, `m` = 8 for 1-D / 16 for radial). **`WTR > 1` → topography-controlled (subdued replica);
  `WTR < 1` → recharge-controlled (mound, disconnected from topography).**
- **TOPMODEL (Beven & Kirkby 1979) depth: `z_i = z̄ − (1/f)·(TWI_i − λ)`**, `TWI = ln(a/tanβ)`,
  `λ` = mean TWI; transmissivity decays as `T = T₀·exp(−f·z)` — **identical to WTM's Fan `fdepth`, so
  `1/f = fdepth`.** WTM already carries the depth scale. Guess: `wtd_i = −(z̄ + fdepth·(λ − TWI_i))`,
  clamped below surface / above sea level.

**Esquibel measurements:**
- **`WTR = 0.011 ≪ 1` → strongly recharge-controlled** (steep, high-K, modest recharge). Confirms the
  regime and *explains the data*: the Dupuit *mound* guess converged; the Tóth *topo-replica* did not.
- **A good physical guess is only half the fix.** On the coarse patch, the Dupuit guess makes plain
  Picard converge at her dt. On the *fine* 384k-cell patch it does not sustain: the crude (shallower)
  Dupuit gets the **first** solve then diverges on cycle 2; the "more accurate" deeper TWI×`fdepth`
  guess (closer to the true near-sea-level mound) **diverges on solve 1**. So **guess accuracy does
  not predict solver convergence**, and the **cycle-2 divergence is the *solver*, not the guess** — no
  guess-tuning removes it.
- **Conclusion:** the physical guess (regime-aware TWI/Dupuit mound) is a validated *initial
  condition*, but on the stiff fine grid it must be **paired with a robust globalization** (trust-region
  or NGMRES-accelerated Newton), per the Synthesis above. Good guess + robust solver, together.

Sources: Haitjema & Mitchell-Bruker 2005 (*Groundwater* 43(6)); Beven & Kirkby 1979 (TOPMODEL);
Fan, Li & Miguez-Macho 2013 (*Science*).
