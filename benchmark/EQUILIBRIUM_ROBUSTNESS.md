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
`deltat` small (default `params.deltat/200`) and, via a Newton-iteration-controlled reject+retry
controller (commit f61d28a), GROWS dt after an easy step (≤ `dtc_easy_iters`), HOLDS when hard, and
REJECTS+shrinks+retries a non-converged step (`update()` returns −1 without committing; the rejected
step's budget accumulators are rolled back). Reject+retry is what survives both the dt overshoot into a
singular Jacobian AND the per-cycle FSM perturbation. On Esquibel 384k it completes 8 cycles **unattended**,
handling 3 rejects gracefully — where a grow-only ramp threw at cycle 4 and the adaptive *Picard* path
died at cycle 2 (`DIVERGED_MAX_IT`, Picard oscillation).

**Robust but not yet fast on stiff grids — the dt ceiling is a CONDITIONING limit, not FSM.** The
controller's dt plateaus at the safe overshoot ceiling (~0.02–0.03 yr on Esquibel) rather than ramping
large. More GW steps between FSM calls (maxiter 12 vs 3) raised the ceiling only modestly (0.02→0.03 yr,
with more rejects), so the cap is fundamental to Esquibel's fine, steep grid (900 cells/degree; T spans
many orders → very stiff Jacobian → low overshoot threshold), not primarily the FSM perturbation
(contrast Corsica, 120 cells/degree, which took dt = 1 yr from the Dupuit guess). So on steep fine grids
it reaches equilibrium via many small-ish steps — robust and unattended, but slow. The real speed lever
there is lifting the conditioning ceiling (a stronger preconditioner, or continuation on the *conditioning*
rather than just dt), not the FSM cadence.

### Methods to permit longer / more stable time steps (2026-08-03 investigation, on cropped Esquibel 900 cpd)
There are TWO distinct dt ceilings, and they need different levers:

**1. Linear-solve ceiling — preconditioner-limited (the Jacobian is ILL-CONDITIONED, not singular).**
Proof: at dt = 0.3 yr from the Dupuit guess the default GAMG fails at iteration 0 (`DIVERGED_LINEAR_SOLVE`),
but **MUMPS direct converges in 27 Newton iters** — the system is well-posed, GAMG just can't precondition
it. Levers, all raising the ceiling from GAMG's ~0.1 yr to ~0.3 yr:
  - **MUMPS** (`-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps`): dt≈0.3 yr, 27 iters.
    Robust; fine for REGIONAL grids (≲1M cells), does not scale to a global grid.
  - **Tuned HYPRE BoomerAMG** (`-pc_type hypre -pc_hypre_type boomeramg
    -pc_hypre_boomeramg_strong_threshold 0.7 -pc_hypre_boomeramg_relax_type_all SOR/Jacobi`): dt≈0.3 yr,
    43 iters. **SCALABLE** — the recommended preconditioner for stiff grids where GAMG stalls.
  - Default GAMG (Chebyshev smoother + smoothed aggregation) is poorly suited to this nonsymmetric,
    high-dynamic-range (T spans many orders) operator; ~0.1 yr.

**2. Nonlinear ceiling — globalization-limited.** Beyond ~0.3 yr from the cold Dupuit guess, NEITHER a
line search NOR a trust region (`-snes_type newtontr`) converges (both `DIVERGED_MAX_IT` at 100 iters):
the Newton *iteration* can't jump that far. The only lever is **continuation** (ramp dt small→large); the
nonlinear ceiling rises as the state warms toward equilibrium.

**Best combination = continuation + a strong preconditioner.** Continuation handles the cold phase; the
strong PC lets the steps grow larger once warm. Continuation+MUMPS ramped to 0.175 yr vs default-GAMG's
~0.05 yr. NB the iteration-based dt controller interacts with linear-solve accuracy: exact solves (MUMPS)
→ fewer Newton iters → the controller grows dt more; inexact AMG → more Newton iters → it holds. Raise
`-wtm_dtc_easy_iters` to let an AMG-preconditioned continuation ramp higher.

**The remaining limiter is the FSM–GW coupling, not the step size.** With FSM on, each cycle's surface-
water routing perturbs the GW state (~26 m inter-cycle changes persisted after 14 cycles on the crop), so
the system approaches a limit cycle rather than a clean fixed point. Reaching a *settled* equilibrium is
a GW↔FSM coupling question (how many GW steps per FSM call, or FSM under-relaxation), separate from the
solver's dt ceiling.

**Full-grid caveat + negative results (be honest about what does NOT help):**
- **On the full 384k grid the preconditioner advantage is muted** in a fresh run: MUMPS continuation
  plateaus ~0.02–0.04 yr over 8 cycles, ≈ GAMG, because the big cold domain stays *nonlinear-limited* for
  many cycles (MUMPS's higher *linear* ceiling only pays off once warm, as the small crop showed by cycle
  6 at 0.175 yr). So on a fresh fine equilibrium run the bottleneck is the **cold nonlinear phase**, and
  the lever is *warming* (continuation, or a warm start), not the linear solver.
- **Trust region** (`newtontr`): no help for the nonlinear ceiling (same `DIVERGED_MAX_IT` as line search).
- **Nonlinearity smoothing** (larger `-wtm_ksat/storativity_*_smoothing_width`): does NOT raise the
  nonlinear ceiling (dt = 1 yr fails at widths 0.1→3.0), and it shifts the near-surface physics — not a
  convergence lever.
- **Recharge / homotopy continuation** (ramp recharge 0→full at large dt): does NOT help — the ceiling is
  set by the *flux* Jacobian conditioning, which the source term doesn't change (10% recharge at dt = 1 yr
  still `DIVERGED_LINEAR_SOLVE`).
- **Most promising untested lever: grid sequencing** (solve a coarse grid to equilibrium — cheap, large dt
  — then interpolate up as a warm fine-grid start, skipping the cold nonlinear phase where the fine solve
  is stuck). Also untested: Levenberg–Marquardt Jacobian regularization (needs code).

**Bottom line for longer steps:** raise the *linear* ceiling with MUMPS (regional) or tuned HYPRE
(scalable) — real, single-step-verified (0.1→0.3 yr); handle the *cold nonlinear* phase with continuation.

### Better first guess + grid sequencing (2026-08-03) — and why the deeper limiter is the drainage timescale
Andy's idea: an analytical-ish first guess using SPATIALLY-VARYING transmissivity (the Dupuit guess is 1-D
constant-coefficient). Built + tested two routes:
- **Frozen-coefficient elliptic guess** — solve the LINEAR `Σ_faces T_face·(L/d)·(h_c−h_nbr) = R_c·A_c`
  with T frozen per cell (WTM's T at a depth estimate; varies via K, fdepth, depth), harmonic-mean faces,
  Dirichlet h=0 at ocean. In principle the 2-D analogue of Dupuit and much closer to equilibrium. IN
  PRACTICE the **free surface defeats it**: freezing T deep (small T) → the solve mounds to the surface →
  next iterate freezes T shallow (large T) → deep, a limit cycle; even damped Picard (ω=0.25) drifts
  surface-saturated. So a *cheap* frozen-T guess is not much better than Dupuit — the free-surface
  nonlinearity resists cheap approximation.
- **Grid sequencing** (coarse equilibrium → interpolate up → warm fine start) — did NOT unlock larger fine
  steps: the fine solve from the interpolated start still failed at dt = 0.3 yr. Two reasons: the coarse
  solve itself never reached a true equilibrium, and a warm start does not change the fine-grid *linear*
  conditioning that caps dt.

**The deeper finding (reframes the whole problem):** with **FSM OFF**, the pure GW continuation STILL does
not settle — per-cycle max|Δwtd| stays 10–25 m over 12 cycles. So the non-settling is NOT mainly the FSM
coupling; it is that the **groundwater equilibration timescale is ~decades** (`t ~ S·L²/T`), while the
conditioning-capped dt (~0.03–0.3 yr) advances only a few years of physical time per run. Reaching
equilibrium = marching through a decades-long drainage transient, which at the dt ceiling is ~100+ steps.
A better *guess* only helps if it is close to the true nonlinear equilibrium, which the cheap guesses are
not; and the dt ceiling caps physical-time progress regardless of the guess.

**So the real accelerators are the two we have not built:**
1. **Direct steady-state (dt→∞) elliptic solve** — skip the transient entirely and solve the equilibrium
   BVP `∇·(T(h)∇h)+R=0` directly. Hard: at dt→∞ the storage term vanishes, so the Jacobian is the pure
   free-surface elliptic operator (ill-conditioned, T spans orders) — needs a good guess + strong PC +
   trust region/continuation *on the elliptic problem itself*.
2. **FAS nonlinear multigrid** (`SNESFAS`) — the gold standard for elliptic equilibria: coarse levels give
   *global* corrections that kill the slow, large-scale drainage modes in O(1) work-units, independent of
   the domain size / drainage timescale. This is the principled fix for "many steps through a slow
   transient," and the biggest-payoff untested investment.
Absent those, the practical path is continuation + reject/retry + MUMPS/HYPRE: robust and unattended, but
~100 steps to equilibrium on a stiff fine grid because of the drainage timescale.

### Direct-analytical exploration + the free-boundary/storage insight (2026-08-03) — the key synthesis
Explored the "direct analytical" route Andy suggested, culminating in the **Kirchhoff / discharge-potential
transform**: define `Φ(wtd)=∫T(wtd')dwtd'` so `T∇h = ∇Φ + T∇topo` and the operator becomes a
CONSTANT-COEFFICIENT Laplacian `∇²Φ` (transmissivity absorbed into the pointwise Φ↔wtd map + a topo-drift
source). For WTM's piecewise T the integral is clean (`Φ = fdepth·T` in the exp regime, quadratic in the
linear regime, linear above surface). Elegant — and it DOES linearize the operator (SPD Laplacian,
multigrid-trivial). **But it does not tame the problem:** the topo-drift source `Σ G·e·(topo_c−topo_nbr)`
is O(90) per face while `R·A` is O(1e-4), so on steep terrain the water table is topography-following and
the frozen-drift Picard oscillates *bimodally* — cells split into "at surface" (T capped) and "very deep"
(T→0) at the wtd=0 free boundary — even under heavy under-relaxation (ω=0.12). The earlier frozen-T guess
failed the same way. scratchpad/kirchhoff.py, frozenT_guess.py.

**THE INSIGHT (ties the whole investigation together): the wtd=0 free surface is a FREE-BOUNDARY problem,
and the STORAGE term is what regularizes it.** Every direct/analytical/steady-state approach removes the
storage term and solves the bare elliptic problem — and every one hits the same surface/deep bimodal
instability at the free boundary where T switches regime. Time-stepping WORKS because the storage term
`S/Δt` adds a positive Jacobian diagonal (`∂storage/∂h = S/Δt > 0`) that keeps the operator diagonally
dominant and non-singular across the free boundary. This is the SAME mechanism as the dt ceiling: **the dt
ceiling is exactly where the storage regularization (`S/Δt`) becomes too weak to hold the free boundary**
(large Δt → small `S/Δt` → the free-boundary elliptic operator goes singular). Direct steady-state (dt→∞)
is therefore the *least* regularized, hardest case — which is why it, and the analytical transforms, fail.

**Consequence for FAS:** bare-steady-state FAS would hit the IDENTICAL free-boundary instability (its
smoother is a local Picard/Newton, which we've shown oscillates there). FAS must run *inside the
time-stepping* — as the linear solver / accelerator for each implicit (storage-regularized) step, or as a
pseudo-transient FAS — NOT on the bare equilibrium. That is the correct, and still substantial, next build
(scope with Andy). The cheaper immediate win remains: tuned HYPRE/MUMPS as the per-step linear solver in
the continuation, which raises the dt ceiling (0.1→0.3 yr) within the storage-regularized framework.

### Kirchhoff discharge-potential (built, `-wtm_kirchhoff`) + the conditioning reframe (2026-08-04)
Since the transmissivity's 7-order dynamic range is the suspected driver, the classic remedy is the
Kirchhoff transform: solve for the discharge potential `Φ = ∫T dwtd` (Φ ≈ fdepth·T) instead of the head,
so the exact chain-rule Jacobian `dF/dΦ = (dF/dh)/T` divides T back out. Fully implemented as an opt-in
change of variable (residual converts Φ→wtd→head; Jacobian column-scaled by 1/T; guess Φ(wtd) in/out).
It reaches the IDENTICAL equilibrium (verified 8.7e-8 m, quadratic convergence) — but **does not raise the
dt ceiling; it makes conditioning WORSE**, and a conditioning diagnostic showed *why the premise was
wrong*:

| at the Dupuit start | dt=0.3yr | dt=1yr | dt=∞ |
|---|---|---|---|
| **cond(J_wtd)** (plain head form) | **1.0e4** | **5.3e3** | 2.7e7 |
| cond(J_Φ) (Kirchhoff, volume-form) | 5.0e7 | 1.8e6 | 4.6e7 |

**The plain Jacobian is WELL-conditioned at finite dt (~1e4).** So the dt=1 yr failure (`DIVERGED_MAX_IT`,
80 iters, where cond is only 5.3e3) is **nonlinear** — the Newton *iteration* can't converge from the far
start — **not linear ill-conditioning.** The transmissivity range causes a hard *nonlinearity* (poor
Newton directions far from the solution), not a bad linear system. Kirchhoff is a *conditioning* remedy,
so it targets the wrong axis, and it makes cond 500–10000× worse (Φ spans the same 7 orders; the head-form
storage term contributes 1/T ~ 1e11 to the diagonal for deep cells). A volume-form residual cuts that
blow-up (1e11 → 5e7, MUMPS-solvable) but Φ is still far worse-conditioned than plain wtd, so it would be
runnable-but-not-better; not built. Kept opt-in as a documented dead-end (`-wtm_kirchhoff`, off by default).

**This refines the earlier "conditioning limit" language:** the GAMG→MUMPS/HYPRE win (0.1→0.3 yr) is
preconditioner *quality* on a T-heterogeneous but only moderately-conditioned operator (MUMPS handles cond
1e4 trivially; GAMG's aggregation just does poorly on it), and the dt ceiling proper is a **nonlinear**
far-from-solution limit. So the effective levers are the nonlinear-globalization ones we already have —
**good initial guess (Dupuit) + continuation** — and, to push further, trust-region or a homotopy on the
recharge/nonlinearity, *not* a conditioning transform.

### Homotopy / natural-parameter continuation — tried both axes; the dt-continuation already IS optimal (2026-08-04)
The reframe (obstacle = nonlinear far-from-solution) points to homotopy: deform an easy problem into the
hard one, tracking the solution. Tested the two physical axes:
- **Recharge homotopy** (R: 0→full, steady state at each): fails at the FIRST step. At R=0 the steady
  solution is h=0, i.e. the water table at **sea level (wtd=-topo, deepest)** → T at its tiniest (7-order
  exp floor) → the elliptic operator is nearly singular, so the first recharge increment needs enormous
  head changes. R-homotopy runs from the *worst* transmissivity regime toward the better one — backwards.
- **Nonlinearity homotopy** (ramp fdepth large→real, so T goes constant→exp; warm-start each): the easy end
  (large fdepth) converges in 8 iters, but each fdepth decrement shifts the equilibrium enough that the
  warm start is "far" for the large-dt Newton, so it stalls early (dt=1 yr fails at the first step; dt=0.1
  yr reaches only fmin=64 of 2.5). Finer steps + smaller dt help but that is just... the dt-continuation.

**Conclusion — the dt-continuation already IS the optimal continuation.** The difficulty is not the *path*
to equilibrium (which any homotopy controls) but that the equilibrium *problem* is nonlinearly stiff, so
Newton's basin is tiny and needs a *very close* next-guess. Only small-dt steps provide that closeness,
because the **storage term uniquely supplies both the free-boundary regularization AND the closest guess**
(the previous physical/pseudo-time state). R and fdepth homotopies don't — they move the solution without
supplying a comparably close guess. So the robust `-wtm_dt_continuation` (+ Dupuit guess + MUMPS/HYPRE) is
not just *a* method, it is the *right* continuation for this problem; the ~100-step cost is the intrinsic
price of a decades-long drainage transient through a stiff free-boundary nonlinearity, not a solver defect.

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
