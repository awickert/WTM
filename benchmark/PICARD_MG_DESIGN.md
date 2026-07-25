# Design Note: Picard-preconditioned multigrid GW solve (mesh-independence prototype)

**Date:** 2026-07-25
**Branch:** `solver-optimization-2`
**Status:** DESIGN / prototype sketch. For a bounded experiment, not a commitment.
**Branch:** do this on a **new branch off `solver-optimization-2`** — it changes the core
solver, so keep it isolated from the validated distributed-dataflow work.
**Authors:** Andy Wickert + Claude

**Prior art to draw on:** GRLP (Wickert, the gravel-river long-profile model) already
prototypes a **Picard iteration for a nonlinear diffusion equation** — a close analogue of
this problem. Look there first for the iteration structure, the under-relaxation/damping
that tames a strong nonlinear diffusivity, and the convergence/stopping criteria; port the
pattern rather than reinvent it.

Sketch of a semi-implicit **Picard iteration** with a **multigrid** inner solve for the
groundwater component — the one credible route to a *mesh-independent* solve, and the
correct reframing of the multigrid lever that failed as a nonlinear Newton solve. See
`SOLVER_NOTES.md` (the clean `-O3` study) and the multigrid experiment for the motivation.

---

## 1. Motivation

The clean scaling study settled that the per-core solve is at parity; the only solver
lever left that could matter *at scale* is **mesh-independent convergence**. The current
default — matrix-free Anderson on the full nonlinear residual — is cheap per iteration but
its iteration count **grows with the grid** (8 → 10 → 15 → 39 across 1000² → 8000²). A
direct Newton+GAMG attempt was **~47× slower** and *not* mesh-independent, for three
reasons: the Jacobian was assembled + GAMG-set-up every Newton step; the Jacobian used a
*smooth* T while the residual used *piecewise* T (inconsistent → no quadratic convergence);
and matrix-free Anderson is a very cheap baseline.

**Picard fixes all three** and is mesh-independent by construction.

---

## 2. The method

WTM's implicit backward-Euler step is, per cell,
`f = (uxx + uyy)·dt/S + x − rech = b`, where the discrete diffusion `(uxx+uyy)` carries a
harmonic-mean transmissivity `T(x)` and `S = S(x)` — both functions of the new head `x`.
Group it as a **semilinear system**:

    A(x) x = b(x),   with   A(x) = I + (dt/S(x))·(−L_{T(x)})

where `L_{T}` is the 5-point diffusion operator with T frozen. **Picard iteration** is

    solve  A(x_k) x_{k+1} = b(x_k)   with T, S frozen at x_k;  repeat until converged.

Each step is a **linear, symmetric (SPD), elliptic** solve — the canonical multigrid
problem. Solve it with **CG + algebraic multigrid (GAMG)** or geometric MG on the DMDA:
mesh-independent, ~O(1) iterations regardless of grid size.

**Mesh-independence argument.** The nonlinearity here is a *local coefficient* (T, S vary
with head), not a change of differential order. So the **outer Picard count depends on the
nonlinearity, not the grid**, and the **inner MG solve is grid-independent**. Their product
is grid-independent total work — versus Anderson's growing count.

---

## 3. Why Picard beats the Newton attempt

- **SPD operator → CG + GAMG** (symmetric, cheap), not GMRES on a nonsymmetric Jacobian.
- **No T-form inconsistency.** Picard needs only `T(x)` itself, never `dT/dh`. So it uses
  the **production piecewise Fan form directly** (`depthIntegratedTransmissivity`) and
  converges to the *same fixed point* as today's Anderson — the whole smooth-vs-piecewise
  Jacobian inconsistency that sank Newton simply does not exist. The `depthIntegrated‑
  TransmissivitySmooth` form is no longer needed on this path.
- **Robust:** no line search; Picard is globally convergent for this class.
- **Amortized setup:** `A` assembles once per outer iteration (T frozen); since T changes
  slowly near convergence, the GAMG hierarchy can be **lagged/reused** across outer
  iterations (`-snes_lag_preconditioner`, `-snes_lag_jacobian`).

**The operator already exists.** `FormJacobianLocal` (`transient_groundwater.cpp`) already
assembles a matrix `P` documented as *"Symmetric Picard preconditioner: freeze T, average S
between neighbors"* — that **is** `A(x)`, SPD and GAMG-compatible. The prototype reuses it
(switching its T from smooth to piecewise, per above).

---

## 4. Implementation sketch (PETSc-native)

Use `SNESSetPicard`, which drives exactly `A(x_k) x_{k+1} = b(x_k)` and can be accelerated
by the *same* Anderson machinery already in the code:

1. **Factor the `A(x)` assembly out of `FormJacobianLocal`** into a standalone
   `FormPicardOperator(SNES, Vec x, Mat A, Mat P, ctx)` that fills the frozen-T SPD stencil
   (the existing `P_east/west/north/south/center` construction) using **piecewise T**.
2. **`FormPicardRHS(SNES, Vec x, Vec b, ctx)`** = the recharge/boundary right-hand side
   (`b(x)` — the `rech` term plus the Dirichlet/ocean contributions; reuse `FormRHS`).
3. Wire it: `SNESSetPicard(snes, r, FormPicardRHS, A, P, FormPicardOperator, &ctx)`.
4. **Inner solve:** `-ksp_type cg -pc_type gamg` (algebraic MG on the SPD `A`), or
   `-pc_type mg` (geometric MG on the DMDA — the LINEAR operator coarsens by Galerkin, so
   the FAS "level-aware residual" blocker does *not* apply here).
5. **Outer acceleration:** `-snes_type nrichardson` = plain Picard; `-snes_type anderson`
   (or `ngmres`) = **Anderson-accelerated Picard** — few outer iterations *and* the
   mesh-independent inner solve. This reuses the existing `-snes_anderson_m` default.

All runtime-selectable; keep matrix-free Anderson as the default and gate this behind flags
(and a config/`CreateSNES` branch) so nothing changes for production until it's proven.

---

## 5. The decisive measurements (the whole point of the prototype)

On the synthetic `run_type test` sweep (`benchmark/scaling`), at grids 1000² … 8000²:

1. **Outer Picard iterations vs grid** — is it *flat*? (The make-or-break number.)
2. **Inner CG+GAMG iterations per outer step vs grid** — also flat? (Confirms MG works.)
3. **Total wall-to-convergence vs current matrix-free Anderson** — and **where the
   crossover is**: expected Anderson-wins at small grids, Picard-MG-wins at large.
4. **Equilibrium fixed point matches** current Anderson to solver tolerance (it should —
   same piecewise-T fixed point; the golden refs should hold within tol, not bit-identical).
5. **Per-rank memory footprint** (assembled `A` + MG hierarchy + CG vectors) vs matrix-free
   Anderson — via `-memory_view`. Confirms the semi-implicit memory cost is DMDA-distributed
   (falls with ranks), not a rank-0 burden (§6).

---

## 6. Risks / open questions

- **Outer count could be large.** WTM's T spans orders of magnitude (exp decay), a *strong*
  nonlinearity → Picard's linear convergence rate may be poor (many outer iterations, or
  needing damping / a line search on the Picard update). Grid-*independent*, but the
  constant matters for the crossover. Anderson acceleration is the main mitigation.
- **Per-iteration cost.** Each outer step is a full MG solve (assemble + setup + cycles) vs
  Anderson's single cheap residual eval. Mesh-independence must beat that cheapness — a
  scale-dependent crossover, which is exactly what measurement (5.3) resolves.
- **Memory cost vs matrix-free Anderson (Andy's constraint).** Assembling `A` + the GAMG
  hierarchy (coarse operators ~1.5–2× the fine matrix) + CG work vectors is O(N) but with a
  *larger constant* than matrix-free Anderson, which stores only ~`m` vectors and no matrix.
  This is the classic semi-implicit memory cost, and on a single node — where rank-0 `arp`
  is already ~50 GB at 8000² — it could be the binding constraint. **Mitigant, and the key
  point:** the operator and its MG hierarchy are DMDA-**distributed** (each rank holds only
  its subdomain's rows), *unlike* the rank-0 `arp`. So this memory spreads across ranks/nodes
  and is relieved by the very memory-splitting lever we already want — multi-node +
  distributing rank-0 `arp` (`DISTRIBUTED_ARP_DESIGN.md`). At single-node scale, budget the
  added footprint; at multi-node scale it should be affordable. Measure the per-rank
  matrix+hierarchy footprint alongside the timing in the prototype.
- **Under MPI.** GAMG and geometric MG both parallelize, and the operator is DMDA-based, so
  this composes with the distributed solve — but re-confirm cross-rank consistency (the
  golden/mpi suites) since it's a new solve path.
- **Storativity nonlinearity.** `S(x)` is also lagged; check it doesn't dominate the outer
  rate (it's a milder nonlinearity than T).

---

## 7. Phasing

1. **Prototype (days, not weeks):** factor `FormPicardOperator` (piecewise T) + RHS, wire
   `SNESSetPicard`, run the sweep with `-snes_type nrichardson -ksp_type cg -pc_type gamg`.
   Answer measurement 5.1/5.2 — *is it mesh-independent?* If no, stop here.
2. **Accelerate + tune:** `-snes_type anderson`, preconditioner lagging, geometric-vs-
   algebraic MG. Find the crossover grid vs matrix-free Anderson (5.3).
3. **Decision:** if the crossover is below the grids you run (or clearly below global 30″),
   make it the default at/above that size (config-selectable); else shelve with the numbers
   recorded. Either way the result retires or confirms the mesh-independence lever with data.

---

## 8. Relationship to the current solver

This does **not** replace Anderson wholesale — matrix-free Anderson stays the default and
wins at small/medium grids. Picard-MG is a *scale* play: it earns its place only where
Anderson's growing iteration count and the per-cell work outrun the MG solve's fixed cost —
i.e. the same regime (global / very large grids) where distributing the rank-0 `arp`
(`DISTRIBUTED_ARP_DESIGN.md`) also becomes necessary. The two are complementary levers for
the massive-scale goal.
