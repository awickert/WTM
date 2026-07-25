# The WTM groundwater step: equation, discretization, and the Picard linearization

**Date:** 2026-07-25
**Branch:** `picard-mg`
**Status:** LIVE — written alongside the semi-implicit Picard prototype.
**Companion:** `PICARD_MG_DESIGN.md` (the *why* and the experiment plan). This note is
the *math*: the continuous equation, its finite-volume discretization, and how the
nonlinear implicit step is recast as a Picard iteration whose inner solve is a
symmetric-positive-definite (SPD) elliptic system suitable for multigrid.

Notation follows the code: `h` is hydraulic head (`src` variable `x`), `wtd` is water-table
depth below the surface (`wtd = h − z_topo`), `z` is surface elevation (`topo`).

---

## 1. The continuous equation

WTM evolves the water table by a nonlinear, heterogeneous **groundwater diffusion**
(Boussinesq) equation. Per unit area, conservation of water in the saturated column is

$$
S(h)\,\frac{\partial h}{\partial t} \;=\; \nabla\!\cdot\!\big(T(h)\,\nabla h\big) \;+\; R,
$$

where

- $h(\mathbf{x},t)$ — hydraulic head $[\mathrm{m}]$ (the unknown; `x` in the solver),
- $S(h)$ — **effective storativity** $[-]$, the volume of water released per unit head
  change per unit area; head-dependent because it switches between specific yield
  (unconfined, water table in the pore space) and a much smaller confined value as the
  table crosses the surface (`updateEffectiveStorativity`),
- $T(h)$ — **depth-integrated transmissivity** $[\mathrm{m^2\,s^{-1}}]$, strongly
  head-dependent: it decays exponentially once the water table drops below the shallow
  soil layer (`depthIntegratedTransmissivity`, Fan et al. 2013 eqns S4/S6),
- $R$ — recharge source $[\mathrm{m\,s^{-1}}]$ (precip − evap, net of runoff), applied
  as a per-step head increment `rech`.

Both coefficients depend on the solution $h$, so the equation is **nonlinear**. The
nonlinearity is entirely in the *coefficients* $S(h), T(h)$ — the differential operator
stays second-order elliptic. This is the structural fact the whole method rests on
(§4): freeze the coefficients and the operator is linear and SPD.

### 1.1 Transmissivity (the strong nonlinearity)

With `shallow` $=1.5\,\mathrm{m}$ and $f$ = e-folding depth (`fdepth`), $K$ = `ksat`:

$$
T(wtd) =
\begin{cases}
0, & f \le 0,\\[2pt]
f\,K\,\exp\!\big((wtd+\text{shallow})/f\big), & wtd < -\text{shallow} \quad(\text{Fan S6}),\\[2pt]
K\,(wtd+\text{shallow}+f), & -\text{shallow}\le wtd \le 0 \quad(\text{Fan S4}),\\[2pt]
K\,(\text{shallow}+f), & wtd > 0 \quad(\text{capped; surface water goes to FSM}).
\end{cases}
$$

The exponential branch makes $T$ span **orders of magnitude** across the grid — the
source of the "strong nonlinearity" caveat in `PICARD_MG_DESIGN.md` §6.

### 1.2 Boundary conditions

Ocean cells (land mask $=0$) are **Dirichlet**: $h=0$ (both $z_\text{topo}$ and $wtd$ are
pinned to 0 there). The physical domain is assumed to be ringed by ocean, so every land
cell has four in-domain neighbors (the code never forms a stencil on a boundary cell).

---

## 2. Time discretization — backward Euler (fully implicit)

One WTM step advances the head from $h^{0}$ (start of step, `starting_wtd + topo`) to
$h$ (end of step) over $\Delta t$ (`deltat`). Backward Euler:

$$
S(h)\,\frac{h - h^{0}}{\Delta t} \;=\; \nabla\!\cdot\!\big(T(h)\,\nabla h\big) + R .
$$

Implicit in both $T$ and $S$ — unconditionally stable, which is what lets WTM take large
groundwater steps. Dividing by $S$ and moving terms, the per-cell residual the solver
drives to zero is (this is exactly `FormFunctionLocal`):

$$
F(h) \;=\; \underbrace{\big(u_{xx}+u_{yy}\big)}_{\nabla\cdot(T\nabla h)}\,\frac{\Delta t}{S(h)}
\;+\; h \;-\; \text{rech} \;-\; h^{0} \;=\; 0 ,
$$

where `rech` is $R\,\Delta t$ folded to a head increment and $h^{0}=$ `starting_wtd + topo`.

---

## 3. Space discretization — finite volume, 5-point stencil

Structured cell-centered grid (PETSc `DMDA`, `DMDA_STENCIL_STAR`). For a land cell
$(i,j)$ with spacings $\Delta x$ (`cellsize_EW`) and $\Delta y$ (`cellsize_NS`), the
divergence of the flux uses **harmonic-mean** interface transmissivities (correct for
a piecewise-constant, discontinuous coefficient — flux continuity across the face):

$$
T_{i+\frac12,j} \;=\; \frac{2}{\,1/T_{i,j} + 1/T_{i+1,j}\,}
\quad(\text{code: } e_E = 2/(T^{-1}_c + T^{-1}_E)),
$$

and likewise for W, N, S. The discrete diffusion operator is

$$
(u_{xx}+u_{yy})_{i,j} =
\frac{e_W\,(h_{i-1,j}-h_{i,j}) + e_E\,(h_{i+1,j}-h_{i,j})}{\Delta y^{2}}
+\frac{e_S\,(h_{i,j-1}-h_{i,j}) + e_N\,(h_{i,j+1}-h_{i,j})}{\Delta x^{2}} .
$$

(The code stores $1/T$ per cell — `my_T` — and takes harmonic means of the neighbors;
the $\Delta x/\Delta y$ pairing with EW/NS follows the source.)

---

## 4. The Picard linearization

### 4.1 Semilinear form

Collect the discrete step as a **semilinear system** — linear in $h$ once the
coefficients are evaluated at a frozen head $\tilde h$:

$$
\boxed{\,A(h)\,h \;=\; b(h)\,}
\qquad
A(h) \;=\; I \;+\; \frac{\Delta t}{S(h)}\,\big(-L_{T(h)}\big),
$$

where $L_{T}$ is the 5-point diffusion operator above with $T$ frozen, and

$$
b(h) \;=\; h^{0} + \text{rech} \;=\; \text{starting\_wtd} + \text{topo} + \text{rech}.
$$

This is algebraically identical to $F(h)=0$ from §2 — just with the linear-in-$h$ part
($I$ and $-L_T\,\Delta t/S$) on the left and the frozen part ($h^0+\text{rech}$) on the
right.

### 4.2 The iteration

**Picard** freezes the coefficients at the current iterate and solves the resulting
*linear* system for the next:

$$
A(h_k)\,h_{k+1} \;=\; b(h_k),\qquad k=0,1,2,\dots \ \text{until}\ \|h_{k+1}-h_k\|<\text{tol}.
$$

Each iteration evaluates $T(h_k), S(h_k)$ (frozen), assembles $A(h_k)$, and does one
**linear** solve. Because the frozen operator is a symmetric elliptic diffusion operator,
$A(h_k)$ is **SPD** (§4.4) — the canonical multigrid problem, solved with **CG + algebraic
multigrid (GAMG)** in $O(1)$ iterations independent of grid size.

**Why mesh-independent.** The nonlinearity is a *local coefficient* ($S,T$ depend on the
local head), not a change of differential order. So the **outer** Picard count depends on
the strength of the nonlinearity, not on the grid; the **inner** MG solve is
grid-independent by construction. Their product is grid-independent total work — unlike
matrix-free Anderson, whose iteration count grows with the grid (8→10→15→39 over
1000²→8000²).

### 4.3 PETSc mapping (`SNESSetPicard`)

PETSc solves $A(x)\,x = b(x)$ by defect correction. Verified against the PETSc 3.24
manual, the two user callbacks are:

| callback | computes | code |
|---|---|---|
| function (`SNESFunctionFn`) | the RHS $b(x)$ | `FormPicardRHS` |
| matrix (`SNESJacobianFn`)   | the operator $A(x)$ | `FormPicardOperator` |

PETSc forms the residual $A(x)x - b(x)$ internally and applies the outer accelerator
(`-snes_type nrichardson` = plain Picard; `anderson`/`ngmres` = Anderson-accelerated
Picard). The inner linear solve is `-ksp_type cg -pc_type gamg`.

The operator $A(x)$ **already existed** in the code as the SPD matrix `P` assembled in
`FormJacobianLocal` ("Symmetric Picard preconditioner: freeze T, average S between
neighbors"). Picard needs only $T(x)$, never $dT/dh$ — so it uses the **production
piecewise** $T$ directly (not the smooth $C^\infty$ blend the Newton Jacobian needed),
and converges to the *same* fixed point as today's Anderson residual.

### 4.4 The SPD operator, and the Dirichlet symmetry fix

Off-diagonal (coupling to a **land** neighbor, E shown), with storativity averaged
between the two cells:

$$
A_{i,j}^{E} = -\,e_E\,\frac{\Delta t}{\tfrac12\big(S_{i,j}+S_{i+1,j}\big)\,\Delta y^2},
\qquad
A_{i,j}^{\text{center}} = 1 - \sum_{d\in\{E,W,N,S\}} A_{i,j}^{d}.
$$

Symmetry of the off-diagonals: $e_E$ (harmonic mean) and $\tfrac12(S_c+S_E)$ are both
symmetric in $(c,E)$, so $A^{E}_{c}=A^{W}_{E}$. With the diagonal strictly dominant
(the $+1$ from the storage term), $A$ is **SPD** ⇒ CG-compatible.

**The one correctness refinement the CG solve forces.** Ocean cells are Dirichlet
identity rows ($A_{oo}=1$, $A_{o,\cdot}=0$). A land cell adjacent to ocean would, naively,
carry an off-diagonal $A_{L,o}\ne 0$ while the ocean row carries no return coupling
$A_{o,L}=0$ — an **asymmetric** matrix, which breaks CG. The fix is standard **symmetric
Dirichlet elimination**: for a land–ocean face,

- **keep** the conductance $e$ in the land cell's diagonal (the flux $e\,(h_L - h_o)$ still
  drains the cell), and
- **drop** the off-diagonal entry to the ocean cell, moving the known term $e\,h_o = e\cdot 0 = 0$
  to the RHS (it contributes nothing since $h_o=0$).

This restores $A_{L,o}=A_{o,L}=0$ (SPD) *and* reproduces the exact flux the Anderson
residual already applies (it drains land to the ocean with $h_o=0$). So the Picard solve
converges to the same physical fixed point; the pre-existing `P` matrix, which kept the
off-diagonal, was only ever a GMRES preconditioner where the asymmetry was harmless.

---

## 5. Summary of the correspondence to code

| math | code (`transient_groundwater.cpp`) |
|---|---|
| residual $F(h)=0$ (§2–3) | `FormFunctionLocal` (Anderson default path) |
| $b(h)=h^0+\text{rech}$ | `FormPicardRHS` (new) |
| $A(h)$, SPD, piecewise $T$, symmetric Dirichlet | `FormPicardOperator` (new) |
| $T(wtd)$ piecewise (§1.1) | `depthIntegratedTransmissivity` |
| $S(h)$ | `updateEffectiveStorativity` |
| harmonic-mean faces (§3) | `e_E,e_W,e_N,e_S` |
| outer Picard + inner CG/GAMG (§4.2) | `SNESSetPicard` + `-snes_type nrichardson -ksp_type cg -pc_type gamg` |

The default solver is unchanged (matrix-free Anderson); the Picard path is gated behind a
runtime flag (`-wtm_picard`). See `PICARD_MG_DESIGN.md` for the experiment that decides
whether it earns a place at scale.
