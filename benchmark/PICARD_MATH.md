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
**linear** solve. The frozen operator is elliptic diffusion; after a diagonal row-scaling
(§4.4) it is symmetric, hence **SPD** — the canonical multigrid problem, solved with **CG +
algebraic multigrid (GAMG)** in $O(1)$ iterations independent of grid size.

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

PETSc forms the residual $A(x)x - b(x)$ internally. Defect-correction Picard is a
**modified Newton** iteration whose "Jacobian" is the *frozen operator* $A(x)$ (not the
true Jacobian): each outer step solves $A(x_k)\,\delta = -(A x_k - b) $, i.e.
$A(x_k)\,x_{k+1}=b(x_k)$. So the OUTER solver must be a Newton type (`-snes_type newtonls`
with `-snes_linesearch_type basic` for the plain full-step update) — **not** `nrichardson`,
which would only do $x\leftarrow x-\lambda F$ with no linear solve. The inner solve is
`-ksp_type cg -pc_type gamg` (verified on PETSc SNES `ex15`, fd/mf_picard). Picard needs
only $T(x)$, never $dT/dh$, so it uses the **production piecewise** $T$ directly (not the
smooth $C^\infty$ blend the Newton Jacobian needed) and converges to the *same* fixed point
as today's Anderson residual (confirmed to $\sim6\times10^{-8}$ on the golden `below_ground`
case).

### 4.4 The SPD operator: centre-storativity with row-scaling

The production residual (§2) divides the **whole** flux divergence by the **centre**
storativity $S_c$. Writing that per row gives the natural operator

$$
N_{i,j}^{E} = -\,\frac{\Delta t}{S_{i,j}}\,\frac{e_E}{\Delta y^2},
\qquad
N_{i,j}^{\text{center}} = 1 - \sum_{d} N_{i,j}^{d}.
$$

This is the correct discretization, but it is **nonsymmetric**: $N^{E}_{c}\propto 1/S_c$
while $N^{W}_{E}\propto 1/S_E$, and $S_c\neq S_E$ where storativity varies. (An early
version used a *face-averaged* $\tfrac12(S_c+S_{\rm nbr})$ to force symmetry — but that
solves a **different** equation and converged to a fixed point off by up to ~15 m from
Anderson. Correctness requires centre $S$.)

**Row-scaling fixes it.** Multiply each row $c$ by $S_c>0$ — which leaves the solution
unchanged — and the $1/S_c$ that broke symmetry is cleared, exposing the symmetric flux
term $\Delta t\,e$:

$$
A_{i,j}^{E} = -\,\frac{\Delta t\,e_E}{\Delta y^2},
\quad
A_{i,j}^{\text{center}} = S_{i,j} - \sum_{d} A_{i,j}^{d},
\qquad
b_{i,j} = S_{i,j}\,\big(h^0_{i,j} + \text{rech}_{i,j}\big).
$$

Now $A^{E}_{c}=A^{W}_{E}=-\Delta t\,e_E/\Delta y^2$ (the harmonic mean $e_E$ is symmetric in
the pair), and the diagonal $S_c+\sum \Delta t\,e/h^2$ is strictly dominant ⇒ $A$ is
**SPD** ⇒ CG-compatible. The matching $S_c$ factor on the RHS (`FormPicardRHS`) cancels the
scaling, so $A(x)x=b(x)$ has exactly the Anderson fixed point. $S_c$ depends on the head, so
$b$ genuinely depends on $x$ (frozen at the outer iterate) — unlike the unscaled RHS.

**The Dirichlet symmetry fix.** Ocean cells are identity rows ($A_{oo}=1$, $A_{o,\cdot}=0$).
A land cell adjacent to ocean would carry an off-diagonal $A_{L,o}\neq0$ while the ocean row
carries no return coupling ($A_{o,L}=0$) — asymmetric, breaking CG. Standard **symmetric
Dirichlet elimination** (`MatZeroRowsColumnsStencil` on the ocean cells) zeros both the row
and the column and sets a unit diagonal; since $h_o=0$ the land RHS needs no correction. This
keeps each land cell's drain-to-ocean conductance in its diagonal (in $A^{\text{center}}$)
while removing the asymmetric off-diagonal, reproducing the exact flux the Anderson residual
applies (draining land to $h_o=0$).

---

## 5. Summary of the correspondence to code

| math | code (`transient_groundwater.cpp`) |
|---|---|
| residual $F(h)=0$ (§2–3) | `FormFunctionLocal` (Anderson default path) |
| $b(h)=S_c\,(h^0+\text{rech})$, row-scaled (§4.4) | `FormPicardRHS` (new) |
| $A(h)$ SPD, centre-$S$ row-scaled, piecewise $T$, symmetric Dirichlet | `FormPicardOperator` (new) |
| $T(wtd)$ piecewise (§1.1) | `depthIntegratedTransmissivity` |
| $S(h)$ | `updateEffectiveStorativity` |
| harmonic-mean faces (§3) | `e_E,e_W,e_N,e_S` |
| outer Picard + inner CG/GAMG (§4.3) | `SNESSetPicard` + `-snes_type newtonls -snes_linesearch_type basic -ksp_type cg -pc_type gamg` |

The default solver is unchanged (matrix-free Anderson); the Picard path is gated behind a
runtime flag (`-wtm_picard`). See `PICARD_MG_DESIGN.md` for the experiment that decides
whether it earns a place at scale.
