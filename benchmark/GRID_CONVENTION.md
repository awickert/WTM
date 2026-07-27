# WTM grid convention and the conservative flux discretisation

**Date:** 2026-07-27
**Status:** convention + corrected-flux design. Motivates and specifies the fix for the
E-W/N-S cell-size swap found via the water budget (see `WATER_BUDGET.md §4a`).
**Companions:** `PICARD_MATH.md` (operator), `WATER_BUDGET.md` (budget), the WTM paper
Callaghan et al. (2025, GMD) Appendix B.

## 1. The canonical index convention (read this before touching the flux)

WTM's domain is a **latitude-longitude raster**. Fix the meaning of the two array indices
once and for all:

| index | axis | compass | code spacing | varies? |
|---|---|---|---|---|
| **`i`** | columns | **East-West (longitude)** | `cellsize_e_w_metres[j]` | yes — with latitude |
| **`j`** | rows | **North-South (latitude)** | `cellsize_n_s_metres` | no — constant |

- Arrays are `field[j][i]` (PETSc `DMDA`) / `arp.field(i, j)`. `i` increases **eastward**;
  `j` increases **northward** (`latitude = southern_edge + j / cells_per_degree`, `irf.cpp`).
- **N-S spacing** `cellsize_n_s_metres = meters_per_degree / cells_per_degree` is **constant**
  (lines of latitude are equally spaced).
- **E-W spacing** `cellsize_e_w_metres[j] = cellsize_n_s · cos(latitude[j])` **shrinks poleward**
  (lines of longitude converge). It is a function of the *row* `j` only.
- Cell area `cell_area[j] = cellsize_n_s · cellsize_e_w_metres[j]`.

### ⚠️ The paper uses the OPPOSITE letters — this is the trap
The WTM paper (Callaghan et al. 2025, Appendix B) labels its axes **`x = (S-N)`** and
**`y = (W-E)`** — i.e. paper-`i` is North-South and paper-`j` is East-West, **transposed** from
the code's array indices above. So:

| paper symbol | paper axis | equals in code |
|---|---|---|
| `Δx` | S-N | `cellsize_n_s` (constant) |
| `Δy` | W-E | `cellsize_e_w[j]` (varies) |

The paper's Appendix-B formula is correct *in the paper's letters* (each axis divided by its own
spacing). The **bug** was implementing that formula **by index letter** on the code's transposed
array: the code divided the `i`-direction (which is **E-W** in the code) by `cellsize_NS²` and the
`j`-direction (**N-S**) by `cellsize_EW²` — i.e. **each direction was divided by the *other*
direction's spacing.** Harmless at the equator (`cos ≈ 1`); off by `cos²(latitude)` elsewhere; and
the root of the north-south volume non-conservation (`WATER_BUDGET.md §4a`).

## 2. The correct, conservative finite-volume flux

Integrate Eq. (B1) over a cell and apply the divergence theorem: the flux through a face is
`F = T_face · (h_nbr − h_c) · L_face / d_face`, with `T_face` the harmonic mean (flux continuity
across a discontinuous coefficient). Face geometry, by orientation:

| face | separates | `L_face` (wall length) | `d_face` (centre distance) |
|---|---|---|---|
| **E / W** (`i±1`) | E-W neighbours | `cellsize_n_s` (N-S wall) | `cellsize_e_w[j]` |
| **N / S** (`j±1`) | N-S neighbours | `cellsize_e_w[j±½]` (E-W wall, **at the face**) | `cellsize_n_s` |

The head-form term entering the residual (`F` divided by the cell area `A_j = cellsize_n_s ·
cellsize_e_w[j]`) is then:

```math
\text{E/W: } \frac{T_{\!E}\,(h_{i\pm1,j}-h_{ij})}{cellsize\_e\_w[j]^2}
\qquad\Longleftarrow\ \text{divide by } \mathbf{cew2}\ (\text{not } cns2)
```

```math
\text{N/S: } \frac{T_{\!N}\,(h_{i,j\pm1}-h_{ij})\;cellsize\_e\_w[j{\pm}\tfrac12]}
{cellsize\_n\_s^{2}\;\,cellsize\_e\_w[j]}
\qquad\Longleftarrow\ \text{divide by } \mathbf{cns2},\ \text{times the FACE E-W length}
```

The E-W term reduces to a plain `÷ cew2` (both cells share row `j`, so it is already conservative
and symmetric). The N-S term carries the **face-centred** E-W length `cellsize_e_w[j±½]`, which is
what the fix adds.

### Why the face length matters (conservation) and volume form (symmetry)
Conservation needs the *volume* flux across a shared face to be equal-and-opposite. With the
face-centred length, `F_N(cell j)` and `F_S(cell j+1)` use the **same** `cellsize_e_w[j+½]`, the
same harmonic-mean `T`, and the same `cellsize_n_s`, so `F_N(j) = −F_S(j+1)` **exactly** →
budget closes to machine zero.

Symmetry (for the SPD Picard operator + CG): the head-form operator is row-divided by `A_j`, which
differs between rows and *breaks* symmetry across N-S faces (this is already true of the current
operator, only more so). The clean cure is to assemble in **volume form** — multiply each cell's
row by `A_j`, so the off-diagonals become the shared face conductances `G_face = T_face · L_face /
d_face`, which *are* symmetric (`G_N(j) = G_S(j+1)`) and conservative. Row-scaling does not change
the solution, and the RHS/storage terms scale with `A_j` to match. (Equivalent: keep head form and
accept the small asymmetry CG already tolerates — but volume form is the correct, exactly-symmetric
choice and is what makes the budget close.)

## 3. Implementation plan (option B — fix the flux, keep the raster layout)

1. **Geometry** (`irf.cpp`, and `run_dephier.cpp`): the N/S cell-edge E-W lengths are already
   computed (`cellsize_e_w_metres_N/_S`) then averaged away — keep them as the **face** lengths
   `cellsize_e_w[j±½]`, and expose them to the solve (DMDA vecs, owned range; each cell uses its own
   N and S face values, which equal the neighbour's by shared-latitude construction — no ghost).
2. **Flux**, consistently in all four places — `FormFunctionLocal` (Anderson residual),
   `FormPicardOperator` + `FormPicardRHS`, `FormJacobianLocal`, and `accumulate_ocean_outflow`:
   E-W `÷ cew2`; N-S `÷ cns2 ×` the face E-W length; assemble in volume form for symmetry.
3. **Prove it**: `exact_budget_residual → machine zero` on a lat-varying grid (the `WATER_BUDGET.md`
   diagnostic is the acceptance test). Regenerate the golden references — results **will** change,
   most away from the equator — and inspect the diffs.
4. **Docs**: update `PICARD_MATH.md §3` to the corrected, paper-consistent form and cross-link here.

Coordinated with Kerry Callaghan (upstream) — fix lands in the fork, then upstream.
