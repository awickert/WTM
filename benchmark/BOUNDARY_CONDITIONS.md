# WTM domain boundary conditions: padding+setEdges → mask-aware ghost nodes

**Date:** 2026-08-14 · **Branch:** `bdf2-adaptive-dt` · **Status:** implemented behind `-wtm_ghost_boundary`
(off by default); cc-validated on Esquibel; goldens to be remade when defaulted. Task #96.

## The problem with padding + `setEdges(0)`

WTM's convention rings the domain with a pad of topo-0 ocean cells, and `land_mask.setEdges(0)`
(irf.cpp) forces the outermost cell ring to ocean (Dirichlet head = 0). This works only where the pad
is genuine sea. Where the pad **fails** — the domain edge clips real high land — `setEdges(0)` forces a
land cell (e.g. a 271 m headland crest) to sea level, creating an **artificial deep sink** that drains
the interior. On Esquibel the south pad failed at a headland (24 land cells, topo 91–273 m, reach the
edge); the resulting sink was the root cause of a long cc-vs-TR-BDF2 transient disagreement (first
mis-read as bistability). The error scales with the edge cell's elevation:
`corr(|cc−tr|, ocean-neighbour topo) = 0.84`.

## The fix: a mask-aware ghost-node boundary (no padding)

With `-wtm_ghost_boundary`, `setEdges(0)` is **skipped** — edge cells keep their real mask — and each
domain-edge cell carries the boundary condition through a **ghost node just outside the domain**,
computed internally (the interior is not altered, no pad ring):

- **Ocean edge** (`mask == 0`): Dirichlet, ghost head = 0 (sea level). This is the existing, correct
  "sea level everywhere" condition, now applied at the real edge rather than a pad.
- **Land edge** (`mask == 1`): Neumann, where the **water-table surface slope equals the land-surface
  slope** — the water table continues parallel to the terrain off-map:

  ```
  ghost_head = head_edge + (topo_edge − topo_inland)
  off-map face flux = −T · (topo_edge − topo_inland) / d
  ```

  The topographic gradient is taken from **fixed** inland data (stable); extrapolating the *water-table*
  gradient instead is unstable and is not done. This unifies the cases: a flat crest → zero slope →
  no-flow (a divide); a hillslope → drainage at the terrain gradient; terrain rising to the edge →
  inflow from off-map upslope (physical).

Why not simpler alternatives: forcing land edges to Dirichlet h=0 (the bug) *over-drains*; a plain
**no-flow** (zero-flux) land edge *over-dams* — it walls in an endorheic region and mounds the water
table without bound (verified: +2584 m runaway). The land-slope Neumann sits between and is physical.

## Implementation notes

- `DM_BOUNDARY_NONE` gives no ghost cells beyond the physical edge, so off-map faces are handled
  **implicitly**: the ghost head is computed from the centre cell and the **inward** reflection
  `(2j−nj, 2i−ni)`, which reads only toward the interior — never out of bounds.
- Every routine that iterates cells and reads neighbours needs an off-map guard once `setEdges` is
  skipped (edge land cells become real): `FormFunctionLocal`, `accumulate_ocean_outflow`,
  `compute_tr_explicit` (Anderson / matrix-free / TR-BDF2 explicit stage), **and now**
  `FormJacobianLocal` (Newton) and `FormPicardOperator`+`FormPicardRHS` (Picard). All four solver
  families handle the off-map land-slope ghost consistently.
- **Matrix-assembly caveat (the sparsity was built for the padded array).** The DMDA matrix is
  preallocated as a 5-point star assuming every edge cell is ocean (a single Dirichlet diagonal). Once an
  edge cell is real *land*, one or two of its stencil neighbours fall off the global grid;
  `MatSetValuesStencil` does **not** silently drop an out-of-range column, it **errors** ("inserting a
  new nonzero" / local index too large) and corrupts the matrix. So the Newton Jacobian and the Picard
  operator now assemble a **variable-length stencil**: one off-diagonal per *in-bounds* face plus the
  centre. The off-map face contributes no column -- for Newton its whole entry (`G·dX·(−τ'_c/τ_c²)`) is on
  the centre diagonal; for Picard its flux is constant in `x` (the centre head cancels) and goes to the
  RHS (`+dt·T_c·G·(topo_c − topo_inland)`), never the SPD operator. With the flag off every land cell is
  interior (5 columns), so this is sparsity- and FP-identical to the old fixed 5-point assembly.
- **Verification.** Newton is FD-verified with `-snes_test_jacobian` on a land-edge fixture:
  `‖J−Jfd‖/‖J‖` is 4–7e-5 with the ghost boundary ON, matching the OFF baseline. cc (Anderson), TR-BDF2,
  BDF2-on-V, and Newton all converge to the SAME steady water table under the ghost boundary
  (max|Δ| ≤ 7e-9 m). **bdf2v-Picard** (`-wtm_picard -wtm_bdf2_on_V`) also matches
  cc to 5.7e-14, so the Picard operator+RHS off-map handling is consistent. **BE-Picard** (plain
  `-wtm_picard`) is the outlier: warm-started at cc's converged field it takes one step then DIVERGES
  (`DIVERGED_MAX_IT`) -- a **pre-existing free-surface contraction failure** (its storage diagonal uses the
  SECANT storativity, which collapses at surface-pinned cells; the bdf2v tangent-Sy diagonal stays
  well-conditioned), **not** caused by the ghost boundary and not a wrong fixed point. (An earlier "~0.33 m
  Picard gap" was retracted as an unconverged-reference artifact -- the reference cc was still relaxing.)
  See `tests/ghost_boundary/` and the memory note on the BE-Picard divergence.
- Behavior is **byte-identical with the flag off**. And with the flag **on**, the full golden suite
  still passes (30/30, all rank counts) — because every golden fixture sets explicit ocean edges
  (`mask[0,:]=…=0`), so the ghost boundary applies the same sea-level Dirichlet there that `setEdges`
  did. **The old goldens are valid under the new BC; no remake is needed** for ocean-edge fixtures. Only
  a test domain with real *land* at the edge would move (none currently do).

## Outlet caveat (open refinement)

No-flow / land-slope at a **watershed outlet** cut by the map margin can dam surface-bound water into an
artificial lake. Two mitigations: (1) in production, FillSpillMerge routes surface water off-map, so the
subsurface no-flow is shed at the surface; (2) the principled fix is a **flow-direction-aware** edge —
use the DEM's routing to distinguish a no-flow divide from a free-draining outlet. Deferred; the
land-slope Neumann is already a large improvement over the sea-level sink for every land edge.

## Status / next

cc and cc+`-wtm_Tbar` agree under the ghost boundary (Esquibel cell (449,833) identical at −36.3 m; field
mean|Δ| 0.0015 m), and the full golden suite passes with the flag on. **Done since:** off-map guards for
the Newton Jacobian and Picard operator+RHS (variable-length stencil), FD-verified; cc/tr/bdf2v/Newton
confirmed to agree domain-wide (a controlled land-edge fixture, `tests/ghost_boundary/`, stands in for the
Esquibel cross-scheme check while MSI was unavailable). **Next:** re-run the Esquibel cross-scheme check on
MSI when it is back; run with FSM; separately, the pre-existing BE-Picard free-surface divergence (secant
storativity diagonal; task #97) -- not a ghost-boundary issue; then default `-wtm_ghost_boundary` (goldens already pass with it on — no remake
for ocean-edge fixtures). Complementary committed fixes: volume-based recharge (`777326d`) and the TR-BDF2
explicit-stage ocean BC (`d8cc249`).
