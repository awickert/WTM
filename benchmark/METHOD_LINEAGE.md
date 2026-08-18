# WTM method lineage & differences — KCallaghan → cc → TR-BDF2 / adaptive

Traces how the WTM groundwater solver evolved from KCallaghan's published v2.0.1 through our corrected
backward-Euler (**cc**) to the higher-order (**TR-BDF2**) and self-tuning (**adaptive**) integrators, and
records exactly what differs at each step. Companion to `CHANGELOG.md`, `REVIEW_NOTES_since_v2.0.1.md`, and
`benchmark/esquibel/SCALING.md`. Last updated 2026-08-18.

## The four things being compared

- **KCallaghan (v2.0.1)** — the published upstream WTM: fixed-1-week **backward-Euler**, matrix-free
  **Anderson** nonlinear solve.
- **cc** ("corrected-Callaghan") — *the same algorithm* (fixed-dt backward-Euler matrix-free Anderson) in our
  corrected code. Flag: `-wtm_anderson`.
- **fixed_tr** — fixed-dt **TR-BDF2** (2nd-order, L-stable, two-stage). Flag: `-wtm_tr_bdf2`. Same solver,
  spatial scheme, and physics as cc — only the time integration differs.
- **adapt** — TR-BDF2 with the self-tuning adaptive-dt controller. Flags: `-wtm_tr_bdf2 -wtm_dt_adaptive`.

Lineage: **KCallaghan v2.0.1** —(correctness fixes)→ **cc** —(2nd-order time integration)→ **fixed_tr**
—(embedded error control on dt)→ **adapt**.

---

## 1. cc vs KCallaghan — same algorithm, corrected

cc is KCallaghan's method with these fixes. It changes *correctness and scalability*, not the numerical
scheme.

1. **MPI ghost-cell boundary bug** (fixed + dedicated validation test). v2.0.1 **SEGVs at MPI n ≥ 4**; cc
   scales cleanly to 128+ ranks. *This is the enabler of all parallelism.* Includes the mask-aware boundary
   (Dirichlet h = 0 on ocean, Neumann no-flow on land edges; task #96).
2. **Conservative finite-volume flux + lon/lat grid-spacing swap.** v2.0.1 divided the east-west and
   north-south fluxes each by the *other* direction's spacing (a cos-lat swap) and was not flux-conservative
   across shared faces. cc fixes both (golden references regenerated). See `benchmark/GRID_CONVENTION.md`.
3. **Mass conservation.** v2.0.1's groundwater + surface-water changes do not cancel → a ~0.48 m³/cycle leak
   = a **~0.59 m water-table floor**; cc cancels to machine zero and reaches ~2.5×10⁻⁵ m, which v2.0.1
   cannot. Exact per-step discrete water budget.
4. **Recharge applied as a fixed volume** (task #93). v2.0.1 applied recharge as a *storativity-scaled head*
   with inconsistent storativity, so different integrators converged to different water tables; cc applies a
   fixed volume (poro·rech).
5. **Smaller correctness fixes:** an integer-division bug that froze transient forcing at its start-time
   values; water-budget diagnostics made MPI-consistent (owned-cells partials + scalar reduction).
6. **Anderson robustness at scale** (so the *same* solver actually converges): damping
   (`-snes_anderson_beta 0.5`) fixes a steep-terrain stall (`DIVERGED_MAX_IT` on real DEMs); a **periodic
   history restart** (on by default) fixes the ~139-million-cell near-convergence residual reversal.

**Net:** cc = KCallaghan's method made (a) **parallel-correct** (runs at scale), (b) **mass-conservative**
(precise — no 0.59 m floor), (c) **discretization-correct** (conservative FV, right grid spacing,
fixed-volume recharge), and (d) **robust** (Anderson converges on steep/large domains). *Same equations,
corrected implementation.*

## 2. fixed_tr (TR-BDF2) vs cc — same model, different clock

Both share the *identical* matrix-free Anderson solver, conservative-FV spatial discretization, physics, and
FillSpillMerge coupling; they differ only in time integration.

| axis | cc (backward-Euler) | fixed_tr (TR-BDF2) |
|---|---|---|
| order in time | 1st | **2nd** (L-stable, 2-stage: trapezoidal + BDF2) |
| solves per step | **1** | 2 staged (so a tr "iteration" ≈ 2× a cc iteration) |
| warm-transient accuracy | plateaus ~0.11 m (over-drains the deep exp-T tail) | converges O(dt²), **no floor** |
| warm-transient stable step | dt ceiling ~2 wk | **~8 wk** |
| warm-transient cost to accuracy | baseline | **~3.7–5× fewer** SNES iterations |
| cold spin-up convergence | **damps** stiff/free-surface cells → per-cycle rms → mm | damps less → per-cycle rms plateaus, worst cell rings |

**Regime split:** fixed_tr wins **warm transients** (2nd-order accuracy, bigger steps, fewer iterations); cc
is the more robust **cold-spin-up** converger (backward-Euler damping). *Caveat:* the cold-spin-up gap was
measured worst at the **artificial N-S tiling cliffs** (unrealistic land-against-ocean-Dirichlet drops), so
it is partly a stress-test artifact; on real terrain TR-BDF2's L-stability should hold up better.

## 3. adapt (adaptive-dt) vs fixed_tr — the self-tuning layer

adapt adds, on top of TR-BDF2: an **embedded error estimator** from the two stages
(`h_pred = [Y_γ − (1−γ)hⁿ]/γ`, no history) driving a **reject/retry + grow/shrink** dt controller with a
free-surface-aware error norm. Consequences (measured):

- **Robust:** converges where fixed-dt TR-BDF2 fails, and grinds safely (small steps) through hard cells
  rather than stalling/crashing — the reason it is the recommended method for large unattended runs.
- **Deterministic run-to-run**, but its *cost* is layout-sensitive (its controller uses a global error norm
  that shifts with rank count / domain size).
- Reaches the finest per-cycle settledness on realistic terrain; under acute cliff stress it grinds (slow)
  while cc's backward-Euler is quietly fast — see `SCALING.md`.

## Which method when

- **Production spin-up of a large / unknown domain:** **adapt** — safe, won't get stuck.
- **Warm transient stepping (paleo timeloop):** **fixed_tr** — 2nd-order, big steps, fewest iterations.
- **Baseline / KCallaghan-comparable equilibrium:** **cc** — robust backward-Euler, fastest to a moderate
  precision under stiff stress.
- **Never:** KCallaghan v2.0.1 for anything at scale — it cannot run past ~4 MPI ranks.

## Interpreting method comparisons (standard)

Compare wall/iterations across methods **only at matched precision**; a shared `eq_tol`/`frac` stop fires at
very different per-cycle wtd-rms per method (e.g. fixed_tr 65 mm vs cc 3.5 mm vs adapt 1.27 mm at the same
frac stop), so those wall numbers are *not* apples-to-apples. Report, for each method: (1) wall + iterations
to reach fixed rms thresholds {100, 10, 1 mm}; (2) the ultimate rms floor; (3) the native stop with the rms
it stopped at. Prefer the FV-consistent **pure-water-depth** rms (`-wtm_eq_metric water`, |S·Δwtd|) — head-rms
floors are contaminated by deep low-storativity cells.

## Provenance / confidence

- CHANGELOG-confirmed: §1 items 1, 2, 5, 6; the TR-BDF2 order/stability description.
- This branch, verified in our testing but not yet in the released CHANGELOG: §1 item 3 (mass-conservation
  floor, from the island v2.0.1-vs-ours test) and item 4 (fixed-volume recharge, task #93). *Worth a
  CHANGELOG entry for the public record.*
- The cold-spin-up cc-vs-tr convergence and the adapt characterization come from the 2026-08 exclusive
  scaling runs + the water-depth / seam-cliff probes (`benchmark/esquibel/SCALING.md`,
  `scaling_multinode_reps.csv`, `scaling_weak_multinode_reps.csv`).
