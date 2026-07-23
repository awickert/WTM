# WTM Nonlinear Solver Investigation Notes

**Date:** 2026-05-24  
**Branch:** `awickert/WTM:solver-optimization` (branched from `KCallaghan/WTM:master` @ `4523547`)  
**Investigator:** Andy Wickert

## Repository / Branch Notes

This branch is on `awickert/WTM`, forked from `KCallaghan/WTM`. At branch creation time, `awickert/master` had **diverged** from `KCallaghan/master` with 5 commits not yet merged upstream:

| Commit | Message |
|--------|---------|
| `944c686` | `-DUSE_GDAL=ON required in cmake to compile WTM` |
| `d7d0666` | `Zenodo DOI badge` |
| `1891cdb` | `set number of CPU threads w/ env var` |
| `9e21dea` | `Merge branch 'KCallaghan:master' into master` |
| `b450aa3` | `Merge branch 'KCallaghan:master' into master` |

This `solver-optimization` branch was created directly from `KCallaghan/WTM:master` (not from `awickert/master`) to keep a clean base. The diverging commits on `awickert/master` should be reconciled separately — consider opening PRs for the useful ones (`944c686`, `d7d0666`, `1891cdb`) back to `KCallaghan/WTM`.

## Goal

Speed up the nonlinear groundwater solver for 20–20,000-year production runs on a global grid (~1000×1000 cells at 1°/10 resolution).

---

## Benchmark Configuration

- Grid: 1000×1000 cells (cells_per_degree=10, southern_edge=-45, region=benchmark)
- Timestep: 1 year (deltat=31536000 s)
- 3 SNES solves: cycle 1 (cold start), cycles 2 and 3 (warm start from prior cycle)
- Benchmark data in `benchmark/surfdata/`
- Run scripts: `benchmark/config_anderson.cfg`, `benchmark/config_newton.cfg`

**IMPORTANT:** All benchmarks below were run **single-process** (`mpirun -n 1`). See the MPI bug note at the bottom.

---

## Summary Table

| Solver | m / KSP | rtol | T formula | iters (c1+c2+c3) | Wall time | Notes |
|--------|---------|------|-----------|-----------------|-----------|-------|
| Anderson | m=1 | 1e-4 | smooth eps=0.01 | 18+5+9 = 32 | ~1.9s | **Production choice** |
| Anderson | m=1 | 1e-4 | smooth eps=0.001 | 18+5+10 = 33 | ~2.0s | No benefit over eps=0.01 |
| Anderson | m=1 | 1e-4 | discontinuous | 18+5+10 = 33 | ~1.8s | Same speed as smooth |
| Anderson | m=1 | 1e-8 | smooth eps=0.01 | ~40+ | ~3.4s | 80% overhead, no physics benefit |
| Anderson | m=5 | 1e-4 | smooth eps=0.01 | DIVERGED | — | Oscillates near solution |
| Newton-Krylov | GMRES+GAMG | 1e-4 | smooth eps=0.01 | FAILS | — | J non-symmetric, GAMG incompatible |
| Newton-Krylov | GMRES+BoomerAMG | 1e-4 | smooth eps=0.01 | converges | ~15s | 8× slower per timestep |

---

## Key Findings

### 1. Anderson (m=1, rtol=1e-4) is the best production solver

- Three-phase convergence: fast geometric drop (F: 1722→1, iters 0–7), near-solution oscillation (iters 7–18), rtol-controlled final push
- Cold start (cycle 1): ~18 iters; warm start (cycles 2+): 5–10 iters
- For 20,000-year runs, warm-start performance dominates: typically 5–10 iters/timestep
- Anderson is set as the default in `CreateSNES.cpp`. Override at runtime:
  ```
  -snes_type newtonls -ksp_type gmres -pc_type ilu
  ```

### 2. rtol=1e-4 is the right production tolerance

- Going from rtol=1e-4 to rtol=1e-8 costs ~80% more wall time
- The function norm stagnates around ‖F‖ ≈ 0.15–0.25 m at rtol=1e-4, which is sub-millimeter per cell — more than adequate for 1-year timesteps over 20,000 years
- Do not use rtol tighter than 1e-5

### 3. T and S smoothing has NO effect on Anderson convergence speed

- eps=0 (hard discontinuous), eps=1mm, eps=1cm all give essentially identical Anderson iteration counts
- **Reason:** the smooth transition zone is a tiny fraction of all cells; it doesn't affect the global Picard contraction rate
- The smooth formulas are still used because they provide C∞ derivatives needed for an accurate analytic Jacobian (Newton-Krylov path)

### 4. Newton-Krylov cannot beat Anderson on this problem

- The analytic Jacobian J is non-symmetric (off-diagonal coupling through ∂T/∂wtd)
- GAMG requires near-SPD matrices and immediately fails on J
- A symmetric Picard preconditioner P was implemented (frozen T, S averaged between neighbors → SPD, GAMG-compatible), but spectral mismatch between J and P means GAMG on P is useless as a preconditioner for J
- BoomerAMG handles non-symmetric J but achieves only ~93% convergence rate per KSP iteration on heterogeneous media → requires many KSP iterations → 8× slower total cost
- Conclusion: Anderson is fundamentally better for this heterogeneous elliptic PDE

---

## Code Changes in This Branch

### `src/CreateSNES.cpp`
- Set `SNESANDERSON` as default SNES type (was no default, i.e., `SNESNEWTONLS`)
- Removed the pre-allocated J and Pmat matrices (they shift PETSc memory layout in a way that exposes the MPI ghost-cell bug in `FormFunctionLocal`)

### `src/transient_groundwater.cpp`
- **`depthIntegratedTransmissivity`**: replaced piecewise if/else with smooth C∞ formula:
  - `wtd_eff = softmin(wtd_T, 0, eps0)` — smooth cap at WTD=0
  - Sigmoid blend between linear (Eq. S4) and exponential (Eq. S6) regimes at WTD=-shallow
  - eps0=eps1=0.01 (1 cm)
- **`dTransmissivityInverseDwtd`**: added analytic derivative of 1/T w.r.t. WTD (needed for Newton Jacobian)
- **`dEffectiveStorativityDnew`**: added analytic derivative of S_eff w.r.t. new WTD (needed for Newton Jacobian)
- **`FormFunctionLocal`**: changed ocean boundary condition from `f=0` to `f=x` (Dirichlet; needed for Newton-Krylov non-singularity; Anderson unaffected)
- **`FormJacobianLocal`**: added full analytic Jacobian (J) and symmetric Picard preconditioner (P). Guarded by SNES-type check — only registered when not using Anderson, to avoid MPI ghost-cell segfault.
- **`update`**: added SNES-type check before `DMDASNESSetJacobianLocal`

### `src/update_effective_storativity.cpp`
- Replaced piecewise-constant formula with smooth C∞ formula based on physical water volume:
  - `V(w) = [w(1+p) + √(w²+ε²)(1-p)] / 2`
  - Returns the chord slope `ΔV/Δwtd` for the WTD step (finite-difference)
  - Analytic derivative for the limit Δwtd→0
  - eps=0.01 (1 cm)

### `CMakeLists.txt`
- Added explicit MPI include/link paths for the local system build environment

### `benchmark/` (new directory)
- `config_anderson.cfg`, `config_newton.cfg`: test configurations
- `run_benchmark.sh`: benchmark runner script
- `surfdata/`: benchmark input data
- `SOLVER_NOTES.md`: this file

---

## Known Bugs / Future Work

### CRITICAL: MPI ghost-cell bug in `FormFunctionLocal`

`FormFunctionLocal` caches `1/T` in a global DMDA vector (`T_vec`) and then accesses it at neighbor indices `[j][i±1]`, `[j±1][i]`. Global DMDA vectors do not have ghost cells, so these accesses read adjacent-process memory under MPI — **incorrect results with >1 MPI process**.

The same bug exists for `my_topo`, `my_fdepth`, `my_ksat` neighbor accesses in `FormJacobianLocal`.

**Symptom:** With the smooth T formula (bounded values), the corrupted ghost cells cause only small errors and the solver converges anyway. With a diverging solver or extreme WTD values, the ghost-cell reads can land on unmapped pages → segfault.

**Fix needed:** Convert all arrays accessed at neighbor indices to local (ghost-enabled) vectors via `DMCreateLocalVector` + `DMGlobalToLocalBegin/End` scatter before each `FormFunctionLocal` call. Specifically: `T_vec`, `topo_vec`, `fdepth_vec`, `ksat_vec`, `porosity_vec`, `starting_wtd`.

This is a pre-existing bug in the HEAD code and must be fixed before any production MPI run.

### Newton-Krylov Jacobian (FormJacobianLocal) also has ghost-cell bug

Same issue: neighbor arrays accessed from global vectors. Currently guarded against use with Anderson (which doesn't need J). For Newton-Krylov to work with MPI, the same ghost-scatter fix is needed.

### Anderson m=1 vs higher m

m=1 outperforms m=5 on this problem. The near-solution oscillation phase is dampened better by m=1 (less memory means simpler, more stable updates). This may be problem-specific; re-test if the grid or forcing changes significantly.

---

# 2026-07-23 — MPI scaling, physics validation, and the replicated-data ceiling

**Branch:** `solver-optimization-2` — the solver work above, rebased on top of `mpi-ghost-fix`
(PR #69). **Investigators:** Andy Wickert + Claude.

## Status update: the CRITICAL ghost-cell bug is FIXED

The ghost-cell bug documented under "Known Bugs / Future Work" above is resolved. PR #69
(`mpi-ghost-fix`) converts the neighbor-accessed fields (`T`, `topo`, `fdepth`, `ksat`) to
local ghost vectors in `FormFunctionLocal`. `solver-optimization-2` is stacked on top of #69
and inherits the fix. Verified: the `tests/ghost_cell` regression passes (1-proc vs 2-proc
agree to 0.000000 m at the MPI boundary). `FormJacobianLocal` still reads global vectors, but
it is guarded off for Anderson (the default) and Newton is dead (Finding 4 above), so this is
latent, not active.

## Physics: the smooth T/S formulas are faithful and bounded

Compared `solver-optimization-2` (smooth T/S) against the `mpi-ghost-fix` discontinuous
baseline on an identical config, tracking the water-table divergence over 30 cycles:

| cycle | mean \|Δwtd\| | p99 \|Δwtd\| | max \|Δwtd\| | mean \|wtd\| | relative |
|-------|-------------|------------|------------|------------|----------|
| 5     | 5.9 mm      | 17 mm      | 0.385 m    | 4.9 m      | 0.12%    |
| 15    | 4.5 mm      | 15 mm      | 0.326 m    | 10.9 m     | 0.041%   |
| 30    | 3.7 mm      | 14 mm      | 0.307 m    | 16.3 m     | 0.023%   |

The difference does **not** accumulate — it saturates and shrinks as a fraction of signal
(0.12% → 0.023%), because as the water table fills and moves deeper, fewer cells sit near the
wtd=0 / wtd=−shallow transitions where the two formulations differ. Differences are localized
to those transition zones; deep cells and the (~50% of domain) ocean cells are essentially
identical. Conclusion: the smooth formulas are a controlled regularization, safe to keep.
(Caveat: measured in `test` mode on the benchmark region; a real transient with time-varying
forcing could differ in detail, but the bounded/saturating character is a strong signal.)

Ocean BC (`f=0` → `f=x`, Finding under Code Changes): introduced only for the now-dead Newton
path. Kept for now — arguably the more correct Dirichlet sea-level BC, and reversible. Open
decision.

## MPI scaling (single-node measurements)

1000×1000 benchmark, `OMP_NUM_THREADS=1`, `maxiter=20` (60 solves), single 16-core node. Two
`PetscLogEvent`s (`SetStartVals`, `FullGridReduce`) were added to split the GW-section time
(commit "Add PetscLogEvents to profile solve-loop O(N) overhead").

| ranks | SNESSolve | SetStartVals | FullGridReduce | overhead share |
|-------|-----------|--------------|----------------|----------------|
| 1     | 50.4 s    | 5.1 s        | 2.3 s          | 12.8%          |
| 4     | 20.0 s    | 5.9 s        | 2.3 s          | 29.0%          |
| 8     | 15.3 s    | 6.9 s        | 3.5 s          | 40.5%          |

- The SNES solve **scales well** (~3.6× at n=8); iteration counts are bit-identical to n=1, so
  the domain decomposition is correct as well as fast.
- `set_starting_values` and the full-grid `MPI_Allreduce` are **replicated O(N) work that does
  not scale** (roughly flat, even rising with ranks — likely memory-bandwidth contention).
  Their share of GW time grows 13% → 40% from n=1 to n=8 and would dominate at higher counts.
- `set_starting_values` (a whole-grid loop run on *every* rank) is the larger of the two
  (~2.5× the allreduce) — not the allreduce, as first assumed.
- n=16 failed on the test box (oversubscription of 16 cores); everything past n=8 is unmeasured.
- All single-node, shared-memory MPI. Cross-node behavior of the full-grid allreduce (InfiniBand)
  is unmeasured and expected to be worse.

## The production ceiling at 141M cells: replicated data structures

Production grids reach ~141,120,000 cells. `ArrayPack` holds **25 `float` + 9 `double`
full-grid arrays, replicated on every MPI rank**:

- (25 × 4 B + 9 × 8 B) × 141.12M ≈ **~26 GB per rank** (before PETSc vectors, halo buffers,
  and the 1.13 GB allreduce buffer).
- On a 128-core / 512 GB Agate node only ~12–16 ranks fit → **~85% of each node's cores are
  stranded by memory**, not compute.
- The per-solve full-grid allreduce is 1.13 GB, done ~500× per cycle = **~565 GB/cycle** of
  collective traffic; over InfiniBand (msilarge) this is the dominant communication cost.

FSM is *not* a concern: it runs ~once per simulated year (many GW solves per FSM) and costs less
than a single GW solve, so its amortized share is negligible.

### Why the structures are replicated (design)

1. **FillSpillMerge and the depression hierarchy are global serial algorithms.** Depressions,
   spill points, and watersheds are global topological structures — they cannot be computed from
   a subdomain. RichDEM's FSM has no MPI, so the surface-water side requires the full grid in one
   place.
2. **Historical:** serial-first richdem model; MPI was added *only* to the GW solve via PETSc.
   The two data models — replicated richdem `arp` vs distributed PETSc DMDA — were never unified;
   the full-grid `MPI_Allreduce` is the seam between them.
3. **Coupling simplicity:** a full-grid `arp` on every rank lets FSM, recharge, ocean accounting,
   and I/O be plain serial loops over global (x,y), with no ownership/halos/communication.

### How the replication is realized (mechanism)

- SPMD: each rank runs `main()` (`WTM.cpp`) and builds its own full-grid `ArrayPack`.
- `initialise()` → `InitialiseTransient` (`irf.cpp`) loads **every input full-grid via GDAL on
  every rank** — no rank guard. The *only* rank-0 guards in the code are for output (`saveGDAL`,
  `WTM.cpp:110,239`).
- The distributed PETSc solve is bridged back to the replicated `arp` by a copy-back of owned
  cells + the full-grid `MPI_Allreduce` (`transient_groundwater.cpp`) after every solve.
- FSM and the depression hierarchy run redundantly on the full `arp` on every rank.

## Path forward

- **Lever #2 — down payment (tractable, contained):** hoist the full-grid allreduce out of the
  per-solve loop (it is needed only once per cycle before FSM/output, not `maxiter` times → ~500×
  fewer 1.13 GB collectives), and make `set_starting_values` owned-cells-only with a scalar
  reduction for the mass-balance diagnostics. Removes both non-scaling GW-section overheads.
  Does **not** fix memory. **Blocked on a decision:** should each rank hold the *global*
  `total_loss_to_ocean` or its *local* contribution? — this determines the correct reduce.
  Requires a mass-balance regression (before/after totals at n=1 and n=8) plus the ghost test.
- **Distributed data model — the real fix (architectural, large):** make `arp` distributed like
  the PETSc vectors (each rank owns its subdomain + halos); gather to a full grid only for the
  rare FSM step and for I/O. Drops per-rank memory from ~26 GB to tens of MB → use all 128
  cores/node, and subsumes lever #2 (no per-solve allreduce, no replicated loops). Touches
  `ArrayPack`, initialization/loading, I/O, and the FSM handoff. Lever #2's owned-only
  `set_starting_values` is literally the first step of this, so it is **not** throwaway work.
