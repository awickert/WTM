# Island comparison: awickert fork vs KCallaghan v2.0.1

Authoritative record of the island equilibrium comparison (2026-08-10). All numbers
below are at a **matched inner solver tolerance** (`snes_stol 1e-6` for both models);
earlier unmatched numbers are superseded and should not be cited.

## Setup

- Domain: the island fixture in `domain/` (117x75 = 8775 cells, cropped from full
  Esquibel; see `make_island.py`), 1-week timestep, cold start (`wtd=0`) to equilibrium.
- Both models solve the same domain and physics. They differ only in invocation:
  - v2.0.1: `-snes_mf -snes_type anderson -snes_stol 1e-6` (its default SNES does not
    converge on this stiff cold start; CLI Anderson is how it is run).
  - ours: `-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6` (Anderson, 1st order
    in time, capillary-derived taper length).
- Single process unless a rank count is given; parallel runs are pure MPI
  (`OMP_NUM_THREADS=1`). Δ = |Σ wtd change| between cycles (lower = more converged).

## Results (matched `snes_stol 1e-6`)

| config | wall to match v2.0.1 (Δ≤0.59) | wall to completion | final Δ | iters/solve |
|---|---|---|---|---|
| v2.0.1 n=1 | 1.59 s | 1.59 s* | 0.590363 | 5.9 |
| ours n=1 | 1.47 s | 7.5 s | 2.7e-5 | 5.5 |
| ours n=2 | 1.02 s | 4.3 s | 3.0e-5 | 5.5 |
| ours n=4 | 0.96 s | 3.0 s | 3.2e-5 | 5.5 |
| ours n=8 | 0.88 s | 2.5 s | 2.6e-5 | 5.5 |

\* v2.0.1 reaches its floor at cycle 21 (1.59 s) and cannot improve; running to 100
cycles wastes ~3.3 s more for no gain. v2.0.1 SEGVs at >=4 ranks (the MPI ghost-cell
bug the fork fixes), so it has no parallel column.

## Findings

1. **Precision comes from the mass-conservation fix, and v2.0.1's 0.59 floor is a mass
   leak.** The output columns show it: at their settled states both models exchange
   ~1755-2381 units between groundwater and surface water each cycle, but v2.0.1's GW
   and SW changes do not cancel (+1754.97 / -1755.45 = -0.48 destroyed per cycle),
   whereas ours cancels to 2.5e-7. That non-cancellation is the 0.590363 floor.
2. **Root cause of the leak: the N-S / E-W cell-size flip.** v2.0.1 divided the E-W flux
   by the N-S cell size and the N-S flux by the (cell-centered, latitude-varying) E-W
   cell size -- swapped. Because the E-W size shrinks poleward, adjacent cells at
   different latitudes compute different fluxes across their shared N-S face, so water
   leaks at every N-S face. The fix uses face-centered geometry in a conservative flux
   sum; the budget closes to machine zero. This is not an FD-vs-FV question -- a
   conservative FD would be identical.
3. **The numerics are not harder.** At matched tolerance ours takes fewer nonlinear
   iterations per solve (5.5 vs 5.9). Ours reaches v2.0.1's accuracy in 9 cycles vs
   v2.0.1's 21 -- each cycle is more productive because it conserves mass -- so ours is
   faster to 0.59 even on a single core, then drives ~23,000x deeper where v2.0.1
   cannot follow.
4. **Method note (a real trap we hit): match the inner solver tolerance before any
   speed comparison.** Ours' PETSc default `snes_stol` is 1e-8, 100x tighter than
   v2.0.1's 1e-6; leaving them unmatched inflated ours' iterations (10.5 vs 5.5) and
   wall time and produced a spurious "ours is slower" conclusion. Matched, it reverses.

## Reproduce

    ./run_island.sh awickert   ../../build/wtm.x                    8
    ./run_island.sh kcallaghan ../../../WTM-mastertest/build/wtm.x  1

The full two-metric table above is produced in one pass by `./compare.sh` (both
models, both wall metrics, n=1/2/4/8, matched tolerance). It expects the KCallaghan
v2.0.1 binary at `../../../kcallaghan-wtm/build/wtm.x` and the fork at `../../build/wtm.x`.
