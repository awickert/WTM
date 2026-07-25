# Picard solver experiments

Reproducible harnesses for the semi-implicit Picard groundwater solver
(`-wtm_picard`; see `../PICARD_MG_DESIGN.md`, `../PICARD_MATH.md`) and the
time-stepping study that motivated `../BDF2_ADAPTIVE_DESIGN.md`. These generated
the numbers quoted in those notes (local runs, 2026-07-25, single node).

## Setup

```bash
# build (from repo root); local desktop build uses system PETSc + GDAL via mpicxx
cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j wtm.x

# rasterio is needed for the .tif I/O (see ../../msi_env.sh / conda env)
export WTM_WORK=/tmp/wtm_picard_bench      # scratch dir (default if unset)
```

All scripts resolve the `wtm.x` binary and the `benchmark/scaling` synthetic-input
tooling relative to the repo (`paths.py`); scratch output goes under `$WTM_WORK`.

## Scripts

| script | question | key result |
|---|---|---|
| `core_grid_sweep.py` | Anderson vs Picard, grids 64²–1024², ranks 1–8 | inner CG+GAMG **flat ~3–5** all grids; Anderson outer iters **grow** (3→18); Picard ~8–20× slower per solve at these sizes (crossover > 1024²) but **strong-scales better** (1024²: 3.2× at n=8 vs Anderson ~1.3×) |
| `make_equil128.py` | build the 128² pure-drainage fixture | 100 m mound draining to an ocean ring, zero forcing → clean monotonic relaxation (τ≈2×10⁴ yr) |
| `timestep_robustness.py` | max stable Δt & steps-to-equilibrium | Anderson **diverges at Δt≥10 yr** (ceiling ~1 yr); Picard unconditionally stable → equilibrium in Δt=1000→**50**, 10⁴→**10**, 10⁵→**4 steps** |
| `equilibrium_accuracy.py` | is Picard's big-step equilibrium correct? | Picard Δt=1000 vs Δt=10⁵ agree to **2×10⁻³ m** (Δt-independent); Anderson Δt=1 converges to the same field but needs **~40,000 steps** |
| `transient_accuracy.py` | transient path error vs Δt (Anderson Δt=1 = truth) | **first-order in Δt** (10× Δt → 10× error), **washes out toward equilibrium** — stability without free accuracy, but first-order-controllable |
| `bdf2_order.py` | is the `-wtm_bdf2` path second-order? | self-convergence ratio → **4 (order 2)** for BDF2 vs → 2 (order 1) for backward Euler; BDF2 is 25–85× more accurate at equal Δt, gap widening as Δt shrinks |

## Run

```bash
python3 make_equil128.py            # once, for the timestep/accuracy scripts
python3 core_grid_sweep.py          # ~minutes; the 1024² Picard rows are the slow part
python3 timestep_robustness.py
python3 equilibrium_accuracy.py     # Anderson 40k-step ground truth is the slow part
python3 transient_accuracy.py
python3 bdf2_order.py               # verifies -wtm_bdf2 is 2nd order
```

`-wtm_bdf2` (Phase A of `../BDF2_ADAPTIVE_DESIGN.md`) turns on second-order BDF2
time integration on the Picard path (it implies `-wtm_picard`); default is
backward Euler.

## Memory (measured, `-memory_view`, 1024²)

Anderson 722 MB (n=1); Picard 891 MB (n=1, +170 MB for the assembled operator +
GAMG hierarchy, ~170 B/cell, DMDA-distributed); Picard n=8 total 1.67 GB, 335 MB
max/proc. The whole sweep peaks under ~1.7 GB.

## Takeaway

At fixed small Δt on a single node Picard costs more per solve (assembly +
GAMG-setup dominated; crossover above 1024²), but it lifts the Δt stability
ceiling from ~1 yr to unbounded — collapsing an intractable ~10⁴-step equilibration
into a handful of steps at the same equilibrium accuracy, and strong-scaling better
on many cores. The equilibrium is Δt-independent; the transient path is first-order
in Δt (hence `../BDF2_ADAPTIVE_DESIGN.md`).
