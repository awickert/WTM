# Island equilibrium test

A small, ocean-ringed sub-island of the Esquibel domain -- 117x75 cells (8775),
4"/900-cells-per-degree, one-week time step, cold start (`wtd = 0`) to
equilibrium. Small enough to iterate on in seconds, but a physically real
cold-start drainage problem. It is the shared fixture for comparing the
KCallaghan v2.0.1 original against the awickert fork.

## Contents

- `domain/` -- the ten input rasters plus `southern_edge.txt`. These files ARE the
  fixture (committed so it survives, independent of any external dataset).
- `make_island.py` -- regenerates `domain/` from the full Esquibel rasters, for
  provenance. The island is the window `col_off=500, row_off=335, 117x75` of the
  full `853x451` domain; that crop reproduces the committed rasters byte-for-byte.
- `eq_kcallaghan.cfg`, `eq_awickert.cfg` -- identical domain and physics; they
  differ only in output paths. Run each from this directory (paths are relative).
- `run_island.sh` -- runs one model at a chosen MPI rank count and reports wall
  time and the convergence trace.
- `results/` -- run outputs and logs (created on first run; not committed).

## The two models

Both solve the same domain and physics. They differ only in how the solver runs:

- **KCallaghan v2.0.1** (the immutable original) has no built-in Anderson, so its
  default solver does not converge on this stiff cold start. It is run with
  PETSc's matrix-free Anderson on the command line -- the only way it converges
  at a one-week step:

      -snes_mf -snes_type anderson -snes_stol 1e-6

  v2.0.1 is expected to **segfault at four or more MPI ranks** (the ghost-cell bug
  the fork fixes); it runs only serially. That parallel incapability is itself an
  axis of the comparison, not a nuisance.

- **awickert fork** (our new version) runs Anderson, first-order in time, with a
  capillary-derived taper length, at the same inner tolerance as v2.0.1:

      -wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6

  The `-snes_stol 1e-6` matters for a fair comparison: ours' PETSc default is 1e-8
  (100x tighter), which inflates iterations/solve. Both models now solve each step
  to the same 1e-6.

  and is expected to run correctly across MPI ranks. First order is deliberate:
  this is a cold-start equilibrium spin-up, where second-order in time
  (`-wtm_bdf2_on_V`) rings in a limit cycle rather than settling (Δ swings
  140-320 over 100 cycles, versus Δ ~2.5e-5 by cycle 20 at first order). BDF2-on-V
  is the tool for a warm transient, not for driving to equilibrium from cold.

## Running

Both binaries are already built:

- fork (our new version): `../../build/wtm.x` (branch `bdf2-adaptive-dt`)
- KCallaghan v2.0.1 baseline: `../../../WTM-mastertest/build/wtm.x` (checked out at
  `4523547`, which differs from the `v2.0.1` tag only by `CITATION.cff`)

From this directory:

    ./run_island.sh awickert   ../../build/wtm.x                    8
    ./run_island.sh kcallaghan ../../../WTM-mastertest/build/wtm.x  1

For a clean-room baseline instead of the WTM-mastertest worktree, clone the tag:

    git clone https://github.com/KCallaghan/WTM.git /tmp/kcallaghan-wtm
    cd /tmp/kcallaghan-wtm && git checkout v2.0.1 && git submodule update --init --recursive
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=/usr/bin/mpicxx -DCMAKE_C_COMPILER=/usr/bin/mpicc \
      -DGDAL_CONFIG=/usr/bin/gdal-config && cmake --build build -j

## Open decisions (pin before reporting numbers)

1. **`-wtm_Tbar`** (log-mean transmissivity): the prior champion combo is
   Anderson + T-bar, but it is not yet part of the stated spec, so `run_island.sh`
   leaves it off. Decide whether it belongs.
2. **Which ksat drives the capillary taper.** The pedotransfer reads the aquifer
   `horizontal_ksat` field; the capillary fringe is physically a near-surface
   soil-zone property. Same field on this island, but a real distinction to
   resolve before the taper is claimed as physical.
3. **Parallel runs must be pure MPI (`OMP_NUM_THREADS=1`).** WTM's parallelism is
   MPI; letting each rank also spawn OpenMP threads oversubscribes the cores and
   the OpenMP region in FormFunctionLocal deadlocks against PETSc's collective
   reductions (hangs at >=4 ranks -- diagnosed by backtrace 2026-08-10). run_island.sh
   now exports OMP_NUM_THREADS=1. With it, n=1/2/4/8 are MPI-consistent; island
   scaling is ~1.9x at n=4 and ~2.0x at n=8 (8775 cells is too small to scale
   further -- communication and the serial FillSpillMerge dominate).
