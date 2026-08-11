# WTM strong-scaling & memory — MSI Agate, groundwater solve only (FSM OFF)

This set benchmarks the **groundwater solve in isolation**: `fsm_on 0`, so
FillSpillMerge is not in the loop. `gw_s` is the clean, bandwidth-bound signal;
`wall_s` additionally carries serial rank-0 setup (see `serial_overhead`).

Machine-readable setup:

```yaml
study: WTM strong-scaling & memory (groundwater solve, FSM off)
date: 2026-08-11
cluster: MSI Agate (Rocky 8)
allocation: shared (non-exclusive)   # -N 1 --ntasks-per-node=32 --mem=32gb -t 1:30:00 -p msilong
nodes: [acn02]
cpu:                                  # TODO: fill from `lscpu` / `scontrol show node acn02`
  model: TODO
  sockets: TODO
  cores_per_socket: TODO
  threads_per_core: TODO
  memory_channels: TODO
  l3_mib: TODO
  mem_gb: TODO
build:
  repo: awickert/WTM
  branch: bdf2-adaptive-dt
  commit: 446e3d0
  build_type: Release                 # -O3
toolchain:                            # `source msi_env.sh`
  petsc: "3.24.5"                     # module petsc/3.24.5-gnu-rocky8; MPICH bundled (--download-mpich)
  gdal: "3.12.1"                      # module gdal/3.12.1-gcc-11.3.0-netcdf-4.9.3
  gcc: "11.3.0"
  cmake: "3.29.2"
  mpi: "MPICH (PETSc-bundled)"        # launcher: mpiexec ($PETSC_DIR/bin)
solver:
  flags: ["-wtm_anderson", "-snes_stol", "1e-6"]   # matrix-free Anderson
config:                               # run_type test synthesizes all fields but topo+slope (make_synthetic.py)
  run_type: test
  fsm_on: 0                           # <-- FillSpillMerge OFF (this set's defining feature)
  evap_mode: 0
  infiltration_on: 0
  runoff_ratio_on: 0
  deltat: 31536000                    # 1 yr
  total_cycles: 1
  maxiter: 5
  fdepth: {a: 200, b: 150, fmin: 2}
sweep:
  grids: [2000, 4000]                 # N x N cells (4.0M, 16.0M)
  ranks: [32, 16, 8, 4, 2, 1]         # descending (most cores first)
  reps: 2
  omp_num_threads: 1                  # pure MPI, one rank per core
fsm:
  in_loop: false                      # fsm_on 0 -> FillSpillMerge runs 0 times per run
  note: >
    The depression hierarchy IS still built once at setup on rank 0
    (WTM.cpp GetDepressionHierarchy, gated only on rank==0, not on fsm_on), so it
    shows in the log; but FillSpillMerge itself does not run.
serial_overhead:                      # why wall_s speedup < gw_s speedup at high rank counts
  - "depression-hierarchy build (rank 0, once, unconditional)"
  - "synthetic-grid read + output-raster write"
metrics:                              # CSV columns
  gw_s: "groundwater solve time ('t GW time') -- the bandwidth-bound signal"
  wall_s: "total wall time (includes serial_overhead)"
  snes_iters: "SNES nonlinear iterations"
  mem_*_gb: "per-rank process memory: total / max / min"
  strong_speedup, parallel_efficiency: "WALL-based, relative to n=1"
```

## Reproduce

```sh
source ~/models/WTM/run_env.sh      # modules + wtmtest + OMP_NUM_THREADS=1
cd ~/models/WTM/benchmark/scaling
python3 scaling_study.py --strong 2000 --ranks 32 16 8 4 2 1 --builds after --reps 2
# preserve: cp results.csv agate_no_fsm/results_2026-08-11_grid<N>.csv   (name must match results_2026-*.csv to be tracked)
```

## Results

### grid 2000 (4.0M cells) — GW solve scales to 10.1× at n=32

| n | wall_s | gw_s | GW speedup | GW eff | mem min (GB) |
|---|---|---|---|---|---|
| 1 | 35.3 | 31.70 | 1.00× | 100% | 2.75 |
| 2 | 21.7 | 19.35 | 1.64× | 82% | 1.09 |
| 4 | 13.9 | 11.59 | 2.73× | 68% | 0.58 |
| 8 | 9.8 | 7.61 | 4.17× | 52% | 0.31 |
| 16 | 7.2 | 4.78 | 6.63× | 41% | 0.19 |
| 32 | 5.9 | 3.13 | 10.13× | 32% | 0.12 |

Data: `results_2026-08-11_grid2000.csv`. Contrast the laptop (8-core mobile APU,
2-channel LPDDR5), where the GW solve flat-lined by n≈4–8 (≈2.3× ceiling); here it
keeps gaining through n=32 — the ~4-core wall was a laptop-memory-bus limit.
Caveat: this grid is ~10× larger than the laptop's Esquibel (384k), and bigger
problems strong-scale better regardless, so part of the gap is problem size; a
matched-grid run would isolate memory-channels from size.

### grid 4000 (16.0M cells)

_pending — add `results_2026-08-11_grid4000.csv` and the table when the run lands._
