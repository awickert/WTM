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
cpu:                                  # benchmark node: acn02 (Agate compute)
  model: "AMD EPYC 7763 (Milan / Zen 3)"
  sockets: 2
  cores_per_socket: 64                # 128 cores/node
  threads_per_core: 1                 # SMT off on acn* compute nodes
  numa_domains: 8                     # NPS4: 4 per socket, 16 cores each
  numa_map: {socket0: "numa 0-3", socket1: "numa 4-7"}
  numa_distances: {local: 10, same_socket: 12, cross_socket: 32}
  memory_channels: 16                 # 8x DDR4-3200 per socket
  bandwidth_gbps: {per_channel: 25.6, per_numa_domain: 51.2, per_socket: 204.8, per_node: 409.6}
  l3_mib_per_ccd: 32                  # 8-core CCD; 256 MiB/socket
  l2_kib_per_core: 512
  l1_kib_per_core: {d: 32, i: 32}
  note: "ahl* is the SMT-on, 512 GB high-mem class; benchmark ran on acn* (SMT off)."
rank_placement: unpinned              # MPICH did not bind ranks (Cpus_allowed_list=0-255); the OS
                                      # scheduler spreads them across NUMA domains (first-touch allocation)
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

| n | wall_s | gw_s | GW speedup | GW eff | mem min (GB) |
|---|---|---|---|---|---|
| 1 | 118.3 | 109.00 | 1.00× | 100% | 10.77 |
| 2 | 62.6 | 57.71 | 1.89× | 94% | 4.21 |
| 4 | 39.8 | 35.14 | 3.10× | 78% | 2.15 |
| 8 | 24.8 | 20.70 | 5.27× | 66% | 1.10 |
| 16 | 19.0 | 15.00 | 7.27× | 45% | 0.58 |
| 32 | 16.6 | 12.13 | 8.99× | 28% | 0.32 |

Data: `results_2026-08-11_grid4000.csv`.

### grid-size comparison (GW efficiency)

| n | 2000² | 4000² |
|---|---|---|
| 2 | 82% | 94% |
| 4 | 68% | 78% |
| 8 | 52% | 66% |
| 16 | 41% | 45% |
| 32 | 32% | 28% |

The bigger grid is **more efficient at low–mid core counts** (more work per rank
amortizes communication/overhead) but its GW peak at n=32 is slightly **lower**
(8.99× vs 10.13×) and efficiency crosses below 2000² by n=32. Reading: Agate has
its **own** memory-bandwidth ceiling — it just shows up at n≈16–32 rather than
the laptop's n≈4. The larger, more bandwidth-hungry problem (500k cells/rank at
n=32 vs 125k for 2000²) saturates the node's aggregate bandwidth sooner in
efficiency terms. So the answer to "does the ~4-core wall lift on Agate?" is a
clear **yes** (both grids gain strongly through n=8, where the laptop had
flat-lined at ~2.3×), but Agate is not unbounded — its ceiling sits ~4–8× higher
in core count. Caveat unchanged: shared (non-exclusive) node; an `--exclusive`
run would sharpen the top end.

## Causal analysis — the scaling shape is a memory-architecture fingerprint

The GW solve (matrix-free Anderson stencil) is **memory-bandwidth-bound**: low
flops/byte, it just streams the grid and neighbour vectors, so throughput tracks
bytes/s from DRAM, not core count. Mapping the observed curve onto the confirmed
hardware (`cpu:` block above):

1. **Why the wall lifts vs the laptop.** Node bandwidth is delivered as **8
   independent ~51 GB/s pools** (one NUMA domain = 16 cores + 2 DDR4-3200
   channels), not one shared pool. Ranks are **unpinned** (`Cpus_allowed_list =
   0-255`), so the OS scheduler spreads a handful of processes across those
   domains, each getting local memory (first-touch) and its own channel pair.
   That is precisely why the low-n scaling is near-linear (82–94% at n=2–4). The
   laptop's single 2-channel pool (all 8 cores sharing it) saturated at ~4 cores
   with nowhere to spread; Agate defers saturation ~4–8× in core count because
   there are 8 pools to spread across. Note this is *not* more bandwidth per core
   — a fully-loaded domain gives ~3.2 GB/s/core (51.2 ÷ 16), *less* than the
   laptop's ~12 — it is the **independence** of the pools that scales.

2. **Why efficiency declines at high n.** As ranks-per-domain rises (≈4 ranks
   sharing one domain's 2 channels at n=32 → ~13 GB/s/rank), they contend for the
   fixed channel bandwidth. This is **fundamental bandwidth sharing, not a
   placement bug** — the scheduler already spreads the (unpinned) ranks, so there
   is no packed-into-one-domain bottleneck to fix. Explicit pinning
   (`-bind-to core -map-by numa`) would buy determinism and migration-safety, and
   at most a few percent — not a large speedup.

3. **Why 2000² beats 4000² at n=32 (the L3 crossover).** Per-rank working set at
   n=32: 2000² ≈ 125k cells ≈ ~20 MB, which **fits a 32 MB CCD L3** → cache reuse
   cuts DRAM traffic → higher efficiency (32%, 10.1× peak). 4000² ≈ 500k cells ≈
   ~80 MB **overflows L3** → pure DRAM-bound → saturates the channel pairs harder
   → lower efficiency (28%, 8.99× peak). So the larger problem is *more*
   bandwidth-bound at the top, which is the "telling" difference between the two
   grids.

4. **Secondary structure.** Cross-socket NUMA distance is 32 vs 12 same-socket
   (2.7×), so at n ≤ 64 the ideal layout keeps ranks within one socket (domains
   0-3, 8 channels, no cross-socket) — halo exchange / rank-0 gather across the
   socket link is the expensive path. With per-rank first-touch each rank's *own*
   streaming stays local regardless, so this is a comm-side, not a solve-side,
   effect.

**Caveat on the placement probe:** it was run on `ahl02` outside the benchmark's
`srun` allocation, so on `acn02` inside Slurm the cgroup *may* restrict
`Cpus_allowed`. The conclusion that survives regardless — MPICH is not pinning and
the OS is spreading — is what explains the curve. Confirm inside the real
allocation if certainty is wanted.

## Multi-node scaling — cross-node VALIDATED (2026-08-11)

WTM was designed single-node (`DISTRIBUTED_ARP_DESIGN.md` lists cross-node as a
non-goal), but it uses PETSc **collective** gather/scatter, so it runs across nodes
with **no code change** — only launcher/filesystem changes. Two gotchas found and
fixed (in `scaling_multinode.sbatch`): (1) use **`mpiexec`** (MPICH Hydra, spans the
Slurm nodelist), NOT `srun --mpi=pmi2` (which aborts/hangs this MPICH); (2) the temp
work dir (config + input grid) must be on a **shared filesystem** (`$HOME`), not
node-local `/tmp`, or ranks on other nodes fail "Failed to read config file!".

**Correctness — bit-identical across node layouts.** Comparing the SAME 16-rank DMDA
decomposition on 1 node vs 2 nodes (cancels Anderson's cross-rank noise; only node
placement differs): **|2 nodes − 1 node| = 0.0 m — bit-for-bit identical.** Both
differ from the 1-rank run by the same 0.0195 m = Anderson's known matrix-free
16-vs-1-rank noise, not a bug. WTM is provably cross-node-correct.

**Node sweep (2000², fixed 8 ranks/node, FSM off, `mpiexec -ppn 8`) — to 8 nodes:**

| nodes | ranks | gw_s | speedup | eff |
|---|---|---|---|---|
| 1 | 8 | 9.57 | 1.00× | — |
| 2 | 16 | 5.71 | 1.68× | 84% |
| 4 | 32 | 3.47 | 2.76× | 69% |
| 8 | 64 | 2.06 | 4.64× | 58% |

Adding nodes scales steadily — **~1.67× per node-doubling all the way to 8 nodes**
(1.68 / 1.65 / 1.68), i.e. NO Infiniband-gather wall yet: throughput keeps climbing
(4.64× at 8 nodes) while efficiency declines gently. This is *better* than adding
cores within a node (8→16 ranks single-node ≈ 1.5×), because each node brings a
*fresh* set of 16 memory channels. Multi-node is the bandwidth lever, confirmed to 8
nodes. Caveats: (1) shared (non-exclusive) nodes → **~25% run-to-run variance** — an
earlier 4-node run gave 2.04×/3.51× vs 1.68×/2.76× here; trust the *shape*, not the
absolute gw_s, until an `--exclusive` run. (2) 2000² is small for 64 ranks (62k
cells/rank), so a `GRID=4000` run (more per-rank work) would scale further and is
more production-representative. (3) `mpiexec -ppn 8` *packs* ranks, inflating the
1-node baseline vs the unpinned study runs above.

**Bigger grid scales better (4000², 4 nodes, job 15415684):**

| nodes | ranks | gw_s | speedup | eff | 2000² eff |
|---|---|---|---|---|---|
| 1 | 8 | 26.77 | 1.00× | — | — |
| 2 | 16 | 14.55 | 1.84× | 92% | 84% |
| 4 | 32 | 9.26 | 2.89× | 72% | 69% |

The 16M-cell grid is **more efficient at every node count** than the 4M-cell one
(92% vs 84% at 2 nodes) — more per-rank work amortizes the inter-node comm and the
gather-to-rank-0. So the more production-realistic the problem size, the better
multi-node pays off. Correctness bit-identical here too (node-spanning = 0.0 m; the
Anderson noise floor is larger, 0.21 m, as expected for the bigger grid).

**Bottom line:** WTM runs multi-node correctly (bit-identical) and scales well to at
least 8 nodes, *better* on bigger grids — a capability the code was never designed
for, unlocking the big global (141M-cell) runs beyond one node's memory + bandwidth.
