# FSM + DH cost fraction — Esquibel (384,703 cells), single node

**Task #77.** How much of a coupled (`fsm_on 1`) equilibrium cycle is the serial rank-0
surface-water work (FillSpillMerge + depression-hierarchy build) versus the distributed
groundwater solve? This sizes the parallelize-FSM prize (#80) and the Amdahl ceiling for
coupled multi-node runs.

## Setup

- Binary: `build/wtm.x`, branch `bdf2-adaptive-dt` (Release `-O3`).
- Config: `eq_awickert.cfg` — `fsm_on 1`, `runoff_ratio_on 1`, `evap_mode 1`,
  `infiltration_on 0`, `deltat 604800` (1 wk), `cells_per_degree 900`, region Esquibel
  (853 × 451 = **384,703 cells**).
- Solver: `-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6`, `OMP_NUM_THREADS=1`,
  `PROJ_DATA=/usr/share/proj`. Single node, 8 MPI ranks (this laptop).
- Timers (WTM.cpp): `t GW time` = maxiter sub-step solve + gather (292–401); `t FSM time`
  = the `FillSpillMerge` call only, rank 0 (413–424); `t WTM update time` = the OUTER
  per-cycle total (dominated by GW, so ≈ GW); DH build (`GetDepressionHierarchy` →
  "Outlets found in") runs ONCE in `run()` setup, not per cycle.

## Result

Per-cycle wall time (one node, 8 ranks):

| cost                  | typical / median | mean over equilibration | notes                              |
|-----------------------|------------------|-------------------------|------------------------------------|
| **GW solve**          | ~2.5–3.8 s       | 3.84 s                  | ~99.7% of the cycle                |
| **FSM (rank-0)**      | 2.4 × 10⁻⁶ s     | 5.5 × 10⁻³ s            | rare routing spikes up to ~4 s     |
| Set recharge          | 0.003–0.006 s    | 0.004 s                 | distributed                        |
| **DH build**          | **one-time <0.05 s** | —                   | static terrain hierarchy, reused   |

- **FSM fraction: 0.00007% at cold start (upper bound: most water to route), 0.142% mean**
  over a full ~790-cycle equilibration (mean pulled up by a few cycles that route a real
  water body; the median cycle routes ~nothing and FSM early-exits in ~2.4 µs).
- **DH build is a one-time setup cost** (<0.05 s at 384k) — confirmed `Outlets found in`
  prints exactly once across a 6-cycle run.
- PETSc `-log_view` (6 cold cycles, n=8): SNESSolve = 15.43 s of 16.65 s total (**92.6%**),
  SNESFunctionEval 7.41 s (matrix-free residual). The cost is the solve, and it is
  memory-bandwidth-bound (see `benchmark/scaling/agate_no_fsm/README.md`).

## Interpretation — reframes the parallelize-FSM motivation (#80)

At Esquibel scale, single node, **FSM compute is not a bottleneck** — it is ~0.14% of
per-cycle wall time and DH build is a one-time <0.05 s. So the case for parallelizing FSM
(#80) does NOT rest on FSM compute being slow. It rests on the two things this single-node
test cannot see:

1. **The per-cycle gather to rank 0** (`gatherToZero`) — single-node it is a cheap memory
   copy (all VecScatter comm here ≈ 0.9 s of 16.65 s, and that is mostly the solve's halo
   exchanges, not the FSM gather). Multi-node it becomes an all-to-one Infiniband transfer
   every cycle → the real Amdahl term.
2. **Global scale** (up to 141M cells) where FSM's O(N log N) grows relative to the
   bandwidth-bound GW, and where the GW solve is spread thin across many nodes so even a
   small serial fraction becomes visible.

**Consequence for sequencing:** the parallelize-FSM payoff is a large-node-count /
global-scale / gather-driven phenomenon, not an Esquibel-scale one. The next measurement
that would actually size it is a **coupled multi-node run on MSI** (the gather cost) and/or
a **global-scale single-node fraction** (#78) — not more single-node Esquibel work.

Feeds #79 (fraction-vs-grid-size table, below) and `benchmark/FSM_PARALLEL_DESIGN.md`.

## Fraction vs grid size (#78/#79) — single-node coupled fsm_on 1

Real topography at every scale via TILING the 384k Esquibel stack (`make_tiled.py`; each
tile a full real domain). Island/Esquibel = this laptop; tiled 1.5M–139M = MSI Agate (acn
EPYC 7763, 32 ranks, msilong; job 15451310, `fsmcost_tiled.sbatch`). GW absolute times are
not cross-hardware comparable — the FSM/DH FRACTION and the DH-build growth are the point.

| cells | DH-build (once) | FSM /cyc (median) | GW /cyc | FSM fraction | where |
|------:|----------------:|------------------:|--------:|-------------:|-------|
| 8,775 (island) | <0.05 s | <0.05 s | ~0.1 s | below timer | laptop |
| 384,703 (Esquibel) | <0.05 s | 2.4e-6 s (0.14% mean) | ~3.8 s | ~0.14% mean | laptop |
| 1,538,812 | 0.20 s | 8.0e-7 s | 2.32 s | 0.00003% | MSI |
| 13,849,308 | 1.70 s | 4.3e-5 s | 23.36 s | 0.00018% | MSI |
| 55,397,232 | 8.60 s | 4.5e-5 s | 127.67 s | 0.00003% | MSI |
| 138,877,783 | 30.40 s | — (cold solve aborted) | — | — | MSI |

**Read of the curve:**
- **FSM per-cycle is negligible at every scale** — microseconds, fraction ~1e-4 % and flat.
  The median cycle routes ~no water (FSM early-exits); it is never the bottleneck.
- **DH-build grows ~O(N log N) but is ONE-TIME setup** (0.2 → 1.7 → 8.6 → 30.4 s over
  1.5M → 139M): each ~2.5–9× cell increase costs ~3.5–8.5× DH time (mildly super-linear =
  the log factor). Rebuilt only decadal–centennial (topography changes slowly), so amortized
  to ~0 per cycle — see `benchmark/GLOBAL_SCALING_DESIGN.md`.
- **GW dominates and scales ~linearly with N** (2.32 → 127.67 s, ~55× for 36× cells; some
  super-linearity from cache/NUMA spill) — ~99.7%+ of every coupled cycle.
- **139M cold solve aborted** (`std::runtime_error` after 2.24 h; DH-build completed at
  30.4 s, zero GW solves finished, NOT OOM — node had 358 GB free, no swap). This is a
  cold-start solver-convergence limit of the *tiled* domain (361 identical hard-depression
  tiles drained from wtd=0 at once), NOT an FSM/DH cost — a separate solver-scaling question.
  A real (non-repeated) global domain, or a warm start, would not hit this.

**Conclusion (single-node, tasks A/#78 done):** FSM+DH compute is not the coupled-model
bottleneck at any single-node scale to 55M; the GW solve is. This confirms the reframe —
the parallelize-FSM driver (#80) is NOT compute. It is (1) the multi-node GATHER to rank 0
(task B) and (2) global-scale MEMORY (the DH-build's full-grid DEM+labels;
`GLOBAL_SCALING_DESIGN.md`). Task B measures (1).

## Task B: the multi-node gather is NOT a growing Amdahl term (jobs 15493615 @13.8M, 15586370 @55M)

Fixed grid 13.8M (tiled), coupled fsm_on 1, node sweep 1/2/4/8 (8 ranks/node) on msilarge,
with a dedicated `t gather time` timer (WTM.cpp) around the per-cycle all-to-one gather to rank 0:

| nodes | ranks | GW/cyc (s) | gather/cyc (s) | FSM/cyc (s) | gather % of cycle |
|------:|------:|-----------:|---------------:|------------:|------------------:|
| 1 | 8  | 104.4 | 0.044 | 5e-5 | 0.04% |
| 2 | 16 | 85.8  | 0.052 | 2e-5 | 0.06% |
| 4 | 32 | 26.2  | 0.025 | 1e-6 | 0.09% |
| 8 | 64 | 25.9  | 0.037 | 1e-6 | 0.14% |

**The gather is FLAT (~0.04 s) across node counts** -- gathering 13.8M doubles (~110 MB) to
rank 0 over Infiniband is ~40 ms and does NOT grow with nodes. It is 0.14% of the cycle even at
8 nodes; FSM stays microseconds. So neither the serial FSM (A) nor the gather (B) is a real cost
up to 8 nodes / 13.8M -- the empirical #80 case is WEAKER than the design note assumed.
CAVEAT keeping #80 alive: GW saturates by 4 nodes here (26->26 s, 4->8 nodes) because 13.8M is
too small to spread over 64 ranks. The Amdahl wall only appears in the UNREACHED regime -- a
global-size grid on hundreds of nodes, where GW/cyc is driven to seconds while the fixed gather
(~7.5 GB to rank 0 at 30" global) + serial FSM (O(N log N) on ~1e9 cells) finally dominate. B
cannot reach that regime, so it neither confirms nor kills #80; it does show the gather is
harmless at all currently-reachable scales. Data: fsmcost_multinode_results.csv.

### Confirmed at 55M (job 15586370)

Same harness, fixed grid 55.4M (tiled 12x12), node sweep 1/2/4/8 (8 ranks/node):

| nodes | ranks | GW/cyc (s) | gather/cyc (s) | gather % of cycle |
|------:|------:|-----------:|---------------:|------------------:|
| 1 | 8  | 487.7 | 0.314 | 0.06% |
| 2 | 16 | 277.9 | 0.165 | 0.06% |
| 4 | 32 | 154.2 | 0.306 | 0.20% |
| 8 | 64 | 119.3 | 0.235 | 0.19% |

At 4x the grid the finding holds and is in fact **stronger**: the gather stays FLAT (~0.2-0.3 s,
~460 MB to rank 0 over Infiniband, no growth with node count) and is <0.3% of the cycle even at
8 nodes. Crucially, unlike 13.8M the GW solve here does **not** saturate by 8 nodes (487->119 s
= 4.1x, still scaling), so the gather is now measured against a solve that is *still shrinking*
with nodes -- and it remains negligible. Two grid sizes (13.8M, 55M) x four node counts now
agree: **the per-cycle all-to-one gather to rank 0 is not a growing Amdahl term at any
currently-reachable scale.** This is a positive result for the massive-scale goal -- the
suspected multi-node serial bottleneck (the gather) is not one in practice; the remaining
multi-node concern is memory (the full-grid DH build), not gather time. Data (both grids):
fsmcost_multinode_results.csv.

## FSM-cadence overhead: FSM every sub-step vs every maxiter (2026-08-21, laptop)

**Motivating question.** To kill the marginal-lake / lakeshore N-dependence (Issue #6) we want *tight*
coupling — run FillSpillMerge every GW sub-step instead of once per cycle (a cycle = `maxiter` sub-steps;
here `maxiter 50`). Is that affordable?

**Setup.** Esquibel (384,703 cells), WARM start (`results/abl_000000020.tif` as `starting_wt`), n=8,
`OMP_NUM_THREADS=1`, `-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6`. Same 10 simulated sub-steps
each way: **tight** = `maxiter 1 × 10 cycles` (10 FSM calls) vs **loose** = `maxiter 10 × 1 cycle`
(1 FSM call). Wall from `date +%s.%N`.

| mode  | FSM cadence            | wall (10 sub-steps) |
|-------|------------------------|---------------------|
| loose | every 10 sub-steps     | 2.146 s             |
| tight | every sub-step         | 2.239 s             |

**Result: ~4% wall overhead for FSM-every-sub-step**, and that ~4% is the per-cycle bookkeeping + gather
done 10× — the FSM *compute* summed to ~0 (microseconds; a warm state routes almost nothing). Single run
each, so read it as "small single-digit %," not a precise figure. **Tight coupling is affordable
single-node.**

**What controls FSM runtime** (the µs→~4 s range seen above): the *volume* of surface water routed and how
far it cascades up the depression hierarchy (fill→spill→merge). Near-equilibrium/dry → nothing to route →
early-exit in ~µs; a cold start / wet event / large water body filling and spilling across many hierarchy
levels → seconds. It is data-dependent, not grid-size-dependent (hence the flat fraction across scales).
This *reinforces* tight coupling: FSM every sub-step routes ~1/50th the water per call, so each call is more
likely to early-exit — it stays in the cheap regime.

**Consequence for parallelize-FSM (#80).** Since FSM compute is negligible, parallelizing FSM is not a
compute-speed play. Combined with the multi-node gather study above (gather stays flat ~0.2–0.3 s and is
<0.3% of a cycle even at 8 nodes / 55M — *not* a growing Amdahl term), the real driver for parallel FSM at
global scale is **memory** (the full grid must be replicated on rank 0 for serial FSM+DH), with the
tight-coupling gather (×`maxiter` more gathers) a secondary factor (~10% at 55M / 8 nodes, vs ~4% laptop).
Net: tight coupling is free-enough now (regional/single-node); parallel FSM is the enabler for *global*
tight coupling, driven mainly by the replicated-grid memory ceiling.
