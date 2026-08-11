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

Feeds #79 (fraction-vs-grid-size table) and `benchmark/FSM_PARALLEL_DESIGN.md`.
