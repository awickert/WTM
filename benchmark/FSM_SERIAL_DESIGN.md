# Design Note: Why FillSpillMerge stays serial on rank 0, and what the gather costs

**Date:** 2026-07-29
**Branch:** `mpi-ghost-fix`
**Status:** ANALYSIS — records the reasoning behind keeping FSM serial; no code change proposed.
**Authors:** Andy Wickert + Claude

Companion to `DISTRIBUTED_ARP_DESIGN.md` (the gather is that design's Phase-1 primitive) and
`GLOBAL_SCALING_DESIGN.md`. Answers two questions raised in review: **is there any advantage in
parallelizing FSM?** and **how expensive is gathering the distributed tiled grids to rank 0 to feed
it?** Line references are to `src/fill_spill_merge.hpp` and `src/WTM.cpp` as of this branch.

---

## 1. What FSM actually is

FillSpillMerge is **not a graph algorithm with a few grid sweeps bolted on**. It is a set of **grid
algorithms stitched together by a lightweight depression-tree traversal**. The tree traversal is the
cheap glue; the per-cell work is the substance. Five stages, run per cycle on rank 0
(`FillSpillMerge`, `fill_spill_merge.hpp:181`):

1. **`ResetDH`** (1766) — zero the per-depression water scalars. O(#depressions). Trivial.
2. **`MoveWaterIntoPits`** (296) — route all surface water downhill into pit cells. **Two modes:**
   - *infiltration OFF:* one O(N) sweep dumping each cell's water into its leaf depression via the
     label (a reduction). Cheap.
   - *infiltration ON:* a **full flow-accumulation** — dependency counts (OpenMP-parallel), peak
     finding, a BFS in flow-direction order moving water cell-to-cell and infiltrating a fraction into
     `wtd` each hop (`CalculateInfiltration`). Genuinely grid-heavy and dependency-ordered.
3. **`CalculateWtdVol`** (514) — O(N) per-cell walk up the tree via `final_label` accumulating
   below-ground storage, then O(#depressions) tree passes.
4. **`MoveWaterInDepHier`** (720) — the **overflow cascade**, a post-order tree traversal. Cheap
   **only when infiltration is off** (the `jump_table`, near-linear in #depressions, shuffles
   `water_vol` scalars between depression records — the true "light graph"). With **infiltration on**
   it calls `OverflowInto → MoveWaterInOverflow` (859), which routes spilled water **cell-by-cell
   across the grid**.
5. **`FindDepressionsToFill`** (1271) → **`FillDepressions`** (1450) — for each depression holding
   water, a **priority-flood from the pit cell**: a PQ grows a flood front cell-by-cell, accumulating
   volume via the Water-Level Equation until the water fits, then `BackfillDepression` sets `wtd`.
   Cost scales with total **lake-cell** count, not the whole grid.

**The infiltration flag is the hinge.** OFF: the heavy work is one label-reduction, one storage sweep,
and a priority-flood over lake cells (≪ N unless very lake-rich); the overflow cascade is graph-only.
ON: two full grid flow-routing passes (pits + overflow), and this is exactly the configuration already
pinned to the serial rank-0 path (`WTM.cpp:60–63` warns it is not parallel-accelerated).

Note this is WTM's **groundwater-coupled fork** of FSM. The infiltration and `wtd`-storage-in-
depressions logic is WTM's addition; r-barnes' original is the pure surface-water algorithm. The
infiltration paths are where WTM diverges and where the grid cost concentrates.

## 2. Is there any advantage in parallelizing FSM?

**Keep the graph glue serial** — the overflow cascade and fill-tree traversal are a sequential
dependency structure, small memory, cheap compute. That is the correct home for them, and the code
confirms it. The graph is *not* where the cost lives.

The only real parallelization candidates are the **grid** stages, and they differ sharply:

- `CalculateWtdVol` and infiltration-off `MoveWaterIntoPits` — trivial per-cell reductions into
  depressions (like the DH marginal-volume pass). Easy, but low absolute cost.
- `FillDepressions` — **independent across disjoint lakes** → embarrassingly parallel *across*
  depressions, though each lake's fill is a serial flood. The one interesting target, and only if the
  landscape is lake-rich.
- Infiltration-on `MoveWaterIntoPits` / `MoveWaterInOverflow` — parallel flow-accumulation, the
  genuinely hard case, and already the serial-bound mode.

Three reasons it is not worth doing now:

1. **Amdahl.** With infiltration off (a common production mode) FSM is light next to the iterative GW
   solve — parallelizing a small fraction buys little. Gate any work on a measured FSM wall-time
   fraction, taken both infiltration-on and -off.
2. **Determinism cost.** `FillDepressions` has explicit tie-break ordering ("equal elevation →
   most-recently-added popped first… need not match the DH ordering"), and `MoveWaterIntoPits` routing
   order matters. Tiling these reintroduces reduction/order sensitivity at exactly the thresholds that
   flip lakes — the rank-dependence the serial-on-rank-0 design currently *prevents*, and which the
   serial≡parallel equilibrium result (see `examples/island_equilibrium/`) depends on. A parallel FSM
   would have to re-earn determinism with the same deterministic-tie-break discipline as the DH's
   Phase-C outlet sort.
3. **The real driver is memory, not speed** (next section).

## 3. How expensive is the gather to rank 0?

Cheap. The cost is not the transfer — it is that rank 0 must **hold** the whole grid.

**What crosses the wire each cycle.** Only the dynamic field(s). The static FSM inputs — `topo`,
`label`, `final_label`, `cell_area`, `porosity`, `flowdirs` — are loaded on rank 0 at init and the DH
is built there (`WTM.cpp:76, 242, 499`); they are **resident on rank 0 and never gathered again**.
Per cycle, before FSM:

- **1 gather: `wtd`** — `gather_wtd_to_all` (`WTM.cpp:348`) → `gatherToZero`
  (`transient_groundwater.cpp:659`). Despite the legacy name it gathers **to rank 0 only, no
  broadcast**.
- **+0–2 more**, config-dependent — `gather_runoff_to_zero` / `gather_sink_removed_to_zero`
  (`transient_groundwater.cpp:684, 737`) only under distributed recharge / the sub-surface sink.
- **1 scatter back** after FSM — `scatter_into_owned` → `scatterFromZero` (`WTM.cpp:388`), only under
  distributed recharge.

So a typical cycle is **one gather + one scatter of an N-cell `double` array** (≤3 gathers in the
heaviest config). The PETSc scatter context (`DMDAGlobalToNatural` + `VecScatterToZero`) is built once
(`CreateSNES.cpp:31`) and reused — no per-cycle setup. See `src/dmda_gather.hpp`.

**Cost.** Volume ≈ N × 8 bytes landing on rank 0.

- **Single node** (WTM's actual MSI deployment): a shared-memory copy, memory-bandwidth-bound. At
  N ≈ 10⁷ that is ~80 MB moved a couple of times — sub-millisecond to millisecond at 10–100 GB/s.
  **Negligible** next to FSM's per-cell floods, invisible next to the GW solve.
- **Multi-node** (future global run): a real interconnect collective, all tiles → one rank. At global
  30″ (N ≈ 10⁹) ~8 GB funnels into rank 0 per gather; at ~10 GB/s, ~sub-second to a second per gather,
  a few seconds/cycle at worst — still likely small next to a full implicit solve cycle.

**The reframe.** The gather's *time* is cheap in both regimes. What is expensive is that FSM forces
rank 0 to **hold** ~7 full-grid arrays (`topo`, `label`, `final_label`, `wtd`, `runoff`, `porosity`,
`cell_area`) at once. *That* is the replicated-memory ceiling on global grid size — the same ceiling no
matter how fast the gather is (`DISTRIBUTED_ARP_DESIGN.md` §1; `GLOBAL_SCALING_DESIGN.md`).

## 4. Bottom line

There is **no speed case** for distributing FSM to avoid the gather — the gather is nearly free, and
FSM's serial parts are cheap and determinism-preserving. The **only** driver that would ever justify a
parallel FSM is **memory residency at global scale**: the day N outgrows one node's RAM, rank 0 can no
longer hold the full grid, and FSM's grid stages (flow-accumulation, lake-fill) must be distributed —
built to preserve cross-rank determinism. Until then, gather-to-rank-0 + serial FSM is the right, cheap
design.
