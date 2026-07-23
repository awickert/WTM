# Design Note: Distributing ArrayPack for Single-Node Many-Core Scaling

**Date:** 2026-07-23
**Branch:** `solver-optimization-2`
**Status:** DESIGN — not yet implemented. For discussion before any code.
**Authors:** Andy Wickert + Claude

This note proposes how to remove WTM's dominant production bottleneck: the full-grid
replication of `ArrayPack` on every MPI rank. It is a design for discussion, not a
commitment. See `SOLVER_NOTES.md` for the measurements that motivate it.

---

## 1. Problem statement

WTM is almost always run **single-node, many-core** on MSI (msismall: 128 cores/512 GB;
msilong: 32 cores/128 GB cap). Production grids reach ~141M cells.

`ArrayPack` holds **34 full-grid 2D arrays** (25 `float` + 9 `double`) plus 3 label/flowdir
arrays, each **replicated on every rank**: ~26 GB/rank at 141M cells. On a single shared-RAM
node this caps usable ranks at:
- **msismall:** ~15 of 128 cores (512 GB ÷ ~26 GB + headroom)
- **msilong:** ~4 of 32 cores (128 GB cap)

So the replicated data model — not compute, not communication — is what strands the many cores
the runs are explicitly provisioned for. The groundwater solve itself already scales well
(measured ~3.6× at n=8; see `SOLVER_NOTES.md`). The goal is to let it use the whole node.

**Non-goal:** cross-node/InfiniBand scaling. Runs are single-node, so the full-grid
`MPI_Allreduce` is shared-memory and cheap; the "allreduce hoist" is explicitly de-prioritized.

---

## 2. The central constraint: FillSpillMerge is global and serial

FillSpillMerge (FSM) and `GetDepressionHierarchy` (dephier) are **inherently global** algorithms:
depressions, spill points, and watersheds are global topological structures that cannot be
computed from a subdomain. Both are built directly on full-grid `richdem::Array2D`, contain no
MPI, and touch ~10 `arp` arrays across the entire grid:

| array | approx. uses in FSM | role |
|-------|--------------------:|------|
| topo | 40 | elevation surface |
| wtd | 31 | water table (read + written) |
| runoff | 23 | surface water routed |
| label / final_label | 16 / 3 | depression membership |
| cell_area | 15 | volume accounting |
| porosity | 13 | storage |
| fdepth | 8 | — |
| flowdirs | 6 | routing |
| slope | 3 | — |

Rewriting FSM to be distributed is research-grade (parallel priority-flood) and **out of scope**.

**Therefore the design is "distribute the solve, gather for FSM," not "distribute everything."**
FSM runs ~once per simulated year with many GW solves between; its amortized cost is small (see
SOLVER_NOTES). So gathering the full grid once per FSM event is an acceptable, rare cost.

---

## 3. Proposed architecture

Split `ArrayPack`'s arrays into three classes by access pattern:

### Class A — Distributed (solve-hot, cell-local or nearest-neighbor)
Carried as DMDA-owned + ghost data, one subdomain per rank. These are the arrays the GW solve
and per-cell recharge/evap loops touch, never needing global structure:
`wtd`, `wtd_old`, `wtd_mid`, `topo`, `ksat`, `fdepth`, `porosity`, `rech`, `runoff`,
`effective_storativity`, `transmissivity`, `infiltration_array`, and the per-cell forcing fields
(`precip`, `evap`, `open_water_evap`, `winter_temp`, `slope`, `runoff_ratio`, `vert_ksat`,
`land_mask`). The `_start`/`_end` transient endpoints distribute the same way.

### Class B — Gathered-for-FSM (transiently full-grid on the FSM rank)
The subset FSM/dephier require as full grids: `topo`, `wtd`, `runoff`, `porosity`, `fdepth`,
`cell_area`, `label`, `final_label`, `flowdirs`, `slope`. Most overlap Class A (they are
distributed for the solve and *gathered* into a full-grid scratch only around the FSM call).
`label`/`final_label`/`flowdirs` are FSM-internal scratch and can live only in the gathered form.

### Class C — Small / replicated (unchanged)
1D lat-dependent geometry (`cell_area`, `cellsize_e_w_metres`: length `ncells_y`, cheap to
replicate) and the scalar accumulators. No change.

### Data flow per cycle
```
[distributed arp]  --solve loop (maxiter GW solves, all distributed, ghost exchange)-->  [distributed arp]
        |                                                                                       |
        |  (once per cycle, only when FSM runs ~1/yr)                                            |
        +--gather Class B to full grid on FSM rank--> [FSM/dephier serial] --scatter wtd back-->-+
```

Mechanism: PETSc `DMDANaturalToGlobal` + `VecScatterCreateToZero` (or `DMDACreateNaturalVector`
gather) to assemble/redistribute a DMDA field to/from a single rank in natural (row-major) order
that matches `richdem::Array2D` layout. Reusable scatter contexts, created once.

---

## 4. Memory outcome

| | current | proposed |
|---|---|---|
| per-rank steady state (141M) | ~26 GB (all arrays) | ~subdomain + halos: tens–hundreds of MB |
| FSM rank transient | (already full) | + Class B full grids ~5 GB, only during FSM |
| usable cores, msismall | ~15 / 128 | up to 128 |
| usable cores, msilong | ~4 / 32 | up to 32 |

The FSM rank needs ~5 GB transient headroom (only the Class B arrays, not all 26 GB); other ranks
stay lean. Gather only the arrays FSM reads; scatter back only `wtd` (the only field it changes
that the solve consumes).

---

## 5. Blast radius (what this touches)

This is invasive. Honest inventory of ripple sites:

1. **`ArrayPack` itself** — arrays become distributed handles (DMDA `Vec` + local array views),
   not `richdem::Array2D`. Accessor `arp.wtd(i,j)` semantics change from global to local(owned)
   indices. **This is the largest and most error-prone part**: ~32 full-grid serial loops
   outside the solve (recharge/evap in `WTM.cpp`, diagnostics in `irf.cpp`/`PrintValues`,
   `set_starting_values`) currently index `[0..ncells)` and must become owned-range loops.
   Step 1 (already committed) converted `set_starting_values` and `PrintValues`; the rest follow
   the same pattern.
2. **Initialization / loading (`irf.cpp`)** — inputs are currently read full-grid via GDAL on
   every rank (no rank guard). Options: (a) rank-0 reads then scatters, or (b) parallel GDAL
   reads of subdomain windows. (a) is simpler and I/O is one-time; prefer it first.
3. **I/O output (`saveGDAL`, rank-0)** — already rank-0-guarded; needs a gather before write
   (same Class B gather mechanism).
4. **FSM handoff (`WTM.cpp`)** — insert gather-before / scatter-after around the FSM + dephier
   calls.
5. **Cell geometry (`run_dephier.cpp`)** — 1D, per-row; stays replicated (Class C). Low risk.

---

## 6. Phasing (each phase independently verifiable, ghost + mass-balance tests as gates)

- **Phase 0 (done):** owned-only diagnostics + scalar reduce (`set_starting_values`,
  `PrintValues`), with the mass-balance regression. First step; proves the owned-range pattern.
- **Phase 1:** introduce the gather/scatter helpers (Class B ⇄ full grid) and wrap FSM + I/O
  with them, *while arp is still replicated*. No behavior change — this lands and tests the
  communication plumbing in isolation. Gate: ghost + mass-balance + bit-identical output vs HEAD.
- **Phase 2:** distribute one array end-to-end (`wtd`) — allocate as DMDA-backed, convert all its
  access sites to owned-range, gather only for FSM/I-O. Gate: same tests, single-array blast
  radius keeps it debuggable.
- **Phase 3:** distribute the rest of Class A in small batches (forcing fields, then derived
  fields), each batch a commit with the tests as gates.
- **Phase 4:** switch loading to rank-0-read-then-scatter; drop the now-unnecessary full-grid
  allocations on non-FSM ranks. This is where the memory win actually lands.

Order matters: the memory benefit only fully arrives at Phase 4, but Phases 1–3 are where
correctness is established incrementally. Do not skip to Phase 4.

---

## 7. Risks and open questions

- **Bandwidth ceiling (measured caveat):** unblocking 15→128 cores is sublinear — 128 cores share
  ~400 GB/s and the stencil is bandwidth-bound. Expected win is large but NOT 8×. Worth measuring
  the single-node saturation curve on MSI before/after to set expectations honestly.
- **richdem coupling:** FSM's reliance on `richdem::Array2D` means the gathered Class B arrays must
  present exactly that type/layout. The natural-ordering gather must reproduce row-major
  `Array2D` indexing precisely — a likely source of subtle bugs; test with the ghost harness.
- **Accessor ergonomics:** `arp.wtd(i,j)` global→owned index change is a footgun across ~32 sites.
  Consider a distinct accessor name (e.g. `arp.wtd_local(i,j)`) so global-index misuse fails to
  compile rather than silently reading the wrong cell.
- **Transient `_start`/`_end` endpoints** double the Class A array count during interpolation;
  confirm they distribute cleanly and don't reintroduce a memory spike.
- **Is rank-0-read-then-scatter fast enough** at 141M cells, or is parallel-windowed GDAL needed?
  One-time cost per run; measure before over-engineering.

---

## 8. Recommendation

Proceed **Phase 1 first** (gather/scatter plumbing around FSM + I/O, arp still replicated): it is
the lowest-risk way to build and prove the one genuinely new mechanism (natural-ordering
gather/scatter) in isolation, with bit-identical output as the gate. Only then distribute arrays
(Phases 2–4), where the memory payoff lands. Take one MSI single-node scaling measurement
(pre-change) to anchor the expected speedup before investing in Phases 2–4.
