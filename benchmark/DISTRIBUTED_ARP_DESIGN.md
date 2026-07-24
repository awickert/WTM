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
- **Accessor ergonomics — DECIDED:** distributed arrays use a distinct accessor name,
  `arp.wtd_local(i,j)` (owned/local indices), rather than reusing the global `arp.wtd(i,j)`.
  Rationale (Andy, 2026-07-23): clearer to the reader, and a global-index misuse then fails to
  *compile* rather than silently reading the wrong cell. Apply this naming to every distributed
  (Class A) array as it is converted.
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

---

# Addendum (2026-07-23): the solve-intake rewrite (the "flip")

Phases 1, 2a, 2b are committed and bit-identical. This addendum specifies the remaining,
correctness-critical work discovered by tracing the code: how the groundwater solve gets its
data, and how to feed it once `ArrayPack` is allocated only on rank 0.

## A. What the solve actually reads and writes from `arp`

The solve (`FanDarcyGroundwater::update`) never touches `arp` inside PETSc; it touches it at four
boundary sites, all of which currently assume `arp` is full on every rank:

| site | reads from `arp` | writes to `arp` |
|------|------------------|-----------------|
| `scatter_static_fields` (init) | topo, fdepth, ksat | — |
| `populate_DMDA_array_pack` (init) | cellsize_e_w_metres, land_mask, porosity | — |
| `set_starting_values` (per solve) | wtd, rech, porosity, land_mask, cell_area | (accumulators only) |
| rech/starting_wtd fill (per solve) | rech, wtd, porosity | — |
| copy-back (per solve) | topo, land_mask, cell_area, porosity | wtd (owned range) |

Two access classes fall out:

- **Static intake** — topo, fdepth, ksat, porosity, land_mask, cellsize_e_w_metres, cell_area.
  Loaded once (topo also changes each transient cycle). Already partly in DMDA vectors
  (topo/fdepth/ksat local ghost vecs; mask/porosity/cellsize global vecs).
- **Per-cycle intake** — `rech` (produced by the recharge loop) and `wtd` (from the prior cycle).

## B. Target end state

`arp` (all 34 full-grid arrays) is allocated **only on rank 0**. The solve reads and writes
**only DMDA-backed data** (never `arp`). The two data models meet at exactly two handoffs:

```
rank 0: load / UpdateTransientArrays / recharge / FSM / diagnostics / output   (full-grid serial)
            |  scatterFromZero(rech), scatterFromZero(wtd)   ^  gatherToZero(wtd)
            v                                                 |
all ranks: DMDA static vecs (scattered once at init) + distributed GW solve    (owned + ghost)
```

- **Static intake:** at init, scatter each static field rank-0→DMDA once (topo/fdepth/ksat already
  do this from `arp`; change the source to be rank-0-only and add porosity, land_mask, cellsize,
  cell_area). `set_starting_values`, the rech/starting_wtd fill, and copy-back are rewritten to
  read these DMDA-local arrays via the `*_local(i,j)` accessors — never `arp`.
- **Per-cycle intake:** the recharge loop runs on rank 0 producing full `rech`; `scatterFromZero`
  distributes it into the solve's `rech_vec`. `wtd` is `scatterFromZero`'d in before the solve and
  `gatherToZero`'d out after (the primitive built and tested for this).
- `cell_area`/`cellsize_e_w_metres` are 1-D per-row (length `ncells_y`); cheap to keep replicated
  on all ranks (Class C) so owned-range loops can index them directly without a scatter.

## C. Increment order (each bit-identical or consistency-gated)

Status as of 2026-07-23. Note that 2d/2e diverged from the original plan: tracing the code
surfaced two pre-existing transient bugs (both since fixed), and 2e became a physics fix rather
than the static-intake conversion. The intake conversion is now folded into 2f.

1. **2c — recharge on rank 0 (DONE, commit 88665c7).** Recharge loop (both evap modes) guarded to
   rank 0; broadcast `rech` and `wtd`. `runoff` needs no broadcast (overwritten before reuse). The
   `rech`/`wtd` broadcasts become `scatterFromZero` at the flip.
2. **2d — UpdateTransientArrays on rank 0 (DONE, commit 6094ced).** Guarded to rank 0 with the
   dephier build; broadcast `topo`. `PrintValues`' loop was already rank-0-only (Phase 0);
   `wtd_old`/`wtd_mid` copies were left replicated (trivial memcpy; folded into 2f).
   - *Bug found + fixed (commit 4b8e13b):* the interpolation weight `f = cycles_done/total_cycles`
     was integer division → all transient forcing frozen at start values. Cast to double.
3. **2e — re-scatter topo/fdepth each cycle (DONE, commit 7fb7b72).** *Not* the originally-planned
   static-intake conversion. Tracing 2e revealed the solve reads topo/fdepth from vectors scattered
   only once at init, so in transient the solve ignored the (now-interpolating) topography change.
   Fix: broadcast `fdepth`, re-scatter topo/fdepth to the solve each transient cycle. Physics change
   (approved); transient golden reference regenerated. Equilibrium/test unaffected.
4. **2f — the flip (IN PROGRESS; the big step).** Scope decision (Andy, 2026-07-23): **full flip** —
   distribute `wtd`/`rech` too, not just the static arrays. Rationale: on msilong's 128 GB cap,
   keeping `wtd`+`rech` replicated (~2.26 GB/rank × 32 = ~72 GB) leaves only ~15 GB headroom (~113 GB
   used); distributing them drops the job to ~43 GB. Distributing them adds **no compute overhead**
   (the bridges are already owned-range) and is communication-neutral; it also naturally folds in the
   lever-#2 gather-hoist. Broken into sub-steps:
   - **2f-A — hoist the per-solve gather (DONE, commit fd2188f).** `gatherToAll` ran once per solve
     (`maxiter`×/cycle); the intermediate solves only need owned wtd. Extracted to
     `gather_wtd_to_all`, called once per cycle after the maxiter loop. Bit-identical; ~499/500
     redundant full-grid gathers per cycle removed. This is "lever #2."
   - **2f-B — distribute the solve dataflow (DONE, commits 06f6b7c wtd, f9a9647 rech).** wtd is now
     carried in `dmdapack.starting_wtd` and rech in a new `dmdapack.rech_dist` (backed by
     `AppCtx::rech_source`): both populated from `arp` once per cycle (owned copy) before the maxiter
     loop, read/written by the bridges over the owned range, wtd advanced in place by the copy-back,
     and wtd assembled back to `arp.wtd` once per cycle in `gather_wtd_to_all`. The per-solve loop no
     longer touches `arp.wtd`/`arp.rech`. Bit-identical (arp still replicated, so the owned copies
     read the same values). The vec-lifecycle worry was sidestepped: while arp is replicated the
     cycle-boundary transfer is a plain owned-range copy through the held local arrays, not a
     VecScatter — so no un-held vec is needed until 2f-C.
   - **2f-C — drop `arp` on non-root + acceptance check (the memory win). Bit-identical prep DONE;
     the coordinated drop REMAINS.**
     - **(a) DONE (commit 6...C(a)):** loop static reads now come from DMDA (`porosity_vec`, `mask`,
       `topo_vec`); `cell_area` stays Class-C. The per-solve loop reads no full-grid arp array.
     - **(c) DONE (commit 9d6e18d):** `wtd_old`/`wtd_mid` copies gated to rank 0.
     - **(b)+(d)+(e) REMAINING — one coordinated, non-bit-identical-in-structure step (do together;
       arp cannot be half-dropped):** replace the cycle-boundary owned copies with `scatterFromZero`
       (`arp.wtd`/`arp.rech` → distributed) and `gatherToAll`→`gatherToZero`; drop the now-unnecessary
       FSM/recharge wtd broadcasts (wtd stays rank-0 through the serial sections) and scatter the
       static fields (topo/fdepth/ksat/porosity/mask/cellsize) to the DMDA vecs from rank-0 `arp`
       instead of broadcasting-then-owned-copy; make loading (`irf.cpp`) a rank-0 GDAL read; allocate
       the full-grid `arp` arrays only on rank 0. Needs a raw-pointer `scatterFromZero` overload and
       float/double handling for the float static fields. **This is where bugs surface as segfaults /
       silent wrong-data (arp finally empty on non-root), not compile errors** — do it as one careful
       unit. Add the **structural acceptance check**: assert the full-grid `arp` arrays are
       empty/unallocated on non-root (structural, not RSS). Gate: full suite at several rank counts +
       the acceptance check.

## D. Ordering invariant (the safety rule)

The flip (2f) must be **last**: `arp` may become rank-0-only only after *every* full-grid access on
non-root ranks has been removed. A single missed non-root `arp(i,j)` read after the flip is an
out-of-bounds / stale-data bug, not a compile error — which is exactly why the `*_local` accessor
rename (compile-time guard) and the test suite matter. Before the flip, grep every `arp.<name>(`
outside rank-0 guards and confirm each is either Class C or converted.

## E. Known pre-existing issues surfaced (not blocking; flagged for the record)

- **Integer-division interpolation** — fixed (4b8e13b).
- **Solve ignored transient topography** — fixed (7fb7b72).
- **Solve still uses `topo_start` reference internally vs current topo in copy-back**: with the
  2e re-scatter the solve now uses current topo, so this is resolved for topo; watch for any other
  field scattered once at init that a transient run mutates.

## G. 2f-C drop progress (2026-07-23)

The solve dataflow is fully converted to source from rank-0; only the arp *allocation* drop remains.

- **drop-1a DONE (3099b75):** topo/ksat/fdepth scattered from rank-0 (`scatterFromZero`, templated on
  float/double); transient topo/fdepth broadcasts removed.
- **drop-1b DONE (1b3eb7b):** mask/porosity scattered from rank-0; cellsize from the 1-D Class-C
  array; `populate_DMDA_array_pack` moved before the `DMDA_Array_Pack` ctor and writes the global
  vecs directly.
- **drop-2 DONE (e5aab70):** wtd/rech scattered from rank-0 into the distributed carriers (via the
  un-held `wtd_global` scratch), `gather_wtd_to_all` uses `gatherToZero` (rank-0-only), and the FSM
  and recharge wtd/rech broadcasts are removed. wtd/rech now live on rank 0 through the serial
  sections. **All bit-identical; full suite green at n=1-8.**
- **Verified:** every non-root full-grid arp access is now either rank-0-guarded (recharge loop,
  PrintValues early-return, FSM, dephier, save, the gather write) or Class-C 1-D (`cell_area`). The
  non-root solve is arp-free.
- **drop-3 DONE (674615c) -- THE MEMORY WIN LANDED.** Loading (`InitialiseTransient/Equilibrium/Test`
  + `InitialiseBoth`) gated to rank 0; `ncells_x/y` broadcast to all ranks; `cell_size_area` (1-D)
  and DMDA setup on all ranks; `arp.check()` rank-0 only. Non-root ranks never allocate the full-grid
  ArrayPack. A structural acceptance check in `main()` asserts `arp.topo.size()` is 0 on non-root and
  `ncells_x*ncells_y` on rank 0 -- shown to bite (un-gating loading throws). Full suite green at
  n=1..8.

**2f (the flip) is COMPLETE.** Per-rank non-root footprint at 141M cells drops from ~26 GB
(replicated ArrayPack) to the DMDA subdomain vectors. Next: measure the actual scaling/memory on MSI
(the pre-change baseline was never taken; now compare before/after). Optional follow-ups: the §F
style cleanup; distributing `wtd_old`/`wtd_mid` is unnecessary (already rank-0-only).

## F. Style note (optional, low priority)

`src/dmda_gather.hpp` (`DMDAFullGridGather`) is a net-new module and is written in a more
encapsulated modern-C++ idiom than the rest of the codebase: a `class` with `private:` members,
deleted copy/assignment, trailing-underscore members (`Mx_`, `da_`), and camelCase methods
(`gatherToAll`/`scatterFromZero`). The surrounding house style uses plain `struct`s with public
members and snake_case methods (`DMDA_Array_Pack::make_global_vectors`, `release`). Edits to
existing (Kerry's) functions were kept in her style; only this new file differs, so it does not make
her code read foreign. Optional future cleanup: restyle to a public-member `struct` with
`gather_to_all`/`scatter_from_zero` etc. to match house conventions. Purely mechanical, no behavior
change; the test suite would confirm bit-identical.

---

# Next phase: distributed per-cycle dataflow (raise the strong-scaling knee)

The flip distributed the *storage* (arp on rank 0, solve distributed). This phase
distributes the *per-cycle dataflow* so the water table stays distributed across
cycles instead of round-tripping through rank 0. Removes the fixed serial fraction
that caps strong scaling at high core counts / global scale.

## Current per-cycle serial work (WTM.cpp update())
1. Scatter arp.wtd + arp.rech -> starting_wtd + rech_dist (start of cycle, ~169-184).
2. gather_wtd_to_all: starting_wtd -> arp.wtd (~192).
3. Recharge: rank-0 O(N) loop (~227-283) computing arp.rech, mutating arp.wtd (evap-0).
The wtd round-trips through rank 0 only so FSM / PrintValues / output can read the
full grid.

## Downstream safety map (verified 2026-07-24)
- arp.rech: ONLY consumer is the next-cycle scatter -> once recharge writes rech_dist
  directly, arp.rech and its scatter both go away.
- wtd_mid / wtd_old / PrintValues: rank-0 DIAGNOSTICS (text file + mass_balance test).
  golden checks the .tif (written from arp.wtd at output). Invariant: arp.wtd correct
  at output (one gather after recharge) + mass-balance reductions correct (already
  owned-range from the flip; accumulated in set_starting_values from rech_dist, so
  distributing the recharge preserves them).

## Increments (each bit-identical, test-gated: golden n=1..8 + mpi/fsm consistency + mass_balance)
- 1a (DONE, commit 4bab462): add + scatter distributed forcing vecs
  precip/evap/open_water_evap/runoff_ratio; unused, bit-identical.
- 1b: distribute the recharge:
  1. Add the 4 forcing arrays to DMDA_Array_Pack.
  2. Gate the start-of-cycle scatter to cycle 0 only (init); after that starting_wtd
     + rech_dist persist.
  3. After FSM: if fsm_on, sync arp.wtd -> starting_wtd (via the wtd_global scratch-copy
     pattern -- the pack HOLDS the DMDA arrays across cycles, so cannot scatter into a
     held vec directly).
  4. Distributed recharge over owned cells: starting_wtd + forcing -> rech_dist, mutate
     starting_wtd (evap-0 surface-water removal). Delete the rank-0 loop.
  5. Gather starting_wtd -> arp.wtd once, for PrintValues + output.
  6. Transient: re-scatter forcing after UpdateTransientArrays (forcing changes each
     cycle; one-directional, cheaper than the round-trip).
- 1f (later): distribute PrintValues (reductions) to drop the post-recharge gather.

## Wrinkles
- Pack holds DMDA arrays across cycles -> scatter-into-held-vec needs the scratch-copy
  dance (steps 3 & 6). Same lock issue the flip navigated.
- Do NOT rush: this is a 6-touch coupled MPI change with 3 lock-navigation points, in
  the same area as the earlier 8x mass-balance bug. Implement as a careful, individually
  -tested pass.

## Payoff (measured, honest)
Per-cycle serial is only ~4% at 8000^2/n=16 (update 172 s vs GW 156 s, minus one-time
dephier), where COMMUNICATION co-binds. So this lifts n=32 from ~11x toward the ~14x
Amdahl-with-comm ceiling -- modest at today's scale. The real payoff is at high core
counts (n>=64) and global / many-node runs, where the O(N) rank-0 recharge + casts
dominate as the solve shrinks. Right lever for the *massive*-scaling goal, not a quick
n<=32 win. (DH+FSM parallelization is a separate lever: ~0 for equilibrium, the Amdahl
wall for transient at high cores.)
