# Design Note (stub): Parallelizing FillSpillMerge

**Status: OPEN — problem statement + constraints only.** To be designed in a *dedicated*
session; this note is a handoff, not a design. Decision recorded 2026-08-11 (Andy): parallelize
FSM rather than optimize the rank-0 gather. This stub deliberately does **not** prescribe an
approach — that is the dedicated session's job.

## Why (the motivation)

WTM's multi-node scaling was validated 2026-08-11 on MSI Agate (`agate_no_fsm/README.md`): the
groundwater solve runs cross-node, **bit-identical**, scaling to 8 nodes. But that was `fsm_on 0`.
The coupled (fsm-on) model is capped by two **serial, un-parallelized** costs on rank 0, incurred
**every cycle**:

1. **FillSpillMerge** — a global spill-and-merge over the depression hierarchy, run serially on
   rank 0 (`WTM.cpp`: `if (mpi_rank == 0) FillSpillMerge(...)`), fixed wall time regardless of nodes.
2. **The full-grid gather to rank 0** (`gatherToZero`) that feeds FSM, plus the scatter back — an
   **all-to-one Infiniband transfer per cycle** cross-node (single-node it is a cheap memory copy).

Both are fixed regardless of node count, so **Amdahl caps the coupled model** even though the solve
now scales — and this is the *same* bottleneck that would cap a GPU port. Optimizing the gather only
band-aids a fundamentally serial algorithm (FSM still runs on one core while all others idle).
**Removing the serial section — a parallel FSM — is the lever:** a parallel FSM riding a parallel DH
could stay distributed and drop the rank-0 gather entirely.

## What is already in place

- **The GW solve is distributed and cross-node-validated** (PETSc DMDA + collectives; bit-identical,
  8-node scaling — `agate_no_fsm/README.md`).
- **DH (depression hierarchy) has parallel work** — the parallel priority-flood / local parallel DH;
  integrating it into WTM (swapping the vendored *serial* dephier) is the DH-integration thread. FSM
  is the one remaining serial piece on top of the hierarchy.
- **Distributed recharge** including the runoff round-trip is done (increments 2b/2c, see memory
  `project_fsm_direction`): runoff is computed distributed and gathered to rank-0 `arp.runoff` before
  the next FSM.

## Constraints the design must honor

- **Bit-identical (or provably equivalent) to the serial FSM.** The entire multi-node validation
  rests on the cross-rank/-node field comparison being ~0; a parallel FSM must preserve it.
- **The recharge → `arp.runoff` → next-FSM coupling** is real in production (`runoff_ratio_on` often
  on). A distributed FSM must produce/consume runoff consistently (see `project_fsm_direction` for
  the three separate toggles: `fsm_on`, `runoff_ratio_on`, `infiltration_on`).
- **FSM threshold sensitivity is inherent:** lake extent can flip on sub-`1e-6` per-rank
  solver-rounding differences at spill thresholds (documented; not a distribution bug). A parallel
  FSM inherits this, so correctness fixtures must use smooth-gradient topos to stay reproducible.
- **`infiltration_on`** (FSM-internal gradual infiltration as water crosses cells) is still serial /
  rank-0 and must be accounted for.

## References

- `benchmark/scaling/agate_no_fsm/README.md` — multi-node validation + the FSM cross-node analysis.
- `benchmark/DISTRIBUTED_ARP_DESIGN.md` — the single-node distribution that made the solve cross-node-ready.
- `benchmark/FSM_SERIAL_DESIGN.md` — the current serial FSM design.
- Barnes et al. parallel priority-flood / depression-hierarchy literature — the DH parallel basis.
- Memory: `project_fsm_direction`, `finding-agate-scaling`, `project_dh_integration`.
