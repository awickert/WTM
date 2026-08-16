# cc / fixed_tr / adaptive scaling — domain size × cores × nodes (Esquibel, FSM-on spin-up)

Companion to [`KC_VS_ADAPT.md`](KC_VS_ADAPT.md). Measures how the three integrators scale with **domain
size**, **core count** (single node), and **node count** (multi-node) on the fully-coupled (`fsm_on 1`)
cold-start spin-up. Integrators:

- **cc** — corrected-Callaghan: KCallaghan's fixed-1-week backward-Euler matrix-free Anderson, in our
  corrected code (1st order).
- **fixed_tr** — fixed-dt TR-BDF2 (`-wtm_tr_bdf2`), 2nd order.
- **adapt** — TR-BDF2 with the one-knob adaptive-dt controller (`-wtm_tr_bdf2 -wtm_dt_adaptive`).

Everything below is the **pre-dedicated (shared-node) set** — the dry run that validated the harness, the
data organization, correctness across ranks, and the hard-domain robustness. The only remaining piece is
the `--exclusive` pass for publication-grade wall (see *Sequencing*).

## Metrics — read this first

- **SNES iterations** are the node-independent signal and the cleanest number here. They are **not a fair
  cross-method cost**: a TR-BDF2 iteration is two staged solves, not one backward-Euler solve, so
  "adaptive/fixed_tr use fewer iterations" is a within-method diagnostic, *not* a speed claim. See the
  correction in `KC_VS_ADAPT.md`.
- **Wall time** is the only method-fair cost. On the **shared node it is noise-limited** (a larger grid has
  come in *faster* than a smaller one across runs; the 64-core 1×1 cc row below is ~10× its neighbors) —
  treat all shared-node wall numbers as directional. Publication-grade wall comes from the **`--exclusive`**
  pass.
- **Precision is not matched at the `eq_tol` stop**: adaptive overshoots to a finer per-cycle settledness
  than cc, so iso-precision (adaptive iters to reach cc's *final* per-cycle RMS) is reported separately —
  it lands at **~2× fewer iterations** than cc across every size (again, iterations, not wall).

## Headline findings

1. **Iterations are mesh-independent for all three methods.** Across 0.38M → 13.85M cells (36×), cc holds
   ~9,800–11,100; fixed_tr ~5,100–6,900 (≈ half of cc); adapt is the outlier — mostly ~5,000–8,700 but
   *layout-sensitive* (see #3).
2. **`fixed_tr`'s cycle count is decomposition-invariant.** It converges in **20–22 cycles everywhere** —
   every grid size, every core count, single- *and* multi-node, well- or over-decomposed. This is the
   standout robustness result: its 2nd-order per-cycle correction drops the `frac` stop-metric cleanly and
   monotonically, so the stop fires at the same cycle regardless of rank layout.
3. **The adaptive controller is non-deterministic under MPI, and it bites at scale.** On the hard 6×6
   seam-cliff domain the stop cycle wandered **18 (2×32) → 42 (4×64) → 20 (8×128)** — the 4×64 layout hit a
   controller resonance and cost *more* than cc. Same effect on 2×2 (cyc 68 at 2×32). The accept/reject/grow
   decisions branch on MPI-reduced values, so they are not reproducible across rank counts. This is the
   controller's cost, paid for the robustness-against-unknown-terrain it buys (see `KC_VS_ADAPT.md`).
4. **cc's `frac` stop-metric jitters under heavy over-decomposition.** 2×2 (1.5M) on 128 ranks (~12k
   cells/rank) ran to **cycle 843** despite a normal iteration count (9,944) — the solve converged, but the
   per-cycle metric never cleanly cleared threshold. This is a **benchmark-hygiene caveat, not a solver
   failure**: keep cells/rank healthy. All properly-sized points (4×4, 6×6 at 128 ranks) sit at cc cyc 13–14.
5. **Multi-node bandwidth scaling is excellent and the gather held.** On 6×6 (13.85M), cc wall
   **1312 → 594 → 305 s** across 32 → 64 → 128 ranks (near-ideal); fixed_tr 1509 → 721 → 395 s. The rank-0
   FSM gather was solid to 128 ranks across nodes, and multi-node ≡ single-node water table (RMS 3.7e-5 m).

## Single-node core sweep (agsmall, shared → wall directional)

Iterations (mesh-independent; the clean signal), representative rows; full table via `scaling_report.py`:

| grid (cells) | cc its | fixed_tr its | adapt its |
|---|--:|--:|--:|
| 1×1 (0.38M) | 9,827 | 5,133 | 4,712–9,707 |
| 2×2 (1.54M) | ~9,943 | 5,389 | 6,264–10,203 |
| 3×3 (3.46M) | ~10,020 | ~5,415 | 5,091–7,283 |
| 4×4 (6.16M) | ~10,570 | ~5,600 | 5,220–8,052 |

Strong-scaling wall (example, 4×4 / 6.16M, speedup vs 2 cores; shared-node noisy):

| cores | cc wall (s) | ×spd | fixed_tr wall (s) | ×spd | adapt wall (s) | ×spd |
|--:|--:|--:|--:|--:|--:|--:|
| 2 | 3021 | 1.0 | 3361 | 1.0 | 4510 | 1.0 |
| 8 | 1425 | 2.1 | 1482 | 2.3 | 2194 | 2.1 |
| 16 | 529 | 5.7 | 1015 | 3.3 | 720 | 6.3 |
| 32 | 526 | 5.8 | 767 | 4.4 | 772 | 5.8 |
| 64 | 319 | 9.5 | 341 | 9.8 | 508 | 8.9 |

## Multi-node (msilarge, shared+cross-node → wall directional)

6×6 (13.85M) — the properly-sized, real multi-node target and the bandwidth story:

| layout (nodes×ranks) | cc its / wall / cyc | fixed_tr its / wall / cyc | adapt its / wall / cyc |
|---|---|---|---|
| 2×32 | 11094 / 1312 / **14** | 6882 / 1509 / **22** | 6362 / 1454 / 18 |
| 4×64 | 11028 / 594 / **14** | 6882 / 721 / **22** | 14147 / 1643 / **42** ⚠ |
| 8×128 | 11021 / 305 / **14** | 6882 / 395 / **22** | 7011 / 423 / 20 |

⚠ adapt 4×64 = the controller resonance (finding #3). Note cc/fixed_tr cycle columns are dead-constant.

Over-decomposition caveat (finding #4), 2×2 (1.54M) on many ranks — watch the `cc_cyc` column:

| layout | cc cyc | fixed_tr cyc | adapt cyc |
|---|--:|--:|--:|
| 2×16 | 43 | **20** | 21 |
| 2×32 | 13 | **20** | 68 |
| 8×128 | **843** | **20** | 26 |

fixed_tr is the only method whose convergence is indifferent to how the (too-small-for-the-ranks) domain is
carved up.

## Pieces

- `scaling.sbatch` — the first run: fixed **N=16**, four grid sizes (1×1/2×2/3×3/4×4 = 0.38/1.5/3.5/6.2M).
- `scaling_ncore.sbatch` — parameterized **single-node core sweep** (`CORES`, `METHODS`), same four sizes,
  seeded with the N=16 rows so the master CSV holds cores {2,4,8,16,32,64}. Idempotent/resumable.
- `scaling_multinode_kc.sbatch` — **multi-node** harness (msilarge, `mpiexec -ppn`; `srun` is broken for
  PETSc's bundled MPICH). Sweeps `NODES_SWEEP` × `PPN` layouts and `TVALS` tile counts (t=6 → 6×6 = 13.85M).
- `scaling_report.py` — organizes both CSVs (+ per-run logs) into the single-node and multi-node tables
  above, including fixed_tr and adaptive iso-precision. Regenerates everything here.
- `iso_prec.py` — the iso-precision crossing (adaptive iters to reach cc's final precision) for the N=16 set.

Master records: `results/scaling/scaling_ncore.csv`, `results/scaling/scaling_multinode.csv`
(`...,method,rc,wall_s,snes_its,stop_cycle`; git-ignored, but the raw data lives only on MSI — the curated
tables above are the committed record). Per-run logs `results/scaling/*.log`.

## Sequencing

1. **Fixed N=16 across sizes** — done. Iterations mesh-independent; iso-precision ratio flat at ~2×.
2. **Single-node core sweep {2,4,8,16,32,64}** — **done** (all three methods, four sizes). Dry run for
   `--exclusive`: validated harness + data organization; iterations clean, wall shared-node noisy.
3. **Multi-node dry run** — **done**. Layouts 2×16 … 8×128 on 2×2/4×4/6×6. De-risked the rank-0 FSM gather
   at 128 ranks; confirmed multi-node ≡ single-node water table (RMS 3.7e-5 m); surfaced the
   over-decomposition metric caveat and the adaptive rank-resonance.
4. **6×6 (13.85M) seam-cliff de-risk** — **done**. Real clipped-land-meets-ocean cliffs (6×6 Esquibel
   tiling); cc and fixed_tr robust/reproducible, adaptive fragile (finding #3), memory/gather fine at scale.
5. **`--exclusive` pass** — NOT started (gated to finalized code). Reuse `scaling_ncore.sbatch` /
   `scaling_multinode_kc.sbatch` with `#SBATCH --exclusive` and whole-node core/node counts, for the clean
   wall / parallel-efficiency curve. Every correctness, robustness, and de-risk question above is already
   answered — this pass measures **only wall** (strong + weak scaling).
6. **`4000²` multi-node** (#82) — the largest-scale publication point, where the FSM all-to-one *gather*
   (not FSM compute) becomes the real cost (#80).
