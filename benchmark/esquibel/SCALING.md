# cc / fixed_tr / adaptive scaling — domain size × cores × nodes (Esquibel, FSM-on spin-up)

Companion to [`KC_VS_ADAPT.md`](KC_VS_ADAPT.md). Measures how the three integrators scale with **domain
size**, **core count** (single node), and **node count** (multi-node) on the fully-coupled (`fsm_on 1`)
cold-start spin-up. Integrators:

- **cc** — corrected-Callaghan: KCallaghan's fixed-1-week backward-Euler matrix-free Anderson, in our
  corrected code (1st order).
- **fixed_tr** — fixed-dt TR-BDF2 (`-wtm_tr_bdf2`), 2nd order.
- **adapt** — TR-BDF2 with the one-knob adaptive-dt controller (`-wtm_tr_bdf2 -wtm_dt_adaptive`).

## Strong vs weak scaling — definitions

Two distinct parallel-scaling questions, measured separately here:

- **Strong scaling** — **fix the problem size, add processors.** The domain is held constant (here 6×6 =
  13.85M cells) while ranks grow 16→128 (1→8 nodes at 16 ranks/node); cells/rank *shrinks* (866k→108k).
  Ideal = wall falls in inverse proportion to processors (2× ranks → ½ wall), i.e. **speedup = wall(1)/wall(n)
  approaches n**. Answers: *"I have this domain — how much faster does throwing more cores make it?"* It is
  bounded by the serial fraction (Amdahl) and by the subdomain becoming too small (comm/bandwidth dominate).
- **Weak scaling** — **fix the work per processor, grow the problem and the processors together.** Here
  cells/rank is held at exactly 384,703 (one Esquibel tile per rank) while both the domain (6.16M→49.3M) and
  ranks (16→128) grow in lockstep. Ideal = **wall stays constant (growth factor 1.0)** as the system grows.
  Answers: *"If I scale up to a continental/global domain on proportionally more nodes, does it stay
  tractable?"* — the question that predicts the 220M production run. It is bounded by cross-node
  communication and any serial/gather term that grows with total size rather than with per-rank work.

Rule of thumb: **strong = same problem, more cores, want it faster; weak = bigger problem, more cores, want
it no slower.**

The `--exclusive` (dedicated-node) results below are the clean-wall answer; the earlier
**pre-dedicated (shared-node) set** (further down) validated the harness, data organization, cross-rank
correctness, and hard-domain robustness, but its wall was noise-limited.

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

## Exclusive (dedicated-node) results — the clean-wall answer

Jobs 16006880 (strong) + 16006881 (weak), 8 exclusive nodes each, 16 ranks/node, cc + fixed_tr + adapt ×
**3 reps**, `-wtm_eq_tol 0.01 -wtm_eq_metric frac`. Medians below; regenerate with `scaling_report.py`
(reads `*_reps.csv`). NB — these stops use the current per-cycle metric; a **matched water-volume
convergence threshold** (ΔV = storativity·Δwtd, not head change, so deep low-storativity cells cannot blow
up the metric — the FV-consistent choice) is the honest cross-method basis and is tracked as follow-up.

**Strong scaling — fixed 6×6 (13.85M), speedup vs 1 node (ideal = n):**

| nodes | ranks | cc: wall / spd / eff | fixed_tr: wall / spd | adapt: wall / cyc |
|--:|--:|---|---|---|
| 1 | 16 | 961 s / 1.0× / 100% | 1113 s / 1.0× | 3287 s / **52** |
| 2 | 32 | 454 s / 2.12× / 106% | 638 s / 1.74× | 645 s / **18** |
| 4 | 64 | 246 s / 3.91× / 98% | 411 s / 2.71× | 963 s / **42** |
| 8 | 128 | 172 s / **5.60×** / 70% | 202 s / 5.51× | 241 s / **19** |

cc is the clean curve (98% efficient to 4 nodes, 70% at 8), fastest on wall at every point. fixed_tr tracks it.

**Weak scaling — fixed 384,703 cells/rank, wall growth vs 1 node (ideal = 1.0):**

| nodes | ranks | cells | cc: wall / growth | fixed_tr | adapt |
|--:|--:|--:|---|---|---|
| 1 | 16 | 6.16M | 325 s / 1.00× | 467 s / 1.00× | 493 s / 1.00× |
| 2 | 32 | 12.3M | 356 s / 1.10× | 445 s / 0.95× | 441 s / 0.90× |
| 4 | 64 | 24.6M | 376 s / 1.16× | 532 s / 1.14× | 499 s / 1.01× |
| 8 | 128 | 49.3M | 587 s / **1.81×** | 725 s / 1.55× | 783 s / 1.59× |

**Near-flat 1→4 nodes (≤1.16×), degrading to ~1.6–1.8× at 8 nodes / 49.3M** — versus the **4.7×** blow-up of
the single-node ladder. The "each node adds a bandwidth pool" thesis holds; this is the direct 220M evidence.

**adapt — the best-converging method (demonstrated, not asserted).** Running adapt at every point (rather
than excluding it, as an earlier draft proposed) is what surfaced its real value:

- **It converges the physics that fixed_tr cannot.** Per-cycle wtd rms settles finest of the three —
  **adapt 1.27 mm vs cc 3.5 mm vs fixed_tr 65 mm** at their stops (8-node 6×6). fixed_tr's rms *plateaus* at
  ~65 mm because a few deep exp-T cells oscillate ~15 m of head every cycle (≈ zero water, but the head
  never settles); **adapt's shrinking dt damps exactly those cells.** To 1 cm wtd rms: adapt and cc tie on
  wall (~129 vs 126 s) but adapt uses ~half the iterations (3,765 vs 8,112); **fixed_tr never reaches 1 cm**
  (see the convergence-threshold table below). So adapt is not merely a "robustness tool" — it has a
  measurable convergence-quality advantage at cc-comparable cost.
- **It is deterministic run-to-run** — iterations bit-identical across all 3 reps at every layout;
  stop-cycle wobbles ≤±1 near threshold. The earlier "non-deterministic" label was wrong.
- Its remaining real caveat is **layout-sensitivity of *cost*** (not of the answer): on 6×6 the stop-cycle
  went 52 → 18 → 42 → 19 across 1/2/4/8 nodes (the 4-node resonance reproduced in all 3 reps), because the
  MPI reduction order that feeds the controller changes with rank count. This is a secondary tuning issue,
  not a reason to discount adapt. (On the weak config's rectangular domains it stayed smooth, cyc 15–19.)

**Convergence to a matched head threshold (8-node; iterations node-independent, wall scales per the table):**

| threshold | cc | fixed_tr | adapt |
|---|---|---|---|
| 10 cm wtd rms | cyc 6 · 4,475 it · ~70 s | cyc 7 · 2,108 it · ~62 s | cyc 6 · 2,153 it · ~74 s |
| **1 cm wtd rms** | cyc 11 · 8,112 it · ~126 s | **never** (floor 65 mm) | cyc 11 · 3,765 it · ~129 s |
| rms floor at stop | 3.5 mm | **65 mm** | **1.27 mm** |

A consistent *head* threshold cannot even rank cc against fixed_tr (fixed_tr's deep cells oscillate in head),
which is why the FV-consistent **pure-water-depth** metric (`-wtm_eq_metric water`, |S·Δwtd|) was added — in
water depth the deep-cell swings vanish and all three become comparable. Water-depth trajectory: follow-up.

## Headline findings

1. **Iterations are mesh-independent for all three methods.** Across 0.38M → 13.85M cells (36×), cc holds
   ~9,800–11,100; fixed_tr ~5,100–6,900 (≈ half of cc); adapt is the outlier — mostly ~5,000–8,700 but
   *layout-sensitive* (see #3).
2. **`fixed_tr`'s cycle count is decomposition-invariant.** It converges in **20–22 cycles everywhere** —
   every grid size, every core count, single- *and* multi-node, well- or over-decomposed. This is the
   standout robustness result: its 2nd-order per-cycle correction drops the `frac` stop-metric cleanly and
   monotonically, so the stop fires at the same cycle regardless of rank layout.
3. **The adaptive controller's *cost* is layout-sensitive — but it is deterministic and the best-converging
   method** (CORRECTED by the exclusive reps; the earlier "non-deterministic" read here was from single-rep
   shared-node data and was wrong). On 6×6 the stop cycle varies with rank count — 18 (2×32) → 42 (4×64) →
   20 (8×128) — because the MPI reduction order feeding the controller changes with the layout; the 4×64
   resonance is real and reproduces *deterministically* across all 3 exclusive reps (iterations bit-identical
   per layout). Crucially, this cost-variability is a *secondary* issue: adapt settles the water table finest
   of the three (1.27 mm rms vs cc 3.5, fixed_tr 65) and damps the deep exp-T cells that fixed_tr leaves
   oscillating — see the exclusive-results adapt section. Do not read the cost-swing as a reason to discount
   it.
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

## Weak scaling — the geometry trap (single-node dry run)

Weak scaling holds **cells/rank fixed** (here 384,703 = one Esquibel tile per rank) and grows domain and
ranks together; ideal = flat wall. The first dry run (`scaling_weak.sbatch`) put **all ranks on one node**,
and the result is *not* flat — but that is the diagnosis confirming itself, not a scaling failure:

| tiles | ranks (1 node) | cc wall | cc weak-eff | fixed_tr wall |
|---|--:|--:|--:|--:|
| 1×1 | 1 | 288 s | 100% | 343 s |
| 2×2 | 4 | 293 s | 98% | 381 s |
| 3×3 | 9 | 328 s | 88% | 443 s |
| 4×4 | 16 | 465 s | 62% | 532 s |
| 5×5 | 25 | 704 s | 41% | 763 s |
| 6×6 | 36 | 1351 s | **21%** | 1505 s |

Wall rises ~4.7×. The cause is **single-node memory-bandwidth saturation**: piling more ranks on one node's
16 channels grows the problem *without* growing the bandwidth. Iterations stay ~flat (cc 9827→11035, ~12%
drift), so it is per-iteration bandwidth cost, not more work. The clean cross-check: the same 13.85M domain
ran **1351 s on 36 ranks / 1 node** here but **305 s on 128 ranks / 8 nodes** in the strong set — 4.4×
faster on the same cells. **The lever is bandwidth pools (nodes), not cores.**

So the single-node weak ladder measures saturation, *not* the production question. The weak-scaling result
that predicts 220M is **16 ranks/node (one per channel) spread across N nodes**, holding cells/rank fixed so
each added node brings its own bandwidth pool — `scaling_weak_multinode.sbatch`, run in the `--exclusive`
pass. If *that* wall is flat as nodes grow, it is the direct green light for 220M on ~16–32 nodes.

## Pieces

- `scaling.sbatch` — the first run: fixed **N=16**, four grid sizes (1×1/2×2/3×3/4×4 = 0.38/1.5/3.5/6.2M).
- `scaling_ncore.sbatch` — parameterized **single-node core sweep** (`CORES`, `METHODS`), same four sizes,
  seeded with the N=16 rows so the master CSV holds cores {2,4,8,16,32,64}. Idempotent/resumable.
- `scaling_multinode_kc.sbatch` — **multi-node** harness (msilarge, `mpiexec -ppn`; `srun` is broken for
  PETSc's bundled MPICH). Sweeps `NODES_SWEEP` × `PPN` layouts and `TVALS` tile counts (t=6 → 6×6 = 13.85M).
  `REPS` repeats each point → rep-columned `scaling_multinode_reps.csv` (dry-run `scaling_multinode.csv` kept).
- `scaling_weak.sbatch` — **single-node weak scaling**: fixed cells/rank (one Esquibel tile per rank,
  ranks = t²), all ranks on one node. Measures single-node bandwidth saturation (see the geometry-trap
  section above), *not* the production weak curve. cc + fixed_tr by default here (adapt's cost is
  layout-sensitive, so its single-node-ladder wall is less directly comparable — but it is deterministic).
- `scaling_weak_multinode.sbatch` — **multi-node weak scaling** (the production-predictive one): 16 ranks/node
  (one per memory channel), node sweep `LADDER="1:4:4 2:4:8 4:8:8 8:8:16"` (nodes:ny:nx tiles), holding
  384,703 cells/rank exactly so each added node brings its own bandwidth pool. Ideal = flat wall vs nodes.
  Extend toward 220M with `16:16:16` (256 ranks, 98.5M). For the `--exclusive` pass add `#SBATCH --exclusive`.
  `REPS` repeats each point → rep-columned `scaling_weak_multinode_reps.csv`.
  `scaling_weak_multinode.csv` currently holds only the **1+2-node shared-node validation** (job 15964419):
  it confirmed cross-node placement, the 4×8 rectangular tiling, and decomposition-invariant convergence
  (iterations bit-stable 10558→10556 / 5534→5534), but its **wall is shared-node noise** (cc falls, fixed_tr
  rises over the same step) — NOT a weak-scaling measurement. The clean curve is the `--exclusive` run.
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
   tiling); all three reproducible (adapt's cost is layout-sensitive but deterministic — finding #3),
   memory/gather fine at scale.
5. **Weak-scaling dry run** — `scaling_weak.sbatch` on shared agsmall, one tile per rank (t=1..6) — **done**.
   Result: single-node wall rises 4.7× (bandwidth saturation, not weak-scaling failure — see the geometry-trap
   section). Its lesson: the single-node ladder is the wrong geometry; the production weak curve needs
   `scaling_weak_multinode.sbatch` (16 ranks/node across nodes), now added to the `--exclusive` pass below.
6. **`--exclusive` pass** — NOT started (gated to finalized code + explicit go-ahead). Measures **only wall**
   (strong + weak); every correctness/robustness/de-risk question above is already answered. **Pinned grid
   (keep cells/rank healthy — the finding-#4 over-decomposition floor):**
   - *Strong scaling* — reuse `scaling_ncore.sbatch` (single-node, `CORES="16 32 64 128"`) and
     `scaling_multinode_kc.sbatch` on the **large domains only** (4×4 = 6.16M and 6×6 = 13.85M). 6×6 stays
     ≥108k cells/rank even at 128 ranks; 4×4 hits ~48k at 128 (borderline — read its 128-rank point with the
     over-decomposition caveat). **Do NOT** run 1×1/2×2 at ≥64 ranks (that is where the `frac` stop-metric
     jittered to cyc 843).
   - *Weak scaling* — `scaling_weak_multinode.sbatch`, 16 ranks/node across a node sweep (`1:4:4 2:4:8
     8:8:16` → 16…128 ranks, 6.16M…49.3M), holding 384,703 cells/rank so each node adds a bandwidth pool.
     This is the curve that predicts 220M; extend with `16:16:16` (98.5M) toward the production point.
   - *Methods & reps* — **all three (cc, fixed_tr, adapt) at every point, `REPS="1 2 3"`.** The reps
     let the data speak on adapt directly (rather than asserting): they showed adapt is **deterministic**
     run-to-run and the **best-converging** method (finding #3 / the exclusive-results adapt section), and
     confirm cc/fixed_tr reproducibility. Rep-columned output goes to the `*_reps.csv` files
     (kept separate from the shared-node dry-run/validation CSVs).
   - Add `#SBATCH --exclusive` to both harnesses (via the CLI `--exclusive` flag).
7. **`4000²` multi-node** (#82) — the largest-scale publication point, where the FSM all-to-one *gather*
   (not FSM compute) becomes the real cost (#80).
