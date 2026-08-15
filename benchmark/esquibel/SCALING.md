# cc-vs-adaptive scaling — domain size × core count (Esquibel, FSM-on spin-up)

Companion to [`KC_VS_ADAPT.md`](KC_VS_ADAPT.md). Measures how the cc (fixed-1-week) vs one-knob-adaptive
spin-up cost scales with **domain size** and **core count**, on the fully-coupled (`fsm_on 1`) run.

## Metrics — read this first

- **SNES iterations** are the node-independent signal and the only clean number here. But they are **not a
  fair cross-method cost**: a TR-BDF2 iteration (two staged solves) is not the same unit of work as a
  backward-Euler one, so "adaptive uses fewer iterations" is a within-method diagnostic, *not* a speed or
  efficiency claim. See the correction in `KC_VS_ADAPT.md`.
- **Wall time** is the only method-fair cost. On the **shared `agsmall` node it is noise-limited** (a larger
  grid has come in *faster* than a smaller one across runs) — treat all wall numbers here as directional.
  The publication-grade wall comes from the **`--exclusive`** pass (below).
- **Precision is not matched at the `eq_tol` stop**: adaptive overshoots to a finer per-cycle settledness
  than cc, so the tables report both the raw stop and the **iso-precision** crossing (adaptive stopped where
  its per-cycle RMS(Δwtd) first reaches cc's *final* value).

## Pieces

- `scaling.sbatch` — fixed **N=16**, four grid sizes (1×1/2×2/3×3/4×4 tilings of Esquibel = 0.38/1.5/3.5/6.2M
  cells), cc + adaptive to equilibrium. The first scaling run.
- `scaling_ncore.sbatch` — the **parameterized single-node core sweep** (`CORES="8 4 2"` default), same four
  sizes, seeded with the N=16 rows so the master CSV holds cores = {2,4,8,16}. Idempotent (resumable),
  reuses the tiled domains in `results/scaling/tiles/`.
- `scaling_report.py` — organizes `results/scaling/scaling_ncore.csv` + logs into per-size cores-sweep tables
  (cc/adapt iters + wall, adaptive iso-precision iters, strong-scaling wall speedups).
- `iso_prec.py` — the iso-precision crossing (adaptive iters to reach cc's final precision) for the N=16 set.

Master record: `results/scaling/scaling_ncore.csv` (`cores,tiles,cells,method,rc,wall_s,snes_its,stop_cycle`;
git-ignored, regenerable). Per-run logs `results/scaling/nc<cores>_<tiles>_<method>.log`.

## Sequencing

1. **Fixed N=16 across sizes** — done. Iterations mesh-independent (cc ~9800–10600, adapt ~5500–6900 across
   16× cells); iso-precision iteration ratio flat at ~2×. Wall unusable (shared-node noise).
2. **Single-node core sweep {2,4,8}(+16)** — this is the **dry run for `--exclusive`**: same grid sizes, same
   information, validating the harness + data organization before spending an exclusive allocation.
3. **`--exclusive` pass** — reuse `scaling_ncore.sbatch` with `#SBATCH --exclusive` and `CORES="32 64 128"`
   (whole node), for the clean wall / parallel-efficiency curve. NOT started (gated to finalized code).
4. **`4000²` multi-node** (#82) — the largest-scale, multi-node point, where the FSM all-to-one *gather*
   (not FSM compute) becomes the real cost (#80).
