# Adaptive-dt benchmark

Does the TR-BDF2 **adaptive time step** (`-wtm_tr_bdf2 -wtm_dt_adaptive`) buy anything over a well-chosen
**constant** dt — in SNES iterations (the node-independent cost) and wall — for the two regimes that matter:

- **transient** — warm from equilibrium, a −20 % dry P−ET step, advanced 8 weeks. Temporally *uniform*
  dynamics, so a single dt is already near-optimal; this is the hard case for adaptive.
- **spin-up** — cold from wtd = 0 (water table at the surface), normal forcing, ~5 yr. Huge dynamic range
  (violent initial drainage → slow settling), so any *fixed* dt is wasteful; this is where adaptive should win.

Each regime compares three schemes at matched physical horizon:
- **const** — `-wtm_tr_bdf2` at a sweep of fixed dt (the finest is the accuracy reference).
- **adaptmax** — adaptive, MAX error norm (default), a sweep of `-wtm_dt_tol`.
- **adaptrms** — adaptive, RMS error norm (`-wtm_dt_norm_rms`), a smaller-tol sweep. The RMS norm is less
  worst-cell-sensitive than MAX (one deep stiff cell otherwise pins dt).

## Files

- `adaptive_bench_msi.sbatch` — the benchmark. Self-contained sbatch (survives login disconnect). Run per
  regime:
  ```
  REGIME=transient sbatch adaptive_bench_msi.sbatch
  REGIME=spinup    sbatch adaptive_bench_msi.sbatch
  ```
  Inputs come from `../esquibel` (`domain/`, `w_eq_correct.tif`, `perturb_pet.py`). Spin-up builds a
  `results/cold_domain/` (domain inputs + a wtd = 0 tif) automatically. All outputs land under `results/`.
- `analyze_adapt_bench.py` — pairs cost (iterations + wall) with accuracy (error vs the finest const dt):
  ```
  python3 analyze_adapt_bench.py transient
  python3 analyze_adapt_bench.py spinup
  ```
- `local_serial_run.sh` — single-run helper for the idle-laptop clean room (n = 1, contention-free), used
  to prototype the comparison; iterations are platform-independent so laptop iteration counts match MSI.

## Results layout (`results/`)

- `adapt_bench_<regime>.csv` — one row per run: `regime,run,param,rc,wall_s,snes_its,nsteps,final_tif`.
  **The CSVs are the committed record of each run** (small; kept in git).
- `<regime>_<run>_<param>_*.tif`, `*.cfg`, `*.log`, `pert_*/`, `cold_domain/` — per-run fields and working
  dirs (large / regenerable; git-ignored, see `results/.gitignore`).

## Findings so far (Esquibel, n = 16, MSI)

- **Transient:** constant dt is hard to beat on uniform dynamics. The **RMS norm beats the MAX norm** (~43
  vs 53 iterations for the same accuracy) and pulls adaptive onto the constant cost-accuracy curve (and
  below it on max error); MAX-norm adaptive sat above it.
- **Spin-up:** the decisive test for adaptive — results recorded here as they complete.
