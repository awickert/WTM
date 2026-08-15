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
- `eq_metric_compare.sh` — equilibrium-stop metric test: for `-wtm_eq_metric max|rms|frac`, measures cost
  (stop cycle, GW solves, wall) vs precision (final field vs a long no-stop reference). Re-run after #108.
- `diagnose_oscillation.py` — analyzes per-cycle wtd tifs to classify the equilibrium-metric behaviour
  (metric artifact vs true head oscillation; deep/shore; kink-crossing; adaptive vs constant).

## Results layout (`results/`)

- `adapt_bench_<regime>.csv` — one row per run: `regime,run,param,rc,wall_s,snes_its,nsteps,final_tif`.
  **The CSVs are the committed record of each run** (small; kept in git).
- `<regime>_<run>_<param>_*.tif`, `*.cfg`, `*.log`, `pert_*/`, `cold_domain/` — per-run fields and working
  dirs (large / regenerable; git-ignored, see `results/.gitignore`).

## Findings (Esquibel, n = 16, MSI; see `results/adapt_bench_*.csv`)

- **Transient:** constant dt is hard to beat on uniform dynamics — adaptive **ties** it. The **RMS norm
  beats the MAX norm** (43 vs 53 iterations for the same accuracy) and pulls adaptive onto the constant
  cost-accuracy curve (below it on max error); MAX-norm adaptive sat above it. No decisive win here.
- **Spin-up: adaptive decisively WINS** — the regime it is for. At matched iterations (~1370), adaptive
  holds max error **~12 m** while constant dt1 / dt2 blow to **399 m / 1638 m** (the violent cold-start
  front the fixed dt can't resolve); adaptive reaches the fine-reference accuracy in fewer iterations
  (1391) than the fine constant needs (2114), and *auto-finds* the safe step without a-priori dt tuning.
- **Worst-cell safety (RMS vs MAX):** it is the *constant-dt* runs whose worst cells blow up; adaptive
  *prevents* it, and **RMS bounds the worst cell (12.26 m) as tightly as MAX (12.33 m)** — the reject/retry
  feasibility floor caps the worst cells regardless of the norm, so the less-conservative RMS norm buys
  efficiency at no worst-cell (stability) cost. TR-BDF2's L-stability damps stiff-mode error rather than
  amplifying it, so a relaxed norm affects local *accuracy*, never stability.

**Bottom line:** adaptive dt is a **robustness / spin-up tool** — it ties well-chosen constant dt on smooth
transients and decisively wins on spin-up (bounded worst-cell error where fixed dt blows up). RMS is the
more efficient norm at no worst-cell cost.

## Equilibrium-stop metric (`-wtm_eq_metric`, default `frac`)

The `-wtm_eq_tol` auto-stop needs a per-cycle "how settled?" number. `diagnose_oscillation.py` showed the
old MAX metric is **worst-cell-hostage**: staggered deep lowland cells fill-and-pin at the surface at
different cycles, so max-over-cells never decays even though the bulk converges monotonically (no physical
oscillation). Measured trade (`eq_metric_compare.sh`; Esquibel warm, eq_tol = 0.05 m, vs a cap-40 ref):

| metric | stops at | GW solves | max \|Δref\| |
|---|---|--:|--:|
| max  | never (cap 30) | 303 | 1.4 m |
| rms  | cycle 8  | 93  | 14.6 m (loose) |
| **frac (99.9%)** | **cycle 22** | **231** | **4.3 m** |

So the default is **`frac`** (converged when < `-wtm_eq_frac` = 0.1 % of land cells exceed `eq_tol`): the
only metric that both *fires* and stays precise. `-wtm_eq_metric max` restores the strict worst-cell
criterion; `rms` is the loose/cheap bulk one. Applies on **every** spin-up pathway (fixed / Newton-
continuation / adaptive).

## Tbar as an opt-in stiff hammer (composes with adaptive; NOT auto)

`-wtm_Tbar` (time-averaged transmissivity) composes with the adaptive controller
(`-wtm_tr_bdf2 -wtm_dt_adaptive -wtm_Tbar`) — kept as an **opt-in hammer for stiff / stiff-long time steps**
(its proven benefit is extending the TR-BDF2 stability ceiling dt8→dt10). It is **not auto-engaged**: a
measured head-to-head (`tbar_vs_shrink`) found adaptive+Tbar equal (tol 1/20/100) or *worse* (tol 50: +2
rejects) than adaptive without it — the controller shrinks dt where it's stiff, so it stays out of the
ceiling regime where Tbar helps. So dt-shrink is the default response to stiffness; reach for `-wtm_Tbar`
only when deliberately pushing long steps near the stability ceiling.
