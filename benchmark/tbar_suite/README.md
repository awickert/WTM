# `-wtm_Tbar` test suite — drivers and machine-readable results

Drivers and captured results for the `-wtm_Tbar` (time-averaged interblock transmissivity) evaluation.
See `benchmark/TBAR_TIME_AVERAGING.md` for the design, math, and the results write-up.

## Files

- `suite.py` — cold-start dt sweep (Anderson/Picard ± T̄) + a Newton ± T̄ pair (`suite.py cold <domain> <tag>`).
- `suite_warm.py` — warm-start perturbation dt-to-failure (`suite_warm.py <src_domain> <eq_raster.tif> <tag> [precip_scale]`).
- `esq_headline.py` — Anderson/Picard ± T̄ cold start on the real 384k-cell Esquibel patch.
- `analyze.py` — turns the JSON into tables + an equilibrium-accuracy comparison (`analyze.py <json...>`).

## Results (machine-readable)

JSON, one record per run, fields: `name, status (OK|FAIL|TIMEOUT), rc, wall (s), tot_iters, nsolves,
div, cyc_run, settle, final_swt, raster, solver, weeks`.

- `s1_cold.json` — cold-start dt sweep on the ocean-ringed Esquibel island (75×117).
- `w2x_warm.json` — warm-start, 2× recharge perturbation, dt-to-failure (island).
- `esq_headline.json` — real 384k Esquibel patch, cold start.

The `.out` files are the human-readable run logs. `raster` paths in the JSON point at the (volatile)
scratchpad tif outputs that existed at run time; the numeric metrics in the JSON are self-contained.

## Provenance / reproduction

Inputs: the island runs used an ocean-ringed crop of Kerry's Esquibel dataset (75×117, 2544 land cells,
~615 m relief); the headline used the full 384k-cell patch (`~/Downloads/Esquibel_Data.../`). Runs used
`OMP_NUM_THREADS=4` (island) / `6` (Esquibel), `PROJ_DATA=/usr/share/proj`, 1-week base `deltat`, default
tapers on. Anderson = `-wtm_anderson`; Picard = default (BDF2-on-V); Newton = `-wtm_stiff`; T̄ adds
`-wtm_Tbar`. The island/Esquibel input rasters are not committed (large, derived from Kerry's dataset);
point the drivers' `domain` argument at an equivalent surfdatadir to regenerate.

Run: `python3 suite.py cold <island_domain> s1 && python3 analyze.py s1_cold.json`.
