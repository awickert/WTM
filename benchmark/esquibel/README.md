# Esquibel comparison (full domain): awickert fork vs KCallaghan v2.0.1

The production-scale version of the island test (`../island/`): the **full Esquibel
domain, 853x451 = 384,703 cells**, 900 cells/degree. This is where the fork's real
advantages show at production scale:

- **accuracy** -- the mass-conserving solution drives orders of magnitude deeper than
  v2.0.1's mass-leak floor (see `../island/RESULTS.md`);
- **v2.0.1's parallel SEGV** -- it cannot run at >=4 ranks at all, while the fork does.

Profiling (`./profile.sh`, PETSc `-log_view` + WTM timers, 2026-08-10) shows the parallel
*speedup* is NOT the story -- but not for the reason first guessed. FillSpillMerge is a no-op
here (~2e-6 s/cycle: the taper holds wtd<=0, nothing to route). The entire cost is the
matrix-free groundwater solve (`SNESSolve`/`SNESFunctionEval`), and it is **memory-bandwidth-
bound**: ~600 Mflop/s per core, tiny MPI traffic, load balance ~1.0, and the scaling curve
(1.00 / 1.42 / 1.79 / 1.94x at n=1/2/4/8) flattens as the cores saturate one shared memory bus.
That ~2x is a laptop-hardware ceiling, **not** a code limit -- expect materially better scaling
on MSI (multiple sockets/nodes, independent memory channels); re-profile there before quoting a
scaling number. On the laptop, run at n=4 (n=8 barely helps) and expect the win to be accuracy +
parallel-capability, not scaling. `profile.sh` re-runs the whole analysis.

**Status: staged (domain populated), NOT YET RUN for the equilibrium comparison.** Probe
estimate: run to equilibrium ~40-80 cycles, ~3-6 min at n=8 (~6-11 min at n=1); the 100-cycle
config is enough.

## The domain (not committed)

`domain/` is 16 MB and gitignored. Regenerate it from the source dataset:

    ./make_esquibel.py                     # copies the 10 input rasters into ./domain
    ./make_esquibel.py /path/to/source     # if the source lives elsewhere

Source (Andy's data): `~/Downloads/Esquibel_Data-20260801T205621Z-1-001/Esquibel_Data`.
Geometry baked into the cfgs: `southern_edge 55.338391020555555`, `cells_per_degree 900`.

## The two models (same as the island)

Both solve the same domain and physics, at the **same inner tolerance** (`snes_stol 1e-6`):

- v2.0.1: `-snes_mf -snes_type anderson -snes_stol 1e-6`
- ours:   `-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6`  (Anderson, 1st order
  in time, capillary taper)

Binaries: fork at `../../build/wtm.x`; KCallaghan v2.0.1 at `../../../kcallaghan-wtm/build/wtm.x`.

## Running (when ready)

    ./make_esquibel.py
    ./run_esquibel.sh awickert   ../../build/wtm.x                 8
    ./run_esquibel.sh kcallaghan ../../../kcallaghan-wtm/build/wtm.x 1

Or the full two-metric comparison in one pass (wall to match v2.0.1's accuracy + wall to
completion, both models, n=1/2/4/8):

    ./compare.sh                           # RANKS="1 2 4 8" TIMEOUT=7200 overridable

`compare.sh` derives the match target from v2.0.1's *own* floor (not the island's 0.59 --
the mass-leak floor scales with the domain), then times ours to that same accuracy.

## Calibrate before trusting (why "not yet run")

The cfgs inherit the island's `deltat 604800` (1 week) and `total_cycles 100`. On a
44x-larger domain those are starting points, not answers:

- **Equilibration.** The drainage timescale grows with the domain, so 100 one-week cycles
  (~1.9 yr) may not reach equilibrium. Watch v2.0.1's Δ(col5) trajectory on the first run:
  if it is still descending at cycle 100, raise `total_cycles` (or `deltat`) until both
  models plateau/settle, then re-run. `compare.sh`'s completion metric is only meaningful
  once the run actually converges.
- **Cold-start stiffness.** Esquibel's cold start is stiffer than the island's; Anderson
  handles it (that is how v2.0.1 was run originally), but expect more work at cycle 0.
- **Runtime.** These are minutes-to-longer per run, not seconds -- hence the 2 h per-run
  timeout in `compare.sh`. Size expectations from the island's per-cell cost x 44 cells.

## Expected result (hypothesis to test)

From the island (`../island/RESULTS.md`): matched tolerance, ours reaches v2.0.1's accuracy
in fewer cycles and then drives ~4-5 orders deeper (mass-conserving) where v2.0.1 floors;
ours parallelizes and v2.0.1 SEGVs at >=4 ranks. Esquibel should widen the parallel margin.
Untested at this scale until run.
