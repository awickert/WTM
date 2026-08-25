# Precision-matched scheme comparison

**Date:** 2026-08-25 · **Branch:** `bdf2-adaptive-dt` · **Fixture:** island, 117×75 = 8,775 cells

`run.sh` sweeps WTM's time-integration / solver schemes and `report.py` reads cost off at **matched
precision**. Raw output of the run below: `RESULTS_island_2026-08-25.txt`.

Reproduce: `benchmark/scheme_bench/run.sh <wtm.x> 4 250`
(cold start from saturated, `dt` = 1 week, FSM on, `runoff_collector implicit`, `-snes_stol 1e-8`,
auto-stop disabled so every arm runs the same budget and reveals its own floor.)

## The rule this enforces

Never compare wall time or iteration counts at each scheme's *own* stopping criterion. A shared
`eq_tol` fires at a **different converged precision** per scheme, so such a table compares a
high-precision method against a low-precision one and calls the sloppy one fast. Cost is therefore
read off at matched water-depth rms, and the native-stop table is printed separately and labelled.

## Headline: iterations and wall time disagree

| scheme | iters → 1 mm | wall → 1 mm | iters/cycle |
|---|---|---|---|
| Anderson BE (secant) | 1835 | 0.9 s | 11.5 |
| Anderson BE (volume ΔV) | 1834 | 0.9 s | 11.5 |
| Picard BDF2-on-V + T̄ | **953** | 2.9 s | 5.8 |
| TR-BDF2 (fixed dt) | 1092 | **0.8 s** | 7.1 |
| TR-BDF2 + adaptive dt | never | never | 15.1 |
| Newton + dt-continuation | never | never | 1045 |

**Picard + T̄ needs the fewest iterations to 1 mm and takes 3.6× the wall time of TR-BDF2.** Its
iterations carry matrix assembly and a linear solve; Anderson's and TR-BDF2's are matrix-free. Fewest
iterations is not fastest — which is why both columns are always reported, never one alone.

At coarser precision TR-BDF2 fixed-dt wins on both axes (96 iters / 0.1 s to 100 mm).

## Precision floors

Anderson 0.1425 mm, Picard + T̄ 0.1377 mm, TR-BDF2 fixed 0.1334 mm — all **still improving** at 250
cycles, so these are budget limits, not floors. Longer runs are needed to find the true floors.

## Two negative results, stated plainly

**TR-BDF2 + adaptive dt REGRESSES on this fixture.** It reaches 1.138 mm at cycle 213 and ends at
**14.4 mm** — an order of magnitude worse than every fixed-dt scheme, and worse than its own best.
This *contradicts* the recorded Esquibel result, where adaptive settled finest (1.27 mm). Different
domain; the mechanism is undiagnosed and worth chasing rather than averaging away.

**Newton + dt-continuation is not competitive here:** 412 s and 261,203 iterations for 250 cycles
(1,045 iters/cycle vs 7 for TR-BDF2), reaching only 3.166 mm. To 10 mm it costs 28.5 s against
TR-BDF2's 0.3 s — roughly 100× the wall time.

**But that Newton number is a lower bound, not a verdict.** Every arm runs
`runoff_collector=implicit` so the physics is identical across schemes, and the implicit exfiltration
kink is **not** in the Newton Jacobian — the run prints a warning saying exactly that, and the
documented remedy is `runoff_collector=explicit`. So Newton is being measured outside its supported
configuration. Correcting it honestly requires a **second matched set at `explicit` for every
scheme**; switching Newton alone would manufacture a difference that is really a configuration
artifact. Until that is run, do not quote the Newton rows as Newton's cost.

Plain Picard and plain Newton both fail from cold at production `dt` (`rc=134`), as documented. Their
working recipes (`-wtm_Tbar`, `-wtm_stiff`) are separate arms so no scheme is judged on a setup it
was never claimed to handle.

## Caveats on the numbers

- Wall time carries ~10% machine noise (shared laptop, 16 cores, `n=4`, `OMP_NUM_THREADS=1`).
  Iterations are the clean algorithmic metric.
- Iso-precision wall is apportioned by iteration share (marked `~`); per-cycle wall is not
  instrumented.
- The dt-continuation arm varies `dt`, so its **cycle** count is not comparable with the fixed-dt
  arms. Its iterations and wall are; precision is matched by rms either way.
- One fixture, one domain, one `dt`. The adaptive regression above is a direct warning against
  generalising any of this to Esquibel or to production scale without re-measuring.
