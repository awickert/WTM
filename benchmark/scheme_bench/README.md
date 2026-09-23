# Precision-matched scheme comparison

**Date:** 2026-09-16 (re-run for #6; the 2026-08-25 files are kept beside these) · **Branch:** `bdf2-adaptive-dt` · **Fixture:** island, 117×75 = 8,775 cells

`run.sh` sweeps WTM's time-integration / solver schemes and `report.py` reads cost off at **matched
precision**. Raw output of the run below: `RESULTS_island_2026-09-16.txt`.

Reproduce: `benchmark/scheme_bench/run.sh <wtm.x> 4 250`
(cold start from saturated, `dt` = 1 week, FSM on, `runoff_collector implicit`, `-snes_stol 1e-8`,
auto-stop disabled so every arm runs the same budget and reveals its own floor.)


## 2026-09-16 re-run (#6): what changed, and one thing that is NOT explained

Newton's rows in the 2026-08-25 matrix predate `bf187ee`, which gave Newton the semismooth active-set
tangent, so they measured a Newton that could not differentiate the constraint it was solving against.
Re-run of the full 2×2. Iterations to reach rms ≤ 10 mm-water:

| Newton + dt-continuation | implicit × between | implicit × during | active-set × between | active-set × during |
|---|---|---|---|---|
| 2026-08-25 | 18041 | 10 | 1780 | 7 |
| 2026-09-16 | **100000** | **100000** | **44** | 8 |

**Under `active_set` the expected result, larger than expected:** 1780 → 44, a 40× improvement. That is
the tangent, and it is the headline this re-run was for.

**Under `implicit` it went the other way, and the reason turns out to matter more than the number.**
Newton + continuation under `implicit` NEVER CONVERGED on this problem. It was exiting on the
head-based RELATIVE-STEP test after 10-14 iterations -- the stagnation test `#61` identified as letting
88-95% of solves exit early -- and `#104` now gates that verdict behind a residual check, so the
premature exit is refused and the solve runs to `max_iterations`.

Established by one controlled run, HEAD binary, same config, one key changed:

| `residual_gate` | first solve | exit reason | total iterations |
|---|---|---|---|
| `1e-05` (shipped) | 10000 | `DIVERGED_MAX_IT` | 122878 |
| `1e+30` (disabled) | **14** | **`CONVERGED_SNORM_RELATIVE`** | 21454 |
| baseline `1b3069b` | 10 | `CONVERGED_SNORM_RELATIVE` | 110179 |

Disabling the gate restores both the count and the exit reason, so the gate is the whole difference.

**So the 2026-08-25 table is the misleading one, not this one.** Its `implicit` Newton rows present a
scheme that was not solving as one that solved cheaply, and anyone choosing a scheme from that table
would have been misled. The rows here are a measurement of non-convergence, which is the honest result.

Two eliminations saved a 520-commit bisect: `bf187ee`'s tangent is gated behind
`g_active_set && !g_kirchhoff`, so it is unreachable under `implicit`; and a fresh build at `1b3069b`
reproduces `32 / 18041 / 29.6` exactly, so the difference was real and in-range. The clue was that
HEAD's FINAL per-cycle rms is **2.27 mm-water** against baseline's **4.983** -- HEAD converges further
while the matrix called it "never", which pointed at the first cycle rather than the trajectory.

**Two harness bugs were fixed to make this run at all**, both of which had made the benchmark
silently unusable rather than loudly broken:

1. `OUT` was relative, and the model joins `outfile_prefix` onto `directory` -- so every arm died with
   "Attempt to create new tiff file ... No such file or directory". All 8 arms in all 4 corners failed
   that way, and `rc=1` with an empty log is indistinguishable from a model failure until you read the
   log. A caller passing an absolute `OUT` never sees it, which is why it survived.
2. `OUT` was derived from the REMAPPED coupling name, so `run.sh` wrote `results_*_impulse` /
   `_continuous` while `matrix.py` reads `results_*_between` / `_during`. A full 2×2 could therefore
   complete and the matrix would still report the PREVIOUS run's numbers -- the worst kind of failure,
   since it produces a plausible table.

Under `active_set` the coupling choice is very nearly inert: `between` and `during` differ by
**1.6466e-09 m** in the final field, which is why their iteration counts are identical rather than
merely close. That bears on #90.


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

---

# The key comparison: FSM between steps vs during the step

`run.sh` takes `COUPLING=between|during`. Both sweeps were run identically
(`results_between/`, `results_during/`); side-by-side output in
`COMPARISON_island_2026-09-16.txt`, produced by `compare.py`.

- **between** — `surface_water.routing: impulse`. FillSpillMerge runs between steps and **overwrites**
  the water table (the original behaviour). `between` is this script's env value; `impulse` is the
  config value it sets.
- **during** — `surface_water.routing: continuous`, and **the shipped default** (#43). FSM's per-cell
  ΔV enters the **next step's source term**. Was `-wtm_fsm_continuous`; that flag is retired (#86).

## Cost at matched settling precision (rms ≤ 1 mm-water), fixed-dt schemes

| scheme | between (iters / ~s) | during (iters / ~s) | iters | wall |
|---|---|---|---|---|
| Anderson BE (secant) | 1835 / 0.9 | 1259 / 1.0 | **1.46× fewer** | 0.92× (slower) |
| Anderson BE (volume ΔV) | 1834 / 0.9 | 1251 / 1.0 | **1.47× fewer** | 0.91× (slower) |
| Picard BDF2-on-V + T̄ | 953 / 2.9 | 918 / 2.5 | 1.04× | 1.16× (faster) |
| TR-BDF2 (fixed dt) | 1092 / 0.8 | 700 / 1.0 | **1.56× fewer** | 0.80× (slower) |

**In-step coupling buys ~1.5× fewer nonlinear iterations and is ~10–20% SLOWER in wall time** on the
matrix-free schemes. The two effects are separate and both real: the smoother state is easier to
solve, but each step now pays an extra O(N) ΔV pass plus a scatter on rank 0. Measured cost per
iteration, Anderson BE: 0.51 ms (`between`) vs 0.81 ms (`during`).

That overhead is per *step*, not per iteration, so it should amortise better where iterations-per-cycle
are higher or the domain is larger. **This is 8,775 cells; do not extrapolate to Esquibel or to
production without re-measuring.**

Settling floors are effectively identical (Anderson 0.1425 vs 0.1427; TR-BDF2 0.1334 vs 0.1338), so
`during` settles just as completely — it is not trading precision for the iteration saving.

## A correction to an earlier number

An earlier note recorded "≈2.4× fewer iterations" for the source coupling on Esquibel (2661 → 1114).
That was a **full-run** comparison at each arm's own stopping point, not precision-matched. The same
inflation appears here: the full-budget ratio is 1.75× where the matched ratio is 1.46×. Treat the
2.4× as an overstatement of the same kind until it is re-measured at matched precision.

## Two arms whose iso-precision rows must not be read as convergence

The precision axis is the per-cycle *change* in the water table, so a scheme deliberately taking tiny
steps reports a tiny rms while nowhere near converged. Both variable-dt arms are affected and are
marked in the output:

- **Newton + dt-continuation** reads "10 mm after 10 iterations" because continuation starts near
  `dt ≈ 0.001 yr`. Its eye-catching full-budget figures (261,203 → 7,426 iterations, 412.5 → 13.3 s,
  ~35× and ~31×) are further confounded: `during` also ends at a **coarser** settling state
  (4.68 vs 3.17 mm), so this is not a like-for-like win and should not be quoted as one.
- **TR-BDF2 + adaptive** regresses under *both* couplings (1.14 → 14.4 mm `between`; 1.59 → 26.7 mm
  `during`), so the coupling is not the cause of that regression.

---

# The 2×2: collector × coupling (`matrix.py`, `MATRIX_island_2026-09-16.txt`)

`run.sh` takes `COLLECTOR=implicit|active_set` and `COUPLING=between|during`, giving four corners.
`implicit × between` is the original model; `active_set × during` was the proposed full stack.

## Cost — full-budget SNES iterations / wall s, every scheme, every corner

| scheme | implicit × between | implicit × during | **active-set × between** | active-set × during |
|---|---|---|---|---|
| Anderson BE (secant) | 2869 / 1.5 | 1643 / 1.3 | **1364 / 1.2** | 1368 / 1.3 |
| Anderson BE (volume ΔV) | 2868 / 1.4 | 1635 / 1.3 | **1364 / 1.2** | 1368 / 1.3 |
| Picard BDF2-on-V (plain) | FAIL 10000 / 19.7 | FAIL 10000 / 17.5 | FAIL 10000 / 17.6 | FAIL 10000 / 19.0 |
| Picard BDF2-on-V + T̄ | 1443 / 4.3 | 1235 / 3.3 | 1438 / 3.6 | 1237 / 3.3 |
| TR-BDF2 (fixed dt) | 1771 / 1.2 | 991 / 1.4 | **957 / 1.4** | 956 / 1.4 |
| TR-BDF2 + adaptive dt | 3769 / 2.0 | 3708 / 2.0 | **957 / 1.4** | 956 / 1.4 |
| Newton (plain) | FAIL 2 / 3.3 | FAIL 2 / 3.7 | FAIL 2 / 3.5 | FAIL 2 / 3.4 |
| Newton + dt-continuation | 261203 / 412.5 | 7426 / 13.3 | **2478 / 5.3** | 2478 / 5.3 |

Three things the two extra rows add:

- **Anderson secant and volume-ΔV are indistinguishable** in every corner (2869 vs 2868; 1364 vs
  1364) and give identical water tables. That is the *empirical* confirmation of an algebraic
  identity found separately while fixing the budget accumulator: for backward Euler,
  `updateEffectiveStorativity` IS the secant of `V`, so `S_c·Δh ≡ V(w¹)−V(w⁰)`. The two "different"
  storage forms are the same expression, and the benchmark shows it.
- **Plain Picard fails from cold in ALL four corners** (hits the 10,000-iteration cap). Active-set
  does *not* rescue it — the cold-start failure is the frozen-coefficient contraction, a separate
  disease from the exfiltration constraint, and `-wtm_Tbar` remains the thing that fixes it.
- **Picard + T̄ is nearly collector-insensitive but coupling-sensitive**: 1443 → 1438 across
  collectors, 1443 → 1235 with `during`. It is the one scheme that gets its saving from #116 rather
  than from active-set — because the active-set pin is in the Anderson residual only, so Picard never
  applies it.

Against the original: Anderson **2.1×** fewer iterations, TR-BDF2 **1.85×**, adaptive **3.9×**, and
Newton + continuation **105× fewer iterations / 78× less wall** (412.5 s → 5.3 s).

**Adaptive becomes identical to fixed-dt** (957 vs 957; 91/91 and 336/335 at the iso-precision
targets). It stops subdividing, because the error estimator is no longer chasing the collector's
dt-dependent artifact. That is the clean confirmation that adaptive was the messenger.

## Answer — `active_set × between` is the only scheme-consistent corner

Final max wtd / ponded-cell count:

| corner | result |
|---|---|
| implicit × between | 5.6986/16, but TR-BDF2 5.3776/16 and adaptive **2.3699**/16 — schemes DISAGREE |
| implicit × during | 3.1198/**169**, Newton 0.6937/198, adaptive 0.7835/166 — schemes DISAGREE |
| **active-set × between** | **5.6986 m / 16 for every single scheme** — exact agreement |
| active-set × during | 0.0000/95, 0.0000/160, Picard 5.8757/176 — **broken, see below** |

Under active-set × between all eight rows agree exactly. dt-independence shows up as
scheme-independence, which is what it should look like.

## `active_set` and the `continuous` coupling are STRUCTURALLY INCOMPATIBLE as built

> **SUPERSEDED 2026-09-10. The measurements below stand; the conclusion does not.**
>
> This section's finding was true of the code as it stood: the lake-aware pin read its stage from
> `starting_wtd`, and the continuous coupling exists precisely to stop FSM writing there, so the two
> could not both work. **#40 removed that collision** by giving the lake stage its OWN array -- the
> obstacle now reads the carried stage rather than the overwritten table -- and the old hard error is
> gone. The two COMPOSE.
>
> Two further things moved underneath the conclusion at the end of this section:
> **#51** re-measured the ~11% evaporation evidence that motivated the coupling and it did NOT survive
> (same fixture, sign and trend both inverted), and **#43** then made `continuous` THE DEFAULT anyway,
> on the physical argument alone -- FSM moved that water, so the next step should see it as a source
> rather than as a rewritten initial condition.
>
> So "park #116, keep it gated off" is not the current position. Read what follows as the record of a
> real incompatibility in a version of the code that no longer exists. The names have also changed:
> `-wtm_fsm_continuous` is `surface_water.fsm_coupling: continuous`, `during` is `continuous`, and
> `between` is `impulse`.


The lake-aware pin takes its lake stage from the water table itself
(`transient_groundwater.cpp:2381`):

```cpp
const double surface_water_depth = std::max(0.0, my_starting_wtd[j][i]);  // lagged FSM lake stage (0 off lakes)
```

`-wtm_fsm_continuous` exists precisely to **stop FSM writing its result into `starting_wtd`**. So
under `during`, `surface_water_depth` is ~0 everywhere, the pin skims at the land surface, and lakes cannot fill —
max wtd 0.0000 m. Active-set's lake-awareness *depends on* the very write that #116 removes.

Confirming detail: **Picard is the exception** (5.8757 m / 176) because the pin lives in the Anderson
residual only, so Picard never applies it — and Picard is exactly the row that does *not* show the
drained lake. That is the mechanism reproducing itself in the one place it predicts an exception.

**Consequence: these two changes cannot both be enabled as currently built.** Either the pin must get
its lake stage from a channel #116 preserves (e.g. FSM's lake stage carried explicitly rather than
inferred from `starting_wtd`), or they stay mutually exclusive.

## Caveat on the `during` corners generally

`implicit × during` moves from 16 ponded cells to **169** with a shallower maximum — water spread
across many shallow ponds instead of one deep lake, and the schemes disagree with each other. #116
conserves mass to 6.9e-11, so this is a **redistribution** difference, not a leak — exactly the class
of error a global budget cannot see and that the per-cell ledger (not yet built) would catch. #116's
answer is therefore **not validated** by this benchmark, whatever its conservation properties.

---

# Correction: active-set SUPERSEDES #116; they are not complementary

An earlier note in `tests/budget_closure/run.sh` read active-set + `-wtm_fsm_continuous` closing the
budget ~50× tighter (8e-8 vs 5.8e-7) as evidence the two changes "belong together." **That inference
was wrong** and is retracted here and there.

**Why it was wrong.** On the same fixture the source arm *halves the ponded water* — max wtd
10.0 → 5.0 m, ponded total 160.00 → 79.04 m. The tighter residual is measured on a **materially
different answer**, not a better version of the same one. Tighter closure of a different state is not
evidence of complementarity.

**Why active-set supersedes it.** `-wtm_fsm_continuous` exists to remove the between-step FSM shock.
Active-set removes that shock *by itself*:

| corner | GW move | FSM jump | shock ratio |
|---|---|---|---|
| implicit × between | 46.38 | 45.67 | **0.985** |
| active-set × between | 0.708 | 3.6e-13 | **0.000** |

Under `implicit`, FSM undoes essentially the whole groundwater step every cycle — a limit cycle, and a
real problem that #116 was a reasonable answer to. Under active-set the solve already places cells at
the stage FSM would set, so FSM has nothing left to change. **#116's premise does not survive
active-set**, and neither does the second-order-accuracy argument built on it.

**Its effect is not even consistent in sign.** Ponded total under `during` vs `between`:
island/implicit **+69%**, island/active-set **−100%**, fsm_test/active-set **−51%**. A systematic
physical improvement would not wander like that.

**Conclusion: enable active-set; park #116.** Keep `-wtm_fsm_continuous` gated off — its
source-delivery machinery is the right mechanism if FSM cadence is ever decoupled from the GW step to
get off the serial-FSM ceiling (option B of the original design) **(But FSM is NOT the ceiling: `benchmark/esquibel/FSM_COST.md` measures it at 0.142% of a cycle, GW solve ~99.7%. Corrected 2026-09-23.)**. That is a *cost* lever, not a
correctness one, and is not what the flag was built for.

**What would reopen it:** if active-set does *not* become the default. A 0.985 shock ratio is severe,
and #116's motivation returns in full.
