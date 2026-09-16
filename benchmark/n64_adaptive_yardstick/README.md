# #64 re-measured, and what it actually turned out to be (2026-09-16)

**Read the conclusion first: there is no time-stepping defect here.** The thing #64 recorded as an
accuracy problem is three cells, and those three cells are where a lake's water leaves the domain.
When two outlets tie, the depression hierarchy picks one arbitrarily -- which `src/dephier.hpp`
documents in as many words -- and this fixture is built so that they always tie.

The suite that establishes that is `tests/fsm_exit_path`. This directory is the measurement that led
there, kept for its numbers and for the two metric mistakes it took to get them right.

## The metric

`|V(run) - V(reference)| ` per cell, from `tests/wtm_volume.py`, in **metres of water** (not head).

- **Reference = the `dt` = 1/1000 yr run of the same routing setting.** Never the previous rung of a
  refinement ladder: under adaptive stepping two runs at different `dt0` are different step PATTERNS,
  not a refinement of one, so differencing them measures nothing. An earlier version of this file did
  exactly that and its tables were meaningless.
- **Land cells only.** Ocean cells are pinned, contribute exactly 0, and only dilute a norm.
- **Median AND max.** Reported together, always. Max alone is what made a three-cell artefact look
  like a model-wide inaccuracy for most of a day.

Fixture `tests/golden` `transient`: 16x16, 196 land cells, 8 simulated years, TR-BDF2, `active_set`.

## Cost to reach an accuracy, routing on

| mode | dt0 | steps | median | max |
|---|---|---|---|---|
| fixed | 1/8 | 64 | 0.0000e+00 | 1.3093e+00 |
| adaptive | 1/8 | **13** | 8.8818e-15 | **1.2502e+00** |
| fixed | 1/32 | 256 | 0.0000e+00 | 9.5980e-01 |
| adaptive | 1/32 | **17** | 0.0000e+00 | 1.0428e+00 |
| fixed | 1/128 | 1024 | 0.0000e+00 | 6.9743e-01 |
| adaptive | 1/128 | **19** | 0.0000e+00 | 9.1864e-01 |

**#64's headline does not reproduce.** "Adaptive is worse than fixed at equal cost" is the reverse:
at matched accuracy adaptive needs 13-15x fewer steps, and at `dt0` = 1/8 yr it is both cheaper and
more accurate. Adaptive also saturates -- refining `dt0` 128x moves it only 1.25 -> 0.92 m -- but that
is adaptive doing what it is for. It targets `error_tol` and stops; it was never a convergence ladder.

**The median is zero almost everywhere.** That is the real content of the table: at every step size,
in both modes, more than half the land matches the 1/1000 yr run bit-for-bit.

## Where the error actually is

| mode | dt0 | land cells with any error | > 1 mm | **> 10 cm** |
|---|---|---|---|---|
| fixed | 1/8 | 95 of 196 | 74 | **3** |
| fixed | 1/128 | 95 | 34 | **3** |
| adaptive | 1/128 | 95 | 60 | **4** |

The three sit in one column, against the ocean, on the side of the domain nearest the pit. The other
~50 land cells with an ocean neighbour are exact. It is not "coastal cells are hard" -- it is the
discharge path. `tests/fsm_exit_path` mirrors the geometry and shows the error mirrors with it.

## Routing off: the control

| mode | dt0 | median | max |
|---|---|---|---|
| fixed | 1/128 | 2.4570e-06 | 1.5039e-03 |
| adaptive | 1/128 | 2.1712e-04 | 3.6280e-02 |

No concentration, nothing above 10 cm anywhere. The effect is entirely an FSM factor.

## What this does NOT establish

- **Wall-clock speed.** It sat at 0.77-1.63 s across runs of 8 to 1024 steps -- startup-dominated at
  256 cells. Every "faster" claim above is in STEPS. A fixture large enough for solve time to dominate
  would be needed to say anything about seconds.
- **Convergence ORDER.** An earlier version of this file quoted observed orders around +0.7 and drew
  conclusions from them. They disagree with the ~2.0 the CHANGELOG records for TR-BDF2 with FSM off,
  and the disagreement was never explained -- three rungs is too thin a basis and the max-norm is one
  cell. Those numbers are withdrawn, not corrected. Use a purpose-built order measurement.
- **Robustness**, which is the claim adaptive actually rests on: whether it completes runs that fixed
  stepping cannot. Untested here, and still untested anywhere.
- **Anything about real terrain.** One synthetic fixture, a square island with a one-cell ocean moat.

## Two metric bugs found here, both of which produced plausible wrong tables

1. A python heredoc nested inside a shell command substitution inside a loop: corrupted 3 of 8 rows.
2. `re.search(r'_(\d+)_', path)` to extract the raster index matched the STEM's number first, so every
   file in an arm sorted equal and `fs[-1]` was whatever order `glob` returned -- sometimes the **0 yr**
   raster. The tell was one value recurring across unrelated arms: the initial condition scored against
   truth. Fixed by matching `_(\d{9})_` and asserting the scored file ends `_8yr.tif`.

Both were caught by the same thing: a number that repeated where it had no business repeating.
