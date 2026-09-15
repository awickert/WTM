# #64 re-measured under the right yardstick (2026-09-15)

Andy: *"Adaptive time stepping may give less accurate results. But they should be more robust, and if
the steps are longer, they might be faster. dt should self-correct."*

#64's evidence scored adaptive on ACCURACY against a dt-refined truth, which is not what adaptive
claims. Re-measured on cost-to-accuracy instead. Fixture `tests/golden` `transient`, 8 yr, 256 cells,
`active_set`, `continuous`. Truth = fixed `dt` = 1/1000 yr.

## 1. Cost to reach an accuracy

| mode | dt0 | steps | max\|dV\| (m) | rms (m) |
|---|---|---|---|---|
| fixed | 1/1 | 8 | 2.8977e+00 | 2.2344e-01 |
| adaptive | 1/1 | 6 | 2.9632e+00 | 2.1729e-01 |
| fixed | 1/8 | 64 | 1.3093e+00 | 1.0762e-01 |
| adaptive | 1/8 | **13** | **1.2502e+00** | 1.0682e-01 |
| fixed | 1/16 | 128 | 1.0549e+00 | 9.4508e-02 |
| adaptive | 1/16 | **17** | 1.0642e+00 | 9.2380e-02 |
| fixed | 1/32 | 256 | 9.5980e-01 | 8.4658e-02 |
| adaptive | 1/32 | **17** | 1.0428e+00 | 8.4125e-02 |
| fixed | 1/128 | 1024 | 6.9743e-01 | 6.0329e-02 |
| adaptive | 1/128 | **19** | 9.1864e-01 | 7.4109e-02 |

**#64's headline does not reproduce.** "Adaptive is worse than fixed at equal cost" is the reverse here:
at matched accuracy adaptive needs **13-15x fewer steps** (17 against 256 for ~1.0 m; 19 against ~256
for ~0.95 m). At 1/8 it is both cheaper AND more accurate -- 13 steps and 1.2502e+00 against 64 steps
and 1.3093e+00.

**But adaptive SATURATES.** Refining `dt0` 128x moves it only 2.96 -> 0.92 and its step count only
6 -> 19, because the controller grows away from `dt0` regardless. Fixed keeps improving (0.70 at 1/128,
still falling). So adaptive reaches a decent answer very cheaply and **cannot be converged**.

## 2. Is the ceiling the missing refinement knob? NO.

`dt0` = 1/128 yr throughout; only `solver.time_step.dt_max` varies.

| dt_max | steps | max\|dV\| (m) |
|---|---|---|
| 1/128 yr (= fixed) | 1024 | 6.9743e-01 |
| 1/32 yr | 259 | 8.7899e-01 |
| 1/8 yr | 70 | 9.2928e-01 |
| 1/2 yr | 26 | 9.0197e-01 |
| 1 yr | 20 | 9.1356e-01 |
| unbounded | 19 | 9.1864e-01 |

The error is FLAT at ~0.9 m across a 14x range of ceilings while the step count moves 19 -> 259.
**259 steps buys nothing over 19.** Only collapsing the ceiling onto `dt0` -- which is fixed stepping,
verified bit-identical elsewhere in this session -- reaches 0.70. So `dt_max` must NOT be offered as an
accuracy control: intermediate values are pure cost.

The shape says the error is dominated by a few large-delta coupling events. Capping partially does not
help, because the damage is done by whatever coarse steps remain.

## What this means for #64

- Adaptive is doing its job. Keep it, and stop treating its accuracy against a refined truth as a defect.
- There is a PLATEAU (~0.9 m here) adaptive cannot pass, and a CLIFF to get below it: 54x the steps for
  a 24% gain. That is a property to DOCUMENT, not a bug to fix.
- No controller change follows from this. A (continuous PI on the coupling error) and C (sub-cycling)
  were both aimed at an accuracy problem that is really a cost cliff.

## Honest limits

- **Wall time cannot be measured on this fixture.** It sat at 0.77-1.63 s across 8 to 1024 steps --
  startup-dominated at 256 cells. Step count is the proxy used throughout. The speed claim needs a
  fixture large enough for solve time to dominate.
- ONE fixture, one 8 yr transient. `fsm_impulse` and `fsm_evap0` are not re-measured.
- Robustness -- does adaptive complete where fixed fails -- is NOT tested here. It is the claim with the
  best physical motivation and the least evidence either way.

## Two metric bugs found and fixed while doing this, both of which produced plausible wrong tables

1. A python heredoc nested inside a shell command substitution inside a loop: corrupted 3 of 8 rows.
2. `re.search(r'_(\d+)_', path)` to get the raster index matched the STEM's number first
   (`w64_fixed_1_000000001_8yr.tif` -> group(1) = "1"), so every file in an arm sorted equal and
   `fs[-1]` was whatever order `glob` returned -- sometimes the **0 yr** raster. The tell was one value,
   `1.1791e+01 / 5.4330e+00 / 196`, recurring across unrelated arms: the initial condition scored
   against truth. Fixed by matching `_(\d{9})_` and asserting the scored file ends `_8yr.tif`.
