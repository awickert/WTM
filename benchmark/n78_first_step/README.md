# #78 reproduction: a one-off first-step mass error on a saturated cold start

`probe.yaml.in` is the purpose-built config that reproduces task #78. It is kept because #78 was
*unbelievable for two years of its life* precisely because every config in the original sweep was
ad-hoc regex surgery on a config written for something else, and the task says so in as many words.
This one is written fresh and pins the things that made the original uninterpretable.

## Running it

Substitute `@INPUTS@ @WORK@ @STEM@ @INTEG@ @MODE@ @DT@ @ROUTING@` and run. Inputs are
`tests/variable_porosity/inputs` (regenerate with that suite's `make_inputs.py`).

```sh
INP=$(readlink -f tests/variable_porosity/inputs); W=/tmp/n78; mkdir -p $W
sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$W|g" -e "s|@STEM@|a|g" -e "s|@INTEG@|backward-euler|g" \
    -e "s|@MODE@|fixed|g" -e "s|@DT@|604800|g" -e "s|@ROUTING@|off|g" probe.yaml.in > $W/a.yaml
./build/wtm.x $W/a.yaml
```

Read `exact_budget_residual` and `total_recharge_added` from the run log **by column name**
(`tests/wtm_log.py`), never by field index.

## What it controls, and why each control is there

| control | why |
|---|---|
| `equilibrium_stop.tol: 0` | no arm may stop early, so every arm covers the same span |
| `time_step.mode: fixed` | **the original sweep shipped `adaptive`**, in which varying `dt` does not vary the step — the controller sizes it. That alone made the first sweep uninterpretable. |
| `report_interval` as a **time** | every arm reports at identical simulated times, and the span is an exact multiple of it at every dt tested (60480000s / dt = 25, 50, 100, 200) |

## What it shows

The error is **one-off, in the first step**, not a growing leak. At dt = 604800 the absolute residual is
`+7.111137e+06` at cycle 0 and `+7.113012e+06` at cycle 9 — constant to four figures. The *ratio* falls
as exactly 1/N only because recharge accumulates in the denominator.

It switches on at a **dt threshold** (between 1.25 and 1.167 wk on this fixture) and **flips sign**
across it: `+5.55e+01` → `-4.52e+07` → `+3.10e+07`. It is not a truncation error.

The mechanism is **surface-water removal on the first step of a saturated cold start**.
`total_surface_removed` at step 0 is `3.8555e+05` above the threshold and `3.9310e+08` below it — 1019×.
Start from a supplied water table 5 m down instead, so there is no surface water to remove, and the
threshold vanishes entirely: residual 2.7e-06 … 3.4e-06 at *every* dt including the finest, with
`total_surface_removed` at cycle 0 exactly `0.0000e+00`.

It is **not backward-euler-specific**. At fixed dt, tr-bdf2 is worse and has no clean regime in this
range (5.26e-02 at 1.25 wk, where backward-euler is 3.27e-06).

`surface_water.routing` is **inert** here — `off` and `continuous` give bit-identical residuals in all
eight pairs, so the FSM coupling is not involved.

Full measurements and the remaining open question are in task #78.

## CORRECTION, same session: dt is NOT the causal variable, and the estimator is NOT blind

Both of the framings above needed testing rather than believing, and both failed.

**"It switches on below a dt threshold" is a proxy, not a cause.** True in `mode: fixed`, but the
`adaptive` arms break it. Two runs, `routing: off`, traced:

| nominal dt | controller settled at | `est` at acceptance (tol 0.5) | cycle-0 residual | final `\|resid\|/rech` |
|---|---|---|---|---|
| 756000 (1.250 wk) | **4.890e+05 (0.809 wk)** | 0.436 | **+1.0201e+02** | 4.06e-07 |
| 604800 (1.000 wk) | **2.225e+05 (0.368 wk)** | 0.444 | **+1.1124e+07** | 1.78e-02 |

The **clean** run ran at the **larger** step, and 0.809 wk is well below the 1.167 wk "threshold". Five
orders of difference in mass error at near-identical estimator readings. Whatever the variable is, it is
not the size of the step.

**And the estimator is not blind.** Every DTTRACE line carries `nest=88` (all land cells in the estimate)
and `ecpl=0.0` (no coupling term — `routing: off`, so there is no FSM and `n_in = 0` cannot arise). At
nominal 604800 the controller *rejected four times*, shrank dt 6.05e5 → 2.39e5, and accepted at
`est=0.444 < tol=0.5`. It did exactly what it is designed to do, and the mass error happened anyway.

So this is **not** an instance of task #58's blindness (`est = 0.0000e+00` because FSM excluded every
cell). It is the sharper statement that **the quantity the controller steers on and the quantity going
wrong are decoupled**: the same `est ≈ 0.44` accompanies both a 1e2 and a 1e7 cycle-0 residual. Compare
`finding_steer_only_on_controllable_error` — the controller was deliberately taught not to steer on error
`dt` cannot reduce; this is the other half of that trade, an error `dt` does not reduce and the estimate
does not see.

### What is established, and what is not

ESTABLISHED, each measured:
1. One-off, in the first cycle; constant to four figures over nine more cycles.
2. Requires surface water present at the start — `initial_water_table: supplied` at −5 m gives
   `total_surface_removed = 0.0000e+00` at cycle 0 and 2.7e-06 … 3.4e-06 at *every* dt.
3. Not backward-euler-specific (tr-bdf2 is worse and has no clean regime in this range).
4. Not the FSM — `routing: off` and `continuous` give bit-identical residuals in all eight pairs.
5. Not dt, and not the estimator (this section).

NOT ESTABLISHED — do not guess it, measure it. What differs between the two adaptive runs above is the
*path through the first cycle*: 2 rejections settling at 4.89e5 versus 4 rejections settling at 2.23e5.
The next experiment is whether the rejection/retry history on the **first** step is the variable, which
would put this next to tasks #13 and #41 rather than anywhere near #58.


## EXPANDED STUDY (same session, after backing off the hypotheses)

Andy: *"Back off of the hypotheses. Redo what caused you to find the first-step and dt dependence. Expand
the study from there."* Both findings were redone with the instrumentation that produced them — **one
report per step** — and the dt axis widened from 4 sampled points to 31. Two of my own conclusions did
not survive the widening.

**Method.** `mode: fixed`, `routing: off`, `saturated` start, one report per step, `total = 4*dt` so every
arm takes exactly four steps. Verified **deterministic**: three arms rerun, `exact_budget_residual`
identical to the last digit (`16188146.6127`, `-45175089.7076`, `11123853.0004`).

### It is NOT a threshold. It is TWO BANDS with clean windows between and outside them.

Step-0 `exact_budget_residual`, backward-euler, all 31 points, nothing dropped (`*` = |resid| > 1e3):

```
  4.0000 +3.27e+00     2.2500 +1.48e-01     1.1250 * +4.11e+07     0.5000 * +1.35e+07
  3.7500 -3.91e-01     2.0000 -3.05e-02     1.0625 * +1.65e+07     0.4375 * +1.37e+07
  3.5000 -4.91e-01     1.8750 +1.76e-03     1.0000 * +1.11e+07     0.3750 * +3.44e+05
  3.2500 -2.15e+00     1.7500 +8.54e-02     0.9375 * +9.65e+06     0.3125 +2.05e-02
  3.0000 * +1.62e+07   1.6250 -4.59e-01     0.8750 * +9.53e+06     0.2500 +1.49e-02
  2.7500 * +3.33e+07   1.5000 -1.30e-01     0.7500 * +1.07e+07     0.1875 -1.75e-02
  2.5000 * +9.53e+07   1.3750 -9.87e-03     0.6250 * +1.23e+07     0.1250 +7.66e-02
                       1.2500 +1.30e-01     1.1875 * -8.27e+07     0.0625 +1.11e-02
```

**Band A** 3.00 → 2.50 wk. **Band B** 1.1875 → 0.4375 wk (0.375 transitional at 3.4e5).
**Clean** ≥ 3.25 wk, 2.25 → 1.25 wk, and ≤ 0.3125 wk. Each band peaks at an *edge* adjacent to a clean
window — A at 2.50 (9.5e7), B at 1.1875 (−8.3e7). `dt = 8 wk` does not converge (DIVERGED_MAX_IT).

**This retracts the "threshold between 1.25 and 1.167 wk".** That came from four sampled points plus a
bisection that assumed a monotone picture. There is no threshold and no monotone trend.

### The error is incurred at step 0 and then FROZEN — across the whole band, not one arm

Cumulative residual over the first four steps: every starred arm is identical to 6–7 significant figures
from step 0 onward (e.g. 2.75 wk: `+3.327940e+07` four times). Every clean arm wanders at O(1) or below
against a step-0 recharge of ~1e6. One exception worth recording: at **0.0625 wk** step 0 is clean
(`+1.11e-02`) and **step 1** jumps to `-1.369e+03`, then freezes — the same shape, three orders smaller.

### `total_surface_removed` does NOT track the residual — retracting a second claim

I wrote that the mechanism *is* surface-water removal, on the strength of two arms (`3.86e+05` clean vs
`3.93e+08` dirty). Across the full sweep that does not hold: the 1019× spike occurs **only** at 1.1875 wk.
At 1.125, 1.0625 and 1.0 wk the removal is `5.12e5 / 4.84e5 / 4.55e5` — on the smooth trend — while the
residual is still 1e7. Removal is mildly elevated inside the bands (≈2× in A, ≈15% in B), nowhere near
enough to account for 1e7. `total_ocean_outflow` also departs from its smooth trend inside the bands.
Surface water at the start is still a **necessary** condition (the −5 m control zeroes the effect at every
dt); it is not the quantity that varies with it.

### The bands MOVE with the integrator

Step-0 residual on a shared 16-point grid:

| dt (wk) | backward-euler | tr-bdf2 | bdf2 |
|---|---|---|---|
| 4.0000 | +3.2668e+00 | **+1.9309e+08** | +3.2668e+00 |
| 3.2500 | −2.1499e+00 | **+2.1243e+07** | −2.1499e+00 |
| 3.0000 | **+1.6188e+07** | **+2.1573e+07** | **+1.6188e+07** |
| 2.5000 | **+9.5302e+07** | **+2.5741e+07** | **+9.5302e+07** |
| 2.2500 | +1.4842e-01 | **+2.8071e+07** | +1.4842e-01 |
| 1.2500 | +1.2964e-01 | **+3.2906e+07** | +1.2964e-01 |
| 1.1875 | **−8.2662e+07** | +4.3366e+05 | **−8.2662e+07** |
| 1.0000 | **+1.1124e+07** | **+3.1363e+07** | **+1.1124e+07** |
| 0.7500 | **+1.0729e+07** | −4.1162e-02 | **+1.0729e+07** |
| 0.4375 | **+1.3738e+07** | −1.7923e-01 | **+1.3738e+07** |
| 0.2500 | +1.4931e-02 | −2.2010e-02 | +1.4931e-02 |

`tr-bdf2` is bad at *large* dt and clean at dt ≤ 0.75 wk — a different structure, not a shifted one.

**backward-euler and bdf2 agree to every printed digit at all 16 points**, which is the expected answer
and a soundness check on the harness: BDF2 is self-starting, so its *first* step is a backward-Euler step.

### Still open

The cause. Not proposing one here — the two I proposed before this sweep (a dt threshold, surface-water
removal) were both products of too few sample points.
