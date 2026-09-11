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
