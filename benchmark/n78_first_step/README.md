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
