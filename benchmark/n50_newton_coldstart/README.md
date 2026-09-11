# #50: does Newton still need the dt-continuation ramp for cold starts from far?

**Yes.** Measured 2026-09-11 on the current binary. The five documentation sites that say so are correct
and were left alone.

## Why the question was open

`tests/newton_solver`'s CONTRACT/a inverted on 2026-09-04 (`7289652`): plain Newton at fixed dt began
converging where it used to abort. But that arm runs **one small, easy fixture** (`fsm_consistency` at
2 yr), and the docs make a broader claim about cold starts from far on hard terrain. One easy fixture
does not refute it, so the sites were deliberately left pending this measurement.

## The measurement

Fixture: `tests/tolerance_independence`'s — `initial_water_table: saturated` on a 100 m plateau draining
to an ocean strip, so the water table must fall tens of metres from its initial guess. Cold, and far.
It is also the fixture `#104` was characterised on, so its behaviour is mapped rather than assumed.

Plain Newton — `solver.method: newton`, `solver.time_step.mode: fixed`, no ramp, no adaptive:

| dt (wk) | rc | outcome |
|---|---|---|
| 4.0000 | 1 | 2 × `DIVERGED_LINE_SEARCH` |
| 2.0000 | 1 | 2 × `DIVERGED_LINE_SEARCH` |
| **1.0000** | **0** | 1 × `CONVERGED_FNORM_ABS`, 1 × `DIVERGED_LINE_SEARCH` |
| 0.5000 | 1 | 2 × `DIVERGED_LINE_SEARCH` |
| 0.2500 | 1 | 2 × `DIVERGED_LINE_SEARCH` |
| 0.1250 | 1 | 2 × `DIVERGED_LINE_SEARCH` |

Five of six abort with `ERROR: The SNES solver has not converged`. The identical configuration with
`mode: ramp` (`newton.dt0: "157680s"`) completes, `rc=0`.

## The single success is not a capability

`dt = 1.0 wk` converges and **both** its neighbours fail. That is the same single-dt trap `#104` produced
earlier the same day, where an error appeared in bands of dt and a four-point sample manufactured a
threshold that did not exist. Sampling only 1.0 wk here would have produced the opposite conclusion and
sent someone to delete a warning their production cold start depends on.

## Reproducing

```sh
INP=$(readlink -f tests/tolerance_independence/inputs); W=/tmp/n50; mkdir -p $W
sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$W|g" -e "s|@STEM@|p|g" -e "s|@DT@|604800|g" -e "s|@WVTOL@|1e-08|g" \
    tests/tolerance_independence/config.yaml | sed -e "s|^  method: anderson|  method: newton|" > $W/p.yaml
./build/wtm.x $W/p.yaml        # mode: fixed -- the plain arm. Vary solver.time_step.dt.
```

## Consequence

`#38` wanted to collapse `adaptive_dt` and `newton.dt_continuation` into one enum and asked whether the
ramp is still a *requirement* anywhere. It is. The enum does not get simpler.
