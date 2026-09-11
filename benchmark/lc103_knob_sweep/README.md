# #103: which parameters permit the surface limit cycle

Reproduction record, 2026-09-11. One knob at a time from the shipped
`tests/storage_equivalence/config.yaml`, everything else held fixed.

## What is measured

Two statistics, both stated here rather than in prose beside the numbers:

- **`settle`** = `max |wtd(last saved raster) − wtd(previous)|` over land cells (column 0 is ocean and is
  excluded), in metres. One report interval of motion at the END of the run.
- **`late/early`** = `max(abs_change_volume_max over the last 10 cycles) / max(over cycles 5..14)`, from
  the model's own per-cycle run-log column. A single snapshot cannot tell a decaying transient from a
  limit cycle; this ratio can. ~1 means the motion is not decaying.

Every arm ran **460.0 simulated years**, verified from `elapsed_time_s`, so the arms are comparable.
`settle` here is NOT the same statistic as the within-cycle max|dw| quoted in #103's original note, so
the two sets of numbers are comparable within themselves and not across.

## Result

| arm | settle (m) | n>1mm | at surface | late/early | verdict |
|---|---|---|---|---|---|
| **baseline** (explicit, secant, adaptive, BE) | 6.4379e-03 | 32/88 | 56/88 | 0.507 | **not decaying** |
| `collection.method: active_set` (+volume) | 4.3109e-07 | 0/88 | 56/88 | 0.000 | settled |
| `collection.method: off` | 3.7256e+00 | 88/88 | 0/88 | 0.310 | mounding, never settles |
| cells/degree 16 | 0.0000e+00 | 0/88 | **88/88** | — | settled (fully saturated) |
| cells/degree 256 | 1.3976e-06 | 0/88 | **0/88** | 0.000 | decays away |
| `storativity_surface: 0` | — | — | — | — | **DIVERGED_FNORM_NAN** |
| `storativity_surface: 0.1` | 5.8567e-03 | 32/88 | 56/88 | 0.208 | not cured |
| `surface_transition: false` | 6.4379e-03 | 32/88 | 56/88 | 0.507 | **inert** (bit-identical) |
| `storage_form: volume` | 1.2151e-04 | 0/88 | 56/88 | 0.329 | ~50x smaller, not cured |
| **`time_step.mode: fixed`** | **7.1054e-14** | 0/88 | 56/88 | 0.000 | **settled** |
| `dt / 4` (still adaptive) | 2.9308e-04 | 0/88 | 56/88 | 0.000 | decays |
| `time_integration: tr-bdf2` | 5.2176e-02 | 32/88 | 56/88 | **1.341** | worse, and growing |

`active_set` is run with `storage_form: volume` because the model refuses `secant x active_set` by name.
That arm therefore moves two knobs; `storage_form: volume` alone is listed separately so its share is
visible (it accounts for ~50x of the improvement, not the remaining four orders).

## The three conditions

Each of these alone removes the cycle, so each is necessary:

1. **A PARTIAL set of cells at the surface.** At 16 cells/degree all 88 are saturated and nothing moves;
   at 256 none are at the surface and the motion decays away. Only the intermediate case, 56 of 88 at
   `wtd == 0` exactly, sustains it.
2. **`collection.method: explicit`.** `active_set` settles at the same cell size, and does it in 118
   nonlinear solves against 74509.
3. **Adaptive time stepping.** `mode: fixed` settles to 7.1e-14 in 6000 solves, with the SAME collector
   and the SAME cell size. This was not known when #103 was written.

## Solve counts, same 460 years

| arm | solves | rejects |
|---|---|---|
| baseline (explicit + adaptive) | **74509** | 1 |
| `mode: fixed` | 6000 | 0 |
| `active_set` | **118** | 2 |

## Reproduce

    export PYTHONPATH=<repo>/tests
    # regenerate fixtures at 16/64/256 cells per degree, then:
    ./sweep.sh
