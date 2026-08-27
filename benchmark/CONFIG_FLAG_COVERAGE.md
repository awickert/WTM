# Flag → YAML coverage: what the config replaces, and what it does not

**Date:** 2026-08-27
**Scope:** every `-wtm_*` flag the model parses (65), against the nested-YAML schema as it currently
stands. PETSc's own flags (`-snes_*`, `-ksp_*`, `-pc_*`) are out of scope — those belong to PETSc and
should keep their CLI surface.

**Why this exists.** Configuration currently arrives through two channels, and 29 of 31 test runners
use *both in the same invocation*. That split does not merely duplicate: it lets one channel override
the other with no record anywhere, because neither system logs the other. `-wtm_extended_soil` versus
`surface_water.collection.method` was exactly that, and cost a day. Unknown YAML keys now abort
(`tests/config_schema`); flags are still accepted silently, so the two channels no longer offer the
same guarantees, and that asymmetry is an argument for finishing the migration rather than a reason to
pause it.

**Not every row wants a 1:1 replacement.** The YAML deliberately simplifies: `solver.method` picks a
strategy and hides which internal flags implement it, and standard settings are defaulted so a normal
user never sees the knobs. The useful question per flag is therefore not "is there a key with this
name" but **"can a user express this intent, at the right level of abstraction, from the config
alone."** The status column answers that.

## Status legend

| status | meaning |
|---|---|
| **1:1** | a YAML key sets exactly this flag; the flag is redundant |
| **ABSTRACTED** | YAML expresses the *intent* at a higher level; the flag is an implementation detail of a YAML value and should never be user-facing |
| **SUPERSEDED** | a YAML mode has replaced it; the flag survives as a legacy alias or a dead switch and is a **retirement candidate** |
| **GAP — user** | a setting a user could legitimately want, with no config path |
| **GAP — advanced** | genuine tuning, and a candidate for the "expose advanced settings under the method that owns them" pattern rather than a top-level key |
| **DEV** | developer/diagnostic escape hatch; exposing it in a user config would be wrong |

## Summary

Counts are per FLAG and were computed from the tables below, not estimated; they sum to the 65 flags
`grep`ed out of `src/`.

| status | count |
|---|---|
| **1:1** | 17 |
| **ABSTRACTED** | 8 |
| **SUPERSEDED** — retirement candidates | 6 |
| **GAP — user** | 7 |
| **GAP — advanced** | 23 |
| **DEV** | 4 |
| **total** | **65** |

**Reachable from a config file today: 26.** That is not simply 1:1 + ABSTRACTED (25), and the two
places it differs are worth stating rather than smoothing over:

- `-wtm_anderson` is classified ABSTRACTED but has no bridge entry, because Anderson is the DEFAULT —
  `solver.method: anderson` reaches it by setting no flag at all.
- `-wtm_dev_allow_aboveground_water_columns` and `-wtm_dev_padded_dirichlet` are classified DEV but
  *are* bridged, under `dev:`. Being developer knobs and being config-reachable are independent.

So: 25 − 1 (anderson, unbridged) + 2 (dev, bridged) = 26, matching the bridge count measured directly
from `apply_config_petsc_options`.

The headline: the config already covers the *common* path well. What it does not cover is (a) six
switches a YAML mode has already replaced, which are now pure hazard, and (b) two clusters of advanced
tuning — the step-size controller and the Anderson/handoff machinery — that have no config expression
at all.

---

## Solver strategy

`solver.method` is the abstraction: it selects a strategy and sets the flags that implement it.
Anderson is the default and needs no flag, which is why `-wtm_anderson` is ABSTRACTED rather than a gap.

| flag | what it does | status | YAML today |
|---|---|---|---|
| `-wtm_picard` | semi-implicit Picard (SPD operator, CG+GAMG) | ABSTRACTED | `solver.method: picard` |
| `-wtm_newton` | Newton-Krylov on the analytic Jacobian | ABSTRACTED | `solver.method: newton` |
| `-wtm_anderson` | Anderson mixing, matrix-free | ABSTRACTED | `solver.method: anderson` (the default) |
| `-wtm_aa_picard` | Anderson-accelerated GAMG-Picard (nonlinear preconditioning) | GAP — advanced | none — a fourth strategy `solver.method` does not offer |
| `-wtm_handoff` | run Anderson, then hand the best iterate to a finisher | GAP — advanced | none |
| `-wtm_handoff_picard` | make that finisher Picard instead of Newton | GAP — advanced | none |
| `-wtm_handoff_patience` | stalled iterations before handing off | GAP — advanced | none |
| `-wtm_handoff_max_it` | cap on the Anderson phase | GAP — advanced | none |
| `-wtm_stiff` | convenience bundle: newton + continuation + eq_tol | GAP — user | none — this is a *preset*, and presets are exactly what a config should carry |
| `-wtm_relax` | sub-step under-relaxation (1 = off) | GAP — advanced | none |
| `-wtm_predict_guess` | predictor-seeded initial guess | GAP — advanced | none |
| `-wtm_kirchhoff` | Kirchhoff variable change | GAP — advanced | none |

The handoff and `aa_picard` cluster is the clearest case for your "advanced settings under the method
that owns them" pattern: they are all *how Anderson behaves*, so they belong under
`solver.method: anderson` as sub-keys, not as top-level switches.

## Anderson restart control

Five flags, none reachable, all tuning one mechanism.

| flag | what it does | status | YAML today |
|---|---|---|---|
| `-wtm_adaptive_restart` | restart Anderson's history when the convergence RATE degrades | GAP — advanced | none |
| `-wtm_ar_rho` | the rho threshold that triggers it | GAP — advanced | none |
| `-wtm_ar_patience` | consecutive degrading iterations before restarting | GAP — advanced | none |
| `-wtm_ar_max_it` | iteration cap | GAP — advanced | none |
| `-wtm_ar_max_restarts` | restart cap | GAP — advanced | none |

## Time integration

| flag | what it does | status | YAML today |
|---|---|---|---|
| `-wtm_tr_bdf2` | TR-BDF2, L-stable 2nd order | ABSTRACTED | `solver.time_integration: tr-bdf2` |
| `-wtm_bdf2_on_V` | BDF2 applied to stored volume V(h) | ABSTRACTED | `solver.time_integration: bdf2` |
| `-wtm_volume_storage` | backward-Euler storage as exact ΔV, not secant S·Δh | ABSTRACTED | `solver.storage: volume` |
| `-wtm_Tbar` | time-averaged interblock transmissivity | 1:1 | `solver.t_bar` |
| `-wtm_bdf2` | the ORIGINAL BDF2 (head form), pre-`bdf2_on_V` | SUPERSEDED | `time_integration: bdf2` resolves to `bdf2_on_V`; this flag reaches an older path nothing selects |

## Adaptive dt and the step-size controller

The single largest gap, and the one with a demonstrated cost: a sweep of `-wtm_dtc_easy_iters` returned
byte-identical results at every setting because the flag was parsed only on the continuation path
(fixed in `395915e`) — a false negative that was recorded as a finding.

| flag | what it does | status | YAML today |
|---|---|---|---|
| `-wtm_dt_adaptive` | enable the adaptive controller | 1:1 | `solver.adaptive_dt` |
| `-wtm_dt_tol` | per-step local-error target, in water volume | 1:1 | `solver.water_volume_timestep_error_tol` |
| `-wtm_dtc_dt_max` | cap on dt | 1:1 | `solver.dt_max` |
| `-wtm_dtc_grow` | growth factor on an easy step | GAP — advanced | none |
| `-wtm_dtc_shrink` | shrink factor on a reject | GAP — advanced | none |
| `-wtm_dtc_easy_iters` | iteration count below which dt may grow | GAP — advanced | none |
| `-wtm_dtc_max_retries` | consecutive rejects before giving up | GAP — advanced | none |
| `-wtm_dtc_dt0` | starting dt for the continuation ramp | GAP — advanced | none |
| `-wtm_dt_continuation` | Newton's dt ramp | **GAP — user** | none — and Newton does **not converge without it** on these fixtures (pinned by `tests/newton_solver`), so a config-only user cannot run Newton at all |
| `-wtm_dt_norm_rms` / `-wtm_dt_norm_max` | adaptive error norm: RMS (default) or MAX | GAP — advanced | none |
| `-wtm_dt_trace` | report (dt, est, tol, factor, iters, accepted) per step | DEV | none — diagnostic |

`-wtm_dt_continuation` is the row to look at first. It is not advanced tuning; it is a *requirement*
of a strategy the config offers, and its absence makes `solver.method: newton` unusable from YAML alone.

## Surface water

The collection selector is the abstraction that already works — and six flags are now redundant
against it. These are the retirement candidates, and they are more than clutter: two channels for one
decision is what produced the `extended_soil` collision.

| flag | what it does | status | YAML today |
|---|---|---|---|
| `-wtm_direct_to_runoff` | in-residual exfiltration removal | SUPERSEDED | `collection.method: implicit` |
| `-wtm_surface_exfiltration_to_runoff` | post-solve clamp | SUPERSEDED | `collection.method: explicit` |
| `-wtm_surface_sink` | sub-surface band sink | SUPERSEDED | `collection.method: legacy` |
| `-wtm_extended_soil` | continue the aquifer above the surface | SUPERSEDED | `collection.method: extended_soil` (this flag is now its documented legacy alias) |
| `-wtm_active_set` | semismooth exfiltration pin | 1:1 **and** ABSTRACTED — reachable **two ways** (`dev.active_set` *and* `collection.method: active_set`) | resolve to one |
| `-wtm_dev_active_set` | the older name for the same thing | SUPERSEDED | `collection.method: active_set` |
| `-wtm_surface_sink_qmax` | band-sink peak removal rate | 1:1 | `collection.sink.qmax` |
| `-wtm_surface_sink_width` | band width below the surface | 1:1 | `collection.sink.width` |
| `-wtm_fringe_source` | capillary-fringe width source | 1:1 | `collection.sink.fringe_source` |
| `-wtm_fringe_cap` | max ψ_a | 1:1 | `collection.sink.fringe_cap` |
| `-wtm_fringe_ksat_coef` | ψ_a = C·√(n/ksat) | 1:1 | `collection.sink.fringe_ksat_coef` |
| `-wtm_fringe_length` | uniform fringe length | 1:1 | `collection.sink.fringe_length` |
| `-wtm_fsm_delta_source` | carry FSM's Δwtd as a source in the next step | DEV | none — experimental |

`-wtm_active_set` deserves attention: it is currently reachable from **two different YAML keys**
(`dev.active_set` and `collection.method: active_set`). That is the same two-channel shape as the bug
just fixed, only entirely inside the config. One of them should go.

## Evaporation and tapers

The bridge exposes the taper *parameters* but not the *toggles*, which is an odd half-migration: a user
can retune the sigmoid but cannot turn it off.

| flag | what it does | status | YAML today |
|---|---|---|---|
| `-wtm_evap_taper_wtdc` | sigmoid centre | 1:1 | `evaporation.et_sigmoid.wtd_center` |
| `-wtm_evap_taper_s` | sigmoid width | 1:1 | `evaporation.et_sigmoid.logistic_width` |
| `-wtm_extinction_depth` | extinction depth | 1:1 | `evaporation.extinction_depth` |
| `-wtm_evap_taper` | taper 2 **on/off** | **GAP — user** | none — parameters are configurable, the switch is not |
| `-wtm_extinction` | taper 3 **on/off** | **GAP — user** | none — same |

## Transmissivity and boundaries

| flag | what it does | status | YAML today |
|---|---|---|---|
| `-wtm_T_bedrock` | additive background transmissivity | 1:1 | `transmissivity.additive_background_transmissivity` |
| `-wtm_land_boundary` | land boundary condition | ABSTRACTED | `boundaries.land` (value translated) |
| `-wtm_ksat_surface_smoothing_width` | round the ksat kink at the surface | GAP — user | none — a modelling choice, not a developer knob |
| `-wtm_ksat_soilbottom_smoothing_width` | round the ksat kink at −1.5 m | GAP — user | none — same |
| `-wtm_storativity_surface_smoothing_width` | round the storativity kink at the surface | GAP — user | none — same |

## Convergence criteria

| flag | what it does | status | YAML today |
|---|---|---|---|
| `-wtm_eq_tol` | equilibrium stop tolerance | 1:1 | `run.equilibrium_stop.tol` |
| `-wtm_eq_metric` | which metric judges equilibrium | 1:1 | `run.equilibrium_stop.metric` |
| `-wtm_eq_frac` | fraction-of-cells threshold | 1:1 | `run.equilibrium_stop.frac` |
| `-wtm_snes_volume_conv` | judge the SNES step in water, not head | GAP — advanced | none |
| `-wtm_snes_volume_conv_govern` | make that judgement authoritative | GAP — advanced | none |
| `-wtm_snes_vol_tol` | its relative tolerance | GAP — advanced | none |

## Developer

| flag | what it does | status | YAML today |
|---|---|---|---|
| `-wtm_dev_allow_aboveground_water_columns` | disable the surface clamp entirely | DEV | `dev.allow_aboveground_water_columns` (already exposed) |
| `-wtm_dev_padded_dirichlet` | verification tool for the ghost scheme | DEV | `dev.padded_dirichlet` (already exposed) |

---

## What this suggests, in order

1. **Retire the six SUPERSEDED flags.** They are duplicate channels for decisions a YAML mode already
   owns, and duplicate channels are what produced the `extended_soil` collision. Zero capability lost.
2. **Resolve `-wtm_active_set`'s two YAML routes** (`dev.active_set` vs `collection.method`). Same
   hazard, already inside the config.
3. **Close the four user-facing gaps that block documented workflows**: `-wtm_dt_continuation` (Newton
   is unusable from YAML without it), the two taper toggles, and `-wtm_stiff` as a preset.
4. **Group the advanced clusters under the method that owns them** rather than as top-level keys — the
   step-size controller under `solver.adaptive_dt`, and the Anderson restart/handoff machinery under
   `solver.method: anderson`. That is the pattern the schema already uses well elsewhere, and it keeps
   the common path uncluttered.
5. **Then, and only then, add a controlled flag list.** Once the remaining flags are known and few, an
   unknown-flag check can be as strict as the YAML one is now, and the asymmetry closes.

The three surface-smoothing widths are the judgement call: they are genuinely modelling choices rather
than developer knobs, so they arguably belong in `transmissivity:` — but they are also default-off and
were shown not to fix the free-boundary order loss (`BDF2_RECHARGE_ORDER.md` §15), so exposing them
prominently may advertise a dead end.
