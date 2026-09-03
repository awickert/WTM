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

> **UPDATE 2026-08-27.** Ten of the seventeen 1:1 rows are now **RETIRED**: the setting moved onto
> `Parameters`, is parsed from the config, is covered by the schema check, and is read directly by its
> consumer; the flag and its bridge entry are deleted. Passing a retired flag now ABORTS (the
> unconsumed-flag check, 750ffb1) rather than being silently ignored. Seven 1:1 rows remain:
> `-wtm_eq_tol`, `-wtm_eq_metric`, `-wtm_Tbar`, `-wtm_active_set`, `-wtm_dt_adaptive`, `-wtm_dt_tol`,
> `-wtm_dtc_dt_max` -- these have live CLI call sites across the suite (eq_tol alone has 61), so each
> needs its callers moved to the config in the same commit.

> **UPDATE 2026-09-01 (latest).** Five of the nine ABSTRACTED flags are RETIRED as well --
> `-wtm_land_boundary`, `-wtm_volume_storage`, `-wtm_dt_continuation`, `-wtm_picard`,
> `-wtm_active_set`. **27 of the 65 flags are gone.** Four ABSTRACTED remain (`newton`, `bdf2_on_V`,
> `tr_bdf2`, `anderson`), then GAP-user 6, GAP-advanced 24, DEV 4.
>
> Each was proved equivalent BEFORE removal, not asserted: the flag route and the config route were run
> and required byte-identical. `-wtm_picard` additionally needed a 96-combination bisect, because the
> first attempt produced a false IMPROVEMENT -- the sweep's dt/8 retry had silently reverted to the
> default solver, since a setting that moves from flag to config reaches only the config-construction
> sites, and that harness has two.

> **UPDATE 2026-09-01 (later).** The three MODE INTERFACE flags are RETIRED too, with
> `collection.method: legacy` and the taper-1 band sink (fork issue #7). 20 of the 65 flags are now gone.
> The `-wtm_fringe_*` rows below describe knobs that no longer exist: they sized the sink's band.

> **UPDATE 2026-09-01.** All **seventeen** 1:1 rows are now RETIRED -- the seven listed above followed.
> Separately, `-wtm_active_set`'s SECOND YAML route (`dev.active_set`) was removed; see the
> surface-water table below. The 2026-08-27 note above is kept as written: it records the state on
> that date, not the state now.

## "Superseded" means two different things — keep them apart

An earlier draft of this document used one word for both, and the confusion produced a wrong
retirement list. They are not the same claim:

- **Classification** — *this flag's function is expressible as a YAML mode.* `-wtm_extended_soil`
  corresponds to `collection.method: extended_soil`.
- **Runtime** — *an explicitly configured method overrides a flag passed on the command line.*
  `collection.method: explicit` plus `-wtm_extended_soil` gives you explicit, with a warning.

The second is not evidence for the first. `method: explicit` does **not** imply extended soil — it is a
mutually exclusive alternative that *disables* it. Citing the runtime override as though it showed the
classification is what put three flags on a retirement list they did not belong on.

## Status legend

| status | meaning |
|---|---|
| **1:1** | a YAML key sets exactly this flag; the flag is redundant |
| **ABSTRACTED** | YAML expresses the *intent* at a higher level; the flag is an implementation detail of a YAML value and should never be user-facing |
| **RETIRED** | done: the setting is a `Parameters` member, parsed from the config and schema-checked; flag and bridge entry deleted. Passing it now aborts |
| **ALIAS** | a documented alternate entry point that RESOLVES to a YAML mode and warns when a configured method overrides it. Removable, but nothing is broken by keeping it |
| **MODE INTERFACE** | the flag IS how a YAML mode is expressed. `collection.method: legacy` hands control back to these deliberately, so deleting the flag deletes the mode |
| **GAP — user** | a setting a user could legitimately want, with no config path |
| **GAP — advanced** | genuine tuning, and a candidate for the "expose advanced settings under the method that owns them" pattern rather than a top-level key |
| **DEV** | developer/diagnostic escape hatch; exposing it in a user config would be wrong |

## Summary

Counts are per FLAG and were computed from the tables below, not estimated; they sum to the 65 flags
`grep`ed out of `src/`.

| status | count |
|---|---|
| **RETIRED** (done) | 10 |
| **1:1** (remaining) | 7 |
| **ABSTRACTED** | 8 |
| **ALIAS** | 2 |
| **MODE INTERFACE** | 3 |
| **GAP — user** | 7 |
| **GAP — advanced** | 23 |
| **DEV** | 4 |
| **total** | **64 rows / 65 flags** (`-wtm_dt_norm_rms` and `-wtm_dt_norm_max` share a row) |

**Reachable from a config file today: 26.** That is not simply 1:1 + ABSTRACTED (25), and the two
places it differs are worth stating rather than smoothing over:

- `-wtm_anderson` is classified ABSTRACTED but has no bridge entry, because Anderson is the DEFAULT —
  `solver.method: anderson` reaches it by setting no flag at all.
- `-wtm_dev_allow_aboveground_water_columns` is classified DEV but *is* bridged, under `dev:`. Being a
  developer knob and being config-reachable are independent. (`-wtm_dev_padded_dirichlet` was the other
  example here; retired 2026-09-04.)

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
| `-wtm_picard` | semi-implicit Picard (SPD operator, CG+GAMG) | **RETIRED** (was ABSTRACTED) | `solver.method: picard` |
| `-wtm_newton` | Newton-Krylov on the analytic Jacobian | ABSTRACTED | `solver.method: newton`, which implies `solver.dt_continuation` (the working recipe). The BARE flag stays plain Newton -- three things pin that |
| `-wtm_anderson` | Anderson mixing, matrix-free | ABSTRACTED | `solver.method: anderson` (the default) |
| `-wtm_aa_picard` | Anderson-accelerated GAMG-Picard (nonlinear preconditioning) | GAP — advanced | none — a fourth strategy `solver.method` does not offer |
| ~~`-wtm_handoff`~~ | ~~run Anderson, then hand the best iterate to a finisher~~ | **RETIRED 2026-09-03** | n/a — removed from the code |
| ~~`-wtm_handoff_picard`~~ | ~~make that finisher Picard instead of Newton~~ | **RETIRED 2026-09-03** | n/a — removed from the code |
| ~~`-wtm_handoff_patience`~~ | ~~stalled iterations before handing off~~ | **RETIRED 2026-09-03** | n/a — removed from the code |
| ~~`-wtm_handoff_max_it`~~ | ~~cap on the Anderson phase~~ | **RETIRED 2026-09-03** | n/a — removed from the code |
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
| `-wtm_volume_storage` | backward-Euler storage assembly: exact ΔV (b=0) vs secant S·Δh (b=h^n); the SAME equation | **RETIRED** (was ABSTRACTED) | `dev.storage_form: volume` (now a DEV key, default volume) |
| `-wtm_Tbar` | time-averaged interblock transmissivity | **RETIRED** (was 1:1) | `solver.t_bar` |
| `-wtm_bdf2` | the ORIGINAL BDF2 (head form), pre-`bdf2_on_V` | GAP — advanced | none. It sets `use_bdf2` WITHOUT `use_bdf2_on_V` — head-form BDF2, a distinct scheme; `time_integration: bdf2` maps to `bdf2_on_V` |

## Adaptive dt and the step-size controller

The single largest gap, and the one with a demonstrated cost: a sweep of `-wtm_dtc_easy_iters` returned
byte-identical results at every setting because the flag was parsed only on the continuation path
(fixed in `395915e`) — a false negative that was recorded as a finding.

| flag | what it does | status | YAML today |
|---|---|---|---|
| `-wtm_dt_adaptive` | enable the adaptive controller | **RETIRED** (was 1:1) | `solver.adaptive_dt` |
| `-wtm_dt_tol` | per-step local-error target, in water volume | **RETIRED** (was 1:1) | `solver.water_volume_timestep_error_tol` |
| `-wtm_dtc_dt_max` | cap on dt | **RETIRED** (was 1:1) | `solver.dt_max` |
| `-wtm_dtc_grow` | growth factor on an easy step | GAP — advanced | none |
| `-wtm_dtc_shrink` | shrink factor on a reject | GAP — advanced | none |
| `-wtm_dtc_easy_iters` | iteration count below which dt may grow | GAP — advanced | none |
| `-wtm_dtc_max_retries` | consecutive rejects before giving up | GAP — advanced | none |
| `-wtm_dtc_dt0` | starting dt for the continuation ramp | GAP — advanced | none |
| `-wtm_dt_continuation` | Newton's dt ramp | **RETIRED** (was ABSTRACTED) | implied by `solver.method: newton`; `solver.dt_continuation: false` opts out (warns) |
| `-wtm_dt_norm_rms` / `-wtm_dt_norm_max` | adaptive error norm: RMS (default) or MAX | GAP — advanced | none |
| `-wtm_dt_trace` | report (dt, est, tol, factor, iters, accepted) per step | DEV | none — diagnostic |

**RESOLVED 2026-09-01.** `-wtm_dt_continuation` was the row to look at first: not advanced tuning but a
*requirement* of a strategy the config offers, and its absence made `solver.method: newton` unusable from
YAML alone -- measured on tests/fsm_consistency as `DIVERGED_LINE_SEARCH` after 4 iterations (rc 134),
against rc 0 the moment the flag was added. `solver.method: newton` now means the working recipe,
`-wtm_newton -wtm_dt_continuation`, and is byte-identical to it (tests/route_equality).

The asymmetry is deliberate and is NOT the `dev.active_set` dual-route hazard. The bare `-wtm_newton`
still means PLAIN Newton, because three things depend on that meaning: `tests/newton_solver`'s CONTRACT
arm pins that plain `-wtm_newton` does NOT converge, `benchmark/scheme_bench` measures a "Newton (plain)"
arm, and `EQUILIBRIUM_ROBUSTNESS.md` documents plain Newton as the thing that needs the recipe. The flags
stay the primitive layer; the config key is the abstraction over them -- exactly the relation
`collection.method: legacy` has to the `-wtm_` surface flags. `solver.dt_continuation: false` opts out
and warns, since Newton as a warm finisher is a real mode where continuation is wasted.

## Surface water

**RESOLVED 2026-09-01 (fork issue #7).** The collection selector is the abstraction, and the three flags
that were the interface of its `legacy` mode are RETIRED along with that mode. They were verified
byte-identical to the modes they wrapped before removal (`legacy` + flag == `explicit` / `implicit`,
max|Δ| = 0.000e+00), so nothing was lost. Two channels for one decision is what produced the
`extended_soil` collision; there is now one.

| flag | what it does | status | YAML today |
|---|---|---|---|
| `-wtm_direct_to_runoff` | in-residual exfiltration removal | **RETIRED** (was MODE INTERFACE) | `collection.method` — the `legacy` mode this was the interface of is gone (issue #7) |
| `-wtm_surface_exfiltration_to_runoff` | post-solve clamp | **RETIRED** (was MODE INTERFACE) | `collection.method` — the `legacy` mode this was the interface of is gone (issue #7) |
| `-wtm_surface_sink` | sub-surface band sink | **RETIRED** (was MODE INTERFACE) | `collection.method` — the `legacy` mode this was the interface of is gone (issue #7) |
| `-wtm_extended_soil` | continue the aquifer above the surface | **RETIRED** (was ALIAS) | `collection.method: extended_soil` — retired 2026-09-01; aborts by name |
| `-wtm_active_set` | semismooth exfiltration pin | **RETIRED** (was ABSTRACTED) | `collection.method: active_set` (the default). The second route, `dev.active_set`, was **removed 2026-09-01** — it silently overrode an explicit method |
| `-wtm_dev_active_set` | the older name for the same thing | **RETIRED** (was ALIAS) | `collection.method: active_set` — retired 2026-09-01; aborts by name |
| `-wtm_surface_sink_qmax` | band-sink peak removal rate | **RETIRED** (was 1:1) | `collection.sink.qmax` |
| `-wtm_surface_sink_width` | band width below the surface | **RETIRED** (was 1:1) | `collection.sink.width` |
| `-wtm_fringe_source` | capillary-fringe width source | **RETIRED** (was 1:1) | `collection.sink.fringe_source` |
| `-wtm_fringe_cap` | max ψ_a | **RETIRED** (was 1:1) | `collection.sink.fringe_cap` |
| `-wtm_fringe_ksat_coef` | ψ_a = C·√(n/ksat) | **RETIRED** (was 1:1) | `collection.sink.fringe_ksat_coef` |
| `-wtm_fringe_length` | uniform fringe length | **RETIRED** (was 1:1) | `collection.sink.fringe_length` |
| `-wtm_fsm_delta_source` | carry FSM's Δwtd as a source in the next step | DEV | none — experimental |

**RESOLVED 2026-09-01.** `-wtm_active_set` had been reachable from **two different YAML keys**
(`dev.active_set` and `collection.method: active_set`) — the same two-channel shape as the bug that
prompted this document, only entirely inside the config. It was worse than a duplicate: `dev.active_set`
silently *overrode* an explicit `collection.method`, so a config asking for `explicit` ran `active_set`,
differing on 54 of 256 cells (max 0.127 m) with nothing in the log. `dev.active_set` is removed from the
schema and the bridge; a config still carrying it aborts and names it.

## Evaporation and tapers

The bridge exposes the taper *parameters* but not the *toggles*, which is an odd half-migration: a user
can retune the sigmoid but cannot turn it off.

| flag | what it does | status | YAML today |
|---|---|---|---|
| `-wtm_evap_taper_wtdc` | sigmoid centre | **RETIRED** (was 1:1) | `evaporation.et_sigmoid.wtd_center` |
| `-wtm_evap_taper_s` | sigmoid width | **RETIRED** (was 1:1) | `evaporation.et_sigmoid.logistic_width` |
| `-wtm_extinction_depth` | extinction depth | **RETIRED** (was 1:1) | `evaporation.extinction_depth` |
| `-wtm_evap_taper` | taper 2 **on/off** | **GAP — user** | none — parameters are configurable, the switch is not |
| `-wtm_extinction` | taper 3 **on/off** | **GAP — user** | none — same |

## Transmissivity and boundaries

| flag | what it does | status | YAML today |
|---|---|---|---|
| `-wtm_T_bedrock` | additive background transmissivity | **RETIRED** (was 1:1) | `transmissivity.additive_background_transmissivity` |
| `-wtm_land_boundary` | land boundary condition | **RETIRED** (was ABSTRACTED) | `boundaries.land` (value translated) |
| `-wtm_ksat_surface_smoothing_width` | round the ksat kink at the surface | GAP — user | none — a modelling choice, not a developer knob |
| `-wtm_ksat_soilbottom_smoothing_width` | round the ksat kink at −1.5 m | GAP — user | none — same |
| `-wtm_storativity_surface_smoothing_width` | round the storativity kink at the surface | GAP — user | none — same |

## Convergence criteria

| flag | what it does | status | YAML today |
|---|---|---|---|
| `-wtm_eq_tol` | equilibrium stop tolerance | **RETIRED** (was 1:1) | `run.equilibrium_stop.tol` |
| `-wtm_eq_metric` | which metric judges equilibrium | **RETIRED** (was 1:1) | `run.equilibrium_stop.metric` |
| `-wtm_eq_frac` | fraction-of-cells threshold | **RETIRED** (was 1:1) | `run.equilibrium_stop.frac` |
| `-wtm_snes_volume_conv` | judge the SNES step in water, not head | GAP — advanced | none |
| `-wtm_snes_volume_conv_govern` | make that judgement authoritative | GAP — advanced | none |
| `-wtm_snes_vol_tol` | its relative tolerance | GAP — advanced | none |

## Developer

| flag | what it does | status | YAML today |
|---|---|---|---|
| `-wtm_dev_allow_aboveground_water_columns` | disable the surface clamp entirely | DEV | `dev.allow_aboveground_water_columns` (already exposed) |
| ~~`-wtm_dev_padded_dirichlet`~~ | verification tool for the ghost scheme | **RETIRED 2026-09-04** | n/a — removed, with its schema key |

---

## What this suggests, in order

**A correction to this document's own first draft, kept visible because it is the useful part.** The
draft said six flags were "pure hazard, retire them, zero capability lost". Checking each against the
code before deleting anything showed that was wrong for five of the six:

- `-wtm_direct_to_runoff`, `-wtm_surface_exfiltration_to_runoff`, `-wtm_surface_sink` are the
  **interface of `collection.method: legacy`** — the selector block is guarded `if (rc != "legacy")`,
  and its own warning tells the user to *"set it to 'legacy' to hand control back to the -wtm_ surface
  flags"*. Deleting them deletes the mode.
- `-wtm_bdf2` is a **distinct scheme**, not a duplicate channel: it sets `use_bdf2` without
  `use_bdf2_on_V`, i.e. head-form BDF2, while `time_integration: bdf2` maps to `bdf2_on_V`.
- All of them, plus `-wtm_extended_soil`, **already warn** when a configured method supersedes them. The
  silent collision that motivated the list was specific to `extended_soil`, and it is fixed.

Only `-wtm_dev_active_set` was what the draft claimed. The lesson is cheap to state and was not cheap to
learn: *a flag being expressible as a YAML mode does not make it redundant* — it may be how that mode is
selected, or the only route to a behaviour the config cannot reach.

1. **Finish the 1:1 retirement (7 left).** `dtc_dt_max` next — zero callers, and its complication (an
   `auto` sentinel and unit parsing) lives in the parser rather than spread across tests. Then
   `dt_tol`, `dt_adaptive`, `eq_metric`, `Tbar`, `active_set`, and `eq_tol` last (61 call sites).
2. **DONE 2026-09-01 — `-wtm_active_set`'s two YAML routes resolved to one.** `dev.active_set` is
   removed from the schema and the bridge. It did not merely duplicate `collection.method`, it silently
   OVERRODE it: 54/256 cells and 0.127 m max on tests/fsm_consistency, with no log line. Pinned by the
   `RETIRED` arm of `tests/config_schema`.
3. **Close the four user-facing gaps that block documented workflows**: `-wtm_dt_continuation` (Newton
   is unusable from YAML without it), the two taper toggles, and `-wtm_stiff` as a preset.
4. **Group the advanced clusters under the method that owns them** — the step-size controller under
   `solver.adaptive_dt`, the Anderson restart/handoff machinery under `solver.method: anderson`.
5. **Decide `-wtm_dev_active_set` and `-wtm_bdf2` on their own merits.** The first is a deprecated alias
   with no callers; the second is a superseded *scheme*, so retiring it is a scientific call about
   whether head-form BDF2 is worth keeping, not config hygiene.

The unknown-flag check (`750ffb1`) is what makes any of this safe: a retired flag now aborts and names
itself instead of being silently ignored, so a missed call site fails loudly on the first run.


The three surface-smoothing widths are the judgement call: they are genuinely modelling choices rather
than developer knobs, so they arguably belong in `transmissivity:` — but they are also default-off and
were shown not to fix the free-boundary order loss (`BDF2_RECHARGE_ORDER.md` §15), so exposing them
prominently may advertise a dead end.
