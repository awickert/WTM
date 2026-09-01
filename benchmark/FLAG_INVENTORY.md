# The remaining `-wtm_*` flags: an inventory, and the question it raises

**What this is.** The 34 `-wtm_*` flags the model still reads, grouped by what they control, with how
many files actually call each one. It exists because the next step in the config migration is a schema
DESIGN, and a design needs the whole set in front of it rather than one parameter at a time.

**The question it raises is not only where each parameter goes.** These flags are the accumulated
creations of development -- every mechanism tried, kept, or half-kept along the way. **20 of the 34 have
no caller anywhere in `tests/` or `benchmark/`.** Giving a config key to a tuning constant for a mechanism
nothing runs puts it in the user-facing schema, in the reference config, and in the maintenance burden,
permanently. So for each group the first question is whether it belongs in the finished model at all;
only then does placement matter.

**Caller counts are files, not occurrences**, and `bench` counts include the ~25 orphaned benchmark
scripts that cannot currently run (they emit the legacy flat `.cfg`), so a non-zero `bench` is weaker
evidence of life than a non-zero `tests`.

**The criterion agreed for what stays a flag** (Andy, 2026-09-01): provenance decides it. A config file is
an archivable artifact and a command line is not, so anything that CHANGES THE ANSWER belongs in the
config -- including developer switches. What may stay a flag is what changes only what is PRINTED.
Separately, PETSc's own dials (`-snes_*`, `-ksp_*`) stay on the CLI because they were never WTM's to own;
the prototype says so explicitly.


## Adaptive step-size controller  (7 flags, 4 with no callers)

*solver.adaptive_dt is a bare boolean today; these are its dials*

| flag | tests | bench | what it does |
|---|---|---|---|
| `-wtm_dtc_dt0` | — | — | starting dt for the continuation ramp |
| `-wtm_dtc_grow` | 1 | — | growth factor on an easy step |
| `-wtm_dtc_shrink` | 1 | — | shrink factor on a reject |
| `-wtm_dtc_easy_iters` | — | — | iteration count below which dt may grow |
| `-wtm_dtc_max_retries` | — | — | consecutive rejects before giving up |
| `-wtm_dt_norm_rms` | — | 1 | adaptive error norm: RMS (default) or MAX |
| `-wtm_dt_norm_max` | — | — | adaptive error norm: RMS (default) or MAX |

## Anderson restart  (5 flags, 4 with no callers)

*one switch plus four tuning constants*

| flag | tests | bench | what it does |
|---|---|---|---|
| `-wtm_adaptive_restart` | 2 | 1 | restart Anderson's history when the convergence RATE degrades |
| `-wtm_ar_rho` | — | — | the rho threshold that triggers it |
| `-wtm_ar_patience` | — | — | consecutive degrading iterations before restarting |
| `-wtm_ar_max_it` | — | — | iteration cap |
| `-wtm_ar_max_restarts` | — | — | restart cap |

## Anderson -> finisher handoff  (4 flags, 4 with no callers)

*run Anderson, hand the best iterate to Newton or Picard*

| flag | tests | bench | what it does |
|---|---|---|---|
| `-wtm_handoff` | — | — | run Anderson, then hand the best iterate to a finisher |
| `-wtm_handoff_picard` | — | — | make that finisher Picard instead of Newton |
| `-wtm_handoff_patience` | — | — | stalled iterations before handing off |
| `-wtm_handoff_max_it` | — | — | cap on the Anderson phase |

## Volume-based SNES convergence  (3 flags, 3 with no callers)

*judge the step in water rather than head*

| flag | tests | bench | what it does |
|---|---|---|---|
| `-wtm_snes_volume_conv` | — | — | judge the SNES step in water, not head |
| `-wtm_snes_volume_conv_govern` | — | — | make that judgement authoritative |
| `-wtm_snes_vol_tol` | — | — | its relative tolerance |

## Alternative schemes  (5 flags, 4 with no callers)

*each a distinct numerical strategy, not a tuning knob*

| flag | tests | bench | what it does |
|---|---|---|---|
| `-wtm_bdf2` | — | 9 | the ORIGINAL BDF2 (head form), pre-`bdf2_on_V` |
| `-wtm_kirchhoff` | — | — | Kirchhoff variable change |
| `-wtm_aa_picard` | — | — | Anderson-accelerated GAMG-Picard (nonlinear preconditioning) |
| `-wtm_predict_guess` | — | — | predictor-seeded initial guess |
| `-wtm_relax` | — | — | sub-step under-relaxation (1 = off) |

## Numerical smoothing  (3 flags, 0 with no callers)

*round a kink so a derivative check is meaningful; storativity's 0.01 is real physics*

| flag | tests | bench | what it does |
|---|---|---|---|
| `-wtm_ksat_surface_smoothing_width` | 2 | 1 | round the ksat kink at the surface |
| `-wtm_ksat_soilbottom_smoothing_width` | 2 | — | round the ksat kink at −1.5 m |
| `-wtm_storativity_surface_smoothing_width` | — | 1 | round the storativity kink at the surface |

## Evaporation taper toggles  (2 flags, 0 with no callers)

*every use is a TEST CONTROL: the arm that shows what happens without the taper*

| flag | tests | bench | what it does |
|---|---|---|---|
| `-wtm_evap_taper` | 6 | 1 | taper 2 **on/off** |
| `-wtm_extinction` | 3 | — | taper 3 **on/off** |

## Preset  (1 flags, 0 with no callers)

*the design's answer is run.type-seeded defaults, which is unbuilt*

| flag | tests | bench | what it does |
|---|---|---|---|
| `-wtm_stiff` | — | 2 | convenience bundle: newton + continuation + eq_tol |

## Developer  (2 flags, 1 with no callers)

*already have dev.* keys, or want them*

| flag | tests | bench | what it does |
|---|---|---|---|
| `-wtm_dev_allow_aboveground_water_columns` | 5 | — | disable the surface clamp entirely |
| `-wtm_dev_padded_dirichlet` | — | — | verification tool for the ghost scheme |

## FSM coupling experiment  (1 flags, 0 with no callers)

*superseded for its original purpose by active_set*

| flag | tests | bench | what it does |
|---|---|---|---|
| `-wtm_fsm_delta_source` | 1 | 3 | carry FSM's Δwtd as a source in the next step |

## Diagnostic output  (1 flags, 0 with no callers)

*changes only what is PRINTED, never the answer*

| flag | tests | bench | what it does |
|---|---|---|---|
| `-wtm_dt_trace` | 1 | — | report (dt, est, tol, factor, iters, accepted) per step |

---

## What I would want decided before designing any schema

1. **Which groups survive.** *Anderson→finisher handoff* (4 flags) and *volume-based SNES convergence*
   (3) have no callers at all. *Alternative schemes* holds four distinct numerical strategies, of which
   only `-wtm_bdf2` has callers and all nine are in scripts that cannot run. If a mechanism is finished
   experimenting, deleting it is cheaper than housing it.

2. **`-wtm_bdf2` specifically.** It is the ORIGINAL head-form BDF2, distinct from `bdf2_on_V`, not a
   duplicate. Superseded in practice, but that is a modelling judgement, not a config one.

3. **`storativity_surface_smoothing_width`.** Default 0.01 m and always on -- the source calls it sub-grid
   roughness, i.e. real physics -- unlike the two `ksat_*` widths, which default to 0 and exist so a
   Jacobian FD check has a smooth tangent. It probably belongs with the physics, not with the diagnostics,
   despite sitting beside them in the source.

4. **`run.type`-seeded preset defaults.** `config_flags_prototype.yaml` specifies them
   ("equilibrium -> adaptive-TR-BDF2 + T-bar; transient -> adaptive-TR-BDF2") and they are NOT implemented;
   `config.yaml` ships explicit values instead. `-wtm_stiff` is a symptom of that gap rather than an item
   of its own, and the gap is larger than a flag migration.

## A shape, once those are settled

The groups above are mostly "one switch plus its constants", which nests the way the prototype already
treats `run.equilibrium_stop` and `evaporation.et_sigmoid`: `solver.adaptive_dt` becomes a block rather
than a bare boolean, `solver.anderson.restart` likewise. That is roughly six blocks instead of 33
top-level keys. Block style throughout, per the prototype's own header.
