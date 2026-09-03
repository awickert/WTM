# The remaining `-wtm_*` flags: an inventory, and the question it raises

**What this is.** The 29 `-wtm_*` flags the model still reads, grouped by what they control, and
classified by whether their code actually *executes* anywhere. It exists because the next step in the
config migration is a schema DESIGN, and a design needs the whole set in front of it rather than one
parameter at a time.

**The question it raises is not only where each parameter goes.** These flags are the accumulated
creations of development -- every mechanism tried, kept, or half-kept along the way. Giving a config key
to a tuning constant for a mechanism nothing runs puts it in the user-facing schema, in the reference
config, and in the maintenance burden, permanently. So for each group the first question is whether it
belongs in the finished model at all; only then does placement matter.

**The criterion agreed for what stays a flag** (Andy, 2026-09-01): provenance decides it. A config file is
an archivable artifact and a command line is not, so anything that CHANGES THE ANSWER belongs in the
config -- including developer switches. What may stay a flag is what changes only what is PRINTED.
Separately, PETSc's own dials (`-snes_*`, `-ksp_*`) stay on the CLI because they were never WTM's to own;
the prototype says so explicitly.


## How coverage is classified

An earlier version of this file counted "callers" and reported that 20 flags had none. That single label
was covering three different situations that need opposite treatment, so it has been replaced.

| term | meaning |
|---|---|
| **varied** | a runnable test or benchmark sets it away from its default -- the knob is genuinely exercised |
| **default-only** | its mechanism executes on ordinary runs, but nothing ever changes the value. The *code* is covered; the *knob* is not |
| **dormant** | a default-off switch that nothing runnable enables, by flag or by config key. **Zero execution coverage** |
| **dormant dial** | a constant belonging to a dormant switch -- a knob on a switch that is never thrown |
| **archive-only** | its only callers are orphaned benchmark scripts that emit the legacy flat `.cfg` and cannot run |

**"Runnable" excludes the orphaned benchmark scripts.** A benchmark script is counted as runnable only if
it calls `tests/emit_config.sh`; the rest still write the pre-migration flat `.cfg` and abort. That
exclusion is why some counts fell against the earlier version: `-wtm_bdf2`'s nine callers are all
orphaned, and so is the sole caller of `storativity_surface_smoothing_width` and of `dt_norm_rms`.

**Why the distinction matters.** For a *default-only* flag, deletion is not on the table -- the mechanism
is load-bearing and runs constantly; the only question is expose-or-hard-code, and a hard-coded constant
can become a key later without breaking anyone. For a *dormant* flag the question is whether the
MECHANISM stays, which is the expensive one: an untested code path in a model being handed off is a
claim the model can do something, with nothing demonstrating that it does.

**What this classification does NOT establish.** It is coverage *within this repository*. Command lines
run by hand, MSI `sbatch` history, and anything held by other users do not appear in it. A flag reached
for personally would classify as dormant here and still be alive.

Reproduce with:

```sh
grep -rhoE '"-wtm_[a-z0-9_]+"' src/ | tr -d '"' | sort -u          # the 34
grep -rl --include='*.sh' --include='*.py' --include='*.yaml' --include='*.cfg' -- "$flag" tests/
grep -q emit_config "$script"                                       # runnable vs orphan
```

Note the `--include` filters must not follow a `--` end-of-options marker; doing so silently disables
them and searches `.md` and `.log` files too, which inflated an earlier draft of these counts.


## Adaptive step-size controller  (7)

*`solver.adaptive_dt` is a bare boolean today; these are its dials. `dtc_dt_max` is ALREADY a config key
(`solver.dt_max`, accepting `auto`) with no flag -- so the precedent for this block exists in the code.*

| flag | coverage | note |
|---|---|---|
| `-wtm_dtc_grow` | **varied** | `tests/estimator_order` sets it to 1 to freeze dt |
| `-wtm_dtc_shrink` | **varied** | same -- the `p = 1.56 1.08 1.00` measurement depends on both |
| `-wtm_dtc_dt0` | default-only | continuation-only; continuation runs in 8 tests |
| `-wtm_dtc_easy_iters` | default-only | the controller's growth gate, and its highest-leverage knob: 0 / 8 / 100000 spans 57 steps to 229506-and-still-running. An earlier sweep recorded it "inert" because the flag was not parsed on the adaptive path at all -- a negative result manufactured by the plumbing |
| `-wtm_dtc_max_retries` | default-only | the abort on both loops -- adaptive (`WTM.cpp:617`) as well as continuation. USED by both but, until 57eed4e, PARSED only on the continuation path, so asking for it on an adaptive run aborted the run |
| `-wtm_dt_norm_rms` | default-only | it *is* the default; redundant with `dt_norm_max` as a pair |
| `-wtm_dt_norm_max` | **dormant** | the MAX-norm path never runs |

## Anderson restart  (5)

| flag | coverage | note |
|---|---|---|
| `-wtm_adaptive_restart` | **varied** | `tests/adaptive_restart` enables it |
| `-wtm_ar_rho` | default-only | the rho threshold that triggers a restart |
| `-wtm_ar_patience` | default-only | consecutive degrading iterations before restarting |
| `-wtm_ar_max_it` | default-only | iteration cap |
| `-wtm_ar_max_restarts` | default-only | restart cap |

## Anderson -> finisher handoff  (4) -- RETIRED 2026-09-03

*was: run Anderson, hand the best iterate to a Newton or Picard finisher (nonlinear preconditioning, #87)*

**Removed from the code** (Andy's call, on the evidence below). The four flags -- `-wtm_handoff`,
`-wtm_handoff_picard`, `-wtm_handoff_patience`, `-wtm_handoff_max_it` -- now abort as unconsumed, so a
script still passing one is told rather than silently getting a plain Anderson run. The mechanism lives
in git history; see the commit for what was deleted and where.

What the walk-through established:

- **Zero test coverage, ever**, and no design note in `benchmark/`.
- **One archived exploratory run** (`esquibel/.../axis2b/handoff`, transient, FSM off, legacy `.cfg`).
- **Still functional** when tested -- it ran clean and announced itself.
- **But inert on anything we can test.** Baseline against `-wtm_handoff` on the same fixture gave
  IDENTICAL results: 3 cycles, 22 solves both. The finisher only engages after Anderson stalls for
  `patience` iterations, and nothing in the suite stalls Anderson. So the mechanism could not be
  evaluated on any fixture we have -- it is a rescue path for a regime with no fixture.
- Found on the way out: it **leaked** `snes_finish` and `handoff_best_x`; neither was ever destroyed.

The judgement, recorded so it is not re-litigated: the regime it exists for (stiff cold starts where
Anderson stalls) is now served by the adaptive TR-BDF2 controller, which is the DEFAULT and IS tested.
The caveat, equally recorded: that supersession is an inference from the transient benchmark, NOT a
head-to-head in the stall regime. If such a case is ever built and the adaptive controller fails it,
this mechanism is in git history and can come back.

## Volume-based SNES convergence  (3) -- KEPT, CONFIG-OWNED, 2 of 3 COVERED

*judge the step in water rather than head*

| flag | now reached by | coverage |
|---|---|---|
| `-wtm_snes_volume_conv` | `output.trace: [water_step]` | **varied** -- `solver_consistency` arm 4 |
| `-wtm_snes_volume_conv_govern` | `solver.convergence.metric: water` | **varied** -- `solver_consistency` arm 5 |
| `-wtm_snes_vol_tol` | `solver.convergence.water_volume_tol` | default-only; a live dial, measured below |

All three are config-owned as of `aa49674`; the flags REMAIN as the internal transport and still
override the config when passed on the CLI, which is the same arrangement as the seventeen settings
migrated before them. The flag census is unchanged at 30 -- migrating a setting does not remove its
flag, and an earlier claim here that this would take 30 -> 27 was wrong.

**What the migration actually retired was two COUPLINGS, not flags.** `CreateSNES.cpp` forced
`conv = vc || govern`, and the print was then gated on `conv && !govern` -- so governing silently
SUPPRESSED the trace, and printing and governing could never be had together. They are now driven by
different config surfaces and all four states are reachable. The printing field is renamed
`vol_step_trace`, since it no longer has anything to do with the convergence criterion.

Route equality before the tests moved across: wtd 0.000000e+00 m, 908 solves both, 7887 trace lines
both -- that last figure with governing ON, where the old coupling would have given zero.

**Decision 2026-09-03: keep all three, and convert the value from historical to standing.**

Asked what these currently brought, the answer was *nothing*. `403aeac` verified something once,
on the day it was written -- "recon == snorm exactly, water/head L2 ratio = 0.250 = phi" -- and that
result had sat in a commit message ever since, with zero callers. A flag nobody sets provides no
ongoing verification. So rather than retire the mechanism, the verification became a test.

**Why the first check earns its place, since it is the least obvious.** PETSc hands a convergence
test `snorm` = ||dx||, the step just taken, and `-snes_stol` converges on `snorm < stol*xnorm`. On the
matrix-free Anderson path that step CANNOT be read back: `SNESGetSolutionUpdate` returns Anderson's raw
PRE-MIXING update, measured ~10x the accepted step. `VolumeStepConverged` therefore keeps its own
previous accepted iterate and differences against it, and `recon == snorm` is the proof that the
reconstruction measures the same step PETSc does. Nothing else in the suite looks at this, so a change
to the Anderson update path would silently invalidate the water-step machinery `eq_tol` and `dt_tol`
are both built on.

Shown to bite: scaling the reconstruction by 1.01 fails that arm at 1.090e-02 against its 1e-3 bound,
while the other two still pass.

**`-wtm_snes_vol_tol` is a live dial**, measured on the solver_consistency fixture, water-governed:

| tol | cycles | solves | from head-governed |
|---|---|---|---|
| 1e-10 | 14 | 899 | 3.865e-07 m |
| 1e-8 (default) | 14 | 908 | 4.995e-07 m |
| 1e-5 | 22 | **2719** | 3.786e-04 m |

The direction is backwards from intuition and worth keeping: a LOOSER per-solve tolerance costs **3x
the solves**, because each solve stops under-converged and the outer equilibrium loop needs more cycles
to reach `eq_tol`. Under-converging the inner solve does not save work, it moves it. Gating that would
add a 2719-solve arm to every suite run for a dial nobody sets, so the measurement is recorded here
instead.

**Still open, and not discharged by any of the above:** `403aeac` parked the governing switch "pending
varying-S validation + precision-matched benchmarking before any default flip". The new arm makes it
COVERED, not VALIDATED -- it would catch the criterion breaking; it does not establish that judging
convergence in water is the right production default.

## Alternative schemes  (5)

*each a distinct numerical strategy, not a tuning knob*

| flag | coverage | note |
|---|---|---|
| `-wtm_bdf2` | **archive-only** | the ORIGINAL head-form BDF2. 9 callers, ALL orphaned. `solver.time_integration: bdf2` maps to BDF2-on-V, so this has NO config route |
| `-wtm_kirchhoff` | **dormant** | Kirchhoff variable change. NOT isolated: it gates the active-set pin in `FormJacobianLocal`, where the SNES variable is the discharge potential |
| ~~`-wtm_aa_picard`~~ | **RETIRED 2026-09-03** | removed; the negative result it produced is kept in `benchmark/AA_PICARD.md` |
| `-wtm_predict_guess` | **dormant** | predictor-seeded initial guess |
| `-wtm_relax` | **dormant** | sub-step under-relaxation; default 1.0 = off |

## Numerical smoothing  (3)

*round a kink so a derivative check is meaningful -- except storativity's 0.01, which is real physics*

| flag | coverage | note |
|---|---|---|
| `-wtm_ksat_surface_smoothing_width` | **varied** | 2 tests; default 0, exists so a Jacobian FD check has a smooth tangent |
| `-wtm_ksat_soilbottom_smoothing_width` | **varied** | 2 tests; the kink at -1.5 m |
| `-wtm_storativity_surface_smoothing_width` | default-only | default 0.01 and always on -- sub-grid roughness. Its only caller is orphaned |

## Evaporation taper toggles  (2)

*tapers always on was the intention; every off-switch use is a TEST CONTROL -- the arm that shows what
happens without the taper*

| flag | coverage | note |
|---|---|---|
| `-wtm_evap_taper` | **varied** | 6 tests: `boundary_analytic`'s analytic baseline, `flicker_evap`'s limit-cycle demo, taper Study C's runaway |
| `-wtm_extinction` | **varied** | 3 tests |

## Preset  (1)

| flag | coverage | note |
|---|---|---|
| `-wtm_stiff` | **varied** | `benchmark/scheme_bench/run.sh` is runnable. Convenience bundle: newton + continuation + eq_tol |

## Developer  (2)

| flag | coverage | note |
|---|---|---|
| `-wtm_dev_allow_aboveground_water_columns` | **varied** | 5 tests; disables the surface clamp entirely. A `dev.` config key exists too |
| `-wtm_dev_padded_dirichlet` | **dormant** | verification tool for the ghost scheme. Its schema key `dev.padded_dirichlet` already exists and is unused by either route |

## FSM coupling experiment  (1)

| flag | coverage | note |
|---|---|---|
| `-wtm_fsm_delta_source` | **varied** | `tests/budget_closure`. Superseded for its original purpose by `active_set` |

## Diagnostic output  (1)

| flag | coverage | note |
|---|---|---|
| `-wtm_dt_trace` | **varied** | `tests/estimator_order`. Changes only what is PRINTED -- the one clear candidate to stay a flag |


## The tally

| classification | count | what the decision is |
|---|---|---|
| varied | **11** | placement only -- these are alive |
| default-only | **9** | expose the knob, or hard-code it? Low stakes; the code is covered |
| dormant + dormant dial | **13** | **keep the mechanism or delete it** -- zero execution coverage |
| archive-only | **1** | `-wtm_bdf2`: a modelling judgement about the head form |

**14 code paths never execute** (13 dormant + `bdf2`), and **9 more run but only ever at their default.**
Those two groups need opposite treatment, which the earlier "no caller" label hid.


## What I would want decided before designing any schema

1. **Which dormant mechanisms survive.** *handoff* (4 flags) and *volume-based SNES convergence* (3) are
   dormant end to end -- neither mechanism has ever run in a test. Of *alternative schemes*, four of five
   are dormant and the fifth is archive-only. If a mechanism is finished experimenting, deleting it is
   cheaper than housing it. `-wtm_kirchhoff` is the one that cannot simply be lifted out: it gates the
   active-set pin in the analytic Jacobian.

2. **`-wtm_bdf2` specifically.** It is the ORIGINAL head-form BDF2, distinct from `bdf2_on_V`, not a
   duplicate, and it has no config route. Superseded in practice, but that is a modelling judgement.

3. **`storativity_surface_smoothing_width`.** Default 0.01 m and always on -- the source calls it sub-grid
   roughness, i.e. real physics -- unlike the two `ksat_*` widths, which default to 0 and exist so a
   Jacobian FD check has a smooth tangent. It probably belongs with the physics, not the diagnostics,
   despite sitting beside them in the source.

4. **`run.type`-seeded preset defaults.** `config_flags_prototype.yaml` specifies them
   ("equilibrium -> adaptive-TR-BDF2 + T-bar; transient -> adaptive-TR-BDF2") and they are NOT
   implemented; `config.yaml` ships explicit values instead. `-wtm_stiff` is a symptom of that gap rather
   than an item of its own, and the gap is larger than a flag migration.

5. **The `auto` idiom.** `solver.dt_max` already accepts the sentinel `auto` meaning "derive from
   `deltat`". If that is the house style, `dtc_dt0` should use it too (`auto` = `deltat/200`) and the
   `_set` boolean pattern in `Parameters` should stop spreading. Whatever this block does, five more
   blocks will copy.

## A shape, once those are settled

The groups above are mostly "one switch plus its constants", which nests the way the prototype already
treats `run.equilibrium_stop` and `evaporation.et_sigmoid`: `solver.adaptive_dt` becomes a block rather
than a bare boolean, `solver.anderson.restart` likewise. That is roughly six blocks instead of 33
top-level keys. Block style throughout, per the prototype's own header. The `dt_norm_rms` /
`dt_norm_max` pair collapses to one enum key -- `norm: rms|max` -- which also removes the current
undefined behaviour when both are set.


---

# Decisions taken

Recorded as they are made, so the walk-through does not have to be re-run from memory. Each entry names
who decided and on what grounds.

## Group 1 -- adaptive step-size controller (in progress)

**Expose all seven as config keys**, rather than hard-coding any. The test applied to a *default-only*
constant: expose it if (a) a test needs to vary it, (b) a user hitting a failure mode needs it to get
unstuck, or (c) varying it moves the answer materially. `dtc_grow` and `dtc_shrink` pass (a) --
`tests/estimator_order` sets both to 1 to freeze dt, and loses that ability the moment the flags go.
`dtc_easy_iters` passes (c). `dtc_max_retries` and `dtc_dt0` pass (b) -- the first IS the abort, on the
adaptive loop as well as the continuation one, and the abort message already tells the user to raise the
second. `dtc_dt0` is continuation-only, so its fate follows Newton's. NOTE this is not a precedent that default-only
implies expose: the four `ar_*` constants in group 2 are expected to fail the same test.

**`dt_norm_rms` / `dt_norm_max` collapse to one enum key, `norm: rms|max`.** Two booleans whose
both-set case is undefined; this is a defect repair, not an exposure decision. 7 flags -> 6 keys.

**`dtc_easy_iters` is renamed `grow_if_niter_leq` in the config** (Andy, 2026-09-02). The old name is
opaque -- it reads as a count of easy iterations rather than the threshold defining "easy". Rejected
along the way, with reasons worth keeping so they are not re-proposed:

- `grow_below_iters` / `grow_if_niter_below` -- **off by one**. Both code paths are inclusive
  (`WTM.cpp:672` grows at `its <= n`; `transient_groundwater.cpp:1969` clamps only at `its > n`), so a
  solve taking exactly 8 iterations DOES grow. A name that says otherwise is worse in a config key than
  in code, because the key is the documentation.
- `max_niter_to_grow` -- parses as a noun phrase, i.e. a cap on the iteration count, which is the
  opposite of what it gates.
- `easy_niter` -- unambiguous and matches the source's own "easy step / hard step" vocabulary, but does
  not stand alone: read in an error message it does not say it controls growth.

`niter` is the conventional spelling for the audience, and `leq` states the inclusive boundary exactly.
Andy accepted its jargon cost on the grounds that each key should stand by itself.

**What `grow_if_niter_leq` actually is**, since the old name hid it: a *solvability* gate on dt growth,
distinct from the *accuracy* gate. `niter` is the nonlinear iteration count of the step just taken --
Newton's on the Newton path, Anderson's on the Anderson path, so the default 8 was tuned on a different
iteration than an Anderson user will be setting it for. On Newton's `dt_continuation` ramp it is the
ENTIRE control law (there is no error estimate on that path). On the adaptive path it is a veto layered
over the PI error controller, which may clamp growth but never force it. It exists because a step can be
perfectly accurate and still have taken 40 iterations near the free-boundary ceiling, where a larger dt
overshoots into a singular Jacobian; iteration count is the proxy for that cliff, and the error estimate
knows nothing about it.

### Still open in group 1

- the block's NAME (`solver.time_step` proposed, on the grounds that these dials serve
  `dt_continuation` as well as `adaptive_dt`, so an `adaptive_`-prefixed name would misdescribe them)
- whether `auto` becomes the house sentinel for "derive this default", following the existing
  `solver.dt_max: auto`, and whether the `_set`-boolean pattern in `Parameters` stops spreading
- whether `solver.dt_max` (1 caller) and `solver.water_volume_timestep_error_tol` (2 callers) move
  into the block
