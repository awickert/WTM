# Three shapes for the finished config

The question underneath the flag walk-through: **should `solver:` exist for an end user at all?**
Going key by key through today's `solver:` block, every one of the eight is mechanism rather than a
modelling decision -- and one of them, `storage`, is a choice the default configuration silently
overrides (`parameters.hpp:70`: "a STARTING value, not the last word -- the active-set enforcement
auto-enables volume storage on top of it").

These are three ways to resolve that. They differ in ONE axis -- how much mechanism the user sees --
and sketch C differs in a second, which is how much mechanism still exists to be seen.

Shared background: `benchmark/FLAG_INVENTORY.md` (the 34 flags and their execution coverage) and
`config_flags_prototype.yaml` (the design record: block style throughout, PETSc's own `-snes_*` /
`-ksp_*` dials stay on the CLI, `run.type`-seeded defaults).


## Sketch A -- expose everything, nested by the solver that reads it

Every flag becomes a config key. Scope is expressed structurally: a key sits under the solver that
consumes it, so `newton:` settings are visibly Newton's.

```yaml
solver:
  method: anderson             # anderson | picard | newton
  tolerance: 1e-6
  max_iterations: auto
  time_integration: tr-bdf2    # backward-euler | bdf2 | tr-bdf2 | bdf2-head
  t_bar: false
  storage: volume              # volume | secant

  time_step:                # serves BOTH adaptive_dt and newton.dt_continuation
    adaptive: true
    error_tol: 1.0e-3
    dt_max: auto
    grow: 1.5
    shrink: 0.25
    grow_if_niter_leq: 8
    max_retries: 15
    norm: rms                  # rms | max

  anderson:
    restart:
      enabled: false
      rho: 0.9
      patience: 3
      max_it: 200
      max_restarts: 10
    handoff:                   # hand the best iterate to a finisher
      enabled: false
      finisher: newton         # newton | picard
      patience: 10
      max_it: 100

  picard:
    anderson_accelerated: false
    relax: 1.0

  newton:
    dt_continuation: true
    dt0: auto
    kirchhoff: false
    predict_guess: false

  convergence:
    volume_based: false
    volume_governs: false
    volume_tol: 1e-3

  smoothing:
    ksat_surface: 0
    ksat_soilbottom: 0
    storativity_surface: 0.01

dev:
  allow_aboveground_water_columns: false
  padded_dirichlet: false

diagnostics:
  dt_trace: false
```

**What it gets right.** Total provenance -- an archived config records every setting that touched the
answer. One home per setting, and the nesting states scope without documentation, which is stronger
than today's `# only meaningful with method: newton` comment. Nothing needs to be decided about
mechanisms before the schema can be written.

**What it costs.** It publishes 13 dormant mechanisms as though they were features. A user handed this
model sees a menu in which `handoff:`, `convergence:` and `picard.relax` look exactly as real as
`method:` -- and none of them has ever executed in a test. It is also the hardest option to walk back:
once a key ships, removing it is a breaking change, so this shape converts "we never got round to
deleting it" into "we support it".


## Sketch B -- hide the mechanism, reorganise by what the setting is about

The user surface carries physics and accuracy targets only. Mechanism stays in the schema -- accepted,
archivable, reachable by tests -- but absent from the reference config.

`config.yaml`, in full:

```yaml
run:
  type: equilibrium
  initial_water_table: saturated
  equilibrium_stop:
    tol: 0.001
    metric: frac
    frac: 0.001

time:
  deltat: 1yr
  total: 1000yr
  report_interval: 100
  save_every_n_reports: 1
  adaptive: true               # error-controlled step size
  error_tol: 1.0e-3            # accuracy target per step

io:      {source: surfdata/, region: Australia_, time_start: "021000", time_end: "021000"}
output:  {directory: results/, outfile_prefix: wtd_, verbosity: normal, if_exists: increment}

boundaries:
  land: neumann_toposlope

transmissivity:
  fdepth:
    a: 200
    b: 150
    fmin: 2
  additive_background_transmissivity: 0

evaporation:
  et_sigmoid:
    wtd_center: 0.05
    logistic_width: 0.1
  extinction_depth: 8

surface_water:
  mode: routed
  infiltration_during_flow: false
  collection:
    method: active_set

parallel:
  threads_per_rank: 1
```

**No `solver:` block.** `run.type` seeds method + time_integration + t_bar + storage, per the
prototype's unbuilt design. The schema still accepts everything in sketch A; it is simply not shown.

**The move that makes it work:** adaptive stepping goes to `time:`, where `deltat` already lives.
`time.adaptive` decides whether `deltat` is a fixed step or a starting value -- that is time
configuration, not solver configuration, and having it under `solver:` is why "should solver disappear?"
appeared to need an exception.

**What it gets right.** The handoff config is ~20 keys and every one is a modelling decision. Exposing
the *accuracy target* while hiding the *mechanism* is how a mature integrator behaves: you give an ODE
solver rtol/atol, not a Butcher tableau.

**What it costs.** It requires building the `run.type` seeding, which does not exist -- `config.yaml`
ships explicit values precisely because nothing seeds them. This is decision #4 in the inventory and it
is larger than any flag migration. It also creates a folklore risk: a user who genuinely needs a hidden
key has to be told it exists, and undocumented-but-accepted keys are how "you have to set
solver.storage" becomes tribal knowledge.


## Sketch C -- recommended: three tiers, and delete what is dormant

C is B's organisation plus one further claim: **the right answer to most of the 34 is neither "expose"
nor "hide" but "delete"**, and several of the rest need no key at all because an existing parameter can
carry the meaning.

### The organising rule

Nest by WHAT THE SETTING IS ABOUT. Decide visibility by WHO NEEDS IT. Three tiers, and the tier is a
property of the key, not of a document:

| tier | meaning | where it appears | runtime |
|---|---|---|---|
| **1 -- the run** | physics, and accuracy targets | `config.yaml` | silent |
| **2 -- numerics** | mechanism: how the answer is computed | schema + an advanced reference | **prints a line when set off-default** |
| **3 -- dev** | VOIDS the answer | schema only | warns, as it already does |

The runtime rule is what keeps the tiers honest and pays for hiding anything: a hidden key that changes
the answer announces itself in the log, so a run's provenance never depends on the reader knowing which
keys exist. This directly answers B's folklore risk.

The `dev` line is drawn at *does this void the answer* -- NOT at *does anyone run it*. That is why
Newton is not in `dev`: a Newton run is a valid run producing a correct answer in a restricted regime,
and it is the independent oracle that certifies the production solver (`tests/solver_consistency`:
"the production solver is matrix-free Anderson, which has no independent Jacobian to validate it").
Filing it under `dev` would say its output is untrustworthy, which contradicts its only job.

### Tier 1 -- `config.yaml` in full

Identical to sketch B's, with two changes:

```yaml
evaporation:
  et_sigmoid:                  # `off` disables the soil<->open-water ET transition (taper 2)
    wtd_center: 0.05
    logistic_width: 0.1
  extinction_depth: 8          # `none` disables the extinction taper (taper 3)
```

**No taper booleans.** `-wtm_evap_taper` and `-wtm_extinction` disappear without becoming keys: the
absence of the parameter IS the off-switch. `extinction_depth: none` means "no extinction depth", which
is what turning taper 3 off means. This also respects the design record's distinction -- `et_sigmoid`
nests because it is a transition FUNCTION with shape parameters; `extinction_depth` is flat because it
is a single physical LIMIT -- and it keeps the test controls alive, which matters because every
off-switch use is a test control (`boundary_analytic`'s analytic baseline, `flicker_evap`'s limit-cycle
demo, taper study C's runaway).

Two flags removed, zero keys added.

### Tier 2 -- `numerics:` (schema; not in the reference config)

Renamed from `solver:` because after `adaptive` moves to `time:` and `method` is seeded by `run.type`,
what remains is discretisation and solver internals together -- `time_integration` and `smoothing` are
not "solver" settings in any useful sense.

```yaml
numerics:
  method: anderson             # seeded by run.type; picard/newton are verification oracles
  time_integration: tr-bdf2    # seeded by run.type
  t_bar: false                 # seeded by run.type
  tolerance: 1e-6
  max_iterations: auto

  time_step:                # ONE controller; serves time.adaptive AND newton.dt_continuation
    grow: 1.5
    shrink: 0.25
    grow_if_niter_leq: 8
    max_retries: 15
    norm: rms
    dt_max: auto

  smoothing:
    ksat_surface: 0            # 0 = off; exists so a Jacobian FD check has a smooth tangent
    ksat_soilbottom: 0
    storativity_surface: 0.01  # sub-grid roughness; ALWAYS ON -- physics, not a check tool

  anderson:
    restart:
      enabled: false
      rho: 0.9
      patience: 3
      max_it: 200
      max_restarts: 10

  newton:
    dt_continuation: true      # implied by method: newton; opting out warns
    dt0: auto

dev:
  allow_aboveground_water_columns: false

output:
  trace: []                    # [dt] -> machine-readable per-step DTTRACE lines
```

`storage:` is **gone as a setting.** It is already not a choice: active_set auto-enables volume storage
over whatever is asked for, so offering it is a key the default configuration overrides -- the same
class of defect as `dev.active_set`, differing only in that it overrides the USER rather than another
key. If the secant form is still wanted for the BE baseline it belongs to `time_integration`
(`backward-euler-secant`), where it is one discretisation among others rather than an orthogonal knob
that silently loses.

`dt_trace` becomes `output.trace: [dt]`, next to `verbosity`. It changes only what is printed, so by
the agreed criterion it MAY remain a flag -- but a list here costs nothing, scales to further traces,
and means the model has no `-wtm_` surface at all.

### What C deletes, and the arithmetic

| deleted | n | why |
|---|---|---|
| handoff (4) | 4 | dormant end to end; the mechanism has never executed in a test |
| volume-based SNES convergence (3) | 3 | dormant end to end |
| `kirchhoff`, `aa_picard`, `predict_guess`, `relax` | 4 | dormant. `kirchhoff` is the one with a cost: it gates the active-set pin in `FormJacobianLocal`, so removing it means removing that guard too |
| `dev_padded_dirichlet` | 1 | dormant by BOTH routes -- its schema key already exists and is unused |
| `bdf2` (head form) | 1 | archive-only; if kept it should be a `time_integration` value, not a flag |
| `stiff` | 1 | dissolves into `run.type` seeding; it is a symptom of that gap |
| **total deleted** | **14** | |

| absorbed without a key | n | how |
|---|---|---|
| `evap_taper`, `extinction` | 2 | `et_sigmoid: off`, `extinction_depth: none` |
| `dt_norm_rms` + `dt_norm_max` | 2 -> 1 | one enum `norm: rms\|max`; removes today's undefined both-set case |

**34 flags -> 0 flags.** 14 deleted, 3 absorbed, 17 become config keys, of which **0 appear in the
reference config**.

### What C assumes, and what it costs

- It assumes the dormant groups are genuinely finished experimenting. That is a modelling judgement and
  it is yours; the inventory's per-group decisions are the input. If any is kept, it moves to tier 2
  rather than reappearing in tier 1.
- It assumes `run.type` seeding gets built. Without it, tier 1 cannot omit method/integrator/t_bar.
  This is the single largest piece of work in any of the three sketches.
- The `ar_*` four remain as tier-2 keys here on the grounds that a hidden key is cheap. The alternative
  -- hard-code them and delete the knobs, keeping the restart mechanism -- is still open and would take
  17 keys to 13.
- Renaming `solver:` -> `numerics:` is a breaking change for existing configs. It is cheap now (the
  block is about to be rewritten anyway) and expensive later.


## Choosing between them

| | A | B | C |
|---|---|---|---|
| keys in the reference config | ~55 | ~20 | ~20 |
| mechanisms shipped that never ran | 13 | 13 | 0 |
| needs `run.type` seeding built | no | **yes** | **yes** |
| decisions required before starting | none | none | **the dormant-group calls** |
| can a hidden setting change the answer unannounced | n/a | yes | no (tier-2 runtime rule) |

A is the option that requires no decisions, which is exactly why it preserves the problem: it turns
thirty-four development artefacts into a supported interface. B fixes the user's experience without
fixing what is underneath. C is more work and needs your judgement on the dormant mechanisms, but it is
the only one where the finished model contains only things that work.


---

# Sketch D -- CHOSEN (Andy, 2026-09-02)

> "For long-term maintainability and reproducibility, I think we should create a config-file structure
> that exposes everything, but with reasonable defaults such that a user-created config file can be
> quite short."

The decision is on the axis of REPRODUCIBILITY, which is the one argument for exposure that does not
reduce to taste. It also dissolves the A-vs-B tension, which was a false one on my part: I had conflated
THE SCHEMA with THE FILE YOU WRITE. The reference is ~55 keys; the file a user writes is ten, because
omission is legal.

## Decisions settled

1. **Everything is exposed as a config key.** No `-wtm_` flag survives as the only route to a setting.
2. **Reasonable defaults, so a user config is short.** Omission is legal for every key except the few a
   run cannot infer (`io.source`, `io.region`, `time.*`).
3. **The run emits a full YAML including defaults, alongside the outputs.** (Andy: "have the run generate
   a full yaml including the defaults that sits alongside the outputs.")
4. **The method must be chosen uniquely.** No setting may change which solver runs.
5. **Constant defaults to begin with**, made `run.type`-dependent only if a need appears. (Andy: "Let's
   start with constant defaults and make them vary with run type if we need to.")
6. **Shared settings appear once, at `solver:` top level**; only genuinely solver-specific settings nest.
   16 of the 17 shared settings are shared by ALL THREE methods, so duplicating them buys nothing and
   costs the inert-key problem.

## A defect this decision surfaces immediately

**The shipped reference config is not the code's defaults.** Three keys disagree:

| key | code default | `config.yaml` ships |
|---|---|---|
| `solver.time_integration` | `""` = backward-euler (`parameters.hpp:80`) | `tr-bdf2` |
| `solver.storage` | `false` = secant (`parameters.hpp:73`) | `volume` |
| `solver.adaptive_dt` | `false` (`parameters.hpp:95`) | `true` |

Today that is survivable because every user starts from `config.yaml` and keeps the lines. Under a
defaults-driven short config it is a trap: DELETING a line changes the numerics, and two configs that
look equivalent are not. A user who writes a ten-line config gets first-order fixed-step backward-Euler
with secant storage -- not the tr-bdf2 + adaptive + volume combination the reference implies is normal.

**Reconciling these is a prerequisite, not a follow-up.** The code defaults should move to what
`config.yaml` ships, since that is the combination actually recommended and tested.

## Requirements the decision implies

**R1 -- resolved-config emission.** Each run writes `resolved_config.yaml` into its output directory: every
schema key with the value actually used, re-runnable as-is. Half of this exists -- `Parameters::print()`
emits `--- resolved configuration ---` -- and `WTM.cpp:1097` already carries the TODO
("also dump the fully-resolved config as run"). What is missing is machine-readable YAML in the run
directory rather than a log block.

The source records why this matters, at `parameters.cpp:463`:

> KEEP THIS IN SYNC with the fields of Parameters. It went dead once -- nothing called it -- and drifted
> behind the config walk's new keys while runs quietly logged nothing; that gap turned a silently
> overridden setting into a wrong conclusion.

A defaults-heavy schema makes that drift more damaging, so R1 needs a test asserting **every schema key
appears in the emitted YAML** -- otherwise the dump silently falls behind the schema again.

**R2 -- unique method selection.** Exactly one key, `solver.method`, chooses the solver. Today five
settings force the path (`CreateSNES.cpp:137,144,152,173,178`), so `method: picard` +
`time_integration: tr-bdf2` silently yields Anderson (task #18). Under D that combination must ABORT with
a message naming both keys. Same for `adaptive_restart`, which currently forces Anderson merely by being
set -- a tuning dial that changes the solver.

**R3 -- contradictions abort, they do not resolve.** The failure this repo keeps producing is a setting
accepted and silently discarded (#27, #28, and suspected #35). With everything exposed, the surface for
that grows, so the rule has to be structural: if two keys cannot both be honoured, name them and stop.

## The schema, with defaults

Defaults below are the CODE's, read from source, except the three marked `[RECONCILE]` where
`config.yaml` disagrees and the reference value should win.

```yaml
run:
  type: equilibrium              # test | equilibrium | transient
  initial_water_table: saturated # saturated | supplied | <path>
  equilibrium_stop:
    tol: 0.001                   # m of water; 0 = never
    metric: frac                 # max | rms | frac
    frac: 0.001

time:
  deltat: <required>
  total: <required>
  report_interval: 100
  save_every_n_reports: 1

io:
  source: <required>
  region: <required>
  time_start: <required>
  time_end: <required>

output:
  directory: results/
  outfile_prefix: wtd_
  run_log: run.log
  if_exists: increment           # increment | overwrite | error
  verbosity: normal              # quiet | normal | verbose
  trace: []                      # [dt] -> per-step DTTRACE lines   (was -wtm_dt_trace)

boundaries:
  land: neumann_toposlope        # neumann_toposlope | dirichlet_sea_level

transmissivity:
  fdepth:
    a: 200
    b: 150
    fmin: 2
  additive_background_transmissivity: 0

evaporation:
  et_sigmoid:                    # `off` disables taper 2      (was -wtm_evap_taper)
    wtd_center: 0.05
    logistic_width: 0.1
  extinction_depth: 8            # `none` disables taper 3     (was -wtm_extinction)

surface_water:
  mode: routed                   # routed | ponded | removed
  runoff_ratio: off              # number in [0,1] | raster | off
  infiltration_during_flow: false
  collection:
    method: active_set           # active_set | explicit | implicit | off | extended_soil

solver:                          # SHARED settings: read whatever the method
  method: anderson               # anderson | picard | newton  -- THE ONLY method selector (R2)
  time_integration: tr-bdf2      # [RECONCILE] code says backward-euler
  storage: volume                # [RECONCILE] code says secant. NOTE active_set forces volume regardless
  adaptive_dt: true              # [RECONCILE] code says false
  t_bar: false
  tolerance: 1.0e-6              # -> snes_stol
  max_iterations: auto           # -> snes_max_it; auto = PETSc's per-method default

  time_step:                  # ONE controller; serves adaptive_dt AND newton.dt_continuation
    error_tol: 0.1               # m of water, per step
    dt_max: auto                 # auto = 1000 * time.deltat
    grow: 1.5
    shrink: 0.25
    grow_if_niter_leq: 8         # renamed from -wtm_dtc_easy_iters
    max_retries: 15
    norm: rms                    # rms | max   (was two booleans with an undefined both-set case)

  smoothing:
    ksat_surface: 0              # 0 = off; for a smooth Jacobian FD tangent
    ksat_soilbottom: 0
    storativity_surface: 0.01    # sub-grid roughness; ALWAYS ON -- physics

  anderson:                      # solver-SPECIFIC
    restart:
      enabled: false
      rho: 0.9
      patience: 2
      max_it: 40
      max_restarts: 30

  newton:                        # solver-SPECIFIC
    dt_continuation: true        # implied by method: newton; opting out warns
    dt0: auto                    # auto = time.deltat / 200

dev:                             # VOIDS the answer; warns at runtime
  allow_aboveground_water_columns: false

parallel:
  threads_per_rank: 1
```

## What a user actually writes

```yaml
run:
  type: equilibrium
time:
  deltat: 1yr
  total: 1000yr
io:
  source: surfdata/
  region: Australia_
  time_start: "021000"
  time_end: "021000"
```

Nine lines. Everything else is defaulted, and `resolved_config.yaml` in the output directory records
what those defaults resolved to for this run, at this version.

## What is still to decide

Only ONE question per remaining flag group, and it is not about placement: **does this mechanism stay in
the code?** Under D, exposure is automatic for whatever survives -- a kept mechanism gets a key, a
deleted one gets nothing. That collapses the walk-through from three questions per group to one.

The groups awaiting that call, all DORMANT (zero execution coverage -- see FLAG_INVENTORY.md):

| group | n | note |
|---|---|---|
| ~~Anderson -> finisher handoff~~ | ~~4~~ | **RETIRED 2026-09-03** -- removed from the code; see FLAG_INVENTORY.md |
| volume-based SNES convergence | 3 | never executed in any test |
| `kirchhoff` | 1 | removing it also removes the guard it places on the active-set pin in `FormJacobianLocal` |
| `aa_picard`, `predict_guess`, `relax` | 3 | |
| `dev_padded_dirichlet` | 1 | dormant by BOTH routes; its schema key already exists, unused |
| `bdf2` (head form) | 1 | archive-only; if kept it becomes a `time_integration` value, not a flag |
| `stiff` | 1 | a preset; dissolves if `run.type` seeding is ever built, otherwise decide on its own terms |
| ~~`fsm_delta_source`~~ | ~~1~~ | **NOT A DECISION -- the row was stale.** Three live test invocations, and #40 made it COMPOSE with active_set rather than be superseded by it. It survives, so under D it needs a config key |

The `ar_*` four are NOT on this list: the restart mechanism has a live test, so under D its constants
simply become keys. Whether to hard-code them instead is no longer worth asking -- a defaulted key that
nobody sets costs nothing.


## R2 as built: method uniqueness, and the one case where it changes an answer

`solver.method` is now the only key that selects the solver. Five settings used to force the path
(`CreateSNES.cpp:137,144,152,173,178`); four of them now REFUSE an incompatible method instead of
overriding it, and `use_picard` no longer reads the integrator.

Refused by name, each naming both keys:

| asked for | outcome before | outcome now |
|---|---|---|
| `method: picard` + `time_integration: tr-bdf2` | silently ran Anderson (task #18) | abort |
| `method: newton` + `time_integration: tr-bdf2` | both `force_anderson` and `use_newton` set; unclear | abort |
| `anderson.restart.enabled` + a non-anderson method | switched the solver to Anderson | abort |
| `-wtm_aa_picard` / `-wtm_handoff` + a non-anderson method | switched the solver | abort |
| `-wtm_stiff` + a method other than newton | Newton won, or did not, by ordering | abort |

**The case that is not a bug fix.** `solver.time_integration: bdf2` with NO method used to select
PICARD, because `use_bdf2` fed the `use_picard` expression. Under "the method is chosen only by
`solver.method`" it would become 2nd-order ANDERSON -- a silent change to the answer of every existing
config relying on the old implication. Andy chose (c), 2026-09-02: **refuse it and make the user state
the method.** Neither silence was acceptable; this breaks such configs deliberately, and the message
names the two fixes:

```
solver.method: picard    -- the BDF2-on-V Picard operator (what this config did before)
solver.method: anderson  -- 2nd-order matrix-free Anderson
```

Rejected: (a) let it become Anderson -- cleanest rule, but changes answers silently, which is the whole
failure class this arc exists to remove. (b) keep `bdf2 => picard` as a documented implication -- the
method would still be chosen by something other than `method`, i.e. R2 not actually done.


## Step 10: the three default flips, and how a default should fail

The plan was "reconcile three code defaults to what config.yaml ships". Checking each first shows they
are NOT alike, and one is worse than a solver switch.

### What each one actually couples

| flip | couples to the solver? | measured |
|---|---|---|
| `storage` secant -> volume | **no** | `active_set` already auto-enables volume, and PRINTS a NOTE while doing it (`transient_groundwater.cpp:1480`). Only fires when no b=0 scheme is selected |
| `time_integration` backward-euler -> tr-bdf2 | **yes, indirectly** | tr-bdf2 runs only on the Anderson path. As a DEFAULT it would make `solver.method: picard` -- alone, a config naming one key -- abort on a key the user never wrote |
| `adaptive_dt` false -> true | **worse than a switch** | it DISABLES Newton's continuation ramp while ANNOUNCING it |

The third, measured on `tests/dt_sensitivity` inputs with `solver.method: newton`:

```
adaptive_dt: true            Newton PTC, dt0=157680. s, grow x1.5 ...   <- announced
                             adaptive dt: 30 steps (0 rejected) ...     <- what actually ran
adaptive_dt absent           Newton PTC, dt0=157680. s, grow x1.5 ...
                             dt-continuation: deltat now 5.24e+08 s ... <- the ramp ran
```

`WTM.cpp:593` is `if (use_dt_adaptive) { ... } else if (use_newton_continuation) { ... }`: adaptive wins
and the ramp never executes, while `InitialiseSNES` has already printed the ramp's banner. Newton needs
that ramp to converge from cold -- `tests/newton_solver`'s CONTRACT arm asserts precisely that -- so
flipping this default would silently break Newton's working recipe on every Newton run.

### The rule that decides failure vs warning

The deciding question is not how bad the combination is. It is **whether the value was EXPLICIT or
DEFAULTED**. A user is answerable for what they wrote; they are not answerable for a default.

| situation | response |
|---|---|
| two EXPLICIT settings contradict, and no resolution can be right | **ABORT**, naming both keys and the fix |
| an EXPLICIT setting cannot be honoured at all | **ABORT** |
| a DEFAULTED value would contradict an explicit one | **RESOLVE** in favour of the explicit one, and **PRINT** what it resolved to |
| honoured, but known-bad or nonphysical | **WARN**, keep running |
| an EXPLICIT value would be overridden | **never** -- this is the defect class (#28 `dev.active_set`) |

An abort must never fire on a key the user did not write. That is the whole reason `auto` exists rather
than a concrete default: a concrete default is indistinguishable from a user's choice, so it can
manufacture a contradiction the user cannot see or fix.

The machinery already exists -- `dt_continuation_set`, `dt_tol_set`, `dtc_dt_max_set` -- and the
resolve-and-print pattern is already in the code, at `transient_groundwater.cpp:1480`.

### Applied: `auto` as the default for all three

PROPOSED resolutions. Each cell is a choice, not a derivation; the rule used was minimum surprise --
preserve today's behaviour except where `config.yaml` already documents otherwise.

```yaml
solver:
  time_integration: auto   # auto | backward-euler | bdf2 | tr-bdf2
  storage: auto            # auto | volume | secant
  adaptive_dt: auto        # auto | true | false
```

| key | `auto` resolves to | explicit conflict |
|---|---|---|
| `time_integration` | anderson -> `tr-bdf2`; picard -> `backward-euler`; newton -> `backward-euler` | `tr-bdf2` + a non-anderson method -> ABORT (built, bcc6638) |
| `storage` | `volume` everywhere (the physically exact form; secant is the 1st-order legacy) | `secant` + `active_set` -> ABORT: active_set needs a b=0 residual path, so secant cannot be honoured. Today it is overridden with a NOTE |
| `adaptive_dt` | newton with continuation -> `false` (the ramp owns the step size); otherwise `true` | `true` + `dt_continuation: true`, both explicit -> ABORT: two step-size controllers, and today one is silently dropped |

Every resolution prints one line and lands in `full_config.yaml`, so a short config stays recoverable.

### Where they sit

Unchanged: all three stay at `solver:` top level. They are read by whichever method runs -- shared, not
method-specific -- which is exactly the criterion that put them there.

### The structural alternative

`adaptive_dt` and `newton.dt_continuation` are two booleans for one thing: WHO SIZES THE STEP. Their
conflict is not a validation problem, it is a modelling error made representable. One enum removes it:

```yaml
solver:
  time_step:
    mode: auto        # auto | fixed | adaptive | continuation
```

`auto` -> continuation on Newton, adaptive elsewhere. The contradiction cannot be written, so no abort is
needed for it, and `WTM.cpp`'s `if (adaptive) ... else if (continuation)` becomes a switch on a stated
mode rather than an implicit precedence.

Cost: a breaking rename with 8 + 10 callers, on top of step 10's own blast radius. Benefit: one fewer
class of contradiction to validate, and the precedence stops being invisible.
