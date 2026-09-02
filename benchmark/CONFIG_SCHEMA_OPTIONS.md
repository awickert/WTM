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

  step_control:                # serves BOTH adaptive_dt and newton.dt_continuation
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

  step_control:                # ONE controller; serves time.adaptive AND newton.dt_continuation
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
