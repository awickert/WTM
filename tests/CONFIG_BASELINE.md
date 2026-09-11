# The test baseline: settings every suite takes at the model's default, on purpose

**What this file is for.** Materialised suite configs (#83) state every setting a run resolves to.
Most of those values are not choices any individual suite made -- they are the model's own defaults,
and every suite takes them. Marking all of them "nobody chose this" in 39 files would be true but
useless: ~1,100 markers nobody will ever clear, which is how "marked but unowned" becomes the new
normal and the marker stops meaning anything.

So they are owned ONCE, here, and the suite configs point at this file instead of carrying a marker.

**What that ownership claims, exactly.** For each key below: *the model's default is the right value
for a test unless that test's subject says otherwise.* It does NOT claim the value is optimal, or
validated, or that the default should not change -- only that a suite which does not mention the key
is content with whatever the model ships, and that this was decided rather than overlooked.

**A suite that cares states the key itself**, which then appears in its config with no marker and no
reference here, because it is that suite's decision.

## The baseline

| key | value | why it is baseline, not a per-suite choice |
|---|---|---|
| `solver.tolerance` | `1e-8` | per-solve step tolerance; suites that probe convergence state their own |
| `solver.max_iterations` | `10000` | a cap, not a target -- a run that reaches it has already failed |
| `solver.convergence.residual_gate` | `1e-5` | #104: the water-step verdict is refused while the residual is above this x its initial value. Relative on purpose -- an absolute bound would not carry from a 96-cell fixture to 384k cells. MEASURED: over a 20-point dt sweep the arms that reached the converged answer exited at worst 5.59e-08, the ones that did not at best 2.12e-02 -- a window of 3.8e+05, so 1e-5 is log-centred rather than tuned. No suite varies it; the suites that vary a per-solve tolerance vary `water_volume_tol` |
| `solver.anderson.restart.*` | off | the rho-driven restart loop; `tests/adaptive_restart` is the suite that turns it on |
| `solver.smoothing.ksat_surface` | `0` | sharp; the smooth form exists for Jacobian FD checks, which state it |
| `solver.smoothing.ksat_soilbottom` | `0` | as above |
| `solver.smoothing.storativity_surface` | `0.01` | sub-grid roughness, always on. MEASURED INERT under active_set (#77) |
| `solver.time_step.grow/shrink/grow_if_niter_leq/max_retries` | `1.5 / 0.25 / 8 / 15` | controller dials; suites that freeze or drive the controller state them |
| `solver.time_step.norm` | `rms` | robust on cold spin-up; `max` is opt-in and hostage to a few kink cells |
| `transmissivity.additive_background_transmissivity` | `0` | v2.0.1 behaviour -- no bedrock floor |
| `dev.under_relaxation` | `1` | off. It VOIDS a transient trajectory, so a suite wanting it must say so |
| `parallel.threads_per_rank` | `1` | tests pin threads for determinism |
| `output.verbosity` | `normal` | printing only |

## NOT baseline -- these stay marked until a suite owns them

They are answer-changing, and which value is right depends on what the suite measures:
`run.equilibrium_stop.metric` / `.frac`, `solver.convergence.metric` / `.water_volume_tol`,
`solver.t_bar`, `dev.storage_form`, `solver.time_integration`, `solver.time_step.mode`,
`solver.time_step.error_tol`, `surface_water.collection.method`, `surface_water.fsm_coupling`,
`boundaries.land`, and all of `evaporation.*`.
