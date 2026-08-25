# FillSpillMerge every timestep — tight surface–subsurface coupling

## The problem

WTM couples a distributed groundwater (GW) solve with FillSpillMerge (FSM), which routes surface water
through the depression hierarchy. Historically FSM ran **once per `maxiter` GW sub-steps** — a batched,
operator-split coupling. The batching interval (`maxiter` / "niter") was a *numerical* parameter, but it
changed the *physics*: marginal-lake and lakeshore equilibria depended on how often FSM ran. This showed up as

- the free-surface **flicker** (a shore cell flooded by FSM, drained by GW between FSM calls, re-flooded),
- the **collector×FSM equilibrium divergence** (explicit vs implicit `runoff_collector` reaching equilibria
  ~24.7 m apart with FSM on, 0.11 m apart with FSM off),

both of which are artifacts of *batching* FSM. The nonlinearity that drives the N-dependence is open-water
evaporation firing at the potential rate only while a cell is flooded: how long a leaky depression stays
flooded per FSM cycle (hence its water balance) depends on the batching interval.

## The change

FSM now runs **after every accepted groundwater timestep** (`src/WTM.cpp`,
`couple_surface_and_recharge<elev_t>()`): gather the water table to rank 0 → hand this step's above-surface
removal to FSM → FillSpillMerge → scatter the post-FSM table back → set recharge for the next step. Recharge
moves per-step because it re-arms `arp.runoff` (= `runoff_ratio*rech`) that the next step's FSM consumes.
`fsm_off` runs call the helper once per report (gather + recharge only). All three solver paths
(fixed / Newton-continuation / adaptive) call it after each accepted step; FSM never runs on a rejected step.

This is encoded in the model structure, not behind a flag — it is the coupling.

## `maxiter`/`niter` retired; new reporting knobs

Batching is gone, so `maxiter` is no longer a coupling interval. What remains useful is a *reporting* cadence,
now its own knob, fully decoupled from FSM:

- **`report_interval`** — steps between the equilibrium check + log line + output, as a step count
  (`report_interval 100`) or a simulated time (`report_interval 50yr` / `1000s`, resolved via `deltat`).
  Default **100 steps + a loud warning** if omitted.
- **`save_nreport_interval`** — save a raster every K reports. Default **1 + a loud warning**.
- `maxiter` / `cycles_to_save` are **deprecated**: parsed, mapped to the new knobs, and warned.

## Cost

FSM *compute* is negligible — microseconds per call at Esquibel (384k cells), flat across scales; running it
every step costs ~4% wall single-node, all of it per-cycle bookkeeping + the rank-0 gather, not FSM
(`benchmark/esquibel/FSM_COST.md`). Because each per-step call routes ~1/50th the water, calls stay in the
cheap early-exit regime. The real driver for parallelizing FSM (#80) at *global* scale is **memory** (the full
grid is replicated on rank 0 for serial FSM+DH), not compute or the gather (which the multi-node study shows
stays flat and <0.3% of a cycle).

## Verification

- `fsm_off` is **byte-identical** (the GW solve is untouched; the full `fsm_off` suite stays green).
- `fsm_on` equilibria are **report-interval-independent** (verified 0.000 m across `report_interval` 1/2/12),
  confirming FSM genuinely runs every step, decoupled from reporting.
- FSM MPI consistency (n=1 vs n=N) still holds.
- The four `fsm_on` golden references were regenerated to the new tight-coupled equilibria (fsm_evap0/1
  +3.5 m, fsm_runoff/hi ~+40 m; below_ground / transient unchanged) — the N-independent answers.

## Where this points

- **Implicit coupling (future, #116):** with FSM every step, the FSM-induced water-table change can be folded
  into the recharge and solved *implicitly* in Anderson. FSM itself (fill-spill-merge) is combinatorial and
  cannot go into a residual, but its *result* — the lake stage — can enter as a Dirichlet head boundary (the
  active-set pin at `topo + surface_water_depth`), with the lake↔aquifer exchange in the residual. See the lake-as-head
  boundary design.
- **Surface-crossing flicker (separate, #114):** the evaporation-discontinuity flicker at `wtd = 0` is a
  distinct, GW-side cause, managed by the smooth evaporation taper (taper 2); the `flicker_evap` test verifies
  it settles.
- **Parallel FSM (#80) + DH swap (#81):** the memory driver for global-scale tight coupling.
