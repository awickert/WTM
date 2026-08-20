# Free-surface flicker: three failure mechanisms at the water table (`wtd = 0`)

**Date:** 2026-08-20 · **Branch:** `bdf2-adaptive-dt`

When the water table sits at the land surface (`wtd = 0`), the model has a **nonsmooth nonlinearity**: a cell
is either subsurface (groundwater) or ponded (surface water), and the transition is not differentiable. A
frozen-coefficient / operator-split iteration that crosses that boundary can **overshoot and settle into a
limit cycle** — the water table bounces above/below the surface and the per-cycle change never decays. This is
the "lakeshore flicker."

## The unifying numerics

Treat one model cycle as a map `F` applied to the water table:
`w_{k+1} = F(w_k)`, where `F` = (maxiter groundwater solves) ∘ (evaporation) ∘ (FSM surface routing).
**Equilibrium is the fixed point `w = F(w)`; the per-cycle change `|w_{k+1} - w_k|` is the fixed-point
residual.** A limit cycle is `F` failing to *contract* near the surface. So every cure is about restoring
contraction of `F` — by removing the nonsmoothness (regularize / clamp), by damping/accelerating the outer
iteration, or by solving the free boundary consistently (active set / semismooth Newton). The literature name
for the coupled-block version is the loosely-coupled surface–subsurface splitting artifact.

There are **three distinct triggers** for the loss of contraction. They live in different parts of the code
and need different guards; the existing `tests/limit_cycle` isolates only the first.

---

## Mechanism 1 — storativity jump + seepage kink (inner GW solve)

- **Trigger:** effective storativity jumps from specific yield (`Sy ≈ porosity`) below the surface to `~1`
  (open water) at `wtd = 0`, plus the seepage/removal kink. The matrix-free Anderson (frozen-coefficient)
  iteration linearizes across that jump and overshoots it each cycle -> period-2 bounce.
- **Where:** the groundwater solve alone. Reproduces with **`fsm_on 0, evap_mode 0`** — nothing else can be
  the cause.
- **Mass is conserved through the flicker — the overshoot is in the SOLUTION (head), not the volume.** The
  backward-Euler storage uses the *secant* effective storativity, and `S_secant·Δh ≡ V(wⁿ⁺¹) − V(wⁿ)`
  exactly (proven to machine precision; see `finding_cc_secant_storage_inconsistency`), the flux form is
  conservative, and recharge is a fixed volume. So every *converged* cycle is an exact volume balance
  `ΔV = recharge − net outflow` to the SNES tolerance, regardless of how far the head overshoots. The limit
  cycle just sloshes water between subsurface and above-surface storage (`V(w>0)` counts surface water at
  storativity 1); nothing is created or destroyed. The water budget closes to ~machine zero
  (`tests/mass_balance`, `finding_surface_water_management_design`).
- **Current management:** the post-solve **exfiltration clamp** (`surface_exfiltration_to_runoff`, now the
  default) projects `wtd <= 0` and routes the excess to runoff/FSM — a *mass-conserving move*, tracked in the
  budget (`total_surface_removed`), not a loss; plus `storativity_surface_smoothing_width` (regularize the
  jump) and `-wtm_relax` (damp). Clamp is the effective one; smoothing/relax are marginal.
- **Principled cure:** solve the free boundary as a complementarity (obstacle) problem — **primal-dual
  active set = semismooth Newton** — so the constraint is enforced *inside* the solve, not post-hoc. Future
  work (see the boundary-framework issue / a follow-on).
- **Test:** `tests/limit_cycle` (to be made a **positive** test: the flicker-prone fixture *settles* under
  the default clamp and the schemes agree; the bare/unmanaged contrast is documented here, not asserted).

## Mechanism 2 — evaporation discontinuity (below vs above ground)

- **Trigger:** phreatic evapotranspiration (below the surface) and open-water evaporation (above it) are
  different rates, so the **removal jumps at `wtd = 0`**. A cell oscillating around the surface sees a
  different sink on each side, which can sustain a cycle independent of the storativity jump.
- **Where:** the evaporation term. Reproduces with **evaporation on, FSM off** (isolates it from the
  coupling of mechanism 3), water table crossing the surface.
- **Current management:** **taper 2** (`evap_taper`) makes ET -> open-water a single **C1 (smooth)
  transition** (`evap_taper_s`, `evap_taper_wtdc`), removing the discontinuity; **taper 3**
  (`extinction`) bounds arid draw-down.
- **Principled cure:** the smooth taper *is* the principled regularization here (a mollified sink). Verify
  it holds where a hard ET/owe switch would flicker.
- **Test:** [NEW] `tests/flicker_evap` — a surface-crossing fixture with evaporation on and FSM off;
  assert the water table settles (per-cycle change decays) with the smooth taper.

## Mechanism 3 — FSM single-cell concentration + evaporation (outer GW↔FSM coupling)

- **Trigger:** FSM concentrates surface water into a single depression cell (a lake); that cell then
  evaporates at the open-water rate and empties; the next FSM refills it. Because the model runs
  **`maxiter` groundwater solves per FSM call**, the GW block over-equilibrates to a surface state that FSM
  then invalidates — an **operator-splitting limit cycle** at the coupling frequency.
- **Where:** the coupling of the GW block and FSM. Needs **`fsm_on 1, evap_mode 1`** and a depression.
- **Current management:** the per-cycle equilibrium metric measures the state *after* FSM (so within-cycle
  sub-step wobble is not mistaken for a cycle); the clamp keeps the GW block from overshooting.
- **Principled cure (relates to the numerics):** the per-cycle water table is a fixed-point sequence of the
  *coupled* operator `F`; flicker = non-contraction of that OUTER iteration. Cures, in order of rigor:
  (a) **iterate the GW↔FSM coupling to a joint fixed point** (converged splitting) or
  **Anderson-accelerate / damp the outer cycle** (as Anderson already accelerates the inner GW solve);
  (b) **tighter split** — reduce `maxiter` so FSM runs more often (cost: FSM is the serial bottleneck);
  (c) **under-relax** the surface-water update between FSM calls.
- **Test:** [NEW] `tests/flicker_fsm_evap` — a sink + FSM + evaporation fixture; assert the coupled state
  settles (per-cycle change decays) under the chosen management. NOTE: an early prototype (shallow table,
  strong open-water evaporation) *diverged* the GW solve on the first cycles — the regime is genuinely
  stiff, so the fixture must be tamed (gentler evap / warmer start / smaller maxiter) before it is a clean
  settling demonstration.

---

## Status / plan

| mechanism | isolate with | management | test |
|---|---|---|---|
| 1 storativity/seepage | fsm 0, evap 0 | exfiltration clamp (default) | `tests/limit_cycle` -> positive |
| 2 evap discontinuity | evap on, fsm 0 | smooth evap taper (taper 2) | `tests/flicker_evap` [NEW] |
| 3 FSM concentration + evap | fsm 1, evap 1, depression | couple-convergence / tighter split / damp | `tests/flicker_fsm_evap` [NEW] |

Each test is **positive** (assert the managed system *settles* — per-cycle change decays and, where two
schemes are run, they agree), not a negative "assert the bad flicker exists," so it is robust and needs no
nonphysical `allow_surface_ponding`. See `finding_lakeshore_flicker` for the original diagnosis.
