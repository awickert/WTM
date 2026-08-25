# How to run the free surface (water-table-meets-ground), and why

**Date:** 2026-08-13 · **Branch:** `bdf2-adaptive-dt`
**Status:** decided + verified. Supersedes the smoothing route in `SURFACE_SINK_DESIGN.md` for this purpose.
**TL;DR:** run **Anderson** equilibria with **`-wtm_surface_exfiltration_to_runoff`** (post-solve clamp).
`-wtm_direct_to_runoff` (in-residual) is **not** deprecated — it is the required variant for the
**Picard / Newton** paths, and is more cold-stable at large dt. Two co-existing, regime-dependent methods.

## The decision (what to run)

For any run where the water table can reach the land surface (all equilibria; transients over wet
terrain), use:

```
-wtm_surface_exfiltration_to_runoff
```

- **fsm OFF:** the shed surface water goes to the runoff array and is removed — this is the **Fan &
  Miguez-Macho** result (equilibrium water table with surface-emergent water shed as runoff).
- **fsm ON:** the shed water is exactly the correct input to FillSpillMerge (it routes/ponds it). Same
  flag, same mechanism, both modes.

## Why (the physics + the numerics)

When recharge + lateral inflow would push a cell's water table **above** the land surface, the table
must sit **at** the surface (wtd = 0) and the excess becomes **surface water** (runoff → FSM). A
groundwater table *above* the solid ground is unphysical.

Two things make this a clean, local problem rather than a hard one:
1. **The exponential transmissivity does not extend to the surface.** `T = fdepth·ksat·exp((wtd)/fdepth)`
   applies only for `wtd < -shallow` (≈ −1.5 m; `dischargePotential`, `transient_groundwater.cpp`). Near
   and above the surface T is the smooth **clamped** form — so the shore is **not** an exp-stiffness
   problem. The only non-smoothness at the surface is (a) the storativity switch (Sy→porosity) and (b)
   the exfiltration removal.
2. **Engage the removal *post-solve*, not inside the residual.** `-wtm_surface_exfiltration_to_runoff`
   lets the table mound *during* the implicit solve (T stays clamped → the mound does not spread
   laterally), then **after** the solve routes the exact above-surface excess
   `storedVolume(wtd) − storedVolume(0)` to runoff/FSM and **clamps `wtd = 0`**. The residual the solver
   sees is smooth (no kink), and the carried state is pinned exactly at the surface.

## The in-residual variant: `-wtm_direct_to_runoff` (when to use it)

`-wtm_direct_to_runoff` puts the removal **inside** the implicit residual as `max(0, wtd)/dt` — a
non-smooth **kink** at wtd = 0, and a rate-relaxation rather than a hard constraint. **Use it for the
Picard and Newton paths:** those solvers build an operator / Jacobian, so the exfiltration must be *in* the
residual (with its tangent `directToRunoffTangent`); a post-solve clamp would break their consistency.
It is also **more cold-stable at large dt** (it holds wtd ≤ 0 *during* the solve → no mounding → the
cold big step doesn't overshoot).

Its two known costs on the **Anderson** path (why the post-solve clamp is preferred there):
- **Residual pond (wrong equilibrium):** at steady state `max(0,w)/dt = net input`, so the table settles
  at **`w = dt·(net input)` above the surface** — a dt-dependent phantom pond (**14,468 cells above
  ground, up to 2.26 m** at the converged dt=1 wk equilibrium). Mass-conserving, but the wrong table.
- **The bounce:** matrix-free Anderson overshoots the kink → the shore cells limit-cycle
  (~0.05 m for cc, ~0.5 m with `-wtm_Tbar`), so `eq_tol` never fires.

So: **Anderson → post-solve clamp; Picard/Newton (or when you need the extra cold big-dt stability) →
in-residual.** Neither is deprecated.

## Why NOT smoothing (`-wtm_storativity_surface_smoothing_width`)

Rounding the kink (the `SURFACE_SINK_DESIGN.md` route) needs a width that grows with the overshoot; a
width wide enough to quiet the worst cells distorts the equilibrium (~4 % at 4 m — see
`SURFACE_SMOOTHING_STABILIZATION.md`). It trades physics for stability. The post-solve clamp needs **no
width** and introduces **no distortion**.

## Evidence

Warm-start from the (ponded) limit-cycle state, fsm off, dt = 1 wk:

| flag | converges? | table above surface |
|---|---|---|
| `-wtm_direct_to_runoff` (in-residual kink) | **no** — limit-cycles, `eq_tol 0.01` never fires | 14,468 cells, up to 2.26 m |
| `-wtm_surface_exfiltration_to_runoff` (post-solve clamp) | **yes** — `eq_tol 0.01` at cycle 20, monotone | **0 cells** (max wtd 0.000 m) |

Cold dt-sweep, `-wtm_surface_exfiltration_to_runoff`, `eq_tol 0.01` (Esquibel, cold from t=0):

| dt | cc | `-wtm_Tbar` |
|---|---|---|
| 1 wk | **EQ @ 39** | **EQ @ 39** (equivalent) |
| 2 wk | **SNES diverges** | bounded (SNES converges) but shore cells limit-cycle ~0.05 m → noeq |
| 4–32 wk | diverge | diverge |

**Verdict: Tbar buys robustness, not speed, for this (well-behaved) equilibrium.** At the only
convergent dt (1 wk) cc and Tbar are identical (39 cycles). At dt=2 Tbar's T-averaging *stabilizes the
solve* (converges where cc's SNES diverges) — a real robustness edge — but the shore free-boundary
limit-cycles, so there's no *converged* larger-dt equilibrium, hence no cycle savings. Tbar's speed
value, if any, remains genuinely stiff cold-starts, not this regime. (Note the dt-ceiling is
flag-dependent: `direct_to_runoff`, which doesn't mound, reached dt=2 for cc — a stability↔correctness
tradeoff between the two variants.)

## Recommended invocation (equilibrium)

```
mpiexec -n N wtm.x <cfg> -wtm_anderson -wtm_surface_exfiltration_to_runoff \
        -wtm_fringe_source ksat -snes_stol 1e-6 -wtm_eq_tol 0.01
```

(`-wtm_surface_exfiltration_to_runoff` will become the **default** once goldens are regenerated; an
explicit `false`-style off-switch will disable it. Until then, pass it explicitly.)
