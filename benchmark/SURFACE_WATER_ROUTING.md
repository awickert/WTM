# Surface-water routing: the `runoff_collector` selector

**Date:** 2026-08-20 · **Branch:** `bdf2-adaptive-dt`

When recharge drives the water table to the land surface, the excess must leave the subsurface: it exfiltrates
at a **seepage face** (`wtd = 0`) and is routed to runoff / FillSpillMerge. This is a single boundary condition
— the free-boundary complementarity `wtd ≤ 0 ⊥ seepage ≥ 0` — with a *choice of how to enforce it*. The
`runoff_collector` config key makes that choice explicit.

```
runoff_collector implicit    # in-residual seepage face (exact, dt-independent). Anderson + Picard. DEFAULT.
runoff_collector explicit    # post-solve clamp (robust on every solver, dt-lagged)
runoff_collector off         # no collection -- NONPHYSICAL, warns
runoff_collector legacy      # the old -wtm_surface_sink band-sink defaults (dt-scaled)
```

**Default is `implicit`** (the exact face); the default *solver* is matrix-free Anderson.

**The selector wins over the legacy `-wtm_` surface flags.** In every mode except `legacy`, the selector sets
`-wtm_surface_sink`, `-wtm_direct_to_runoff` and `-wtm_surface_exfiltration_to_runoff` itself, so passing one of
those on the command line has no effect — the *config key*, not the command line, decides which enforcement
runs. Because `runoff_collector` defaults to `implicit`, this holds even for a config that never mentions it.
If you passed such a flag and the selector changed its effect, the run prints a one-line `NOTE [runoff_collector=…]`
saying so; the resolved value is also on the `c runoff_collector = …` line of the config echo at the top of every
run log. Set `surface_water.collection.method: legacy` to hand control back to the flags.

**Adaptive-dt and the implicit kink.** The implicit seepage's discontinuous `max(0,wtd)/dt` would spike the
adaptive-dt controller's error estimate at a cell crossing the surface (a projection jump that does not shrink
with `dt`), so it once could not be adaptively stepped. Fixed by **clamping the error predictor to the
feasible set** (`h_pred = min(h_pred, topo)`) in the norm: a cell *rising* to the surface still contributes its
true rise (bounding `dt` in Anderson's stable range), a *pinned* cell contributes ~0. So `implicit` now works
under `-wtm_dt_adaptive` too (verified: implicit-adaptive == implicit-fixed-cc to ~1 cm). *Excluding* the
constraint cells instead would unbound `dt` and pile the water — the predictor clamp is the right treatment.

All three modes route the above-surface excess to the **same** destination — `total_surface_removed` (the water
budget) and `arp.runoff → FillSpillMerge`. They differ only in *when the constraint meets the solver*.

## The three enforcements

| mode | mechanism | where | dt-dependence | solvers |
|---|---|---|---|---|
| `implicit` | in-residual seepage `max(0,wtd)/dt` (`-wtm_direct_to_runoff`) | inside `F(w)`, solved-for | none (exact) | Anderson today; Picard/Newton need active-set (Issue #7) |
| `explicit` | post-solve clamp (`-wtm_surface_exfiltration_to_runoff`) | after each step, projected | small (~1 cm, → 0 as dt→0) | all |
| `off` | none | — | — | all (nonphysical) |

- **`implicit`** adds the seepage removal to the residual, so the lateral flow field equilibrates *against* a
  surface pinned at `wtd = 0`. It is the exact seepage face and is dt-independent. Its removal is a
  discontinuous step at `wtd = 0`; the matrix-free Anderson path tolerates that kink, but the Picard operator /
  Newton Jacobian do not (Picard lands ~0.1 m off; Newton diverges), so `implicit` **warns** on those solvers.
  It runs **alone** — no post-solve clamp backstop — deliberately: a backstop would silently mop up any
  implicit overshoot and *hide* a bug, so the modes are mutually exclusive and implicit's misbehaviour stays
  visible (e.g. a small SNES-tolerance overshoot of a few cm shows as `max wtd > 0`).
- **`explicit`** lets the GW step solve *without* the constraint (the table mounds), then projects the overshoot
  back to `wtd = 0` and collects it. The flow field never sees the pin during the solve, so it is a lower-order,
  dt-lagged form of the same face — but robust on every solver (no tangent) and within ~1 cm of `implicit`,
  converging to it as `dt → 0`.
- **`off`** collects nothing; above-surface water piles up (hundreds of metres in a supply-rich basin). This is
  the nonphysical developer/diagnostic case (the former `-wtm_dev_allow_aboveground_water_columns`); it warns
  loudly.

In numerical terms this is the classic obstacle-problem split: `implicit` is the constraint solved *in* the
nonlinear system (active-set / complementarity), `explicit` is *solve-then-project* onto the feasible set.

## The band sink (taper 1) — a different strategy, turned off by every mode

The legacy default is the **taper-1 sub-surface sink** (`-wtm_surface_sink`): a smooth removal in a band of
width `2·qmax·dt` *below* the surface that holds the table strictly sub-surface (`wtd < 0`) so no cell ever
crosses the free boundary. It dodges the seepage face rather than enforcing it — keeping the solve smooth
(differentiable for Picard/Newton) and 2nd-order (BDF2-on-V "no-crossing" regime). The cost is that the
equilibrium table sits in a **dt-scaled band**, so it is **dt-dependent** (see Issue #6). The `runoff_collector`
selector turns the band sink **off** in every mode; fully retiring it (and making a mode the default) is a
later, regold-bearing step (Issue #7), with `explicit` the robust default and `implicit` the exact opt-in.

## A CLI hazard worth knowing

`-wtm_surface_sink 0` does **not** reliably mean "sink off" — that CLI form mis-parses (it can leave the surface
unmanaged, so water piles). Use `-wtm_surface_sink false`, or better, drive the choice through
`runoff_collector`, whose test asserts each mode by the config key and so cannot be fooled by the `0` form.
