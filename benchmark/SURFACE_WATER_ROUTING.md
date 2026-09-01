# Surface-water routing: the `runoff_collector` selector

**Date:** 2026-08-20 · **Branch:** `bdf2-adaptive-dt`

When recharge drives the water table to the land surface, the excess must leave the subsurface: it **exfiltrates**
through the ground (`wtd = 0`) and is routed to runoff / FillSpillMerge. This is a single boundary condition
— the free-boundary complementarity `wtd ≤ 0 ⊥ q_exfiltration ≥ 0` — with a *choice of how to enforce it*. The
`runoff_collector` config key makes that choice explicit.

```
runoff_collector active_set  # semismooth constraint solved INSIDE the residual. DEFAULT (Anderson).
                             #   The only enforcement with no SPURIOUS dt-dependence.
runoff_collector implicit    # in-residual exfiltration constraint. The FORMER default.
                             #   NOT dt-independent: leaves a residual head ~ dt*inflow (see below)
runoff_collector explicit    # post-solve clamp (robust on every solver, dt-lagged)
runoff_collector off         # no collection -- NONPHYSICAL, warns
runoff_collector extended_soil  # aquifer continues above the surface -- NONPHYSICAL, [WIP]
```

**Default is `active_set`**; the default *solver* is matrix-free Anderson.

**The default is solver-dependent.** The active-set pin lives in the matrix-free Anderson residual and,
since the semismooth Jacobian row landed, in the analytic Newton Jacobian (`FormJacobianLocal`; not under
`-wtm_kirchhoff`, where the SNES variable is the discharge potential and the pinned row would need
`1/Φ'(wtd)`). The **Picard** operator still carries no tangent for it, and selecting active-set also
switches every collector removal off, so Picard would run with the constraint effectively UNENFORCED.
So when `surface_water.collection.method` is **unset**, the default resolves to `active_set` on Anderson
and Newton, and to `explicit` on Picard, with a NOTE. An explicit choice is always honoured (and warns).

**There is one channel.** The `-wtm_` surface flags that this selector used to supersede --
`-wtm_surface_sink`, `-wtm_direct_to_runoff`, `-wtm_surface_exfiltration_to_runoff` -- were retired with
the `legacy` mode on 2026-09-01 (fork issue #7). Passing any of them now aborts by name. The config key
is the only way to choose an enforcement, and the resolved choice is stated once per run:

```
surface-water exfiltration enforcement: explicit  [surface_water.collection.method]
```

**Adaptive-dt and the implicit kink.** The implicit exfiltration's discontinuous `max(0,wtd)/dt` would spike the
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
| `implicit` | in-residual exfiltration `max(0,wtd)/dt` (`-wtm_direct_to_runoff`) | inside `F(w)`, solved-for | **LINEAR in dt** (measured) | Anderson today; Picard/Newton need active-set (Issue #7) |
| `explicit` | post-solve clamp (`-wtm_surface_exfiltration_to_runoff`) | after each step, projected | small (~1 cm, → 0 as dt→0) | all |
| `off` | none | — | — | all (nonphysical) |

- **`implicit`** adds the exfiltration removal to the residual, so the lateral flow field equilibrates *against* a
  surface pinned at `wtd = 0`. Its removal is a
  discontinuous step at `wtd = 0`; the matrix-free Anderson path tolerates that kink, but the Picard operator /
  Newton Jacobian do not (Picard lands ~0.1 m off; Newton diverges), so `implicit` **warns** on those solvers.
  It runs **alone** — no post-solve clamp backstop — deliberately: a backstop would silently mop up any
  implicit overshoot and *hide* a bug, so the modes are mutually exclusive and implicit's misbehaviour stays
  visible (e.g. a small SNES-tolerance overshoot of a few cm shows as `max wtd > 0`).
- **`explicit`** lets the GW step solve *without* the constraint (the table mounds), then projects the overshoot
  back to `wtd = 0` and collects it. The flow field never sees the pin during the solve, so it is a lower-order,
  dt-lagged form of the same constraint — but robust on every solver (no tangent) and within ~1 cm of `implicit`,
  converging to it as `dt → 0`.
- **`off`** collects nothing; above-surface water piles up (hundreds of metres in a supply-rich basin). This is
  the nonphysical developer/diagnostic case (the former `-wtm_dev_allow_aboveground_water_columns`); it warns
  loudly.

In numerical terms this is the classic obstacle-problem split: `implicit` is the constraint solved *in* the
nonlinear system (active-set / complementarity), `explicit` is *solve-then-project* onto the feasible set.

## The band sink (taper 1) — RETIRED

The former default was the **taper-1 sub-surface sink** (`-wtm_surface_sink`): a smooth removal in a band of
width `2·qmax·dt` *below* the surface that held the table strictly sub-surface (`wtd < 0`) so no cell ever
crossed the free boundary. It dodged the exfiltration constraint rather than enforcing it — keeping the solve
smooth (differentiable for Picard/Newton) and 2nd-order (the BDF2-on-V "no-crossing" regime). The cost is that
the equilibrium table sat in a **dt-scaled band**, so it was **dt-dependent** (Issue #6).

**Retired 2026-09-01 (Issue #7)**, with `collection.method: legacy` and the three surface flags. Two things
turned out differently from the plan in that issue:

1. The replacement is **`active_set`, not `implicit`.** #7 described `implicit` as the dt-independent exact
   face; it is not — its retained head is ~linear in `dt` (1.97 / 0.68 / 0.34 m at `dt` = 1, 1/3, 1/6 week,
   FSM off), and with FSM on the *lake count* moves with `dt`. Retiring the taper in favour of `implicit`
   would have swapped one dt-dependence for another.
2. It was **not a golden regold.** Golden configs select no collector, so they already resolved to the
   default, and the selector had been forcing the sink off for them.

The semismooth active-set Newton that #7 prescribed as the way to "let the taper go for good" is what shipped,
and is what made the retirement safe.

## A CLI hazard worth knowing

`-wtm_surface_sink 0` does **not** reliably mean "sink off" — that CLI form mis-parses (it can leave the surface
unmanaged, so water piles). Use `-wtm_surface_sink false`, or better, drive the choice through
`runoff_collector`, whose test asserts each mode by the config key and so cannot be fooled by the `0` form.

## Measured: `implicit` is dt-DEPENDENT, and it propagates into lake depth

**Date:** 2026-08-25 · island fixture (117×75), cold start, TR-BDF2, 250 weeks of simulated time held
constant while only `dt` changes.

The `implicit` siphon removes above-surface water at rate `max(0,wtd)/dt`, so at steady state the
retained head balances inflow against a `1/dt` conductance: **`wtd_above ∝ dt`**. With FSM **off**
(no routing, so the face is measured alone):

| `dt` | max wtd [m] | ratio vs 1 week | ideal if ∝ `dt` |
|---|---|---|---|
| 1 week | 1.96615 | 1.000 | 1.000 |
| 1/3 week | 0.68122 | 0.346 | 0.333 |
| 1/6 week | 0.34425 | 0.175 | 0.167 |

Essentially linear. This confirms the mechanism already noted in `tests/dt_sensitivity` and
**contradicts the earlier "dt-independent, exact" claim in this document**, now corrected above.

With FSM **on**, that dt-dependent excess is what FillSpillMerge routes, so **lake depth inherits the
dependence** — a ~1.6 m face artifact becomes a ~3.4 m difference in modelled lake depth:

| `dt` | `implicit` max wtd [m] | `-wtm_active_set` max wtd [m] |
|---|---|---|
| 1 week | 5.3776 | **5.6986** |
| 1/3 week | 2.4962 | **5.6986** |
| 1/6 week | 2.0171 | **5.6986** |

**The active-set exfiltration constraint removes it completely**: max &#124;Δwtd&#124; = 1.1e-3 m
(rms 6e-5 m) across a 6× `dt` range, and it agrees with the value the well-converged fixed-dt
Anderson and Picard+T̄ runs reach (5.6986 m). It was also *cheaper* here — 957 vs 1771 SNES
iterations at `dt` = 1 week.

### Why this matters beyond the collector

This is what made **TR-BDF2 + adaptive dt** look like it was regressing in
`benchmark/scheme_bench`. The controller subdivides each week into ~3 sub-steps and **rejects
nothing** — it is behaving correctly. It simply lands on the small-`dt` branch of a model whose
equilibrium lake depth depends on `dt` (adaptive: 2.37 m, right on the `dt` = 1/3 week line at
2.50 m). **Adaptive was the messenger, not the fault.** Any scheme that shortens the step will
disagree with the production `dt` = 1 week answer for as long as `implicit` is the default.

## Measured: why the Picard/Newton fallback is `explicit`, not `implicit`

`implicit`'s tangent IS wired into the Picard operator (only the Newton Jacobian lacks it), so
`implicit` was a live option for the Picard fallback. It was not chosen, and the reason is measured
rather than inherited from this document's earlier claims.

Multi-lake fixture (`tests/multilake`), same 150 yr of simulated time, `dt` halved from 0.5 to 0.25 yr:

| enforcement | multi-cell lakes at `dt` = 0.5 → 0.25 yr | |
|---|---|---|
| `active_set` | 4 → 4 | topology stable |
| `implicit` | **6 → 5** | **topology changes** |
| `explicit` | 6 → 6 | topology stable |

`implicit` does not merely shift the lake *stages* with `dt` — it changes how many lakes there are,
because its `dt`-proportional retained head is routed by Fill-Spill-Merge into a different set of
depressions. `explicit` is `dt`-stable in topology, so it is the better fallback for the solvers that
cannot carry the active-set pin.

**Honest caveat:** `explicit` and `active_set` do not agree with each other on the answer (6 lakes vs
4 on this fixture). `explicit` is the best *available* enforcement on Picard/Newton, not an equivalent
one. Closing that gap needs the active-set tangents for those paths.
