# Design Note: an order-preserving *smooth surface sink* for the water-table free boundary

**Date:** 2026-07-27
**Branch:** `bdf2-adaptive-dt`
**Status:** design + math, validated in principle; prototype (residual + Jacobian) next.
**Companions:** `BDF2_RECHARGE_ORDER.md` (why recharge dropped BDF2-on-V to 1st order — the
free boundary), `BDF2_ADAPTIVE_DESIGN.md`, `PICARD_MATH.md`.

## 0. The general lesson (why this note is worth keeping)

A *hard constraint* in a time-dependent finite-difference / finite-volume solution — an obstacle
(`h ≤ z`), a contact condition, a sharp phase boundary — generically **reduces the temporal order of
accuracy to 1**, no matter how good the integrator. The reason is not the discretization: it is that
the *exact* solution loses smoothness (develops a corner in time) at the instant a cell activates the
constraint, and every multistep/Runge–Kutta order estimate rides on a derivative that becomes
unbounded there.

The general remedy — the thing worth remembering — is to **regularize the hard constraint into a
smooth relaxation folded into the implicit operator**, trading a small, `Δt`-independent *modeling*
error for restored order *and* better conditioning. This is the same idea as penalty / smoothed
methods for variational inequalities, smoothed contact mechanics, and phase-field regularization of
sharp interfaces. Below it is worked out concretely for the WTM water table crossing the land
surface, with WTM's own parameters, but the pattern transfers directly to other hard boundaries.

## 1. The problem: the water table as a moving obstacle

WTM stores a groundwater volume `V(h)` per cell and integrates (BDF2-on-V; see `BDF2_ADAPTIVE_DESIGN.md`)

```math
\frac{\mathrm{d}V}{\mathrm{d}t} \;=\; \nabla\!\cdot\!\big(T\,\nabla h\big) \;+\; R \;-\; Q ,
```

with head `h`, water-table depth `wtd = h - z` (`z` = land surface), transmissivity `T`, recharge
`R`, and a surface-water removal `Q`. In **extended-soil** mode the storage is linear everywhere,
`V = \phi\,h` with porosity `\phi` (no jump to `\phi=1` at the surface) and `T` continues smoothly
past `wtd=0` (no clamp) — this is what removes the *coefficient* nonsmoothness. See
`BDF2_RECHARGE_ORDER.md` §15.

Production still has to dispose of water that rises above the surface (to FillSpillMerge, or removed
Fan-style). The status quo does this as a **hard clip** applied *between* solves:

```math
h \leftarrow \min(h,\; z), \qquad
\text{removed volume} = \phi\,(wtd)_+ \;\;\text{(to FSM, or discarded).}
```

## 2. Why the hard clip is 1st order — and why that is fundamental

Take a single cell, drop lateral flow, constant recharge `R`. With `V=\phi h` the free rise is
`h(t) = h_0 + Rt/\phi`; the hard clip pins it at the surface, so the *exact* solution is

```math
h(t) \;=\; \min\!\Big(h_0 + \tfrac{R}{\phi}\,t,\;\; z\Big).
```

At the crossing time `t_\*` this has a **corner**: `h'` jumps from `R/\phi` to `0`, so `h''` carries a
Dirac and `h'''` is worse. BDF2's local truncation error is `\propto \Delta t^{3} h'''(\xi)`; the step
straddling `t_\*` therefore contributes `O(\Delta t)`, and the global order collapses to **1**. Because
this is a property of the *solution's regularity*, no time integrator — not even an exact
variational-inequality / obstacle solve — is globally 2nd order across the corner.

Measured on the 128² fixture (`benchmark/picard/recharge_free_boundary.py`), truncation caps the
order at ~1 at **every** cadence; the extended-soil benefit only survives *between* truncations:

| configuration | truncation cadence | err @ Δt=1 yr | order |
|---|---|---|---|
| extended soil, truncate **every step** | every step | 1.71 mm | ~1.0 |
| extended soil, truncate every 10th step | every 10 steps | 0.63 mm | ~1.0 |
| extended soil, **no** truncation | never | 0.0019 mm | **~2.0** |

## 3. The fix: replace the clip with a smooth implicit sink

Fold the surface removal into the operator as a smooth outflow that switches on as the water table
rises above the surface:

```math
Q(h) \;=\; \lambda\,\phi\,S_\varepsilon(wtd), \qquad
S_\varepsilon(x) \;=\; \varepsilon \ln\!\big(1 + e^{x/\varepsilon}\big)
\;\xrightarrow[\varepsilon\to0]{}\; (x)_+ ,
```

with a removal rate `\lambda` `[\mathrm{s}^{-1}]` and a head smoothing scale `\varepsilon` `[\mathrm{m}]`.
`S_\varepsilon` (softplus) is `C^\infty` for `\varepsilon>0`. The local balance becomes

```math
\phi\,\frac{\mathrm{d}h}{\mathrm{d}t} \;=\; R \;-\; \lambda\,\phi\,S_\varepsilon(h - z).
```

Now the right-hand side is `C^\infty` in `h` and as smooth in `t` as `R(t)`, so `h(t)` is smooth —
**the corner is gone and BDF2 is 2nd order again.** The removed water is a genuine flux integrated by
the same 2nd-order scheme (see §6), not an abrupt post-hoc deletion.

## 4. Two regimes, both 2nd order; the price is a thin overshoot

- **Quasi-steady overshoot.** Below the surface, `S_\varepsilon\approx0` and the cell rises freely.
  Above it, it settles where inflow balances outflow, `R = \lambda\phi\,(h_{ss}-z)`, i.e.

  ```math
  \boxed{\,h_{ss} - z \;=\; \delta \;=\; \dfrac{R}{\lambda\,\phi}\,}.
  ```

  The surface floats a hair *above* `z` by `\delta`. This is the **entire** price, and it is a
  *modeling* error **independent of `\Delta t`** — the temporal order stays 2; `\lambda` tunes fidelity.

- **Stiffness is harmless to order.** A sharp surface (large `\lambda`, relaxation time
  `\tau=1/\lambda \ll \Delta t`) is stiff, but the *implicit* solve simply tracks the quasi-steady
  manifold `h_{ss}(t) = z + R(t)/(\lambda\phi)`, which is smooth in `t` — still 2nd order. Stiffness
  costs conditioning, not order (and the sink *improves* conditioning; see §5).

## 5. Jacobian: SPD-preserving and stabilizing

The implicit operator gains one diagonal term,

```math
\frac{\partial Q}{\partial h} \;=\; \lambda\,\phi\,S_\varepsilon'(wtd)
\;=\; \lambda\,\phi\,\sigma\!\big(wtd/\varepsilon\big), \qquad
\sigma(u) = \frac{1}{1+e^{-u}}\in(0,1),
```

with `\sigma` the logistic. It is **smooth, non-negative**, and adds to the storage diagonal —
so it is **SPD-preserving** and makes the operator *more* diagonally dominant exactly above the
surface, where the old free boundary hurt conditioning most. Its tangent is exact (no secant
approximation — the same discipline as BDF2-on-V's `\mathrm{d}V/\mathrm{d}h`). Drop-in points: the
residual in `FormFunctionLocal` and the operator diagonal in the Picard assembly.

## 6. Mass balance and the data flow (does it need an extra array?)

The removed volume over a step is the implicit flux `\;Q(h^{n+1})\,\Delta t = \lambda\phi\,
S_\varepsilon(wtd^{n+1})\,\Delta t\;`, evaluated at the new state — so the bookkeeping is itself
**2nd order**, with no water lost to an abrupt clip. Where it goes splits exactly on FSM:

- **No-FSM (Fan-style, `evap_mode 0`):** the flux is discarded. **No array** — only a scalar
  mass-balance accumulator (`total_surface_removed`), paralleling `total_loss_to_ocean_gw`.
- **FSM path:** accumulate the per-cell flux across the cycle's `maxiter` substeps into **one
  per-cell array**; that sum is FSM's surface-water input for the cycle. It is a DMDA-**distributed**
  array (owned cells only), gathered to rank 0 once per cycle like the existing surface-water
  handoff — so *no new full-grid replicated array* (which matters: replicated memory is the
  global-scale bottleneck). Conceptually it is a second "runoff" channel — water that left *upward*
  — and slots in beside `runoff_dist_vec`.

The change vs today is *when* the water is counted: not as a standing pool at cycle end (`wtd>0`),
but as the flux that left continuously during the solve, leaving `wtd \approx z` behind.

## 7. How thick is the overshoot layer? (WTM parameters)

A natural non-dimensional choice ties the removal rate to the substep, `\lambda = C/\Delta t`
(`C = \lambda\Delta t` = number of e-foldings of removal per substep), giving

```math
\delta \;=\; \frac{R}{\lambda\phi} \;=\; \frac{R\,\Delta t}{\phi\,C}
\;=\; \frac{1}{C}\cdot\underbrace{\frac{R\,\Delta t}{\phi}}_{\text{one substep's recharge rise}} .
```

So the water table is held within *one relaxation-window's worth of recharge rise* of the surface.
With WTM production values — `\Delta t = 86{,}400\ \mathrm{s}` (1 day), `\phi = 0.25`,
`R` net recharge, `C=1` (loosest) and `C=10`:

| `R` (m/yr) | `\phi` | `\delta`, `C=1` | `\delta`, `C=10` | `\delta`, weekly relax (`C=1`, `\lambda=1/(7\Delta t)`) |
|---|---|---|---|---|
| 0.1 | 0.25 | 1.1 mm | 0.11 mm | 7.7 mm |
| 0.3 | 0.25 | 3.3 mm | 0.33 mm | 23 mm |
| 1.0 | 0.25 | 11 mm | 1.1 mm | 77 mm |
| 2.0 | 0.25 | 22 mm | 2.2 mm | 153 mm |
| 2.0 | 0.05 | 110 mm | 11 mm | 767 mm |

For realistic net recharge the layer is **millimetres to ~1 cm**; only the extreme corner (very wet
into very low porosity, loosest `\lambda`) reaches ~10 cm, and even that shrinks tenfold at `C=10`.
Against WTM's topography (metre-scale relief, ~arc-second cells) this is **negligible** — the layer
is far thinner than the vertical resolution at which the surface itself is known. Recommended default:
relax per substep at `C \approx 1`–`10` (i.e. `\lambda \approx (1\text{–}10)/\Delta t`), leaning
higher where memory/conditioning allow, so `\delta` stays sub-centimetre.

## 8. Interaction with evaporation (important)

The overshoot layer is a **numerical stabilisation artifact, not a real pond**, so open-water
evaporation must **not** act on it — otherwise the `wtd>0 \Rightarrow R = (\text{precip} -
\text{open\_water\_evap})` branch would (a) spuriously evaporate the stabilising water and (b)
mis-attribute numerical overshoot as real open-water loss. Two design rules follow:

1. **Physical surface water = the sink flux, not the standing overshoot.** Evaporation / FSM should
   act on the *accumulated removed flux* (§6), which is the water that genuinely left the column —
   not on the thin `\delta`-layer left in `wtd`.
2. **Treat the overshoot as `wtd \approx 0` for the recharge/evap decision.** Because `\delta` is
   sub-centimetre (§7), clamping the evap-mode test to `\max(wtd - \delta, 0)` (or simply using the
   below-surface recharge branch whenever `wtd \lesssim \delta`) is safe and keeps the recharge
   partition physically correct. `evap_mode 0` needs no special case — it discards the flux, so no
   open-water evaporation is applied at all, consistently.

This is the one behavioural subtlety to get right in the prototype; the thickness estimate in §7 is
what makes "ignore the overshoot for evaporation" defensible.

## 9. Proposed parameters and defaults

| symbol | flag (proposed) | meaning | default (proposed) |
|---|---|---|---|
| `\lambda` | `-wtm_surface_sink_rate` | removal rate `[1/\mathrm{s}]`, or as `C=\lambda\Delta t` | `\lambda = C/\Delta t`, `C\approx`5 |
| `\varepsilon` | `-wtm_surface_sink_width` | head smoothing scale `[\mathrm{m}]` of `S_\varepsilon` | a few cm (≥ `\delta`) |

The sink is meaningful only with extended-soil (it needs the smooth above-surface operator to be
well-posed), so it is enabled together with `-wtm_extended_soil`. Whether extended-soil + sink becomes
the model **default** (Andy's intent) is a separate, wide-blast-radius step (it rebaselines the golden
tests and requires truncation always active) — to be done deliberately, with diffs shown, after the
prototype confirms the order-2-with-removal result below.

> **Note on the smoother (Andy, 2026-07-27):** the square-root / conic form is *not* an external
> convention — it was introduced by this project's own recent transmissivity-smoothing work, so the
> choice of `S` is genuinely open. Softplus (§3) is used above for the derivation; the
> **compact-support** variant in §11 is now the leading candidate, for the reasons there.

## 11. Leading candidate: a *sub-surface* compact-support ramp (keep `wtd ≤ 0` always)

**The thought (Andy, 2026-07-27) — and why it simplifies the whole task.** Instead of letting the
water table rise *above* the surface and removing the overshoot, place a **compact-support
piecewise-polynomial ramp** in a thin band *below* the surface, `wtd \in [-w, 0]`, that removes water
ever faster as the table approaches the surface. Sized so its maximum removal outpaces the recharge
influx, it **holds the water table strictly below the surface** while the removed water flows on to
fill the FillSpillMerge reservoirs — *real* water, correctly routed.

```math
Q(wtd) = \lambda\,\phi\,g_w(wtd), \qquad
g_w(s) =
\begin{cases}
0, & s \le -w \\[2pt]
p\!\big((s+w)/w\big), & -w < s < 0 \quad (p:\text{ } C^2\text{ ramp, } p(0)=0,\ p(1)=1,\ p'(0)=p'(1)=0) \\[2pt]
1, & s \ge 0
\end{cases}
```

with, e.g., the quintic smoothstep `p(u)=u^3(6u^2-15u+10)` (`C^2`, compact). Equilibrium sits at some
`wtd = -\delta' \in (-w, 0)` where `\lambda\phi\,g_w(-\delta') = R` — **strictly below the surface**.

**Three consequences, in order of importance:**

1. **It may remove the need for extended-soil entirely — the real simplification.** The order-1
   disease was *crossing* `wtd=0` (the storativity jump `\phi\to1`, the `T`-clamp, the corner). If the
   ramp keeps every cell *below* the surface, the model **never touches that free boundary** — it lives
   permanently in the smooth sub-surface regime where storativity `=\phi` and `T` is the ordinary Fan
   form. That is exactly the **Exp A ("no cell crosses → order 2")** regime of
   `benchmark/picard/recharge_free_boundary.py`, achieved *natively*. So this single mechanism could
   replace {extended-soil operator + above-surface sink + overshoot bookkeeping} with just {a
   sub-surface ET-like sink}, and still be 2nd order because the ramp is smooth (`C^2`) and the
   solution never develops a corner.

2. **It eliminates the evaporation caveat (§8) outright — not mitigates it, removes it.** With
   `wtd \le 0` always, there is **no above-surface water in the groundwater field**, so there is *no
   numerical layer for open-water evaporation to spuriously act on*. The `wtd>0` recharge/evap branch
   is simply never entered by the stabiliser. The physical surface water is exactly the accumulated
   sink flux handed to FSM.

3. **It unintentionally emulates a near-surface soil-water / evapotranspiration profile (Andy).** A
   removal that grows as the water table nears the surface is, physically, what near-surface
   **evaporation and transpiration** do — roots and capillary rise draw down water most strongly where
   the table is shallow. So the numerical stabiliser doubles as a crude ET parameterisation; `w` maps
   naturally to an effective rooting/extinction depth and could later be tied to real ET data. A
   "spoof" of physics that happens to point the right way.

4. **It removes extended-soil's one residual physical risk: wrong-direction flow (Andy).** Extended
   soil lets water stand *above* the surface, and that fictional above-surface head participates in
   the lateral GW flow (`T` continues past `wtd=0`). The extended-soil note bounded — but could not
   zero — the chance that a tall enough mound reverses a gradient **across a topographic divide** and
   misroutes water (see `BDF2_RECHARGE_ORDER.md`, "Why it is physically safe"). With the sub-surface
   ramp there is *no* above-surface head anywhere, so `h \le z` at every cell and lateral flow only
   ever follows *real* sub-surface gradients — cross-divide misrouting is **impossible by
   construction**, not merely improbable. (Extended-soil was not a bad idea — it proved order 2 is
   reachable and pinned the cause to the free boundary; the ramp keeps that win and drops this risk.)

**Open questions to settle in the prototype (held honestly):**

- **Sizing / guarantee.** Keeping `wtd<0` requires `\max Q = \lambda\phi \ge R_{\max}` *within a
  substep's influx* — an extreme storm into low porosity could still punch through the band. Do we
  size `\lambda` for the worst case, adapt it, or keep a hard clamp / extended-soil as a **backstop**
  for the rare breach? (A breach is graceful — it just reverts locally to the §2–§9 above-surface
  behaviour — but we should know when it happens.)
- **Double-counting ET.** If `evap_mode`/FSM already account for evaporation, a sub-surface ET-sink
  that *also* removes near-surface water risks double-counting real ET. The sink's removal must be
  reconciled with (or replace part of) the existing evap term, not stack on top of it.
- **Mass balance & routing.** Verify the removed flux is conserved into FSM at the right cell, time,
  and cadence (the §6 accumulator), and that shunting it below-surface-directly (vs letting it pond
  first) gives the intended reservoir filling.
- **`w` and `p`.** Band width and ramp shape: numerically `w` a few `\delta` keeps the equilibrium
  comfortably inside the band; physically `w` = effective ET extinction depth. These two readings may
  or may not want the same value — worth checking whether one `w` serves both.

## 12. Status / next

- **Analysis:** order-2 recovery, overshoot `\delta = R/\lambda\phi`, SPD Jacobian — done (this note).
- **Leading design (§11):** sub-surface compact-support ramp keeping `wtd\le0`; likely obviates
  extended-soil, kills the evaporation caveat, and doubles as a crude ET profile. **Prototype this
  variant first.**
- **Prototype:** add `Q` and `\partial Q/\partial h` to the residual + Picard operator; measure (i)
  order *with active removal*, (ii) that `wtd` stays `<0` across the `R/\phi` range and cell sizes,
  (iii) mass conservation of the removed flux into FSM. Held skeptically: whether one global
  `(\lambda, w)` behaves across the domain, and how breaches (if any) present.
- **Then:** the FSM-input accumulator (§6), the ET reconciliation (§11), and — deliberately, diffs
  shown — the default flip.
