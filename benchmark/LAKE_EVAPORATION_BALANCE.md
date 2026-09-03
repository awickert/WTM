# What lake evaporation does to the equilibrium water balance

**Date:** 2026-09-03
**Companions:** `WATER_BUDGET.md` (the budget and its columns), `SURFACE_SINK_DESIGN.md` §14
(the taper), `tests/lake_evap_equals_et` (the invariant this study leans on).

WTM has no switch between soil ET and lake evaporation. It blends them as a logistic in
water-table depth (`evaporation.et_sigmoid`),

```math
E_\text{eff}(w) \;=\; \mathrm{ET} \;+\; (\mathrm{owe}-\mathrm{ET})\,
\sigma\!\left(\frac{w - w_c}{s}\right),
```

so `owe` is not "the lake's evaporation rate" so much as the upper end of a continuum the
water table slides along. This note measures what moving that upper end does to the balance
a run settles into.

## The experiment

One fixture, six runs, and the ONLY field that differs between them is
`open_water_evaporation`. Topography (a 100 m plateau with a 10 m-deep off-centre pit),
mask, precipitation (1.0 m/yr), ET (0.2 m/yr), ksat, porosity and the supplied initial table
are written from the same arrays, so this is a same-method comparison rather than six
experiments. `active_set` collection, FillSpillMerge on, Anderson, fixed `dt` = 1 yr, 400 yr
to equilibrium — per-cycle storage change at the end is ≤ 0.011 % of cycle recharge in every
arm, and ≤ 0.0001 % in five of the six.

Terms are per 20 yr report cycle, in m³.

| `owe` | P in | evaporation | Darcy ocean | FSM spill | closure | evap as % of P | lake cells | max wtd (m) |
|---|---|---|---|---|---|---|---|---|
| 0.2 (= ET) | 3.47464e11 | 6.94928e10 | 6.71268e10 | 2.10845e11 | 1.7e-13 | 20.00 % | 16 | 10.0000 |
| 0.35 | 3.47464e11 | 8.90086e10 | 6.71263e10 | 1.91329e11 | −2.9e-11 | 25.62 % | 16 | 10.0000 |
| 0.5 | 3.47464e11 | 1.08197e11 | 6.71193e10 | 1.72148e11 | −1.4e-08 | 31.14 % | 16 | 10.0000 |
| 1.0 | 3.47464e11 | 1.71398e11 | 6.71074e10 | 1.08958e11 | −4.2e-08 | 49.33 % | 16 | 10.0000 |
| 2.0 | 3.47464e11 | 2.80422e11 | 6.70748e10 | **0** | 1.3e-05 | 80.71 % | 16 | 0.4898 |
| 4.0 | 3.47464e11 | 2.80439e11 | 6.70254e10 | **0** | −8.7e-09 | 80.71 % | 0 | −0.0599 |

The closure column is `P − (E + Darcy + spill + ΔS)` relative to P. That it sits at 1e-13 on
the first row is worth stating plainly: it is only measurable at all because the budget
baseline defect (`WATER_BUDGET.md` §4) was fixed first. Before that, the same quantity was
tens of percent and this study could not have been read.

## Three results

**1. Lake evaporation trades against the FSM ocean spill, essentially one for one, and
leaves groundwater outflow alone.** Referenced to the `owe == ET` arm:

| `owe` | Δ evaporation | Δ Darcy ocean | Δ FSM spill |
|---|---|---|---|
| 0.35 | +5.617 % of P | −0.000 % | −5.617 % |
| 0.5 | +11.139 % | −0.002 % | −11.137 % |
| 1.0 | +29.328 % | −0.006 % | −29.323 % |
| 2.0 | +60.705 % | −0.015 % | −60.681 % |

Darcy outflow moves by 0.15 % across a **20×** change in `owe`. The groundwater system barely
notices; what changes is which of the two *surface* exits the water leaves by.

**2. While the lake spills, its geometry is completely buffered.** For every arm from 0.2 to
1.0 the lake holds 16 cells at `max wtd = 10.0000 m` — exactly the pit depth, i.e. filled to
the brim and spilling. Evaporation up to five times ET does not lower it by a millimetre. A
spilling lake is pinned at its outlet: extra evaporation is paid for out of the spill, not
out of storage, and the balance absorbs it entirely through the routing term.

**3. The influence saturates once the lake stops spilling.** At `owe = 2.0` the spill reaches
zero, and from there evaporation cannot grow: 80.71 % of P at both `owe = 2.0` and `owe = 4.0`,
identical to four significant figures, with Darcy outflow making up the remaining 19.3 %.
Evaporation has become **supply-limited** at `P − Darcy`, and the water table, not the rate,
is what adjusts — the lake falls from 10 m to 0.49 m at `owe = 2.0` and is gone at `owe = 4.0`,
where the table settles 6 cm below ground. It settles *there* because the sigmoid is still
partly open at that depth (`w_c = 0.05`, `s = 0.1` puts `σ(−1.1) ≈ 0.25`), so a sub-surface
table can still deliver near-lake evaporation rates. The model self-regulates by depth once
it can no longer regulate by area.

## What to take from it

For a spilling lake, the lake-evaporation rate is a **routing** parameter, not a storage or a
water-table parameter: it decides how much of the surplus leaves as vapour and how much leaves
down the outlet, and it does not touch the water table at all. It becomes a first-order control
on state only once evaporation is large enough to close the outlet. Calibration effort spent on
`owe` is therefore effort spent on the surface-water partition, and is invisible in the water
table, right up to the point where it is abruptly the only thing that matters.

Two limits worth naming. This is one pit on one synthetic plateau: the saturation threshold
(`owe ≈ 2.0` here) is a property of this catchment's surplus, not a general number. And the
`owe = 4.0` arm shows the result depends on the sigmoid width as well as on `owe` — with a
narrower transition the sub-surface table could not sustain that evaporation, and the balance
after collapse would differ. Neither limit affects results 1 and 2, which are measured while
the lake spills.
