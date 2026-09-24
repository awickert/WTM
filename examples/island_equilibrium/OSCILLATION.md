# Why corsica never reaches equilibrium

**Status: the behaviour is fully characterised and the mechanism is NOT known.** SEVEN candidate
mechanisms have been excluded by measurement and five explanations of my own have been refuted. This
document records all of it, because the exclusions cost a measurement each and are the part worth not
repeating.

Measured 2026-09-17/18 on `examples/island_equilibrium`, corsica at 30" with DEM-derived slope.
Reproduce with `demo.py corsica --solver anderson --equilibrium`.

## The behaviour

Running to equilibrium, corsica reports **did not settle**, however long it is given. Over a full
20 000 yr run the stop metric (fraction of land cells moving more than 1 mm of water per cycle) bottoms
out at **frac = 0.001351** against its 0.001 threshold and never goes below.

**32 of 14 064 land cells** never quiet — 0.23% of the domain. They are the high, steep ones:

| | offenders | all land |
|---|---|---|
| median topography | 1172 m | 428 m |
| median slope | 0.264 | 0.134 |
| implied `fdepth` | ~5 m | 9.5 m |

Two clusters: rows 119–123 × cols 61–65, and rows 98–99 × cols 75–77. Within each, 2–3 cells actually
drive and the rest are neighbours responding.

**It is a true limit cycle.** The orbit in (`wtd_A`, `wtd_B`) returns to within a mean 3.0 m after 21
reports, against amplitudes of 24.5 m and 13.1 m. **Period 210 yr.** The driver fills for ~100 yr,
turns smoothly at about −0.06 m, plunges up to 24.5 m over ~60 yr, and refills. The two cells run
antiphase (corr −0.410 at a −6 report lag, about half a period).

**Read it with the median, not the max.** The rms is ~100× smaller than the max and that ratio is
stable over 2000 cycles. Almost the whole domain is quiet. `dvol_max` alone reads as "corsica never
equilibrates", which is wrong and alarming; the max/rms contrast is what shows it is 32 cells.

## What controls the coupling (measured, and correct as far as it goes)

Flux between the driver (1084 m) and its downhill neighbour (874 m) is `T_harmonic × head-drop`, and
only one factor moves:

```
head DROP  : 202 .. 233 m        varies 1.16x     corr(flux, drop)   = -0.444
T_harmonic : 7.5e-6 .. 6.1e-5    varies 8.1x      corr(flux, T_harm) = +0.998
```

The gradient is pinned by the **210 m topographic drop**, so a 24 m water-table swing perturbs it by
16% — and in the wrong direction. The harmonic mean is dominated by the low side, which is the
**downhill** cell on 37 of 41 reports; its `T` varies 17× over the cycle.

**This is a statement about the coupling, not an explanation of the oscillation.** It was promoted to a
mechanism and refuted — see R5.

**Above-surface water does not conduct.** `transient_groundwater.cpp:115` clamps `T` to its `wtd = 0`
value above ground, because surface water moves in FillSpillMerge rather than by Darcy flow. And the
clamp never engages here: 0 of 41 reports have the driver above ground.

## Excluded by measurement

| # | candidate | how it died |
|---|---|---|
| 1 | the metric read pre-FSM state | a real defect, found here and fixed separately (`92051e4`) — but pre- and post-FSM metrics agree to **1e-12** on this run |
| 2 | FSM outlet switching (`#64`'s mechanism) | lake cell count 254 and lake volume 4751.7081 m **bit-identical** for 60 consecutive cycles |
| 3 | time discretisation | 4× `dt` refinement from an identical restart moves the amplitude by **≤0.5%** (ratios 0.995 / 0.997 / 1.000) |
| 4 | operator splitting | ruled out by the same sweep once measured: `FSMTRACE` shows FSM runs once per **step** (40 calls in 40 steps), so refining `dt` refined the coupling 4× too |
| 5 | which removal mechanism | `active_set` / `explicit` / `implicit` give spans 24.4966 / 24.4085 / 24.4877 m, same phase, same 8 of 41 reports at the surface |
| 6 | a lagged nonlinear coefficient | `benchmark/twocell_numerics` shows a lagged `T` **does** manufacture a limit cycle — but on the real model a 10 000× tighter solve (`water_volume_tol` 1e-8→1e-12, `residual_gate` 1e-5→1e-9, both confirmed resolved) leaves it at **24.4915 → 24.4914 m, ratio 1.0000** |
| 7 | the lagged FSM→recharge source (`continuous` feeds step *n*'s delta to step *n+1*) | measured directly from the two raster series: the step-to-step relative change in that source is **1.58e-06** (median, settled), against a 24 m groundwater swing. It decays ~3× per step from a 17.6% transient peak. FSM's output is constant because the lakes are — volume 4751.58 → 4751.71 over 54 steps |

## Refuted explanations of mine

- **R1 — open-water evaporation exceeding precipitation.** The forcing does change sign at the surface
  (+0.060 m/yr below, −0.080 m/yr at it), and that is real. It is not the driver: **halving** `E_ow` to
  0.15, below `P`, leaves the span at 24.4915 → 24.5777 m. I published this on the strength of a
  fill-limb timing prediction that matched (102 yr against 110 measured) without testing the mechanism
  itself — and that prediction depends on the *below-surface* balance, the half the ablation did not
  touch.
- **R2 — `active_set` failing on steep terrain.** Not flicker (a smooth 210 yr cycle, not per-cycle
  alternation), and `active_set` is not implicated: `explicit` and `implicit` give the same cycle.
- **R3 — the head cap forbidding an equilibrium.** A 10 cm ponding allowance patched into the live
  obstacle writer changes the span by 0.003%, **and the cell never uses it** — still turning at
  −0.06 m. The obstacle was never binding.
- **R4 — the extinction-depth taper.** 1 m / 8 m / 30 m give **bit-identical** results. `A(wtd)`
  multiplies only `(E_eff − P)₊`, which is identically zero here (`P` 0.22 > `E_soil` 0.10). Correct
  behaviour, not an inert key — it would bite on an arid fixture.
- **R5 — the two-cell valve.** A faithful ODE reduction does not oscillate. Built with the model's own
  forms — piecewise `T`, harmonic-mean interfaces, 5-point stencil, `specificYield` storage, real
  topography, real `fdepth`, real cell sizes, surface cap as the collector, outer ring Dirichlet at
  measured time-means:

  ```
  5x5  block:  driver -> -19.860, neighbour -> -14.383   span 0.001 m
  11x11 block: driver ->  -6.342, neighbour -> -13.227   span 0.031 m
  ```

  Both converge **monotonically** to a stable fixed point, with the whole oscillating cluster strictly
  interior in the 11×11.

## The one thing that changes it

**Surface removal is necessary.** With `routing: off` *and* `collection.method: off`, the driver rests
at +0.042 m, the downhill cell stays dry (`T` 1.4e-6 … 3.9e-6, varying only 2.9×), and nothing cycles.
**Either remover alone restores the full cycle** — FSM off with the collector on gives 24.66 m, FSM on
with the collector off gives 24.38 m.

Nothing else varied has ever moved the amplitude by more than 1%.

## What is still untested

Two of the three candidates below are now dead, and a fourth has taken their place. Struck through
with the evidence, so neither is re-walked:

1. ~~**FSM's surface routing of the `runoff_ratio` share downhill.**~~ **REFUTED by the table above,
   without a new run.** "Either remover alone restores the full cycle — FSM off with the collector on
   gives 24.66 m". With FSM off there is no overland transport at all, and the full cycle persists.
   Transport cannot be necessary. What is necessary is a **local sink at the surface**.
2. The ET tapers and the open-water switch — and see the vacuity note below, which changes what an
   ablation here can mean.
3. ~~**Full-domain boundaries rather than a fixed ring.**~~ **REFUTED by measurement 2026-09-24.**
   Peak-to-peak amplitude over 3 full periods (63 reports, 19 380–20 000 yr), by Chebyshev distance
   from the nearer cluster centre:

   | distance | land cells | max amplitude |
   |---|---|---|
   | 0–2 | 45 | 24.4833 m |
   | 3–5 | 176 | 0.2170 m |
   | 6–10 | 608 | 0.0004 m |
   | 11–20 | 1914 | 0.0000 m |

   The 11×11 block puts its ring at distance 5, where the real field moves by at most 0.217 m against
   24.5 m in the middle. The ring genuinely does not move; fixing it at time-means cannot be what
   suppressed the cycle.

## THE OSCILLATING CELLS NEVER REACH THE SURFACE, and that makes several ablations vacuous

Measured on the same 63 reports. Of the 22 cells with amplitude > 0.1 m, **one** ever attains
`wtd >= 0`. Both drivers turn just short of it and stay there:

    driver A (120,63)   amplitude 24.483 m   wtd range -24.545 .. -0.0621
    driver B (98,75)    amplitude  9.900 m   wtd range  -9.970 .. -0.0701

**So any ablation acting only on surface water could not have moved this cycle, and its null result is
not evidence about the mechanism.** That covers R1 (halving open-water evaporation — `E_ow` applies to
water these cells never have) and R3 (the ponding allowance, where the document already noticed "the
cell never uses it" without drawing the conclusion). Both should be read as *inconclusive*, not as
refutations. R4 and rows 2–6 are unaffected; they ablate things that act below the surface.

## What IS at the surface: a PINNED neighbour, downhill

The count above was nearly a fourth vacuous test of my own. A cell **pinned** at `wtd = 0` has zero
amplitude by construction, so filtering on "amplitude > 0.1 m" excludes exactly the cells the removers
are acting on. Asked without that filter — how many cells near each driver ever reach `wtd >= 0`,
whatever their amplitude:

    driver A: 1 such cell within 2 cells, 4 within 3, 13 within 8 -- every one at amplitude 0.0000 m
    driver B: 1 within 2, 4 within 3, 17 within 8

and every one of them sits **below** the driver, by 314 to 597 m of topography.

So the picture is a violently oscillating cell one or two cells away from a neighbour that the remover
holds at the surface and which never moves at all.

**HYPOTHESIS, stated as one and not yet tested.** This is a relaxation oscillator whose sink is the
pinned neighbour and whose valve is `T`. A cell held at `wtd = 0` by a remover is not a cell with a
cap on it — it is a **Dirichlet boundary at fixed head with unlimited capacity**, because whatever
arrives is taken away. Conductance from the driver to it goes as `exp(wtd / fdepth)` with
`fdepth ≈ 5 m` here. The driver rises, the valve opens, it drains into a sink that cannot fill, it
falls until the exponential shuts the valve — 24.5 m is ≈ 5 `fdepth` — and then refills over ~100 yr
on the +0.060 m/yr it gets below the surface.

If that is right it accounts for every measured fact already in this document: why **either** remover
suffices and neither is special (all three collectors pin identically); why removal is *necessary*
(with none, nothing is pinned, and the driver rests ponded at +0.042 m); why the offenders are the
high, steep, thin-`fdepth` cells (small `fdepth` is a sharper valve); and why the ODE reduction did
not oscillate (R5 gave the **driver** a surface cap, which is a limit on one cell, rather than giving
it a **neighbour pinned at zero with unlimited removal**, which is a boundary condition).

**The prediction that would falsify it:** rebuild the R5 reduction with the downhill neighbour held at
`wtd = 0` as a Dirichlet sink instead of capping the driver. Everything else unchanged. If it still
converges monotonically, this is wrong.

### TESTED THE SAME DAY, AND THE SIGN IS BACKWARDS — the hypothesis above is WITHDRAWN

`benchmark/twocell_numerics/twocell.py` already carries the test, and it already contained the
configuration the hypothesis proposed. Its baseline settles at

    hC = -30 (shipped):   A = -32.822   B =  0.0000   <- B IS PINNED AT THE SURFACE, and it is STABLE
    hC =   0 (pinned far
         boundary, swept
         -30/-10/-3/-1/-0.1/0):          every scheme, every hC: amplitude 0.0000

So a neighbour pinned at `wtd = 0` next to the driver is not what starts a cycle — in this reduction it
is what **ends** one. Pinning goes with stability, and sweeping the far boundary from 30 m below its
surface up to pinned at it produced zeros in explicit, lagged AND implicit at the shipped 1 yr step.

**The discriminator is the other way round, and it is sharper for it.** Compare the same two cells:

    reduction: B pins at 0.0000 and the system settles
    real:      B (121,63) cycles -23.767 .. -11.064 and NEVER comes within 11 m of the surface

The reduction does not fail by missing a sink. It fails by **over-filling B** — B has exactly one
outlet (to C), backs up, and pins, while the real (121,63) drains in four directions into cells 300 to
600 m lower and so never gets near the surface. The reduction settles in a regime the real cells never
visit, which is exactly the caveat twocell.py states about itself; what is new is knowing WHICH way the
mismatch runs and WHY.

### AND THE REDUCTION'S FIXED POINT IS NOT PHYSICAL — it is the discard absorbing a shut boundary

Andy, on the uncapped run piling to +283 m: *"Is this because you grabbed just two cells without
opening their other boundaries?"* Yes, and it is measurable.

B's only outlet is to C, and C is held at `hC = -30` — 30 m below its OWN surface — which throttles
the interface transmissivity:

    hC = -30 (shipped)   T_C = 9.824e-07    624x smaller than T at the surface
    hC =   0 (opened)    T_C = 6.130e-04

To pass `2R` (its own recharge plus A's) through the shipped outlet, B would need `h_B ≈ +1435 m`. No
physical head balances it, so uncapped B simply fills — monotone rising in all 1999 steps of the
second half, ending +283 m above ground. Open the outlet (`hC = 0`) and it settles at -19.7 m instead.

**With the cap on, that shut boundary is invisible, because the discard is doing the draining.** At
the settled state, in m/yr:

    recharge in (2R)          0.1200
    out via the outlet QBC    0.0167    14%
    out via the CAP           0.1033    86%
    balance                   0.0000

So the reduction's "stable fixed point at A = -32.8, B = 0" is not the local physics reaching
equilibrium. It is 86% of the water being thrown away at B's surface because the only way out is
closed. A fixed point held up by a discard is not evidence that the physics has one.

**That is the THIRD independent reason this reduction cannot speak to corsica**, and they stack:

1. B pins at the cap, so the system is scalar in `h_A` — and a scalar autonomous ODE cannot oscillate.
   The null was guaranteed before any physics was evaluated.
2. Two cells cannot carry a five-cell mode. The real oscillation is a stripe down column 63 —
   amplitudes 5.97 / **24.48** / 12.70 / 5.13 / 1.17 at rows 119–123 — peaking at A and decaying both
   ways. A and B are a two-cell slice through it.
3. Its equilibrium is maintained by discarding 86% of the inflow through a boundary that is shut.

None of these is about the physics being wrong. All three are about the REDUCTION being the wrong
shape, and each one alone is enough to void its result.

**So the open question is now specific:** what sustains a 24.5 m cycle in cells that are well drained,
never reach the surface, and sit next to pinned cells that do not move? An adequate reduction has to
reproduce a FREE, well-drained B — four outlets, not one — before its null means anything.

The implicit-solve candidate is closed by row 6 above.

## Caveats on my own numbers

- **The hand-rolled flux budget does not close.** Adding the downhill cell's recharge to its measured
  lateral net predicts it wets *without* removal, the opposite of the measured trajectory. The estimate
  is adequate for the 1.16×-versus-8.1× contrast and **not** for a budget. Re-derive from the model's
  own fluxes before building on it.
- **The two-cell reduction settles at `A = −32.8, B = 0`**, which is not where the real cells sit (they
  cycle 0 to −24.5). A null in the reduction does not prove a null in the model.

## No model change is implied

The behaviour is `dt`-invariant, coupling-invariant, collector-invariant, ponding-invariant, and
solver-tolerance-invariant. Nothing here says the model is computing incorrectly. `--equilibrium` on
corsica reporting "did not settle" is a true statement about this domain at this tolerance.
