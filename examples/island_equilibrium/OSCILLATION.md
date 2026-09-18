# Why corsica never reaches equilibrium

**Status: the behaviour is fully characterised and the mechanism is NOT known.** Six candidate
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
| 7 | the lagged FSM→recharge source (`continuous` feeds step *n*'s delta to step *n+1*) | measured directly from the two raster series: the step-to-step relative change in that source is **1.58e-06** (median, settled), against a 24 m groundwater swing. It decays ~3× per step from a 17.6% transient peak. FSM's output is constant because the lakes are — volume 4751.58 → 4751.71 over 54 steps |
| 6 | a lagged nonlinear coefficient | `benchmark/twocell_numerics` shows a lagged `T` **does** manufacture a limit cycle — but on the real model a 10 000× tighter solve (`water_volume_tol` 1e-8→1e-12, `residual_gate` 1e-5→1e-9, both confirmed resolved) leaves it at **24.4915 → 24.4914 m, ratio 1.0000** |

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

Everything that settled omitted the same things. In the order I would take them:

1. **FSM's surface routing of the `runoff_ratio` share downhill.** Half the net `P − E` (0.06 m/yr)
   leaves as runoff and is redistributed by FSM. No reduction here modelled that transport at all.
   Note this is the **transport**, not the one-step lag in handing it over — the lag is excluded above
   (row 7). What is untested is that half the water budget moves overland and no reduction had it.
2. The ET tapers and the open-water switch.
3. Full-domain boundaries rather than a fixed ring.

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
