# Time-averaged interblock transmissivity (`-wtm_Tbar`)

## Motivation

The Fan et al. depth-integrated transmissivity is exponential in the water-table depth,
`T(wtd) = fdepth·ksat·exp((wtd+1.5)/fdepth)` in the deep regime, spanning ~7 orders of magnitude over
the column. That exponential is the dominant nonlinearity in the groundwater diffusion
`∂(S h)/∂t = ∇·(T ∇h) + R`, and it is what caps the usable time step:

- **Frozen-coefficient solvers (Anderson, Picard)** evaluate `T` at the *current iterate* — i.e. near
  the start of the step. On a stiff step the head moves far, so the true within-step transmissivity is
  poorly represented by its start-of-step value; the outer iteration then **oscillates / overshoots**
  and either stalls or diverges. This is the observed cold-start Picard hang (`benchmark/EQUILIBRIUM_ROBUSTNESS.md`).
- **Newton** linearizes `T` to first order (`T + T'Δ`), whose validity basin is small because the
  exponential curvature is large.

## The idea: average `T` over the step, not freeze it at the start

For the flux coefficient use each cell's **time-averaged transmissivity over the step** `wtd^n → wtd^{n+1}`,
rather than the instantaneous `T`. Because `∂Φ/∂wtd = T`, where `Φ` is the piecewise Kirchhoff
(discharge) potential `Φ(wtd) = ∫T dwtd` (already in the code as `dischargePotential`), the exact
wtd-average of `T` between the two states is the **Kirchhoff-potential difference**:

```
T̄ = (Φ(wtd^{n+1}) − Φ(wtd^n)) / (wtd^{n+1} − wtd^n)
```

Evaluated per cell with that cell's own `fdepth, ksat`, this is:

| regime | `T(wtd)` | `T̄` reduces to |
|---|---|---|
| deep (`wtd < −1.5`) | `fdepth·ksat·exp((wtd+1.5)/fdepth)` | the **log-mean** of `T^n, T^{n+1}` (exact, since `ln T` is linear in wtd) |
| soil (`−1.5 ≤ wtd ≤ 0`) | `ksat·(wtd+1.5+fdepth)` (affine) | the **arithmetic mean** |
| surface (`wtd > 0`) | `ksat·(1.5+fdepth)` (constant) | the **constant** surface `T` |

and it is **continuous (C1)** across the `−1.5` and `0` regime boundaries (Φ is C1 there), so there is
no kink to reintroduce. As `wtd^{n+1} → wtd^n` (small step / steady state) `T̄ → T`, so the method
changes nothing at equilibrium — the steady state is identical.

## What changes, and what does not

`-wtm_Tbar` changes **only** the per-cell transmissivity that feeds the (unchanged) **harmonic**
interblock mean `e = 2/(1/T̄_i + 1/T̄_j)`. The spatial discretization — harmonic face averaging, the
conservative finite-volume flux, the storage term — is untouched. This is deliberate: the spatial role
(neighbours = different media, in series → harmonic) and the temporal role (`T` evolving over the step →
time-average) are **separate**, and only the temporal one is changed. Same physics, same equilibrium,
better-conditioned transient steps.

It composes with **any** solver because it is a residual-level change:
- **Anderson** (matrix-free) evaluates the T̄ residual directly.
- **Picard** (BDF2-on-V) assembles its operator with the T̄ face conductance.
- **Newton** additionally needs the tangent `d(1/T̄)/dw = −T̄'/T̄²`, with `T̄' = [T(w) − T̄]/(w − w^n)`
  (Δ→0 limit `T'(w)/2`). Because T̄ is built from the piecewise `T`, this is the *exact* analytic
  tangent of the T̄ residual (FD-verified to 1e-9 in isolation).

The water-budget ocean-outflow accounting rebuilds the same T̄ face conductance, so the budget closes
consistently on the T̄ path.

## Requirements / restrictions

`Φ` is the antiderivative of the **piecewise** Fan `T`, so `-wtm_Tbar` requires the piecewise form and
is refused with `-wtm_ksat_*_smoothing_width`, `-wtm_extended_soil`, or `-wtm_kirchhoff` (which redefines
the solve variable). A neighbour's T̄ needs its `w^n`, so `w^n` is ghost-scattered once per solve into
`starting_wtd_local`. Off by default → the production path is byte-identical.

## Correctness verification

- **Jacobian tangent** FD-checked in isolation: relerr 7e-11 (deep), 1.4e-9 (soil), 8.8e-10 (deep↔soil
  crossing). Whole-Jacobian `-snes_test_jacobian` ratio 6.4e-3 = identical to the baseline's (the shared
  piecewise-kink + secant-storativity inexactness, not a T̄ error).
- **MPI**: serial vs 4-rank agree to 8.4e-7 m (Anderson) / 2.3e-5 m (Picard) — the cross-rank
  reduction-order floor. The ghost `w^n` scatter is correct.
- **Equilibrium**: Anderson±T̄ reach the same steady state (max 0.10 m at 60 cycles, still settling; T̄→T
  at steady state).

## Empirical results

See `benchmark/tbar_suite/` for the driver. Results table appended by the suite run (see the session
notes / memory `finding-logmean-transmissivity`).

<!-- RESULTS_PLACEHOLDER: cold-start dt sweep, warm perturbation ceilings, Esquibel headline, wall-clock vs Anderson -->
