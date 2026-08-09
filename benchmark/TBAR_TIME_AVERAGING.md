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

Driver: `scratchpad/tbar_suite/` (cold dt sweep, warm perturbation, Esquibel headline). Test bed: the
ocean-ringed Esquibel island (75×117, 2544 land cells, ~615 m relief), 1-week base dt (Kerry's setting),
default tapers on. `T̄` on = `-wtm_Tbar`.

### Cold start to equilibrium — maximum stable dt (stability vs time-step size)

| solver | ceiling without T̄ | ceiling with T̄ |
|---|---|---|
| Anderson | 4 wk | **8 wk (2×)** |
| Picard (BDF2-on-V, default) | *fails at 1 wk* (frozen-coefficient hang, 10000-iter divergence) | **1 wk (rescued)** |
| Newton (`-wtm_stiff`) | reaches equilibrium but slow (per-iteration GMRES+GAMG cost; times out at 400 s where Anderson takes 2 s) | same, settles ~2 cycles sooner |

- **T̄ doubles Anderson's stable cold step (4→8 wk)** and **rescues the default Picard's cold start** at
  Kerry's 1-week dt (baseline Picard diverges cold; this is the Kerry-Picard-hang, cured).
- **Accuracy is unchanged by T̄.** At each dt Anderson and Anderson+T̄ reach the *same* equilibrium
  (2 wk: 4.735 vs 4.731 m; 4 wk: 11.664 vs 11.664 m max vs the Anderson@1wk reference). T̄ changes
  *convergence*, not the answer. (The equilibrium's own drift with dt — 0.17 m@1wk → ~12 m@4wk vs the
  fine-dt reference — is a pre-existing dt-dependence of the coupled GW↔FillSpillMerge steady state,
  shared by both, not a T̄ effect.)
- Iteration reduction grows with stiffness: ~4 % fewer Anderson iters at 1 wk, ~11 % at 4 wk.

### Warm-start perturbation (2× recharge from equilibrium) — step ceiling to failure

| solver | ceiling without T̄ | ceiling with T̄ | iters where both converge |
|---|---|---|---|
| Anderson | 4 wk | **8 wk (2×)** | 4 wk: 686 → 598 (~13 % fewer) |
| Picard (default) | 1 wk | **4 wk (4×)** | 1 wk: 200 → 141 (~30 % fewer) |

T̄ roughly **doubles Anderson's and quadruples the default Picard's** stable step under perturbation, and
cuts the iteration count where both converge.

### Wall-clock vs Anderson

The step-ceiling and iteration-count wins do **not** make the implicit solvers beat Anderson on
wall-clock in the small-dt regime: Anderson's matrix-free iterations carry no linear solve (~2 s for a
full island equilibrium), whereas each Picard/Newton iteration pays a preconditioned solve (Picard+T̄ ~26 s,
Newton times out at 400 s for the same equilibrium). T̄'s value is **robustness and step size** (bigger
stable steps, fewer nonlinear iterations, curing the cold-Picard hang), not making Picard/Newton faster
than Anderson at small dt. Applied to Anderson itself, T̄ is a pure win: same cheap iterations, 2× the
stable step, identical equilibrium.

### Esquibel headline — real 384k-cell patch (Kerry's domain, 166k land cells)

The island result holds on real terrain (cold start, `run_type=equilibrium`):

| solver | cold ceiling without T̄ | cold ceiling with T̄ |
|---|---|---|
| Anderson | 1 wk (2 wk diverges) — *this is Kerry's exact setting* | **2 wk (2×)**; and 1 wk slightly faster (19.5 vs 21.1 s, 1926 vs 2001 iters) |
| Picard (default) | *times out cold* (no convergence in 600 s) | **converges** (492 s, 1967 iters) — cold-Picard rescue holds on real terrain |

So at Kerry's 1-week setting T̄ gives Anderson headroom to 2 weeks and makes the default Picard usable cold.
Wall-clock ranking is unchanged (Anderson+T̄ ≈ 20 s vs Picard+T̄ ≈ 492 s for the same patch): T̄'s payoff is
step size and robustness, and on Anderson it is a pure win.

### Bottom line

`-wtm_Tbar` is a small, physics-preserving, off-by-default residual-level change that **enlarges the
stable time step ~2–4× and cures the default-Picard cold-start divergence**, at no accuracy cost and
composing with every solver. It is the numerical realization of managing the exponential transmissivity
by *integrating it over the step* rather than freezing it at the step's start.
