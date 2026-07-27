# The WTM water budget: physical fluxes and a budget-closing check

**Date:** 2026-07-27
**Companions:** `SURFACE_SINK_DESIGN.md` (the sub-surface sink), `BDF2_ADAPTIVE_DESIGN.md` (the
time integrator), `PICARD_MATH.md` (the operator).

WTM now reports a closed water budget. This note states the budget, defines every reported term,
and — importantly — keeps **physically meaningful** quantities separate from **budget-closing**
(numerical-consistency) quantities, because they are *not* the same and their difference is itself a
useful diagnostic.

## 1. The budget

Over any interval, conservation of water is

```math
\underbrace{R}_{\text{recharge in}}
\;=\;
\underbrace{\Delta S}_{\text{change in storage}}
\;+\;
\underbrace{O}_{\text{ocean outflow}}
\;+\;
\underbrace{Q}_{\text{surface sink}} .
```

Every land cell obeys the discrete balance the solver actually solves (BDF2-on-V; `PICARD_MATH.md`),

```math
a_c V(w^{n+1}) - b_c V(w^{n}) + c_c V(w^{n-1})
\;+\; \Delta t\,(\text{lateral outflow})
\;+\; \Delta t\,Q(w^{n+1})
\;=\; S_y\,r ,
```

with `V = storedVolume(w)` the stored water, `S_y` the specific yield, `r` the per-step recharge, and
`(a_c,b_c,c_c) = ((1+2\omega)/(1+\omega),\,1+\omega,\,\omega^2/(1+\omega))` the variable-step BDF2
weights (`3/2,\,2,\,1/2` at constant step). Summed over **all cells**, the interior lateral fluxes
cancel pairwise (flux `c\!\to\!n` is minus flux `n\!\to\!c`), leaving only the **land→ocean boundary**
flux. Summed over **all steps**, this is the budget above. So the budget closes *by construction of
the discretisation* — provided each reported term is the solver's exact discrete term.

## 2. Physically meaningful vs budget-closing quantities

The reported quantities fall in two groups, kept separate on purpose:

| quantity | column | kind | definition |
|---|---|---|---|
| `total_recharge_added` | 9 | **physical** | water delivered, `Sum r_{dist}\cdot A` (precip−ET as a depth) |
| `total_ocean_outflow` | 13 | **physical** | direct Darcy flux across land→ocean faces, `Sum e\,\tfrac{\Delta t}{(\text{cell})^2} h\, A` |
| `total_surface_removed` | 12 | **physical** | sub-surface sink removal, `Sum \Delta t\, Q(w^{n+1})\, A` |
| `stored_volume` | 14 | **physical** | exact stored water, `Sum storedVolume(w)\cdot A` |
| `ocean_loss_closing` | 15 | **budget-closing** | ocean loss *inferred by difference*: `recharge − sink − \Delta(stored\_volume)` |
| `budget_residual` | 16 | **budget-closing** | `ocean_loss_closing − total_ocean_outflow` |

- The **physical** quantities are what science uses: how much water entered, where and how fast it
  left through the coast (a real Darcy flux, per-cell-mappable), how much the sink removed, how much
  is stored. Each is a genuine physical measure, computed directly.
- The **budget-closing** quantities exist to *test conservation*. `ocean_loss_closing` is not a
  physical calculation at all — it is whatever value makes the books balance. Comparing it to the
  physical `total_ocean_outflow` gives `budget_residual`: **≈0 means the physical flux is
  conservative**; a nonzero value is the *numerical-consistency gap*, not a leak (§4).

Reporting ocean loss **both ways** — once physically (direct flux) and once by closure (difference) —
is the whole point: their agreement is the conservation proof, and their disagreement is a
quantified, interpretable discretisation signal.

## 3. Why this was needed

WTM's ocean boundary is Dirichlet `h=0` (ocean cells have `topo=0`; the domain edge is forced ocean
by `land_mask.setEdges(0)`). Crossing water is *absorbed* at that boundary — it never accumulates as
ocean-cell content. The former `total_loss_to_ocean_gw` counted ocean-cell content, which is pinned
at zero, so it measured essentially nothing: a no-crossing, no-sink baseline "lost" 233% of its
recharge with the drained water entirely unaccounted. The interface-flux `total_ocean_outflow`
(§2) is what the boundary actually passes, and it brings the budget from a 233% gap to ≈1%.

## 4. Why the residual is small but not exactly zero (the BDF2 subtlety)

With the *physical* storage change `\Delta S = \sum storedVolume(w^{\text{now}}) - \sum
storedVolume(w^{0})`, the residual is small (≈0.04% once spun up; up to ≈2% on a cold start) but not
machine-zero. Two consistency gaps explain it exactly:

**(a) BDF2 storage does not telescope to `V_{\text{final}}-V_{\text{initial}}`.** Summing the scheme's
storage term,

```math
\sum_{n=1}^{N} \frac{3V^{n+1}-4V^{n}+V^{n-1}}{2}
\;=\;
\frac{3V^{N+1}-V^{N}}{2} \;-\; \frac{3V^{1}-V^{0}}{2} .
```

Near steady state both endpoints reduce to `V` (they telescope), but the **startup term**
`\tfrac{3}{2}(V^{1}-V^{0})` is nonzero during a transient — and a cold start from a deep initial water
table makes the first step large. That startup term is the dominant residual on cold starts (hence
≈2% there, ≈0.04% once the first-step jump is small). Backward Euler *does* telescope exactly
(`V^{n+1}-V^{n} \to V^{N}-V^{0}`), so a BE run closes cleanly; it is the multistep scheme that carries
the boundary terms.

**(b) Specific-yield recharge.** The solver adds `S_y\,r` to the volume balance, while the physical
`total_recharge_added` counts the delivered depth `r`. Since `S_y = storedVolume'(w)` differs from a
plain porosity factor near the surface, `S_y r` and `r` differ slightly there — a second `O` (small)
consistency term.

**Making it exact (optional).** Accumulate the solver's *exact per-step discrete* terms instead of the
physical snapshots: the storage term `\sum (a_c V^{n+1}-b_c V^{n}+c_c V^{n-1})A` (which telescopes to
the endpoints above automatically) and the solver recharge `\sum S_y r\,A`. Then the residual falls to
the SNES convergence tolerance. We deliberately report the **physical** quantities plus the residual
instead, so the numbers mean what a scientist expects and the residual openly shows the numerical gap.

## 5. Verification

- **Budget closes:** residual 233% → ≈1% (drainage now accounted); ≈0.04% once spun up.
- **MPI-consistent:** `total_ocean_outflow`, `stored_volume`, and `budget_residual` are byte-identical
  at n=1 and n=4 (per-rank owned-cell partials reduced with `MPI_Allreduce`; the ocean flux uses a
  ghost `mask_local` so land→ocean faces at rank boundaries are counted exactly once).
- **Non-invasive:** the accounting only *reads* the converged head; the golden regression is
  byte-clean.

## 6. Reported columns (textfile)

`... total_recharge_added(9) total_loss_to_ocean(10) sum_of_water_tables(11) total_surface_removed(12)
total_ocean_outflow(13) stored_volume(14) ocean_loss_closing(15) budget_residual(16)`

`sum_of_water_tables` (11) is the legacy stored-water proxy (`Sum w\cdot\phi\cdot A` below the surface,
`Sum w\cdot A` above); `stored_volume` (14) is the exact `Sum storedVolume(w)\cdot A` used for the
budget. `total_loss_to_ocean` (10) is the legacy ocean-content counter (≈0 under the Dirichlet BC),
retained for continuity; `total_ocean_outflow` (13) supersedes it as the physical ocean loss.
