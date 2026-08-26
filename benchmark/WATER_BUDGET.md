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
| `total_recharge_added` | 9 | **physical** | TOTAL external water in = col 19 + col 20 (precip−ET as a depth) |
| `recharge_direct` | 19 | **physical** | the `(1−runoff_ratio)` share, straight into the aquifer as the step's source |
| `runoff_to_surface` | 20 | **physical** | the `runoff_ratio` share, diverted to the surface for FillSpillMerge to route |
| `total_ocean_outflow` | 13 | **physical** | direct Darcy flux across land→ocean faces, `Sum e\,\tfrac{\Delta t}{(\text{cell})^2} h\, A` |
| `total_surface_removed` | 12 | **physical** | sub-surface sink removal, `Sum \Delta t\, Q(w^{n+1})\, A` |
| `stored_volume` | 14 | **physical** | exact stored water, `Sum storedVolume(w)\cdot A` |
| `ocean_loss_closing` | 15 | **budget-closing** | total ocean loss *inferred by difference*: `recharge − evap − \Delta(stored\_volume)` |
| `budget_residual` | 16 | **budget-closing** | `ocean_loss_closing − total_ocean_outflow − total_loss_to_ocean` (both ocean channels: Darcy + FSM spill; carries the BDF2 gap) |
| `exact_budget_residual` | 17 | **budget-closing** | `solver_recharge − storage_change − ocean − sink` from the solver's exact per-step discrete terms; ≈0 to SNES tolerance on every solver path, TR-BDF2 included (its two stages telescope — see below). Reported as `nan`, never as a stale zero, if a scheme ever cannot be expressed as one per-step identity |

- The **physical** quantities are what science uses: how much water entered, where and how fast it
  left through the coast (a real Darcy flux, per-cell-mappable), how much the sink removed, how much
  is stored. Each is a genuine physical measure, computed directly.
- The **budget-closing** quantities exist to *test conservation*. `ocean_loss_closing` is not a
  physical calculation at all — it is whatever value makes the books balance. Comparing it to the
  physical `total_ocean_outflow` gives `budget_residual`: **≈0 means the physical flux is
  conservative**; a nonzero value is the *numerical-consistency gap*, not a leak (§4).
- **`total_surface_removed` is NOT a budget sink.** It is an *internal* GW→FSM transfer (the sub-surface
  sink / active-set skim hands above-surface water to FillSpillMerge). Its fate is already captured
  elsewhere: FSM either keeps it in a lake (counted in `stored_volume`) or routes it to the ocean
  (`total_loss_to_ocean`). Counting it as a sink double-counts water that FSM recycles into a persistent
  lake — a closed lake re-skims the same recharge every step, inflating `total_surface_removed` to a large
  gross flux while the water sits in storage (this surfaced once the active-set skim actually delivered its
  captured water to FSM; see `tests/fsm_fullness` / `tests/fsm_conservation`). So `total_surface_removed`
  stays a diagnostic of the gross removal, but the conservation budget uses `stored_volume` + evap + the two
  ocean channels only.

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

**Making it exact.** `exact_budget_residual` (column 17) does exactly this: it accumulates the
solver's *exact per-step discrete* terms — the storage term `\sum (a_c V^{n+1}-b_c V^{n}+c_c
V^{n-1})A` (which telescopes to the endpoints above automatically) and the solver recharge `\sum S_y
r\,A` — over owned land cells. By the discrete balance, `storage_change =
solver_recharge − ocean_outflow − surface_removed` to the SNES tolerance, so this residual is ~0
regardless of cold-start transients. We report it *alongside* the physical quantities (not instead),
so the headline numbers mean what a scientist expects while the exact residual proves conservation.

**It is now solver-agnostic** (it was Picard-only, which left the default matrix-free Anderson path
without an exact check). `accumulate_budget_terms` mirrors the residual's storage form in **volume**
units, i.e. without the `1/S_y` head-scaling the Anderson residual applies (a positive per-cell scale
that leaves the root unchanged but would corrupt a budget).

Only **two** storage forms are needed, not one per solver flag, and the reason is worth recording
because it is easy to get wrong in the other direction. `-wtm_volume_storage` is a backward Euler
whose storage is the exact volume change `V(w^{n+1}) - V(w^{n})`, and the *secant* form already
computes precisely that: `updateEffectiveStorativity(w^n, w^{n+1})` is **defined** as the secant
`(V(w^{n+1}) - V(w^{n}))/(w^{n+1} - w^{n})` (pinned in `src/test_storage_math.cpp`), and
`h^{n+1} - h^{n} = w^{n+1} - w^{n}`, so `S_c(h^{n+1}-h^{n}) \equiv V(w^{n+1}) - V(w^{n})`. The two
separate only once the BDF2 weights are not `(1,1,0)`, because `S_c(a_c h^{n+1} - b_c h^{n} + c_c
h^{n-1})` is then *not* the weighted volume difference. So: secant for everything backward-Euler,
volume for BDF2-on-V. **TR-BDF2 needs neither of them** — see below, where its two stages are shown
to telescope onto the backward-Euler storage form exactly.

### TR-BDF2: the step balance its two stages define

TR-BDF2 takes two implicit stages per step and each satisfies its own discrete balance. Neither is
the step's, so for a long time column 17 was reported as `nan` under TR-BDF2 and the scheme was
simply excluded. That was honest but expensive: TR-BDF2 is the integrator the adaptive controller
drives, and it was the one scheme whose conservation nothing could check.

The stages *do* combine. Take `C1 × (stage 1) + (stage 2)`, with `γ = 2 − √2`, `C1 = 1/(γ(2−γ))`,
`C2 = (1−γ)²/(γ(2−γ))` and `C3 = (1−γ)/(2−γ)`. The storage terms telescope because `C1 − C2 = 1`
exactly, and the recharge collapses because `C1·γ + C3 = 1`, leaving

```
V(w^{n+1}) − V(w^n)  =  R  −  [ W_OLD·F(w^n) + W_YGAMMA·F(Y_γ) + W_NEW·F(w^{n+1}) ]  −  [ C1·E1 + E2 ]
```

with `W_OLD = W_YGAMMA = C1·γ/2` and `W_NEW = C3`. So **TR-BDF2's per-step balance is the
backward-Euler balance with exactly two substitutions**: every flux and removal term becomes a
three-point quadrature over the states `(w^n, Y_γ, w^{n+1})`, and the active-set exfiltration
multiplier becomes `E = C1·E1 + E2`. Storage and recharge are unchanged — which is why those halves
of the budget were always right under TR-BDF2 and only the flux, removal and exfiltration halves
were wrong.

The weights sum to 1 (consistency) and satisfy `W_YGAMMA·γ + W_NEW = 1/2` exactly, which is the
second-order condition: the budget is second-order accurate, not merely conservative. It is not
exact on a quadratic (0.414214 against 1/3), the expected order barrier. All of this is derived in
`src/tr_bdf2_coefficients.hpp` and pinned in `src/test_tr_bdf2_balance.cpp`.

**Why it mattered, quantitatively.** Stage 1 carries `C1·γ = 70.71%` of the step and stage 2 carries
`C3 = 29.29%`. Reading the exfiltration multiplier off the stage-2 residual alone — which is what
happened, because every residual evaluation overwrites the vector and stage 2 evaluates last —
recovered 29.29% of the step and understated by `1/C3 = 3.4142`. Measured on `tests/multilake` under
`active_set` at `dt = 0.25 yr`: 5.97e11 m³ delivered to FillSpillMerge against backward Euler's
2.00e12 m³ (ratio 3.356 against the predicted 3.414), and a physical budget residual of 9.5% of
recharge where BDF2-on-V — also multi-level, also second order — closes at 0.2%.

With all three terms weighted, column 17 closes at 2.2e-8 of recharge under TR-BDF2 against
backward Euler's 3.1e-8, and is tolerance-limited (1.7e-8 per cycle at `-snes_stol 1e-8`).
`tests/budget_closure` holds both plain TR-BDF2 and TR-BDF2 + active-set to that standard.

### The two input CHANNELS (columns 19 and 20)

Precipitation-less-evaporation is split by the runoff ratio the moment it is computed, in
`WTM.cpp`'s `distributed_recharge` (and its serial rank-0 twin):

```cpp
const double runoff = rratio_f * dmdapack.rech_dist[j][i];
dmdapack.runoff_dist[j][i] = runoff;   // -> FillSpillMerge      (column 20)
dmdapack.rech_dist[j][i] -= runoff;    // -> the step's source   (column 19)
```

Both halves are external water entering the domain; they differ only in *route*. Column 9 used to
sum the direct half alone, so the routed share was missing from the budget's "water in" while the
lakes FSM builds from it were present in `stored_volume` — which is why `budget_residual` could not
close with `runoff_ratio > 0`. Measured: col 9 scaled as exactly `(1 − runoff_ratio)` (2.43225e10 at
`rr=0.3` against 3.47464e10 at `rr=0`, ratio 0.700000).

Two things to know when reading the pair:

- **`col20/col19` is *not* `rr/(1−rr)`.** The runoff share is taken only where `rech_dist > 0`, while
  the direct channel also carries the *negative* (net-evaporation) cells and is clipped at the land
  surface.
- **The totals legitimately move with `runoff_ratio`**, because `rech_dist` is state-dependent: a
  wetter table puts more cells on the `(precip − open_water_evap)` branch, and open-water evaporation
  can exceed precipitation. This is the evaporation model, not an accounting error. Verified at
  `runoff_ratio = 1.0`, where column 19 is exactly 0 and the whole input lands in column 20.

Column 19 is accumulated **per sub-step and scaled by `rech_dt_scale`**, so a sub-stepped cycle books
the cycle total once rather than once per sub-step. Without that scaling column 9 tracked the *solve
count* instead of elapsed time (2.55× inflation under adaptive dt).

**RESOLVED (2026-08-26): the routed channel is now scaled by elapsed time.** It used to be *delivered*
to FillSpillMerge at full nominal-step size on every accepted sub-step and never scaled, so under
sub-stepping the model routed an amount proportional to the SOLVE COUNT — a mass error, not a
reporting one. Measured at `rr=0.3`: 1.36148e10 over 20 solves (fixed `dt`) against 9.53038e09 over 14
(adaptive), a ratio of 0.700 against a solve-count ratio of 0.700, and a 6.6 % divergence in
`stored_volume` against 0.16 % with the channel off.

The fix makes the routed channel **lazy**, like the direct one. The direct share survives sub-stepping
because `rech_dist` holds a nominal depth scaled at the point of *use*, when `dt` is final; the routed
share was baked and consumed eagerly, before the step it would be routed over had been chosen. So
`runoff_dist` (and `arp.runoff_nominal` on the serial path) now hold the nominal depth, and both the
delivery *and* the column-20 booking happen at the handoff, scaled by `dt_committed` — the `dt` of the
step that was accepted. Scaling at preparation would still be wrong: the loop can clamp `dt` to the
cycle remainder afterwards, and a rejected step re-runs smaller.

Result: column 20 is now **exactly invariant** to solve count (0.000e+00), and `stored_volume`'s
fixed-vs-adaptive spread at `rr=0.3` falls to 7.293e-04 — below the `rr=0` control. One deliberate
change at fixed `dt`: the final step's prepared runoff is no longer booked, because it is never
delivered, so column 20 counts N−1 handoffs rather than N preparations. `tests/dt_invariance` gates
this.

### Two definitions of "recharge", and which one this uses

There are two accumulators, and they mean different things:

| accumulator | sums | meaning |
|---|---|---|
| `total_recharge_added` (col 9) | `rech_dist`·`rech_dt_scale` + `runoff_dist` | **external** water entering the domain, BOTH channels (cols 19 + 20) |
| `total_solver_recharge` (col 17's input) | `rech_vec`, the source term the residual actually integrates | **everything the scheme treats as an input during the step** |

They agree whenever all input arrives as precipitation. They diverge under
`-wtm_fsm_delta_source` (#116), where FillSpillMerge's delivery is folded into the step's source term
rather than applied as a between-step overwrite of the water table. Once the water arrives *during*
the step, the scheme's own conservation law counts it as an input, and only the second definition
describes the scheme being run — so the exact budget uses it.

Measured on the dome fixture (Anderson, FSM on, at steady state), the exact residual relative to
recharge is `1.6e-6` for the overwrite path and **`6.9e-11`** for `-wtm_fsm_delta_source`. The source
path conserves *more* tightly, and for a structural reason: its coupling flux is an explicit term in
the residual, which the solver drives to its tolerance, whereas the overwrite arrives as a state jump
that no per-step discrete identity can see. Column 16 (the physical residual, built on the external
definition) still reads ~18% for the source path; that is the definitional mismatch, not a leak.

## 4a. What the exact residual then uncovered: N–S flux on a lat-lon grid

Driving the numerics to machine zero turned the budget into a probe, and it found a real property:
`exact_budget_residual` is machine-zero (≈`10^{-11}` relative) on a **constant-area** grid, but on a
latitude-varying grid it is a small, constant **per-step** term that scales with the meridional area
gradient (≈0.25% on a coarse 12.8°-span test grid; it shrinks by ~`500\times` when the latitude span
and cell size shrink `10\times`, and is negligible on fine grids).

Cause: the flux across a **north–south** face between rows `j` and `j{+}1` uses each cell's *own* area
(`cell\_area[j]` vs `cell\_area[j{+}1]`), which differ because cells shrink poleward. So
`flux(c\!\to\!n)\,A_c \neq flux(n\!\to\!c)\,A_n` and the pair does **not** cancel in volume; **east–west**
faces (same latitude, equal area) cancel exactly. The discretisation is thus volume-conservative to
`O(\text{area gradient})`, not exactly, on a varying grid. This is a genuine (small) non-conservation
surfaced *by* the exact budget check — not an accounting error (the check closes to machine zero where
the grid area is constant). Whether to make the meridional flux face-area-symmetric (true conservation
on all grids, but a change to the core operator that rebaselines results) is a separate decision; for
now the check *measures* it.

## 5. Verification

- **Budget closes:** residual 233% → ≈1% (drainage now accounted); ≈0.04% once spun up.
- **MPI-consistent:** `total_ocean_outflow`, `stored_volume`, and `budget_residual` are byte-identical
  at n=1 and n=4 (per-rank owned-cell partials reduced with `MPI_Allreduce`; the ocean flux uses a
  ghost `mask_local` so land→ocean faces at rank boundaries are counted exactly once).
- **Non-invasive:** the accounting only *reads* the converged head; the golden regression is
  byte-clean.

### RESOLVED: the post-solve collectors now close too

Uncovered while building the TR-BDF2 arms, fixed 2026-08-26. It was never TR-BDF2's, and it was never
one collector: **both** post-solve removals — `explicit` and `legacy` — broke the exact budget, on
**every** solver.

| collector | Anderson | Picard | in the residual? |
|---|---|---|---|
| `active_set` | −3.744e-07 | hard error | yes (semismooth pin) |
| `implicit` | +4.197e-09 | −1.890e-10 | yes (in-residual siphon) |
| `off` | −1.753e-07 | −3.370e-10 | n/a |
| `explicit` | −9.165e+00 → **−1.670e-07** | −9.331e+00 → **−3.328e-10** | **no — post-solve clamp** |
| `legacy` | −7.998e+00 → **−8.461e-08** | −8.865e+00 → **−3.289e-10** | **no — post-solve clamp** |

**The cause was a state mismatch, not a missing term.** `accumulate_budget_terms` read the storage
from `dmdapack.x` — the *pre-clamp* `w^{n+1}` — while the commit loop then clamped and stored the
post-clamp value. The budget therefore described a state the model does not carry forward. Under a
post-solve collector the residual has no removal term, so the solve satisfies

```math
\Delta V_{\text{pre}} = \text{solver\_recharge} - \text{ocean} - \text{evap}
```

and the reported residual came out at exactly `−total_surface_removed`. With FSM on, the same water is
returned and re-skimmed every step, so what accumulated was a **gross** flux — which is why the
magnitude exceeded 1 rather than being a small percentage. Substituting the committed state,
`ΔV_post = ΔV_pre − excess_depth`, closes the identity algebraically and in practice.

**No water was ever lost.** `excess_depth` is handed to FillSpillMerge on the line above, so this was
a defect in the conservation *check*, not in conservation. It mattered because `explicit` is what the
**Picard** solver resolves to when the collection method is unset — so the budget was unusable in
Picard's default production configuration.

Every collector that already closed is unchanged to the digit, which is what makes the correction
surgical. `tests/budget_closure` now covers all five collectors on both solvers, plus each solver at
its own resolved default.

## 5a. Adding a new water channel

Every defect found in the budget on 2026-08-26 was the same shape: a channel that was booked, scaled
or delivered at the wrong moment. Not physics, not numerics — *when*. Six of them, all in one loop.

So when you add a channel, add its **equivalent-model-time** arms at the same time. The idea is simple:
run the same model time by a different numerical path, and require the cumulative totals to agree. Two
paths are worth varying, and they catch different things.

**Different step count.** Adaptive `dt` subdivides; a larger `dt` takes fewer steps. A cumulative
quantity must scale with elapsed time, never with the number of solves. This caught the routed channel
being delivered once per solve at nominal size:

| arm | solves | elapsed | routed water |
|---|---|---|---|
| fixed `dt` | 20 | 20.00 yr | 1.36148e+10 |
| adaptive | 14 | 20.00 yr | 9.53038e+09 |

`9.53/13.61 = 0.700`, and `14/20 = 0.700`. Gated by `tests/dt_invariance`.

**Different bookkeeping cadence.** Be precise about whose cadence: **FillSpillMerge is not
configurable — with `fsm_on` it runs once per accepted step, always.** Three of the four
`couple_surface_and_recharge` call sites sit inside the step loops. The fourth fires once per *report*
and skips FSM entirely, so with no surface-water model the recharge/runoff bookkeeping happens per
report and `report_interval` sets how often. That is the only cadence there is to vary, and varying it
caught the routed share being booked `(N−1)/N` — 3/4, 1/2 and 0/1 of its true value at 4, 2 and 1
cycles. Gated by `tests/local_ledger` arm C.

**Why the exact budget does not substitute for either.** `budget_closure` passed throughout both of
those bugs. It checks the *solver's own* discrete identity, which stays self-consistent when an input
channel is mis-booked — the solve does not know or care what the bookkeeping did. Closure and
correct accounting are different claims.

**One more habit, learned expensively.** Pick a fixture where recharge is strictly positive when you
need an exact expectation. `rech_dist` goes negative where evaporation exceeds precipitation, and the
runoff split only fires where `rech_dist > 0`, so the direct channel carries negatives the routed one
never sees. An hour went into a "35% under-count" that was the yardstick, not the code.

## 6. Reported columns (textfile)

`... total_recharge_added(9) total_loss_to_ocean(10) sum_of_water_tables(11) total_surface_removed(12)
total_ocean_outflow(13) stored_volume(14) ocean_loss_closing(15) budget_residual(16)
exact_budget_residual(17) total_evap_removed(18) recharge_direct(19) runoff_to_surface(20)
elapsed_time_s(21) solves_done(22) rejects_done(23)`

### Columns 21–23 are the denominators, and they exist to make one invariant checkable

> **No cumulative quantity may be proportional to the SOLVE COUNT. Every one must be proportional to
> ELAPSED TIME, or be a difference of states.**

Three separate bugs violated that, and each showed up as a column tracking solves rather than time.
With `elapsed_time_s` and `solves_done` both in the file, the violation is visible by inspection —
run the same physical problem at fixed and adaptive `dt` and compare:

| arm | elapsed_yr | solves | col 9 | col 19 direct | col 20 routed |
|---|---|---|---|---|---|
| `rr=0` fixed | 20.00 | 20 | 3.47464e10 | 3.47464e10 | 0 |
| `rr=0` adaptive | 20.00 | 15 | 3.47464e10 | 3.47464e10 | 0 |
| `rr=0.3` fixed | 20.00 | 20 | 3.79373e10 | 2.43225e10 | **1.36148e10** |
| `rr=0.3` adaptive | 20.00 | 14 | 3.38529e10 | 2.43225e10 | **9.53038e09** |

Columns 9 and 19 are invariant to the solve count; column 20 tracks it (0.700 against 14/20 = 0.700).
That is the routed-channel mass defect above, legible without a special harness.

**Deliberately no derived rate columns.** A rate computed as `Δ(cumulative)/Δt` is exactly recoverable
from what is already here, so it adds convenience but no information — and no checking power, since a
rate derived from a wrong amount is wrong in the same proportion. Only the denominators let you test
the amount.

**Off-by-one, stated because getting it wrong corrupts every rate a reader derives:**
`elapsed_time_s = (cycles_done + 1) × report_seconds`. `PrintValues` is called for a cycle that has
just *completed*, and `cycles_done` is incremented after the call.

**Columns 19 and 20 are APPENDED, never inserted.** Three test scripts (`tests/budget_closure`,
`tests/fsm_cascade`, `tests/fsm_conservation`) parse this file positionally, so every existing index
must stay put.

`sum_of_water_tables` (11) is the legacy stored-water proxy (`Sum w\cdot\phi\cdot A` below the surface,
`Sum w\cdot A` above); `stored_volume` (14) is the exact `Sum storedVolume(w)\cdot A` used for the
budget. `total_loss_to_ocean` (10) is the legacy ocean-content counter (≈0 under the Dirichlet BC),
retained for continuity; `total_ocean_outflow` (13) supersedes it as the physical ocean loss.
