# Taming the transmissivity–depth nonlinearity — a decision tree

**Status:** plan (2026-08-05). Two parallel tracks; A is physics-preserving and tried first, B is a
modeling change saved for later. See `EQUILIBRIUM_ROBUSTNESS.md` for everything already tried and why.

## The framing (confirmed root cause)
The equilibrium solve is hard because transmissivity rises **exponentially** as the water table rises:
`T = fdepth·ksat·exp((wtd+1.5)/fdepth)` below the soil layer, spanning ~**7 orders of magnitude** over the
wtd range on stiff terrain. This is the driver of the **nonlinearity** (not linear ill-conditioning:
cond(J) ~1e4 is fine). Proven by the controlled test: flattening the profile (large fdepth floor → T≈const)
makes the exact failing case — dt=1 yr from the Dupuit guess — converge in **8 iterations**; nothing else
changed. The exponential was chosen for **analytical/store–discharge convenience** (TOPMODEL; Beven &
Kirkby 1979), not physical necessity — so leaning on it where it is numerically inconvenient is wasteful.
Hence two tracks, pursued in parallel:

---

## Track A — make the EXP work better (numerics only; physics & answer UNCHANGED)
Goal: widen the Newton basin / cut the step count **without changing the model**, so no re-validation is
needed. Test each on a small Esquibel sub-domain (below). Baselines to beat: dt-ceiling ~0.3 yr from
Dupuit; ~100 steps to equilibrium via `-wtm_dt_continuation`.

- **A1 — exploit `log T` linearity (the untried analytical property; Andy's #2).**
  `log T` is *exactly linear* in wtd (in the exp regime), so the nonlinearity is a pure exponential of a
  linear function. Candidate ways to use it:
  - a **log-space line search / damping** (measure step acceptance in a metric where the exp is linear);
  - a within-Newton **linearization of T about the iterate in log space** (Newton on a `log T`-aware
    residual / a variable substitution `s = wtd` with T handled analytically), i.e. treat the exp
    *exactly* over the step instead of via its first-order Taylor term (which is what blows the basin).
  - Watch out: this is *not* Kirchhoff. Kirchhoff (`Φ=f·T`) is a change of variable that fixes
    *conditioning* and was a dead end (worsens conditioning; wrong axis). A1 aims at the *basin*
    (nonlinear step quality), keeping wtd as the variable.
  - **Success = converges at larger dt / far fewer iters than plain Newton on the sub-domain.**

- **A2 — 1-D analytical steady-state initial guess (Andy's #3).**
  The exponential-store hillslope (Boussinesq / TOPMODEL exponential-recession) has closed-form 1-D steady
  states. Seed the fine solve with that instead of the constant-coefficient Dupuit mound — a much closer
  guess exactly because it uses the real exp profile. **Success = Newton converges at larger dt from the
  analytical guess** (a closer guess widens the effective basin). Cheap, purely an initial condition.

- **A0 (DONE, dead end):** Kirchhoff `Φ=∫T dwtd = f·T`. Reaches the identical equilibrium but worsens
  conditioning; kept opt-in `-wtm_kirchhoff`, off. Do not revisit as a speed lever.

---

## Track B — a BETTER formulation (physics change; requires re-validation) — LATER
Goal: replace the exp with a T(depth) that **matches data but is numerically gentle** (polynomial, not
exponential, dynamic range → gentler `dT/dwtd` → wider basin). Saved for after Track A.

- **B1 — power-law / parabolic transmissivity–depth profile.** e.g. `T ∝ (1 + wtd/d)^n` (power) or a
  linear/parabolic decline to a base depth. Much smaller dynamic range than the exp for the same physical
  decline. Prototype as a Python POC on the sub-domain first (like Kirchhoff/homotopy), then decide.
- **B2 — literature review (do this before B1 coding).** Verify the alternative forms and their
  data-fit / calibration precedents; ground it in sources, don't invent:
  - **Ambroise, Beven & Freer (1996, *Water Resources Research*)** — generalizing TOPMODEL to
    **exponential, power, and parabolic** transmissivity profiles; shows non-exp forms fit recession data.
    *(Verify the exact functional forms + parameters.)*
  - Iorgulescu & Musy (1997); Duan & Miller (1997) — other store–discharge/T(z) forms.
  - Fan, Li & Miguez-Macho (2013, *Science*) + Cuthbert et al. — the exp e-folding-depth calibration WTM
    uses; what a power-law re-calibration would need (fdepth_a/b/fmin are exp-specific).
  - Key question to answer from the literature: **does a power-law fit water-table / baseflow data
    *comparably* to the exponential (equifinality)?** If yes, B1 is defensible.
- **Caveat (honest):** any B change re-opens the model's calibration and validation against data. Not a
  free swap; it changes the computed water table. Decide with Andy + data in hand.

---

## Decision tree (how we choose)
1. **Prototype A1 (log T) and A2 (analytical guess)** on the small Esquibel sub-domain. Both are cheap and
   physics-preserving.
   - If either gives a meaningful win (dt ceiling up ≥2×, or step count materially down) → **ship it**
     (no model change, no re-validation). Likely done.
   - If neither helps enough → the nonlinearity is *intrinsic to the exp*, which is itself evidence that
     motivates **escalating to Track B**.
2. **Track B** only if A is insufficient **and** we accept re-calibration: do **B2 (lit review)** first,
   then a **B1 Python POC**, then the physics + calibration decision with Andy and data.

Rule of thumb: prefer the change that (a) preserves the answer, (b) is smallest, (c) is defensible against
data — in that order. A1/A2 satisfy (a); B needs (c) demonstrated.

---

## Test harness (for continuity across compaction)
- **Sub-domain:** `scratchpad/esq_crop` (200×200 window of Esquibel at full 900 cpd → keeps the 7-order
  stiffness) or its 60×60 sub-window (`SL=(70:130,70:130)` in the POC scripts) for fast Python iteration.
  Rebuild the Dupuit guess with the standard snippet used throughout (dist-transform × sqrt(R/K)).
- **Existing POC scaffolding to reuse:** `scratchpad/kirchhoff_newton.py` has a correct steady-state
  residual + analytic wtd-Jacobian + the piecewise T/Φ and the crop loader — A1/A2 can build on it (but
  its hand-rolled Newton line search is weak; use a robust solver, e.g. `scipy.optimize.root`, or damp
  properly, so results aren't confounded by solver crudeness, as they were before).
- **Metrics:** (1) dt ceiling = largest dt that converges from the Dupuit (or analytical) guess; (2) total
  Newton iters to equilibrium; (3) for B, does it match the exp equilibrium / fit data.
- **Baselines already measured:** plain head-form Newton ceiling ~0.3 yr; flattened profile converges
  dt=1 yr in 8 iters; dt-continuation reaches equilibrium in ~100 steps; Kirchhoff = no help.

## Success criteria
- **Track A:** a physics-preserving change that raises the dt ceiling or cuts the step count ≥2×.
- **Track B:** a T(depth) form that fits data comparably to the exp AND materially widens the Newton basin.

## One-line status of the two ideas
- **A1 (log T):** untried; the one analytical property of the exp we have NOT exploited; Andy hopeful,
  low prior intuition — prototype first.
- **A2 (analytical guess):** untried; cheap; strictly an initial condition, so low risk.

---

## Results (2026-08-05) — Track A prototyped on the 60×60 Esquibel sub-window
Vectorized, FD-verified head-form Newton (`scratchpad/logT_newton.py`, `picard_ref.py`, `a2_guess.py`,
`cont_seed.py`). Baseline reproduced the validated C++ behavior: plain Newton from the Dupuit guess
converges at **dt=0.1 yr (7 iters)** and stalls at **dt≥0.3 yr** (line search collapses; tiny basin).

**A1 (exploit log-T linearity) — NEGATIVE.** Three independent ways of using `log T = log(fk)+(wtd+SH)/f`
being exactly linear in wtd all leave the ~0.3-yr ceiling unmoved:
- *Magnitude* — log-T trust region (cap per-cell `|Δwtd| ≤ c·fdepth` ⇒ `|Δ log T| ≤ c`, c=1,2):
  identical stall. Clamping the step magnitude is redundant with the backtracking line search; it does
  not change the Newton *direction*, which is the problem.
- *Coefficient-lag* — frozen-T Picard (the exact exponential coefficient, lagged): undamped **oscillates**
  (one step overshoots to −114 m then swings back — reproduces the documented Kerry cold-start oscillation);
  damped (Armijo) converges only to **dt≈0.1–0.3 yr**, same ceiling as Newton.
- *Change of variable* — Kirchhoff `Φ=∫T dwtd` (already done, `EQUILIBRIUM_ROBUSTNESS.md`): dead end;
  topography keeps T explicit so it does not linearize, and it worsens conditioning.
- **Conclusion:** the tiny basin is **not** a property of the linearization scheme — Newton and damped
  Picard hit the same wall. It is set by the exp physics + being far from the solution.

**A2 (1-D analytical / TOPMODEL steady-state guess) — MARGINAL & FRAGILE.** Local-equilibrium depth
`z = fdepth·(ln(T0/R) − TI)`, `TI=ln(a/tanβ)`, `a` from D8 flow accumulation on the crop DEM, `T0` the
WTM surface transmissivity. The *raw* guess is **worse than Dupuit** (deeper initial residual; flat-area
`z` blows up to −79 m — the known TOPMODEL flats pathology). Adding TOPMODEL's free catchment-mean-depth
offset `z_bar`, there is a **band of offsets (+10..+60 m) where Newton converges at dt=0.3 yr (44–99 iters)**
— a real ~3× ceiling bump over Dupuit. BUT:
- The natural selection criterion — pick `z_bar` that minimizes the static initial residual — **picks the
  wrong offset**: static `‖F‖` falls monotonically to its min at +80 m, yet +80 m *fails* while +40 m
  (higher residual) succeeds. So closeness in L2 norm ≠ wider basin; there is no cheap a-priori signal to
  land in the good band.
- Even the best A2 only reaches **dt≈0.3 yr** — the same ceiling MUMPS/HYPRE already give, and far short of
  the dt=10–1000 yr regime a big-step equilibrium solve wants.
- Seeding dt-continuation from A2 vs Dupuit: **inconclusive** — the toy continuation controller here
  collapsed dt for *all* seeds (it is cruder than the committed C++ reject/retry controller, which works
  from Dupuit on the real 384k case); within that crude test A2 seeds were no better (the min-residual +80
  seed was worst).

**Net (decision tree).** Track A yields **no robust, cleanly-shippable ≥2× win**. Both A1 and A2 point to
the same thing: the difficulty is **intrinsic to the exponential**, not to the solver or the guess-in-L2.
This is exactly the branch the plan named as motivating **Track B** (a gentler T(depth) formulation). The
already-committed **dt-continuation** remains the reliable production path to equilibrium.

---

## Track B — B2 literature review (2026-08-05)
Sourced from the generalized-TOPMODEL literature (see Sources). The question Andy posed: *is there a
numerically-gentler T(depth) that still fits data comparably to the exponential?* Short answer: **yes —
the power-law profile, which contains WTM's exponential as a limiting case.**

**The three classical profiles (Ambroise, Beven & Freer 1996a; Beven "History of TOPMODEL" 2021).**
`D` = gravity-drainage storage deficit (depth to water table), `T0` = transmissivity at saturation
(`D=0`), `m`/`M` = a depth scaling parameter. Each profile implies a distinct topographic index and a
distinct baseflow recession shape (the recession is the field-observable that discriminates them):

| profile | `T(D)` | topographic index | recession (linear when plotted) |
|---|---|---|---|
| **exponential** (WTM's) | `T0·exp(−D/m)` | `ln(a/tanβ)` | `1/Qb` vs t (1st-order hyperbolic) |
| **linear** | `T0·(1 − D/M)` | (a/tanβ)-based | `ln Qb` vs t (exponential) |
| **parabolic** | `T0·(1 − D/M)²` | `√(a/tanβ)`-based | `1/√Qb` vs t (2nd-order hyperbolic) |

- Exponential: `q = T0 tanβ exp(−D/m)` (Beven 2021, Eq. 6); `Qb = Q0 exp(−D̄/m)` (Eq. 8). This is exactly
  WTM's `T = fdepth·ksat·exp((wtd+1.5)/fdepth)` with `m=fdepth`. It is the analytically convenient one
  (closed-form store–discharge) but **never reaches zero** — hence the ~7-order dynamic range that our
  Track-A tests proved is the nonlinearity driver.
- Linear & parabolic reach **T=0 at a finite deficit `D=M`** (a physical bedrock/base depth). Their
  dynamic range is ~1–2 orders (vs the exp's 7) and `dT/dD` is bounded and gentle → a structurally milder
  Jacobian → wider Newton basin.

**Power-law profile — the generalization that matters (Iorgulescu & Musy 1997; Duan & Miller 1997).**
`T(D) = T0·(1 − D/M)^n`. This *generalizes* linear (n=1) and parabolic (n=2), and — the key property —
**recovers WTM's exponential exactly as n→∞** with `M = n·m` (verified numerically: n=1000, M=n·fdepth
reproduces `exp(−D/fdepth)` to 4 decimals; `scratchpad/` check). So the exponential is not discarded; it
becomes the `n→∞` end of a **one-parameter family with a numerical-convenience knob**: small `n` = bounded
dynamic range + finite depth = gentle Newton basin; large `n` = today's calibration and behavior.

**Trade-offs (honest).**
- *For the numerics:* the win is exactly what our controlled flattening test predicted — compressing the
  dynamic range widens the basin. Power-law at finite `n` does this by construction.
- *New hazard:* the finite depth `T→0` at `D=M` can make fully-drained cells **disconnected/degenerate**
  (zero lateral transmissivity) — a *different* numerical failure mode the exp never has (it stays
  positive asymptotically). Mitigable by choosing `M` large enough that `D<M` everywhere, or a small `T`
  floor; must be checked, not assumed.
- *For the data / calibration:* Ambroise/Beven/Freer developed these **because** the exponential recession
  often does *not* match observations — the exp "was found to be not appropriate" for the Ringelbach
  catchment, where the parabolic form fit better. So non-exp forms are **data-defensible and roughly
  equifinal** in the recession sense (you pick the profile from the observed recession shape). **Caveat:**
  that evidence is small-catchment *recession* fitting; WTM's use is *global equilibrium water-table depth*,
  calibrated on Fan/Cuthbert exponential e-folding depths (`fdepth_a/b/fmin` are exp-specific). A power-law
  migration needs its own water-table validation and a rule to set `(n, M)` from the existing `fdepth`
  data — the `n→∞` limit gives a safe, continuous starting point (reproduce current results, then lower
  `n` and watch the water table + basin).

**Recommendation for the B-branch decision (Andy's call).** The power-law profile is the defensible Track-B
form: a literature-grounded superset of the current exp, tunable from "identical to today" (`n→∞`) toward
"numerically gentle" (small `n`), addressing exactly the "exp is convenient-until-it-isn't" framing. The
low-risk next step is a **B1 Python POC on the sub-window**: implement `T=T0(1−D/M)^n`, confirm large-`n`
reproduces the exp basin/ceiling, then sweep `n` down to measure how much the dt-ceiling widens and where
the finite-depth degeneracy starts to bite — *before* touching WTM's physics or calibration.

## Track B — B1 power-law POC results (2026-08-05) — NEGATIVE for the numerics
Implemented `T=T0(1−d/M)^n` with FD-verified tangent (`scratchpad/pow_newton.py`), on the same 60×60
sub-window and Newton harness as Track A. Two parameterizations tested:
- **`M = n·fdepth`** (the clean exp-limit family, `n→∞` = exp): large `n` reproduces the exp ceiling
  (n=8 converges at dt=0.1, stalls at 0.3, exactly like the exp). Lowering `n` **worsens** the basin
  (n=2, n=1 fail even at dt=0.1) because `M` shrinks with `n`, so the deep water table breaches the finite
  depth: **T=0 fully-drained cells = 1.7% (n=8) → 25% (n=2) → 47% (n=1)** at the dt=0.1 solution. The
  degeneracy hazard flagged in B2 bites hard and early.
- **`M = Mf·fdepth`** (M decoupled from `n`, kept deep to avoid degeneracy): scanned `n∈{1,2,4,8}` ×
  `Mf∈{8,16,32}`. **No combination beats the exp basin — every profile still stalls at dt=0.3 yr**, the
  exp's own ceiling. The combos that compress the dynamic range to ~2 orders (large `Mf`, small `n`) fail
  even at dt=0.1 *and* change the physical answer.

**Why this is a hard negative (the premise was flawed).** The transmissivity dynamic range is **physically
pinned by the boundary values**: T must fall from its surface value to ≈0 at the deepest water table
(~6–7 orders on this terrain, set by the physics of a deep equilibrium table). *Any* profile matching the
same physics spans the same range — the exp's 7 orders are not an artifact of the exp, they are the
physics. Compressing the range (which was the whole hope) either fails to converge anyway or changes the
computed water table. This mirrors the old "flattening" observation correctly: a *near-constant* T is easy,
but only because it is a *different, wrong* problem — a 2-order-varying T is still hard. **The dt-ceiling is
invariant to the transmissivity functional form.**

**Consequence — the investigation closes.** The equilibrium cold-start difficulty is **intrinsic to the
far-from-solution nonlinear steady-drainage problem on stiff terrain** — not the solver linearization (A1),
not the initial guess in L2 (A2), not the transmissivity functional form (B1). **dt-continuation** (a
sequence of nearby solves; committed, reliable on the real 384k Esquibel case) is the correct tool, not a
workaround for a fixable formulation. *Separate, not foreclosed:* a power-law/parabolic profile may still
be worth adopting for **data-fit / physical-realism** reasons (Ambroise/Beven/Freer's original motivation —
matching observed recessions), but B1 removes "it will speed up the equilibrium solve" from the reasons to
do so; that is now a modeling decision for Andy + data, decoupled from solver performance.

### Sources
- Beven, K. (2021), *A history of TOPMODEL*, HESS 25, 527–549 — https://hess.copernicus.org/articles/25/527/2021/ (Eqs. 6–8; Fig. 7 profile/recession caption)
- Ambroise, Beven & Freer (1996a), *Toward a generalization of the TOPMODEL concepts: topographic indices of hydrological similarity*, WRR 32(7), 2135–2145 — https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/95WR03716
- Ambroise, Freer & Beven (1996b), *Application of a generalized TOPMODEL to the small Ringelbach catchment*, WRR 32(7) — https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/95wr03715
- Iorgulescu & Musy (1997), *Generalization of TOPMODEL for a power law transmissivity profile*, Hydrol. Process. 11, 1353–1355 — https://onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1099-1085(199707)11:9%3C1353::AID-HYP585%3E3.0.CO;2-U
- Duan & Miller (1997), *A generalized power function for the subsurface transmissivity profile in TOPMODEL*, WRR 33(11), 2559–2562
