# Why early Picard tests looked promising — post-mortem (transmissivity stiffness)

## The claim that drove the Picard-default decision

Early tests reported Picard (BDF2-on-V, Newton+GAMG on the frozen SPD operator) taking **large, stable
time steps with a nearly step-size-independent cost** — "Corsica: ~28 SNES iters flat from dt = 1 to
1000 yr." That justified making Picard the default and treating it as *the* big-step equilibrium solver.

## What actually controls Picard's step ceiling

Picard's usable `dt` is set by whether the **storage regularization `S/dt`** (on the operator diagonal)
dominates the **flux operator's transmissivity range**. As `dt` grows, `S/dt` shrinks; if the exponential
`T(wtd) = fdepth·ksat·exp((wtd+1.5)/fdepth)` spans ~7 orders of magnitude, the frozen operator **loses
diagonal dominance** and the frozen-coefficient iteration oscillates → diverges. At the extreme the
*linear* operator itself goes singular/indefinite (confirmed earlier: both GAMG **and** MUMPS fail — not a
preconditioner issue).

**So the ceiling is `dt` × T-stiffness, NOT distance-from-equilibrium.** Verified two ways:
- *At equilibrium, huge `dt` still fails.* A supplied-equilibrium (no forcing change) 1-year Picard step
  diverges just like a perturbed one — ruling out "the early tests were merely near equilibrium."
- *Gentler T lifts the ceiling at fixed `dt`.* Warm, `dt = 8 wk` (where the stiff real T fails), Picard+T̄:

  | `fdepth_fmin` | T character | Picard+T̄ @ 8 wk |
  |---|---|---|
  | 2.5 (Kerry's real) | stiff exp (~7 orders) | **FAIL** (oscillates, 10000 it) |
  | 100 | gentle exp | **OK, ~8 it/solve** |
  | 10000 | ~flat | OK, ~20 it/solve |

  Same solver, same `dt`, same domain — only the transmissivity gentleness differs.

## Diagnosis

**The early promising Picard tests ran on gentler transmissivity (less-stiff conditions) than Kerry's real
Esquibel.** "Flat ~28 iters, dt = 1→1000 yr" is a real property **of gentle T** — where diagonal dominance
survives to huge `dt` — and does **not** transfer to the sharp exponential T of the production problem,
where Picard caps at ~1 wk and hangs cold. Two compounding measurement errors:
1. **Gentler T** in the promising tests (dominant factor).
2. Some early Esquibel runs used the config's **73-day `dt`** (10× Kerry's 1-week), even further past the
   ceiling — so those "failures" were also `dt` artifacts.

## Consequence

This is the root justification for **not defaulting to Picard** (the merge regression, see
`PORT_TO_UPSTREAM.md`) and for the **Anderson + T̄** workhorse: matrix-free Anderson never inverts the
stiff operator, so it sidesteps the diagonal-dominance/singularity barrier that caps Picard on sharp
exp T. Picard's "large stable steps" is a gentle-T capability, not a general one.
