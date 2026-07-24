# Design Note: Generalizing WTM to Multiple Layers

**Date:** 2026-07-24
**Branch:** `solver-optimization-2`
**Status:** DESIGN — exploratory. For discussion before any code.
**Authors:** Andy Wickert + Claude

This note sketches what it would take to extend WTM from a single vertically-integrated
water table to **multiple layers** (e.g. an unconfined aquifer over a confined one), and
weighs that against the alternative of bolting WTM's surface-water engine onto a different
groundwater model. It is a design for discussion, not a commitment.

---

## 1. Motivation

WTM today solves a single depth-integrated water table (Fan et al. Dupuit–Forchheimer
approximation): one head per cell, 2D nonlinear diffusion. Two capabilities are out of
reach in that form:

- **Confined aquifers** — a lower aquifer under a confining unit, whose potentiometric
  surface can sit *above* the unit's top, decoupled from the water table above it.
- **Perched systems** — a shallow saturated zone perched on a low-K lens above the
  regional water table, with an unsaturated gap between.

Both are genuinely multi-layer. The question is whether WTM's architecture extends to
them cleanly.

---

## 2. The central reframe: FSM is the crown jewel; the GW solver is commodity

WTM is two pieces:

- **DH + FSM** (`GetDepressionHierarchy` + `FillSpillMerge`, `fill_spill_merge.hpp`) — a
  *specialized* surface-water engine: topologically-exact depression filling, spill,
  merge, and cheap lake placement **without time-stepping overland flow**. This is the
  distinctive science and the hard-to-replace part.
- **The groundwater solver** (`FormFunctionLocal` in `transient_groundwater.cpp`) — a
  *simple, replaceable* 2D Dupuit diffusion, roughly one screen of code.

So "add layers to WTM" means extending the **commodity** part you control; "add DH+FSM to
another model" means porting the **specialized** part onto foreign infrastructure. The
first is almost always the smaller, safer move.

---

## 3. What a multi-layer WTM changes (grounded in the code)

The encouraging headline: **PETSc's DMDA already carries multiple degrees of freedom per
node natively, so the distributed infrastructure built in `DISTRIBUTED_ARP_DESIGN.md`
transfers unchanged.** The work is concentrated in the residual and the physics.

1. **DMDA `dof = 1 → N`.** `CreateSNES.cpp`: `DMDACreate2d(..., 1 /*dof*/, 1 /*stencil*/, ...)`.
   Bump `dof` to the layer count N. PETSc interleaves N heads per node; the SNES solve,
   ghost exchange, and every gather/scatter we wrote (`dmda_gather.hpp`, the per-cycle
   dataflow) handle `dof > 1` with no change. This is the cheap, mechanical part.

2. **The residual — the core work.** `FormFunctionLocal` today writes **one** equation
   per cell: `f = (uxx + uyy)·dt/S + x − rech`. For N layers it writes **N coupled**
   equations per cell — each layer's horizontal 5-point diffusion (with its own T, S)
   **plus a local vertical leakage term** `L·(h_l − h_{l±1})`, where `L = K'/b'` is the
   aquitard leakance. Access moves to `x[j][i][l]` (`DMDAVecGetArrayDOF`). Crucially the
   vertical coupling is *same-cell, adjacent-layer* — **no new halo exchange**, so the
   communication pattern is unchanged.

3. **Per-layer transmissivity & storativity — the physics.**
   `depthIntegratedTransmissivity` and `updateEffectiveStorativity`
   (`transient_groundwater.cpp`) are written for one unconfined table. Keep them for layer
   1 (unconfined: specific yield, T from the Fan S4/S6 form). Add a **confined** branch
   for lower layers (T ≈ K·thickness, constant; storativity = specific storage × thickness,
   tiny), with **convertible-layer** logic when a confined head drops below the unit top —
   the exact case MODFLOW handles. New per-layer parameters: `ksat_l`, layer top/thickness,
   and the inter-layer aquitard leakance.

4. **FSM and recharge stay layer-1-only — and stay clean.** FillSpillMerge only touches
   *surface* water (`arp.wtd > 0`), which is layer 1; confined layers never surface. So FSM
   — and the whole gather-for-FSM machinery this branch optimized — couples to layer 1
   alone. Likewise recharge (including the 1b/2b/2c distributed recharge) applies to the
   top-layer equation; lower layers receive water only through leakage. **The two things
   that dominated the scaling work barely change.**

5. **Storage & I/O.** `ArrayPack`/`AppCtx` static fields become per-layer (or `dof=N`
   vecs); `saveGDAL` writes N bands / N files; `irf.cpp` loads per-layer inputs.

6. **Jacobian (Newton path only).** `FormJacobianLocal` becomes an N×N block per cell
   (leakage couples layers) — more complex but structured. Anderson (the production
   solver) needs none of it.

**Perched systems are the harder case.** A Dupuit two-*saturated*-layer model has no
unsaturated gap — it assumes each layer is saturated to its head. It can *caricature* a
perched wetland as a weakly-connected shallow layer, but true perching needs
unsaturated-zone physics (Richards, or a dedicated perched formulation). Confined aquifers
are a clean `dof=2` extension; perched systems are a larger modeling question.

---

## 4. Alternative: put DH+FSM on a different groundwater model?

Considered and, for the layer-adding goal, **not recommended** — for three concrete
reasons:

1. **FSM is global and serial.** `GetDepressionHierarchy` builds a whole-grid topological
   structure; FSM needs the full grid on one rank. That is exactly the constraint the
   distributed-arp effort navigates ("distribute the solve, gather for FSM"). Bolt FSM
   onto a *distributed* host (ParFlow, parallel MODFLOW) and you rebuild that gather-for-FSM
   machinery from scratch, against someone else's grid abstraction.

2. **A capable host likely already has surface water — and may fight FSM.** ParFlow
   resolves integrated surface–subsurface flow via overland flow; it does not need FSM and
   is far costlier at global/equilibrium scale (which is the whole reason FSM exists).
   MODFLOW is layered and mature but has its own lake package (LAK) and grid/data model;
   FSM's global-serial fill-spill sits awkwardly beside it.

3. **The niche is a matched pair.** WTM + FSM are both built for cheap, large-scale,
   equilibrium-oriented computation. FSM's value is "given this much water, where do lakes
   form — without simulating overland flow." That pairs naturally with a fast equilibrium
   GW solver, not a heavy transient model.

**Decision rule.** If the driver is "WTM but with confined aquifers / multiple layers" →
extend WTM's own solver (`dof=N`); FSM stays coupled to layer 1 unchanged, the niche is
preserved, and it's a contained change. If the driver is fundamentally richer physics that
*abandons* WTM's Dupuit/equilibrium niche → a mature model is right, but in that world FSM
is probably redundant (the host models surface water itself), so you'd retire FSM rather
than port it. The one narrow case for "FSM + different model" — needing one confined lower
aquifer while keeping FSM's cheap lakes — is exactly what a two-layer WTM already gives.

---

## 5. Incremental path

1. **Two layers** (upper unconfined + lower confined), `dof=2`, no leakage — verify the
   solve runs and the top layer reproduces the current single-layer result when the layers
   are decoupled (bit-identical gate, like the distributed-arp increments).
2. **Add vertical leakage** `L·(h_1 − h_2)`; verify against an analytic leaky-aquifer case.
3. **Per-layer T/S** with convertible-layer logic.
4. **Confirm FSM + recharge stay on layer 1**; add per-layer I/O.
5. **Generalize to N** once two layers are solid.

Cost is roughly 2–4× the solve (N× unknowns + coupling); the memory/scaling work reuses
directly, and a `dof=N` DMDA decomposes identically to `dof=1`.

---

## 6. The real limiter is data, not code

Confined aquifers are a well-posed `dof=2` extension of what's already here. The honest
constraint for a *global* two-layer WTM is parameterization: aquitard geometry and vertical
K at 30″ globally are far sparser than the topography/soil-derived `fdepth`/`ksat`/
`porosity` the single-layer model runs on. The solver extends; the global boundary
conditions are the hard part. Worth scoping data availability before investing in Phase 3+.

---

## 7. Open questions

- Which capability is actually driving this — confined aquifers, transient regional flow,
  transport — and is the global/equilibrium niche being kept or left? (This picks `dof=N`
  vs. a different model, per §4.)
- Do the transient endpoints (`_start`/`_end`) and the storativity nonlinearity behave for
  a convertible confined layer, or does the confined→unconfined switch need damping?
- Is `dof=N` interleaving or a per-layer field-split preconditioner better for the Newton
  path at scale? (Anderson is agnostic.)
