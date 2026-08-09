# Porting checklist: `bdf2-adaptive-dt` → Callaghan's WTM (v2.0.1)

**Purpose.** This branch is a Skunkworks that explored many solver/numerics/physics paths. The intended
deliverable to upstream (KCallaghan `master` = v2.0.1) is a **slim, distilled set of the best edits**, not
a merge of the whole branch. This file is the living, maintainable checklist of *what to port and why*,
with the "why" graded as **accuracy** (quantified), **speed** (quantified), **correctness** (bug fix), or
**other**. Keep it current as items land or are dropped.

**Guiding principles for the port**
- Minimize the diff against v2.0.1; prefer **additive, off-by-default flags** (production path byte-identical).
- Lead with what measurably helps Kerry's **two first-class modes — equilibrium and transient**.
- **Kerry runs Anderson** (`-snes_type anderson`) for both modes, and it is the wall-clock workhorse
  (matrix-free, MPI bit-exact; ~25× faster than Picard at 384k cells). So the solver strategy is
  **Anderson-centric**: port the improvements that apply to Anderson; keep Picard/Newton as opt-in
  specialists. (Do **not** port a Picard *default* — it diverges on Kerry's cold 1-week-dt workflow.)
- Separate **numerics that don't change the answer** (safe) from **physics/discretization changes that do**
  (must be deliberate, reviewed, and communicated to Kerry).

Legend — Port?: ✅ recommend · 🔬 promising, finish/verify first · ⚠️ deliberate change, review with Kerry · ❌ do not port.
Status is on THIS branch. All flags are off by default unless noted.

---

## A. Solver / numerics improvements applicable to Anderson (the priority set)

| Feature (flag) | Why — graded & quantified | Changes the answer? | Status | Port? |
|---|---|---|---|---|
| **Time-averaged transmissivity `-wtm_Tbar`** | **Speed/robustness.** 2× larger stable cold step (island 4→8 wk; **real 384k Esquibel 1→2 wk at Kerry's 1-wk setting**); ~4 % fewer Anderson iters at 1 wk → ~11 % at 4 wk; **cures the cold-start Picard divergence** on island & Esquibel. | **No** — same equilibrium (Anderson ± T̄ agree to ≤0.005 m mean at each dt). | Shipped, verified (FD, MPI, regression, off-path byte-identical). | ✅ |
| **2nd-order-in-time Anderson `-wtm_anderson -wtm_bdf2_on_V`** | **Accuracy.** Upgrades Anderson from 1st-order backward-Euler (what v2.0.1/Kerry run today) to genuine 2nd-order. Transient RMS ≈ **0.023 m (BDF2) vs 0.044 m (BE)** at 1-wk sub-steps (~2×; `finding-solver-by-regime`). Anderson's stable step **unchanged** (4 wk BE = 4 wk BDF2). | **No** at steady state (T̄→T; same equilibrium as Picard BDF2-on-V to 1.1e-2 m); differs mid-transient (that's the accuracy gain). | Shipped. MPI-consistent to 1.2e-3 m (looser than BE's µm; near-steady V-difference sensitivity). | ✅ |
| **TR-BDF2 `-wtm_tr_bdf2`** | **Accuracy + robustness (quantified).** L-stable, *strongly/monotonically* damped 2nd-order. Warm 2× perturbation: **2× the stable step of BE/BDF2-on-V (8 vs 4 wk)** and **~6× fewer iters near the ceiling** (387 vs 2563 at 4 wk — no BDF2 ringing), for ~11 % more work/step (two staged solves). | **No** — same equilibrium as BDF2-on-V (mean 0.039 m). | Shipped, verified (pure-GW sane, MPI 2.7e-3 m, off-path byte-identical). | ✅ (better 2nd-order option than plain BDF2 on Anderson) |
| **Predictor-seeded guess `-wtm_predict_guess`** | **Speed (quantified).** Seeds the guess (hence iter-1 T̄) with a guarded 2nd-order history extrapolation (forward-Euler on the first step). With T̄: ~11 % fewer iters on a 2-wk warm transient, **~48 % at 8-wk steps** (grows with dt), ~34 % on the first step. Helps *only* with T̄ (no coefficient to improve otherwise). **Does NOT raise the step ceiling** (that's set by the operator, not the guess) — measured. | **No** — same answer to solver tolerance. | Shipped, verified (regression, off-path byte-identical). | ✅ pairs with T̄ |
| **`-wtm_stiff` (Newton + dt-continuation + eq_tol)** | **Robustness (other).** Reaches cold-start equilibrium on stiff terrain at large dt where Anderson diverges and plain Newton/Picard fail cold. NOT a speed win (slower than Anderson where Anderson works). | Same equilibrium (it's a path to it). | Shipped. | 🔬 keep as opt-in specialist for hard cold-starts; not a default. |
| **Bedrock transmissivity floor `-wtm_T_bedrock`** | **Physics-correctness only (no measured solver win).** Additive background T (Manning–Ingebritsen) removing the unphysical exp→0 extrapolation and the deep-cell operator singularity. But NO operating window at production dt: small dt self-regularizes (S/dt), so it's inert at Kerry's 1 wk (2028 vs 2021 iters) and on shallow domains. `finding-bedrock-floor`. | Negligible at production dt (~0.01%); would shift the deep table only if raised. | Shipped, off by default (0). | 🔬 keep as opt-in physics knob; LOW priority — not a speed lever. |

---

## B. Parallelism / scaling

*(Key category per project lead. Quantification of speedup vs the v2.0.1 baseline is a TODO — see below.)*

| Feature | Why — graded | Status | Port? |
|---|---|---|---|
| **MPI-distributed solve (DMDA) correctness** — ghost-cell exchange, MPI-consistency across rank counts | **Correctness/other.** Serial ≡ parallel verified: ghost-cell test 0.0 m; MPI-consistency matrix (evap×fsm × np 2/4/6/8) all pass; golden results rank-independent. | Verified (regression suite). | ✅ (correctness backbone for any scaling) |
| **Distributed forcing / reduced replicated memory** (`DISTRIBUTED_ARP_DESIGN.md`) | **Other (scaling).** Replicated memory is the deployment bottleneck (single-node many-core on MSI); distributing the forcing fields reduces per-rank memory → larger domains / more ranks. | On branch. | ✅ (aligns with the parallelism goal) |
| **FSM parallelization** | **Other (scaling).** FSM is the serial ceiling. | DECIDED-park (memory-only driver, no speed case). | ❌/park unless a speed case appears |
| **TODO — quantify parallel scaling vs v2.0.1** | **Speed (unquantified).** No measured strong/weak-scaling speedup vs the v2.0.1 baseline this session, and it is unconfirmed how much MPI v2.0.1 already had. **Measure before claiming a speed "why" for parallelism.** | Open | — |

---

## C. Physics / discretization changes that MOVE Kerry's results (deliberate, review with Kerry)

These are **not** transparent add-ons — they change her numbers. Port each as its own reviewed change with the effect stated.

| Change | Why — graded & quantified | Port? |
|---|---|---|
| **Conservative-FV flux + grid-convention fix** (E-W/N-S cell-size swap, cos²lat, non-conservative N-S faces; `GRID_CONVENTION.md`) | **Correctness (bug fix) — but LARGE effect.** Dominant divergence from v2.0.1: **~2.5 m deep-cell mean, up to ~16 m**; v2.0.1 drains deeper (range −35 vs −30 m). This is ~80× the taper effect and is the single biggest behavior change. | ⚠️ Port (it's a genuine correction) but **flag explicitly to Kerry** — it silently shifts her deep water table by meters. |
| **Surface-transition tapers 1–3 + FSM-handoff sink** (`-wtm_surface_sink`, `-wtm_evap_taper`, `-wtm_extinction`; default ON) | **Physics model (other).** Smooth surface transition replacing the hard wtd=0 switch. **Small effect: ~0.009 m mean, 0.60 m max** (larger in deep cells via the changed surface BC). Toggleable; all-off = the legacy hard-switch = v2.0.1 surface physics. | ⚠️ Port as an **opt-in** (consider default-off upstream to preserve v2.0.1 behavior unless Kerry wants the smooth model). |
| **Recharge rescale to rate×(actual dt)** | **Correctness** for variable-dt paths (adaptive / continuation); **byte-identical on fixed-dt** (=1.0). | ✅ (safe; needed by any variable-dt feature) |

---

## D. Do NOT port (negative results — recorded so they aren't re-tried)

| Item | Why not |
|---|---|
| **Kirchhoff transform as the solve variable** (`-wtm_kirchhoff`) | Reaches identical equilibrium (8.7e-8 m) but **worsens conditioning** — the storage term's 1/T blows up the diagonal for deep cells. Conditioning is *conserved* under a variable change, not removed. |
| **Log-transform of the equation (variable)** | Same reason — the water-table depth is already ~log(T) in the deep regime; a log/Kirchhoff variable moves ill-conditioning from flux to storage, no net gain. The log/Kirchhoff structure pays off only as a *coefficient* (= `-wtm_Tbar`). *(Volume-form Kirchhoff untested — the one open variant.)* |
| **Picard (BDF2-on-V) as the DEFAULT** | Merge regression for Kerry: diverges on her cold + 1-week-dt workflow; ~25× slower than Anderson at 384k; GAMG not mesh-independent under exp-T. Keep Picard opt-in only. |
| **Auto-Dupuit initial guess** | Dropped — dt-continuation makes the guess irrelevant. |
| **Anderson-accelerated GAMG-Picard** (`-wtm_aa_picard`) | Converges warm but expensively (a GAMG solve every iteration); **fails cold-start-to-equilibrium** (the Picard nonlinear preconditioner hard-fails mid-drainage → `DIVERGED_INNER`, aborting the outer solve). Buys GAMG's per-iteration cost + a new fragility, not steps+robustness. Kept off-by-default as a documented dead-end. See `benchmark/AA_PICARD.md`. |
| **Power-law transmissivity** | Would need data calibration (changes physics); did not help the cold-start Newton basin. |

---

## Evidence / provenance

Measurements: `benchmark/tbar_suite/` (drivers + machine-readable JSON), `benchmark/TBAR_TIME_AVERAGING.md`,
`benchmark/EQUILIBRIUM_ROBUSTNESS.md`, `benchmark/GRID_CONVENTION.md`. Test bed: ocean-ringed Esquibel
island (75×117) + real 384k Esquibel patch, 1-week dt (Kerry's setting). v2.0.1 comparison binary:
`/home/awickert/models/WTM-mastertest`.

**Open quantification TODOs:** (1) parallel scaling speedup vs v2.0.1; (2) a from-scratch
order-of-accuracy study on the Anderson BDF2 / TR-BDF2 paths (confirm p≈2); (3) confirm the v2.0.1 MPI
baseline. [TR-BDF2 ceiling/robustness measured: 2× the BDF2 step, ~6× fewer iters near the ceiling.]
