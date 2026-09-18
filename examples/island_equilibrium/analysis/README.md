# The analysis behind OSCILLATION.md

These are the scripts that produced every number in `../OSCILLATION.md`. They are kept because the
document states conclusions and the conclusions are only as good as the method — and re-deriving that
method from prose is a rewrite, not a check.

**They are investigation scripts, not a suite.** They read rasters a run has already written, print,
and assert nothing. Nothing in `tests/` depends on them. Paths are hard-coded to the work directories
the demo produces, so they need editing to point somewhere else.

## What each one answered

| script | question | result |
|---|---|---|
| `remeasure.py` | recompute the stop metric over a whole run, from post-FSM snapshots | `frac` bottoms at 0.001351 against a 0.001 threshold and never goes below |
| `space.py` | which cells never quiet, and what do they share | 32 of 14064; median topo 1172 m vs 428, median slope 0.264 vs 0.134 |
| `timeseries.py` | is it flicker or something slower | a smooth ~210 yr cycle, 5–7 sign flips in 59 steps — **not** period-2 |
| `lakes.py` | is FSM doing something different each cycle | lake count 254 and volume 4751.7081 m **bit-identical** for 60 cycles |
| `flux.py` | per-face flux between the driver and its 8 neighbours, through the cycle | the outflow to the downhill cell surges 6× while the driver is *falling* |
| `valve2.py` | is the flux set by the gradient or by the conductance | gradient 1.16×, conductance 8.1×, `corr(flux, T_harm) = +0.998`. Uses the model's TRUE piecewise `T` |
| `osc.py` | does the orbit close, and what is the storativity at rest | closes to 3.0 m after 21 reports against 24.5 m amplitude; `S = 0.9896` at the rest point |
| `bbudget.py` | where does the downhill cell get its water | **its budget does NOT close** — see the caveat in `OSCILLATION.md` and do not reason from it |
| `block.py` / `block11.py` | does a faithful ODE reduction of the local physics oscillate | **NO.** 5×5 and 11×11 patches of the real terrain converge monotonically to a stable fixed point |

`block11.py` is generated from `block.py` by the `sed` in the commit that added them; they differ only
in the patch size and which cell indices are read.

## The two that matter most

**`valve2.py`** carries the one decomposition that survived: flux is conductance-controlled, not
gradient-controlled. It also carries the correction that made it trustworthy — an earlier version used
the exponential `T` branch at all depths, where the model's `T` is piecewise. The error was 2.2% in the
shallow band and exactly zero below −1.5 m, which is where the neighbour lives, so the conclusion held.
Check that before trusting any new script here.

**`block.py` / `block11.py`** carry the strongest negative result in the investigation, and it refuted
a mechanism that had already been written up. The equations do not oscillate. Anything proposing a
local mechanism has to explain why an 11×11 patch of the same terrain, with the same `T`, the same
harmonic means, the same stencil and the same storage, settles.

## A caveat kept with the scripts, not only in the document

`bbudget.py`'s flux budget **does not close**: adding the downhill cell's recharge to its measured
lateral net predicts it wets *without* surface removal, which is the opposite of the measured
trajectory. The estimate ignores the real finite-volume discretisation, the per-face areas and the
within-cycle timing. It is adequate for the 1.16×-versus-8.1× contrast in `valve2.py` and **not**
adequate for a budget. Re-derive from the model's own fluxes before building on it.

## REPRODUCING ANY OF IT — read this before running a script

**The scripts point at data that no longer exists.** They were written against
`/tmp/claude-1000/dtsweep` and `/tmp/claude-1000/calm`, which do not survive a reboot, and against
`../_work_corsica`, which is gitignored and therefore local to whichever machine produced it. Committing
the scripts without saying this would leave someone editing paths for an hour before discovering there
is nothing behind them.

**The configs ARE preserved**, in `configs/` — 17 of them, one per ablation arm, with the vanished
`/tmp` paths replaced by an `ANALYSIS_DIR` placeholder you must substitute. They are the expensive part
to reconstruct; the runs themselves are one command each.

### The order that matters

Everything except the first step starts from **one restart raster**: the settled 20000 yr state,
`_work_corsica/anderson_fixed_dt31536000_eq_n1_000002000_20000yr.tif` (498 448 bytes). Every ablation
is a 400 yr continuation from it, which is why they are cheap and the spin-up is not.

```
# 1. THE EXPENSIVE ONE, hours. Produces the restart state and the 2001 snapshots
#    remeasure.py / space.py / timeseries.py / lakes.py read.
python3 examples/island_equilibrium/demo.py corsica --solver anderson --equilibrium --ranks

# 2. EVERY ABLATION, ~10-30 min each. Substitute ANALYSIS_DIR in the config first.
sed -i "s|ANALYSIS_DIR|$PWD/examples/island_equilibrium/analysis|" <config>
./build/wtm.x examples/island_equilibrium/analysis/configs/<arm>.yaml
```

### Which config produced which result

| config | arm | what it showed |
|---|---|---|
| `a_dt1`, `b_dt025` | `dt` 1 yr vs 0.25 yr | amplitude ratio 0.995–1.000 — **`dt`-invariant** |
| `c_active_set`, `c_explicit`, `c_implicit` | three collectors at `impulse`/`fixed` | spans 24.4966 / 24.4085 / 24.4877 — the removal law does not matter |
| `nofsm` | `routing: off` AND `collection: off` | the driver rests at +0.042 m — **removal is necessary** |
| `A_noFSM_as`, `B_FSM_nocoll` | the two removers split | either one alone restores the full cycle |
| `calm`, `lowP`, `fdb` | `E_ow` 0.15, `P` 0.16, `fdepth_b` 15 | `E_ow` refuted; the other two were **invalid** — a domain-wide change from a state equilibrated under the old one is a transient, not a limit cycle |
| `ext1`, `ext30` | extinction depth 1 m / 30 m | **bit-identical** — `A(wtd)` multiplies `(E_eff−P)₊`, zero here |
| `pond` | 10 cm ponding allowance (a SOURCE PATCH, not a config — see `#111`) | span changes 0.003%, and the cell never uses it |
| `tight` | `water_volume_tol` 1e-12, `residual_gate` 1e-9 | ratio 1.0000 — the solve was already converged |
| `lag` | `post_groundwater` on, every step | the FSM source is stale by 1.58e-06 settled, 17.6% in transient |
| `tr` | FSM trace on | FSM runs once per **step**, not per cycle |

`pond` cannot be reproduced from its config alone: it needed a one-line patch to `WTM.cpp:499`
(`std::max(0.0, ...)` → `std::max(0.10, ...)`), the writer `continuous` actually uses. The first
attempt patched `transient_groundwater.cpp:2336` instead and measured nothing, because that writer is
not on this path.
