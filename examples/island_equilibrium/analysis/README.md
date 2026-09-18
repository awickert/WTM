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
