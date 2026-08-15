#!/usr/bin/env python3
"""Diagnose the deep per-cycle oscillation in the equilibrium metric.

Given per-cycle wtd tifs (saved with cycles_to_save 1), answer:
  - Does the per-cycle max|Δwtd| metric oscillate (non-monotone), and for which run(s)?
  - WHICH cells dominate it, and WHERE are they (deep wtd<-1.5 / soil -1.5..0 / near-surface >-0.05)?
  - Does the HEAD at those cells oscillate (sign-flipping Δ = a limit cycle) or decay monotonically?
  - Is it a KINK-CROSSING (wtd crosses the -1.5 m soil/deep boundary or 0 surface between cycles)?
  - Is it FEW cells (a metric artifact -> robust metric) or MANY (a real solution oscillation)?
  - Is it adaptive-specific (compare the two runs)?

Usage: diagnose_oscillation.py <stemA> <stemB> <ncycles>   (stems e.g. .../osc_diag/adapt .../osc_diag/const)
Reads <stem>_000000001.tif .. <stem>_00000000N.tif (wtd, negative below surface).
"""
import sys, numpy as np, rasterio

SHALLOW = 1.5  # soil/deep transmissivity kink at wtd = -1.5 m; surface kink at wtd = 0
ncyc = int(sys.argv[3])

def load(stem):
    a = [rasterio.open("%s_%09d.tif" % (stem, n)).read(1).astype(float) for n in range(1, ncyc + 1)]
    return np.array(a)  # (ncyc, ny, nx), wtd

def analyze(name, stem):
    w = load(stem)                       # w[c] = wtd at cycle c+1
    fin = np.all(np.isfinite(w), axis=0) # land cells finite in every cycle
    dw = np.abs(np.diff(w, axis=0))      # (ncyc-1, ny, nx) per-cycle |Δwtd|
    dw_masked = np.where(fin[None], dw, 0.0)
    metric = dw_masked.reshape(ncyc - 1, -1).max(axis=1)
    print("\n=== %s ===" % name)
    print("per-cycle max|Δwtd| (m): " + "  ".join("%.2f" % m for m in metric))
    # the persistent oscillator: cell with the largest TOTAL variation over the LATE half (cycles > ncyc/2)
    late = dw_masked[ncyc // 2:].sum(axis=0)
    jy, jx = np.unravel_index(np.argmax(late), late.shape)
    traj = w[:, jy, jx]
    d = np.diff(traj)
    signflips = int(np.sum(d[:-1] * d[1:] < 0))
    def regime(v): return "deep" if v < -SHALLOW else ("soil" if v < 0 else "surface")
    crosses = int(np.sum((traj[:-1] + SHALLOW) * (traj[1:] + SHALLOW) < 0)  # crosses -1.5
                  + np.sum(traj[:-1] * traj[1:] < 0))                        # crosses 0
    print("dominant late-oscillator cell (y=%d,x=%d):" % (jy, jx))
    print("  wtd trajectory: " + "  ".join("%.2f" % v for v in traj))
    print("  regime: %s (final wtd %.2f m); per-step Δ sign-flips=%d (oscillation if >0); kink-crossings=%d"
          % (regime(traj[-1]), traj[-1], signflips, crosses))
    # metric vs solution: how many cells exceed thresholds on the LAST transition
    last = dw_masked[-1]
    for thr in (1.0, 0.1, 0.01):
        print("  cells with |Δwtd|>%.2fm on last cycle: %d" % (thr, int((last > thr).sum())))
    # regime breakdown of the top-100 movers on the last transition
    idx = np.argsort(last.ravel())[::-1][:100]
    vals = w[-1].ravel()[idx]
    nd = int((vals < -SHALLOW).sum()); nso = int(((vals >= -SHALLOW) & (vals < 0)).sum()); nsu = int((vals >= 0).sum())
    print("  top-100 movers by regime: deep=%d soil=%d surface=%d" % (nd, nso, nsu))
    return metric

mA = analyze("A: " + sys.argv[1], sys.argv[1])
mB = analyze("B: " + sys.argv[2], sys.argv[2])
print("\n=== monotone? (metric strictly decreasing = converging; else oscillating) ===")
for nm, m in (("A", mA), ("B", mB)):
    mono = all(m[i] >= m[i + 1] for i in range(len(m) - 1))
    print("  %s: %s" % (nm, "monotone-decreasing (converging)" if mono else "NON-monotone (oscillating)"))
