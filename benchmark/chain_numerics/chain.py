#!/usr/bin/env python3
"""DOES THE LOCAL PHYSICS OSCILLATE IF THE REDUCTION HAS THE RIGHT SHAPE?  (#111)

WHY THIS REPLACES benchmark/twocell_numerics.  That study concluded the continuous equations do not
oscillate, from a two-cell reduction that settles.  Three measurements (2026-09-24,
examples/island_equilibrium/OSCILLATION.md) show its NULL COULD NOT HAVE BEEN OTHERWISE:

  1. its cell B pins at the surface cap, leaving h_A the only free variable -- and a SCALAR autonomous
     ODE cannot oscillate, so the answer was fixed before any physics was evaluated;
  2. the real oscillation is a FIVE-CELL mode down column 63 (amplitudes 5.97 / 24.48 / 12.70 / 5.13 /
     1.17 at rows 119-123), and two cells cannot carry it;
  3. its equilibrium is held up by a DISCARD: 86% of the inflow leaves through the surface cap because
     its only outlet is throttled 624x by a boundary cell held 30 m below its own surface.

So this rebuild fixes the SHAPE rather than the physics, which is unchanged from twocell.py:

  GEOMETRY   the column-63 chain, rows 118-124, every cell FREE -- no cell is pinned by construction.
  OUTLETS    each chain cell keeps all FOUR lateral outlets, not one: up and down the chain, plus east
             and west to the flanking terrain.
  BOUNDARY   the flanking cells are held at their MEASURED TIME-MEANS, which is fair here and was
             checked rather than assumed: amplitude falls to <=0.217 m by Chebyshev distance 3-5
             against 24.5 m in the middle.  Flanking cells that the real model PINS at the surface are
             held at 0, not at a deep head -- defect 3 above.
  SPACING    anisotropic, from the geotransform: 928 m north-south, 687 m east-west at 42.2 N.
  PARAMETERS the model's own: fdepth = max(200/(1+150*slope), 2) on the real DEM (reproduces
             twocell.py's measured fA 6.37 / fB 4.63 exactly), ksat 1e-4, porosity 0.25, R 0.060 m/yr.

THE DISCRIMINATOR.  If this oscillates with roughly the model's period (220 yr) and amplitude
(24.5 m at the driver), the limit cycle is in the LOCAL CONTINUOUS PHYSICS and the reductions were
simply too small.  If it settles, the mechanism needs something this still omits -- and the omissions
are now few and nameable: the flanking cells' own dynamics, FSM's overland routing, and the rest of
the domain.
"""
import numpy as np, rasterio, math, os, sys

YR = 31536000.0
k, phi = 1e-4, 0.25
R = 0.060 / YR
# THE FREE SET IS CHOSEN BY MEASUREMENT, not by eye.  Every cell whose water table moves more than
# FREE_MIN over one period is free; everything adjacent to one is a boundary.  The first attempt here
# used "the column-63 chain", which LOOKED right on an amplitude map and froze 23% of the moving
# amplitude (cols 64-65 carry 4.59 / 6.18 / 2.85 m) -- it came out 387x too small.
FREE_MIN = 0.1

def T(h, f):
    h = np.asarray(h, float)
    return np.where(h < -1.5, f*k*np.exp((h+1.5)/f), np.where(h > 0, k*(1.5+f), k*(h+1.5+f)))
def S(h, eps=0.01):
    return 0.5*((1+phi) + h*(1-phi)/np.sqrt(h*h + eps*eps))
def harm(a, b): return 2*a*b/(a+b)

def load():
    here = os.path.dirname(os.path.abspath(__file__))
    dem = os.path.join(here, "..", "..", "examples", "island_equilibrium", "corsica_gebco.tif")
    src = rasterio.open(dem); topo = src.read(1).astype(float); gt = src.transform
    lat = gt.f + gt.e*120
    dy = abs(gt.e)*111320.0
    dx = abs(gt.a)*111320.0*math.cos(math.radians(lat))
    gyy, gxx = np.gradient(topo, dy, dx)
    fd = np.maximum(200.0/(1+150.0*np.hypot(gxx, gyy)), 2.0)
    return topo, fd, dx, dy

def build(series):
    """series: (nt, H, W) from the real run.  Returns the free list and the boundary dict."""
    topo, fd, dx, dy = load()
    amp = np.nan_to_num(series.max(0) - series.min(0))
    mx, mean = series.max(0), series.mean(0)
    free = [tuple(v) for v in np.argwhere(amp > FREE_MIN)]
    bnd = {}
    for (r, c) in free:
        for nb in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if nb not in free and nb not in bnd:
                bnd[nb] = 0.0 if mx[nb] >= -1e-9 else float(mean[nb])
    return topo, fd, dx, dy, free, bnd

def rates(h, topo, fd, dx, dy, free, bnd):
    """net inflow rate (m/s of water) for each free cell."""
    idx = {rc: i for i, rc in enumerate(free)}
    out = np.zeros(len(free))
    for i, (r, c) in enumerate(free):
        z0, h0, f0 = topo[r, c], h[i], fd[r, c]
        q = R
        for nb, L in (((r-1,c), dy), ((r+1,c), dy), ((r,c-1), dx), ((r,c+1), dx)):
            hn = h[idx[nb]] if nb in idx else bnd[nb]
            zn, fn = topo[nb], fd[nb]
            q -= harm(float(T(h0, f0)), float(T(hn, fn))) * ((z0+h0) - (zn+hn)) / L**2
        out[i] = q
    return out

def step(h, dt, topo, fd, dx, dy, free, bnd):
    n = h.copy()
    for _ in range(80):                      # backward Euler, Newton with a numerical Jacobian
        res = n - h - dt*rates(n, topo, fd, dx, dy, free, bnd)/S(h)
        if np.abs(res).max() < 1e-11: break
        J = np.zeros((len(h), len(h))); d = 1e-6
        for c in range(len(h)):
            p = n.copy(); p[c] += d
            J[:, c] = ((p - h - dt*rates(p, topo, fd, dx, dy, free, bnd)/S(h)) - res)/d
        try: n = n + np.linalg.solve(J, -res)
        except np.linalg.LinAlgError: break
    return np.minimum(n, 0.0)                # the surface remover

def run(series, years=800, dt_yr=1.0):
    topo, fd, dx, dy, free, bnd = build(series)
    h = np.array([float(series[-1][rc]) for rc in free])      # start from the real state
    keep = []
    for i in range(int(years/dt_yr)):
        h = step(h, dt_yr*YR, topo, fd, dx, dy, free, bnd)
        if not np.all(np.isfinite(h)): return None, free, bnd
        keep.append(h.copy())
    return np.array(keep), free, bnd

# ============================== RESULTS, 2026-09-24 ==============================
#
# THE REDUCTION SETTLES, AND THAT IS NOW A TRUSTWORTHY NULL.
#
#   free set (measured, amp > 0.1 m over one period): 13 cells, rows 118-123, cols 60-65
#   boundaries: 19 cells, 3 of them PINNED at wtd = 0 as the real model pins them
#
#   cell        reduction span     MODEL span
#   (120,63)        0.0046 m         24.52 m      <- the driver
#   (121,63)        0.0018           13.16
#   (119,63)        0.0009            6.00
#   (121,64)        0.0009            6.18
#   (119,64)        0.0008            4.59
#   ... all 13:     < 0.005 m        up to 24.52
#   driver period:  none             220 yr
#
# AND FREEING MORE CELLS MADE IT MORE STABLE, NOT LESS. A first attempt using "the column-63 chain"
# -- which looked right on an amplitude map -- froze cols 64-65 (4.59 / 6.18 / 2.85 m, 23% of the
# moving amplitude) and gave 0.063 m. Freeing the whole measured mode gave 0.005 m. So "the reduction
# was too small" is REFUTED: the local continuous physics does not carry this cycle at any of the
# three sizes tried.
#
# PARAMETERS WERE VERIFIED AGAINST THE FIXTURE rather than assumed, because a null is only as good as
# its inputs: R = (P-E)*(1-runoff_ratio) = (0.2200-0.1000)*0.5 = 0.0600 m/yr, ksat 1.000e-04,
# porosity 0.250, and fdepth from max(200/(1+150*slope), 2) on the real DEM, which reproduces
# twocell.py's measured fA 6.37 / fB 4.63 exactly.
#
# WHAT THIS MEANS FOR #111. twocell.py's conclusion -- "the continuous equations do not oscillate" --
# was RIGHT, but its support was not: that reduction pinned its second cell (making the system
# scalar, where a cycle is impossible by construction), carried two cells of a thirteen-cell mode,
# and held its fixed point up by discarding 86% of the inflow through a shut boundary. This rebuild
# removes all three objections and reaches the same verdict. The null is now worth something.
#
# So the mechanism is NOT in the local continuous physics, and the remaining candidates are few:
# the DISCRETISATION (twocell.py showed a lagged coefficient manufactures cycles, though at forcing
# far from the shipped values), or an interaction with FSM / surface removal that no groundwater-only
# reduction can express -- note that removal is NECESSARY for the real cycle (OSCILLATION.md) and
# this reduction models it only as a cap, never as water routed somewhere else.
# ================================================================================


if __name__ == "__main__":
    # REGENERATES THE TABLE ABOVE from the kept rasters, so the recorded numbers can be re-derived
    # rather than trusted.  Default data: examples/island_equilibrium/_work_111/annual_restart_630yr/
    # (untracked but persistent; see that directory's READINESS.md).  Pass another directory as argv[1].
    import glob, re
    here = os.path.dirname(os.path.abspath(__file__))
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        here, "..", "..", "examples", "island_equilibrium", "_work_111", "annual_restart_630yr")
    fs = sorted(glob.glob(os.path.join(d, "*.tif")),
                key=lambda f: int(re.search(r'_(\d{9})_', f).group(1)))[-220:]   # one full period
    if not fs:
        sys.exit("no rasters in %s -- regenerate with _work_corsica/cfg_REPRO_annual_restart.yaml" % d)
    win = rasterio.windows.Window(55, 112, 20, 20)
    sub = np.stack([rasterio.open(f).read(1, window=win) for f in fs])
    series = np.full((len(fs), 240, 156), np.nan); series[:, 112:132, 55:75] = sub
    model = np.nan_to_num(series.max(0) - series.min(0))

    h, free, bnd = run(series, years=800)
    print("%d reports from %s" % (len(fs), os.path.normpath(d)))
    print("free cells %d (amp > %.2f m); boundaries %d, of them pinned at 0: %d\n"
          % (len(free), FREE_MIN, len(bnd), sum(1 for v in bnd.values() if v == 0.0)))
    if h is None:
        sys.exit("DIVERGED")
    sec = h[len(h)//2:]
    print("%-12s %-14s %-12s" % ("cell", "reduction (m)", "MODEL (m)"))
    for i, rc in enumerate(free):
        print("(%d,%d)%s %-14.4f %-12.2f" % (rc[0], rc[1], " "*(6-len(str(rc[1]))),
                                             sec[:, i].max()-sec[:, i].min(), model[rc]))
    drv = free.index(max(free, key=lambda rc: model[rc]))
    span = sec[:, drv].max() - sec[:, drv].min()
    print("\ndriver (%d,%d): reduction %.4f m vs model %.2f m -- %s"
          % (free[drv][0], free[drv][1], span, model[free[drv]],
             "SETTLES" if span < 0.1 else "OSCILLATES, re-read the conclusion above"))
