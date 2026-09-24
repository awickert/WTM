#!/usr/bin/env python3
"""IS THE CORSICA LIMIT CYCLE A NUMERICAL ARTIFACT?  Two cells, three schemes, a sweep in dt.

SUPERSEDED AS A REDUCTION (2026-09-24) -- KEPT as the discretisation oracle. Three measured defects
mean its NULL ("the reduction settles, so the continuous equations do not oscillate") could not have
come out otherwise: cell B pins at the cap, leaving a SCALAR system where a cycle is impossible by
construction; two cells cannot carry the THIRTEEN-cell mode the real oscillation occupies; and its
fixed point is held up by DISCARDING 86% of the inflow through an outlet throttled 624x by a boundary
30 m below its own surface. See examples/island_equilibrium/OSCILLATION.md and, for the rebuilt
reduction that fixes all three (and still settles -- so the conclusion below stands, on better
evidence), benchmark/chain_numerics/chain.py.

What this file is still GOOD FOR, and why it is not deleted: the LAGGED-vs-IMPLICIT sweep below is a
cheap oracle for "can the discretisation manufacture a cycle like this?" -- run it before attributing
any new oscillation to physics.

WHY THIS EXISTS. examples/island_equilibrium reports that corsica never satisfies the equilibrium stop:
~32 of 14064 cells run a ~210 yr limit cycle of up to 24.5 m. Five explanations were refuted by
measurement (task #111). The last of them was refuted by writing the local physics out as an ODE and
integrating it: with the model's own piecewise T, harmonic-mean interfaces, 5-point stencil and
specificYield storage, an 11x11 patch of the REAL terrain converges MONOTONICALLY to a stable fixed
point. The continuous equations do not oscillate.

That leaves the DISCRETISATION. This script asks the question directly, on the smallest system that can
carry it: two cells, the real parameters, and three time-integration schemes swept over dt.

    EXPLICIT      h^{n+1} = h^n + dt*f(h^n)                    forward Euler
    LAGGED        backward Euler on the linear part, with T FROZEN at the old state.
                  A semi-implicit scheme: unconditionally stable in the linear sense, but the
                  nonlinear coefficient is one step behind. This is the classic source of spurious
                  oscillation in nonlinear diffusion, and it is what a Picard/Anderson iteration
                  DEGENERATES TO if the solve exits before the coefficient has converged -- which is
                  exactly the failure #61 and #104 were opened about.
    IMPLICIT      backward Euler with T at the NEW state, solved by Newton to 1e-12.

THE DISCRIMINATOR. If LAGGED oscillates where IMPLICIT does not, the limit cycle is a lagged-coefficient
artifact and the fix is a convergence question, not a physics one. If none of them oscillates, the two-
cell reduction is too small and the answer is elsewhere. If ALL of them oscillate at large dt but the
amplitude falls with dt, it is ordinary truncation error and not a limit cycle at all.

PARAMETERS are the measured ones for the driver (120,63) and its downhill neighbour (121,63).
"""
import numpy as np

YR = 31536000.0
zA, zB, zC = 1084.0, 874.0, 671.0
fA, fB     = 6.37, 4.63
k, phi, eps = 1e-4, 0.25, 0.01
L   = 928.0
R   = 0.060 / YR
hC  = -30.0

def T(h, f):
    h = np.asarray(h, float)
    return np.where(h < -1.5, f*k*np.exp((h+1.5)/f), np.where(h > 0, k*(1.5+f), k*(h+1.5+f)))
def S(h):
    return 0.5*((1+phi) + h*(1-phi)/np.sqrt(h*h + eps*eps))
def harm(a, b): return 2*a*b/(a+b)

def fluxes(hA, hB, hA_T, hB_T):
    """Mass fluxes.  hA,hB set the HEADS; hA_T,hB_T set the TRANSMISSIVITIES (so they can be lagged)."""
    TA, TB, TC = float(T(hA_T, fA)), float(T(hB_T, fB)), float(T(hC, fB))
    QAB = harm(TA, TB) * ((zA+hA) - (zB+hB)) / L**2
    QBC = harm(TB, TC) * ((zB+hB) - (zC+hC)) / L**2
    return QAB, QBC

def step(hA, hB, dt, scheme, cap):
    if scheme == "explicit":
        QAB, QBC = fluxes(hA, hB, hA, hB)
        nA = hA + dt*(R - QAB)/S(hA)
        nB = hB + dt*(R + QAB - QBC)/S(hB)
    else:
        nA, nB = hA, hB
        for _ in range(60):                       # Newton / fixed point on the backward-Euler system
            tA, tB = (hA, hB) if scheme == "lagged" else (nA, nB)
            QAB, QBC = fluxes(nA, nB, tA, tB)
            rA = nA - hA - dt*(R - QAB)/S(hA)
            rB = nB - hB - dt*(R + QAB - QBC)/S(hB)
            if abs(rA) < 1e-12 and abs(rB) < 1e-12: break
            d = 1e-6                               # numerical Jacobian, 2x2
            J = np.zeros((2, 2))
            for c, (pa, pb) in enumerate(((d, 0.0), (0.0, d))):
                tA2, tB2 = (hA, hB) if scheme == "lagged" else (nA+pa, nB+pb)
                q1, q2 = fluxes(nA+pa, nB+pb, tA2, tB2)
                J[0, c] = ((nA+pa) - hA - dt*(R - q1)/S(hA) - rA)/d
                J[1, c] = ((nB+pb) - hB - dt*(R + q1 - q2)/S(hB) - rB)/d
            try: dx = np.linalg.solve(J, [-rA, -rB])
            except np.linalg.LinAlgError: break
            nA += dx[0]; nB += dx[1]
    if cap:
        nA = min(nA, 0.0); nB = min(nB, 0.0)
    return nA, nB

def run(dt_yr, scheme, years=4000, cap=True):
    hA, hB = -8.89, -23.74
    n = int(years/dt_yr); keep = []
    for i in range(n):
        hA, hB = step(hA, hB, dt_yr*YR, scheme, cap)
        if not np.isfinite(hA) or not np.isfinite(hB) or abs(hA) > 1e4: return None
        if i > n//2: keep.append((hA, hB))
    a = np.array(keep)
    return a[:,0].max()-a[:,0].min(), a[:,1].max()-a[:,1].min(), a[:,0].mean()

# ============================== RESULTS, 2026-09-18 ==============================
#
# THE SCHEME MATTERS AND THE STEP SIZE DOES NOT (at shipped forcing). Cell-A amplitude in the second
# half of a 4000 yr run, sweeping recharge as well as dt, because the threshold moves with both:
#
#            LAGGED T (semi-implicit)                  IMPLICIT T (Newton)
#   R m/yr   0.25    1.00    4.00   16.00   64.00      every dt tested
#   0.02    0.000   0.000   0.000   0.000   0.000          0.000
#   0.06    0.000   0.000   0.000   0.000   8.150          0.000
#   0.15    0.000   0.000   0.000   0.000  97.489          0.000
#   0.40    0.000   0.000   0.000  42.389  48.129          0.000
#   1.00    0.000   0.000   4.450  33.756   0.000          0.000
#
# SO: a LAGGED nonlinear coefficient DOES manufacture a limit cycle in this system, and a fully
# implicit one never does, at any dt or recharge tested. The mechanism class is real and it is a
# numerical artifact, not physics. The threshold moves with recharge -- R 0.06 needs dt 64 yr,
# R 1.00 needs only 4 yr -- so it is the PRODUCT of forcing and step that matters, not dt alone.
#
# BUT IT IS NOT WHAT CORSICA IS DOING, and that was tested rather than assumed. At the model's own
# parameters (R 0.06, dt 1 yr) the lagged scheme gives 0.000 here. And on the REAL model, a 10,000x
# tighter solve -- water_volume_tol 1e-8 -> 1e-12, residual_gate 1e-5 -> 1e-9, both confirmed resolved
# in full_config.yaml -- leaves the oscillation untouched:
#     cell (120,63)  span 24.4915 -> 24.4914   ratio 1.0000
#     cell (121,63)  span 13.1483 -> 13.1483   ratio 1.0000
#     max |baseline - tight| over the whole domain at the final state: 1.93e-04 m
# A solve that exits early behaves like a lagged coefficient (the #61 / #104 failure). This run shows
# the shipped solve is NOT exiting early here: it is already converged, so the coefficient is not
# effectively lagged and this mechanism is excluded.
#
# WHAT THIS SCRIPT IS GOOD FOR GOING FORWARD: it is a cheap oracle for "can the discretisation do
# this?" -- run it before attributing any new oscillation to physics. It answers in seconds where the
# full model takes an hour.
#
# CAVEAT, stated because it bounds the conclusion: two cells settle at A = -32.8, B = 0 (the cap
# active and stable), which is NOT where the real model's cells sit (they cycle 0 to -24.5). The
# reduction is not visiting the same regime, so a null here does not prove a null there. An 11x11
# patch of the real terrain also settles -- see examples/island_equilibrium/demo.py and task #111.
# ================================================================================

print("TWO CELLS, REAL PARAMETERS. Amplitude in the SECOND HALF of a 4000 yr run.")
print("An amplitude that PERSISTS is an oscillation; one that SHRINKS with dt is truncation error.\n")
print("%9s   %-26s %-26s %-26s" % ("", "EXPLICIT", "LAGGED T (semi-implicit)", "IMPLICIT T (Newton)"))
print("%9s   %10s %10s   %10s %10s   %10s %10s" % ("dt (yr)", "A span", "B span", "A span", "B span", "A span", "B span"))
for dt in (0.05, 0.25, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
    row = "%9.2f  " % dt
    for scheme in ("explicit", "lagged", "implicit"):
        r = run(dt, scheme)
        row += ("  %10s %10s" % ("DIVERGED", "")) if r is None else ("  %10.4f %10.4f" % (r[0], r[1]))
    print(row)
