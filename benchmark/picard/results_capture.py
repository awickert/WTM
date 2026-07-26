#!/usr/bin/env python3
"""Assemble the Picard/BDF2/adaptive results CAPTURED during development (2026-07-25/26)
into a single queryable CSV. Timing columns (gw_time_s, wall_s) from runs marked
slowed_by_concurrent_processes=yes overlapped other processes and should be RE-RUN clean via the
benchmark/picard harnesses; iteration counts, step counts, and errors are
timing-insensitive and reliable regardless. Long/tidy format: one row per measurement,
empty cell = not-applicable for that experiment.
"""
import csv, os

COLS = ["experiment", "solver", "grid", "ranks", "dt_yr", "tol_mm", "T_yr", "steps",
        "gw_time_s", "wall_s", "outer_its", "inner_its", "steps_to_equil", "converged",
        "mean_err_mm", "max_err_mm", "err_ref", "mem_peak_gb", "slowed_by_concurrent_processes",
        "run_date", "notes"]

rows = []
def add(**kw): rows.append({c: kw.get(c, "") for c in COLS})

# ---------- core+grid sweep: fixed BDF2 (run_type test, maxiter=10, best-of-3) ----------
bdf2_sweep = {  # grid: {n: (gw_s, outer, inner)}
 64:{1:(0.100,2.1,3.0),2:(0.100,2.1,3.0),4:(0.064,2.1,3.0),8:(0.052,2.1,3.0)},
 128:{1:(0.300,1.9,3.0),2:(0.200,1.8,3.0),4:(0.100,1.8,3.0),8:(0.100,1.8,3.0)},
 256:{1:(2.100,3.5,2.0),2:(1.058,3.5,2.0),4:(0.587,3.5,2.0),8:(0.429,3.5,2.0)},
 512:{1:(9.000,4.7,2.8),2:(3.880,4.7,2.0),4:(2.693,4.7,2.0),8:(2.256,4.7,2.4)},
 1024:{1:(91.500,10.1,4.2),2:(42.078,10.1,3.0),4:(31.040,10.1,3.0),8:(30.934,10.1,3.0)}}
for g, d in bdf2_sweep.items():
    for n,(gw,o,i) in d.items():
        add(experiment="core_sweep", solver="bdf2", grid=g, ranks=n, gw_time_s=gw,
            outer_its=o, inner_its=i, slowed_by_concurrent_processes="yes", run_date="2026-07-26",
            notes="run_type test maxiter=10 best-of-3; ran concurrently w/ dt-robustness")

# ---------- core+grid sweep: Anderson vs Picard-BE (n=1 comparison points) ----------
# GWtime (s), outer_avg, inner_avg. Small grids read ~0 (below timer resolution).
cmp_sweep = {  # (grid): {solver: {n: (gw, outer, inner|None)}}
 256:{"anderson":{1:(0.100,4.5,None),2:(0.087,4.5,None),4:(0.061,4.5,None),8:(0.055,4.5,None)},
      "picard-be":{1:(1.700,3.6,2.0),2:(0.800,3.6,2.0),4:(0.500,3.6,2.0),8:(0.422,3.6,2.0)}},
 512:{"anderson":{1:(0.500,4.7,None),2:(0.407,4.7,None),4:(0.347,4.7,None),8:(0.318,4.7,None)},
      "picard-be":{1:(9.300,5.2,2.8),2:(5.240,5.2,2.0),4:(3.948,5.2,2.0),8:(3.961,5.2,2.8)}},
 1024:{"anderson":{1:(5.400,18,None),2:(4.306,18,None),4:(4.270,18,None)},
       "picard-be":{1:(103.900,None,4.7),2:(44.860,None,3.0),4:(35.923,None,3.0),8:(32.340,None,3.0)}}}
for g,ss in cmp_sweep.items():
    for sv,dd in ss.items():
        for n,(gw,o,i) in dd.items():
            add(experiment="core_sweep", solver=sv, grid=g, ranks=n, gw_time_s=gw,
                outer_its=("" if o is None else o), inner_its=("" if i is None else i),
                slowed_by_concurrent_processes="yes", run_date="2026-07-25",
                notes="run_type test maxiter=10 best-of-3")

# ---------- dt robustness (128^2 drainage, maxiter=1): steps to equilibrium ----------
for dt,conv,seq,inn,fc in [(1,"yes",">399",5,""),(10,"yes",">399",5,""),(100,"yes","341",6,""),
                           (1000,"yes","45",6,""),(10000,"yes","11",7,""),(100000,"yes","6",7,"")]:
    add(experiment="dt_robustness", solver="bdf2", grid=128, ranks=1, dt_yr=dt,
        steps_to_equil=seq, inner_its=inn, converged=conv, slowed_by_concurrent_processes="no",
        run_date="2026-07-26", notes="128^2 drainage; unconditionally stable")
for dt,conv in [(1,"yes"),(10,"no"),(100,"no"),(1000,"no"),(10000,"no"),(100000,"no")]:
    add(experiment="dt_robustness", solver="anderson", grid=128, ranks=1, dt_yr=dt,
        converged=conv, slowed_by_concurrent_processes="no", run_date="2026-07-25",
        notes="stability ceiling ~1 yr (diverges at dt>=10)")

# ---------- temporal order: self-convergence vs dt=125yr ref, T=8000yr, mean err ----------
for dt,be,bd in [(2000,4.041,0.7147),(1000,1.949,0.0778),(500,0.851,0.0156),(250,0.287,0.0034)]:
    add(experiment="order", solver="backward-euler", grid=128, ranks=1, dt_yr=dt, T_yr=8000,
        mean_err_mm=be*1000, err_ref="dt=125yr", slowed_by_concurrent_processes="no", run_date="2026-07-26",
        notes="self-convergence; order ->1")
    add(experiment="order", solver="bdf2", grid=128, ranks=1, dt_yr=dt, T_yr=8000,
        mean_err_mm=bd*1000, err_ref="dt=125yr", slowed_by_concurrent_processes="no", run_date="2026-07-26",
        notes="self-convergence; order ~2 coarse dt, ->1 fine dt (C0 Fan T)")

# ---------- error vs dt (BDF2, 128^2, T=8000yr) ----------
for dt,mx,mn,ref in [(10,2.47e-4,1.04e-4,"dt=5yr"),(20,5.75e-4,2.36e-4,"dt=5yr"),
                     (25,7.13e-4,2.89e-4,"dt=5yr"),(40,1.13e-3,4.41e-4,"dt=5yr"),
                     (50,1.44e-3,5.49e-4,"dt=5yr"),(80,2.60e-3,9.33e-4,"dt=5yr"),
                     (100,3.58e-3,1.25e-3,"dt=5yr"),
                     (1,1.86e-4,5.13e-5,"dt=0.5yr"),(2,3.05e-4,1.02e-4,"dt=0.5yr"),
                     (4,4.76e-4,1.75e-4,"dt=0.5yr")]:
    add(experiment="error_vs_dt", solver="bdf2", grid=128, ranks=1, dt_yr=dt, T_yr=8000,
        mean_err_mm=mn*1000, max_err_mm=mx*1000, err_ref=ref, slowed_by_concurrent_processes="no",
        run_date="2026-07-26", notes="fixed-step BDF2")

# ---------- adaptive sweep (equil drainage, tol per-step deviation): steps + gw_time ----------
adapt = {  # grid: {n: (steps, gw_s)} at tol=1.0 mm... note tol here is 1.0 m = 1000 mm? controller tol was 1.0 m
 64:{1:(39,0.400),2:(39,""),4:(39,0.307),8:(39,0.329)},
 128:{1:(39,2.000),2:(39,1.322),4:(39,0.925),8:(39,1.000)},
 256:{1:(38,10.400),2:(38,9.147),4:(38,3.749),8:(38,3.500)},
 512:{1:(40,71.300),2:(40,43.978),4:(40,34.372),8:(40,28.352)},
 1024:{1:(41,363.700),2:(41,274.686),4:(41,258.522),8:(41,239.822)}}
for g,d in adapt.items():
    for n,(st,gw) in d.items():
        add(experiment="adaptive_sweep", solver="adaptive-bdf2", grid=g, ranks=n, tol_mm=1000.0,
            steps=st, gw_time_s=gw, slowed_by_concurrent_processes="yes", run_date="2026-07-26",
            notes="tol=1.0 m per-step; step count invariant to ranks + ~grid-independent")
for g,st in [(64,17),(128,17),(256,17),(512,17),(1024,17)]:
    add(experiment="adaptive_sweep", solver="adaptive-bdf2", grid=g, ranks=1, tol_mm=10000.0,
        steps=st, slowed_by_concurrent_processes="yes", run_date="2026-07-26", notes="tol=10 m per-step (n=1)")

# ---------- adaptive vs fixed dt=1yr at matched accuracy (128^2 drainage, T=8000yr) ----------
# NEGATIVE RESULT: adaptive does NOT beat fixed-1yr here; error is non-monotonic in tol (U-shape,
# min at tol=1e-3), and tight tol explodes the step count -- the controller chases a per-step
# deviation that cannot shrink at the C0 Fan-T threshold kinks. err vs dt=0.5yr reference (mm).
add(experiment="adaptive_vs_fixed", solver="bdf2", grid=128, ranks=1, dt_yr=1, T_yr=8000,
    steps=8000, wall_s=135.3, mean_err_mm=0.0513, max_err_mm=0.1862, err_ref="dt=0.5yr",
    slowed_by_concurrent_processes="maybe", run_date="2026-07-26", notes="fixed dt=1yr baseline")
for tol_m, st, wall, me, mx in [(1e-4,518972,3718.1,1.0775,2.4577),(3e-4,172522,1273.2,0.7196,1.6563),
                                (1e-3,51819,573.5,0.0590,0.1926),(3e-3,17242,229.2,1.5918,3.7669),
                                (1e-2,5160,75.2,15.1160,35.2344)]:
    add(experiment="adaptive_vs_fixed", solver="adaptive-bdf2", grid=128, ranks=1, tol_mm=tol_m*1000,
        T_yr=8000, steps=st, wall_s=wall, mean_err_mm=me, max_err_mm=mx, err_ref="dt=0.5yr",
        slowed_by_concurrent_processes="maybe", run_date="2026-07-26",
        notes="adaptive loses to fixed-1yr; step explosion + non-monotonic err at tight tol (C0-T kinks)")

# ---------- memory (1024^2, -memory_view, peak process RSS) ----------
add(experiment="memory", solver="anderson", grid=1024, ranks=1, mem_peak_gb=0.722,
    slowed_by_concurrent_processes="no", run_date="2026-07-25", notes="peak process RSS")
add(experiment="memory", solver="picard-be", grid=1024, ranks=1, mem_peak_gb=0.891,
    slowed_by_concurrent_processes="no", run_date="2026-07-25", notes="+170 MB for A + GAMG hierarchy")
add(experiment="memory", solver="picard-be", grid=1024, ranks=8, mem_peak_gb=1.674,
    slowed_by_concurrent_processes="no", run_date="2026-07-25", notes="total across 8 ranks; 0.335 max/proc")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} rows to {out}")
