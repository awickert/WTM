#!/usr/bin/env python3
"""Report, for each equilibrium run: where it CROSSED the stop tolerance, and where it actually STOPPED.

THE TWO ARE NOT THE SAME, and the gap is the point. The stop test is only evaluated every
report_interval steps, so a run does not halt AT the tolerance -- it halts at the first CHECK after it
got there. A solver taking small steps lands just under the bar; one whose step is growing sails a long
way past it between checks. Reading only "it converged" hides a three-order-of-magnitude difference in
how converged, and turns a cycle count into a false efficiency comparison.

WHAT THE STOP TEST ACTUALLY IS, since "converged" on its own is not a quantity:
    run.equilibrium_stop.metric: frac  with  tol 0.001, frac 0.001
    -> stop once FEWER THAN 0.1% of land cells moved more than 1 mm OF WATER in a whole cycle.
`tol` is metres of WATER (|S*dwtd|), not metres of water table: at porosity 0.25, 1 mm of water is 4 mm
of table movement. The run log reports the whole-cycle motion two ways -- abs_change_volume_max (the
worst single cell) and abs_change_volume_rms -- and when the MAX is already below tol, zero cells
exceed it and the frac test is satisfied with room to spare.

THE LOGS APPEND. Re-running a configuration writes a second block to the same file rather than
replacing it, so a naive line count double-counts. This reads the LAST block only, detected by the
cycle counter restarting.
"""
import glob
import os
import re
import sys

TOL = 0.001


def read_rows(path):
    rows = []
    for line in open(path, errors="replace"):
        f = line.split()
        if not f or not f[0].isdigit():
            continue
        try:
            rows.append((int(f[0]), float(f[23]), float(f[24]), float(f[20]), int(f[21])))
            # f[20] is the column HEADED elapsed_time_s. It is SIMULATED seconds, not wall clock:
            # transient_groundwater.cpp:2466 accumulates deltat into it. There is NO wall time in
            # the run log -- if you need it, time the process.
        except (IndexError, ValueError):
            continue
    # keep only the final block: cycle numbers restart when a run is appended
    start = 0
    for i in range(1, len(rows)):
        if rows[i][0] <= rows[i - 1][0]:
            start = i
    return rows[start:]


def sim_years(d, stem):
    """Simulated time at the end, taken from the last output raster's name."""
    yrs = []
    for p in glob.glob(os.path.join(d, f"{stem}_*yr.tif")):
        m = re.search(r"_(\d+)yr\.tif$", p)
        if m:
            yrs.append(int(m.group(1)))
    return max(yrs) if yrs else None


def main(dirs):
    hdr = ("%-26s %7s %7s %9s %9s %12s %12s %8s"
           % ("run", "stop@", "prev@", "sim yr", "yr/chk", "dvol_max", "prev dvol", "steps"))
    print(hdr)
    print("-" * len(hdr))
    for d in dirs:
        for logp in sorted(glob.glob(os.path.join(d, "out_*_eq_n1.txt"))):
            stem = os.path.basename(logp)[4:-7]          # out_<stem>_n1.txt
            rows = read_rows(logp)
            if not rows:
                print("%-26s  (no data rows)" % stem[:26])
                continue
            cyc, dmax, drms, wall, solves = rows[-1]
            prev = rows[-2] if len(rows) > 1 else None
            # the first cycle in this block at or below tol -- normally the last one, but not always
            crossed = next((r for r in rows if r[1] <= TOL), None)
            yr = sim_years(d, stem)
            sim_yr = wall / 31536000.0          # f[20]: SIMULATED seconds (see read_rows)
            per_check = sim_yr / cyc if cyc else float("nan")
            if yr is not None and abs(yr - sim_yr) > 1.0:
                print("   WARNING %s: raster says %s yr, log says %.1f yr" % (stem[:26], yr, sim_yr))
            print("%-26s %7s %7s %9.0f %9.2f %12.3e %12s %8d"
                  % (stem[:26],
                     crossed[0] if crossed else "never",
                     prev[0] if prev else "-",
                     sim_yr, per_check, dmax,
                     ("%.3e" % prev[1]) if prev else "-",
                     solves))
    print()
    print(f"stop@     first cycle whose whole-cycle motion is at or below tol = {TOL} m of water")
    print("prev@     the check before it -- the run was still above tol here")
    print("dvol_max  abs_change_volume_max AT THE STOP: how converged it actually got, not the bar")
    print("prev dvol the same quantity one check earlier: the size of the jump it took past the bar")
    print("sim yr    SIMULATED years at the stop, from the column headed elapsed_time_s, which")
    print("          accumulates deltat -- it is NOT wall clock, and the log carries no wall time")
    print("yr/chk    simulated years per equilibrium check: how big a bite each cycle took")
    print("steps     solves_done")


if __name__ == "__main__":
    main(sys.argv[1:] or [os.path.join(os.path.dirname(os.path.abspath(__file__)), "_work_spectral")])
