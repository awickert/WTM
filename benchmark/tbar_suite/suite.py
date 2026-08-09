#!/usr/bin/env python3
"""-wtm_Tbar test suite. Drives wtm.x over a solver x Tbar x dt matrix and records, per run:
convergence/stability, wall-clock, total nonlinear iterations, cycles-to-settle, and the final
equilibrium (for accuracy). Two suites: (1) cold-start-to-equilibrium dt sweep; (2) warm-start
perturbation dt-to-failure. Emits CSV + a human-readable table.

Usage: suite.py <suite:cold|warm> <domain_dir> <out_tag>
Env: PROJ_DATA=/usr/share/proj must be set; OMP_NUM_THREADS as desired."""
import os, sys, time, subprocess, re, glob, json
import numpy as np

WTM = "/home/awickert/models/WTM/build/wtm.x"
WEEK = 604800

def read_se(domain):
    p = os.path.join(domain, "_se.txt")
    return open(p).read().strip() if os.path.exists(p) else None

def base_cfg(domain):
    # prefer an existing equilibrium cfg in the domain as the template
    for c in ("eq_anderson.cfg", "island.cfg", "dmon.cfg"):
        p = os.path.join(domain, c)
        if os.path.exists(p):
            return p
    raise SystemExit("no template cfg in " + domain)

def make_cfg(domain, name, deltat, cycles, maxiter, run_type="equilibrium", supplied=0, save_every=None):
    tmpl = open(base_cfg(domain)).read()
    se = read_se(domain)
    save = save_every if save_every else cycles  # save first + last only by default
    subs = {
        r"^surfdatadir.*":     f"surfdatadir        {domain}",
        r"^deltat.*":          f"deltat             {deltat}",
        r"^total_cycles.*":    f"total_cycles       {cycles}",
        r"^maxiter.*":         f"maxiter            {maxiter}",
        r"^run_type.*":        f"run_type           {run_type}",
        r"^supplied_wt.*":     f"supplied_wt        {supplied}",
        r"^cycles_to_save.*":  f"cycles_to_save     {save}",
        r"^textfilename.*":    f"textfilename       {domain}/{name}.txt",
        r"^outfile_prefix.*":  f"outfile_prefix     {domain}/{name}_",
    }
    if se is not None:
        subs[r"^southern_edge.*"] = f"southern_edge      {se}"
    out = tmpl
    for pat, rep in subs.items():
        out = re.sub(pat, rep, out, flags=re.M)
    path = os.path.join(domain, name + ".cfg")
    open(path, "w").write(out)
    # clear stale text output so cycle parsing is fresh
    open(os.path.join(domain, name + ".txt"), "w").close()
    return path

def parse_log(text):
    text = text.replace("\r", "\n")
    conv = len(re.findall(r"CONVERGED", text))
    div = len(re.findall(r"DIVERGED", text))
    iters = [int(m) for m in re.findall(r"nonlinear iterations = (\d+)", text)]
    return conv, div, sum(iters), len(iters)

def parse_txt_settle(domain, name, thresh):
    """Return (cycles_run, settle_cycle_or_None, final_sum_wtd). col1=cycle, col5=abs_total_wtd_change,
    col11=sum_of_water_tables."""
    p = os.path.join(domain, name + ".txt")
    cyc, col5, col11 = [], [], []
    for line in open(p):
        s = line.split()
        if s and re.match(r"^\d+$", s[0]):
            cyc.append(int(s[0]))
            col5.append(float(s[4]) if len(s) > 4 else np.nan)
            col11.append(float(s[10]) if len(s) > 10 else np.nan)
    if not cyc:
        return 0, None, np.nan
    # relative settle: first cycle where the domain-total |Δwtd| falls below 1% of its cycle-1 value
    # (or the absolute floor `thresh`, whichever is larger) -- a cross-solver-comparable equilibrium proxy.
    ref = max(thresh, 0.01 * (col5[0] if col5 else thresh))
    settle = next((c for c, v in zip(cyc, col5) if v < ref), None)
    return cyc[-1], settle, col11[-1]

def final_raster(domain, name):
    tifs = sorted(glob.glob(os.path.join(domain, name + "_*.tif")))
    return tifs[-1] if tifs else None

def run_one(domain, name, flags, deltat, cycles, maxiter, run_type="equilibrium", supplied=0,
            timeout=300, settle_thresh=0.5):
    make_cfg(domain, name, deltat, cycles, maxiter, run_type, supplied)
    cmd = [WTM, os.path.join(domain, name + ".cfg")] + flags
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        wall = time.time() - t0
        rc = r.returncode
        log = r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        wall = time.time() - t0
        rc = -9
        log = (e.stdout or "") + (e.stderr or "") if isinstance(e.stdout, str) else "TIMEOUT"
    conv, div, tot_iters, nsolves = parse_log(log)
    cyc_run, settle, final_swt = parse_txt_settle(domain, name, settle_thresh)
    status = "OK" if (rc == 0 and div == 0) else ("TIMEOUT" if rc == -9 else "FAIL")
    return dict(name=name, status=status, rc=rc, wall=round(wall, 1), tot_iters=tot_iters,
                nsolves=nsolves, div=div, cyc_run=cyc_run, settle=settle,
                final_swt=final_swt, raster=final_raster(domain, name))

# Solver definitions (cold-start). Newton needs dt-continuation to start from cold.
# Newton is EXCLUDED from the dt sweep: its dt-continuation controls the step internally (config dt only
# caps the ramp), so a config-dt sweep is not meaningful for it. Newton +/- Tbar is compared separately
# at a single setting (see newton_pair below). The sweep compares the frozen-coefficient solvers whose
# step size IS the config dt -- exactly where Tbar's temporal averaging should matter.
SOLVERS = [
    ("anderson",      ["-wtm_anderson"]),
    ("anderson_tbar", ["-wtm_anderson", "-wtm_Tbar"]),
    ("picard",        []),                       # default = Picard BDF2-on-V
    ("picard_tbar",   ["-wtm_Tbar"]),
]

def newton_pair(domain, tag, wk=1, cycles=40, maxiter=20, timeout=400):
    """Newton +/- Tbar at one dt via -wtm_stiff (continuation + eq_tol early-stop). Iteration count and
    wall to equilibrium are the comparison (does Tbar reduce Newton's cost / continuation length?)."""
    out = []
    for sname, flags in (("newton", ["-wtm_stiff"]), ("newton_tbar", ["-wtm_stiff", "-wtm_Tbar"])):
        name = f"{tag}_cold_{sname}_{wk}wk"
        res = run_one(domain, name, flags, wk * WEEK, cycles, maxiter, "equilibrium", 0, timeout)
        res["solver"] = sname; res["weeks"] = wk
        out.append(res)
        print(f"  {sname:14s} {wk:2d}wk: {res['status']:7s} wall={res['wall']:6.1f}s "
              f"iters={res['tot_iters']:6d} settle@{res['settle']} swt={res['final_swt']:.4g}", flush=True)
    return out

def suite_cold(domain, tag, weeks=(1, 2, 4, 8, 16), cycles=40, maxiter=20, timeout=200):
    results = []
    for wk in weeks:
        dt = wk * WEEK
        for sname, flags in SOLVERS:
            name = f"{tag}_cold_{sname}_{wk}wk"
            res = run_one(domain, name, flags, dt, cycles, maxiter, "equilibrium", 0, timeout)
            res["solver"] = sname; res["weeks"] = wk
            results.append(res)
            print(f"  {sname:14s} {wk:2d}wk: {res['status']:7s} wall={res['wall']:6.1f}s "
                  f"iters={res['tot_iters']:6d} settle@{res['settle']} swt={res['final_swt']:.4g}",
                  flush=True)
    return results

if __name__ == "__main__":
    suite = sys.argv[1] if len(sys.argv) > 1 else "cold"
    domain = sys.argv[2] if len(sys.argv) > 2 else \
        "/tmp/claude-1000/-home-awickert-models-WTM/ff1a9122-d3f3-4054-acc7-66b5a35ca781/scratchpad/esq_island"
    tag = sys.argv[3] if len(sys.argv) > 3 else "s1"
    print(f"SUITE={suite} DOMAIN={domain} TAG={tag}", flush=True)
    if suite == "cold":
        res = suite_cold(domain, tag)
        print("Newton +/- Tbar (single setting; continuation-driven):", flush=True)
        res += newton_pair(domain, tag, wk=1)
    else:
        raise SystemExit("warm suite in suite_warm.py")
    outdir = os.path.dirname(os.path.abspath(__file__))
    json.dump(res, open(os.path.join(outdir, f"{tag}_{suite}.json"), "w"), indent=1)
    print("DONE ->", os.path.join(outdir, f"{tag}_{suite}.json"), flush=True)
