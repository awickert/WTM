#!/usr/bin/env python3
"""Esquibel headline: does T̄ help on the real 384k-cell patch (Kerry's domain)?
Cold-start Anderson +/- T̄ (the primary ask) at a few dt; plus Picard +/- T̄ at 1wk to see whether the
cold-Picard rescue holds on real terrain. Bounded cycles to stay tractable on 166k land cells."""
import os, sys, json
import suite

DOWN = os.path.expanduser("~/Downloads/Esquibel_Data-20260801T205621Z-1-001/Esquibel_Data")

def esq_cfg_template(domain):
    # reuse Kerry's cfg as the template; the driver rewrites dt/cycles/maxiter/output
    import shutil
    src = os.environ.get("WTM_ESQ_CFG",
                         os.path.join(os.environ.get("WTM_SCRATCH", "/tmp/wtm_scratch"), "esq_kerry", "anderson.cfg"))
    dst = os.path.join(domain, "eq_anderson.cfg")
    if not os.path.exists(dst):
        shutil.copy(src, dst)

RUNS = [
    ("anderson",      ["-wtm_anderson"],                 [1, 2, 4]),
    ("anderson_tbar", ["-wtm_anderson", "-wtm_Tbar"],    [1, 2, 4]),
    ("picard",        [],                                [1]),
    ("picard_tbar",   ["-wtm_Tbar"],                     [1]),
]

if __name__ == "__main__":
    domain = DOWN
    esq_cfg_template(domain)
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    maxiter = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 900
    print(f"ESQUIBEL headline (384k cells) cycles={cycles} maxiter={maxiter} timeout={timeout}", flush=True)
    results = []
    for sname, flags, weeks in RUNS:
        for wk in weeks:
            name = f"esqh_{sname}_{wk}wk"
            res = suite.run_one(domain, name, flags, wk * suite.WEEK, cycles, maxiter,
                                "equilibrium", 0, timeout, settle_thresh=5.0)
            res["solver"] = sname; res["weeks"] = wk
            results.append(res)
            print(f"  {sname:14s} {wk:2d}wk: {res['status']:7s} wall={res['wall']:7.1f}s "
                  f"iters={res['tot_iters']:7d} settle@{res['settle']} swt={res['final_swt']:.4g}", flush=True)
    outdir = os.path.dirname(os.path.abspath(__file__))
    json.dump(results, open(os.path.join(outdir, "esq_headline.json"), "w"), indent=1)
    print("DONE", flush=True)
