#!/usr/bin/env python3
"""Side-by-side comparison of the two FSM couplings, per solver scheme.

  between -- FillSpillMerge runs between steps and OVERWRITES the water table (original behaviour)
  during  -- FSM's per-cell volume change enters the NEXT step's source term (-wtm_fsm_delta_source)

WHAT THIS DOES AND DOES NOT MEASURE. The two couplings converge to genuinely DIFFERENT equilibria --
under `during`, ponded cells infiltrate instead of being re-pinned full every step. So the per-cycle
rms used here measures how fast each configuration STOPS CHANGING, not how close it is to a true
answer. Read the tables as settling rate and cost. They say nothing about which water table is right.

Usage: compare.py [results_between] [results_during]
"""
import os, sys, importlib.util

BETWEEN = sys.argv[1] if len(sys.argv) > 1 else "results_between"
DURING  = sys.argv[2] if len(sys.argv) > 2 else "results_during"
TARGETS = [100.0, 10.0, 1.0]

# Arms whose dt VARIES during the run. The precision axis here is the per-cycle CHANGE in the water
# table, so a scheme taking deliberately tiny steps reports a tiny rms while being nowhere near
# converged -- dt-continuation starts at dt ~ 0.001 yr and reads as "10 mm after 10 iterations",
# which is a step-size artifact, not convergence. Iso-precision rows for these arms are therefore
# marked and must not be read as cost-to-converge. Fixed-dt arms are unaffected and comparable.
VARIABLE_DT = {"tr_adapt", "newton_cont"}

# reuse report.py's parsers so the two tools can never drift apart
spec = importlib.util.spec_from_file_location("rep", os.path.join(os.path.dirname(__file__), "report.py"))


def load(dirname):
    """{stem: {label, rc, wall, iters, cycles, traj}}"""
    import re
    ITER_RE = re.compile(r"Number of nonlinear iterations = (\d+)")
    CYC_RE = re.compile(
        r"cycle (\d+): per-cycle \|.wtd\| max=[0-9.eE+-]+ rms=([0-9.eE+-]+) .*?rms=([0-9.eE+-]+) mm-water")
    out = {}
    p = os.path.join(dirname, "summary.csv")
    if not os.path.exists(p):
        return out
    for ln in open(p).readlines()[1:]:
        f = ln.rstrip("\n").split(",")
        if len(f) < 6:
            continue
        stem = f[0]
        cum, traj = 0, []
        lp = os.path.join(dirname, stem + ".log")
        if os.path.exists(lp):
            for line in open(lp, errors="ignore"):
                m = ITER_RE.search(line)
                if m:
                    cum += int(m.group(1)); continue
                c = CYC_RE.search(line)
                if c:
                    traj.append((int(c.group(1)), cum, float(c.group(3))))
        out[stem] = dict(label=f[1], rc=int(f[2]), wall=float(f[3]),
                         iters=int(f[4]), cycles=int(f[5]), traj=traj)
    return out


def cost_at(arm, target):
    """(cycles, iters, ~wall) to first reach target rms, or None."""
    for cyc, it, rms in arm["traj"]:
        if rms <= target:
            frac = it / arm["iters"] if arm["iters"] else 0.0
            return cyc, it, arm["wall"] * frac
    return None


def ratio(a, b):
    return f"{a / b:.2f}x" if (b and a) else "--"


B, D = load(BETWEEN), load(DURING)
if not B or not D:
    sys.exit(f"need summary.csv in both {BETWEEN} and {DURING} -- run both couplings first")

print("FSM COUPLING COMPARISON -- island 117x75 (8775 cells), cold start, dt = 1 week, n=4")
print(f"  between = FSM overwrites the table between steps (original)   [{BETWEEN}]")
print(f"  during  = FSM's dV enters the next step's source (-wtm_fsm_delta_source, #116)  [{DURING}]")
print()
print("The two converge to DIFFERENT equilibria, so the rms below is a SETTLING-RATE metric")
print("(how fast each stops changing), NOT accuracy. Speedup > 1 means `during` is cheaper.")
print()

for target in TARGETS:
    print("=" * 108)
    print(f"COST TO REACH rms <= {target:g} mm-water")
    print("=" * 108)
    print(f"{'scheme':<28}{'between: it / ~s':>22}{'during: it / ~s':>22}{'iter speedup':>16}{'wall speedup':>16}")
    print("-" * 108)
    for stem, b in B.items():
        d = D.get(stem)
        lbl = b["label"]
        if not b["traj"] or not d or not d["traj"]:
            note = "did not run" if not b["traj"] else "no `during` arm"
            print(f"{lbl:<28}{note:>22}")
            continue
        cb, cd = cost_at(b, target), cost_at(d, target)
        sb = f"{cb[1]} / {cb[2]:.1f}" if cb else "never"
        sd = f"{cd[1]} / {cd[2]:.1f}" if cd else "never"
        it_sp = ratio(cb[1], cd[1]) if (cb and cd) else "--"
        wl_sp = ratio(cb[2], cd[2]) if (cb and cd) else "--"
        mark = "  <-- variable dt: rms is a step-size artifact, NOT convergence" if stem in VARIABLE_DT else ""
        print(f"{lbl:<28}{sb:>22}{sd:>22}{it_sp:>16}{wl_sp:>16}{mark}")
    print()

print("=" * 108)
print("FULL-BUDGET TOTALS (same cycle budget both sides)")
print("=" * 108)
print(f"{'scheme':<28}{'between it / s':>22}{'during it / s':>22}{'iter ratio':>16}{'wall ratio':>16}")
print("-" * 108)
for stem, b in B.items():
    d = D.get(stem)
    if not d:
        continue
    sb = f"{b['iters']} / {b['wall']:.1f}"
    sd = f"{d['iters']} / {d['wall']:.1f}"
    print(f"{b['label']:<28}{sb:>22}{sd:>22}{ratio(b['iters'], d['iters']):>16}"
          f"{ratio(b['wall'], d['wall']):>16}")

print()
print("=" * 108)
print("SETTLING FLOOR reached within the budget (lower = stopped changing more completely)")
print("=" * 108)
print(f"{'scheme':<28}{'between best/final':>26}{'during best/final':>26}")
print("-" * 108)
for stem, b in B.items():
    d = D.get(stem)
    if not d or not b["traj"] or not d["traj"]:
        continue
    bb, bf = min(r[2] for r in b["traj"]), b["traj"][-1][2]
    db, df = min(r[2] for r in d["traj"]), d["traj"][-1][2]
    fb = f"{bb:.4g} / {bf:.4g}" + (" REGR" if bf > bb * 1.5 else "")
    fd = f"{db:.4g} / {df:.4g}" + (" REGR" if df > db * 1.5 else "")
    print(f"{b['label']:<28}{fb:>26}{fd:>26}")
