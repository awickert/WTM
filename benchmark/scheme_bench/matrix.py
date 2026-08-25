#!/usr/bin/env python3
"""Full 2x2 comparison: {implicit, active_set} collector x {between, during} FSM coupling.

  implicit   -- in-residual siphon max(0,wtd)/dt. MEASURED dt-DEPENDENT (retained head ~ linear in dt).
  active_set -- semismooth pin at wtd=0 inside the solve. MEASURED dt-INDEPENDENT.
  between    -- FSM overwrites the water table between steps (original WTM).
  during     -- FSM's per-cell dV enters the next step's source term (-wtm_fsm_delta_source, #116).

So `implicit x between` is the ORIGINAL model and `active_set x during` is the full proposed stack.

Reports cost at MATCHED settling precision, and -- separately -- what water table each corner actually
produces. Cost and answer are different questions and are never merged into one "better" column.

Usage: matrix.py [results_root]
"""
import os, re, sys, glob

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
CORNERS = [
    ("results_implicit_between",   "implicit x between  (ORIGINAL)"),
    ("results_implicit_during",    "implicit x during   (#116 only)"),
    ("results_active_set_between", "active-set x between (AS only)"),
    ("results_active_set_during",  "active-set x during  (BOTH)"),
]
TARGETS = [100.0, 10.0, 1.0]
VARIABLE_DT = {"tr_adapt", "newton_cont"}   # per-cycle rms is a step-size artifact for these

ITER_RE = re.compile(r"Number of nonlinear iterations = (\d+)")
CYC_RE = re.compile(
    r"cycle (\d+): per-cycle \|.wtd\| max=[0-9.eE+-]+ rms=([0-9.eE+-]+) .*?rms=([0-9.eE+-]+) mm-water")


def load(dirname):
    out = {}
    p = os.path.join(ROOT, dirname, "summary.csv")
    if not os.path.exists(p):
        return out
    for ln in open(p).readlines()[1:]:
        f = ln.rstrip("\n").split(",")
        if len(f) < 6:
            continue
        stem, cum, traj = f[0], 0, []
        lp = os.path.join(ROOT, dirname, stem + ".log")
        if os.path.exists(lp):
            for line in open(lp, errors="ignore"):
                m = ITER_RE.search(line)
                if m:
                    cum += int(m.group(1)); continue
                c = CYC_RE.search(line)
                if c:
                    traj.append((int(c.group(1)), cum, float(c.group(3))))
        out[stem] = dict(label=f[1], rc=int(f[2]), wall=float(f[3]),
                         iters=int(f[4]), cycles=int(f[5]), traj=traj, dir=dirname)
    return out


def cost_at(arm, target):
    for cyc, it, rms in arm["traj"]:
        if rms <= target:
            frac = it / arm["iters"] if arm["iters"] else 0.0
            return it, arm["wall"] * frac
    return None


DATA = {d: load(d) for d, _ in CORNERS}
present = [(d, lbl) for d, lbl in CORNERS if DATA[d]]
if len(present) < 2:
    sys.exit("need at least two corners; run run.sh with COLLECTOR/COUPLING set")

stems = list(DATA[present[0][0]].keys())

print("SCHEME x COUPLING x COLLECTOR MATRIX -- island 117x75 (8775 cells), cold start, dt = 1 week, n=4")
for d, lbl in present:
    print(f"  {lbl:<34} [{d}]")
print()
print("Cost is read at MATCHED settling rms. The corners converge to DIFFERENT water tables, so cost")
print("and answer are reported separately -- see the final table for which answer each produces.")
print()

for target in TARGETS:
    print("=" * 118)
    print(f"SNES ITERATIONS / ~WALL s TO REACH rms <= {target:g} mm-water")
    print("=" * 118)
    hdr = f"{'scheme':<26}"
    for _, lbl in present:
        hdr += f"{lbl.split('(')[0].strip():>23}"
    print(hdr)
    print("-" * 118)
    for stem in stems:
        row = f"{DATA[present[0][0]][stem]['label']:<26}"
        for d, _ in present:
            a = DATA[d].get(stem)
            if not a or not a["traj"]:
                row += f"{'did not run':>23}"; continue
            c = cost_at(a, target)
            row += f"{(f'{c[0]} / {c[1]:.1f}' if c else 'never'):>23}"
        mark = "   <-- variable dt: rms is a step-size artifact" if stem in VARIABLE_DT else ""
        print(row + mark)
    print()

print("=" * 118)
print("FULL-BUDGET TOTALS: SNES iterations / wall s  (same 250-cycle budget everywhere)")
print("=" * 118)
hdr = f"{'scheme':<26}"
for _, lbl in present:
    hdr += f"{lbl.split('(')[0].strip():>23}"
print(hdr)
print("-" * 118)
for stem in stems:
    row = f"{DATA[present[0][0]][stem]['label']:<26}"
    for d, _ in present:
        a = DATA[d].get(stem)
        cell = "--" if not a else "{} / {:.1f}".format(a["iters"], a["wall"])
        row += f"{cell:>23}"
    print(row)
print()

# ------------------------------------------------------------------ the answer, not the cost
print("=" * 118)
print("THE ANSWER EACH CORNER PRODUCES: final max wtd [m] (lake depth) and ponded-cell count")
print("=" * 118)
try:
    import rasterio, numpy as np
    dom = os.path.join(ROOT, "..", "island", "domain")
    mask = rasterio.open(os.path.join(dom, "Esquibel_010000_mask.tif")).read(1).astype(float) > 0
    hdr = f"{'scheme':<26}"
    for _, lbl in present:
        hdr += f"{lbl.split('(')[0].strip():>23}"
    print(hdr)
    print("-" * 118)
    for stem in stems:
        row = f"{DATA[present[0][0]][stem]['label']:<26}"
        for d, _ in present:
            fs = sorted(glob.glob(os.path.join(ROOT, d, f"{stem}_*_5yr.tif")))
            if not fs:
                row += f"{'--':>23}"; continue
            w = rasterio.open(fs[-1]).read(1).astype(float)[mask]
            row += f"{f'{w.max():.4f} m / {int((w > 0).sum())}':>23}"
        print(row)
    print()
    print("Lake depth is the diagnostic that exposed the collector's dt-dependence: under `implicit` it")
    print("varies with dt (5.38 / 2.50 / 2.02 m at dt = 1, 1/3, 1/6 week), under active-set it does not")
    print("(5.6986 m at every dt). Any corner whose schemes DISAGREE with each other on lake depth is")
    print("showing that dependence, since the schemes differ in effective step size.")
except ImportError:
    print("(rasterio not available -- skipping the water-table table)")
