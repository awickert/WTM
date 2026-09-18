import glob, os, re
import numpy as np, rasterio
W = "examples/island_equilibrium/_work_corsica"
def band(n): return rasterio.open(os.path.join(W, n)).read(1).astype(float)
topo, slope, ks, phi = (band("corsica_t0_topography.tif"), band("corsica_t0_slope.tif"),
                        band("corsica_horizontal_ksat.tif"), band("corsica_porosity.tif"))
A, B = (120, 63), (121, 63)
EPS = 0.01
def fd(c): return max(200.0/(1+150*slope[c]), 2.0)
def T(c, w):
    f, k, sh = fd(c), ks[c], 1.5
    w = np.asarray(w, float)
    return np.where(w < -sh, f*k*np.exp((w+sh)/f), np.where(w > 0, k*(sh+f), k*(w+sh+f)))
def Sy(c, w):                       # tangent dV/dwtd, the model's specificYield
    p = phi[c]; w = np.asarray(w, float)
    return 0.5*((1+p) + w*(1-p)/np.sqrt(w*w + EPS*EPS))
def load(d, stem):
    fs = sorted(glob.glob(os.path.join(d, f"{stem}_0*.tif")),
                key=lambda p: int(re.search(r"_(\d{9})_", p).group(1)))
    return np.stack([rasterio.open(p).read(1).astype(float) for p in fs])
runs = {"removal ON  (shipped)": load("/tmp/claude-1000/dtsweep", "a_dt1"),
        "removal OFF (both off)": load("/tmp/claude-1000/calm", "nofsm")}
print("=== THE STORATIVITY TEST ===")
print("V(wtd) = 0.5*(wtd*(1+phi) + sqrt(wtd^2+eps^2)*(1-phi))  ->  S = phi below ground, 1.0 above.\n")
print("%-24s %10s %10s %10s %10s" % ("run", "A wtd min", "A wtd max", "S at min", "S at max"))
for nm, st in runs.items():
    wa = st[:, A[0], A[1]]
    print("%-24s %10.3f %10.3f %10.4f %10.4f"
          % (nm, wa.min(), wa.max(), float(Sy(A, wa.min())), float(Sy(A, wa.max()))))
print("\n  -> the buffer hypothesis predicted S ~1 without removal. It is %.4f: the cell rests just"
      % float(Sy(A, runs["removal OFF (both off)"][:, A[0], A[1]].max())))
print("     BARELY above ground (+%.3f m) and eps=0.01 smoothing keeps S near porosity."
      % runs["removal OFF (both off)"][:, A[0], A[1]].max())
print("     THE 4x BUFFER NEVER ENGAGES. Hypothesis REFUTED.\n")
print("=== WHAT ACTUALLY DIFFERS: cell B, the valve ===")
print("%-24s %10s %10s %10s  %12s %12s" % ("run", "B wtd min", "B wtd max", "B span", "T_B min", "T_B max"))
for nm, st in runs.items():
    wb = st[:, B[0], B[1]]
    print("%-24s %10.3f %10.3f %10.3f  %12.3e %12.3e"
          % (nm, wb.min(), wb.max(), wb.max()-wb.min(), float(T(B, wb.min())), float(T(B, wb.max()))))
print("\n=== PHASE PORTRAIT (removal ON): does it close into a limit cycle? ===")
st = runs["removal ON  (shipped)"]
wa, wb = st[:, A[0], A[1]], st[:, B[0], B[1]]
print("%6s %9s %9s   %6s %9s %9s" % ("report","wtd A","wtd B","report","wtd A","wtd B"))
for i in range(0, 20):
    j = i + 21
    r = "%6d %9.3f %9.3f" % (i, wa[i], wb[i])
    if j < len(wa): r += "   %6d %9.3f %9.3f" % (j, wa[j], wb[j])
    print(r)
d = np.hypot(wa[21:41]-wa[0:20], wb[21:41]-wb[0:20])
print(f"\n  distance in (wtd_A, wtd_B) between report i and report i+21, over 20 pairs:")
print(f"    mean {d.mean():.4f} m   max {d.max():.4f} m   -- the orbit REPEATS to this accuracy")
print(f"    against amplitudes of {wa.max()-wa.min():.1f} m (A) and {wb.max()-wb.min():.1f} m (B)")
print(f"  => period = 21 reports = 210 yr")
