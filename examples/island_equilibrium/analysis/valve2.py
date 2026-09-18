import glob, os, re
import numpy as np, rasterio
W = "examples/island_equilibrium/_work_corsica"; S = "/tmp/claude-1000/dtsweep"
def band(n): return rasterio.open(os.path.join(W, n)).read(1).astype(float)
topo, slope, ks = band("corsica_t0_topography.tif"), band("corsica_t0_slope.tif"), band("corsica_horizontal_ksat.tif")
fs = sorted(glob.glob(os.path.join(S, "a_dt1_0*.tif")), key=lambda p: int(re.search(r"_(\d{9})_", p).group(1)))
st = np.stack([rasterio.open(p).read(1).astype(float) for p in fs])
A, B = (120, 63), (121, 63)
def fd(c): return max(200.0/(1+150*slope[c]), 2.0)
def T_true(c, w):                       # the model's piecewise form, transcribed from :102-122
    f, k, sh = fd(c), ks[c], 1.5
    w = np.asarray(w, dtype=float)
    return np.where(w < -sh, f*k*np.exp((w+sh)/f),
           np.where(w > 0.0, k*(sh+f), k*(w+sh+f)))
def T_expo(c, w):                       # what my earlier scripts used
    f, k = fd(c), ks[c]
    return f*k*np.exp((np.asarray(w,dtype=float)+1.5)/f)
wa, wb = st[:, A[0], A[1]], st[:, B[0], B[1]]
print("HOW WRONG WAS MY EARLIER T?  (driver cell, fdepth %.2f)" % fd(A))
print("%9s %12s %12s %9s" % ("wtd", "T piecewise", "T exponential", "error"))
for w in (-24.5, -10.0, -1.5, -0.5, -0.065, 0.0):
    t1, t2 = float(T_true(A, w)), float(T_expo(A, w))
    print("%9.3f %12.4e %12.4e %8.1f%%" % (w, t1, t2, 100*(t2-t1)/t1))
print("\n  -> the two agree exactly below -1.5 m and differ by ~2% in the shallow band.")
print("     Cell B spends the whole cycle between -23.7 and -10.6 m, i.e. entirely in the")
print("     exponential branch, so the VALVE analysis used the right formula throughout.\n")
Ta, Tb = T_true(A, wa), T_true(B, wb)
Th = 2*Ta*Tb/(Ta+Tb); drop = (topo[A]+wa) - (topo[B]+wb); flux = Th*drop
def nz(v): return (v-v.mean())/v.std()
print("VALVE DECOMPOSITION, recomputed with the model's true piecewise T:")
print("  head DROP  : %7.1f .. %7.1f   varies %4.2fx" % (drop.min(), drop.max(), drop.max()/drop.min()))
print("  T_harmonic : %.3e .. %.3e   varies %4.1fx" % (Th.min(), Th.max(), Th.max()/Th.min()))
print("  corr(flux, head drop) = %+.3f" % float(np.mean(nz(flux)*nz(drop))))
print("  corr(flux, T_harm)    = %+.3f" % float(np.mean(nz(flux)*nz(Th))))
print("  B is the low side of the harmonic mean on %d of %d reports" % (int((Tb<Ta).sum()), len(Tb)))
print("\n  T_A range over the cycle: %.3e .. %.3e  (%.1fx)   CLAMPED above wtd=0 at %.3e"
      % (Ta.min(), Ta.max(), Ta.max()/Ta.min(), float(ks[A]*(1.5+fd(A)))))
print("  reports with wtd_A > 0 (clamp actually engaged): %d of %d" % (int((wa>0).sum()), len(wa)))
