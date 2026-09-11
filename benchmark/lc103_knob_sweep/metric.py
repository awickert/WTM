# METRIC, stated once and used identically for every arm:
#   settle = max |wtd(last saved raster) - wtd(previous saved raster)| over LAND cells (column 0 is
#            ocean and is excluded), in metres. One report interval of motion at the END of the run.
#   n1mm   = how many land cells moved more than 1 mm over that same interval.
#   atsurf = how many land cells sit exactly at wtd == 0.0 in the final raster.
# A converged run drives settle to ~0. A run still in a limit cycle holds it at ~cm indefinitely.
import sys, glob, re, numpy as np, rasterio
stem = sys.argv[1]
fs = sorted(glob.glob(stem + "_0*.tif"), key=lambda p: int(re.search(r"_(\d+)_", p).group(1)))
if len(fs) < 2:
    print("INSUFFICIENT_OUTPUT", len(fs)); sys.exit(3)
a = rasterio.open(fs[-2]).read(1).astype(float)
b = rasterio.open(fs[-1]).read(1).astype(float)
d = np.abs(b - a)[:, 1:]
land = b[:, 1:]
print("%.4e %d %d %d" % (d.max(), int((d > 1e-3).sum()), int((land == 0.0).sum()), land.size))
