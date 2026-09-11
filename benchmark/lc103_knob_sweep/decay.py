# METRIC, stated once, identical for every arm. Read from the run log, not from rasters, so it is a
# TIME SERIES and can separate a decaying transient from a non-decaying limit cycle.
#   series = abs_change_volume_max, the model's own per-cycle |S*dwtd| max, one value per cycle.
#   early  = max over cycles 5..14   (past the initial shock)
#   late   = max over the last 10 cycles
#   ratio  = late/early.  ~1 means the motion is NOT decaying; << 1 means it is a transient dying out.
import sys
L=[l for l in open(sys.argv[1]) if l.strip()]
h=[l for l in L if l.startswith("Cycles_done")][0].split()
rows=[dict(zip(h,l.split())) for l in L if l and l[0].isdigit()]
s=[float(r["abs_change_volume_max"]) for r in rows]
if len(s) < 25: print("SHORT %d" % len(s)); sys.exit(3)
early=max(s[5:15]); late=max(s[-10:])
print("%d %.4e %.4e %.3f" % (len(s), early, late, (late/early) if early else float("nan")))
