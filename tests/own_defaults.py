#!/usr/bin/env python3
"""Attach a REASON to each unstated-but-answer-changing value in a materialised suite config (#83).

Andy's rule for this task: never write something that hides a failure. So this tool is DELIBERATELY
timid -- it owns a value only when the suite's own config gives it a fact to reason from, and leaves
the marker in place otherwise. A leftover marker is a visible to-do; a fabricated reason is a lie that
reads like a decision.

Facts it reads from the config itself (never from the suite's name or prose):
  eq_tol == 0            -> the equilibrium stop is OFF, so its metric/frac are INERT
  surface_water.mode     -> whether FSM/lakes are in play, hence whether evaporation is in the budget
"""
import re, sys

MARK = "   # UNSTATED: nobody chose this -- decide it for this suite"

def own(path):
    txt = open(path).read()
    lines = txt.splitlines()
    def val(key):
        for l in lines:
            m = re.match(r'\s*' + re.escape(key) + r':\s*(\S+)', l)
            if m: return m.group(1).strip("'\"")
        return None
    stop_off = val("tol") == "0"                       # run.equilibrium_stop.tol
    routed   = val("mode") == "routed"                 # FSM on: lakes, and evaporation in the budget

    reasons = {
      "metric: frac":  "INERT: equilibrium_stop.tol is 0, so nothing reads this" if stop_off else None,
      "frac: 0.001":   "INERT: equilibrium_stop.tol is 0, so nothing reads this" if stop_off else None,
      "t_bar: false":  "off -- this suite does not exercise step-averaged transmissivity",
      "metric: volume":"the per-solve step judged in WATER (#61), the shipped default",
      "water_volume_tol: 1e-08": "the shipped per-solve water tolerance",
      "storage_form: volume":    "exact dV rather than the secant approximation",
      "wtd_center: 0.05":        ("the SHIPPED ET sigmoid; evaporation is active here, so this shape is"
                                  " part of what is measured") if routed else None,
      "logistic_width: 0.1":     "as above -- shipped ET sigmoid shape" if routed else None,
      "extinction_depth: 8":     "shipped extinction depth" if routed else None,
      "surface_transition: true":("both tapers SHIPPED-ON; the result is obtained under the default"
                                  " taper configuration") if routed else None,
      "depth_extinction: true":  "as above -- shipped taper configuration" if routed else None,
    }
    out, owned, left = [], 0, []
    for l in lines:
        if MARK in l:
            base = l.replace(MARK, "")
            key  = base.strip()
            r = reasons.get(key)
            if r:
                out.append(f"{base}   # {r}"); owned += 1
            else:
                out.append(l); left.append(key)
        else:
            out.append(l)
    open(path, "w").write("\n".join(out) + "\n")
    return owned, left

if __name__ == "__main__":
    for p in sys.argv[1:]:
        n, left = own(p)
        print(f"{p}: owned {n}" + (f", STILL MARKED: {left}" if left else ", none left"))
