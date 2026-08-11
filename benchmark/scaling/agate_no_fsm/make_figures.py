#!/usr/bin/env python3
"""Regenerate the WTM MSI-Agate scaling figures from the CSVs in THIS folder.

Self-contained: reads the three committed data files next to it --
  results_2026-08-11_grid2000.csv, results_2026-08-11_grid4000.csv  (single-node core sweep)
  results_2026-08-11_multinode.csv                                   (node sweep, 2 runs/grid)
-- and writes PNGs alongside. All figure metadata (units, series, the shared-node
variance) is baked in here, so `python3 make_figures.py` reproduces the figures
exactly from the data + this script. The groundwater solve only (fsm_on 0); gw_s is
the bandwidth-bound signal.

Requires: numpy + matplotlib (wtmtest ships numpy; `conda install -n wtmtest matplotlib`
if matplotlib is missing).

    python3 make_figures.py
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDS = [(2000, "2000² (4M cells)", "tab:blue"), (4000, "4000² (16M cells)", "tab:red")]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(r for r in f if not r.lstrip().startswith("#")))


def single(grid):
    """Single-node core sweep -> (cores, gw_speedup, gw_efficiency)."""
    rows = read_csv(os.path.join(HERE, f"results_2026-08-11_grid{grid}.csv"))
    n = np.array([int(r["nranks"]) for r in rows])
    gw = np.array([float(r["gw_s"]) for r in rows])
    o = np.argsort(n); n, gw = n[o], gw[o]
    sp = gw[n == 1][0] / gw                    # GW-solve speedup vs 1 core
    return n, sp, sp / n


def multi(grid):
    """Node sweep -> list of (job_id, nodes, node_speedup, node_efficiency), one per run."""
    rows = [r for r in read_csv(os.path.join(HERE, "results_2026-08-11_multinode.csv"))
            if int(r["grid"]) == grid]
    runs = {}
    for r in rows:
        runs.setdefault(r["job_id"], []).append(
            (int(r["nodes"]), float(r["node_speedup"]), float(r["node_efficiency"])))
    out = []
    for job, pts in runs.items():
        pts.sort()
        a = np.array(pts)
        out.append((job, a[:, 0], a[:, 1], a[:, 2]))
    return out


def _pow2_axes(ax, xt, xmax):
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks(xt); ax.set_xticklabels([str(x) for x in xt])
    ax.set_yticks(xt); ax.set_yticklabels([str(x) for x in xt])
    lim = [1, xmax]; ax.plot(lim, lim, "k--", lw=0.8, label="ideal (linear)")
    ax.grid(True, which="both", alpha=0.3)


# --- Figure 1: single-node strong scaling (GW-solve speedup vs cores) -----------
fig, ax = plt.subplots(figsize=(5, 4))
for g, label, c in GRIDS:
    n, sp, _ = single(g)
    ax.plot(n, sp, "o-", color=c, label=label)
_pow2_axes(ax, [1, 2, 4, 8, 16, 32], 32)
ax.set_xlabel("MPI ranks (cores, one node)"); ax.set_ylabel("GW-solve speedup vs 1 core")
ax.set_title("Single-node strong scaling — MSI Agate (2× EPYC 7763)")
ax.legend()
fig.savefig(os.path.join(HERE, "fig_single_node_scaling.png"), dpi=200, bbox_inches="tight")

# --- Figure 2: multi-node strong scaling (both runs/grid show the variance) ------
fig, ax = plt.subplots(figsize=(5, 4))
for g, label, c in GRIDS:
    for i, (job, nodes, sp, _) in enumerate(multi(g)):
        ax.plot(nodes, sp, "o-", color=c, alpha=0.85, label=(label if i == 0 else None))
_pow2_axes(ax, [1, 2, 4, 8], 8)
ax.set_xlabel("nodes (8 ranks/node)"); ax.set_ylabel("GW-solve speedup vs 1 node")
ax.set_title("Multi-node strong scaling — FSM off\n(2 runs/grid = shared-node variance)")
ax.legend()
fig.savefig(os.path.join(HERE, "fig_multinode_scaling.png"), dpi=200, bbox_inches="tight")

# --- Figure 3: parallel efficiency (single-node vs cores | multi-node vs nodes) --
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
for g, label, c in GRIDS:
    n, _, eff = single(g)
    a1.plot(n, 100 * eff, "o-", color=c, label=label)
a1.set_xscale("log", base=2); a1.set_xticks([1, 2, 4, 8, 16, 32]); a1.set_xticklabels([1, 2, 4, 8, 16, 32])
a1.set_xlabel("cores (one node)"); a1.set_ylabel("parallel efficiency (%)")
a1.set_title("Single-node"); a1.axhline(100, color="k", ls="--", lw=0.8); a1.grid(alpha=0.3); a1.legend()
for g, label, c in GRIDS:
    for i, (job, nodes, _, eff) in enumerate(multi(g)):
        a2.plot(nodes, 100 * eff, "o-", color=c, alpha=0.85, label=(label if i == 0 else None))
a2.set_xscale("log", base=2); a2.set_xticks([1, 2, 4, 8]); a2.set_xticklabels([1, 2, 4, 8])
a2.set_xlabel("nodes (8 ranks/node)"); a2.set_ylabel("parallel efficiency (%)")
a2.set_title("Multi-node"); a2.axhline(100, color="k", ls="--", lw=0.8); a2.grid(alpha=0.3); a2.legend()
fig.suptitle("GW-solve parallel efficiency (bigger grid = better)")
fig.savefig(os.path.join(HERE, "fig_efficiency.png"), dpi=200, bbox_inches="tight")

print("wrote: fig_single_node_scaling.png, fig_multinode_scaling.png, fig_efficiency.png")
