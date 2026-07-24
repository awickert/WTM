#!/usr/bin/env python3
"""Publication figures for the WTM strong-scaling & memory study.

Reads the tidy results CSV (../results_2026-*.csv by default) and writes PNGs
into this folder. The PNGs are .gitignore'd; this script is tracked, so the
figures can be regenerated any time the data changes.

    source ~/models/WTM/msi_env.sh test   # (or any env with pandas + matplotlib)
    python3 make_figures.py [path/to/results.csv]

Figures:
  fig1_strong_scaling      speedup vs cores, per grid (+ ideal linear)
  fig2_parallel_efficiency efficiency vs cores, per grid
  fig3_memory_scaling      total memory vs ranks: after (flat) vs before (grows -> OOM)
  fig4_realized_speedup    after / kcallaghan(n=1) vs cores, per grid
  fig5_cost_tradeoff       speedup vs compute cost (core-seconds), 64M grid -- the
                           diminishing-returns / "how many cores" figure
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

GRIDS = [1000, 2000, 4000, 8000]
GLABEL = {1000: "1000² (1M)", 2000: "2000² (4M)", 4000: "4000² (16M)", 8000: "8000² (64M)"}
# Paul Tol bright, colorblind-friendly
COLOR = {1000: "#4477AA", 2000: "#66CCEE", 4000: "#EE6677", 8000: "#228833"}
MARK = {1000: "o", 2000: "s", 4000: "^", 8000: "D"}
CORES = [1, 2, 4, 8, 16, 32]


def load(path=None):
    if path is None:
        cands = sorted(glob.glob(os.path.join(HERE, "..", "results_*.csv")))
        if not cands:
            sys.exit("no results_*.csv found next to figures/ -- pass a path")
        path = cands[-1]
    df = pd.read_csv(path, comment="#")
    print(f"loaded {path}  ({len(df)} rows)")
    return df


def _log2_cores(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(CORES)
    ax.set_xticklabels(CORES)
    ax.set_xlabel("MPI ranks (cores)")


def _save(fig, name):
    for ext in ("png",):
        p = os.path.join(HERE, f"{name}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)


def fig_strong(df):
    a = df[(df.build == "after") & df.strong_speedup.notna()]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for g in GRIDS:
        s = a[a.grid == g].sort_values("nranks")
        ax.plot(s.nranks, s.strong_speedup, marker=MARK[g], color=COLOR[g],
                lw=1.8, ms=6, label=GLABEL[g])
    ax.plot([1, 32], [1, 32], ls="--", color="0.5", lw=1.2, label="ideal (linear)")
    _log2_cores(ax)
    ax.set_yscale("log", base=2)
    ax.set_yticks([1, 2, 4, 8, 16])
    ax.set_yticklabels([1, 2, 4, 8, 16])
    ax.set_ylabel("strong-scaling speedup vs 1 core")
    ax.set_title("Strong scaling of the distributed groundwater solve (after)")
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.6)
    ax.legend(fontsize=8, title="grid (cells)", title_fontsize=8)
    _save(fig, "fig1_strong_scaling")


def fig_efficiency(df):
    a = df[(df.build == "after") & df.parallel_efficiency.notna()]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for g in GRIDS:
        s = a[a.grid == g].sort_values("nranks")
        ax.plot(s.nranks, 100 * s.parallel_efficiency, marker=MARK[g],
                color=COLOR[g], lw=1.8, ms=6, label=GLABEL[g])
    ax.axhline(100, ls="--", color="0.5", lw=1.0)
    ax.axhline(50, ls=":", color="0.6", lw=1.0)
    ax.text(1.05, 51, "50%", fontsize=7, color="0.4")
    _log2_cores(ax)
    ax.set_ylabel("parallel efficiency (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Parallel efficiency (after) -- the knee tracks cells/rank")
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.6)
    ax.legend(fontsize=8, title="grid (cells)", title_fontsize=8)
    _save(fig, "fig2_parallel_efficiency")


def fig_memory(df, ceiling=64.0):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for g in (4000, 8000):
        for build, ls, fill in (("after", "-", True), ("before", "--", False)):
            s = df[(df.build == build) & (df.grid == g) & (df.rc == 0)].sort_values("nranks")
            ax.plot(s.nranks, s.mem_total_gb, ls=ls, marker=MARK[g],
                    color=COLOR[g], lw=1.8, ms=6,
                    mfc=(COLOR[g] if fill else "white"),
                    label=f"{build} — {GLABEL[g]}")
        # mark before's first OOM rank with an X at the ceiling
        fail = df[(df.build == "before") & (df.grid == g) & (df.rc != 0)].sort_values("nranks")
        if len(fail):
            ax.scatter([fail.nranks.iloc[0]], [ceiling], marker="x", s=90,
                       color=COLOR[g], zorder=6, lw=2)
    ax.axhline(ceiling, color="black", ls=":", lw=1.2)
    ax.text(1.05, ceiling + 1.5, f"{ceiling:.0f} GB session ceiling (× = before OOMs)", fontsize=8)
    _log2_cores(ax)
    ax.set_ylabel("peak total memory (GB, summed over ranks)")
    ax.set_title("Memory: distributed (after, flat) vs replicated (before, grows → OOM)")
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.6)
    ax.legend(fontsize=8)
    _save(fig, "fig3_memory_scaling")


def fig_realized(df):
    a = df[(df.build == "after") & df.realized_speedup_vs_kcallaghan.notna()]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for g in GRIDS:
        s = a[a.grid == g].sort_values("nranks")
        ax.plot(s.nranks, s.realized_speedup_vs_kcallaghan, marker=MARK[g],
                color=COLOR[g], lw=1.8, ms=6, label=GLABEL[g])
    ax.axhline(1.0, ls="--", color="0.5", lw=1.0)
    ax.text(1.05, 1.03, "parity with published (kcallaghan, 1 core)", fontsize=7, color="0.4")
    _log2_cores(ax)
    ax.set_ylabel("realized speedup vs published v2.0.1 (n=1)")
    ax.set_title("End-to-end speedup vs the published single-process model")
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.6)
    ax.legend(fontsize=8, title="grid (cells)", title_fontsize=8)
    _save(fig, "fig4_realized_speedup")


def fig_cost(df, grid=8000):
    """Speedup vs compute cost (core-seconds), for the 64M grid: the
    diminishing-returns / how-many-cores figure. Cost is normalized to n=1."""
    a = df[(df.build == "after") & (df.grid == grid) & (df.rc == 0)].sort_values("nranks")
    base = float(a[a.nranks == 1].wall_s.iloc[0])
    a = a.assign(cost=a.nranks * a.wall_s / base, speedup=base / a.wall_s)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(a.cost, a.speedup, "-", color="0.6", lw=1.2, zorder=1)
    ax.scatter(a.cost, a.speedup, c=[COLOR[grid]], s=60, zorder=2)
    for _, r in a.iterrows():
        ax.annotate(f"{int(r.nranks)}c", (r.cost, r.speedup),
                    textcoords="offset points", xytext=(6, -2), fontsize=8)
    ax.set_xlabel("compute cost  (core-seconds, ÷ 1-core run)")
    ax.set_ylabel("speedup vs 1 core")
    ax.set_title(f"Speedup vs compute cost — {GLABEL[grid]}\n"
                 "(labels = core count; flat-right = diminishing returns)")
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)
    _save(fig, "fig5_cost_tradeoff")


def main():
    df = load(sys.argv[1] if len(sys.argv) > 1 else None)
    fig_strong(df)
    fig_efficiency(df)
    fig_memory(df)
    fig_realized(df)
    fig_cost(df)
    print("done.")


if __name__ == "__main__":
    main()
