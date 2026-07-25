"""Shared path resolution for the benchmark/picard harnesses.

Keeps the scripts portable: the WTM binary and the synthetic-input tooling are
found relative to the repository, and all scratch output goes under $WTM_WORK
(default /tmp/wtm_picard_bench).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))          # benchmark/picard -> repo root
WTM = os.path.join(REPO, "build", "wtm.x")
SCALING = os.path.join(REPO, "benchmark", "scaling")   # make_synthetic.py / scaling_study.py
WORK = os.environ.get("WTM_WORK", "/tmp/wtm_picard_bench")
os.makedirs(WORK, exist_ok=True)
sys.path.insert(0, SCALING)                            # so `import scaling_study` works
