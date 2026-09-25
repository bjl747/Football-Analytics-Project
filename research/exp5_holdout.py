"""Holdout check (2023-2025) for experiment 5 finalists. Run once."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp5_full_model as E  # noqa: E402  (rebuilds features; prints dev results)
from lab import run  # noqa: E402

print("\n=== HOLDOUT 2023-2025 ===")
for label, X, M in (("ridge points only", E.feats(["points"], ["home"]), E.RIDGE),
                    ("ridge perf+mkt", E.feats(E.PERF + ["mkt_points"], ["home"]), E.RIDGE),
                    ("ridge perf+mkt+ctx", E.feats(E.PERF + ["mkt_points"], E.CTX), E.RIDGE),
                    ("xgb perf+mkt+ctx", E.feats(E.PERF + ["mkt_points"], E.CTX), E.XGB),
                    ("ridge mkt+ctx+inj", E.feats(["mkt_points", "points"], E.CTX + E.INJ), E.RIDGE)):
    run(E.g, X, M, holdout=True, label=label)
