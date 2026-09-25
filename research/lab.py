"""Experiment harness: test feature sets and models against real results
and the closing line.

Protocol (to avoid fooling ourselves):
* Development: leave-one-season-out cross-validation over 2012-2022.
* Holdout: 2023-2025, only checked for finalists (``--holdout``).

Scores reported:
  mae        average miss on final margin (lower is better; market ~ 10.2 dev)
  w_mkt      optimal weight on our model when blended with the closing spread
  blend_gain MAE improvement of that blend over the market alone
  ats@k      against-the-spread win rate when |model - line| >= k points
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from superalgo.data import DATA_DIR, load_games  # noqa: E402
from superalgo.features import canon  # noqa: E402

DEV = list(range(2012, 2023))
HOLD = [2023, 2024, 2025]


def load_game_frame() -> pd.DataFrame:
    kp = pd.read_parquet(DATA_DIR / "kalman_preds.parquet")
    g = load_games()
    g = g[(g["season"] >= 2011)].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].map(canon)
    kcols = [c for c in kp.columns if c.startswith("k_") or c.startswith("kv_")]
    h = kp[["game_id", "team"] + kcols].rename(columns={c: "h_" + c for c in kcols}).rename(columns={"team": "home_team"})
    a = kp[["game_id", "team"] + kcols].rename(columns={c: "a_" + c for c in kcols}).rename(columns={"team": "away_team"})
    g = g.merge(h, on=["game_id", "home_team"], how="left").merge(a, on=["game_id", "away_team"], how="left")
    g["neutral"] = (g["location"] == "Neutral").astype(float)
    g["home"] = 1 - g["neutral"]
    g["rest_diff"] = (g["home_rest"].fillna(7) - g["away_rest"].fillna(7)).clip(-7, 7)
    g["wind"] = np.where(g["roof"].isin(["dome", "closed"]), 0, g["wind"].fillna(8))
    g["temp"] = np.where(g["roof"].isin(["dome", "closed"]), 70, g["temp"].fillna(60))
    g["dome"] = g["roof"].isin(["dome", "closed"]).astype(float)
    g["div"] = g["div_game"].fillna(0)
    g["prime"] = g["gametime"].fillna("13:00").str[:2].astype(int).ge(19).astype(float)
    g["playoff"] = (g["game_type"] != "REG").astype(float)
    g["ats_resid"] = g["result"] - g["spread_line"]
    return g


def margin_features(g: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=g.index)
    for m in metrics:
        X[f"d_{m}"] = g[f"h_k_{m}"] - g[f"a_k_{m}"]
    return X


def total_features(g: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=g.index)
    for m in metrics:
        X[f"s_{m}"] = g[f"h_k_{m}"] + g[f"a_k_{m}"]
    return X


def score(pred: np.ndarray, g: pd.DataFrame, target="result", line="spread_line") -> dict:
    y, mk = g[target].to_numpy(float), g[line].to_numpy(float)
    d, r = pred - mk, y - mk
    w = float(np.clip(np.dot(d, r) / max(np.dot(d, d), 1e-9), 0, 1))
    blend = mk + w * d
    out = {"n": len(g), "mae": np.mean(np.abs(pred - y)), "mae_mkt": np.mean(np.abs(mk - y)),
           "w_mkt": w, "blend_gain": np.mean(np.abs(mk - y)) - np.mean(np.abs(blend - y))}
    for k in (1.5, 3, 5):
        sel = np.abs(d) >= k
        win = np.sign(r[sel]) == np.sign(d[sel])
        push = r[sel] == 0
        out[f"ats@{k}"] = float(win[~push].mean()) if (~push).sum() else np.nan
        out[f"n@{k}"] = int(sel.sum())
    return out


def cv_predict(g: pd.DataFrame, X: pd.DataFrame, y: pd.Series, make_model, seasons=DEV,
               holdout=False) -> tuple[np.ndarray, pd.Index]:
    """Leave-one-season-out predictions over ``seasons`` (or train-on-DEV,
    predict-HOLD when ``holdout``)."""
    ok = X.notna().all(axis=1) & y.notna() & g["spread_line"].notna()
    preds, idx = [], []
    folds = [(DEV, HOLD)] if holdout else [([s for s in seasons if s != t], [t]) for t in seasons]
    for tr_s, te_s in folds:
        tr = ok & g["season"].isin(tr_s)
        te = ok & g["season"].isin(te_s)
        m = make_model().fit(X[tr], y[tr])
        preds.append(m.predict(X[te])); idx.append(g.index[te])
    return np.concatenate(preds), np.concatenate(idx)


def run(g, X, make_model, target="result", line="spread_line", holdout=False, label=""):
    p, idx = cv_predict(g, X, g[target], make_model, holdout=holdout)
    s = score(p, g.loc[idx], target, line)
    print(f"{label:<40} " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
    return s, p, idx
