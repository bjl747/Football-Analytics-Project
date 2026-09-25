"""Count distributions for pricing discrete events: Poisson, Negative
Binomial, Zero-Inflated Poisson (ZIP) and Zero-Inflated Negative Binomial
(ZINB), with maximum-likelihood fitting and AIC-based model choice.

* Poisson: P(X=k) = lambda^k e^-lambda / k!  (variance = mean)
* Negative Binomial: adds a dispersion parameter for variance > mean.
* Zero-inflated versions: a logistic "gate" first decides whether the event is
  structurally impossible (e.g. a backup TE who never gets a target), then the
  count model runs. P(0) = pi + (1-pi) f(0).

The game engine uses the drive simulator (see simulate.py). These models are
for event counts where they fit the data: player props (the secondary
project), sacks, turnovers, etc.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, special, stats


def poisson_pmf(k, lam):
    return stats.poisson.pmf(k, lam)


def nb_pmf(k, mean, alpha):
    """Negative binomial with mean ``mean`` and variance mean + alpha*mean^2."""
    if alpha <= 0:
        return poisson_pmf(k, mean)
    r = 1.0 / alpha
    return stats.nbinom.pmf(k, r, r / (r + mean))


def zip_pmf(k, lam, pi):
    k = np.asarray(k)
    return np.where(k == 0, pi, 0.0) + (1 - pi) * poisson_pmf(k, lam)


def zinb_pmf(k, mean, alpha, pi):
    k = np.asarray(k)
    return np.where(k == 0, pi, 0.0) + (1 - pi) * nb_pmf(k, mean, alpha)


def _nll(dist, params, x):
    if dist == "poisson":
        p = poisson_pmf(x, np.exp(params[0]))
    elif dist == "negbin":
        p = nb_pmf(x, np.exp(params[0]), np.exp(params[1]))
    elif dist == "zip":
        p = zip_pmf(x, np.exp(params[0]), special.expit(params[1]))
    elif dist == "zinb":
        p = zinb_pmf(x, np.exp(params[0]), np.exp(params[1]), special.expit(params[2]))
    else:
        raise ValueError(dist)
    return -np.sum(np.log(np.clip(p, 1e-300, None)))


_N_PARAMS = {"poisson": 1, "negbin": 2, "zip": 2, "zinb": 3}


def fit(x, dist: str) -> dict:
    """Maximum-likelihood fit of one distribution to observed counts ``x``."""
    x = np.asarray(x, dtype=int)
    m = max(x.mean(), 1e-3)
    start = {"poisson": [np.log(m)], "negbin": [np.log(m), np.log(0.1)],
             "zip": [np.log(m), -2.0], "zinb": [np.log(m), np.log(0.1), -2.0]}[dist]
    res = optimize.minimize(lambda p: _nll(dist, p, x), start, method="Nelder-Mead",
                            options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
    p = res.x
    out = {"dist": dist, "nll": float(res.fun), "aic": 2 * _N_PARAMS[dist] + 2 * float(res.fun)}
    out["mean_param"] = float(np.exp(p[0]))
    if dist in ("negbin", "zinb"):
        out["alpha"] = float(np.exp(p[1]))
    if dist == "zip":
        out["pi"] = float(special.expit(p[1]))
    if dist == "zinb":
        out["pi"] = float(special.expit(p[2]))
    return out


def best_fit(x, candidates=("poisson", "negbin", "zip", "zinb")) -> dict:
    """Fit every candidate and return the one with the lowest AIC, plus all fits."""
    fits = [fit(x, d) for d in candidates]
    best = min(fits, key=lambda f: f["aic"])
    return {"best": best, "all": fits, "mean": float(np.mean(x)), "var": float(np.var(x, ddof=1))}


def pmf_from_fit(f: dict, k):
    d = f["dist"]
    if d == "poisson":
        return poisson_pmf(k, f["mean_param"])
    if d == "negbin":
        return nb_pmf(k, f["mean_param"], f["alpha"])
    if d == "zip":
        return zip_pmf(k, f["mean_param"], f["pi"])
    return zinb_pmf(k, f["mean_param"], f["alpha"], f["pi"])


def prob_over(f: dict, line: float, kmax: int = 200) -> float:
    """P(X > line) for a fitted count model, e.g. an over/under prop."""
    k = np.arange(0, kmax + 1)
    return float(pmf_from_fit(f, k)[k > line].sum())
