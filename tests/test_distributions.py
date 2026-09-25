import numpy as np
import pytest

from superalgo import distributions as D


def test_pmfs_sum_to_one():
    k = np.arange(0, 200)
    assert D.poisson_pmf(k, 3.2).sum() == pytest.approx(1)
    assert D.nb_pmf(k, 3.2, 0.4).sum() == pytest.approx(1)
    assert D.zip_pmf(k, 3.2, 0.3).sum() == pytest.approx(1)
    assert D.zinb_pmf(k, 3.2, 0.4, 0.3).sum() == pytest.approx(1)


def test_zip_fit_recovers_parameters():
    rng = np.random.default_rng(1)
    n = 20000
    x = np.where(rng.random(n) < 0.3, 0, rng.poisson(4.0, n))
    res = D.best_fit(x)
    assert res["best"]["dist"] in ("zip", "zinb")
    f = D.fit(x, "zip")
    assert f["mean_param"] == pytest.approx(4.0, rel=0.05)
    assert f["pi"] == pytest.approx(0.3, abs=0.02)


def test_negbin_chosen_for_overdispersed():
    rng = np.random.default_rng(2)
    x = rng.negative_binomial(2, 2 / (2 + 5.0), 20000)
    assert D.best_fit(x, ("poisson", "negbin"))["best"]["dist"] == "negbin"
    assert 0 < D.prob_over(D.fit(x, "negbin"), 4.5) < 1
