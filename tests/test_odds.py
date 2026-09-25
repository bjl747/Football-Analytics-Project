import pytest

from superalgo.odds import (american_to_decimal, decimal_to_american, devig, expected_value,
                            implied_prob, kelly_fraction, prob_to_american)


def test_conversions_round_trip():
    for a in (-250, -110, 100, 150, 400):
        assert decimal_to_american(american_to_decimal(a)) == pytest.approx(a)


def test_implied_and_devig():
    assert implied_prob(-110) == pytest.approx(110 / 210)
    p = devig(-110, -110)
    assert p.sum() == pytest.approx(1.0) and p[0] == pytest.approx(0.5)
    assert prob_to_american(0.5) == pytest.approx(100)


def test_kelly_matches_formula():
    # +100 (b = 1), p = 0.55: full Kelly = (1*0.55 - 0.45)/1 = 0.10
    assert kelly_fraction(0.55, 100, fraction=1.0, max_stake=1.0) == pytest.approx(0.10)
    assert kelly_fraction(0.55, 100, fraction=0.25, max_stake=1.0) == pytest.approx(0.025)


def test_kelly_no_edge_and_uncertainty_shrink():
    assert kelly_fraction(0.50, -110) == 0.0
    assert kelly_fraction(0.56, -110, p_std=0.03) < kelly_fraction(0.56, -110)
    assert kelly_fraction(0.9, 100, max_stake=0.03) == 0.03


def test_expected_value():
    assert expected_value(0.5, 100) == pytest.approx(0.0)
    assert expected_value(0.6, 100) == pytest.approx(0.2)
