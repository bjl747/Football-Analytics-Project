"""Dual-state Monte Carlo game simulator (a drive-level Markov chain).

A game is modelled as a chain of drives. Each drive's outcome (TD, FG,
defensive TD, safety, or no score) depends only on the current state, which
is who has the ball, the score margin, and whether we are in the final
minutes, not on how the game got there (the Markov property).

Dual-state design:
* State 1, "pure strength": team scoring rates come from power ratings that
  were fitted with garbage time stripped out.
* State 2, "late game": in the final ~6 minutes the drive outcome rates are
  multiplied by empirically measured late-game factors for each score state.
  Trailing teams keep chasing TDs while leading teams bleed the clock. This
  re-injects the real-world variance that creates backdoor covers.

The same simulator prices pre-game markets (start from 0-0) and live markets
(start from the current score, possession and remaining drives).

Data note: NFL team TD counts are *under*-dispersed (variance < mean) because
drives are a fixed budget shared between TDs, FGs and empty possessions. A
Poisson or Negative Binomial on raw TD counts would overstate the spread of
outcomes, so the game engine uses this drive model. The count distributions
in ``distributions.py`` are used where they fit (player props, low-count events).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import calibrate as _cal
from .calibrate import BUCKETS, BUCKET_LABELS, OUTCOMES


@dataclass
class GameState:
    home_score: int = 0
    away_score: int = 0
    drives_remaining: float | None = None   # None = full game
    home_has_ball: bool | None = None       # None = coin toss
    seconds_remaining: float = 3600.0
    yardline_100: float | None = None      # yards to goal for the team with the ball (live)


@dataclass
class SimResult:
    home: np.ndarray
    away: np.ndarray
    info: dict = field(default_factory=dict)
    weights: np.ndarray | None = None   # key-number importance weights (sum to 1)

    def _w(self) -> np.ndarray:
        if self.weights is None:
            return np.full(len(self.home), 1.0 / len(self.home))
        return self.weights

    def prob(self, mask: np.ndarray) -> float:
        return float(np.sum(self._w() * mask))

    @property
    def margin(self) -> np.ndarray:
        return self.home - self.away

    @property
    def total(self) -> np.ndarray:
        return self.home + self.away

    # ---- market probabilities ------------------------------------------
    def p_home_win(self, ties_half: bool = True) -> float:
        m = self.margin
        return self.prob(m > 0) + (0.5 * self.prob(m == 0) if ties_half else 0.0)

    def p_home_cover(self, home_spread: float) -> dict:
        """home_spread in betting notation (-3.5 = home favoured by 3.5)."""
        adj = self.margin + home_spread
        return {"win": self.prob(adj > 0), "push": self.prob(adj == 0), "lose": self.prob(adj < 0)}

    def p_over(self, line: float) -> dict:
        t = self.total
        return {"over": self.prob(t > line), "push": self.prob(t == line), "under": self.prob(t < line)}

    def fair_spread(self) -> float:
        """Home line (betting notation) at which home covers ~50%."""
        return -_wquantile(self.margin, self._w(), 0.5)

    def most_likely_scores(self, k: int = 5) -> list[tuple[int, int, float]]:
        pairs, inv = np.unique(np.stack([self.home, self.away], 1), axis=0, return_inverse=True)
        probs = np.bincount(inv.ravel(), weights=self._w(), minlength=len(pairs))
        top = np.argsort(-probs)[:k]
        return [(int(pairs[i, 0]), int(pairs[i, 1]), float(probs[i])) for i in top]

    def margin_distribution(self, lo: int = -30, hi: int = 30) -> dict[int, float]:
        m = self.margin
        return {int(v): self.prob(m == v) for v in range(lo, hi + 1)}

    def summary(self) -> dict:
        w = self._w()
        mean = lambda x: float(np.sum(w * x))
        sd = lambda x: float(np.sqrt(np.sum(w * (x - mean(x)) ** 2)))
        return {
            "home_points": mean(self.home), "away_points": mean(self.away),
            "margin_mean": mean(self.margin), "margin_sd": sd(self.margin),
            "total_mean": mean(self.total), "total_sd": sd(self.total),
            "p_home_win": self.p_home_win(), "fair_home_spread": self.fair_spread(),
            "fair_total": _wquantile(self.total, w, 0.5),
            "most_likely_scores": self.most_likely_scores(),
        }


def _wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, q * cw[-1])])


class GameSimulator:
    def __init__(self, calibration: dict | None = None, seed: int | None = None):
        self.cal = calibration or _cal.load()
        self.rng = np.random.default_rng(seed)
        self.late_mult = np.array([[self.cal["late_multipliers"].get(b, {}).get(o, 1.0)
                                    for o in OUTCOMES] for b in BUCKET_LABELS])
        self.use_key_numbers = True

    # ---- rate model ------------------------------------------------------
    def rates_from_ppd(self, ppd: np.ndarray) -> np.ndarray:
        """Offensive points-per-drive -> [p_td, p_fg, p_opp_td, p_safety]."""
        c = self.cal
        ppd = np.clip(ppd, 0.3, 4.5)
        p_td = np.clip(c["td_rate_slope"] * ppd + c["td_rate_intercept"], 0.03, 0.60)
        p_fg = np.clip((ppd - 6.96 * p_td) / 3.0, 0.02, 0.40)
        p_td = np.clip((ppd - 3.0 * p_fg) / 6.96, 0.02, 0.60)
        n = np.broadcast_to
        br = c["base_rates"]
        return np.stack([p_td, p_fg, n(br["opp_td"], ppd.shape), n(br["safety"], ppd.shape)], -1)

    def _td_points(self, n: int) -> np.ndarray:
        pat = self.cal["pat"]
        kick = self.rng.random(n) < pat["kick_share"]
        made = np.where(kick, self.rng.random(n) < pat["kick_make"], self.rng.random(n) < pat["two_make"])
        return 6 + np.where(kick, made * 1, made * 2)

    # ---- simulation -----------------------------------------------------
    def simulate(self, home_exp_pts: float, away_exp_pts: float, n_sims: int = 20000,
                 state: GameState | None = None, pace: float = 1.0,
                 dual_state: bool = True, _tune: bool = True) -> SimResult:
        """Simulate a game where the home/away teams would score
        ``home_exp_pts``/``away_exp_pts`` against each other on average in a full
        game (use ratings.expected_points to get these)."""
        c = self.cal
        state = state or GameState()
        mean_drives = c["drives_per_game_mean"] * pace
        # iterate so simulated means hit the targets (late-game effects shift them)
        if _tune:
            h_ppd, a_ppd = self._tuned_ppd(home_exp_pts, away_exp_pts, mean_drives, pace, dual_state)
        else:
            h_ppd, a_ppd = home_exp_pts / (mean_drives / 2), away_exp_pts / (mean_drives / 2)
        return self._run(h_ppd, a_ppd, n_sims, state, mean_drives, dual_state)

    def _tuned_ppd(self, h_pts, a_pts, mean_drives, pace, dual_state):
        h, a = h_pts / (mean_drives / 2), a_pts / (mean_drives / 2)
        rng_state = self.rng.bit_generator.state
        self.rng = np.random.default_rng(12345)
        for _ in range(3):
            r = self._run(h, a, 6000, GameState(), mean_drives, dual_state)
            h *= h_pts / max(r.home.mean(), 1e-6)
            a *= a_pts / max(r.away.mean(), 1e-6)
        self.rng.bit_generator.state = rng_state
        return h, a

    def _run(self, h_ppd, a_ppd, n, state: GameState, mean_drives, dual_state) -> SimResult:
        c, rng = self.cal, self.rng
        sd = c["drives_per_game_sd"] * c.get("pace_sd_scale", 1.0)
        frac = 1.0 if state.drives_remaining is None else None
        if state.drives_remaining is None:
            drives = np.maximum(np.rint(rng.normal(mean_drives, sd, n)), 10).astype(int)
        else:
            dr = state.drives_remaining
            drives = np.maximum(np.rint(rng.normal(dr, max(0.15 * dr, 0.3), n)), 0).astype(int)
        late_n = np.rint(c["late_drive_share"] * mean_drives)
        # game-day form: each team's efficiency varies game to game
        fs = c["form_sd"] if frac else c["form_sd"] * 0.5
        h_eff = h_ppd * rng.lognormal(-fs ** 2 / 2, fs, n)
        a_eff = a_ppd * rng.lognormal(-fs ** 2 / 2, fs, n)
        h_rates, a_rates = self.rates_from_ppd(h_eff), self.rates_from_ppd(a_eff)

        home = np.full(n, state.home_score, dtype=int)
        away = np.full(n, state.away_score, dtype=int)
        if state.home_has_ball is None:
            home_ball = rng.random(n) < 0.5
        else:
            home_ball = np.full(n, state.home_has_ball)

        fp = self.cal.get("drive_by_yardline")
        for j in range(int(drives.max()) if n else 0):
            active = j < drives
            late = dual_state & (j >= drives - late_n)
            off_rates = np.where(home_ball[:, None], h_rates, a_rates)
            if j == 0 and state.yardline_100 is not None and fp:
                # the drive in progress: league odds from this field position,
                # scaled by how good this offence is relative to average
                b = int(np.clip(state.yardline_100 // 10, 0, 9))
                base = np.array(fp["rates"][fp["buckets"].index(b)] if b in fp["buckets"] else fp["rates"][-1])
                ppd = np.where(home_ball, h_eff, a_eff)
                k = np.clip(ppd / 2.0, 0.6, 1.5)[:, None]
                off_rates = np.column_stack([base[0] * k[:, 0], base[1] * k[:, 0],
                                             np.full(n, base[2]), np.full(n, base[3])])
                off_rates[:, :2] = np.minimum(off_rates[:, :2], 0.95)
            diff = np.where(home_ball, home - away, away - home)
            b = np.clip(np.searchsorted(BUCKETS, diff, side="left") - 1, 0, len(BUCKET_LABELS) - 1)
            rates = np.where(late[:, None], off_rates * self.late_mult[b], off_rates)
            s = rates.sum(1, keepdims=True)
            rates = np.where(s > 0.95, rates * 0.95 / s, rates)
            cum = np.cumsum(rates, 1)
            u = rng.random(n)
            outcome = (u[:, None] > cum).sum(1)  # 0 td, 1 fg, 2 opp_td, 3 safety, 4 none
            pts_off = np.where(outcome == 0, self._td_points(n), np.where(outcome == 1, 3, 0))
            pts_def = np.where(outcome == 2, self._td_points(n), np.where(outcome == 3, 2, 0))
            pts_off *= active; pts_def *= active
            home += np.where(home_ball, pts_off, pts_def)
            away += np.where(home_ball, pts_def, pts_off)
            home_ball = np.where(active, ~home_ball, home_ball)

        self._overtime(home, away, h_rates, a_rates)
        res = SimResult(home, away, {"home_ppd": float(np.mean(h_ppd)), "away_ppd": float(np.mean(a_ppd))})
        kn = self.cal.get("key_number_weights")
        if kn and self.use_key_numbers:
            table = np.asarray(kn["weights"])
            off = kn["max_margin"]
            m = np.clip(res.margin, -off, off) + off
            w = table[m]
            res.weights = w / w.sum()
        return res

    def _overtime(self, home, away, h_rates, a_rates, max_drives: int = 4):
        """Modern NFL OT: both teams get a possession, then sudden death."""
        tied = np.flatnonzero(home == away)
        if not len(tied):
            return
        rng = self.rng
        h_first = rng.random(len(tied)) < 0.5
        done = np.zeros(len(tied), bool)
        for j in range(max_drives):
            h_ball = h_first ^ (j % 2 == 1)
            r = np.where(h_ball[:, None], h_rates[tied], a_rates[tied])[:, :2]
            u = rng.random(len(tied))
            pts = np.where(u < r[:, 0], self._td_points(len(tied)), np.where(u < r[:, 0] + r[:, 1], 3, 0))
            pts = np.where(done, 0, pts)
            home[tied] += np.where(h_ball, pts, 0)
            away[tied] += np.where(h_ball, 0, pts)
            if j >= 1:  # after both teams have had the ball, first score wins
                done |= home[tied] != away[tied]


def calibrate_noise(sim: GameSimulator, target_margin_sd: float, target_total_sd: float,
                    exp_pts: float = 22.8, n: int = 40000) -> dict:
    """Grid-search game-day form noise and pace spread so that simulated
    outcomes are as uncertain as real NFL results are around the closing line."""
    best = None
    for fs in np.arange(0.0, 0.31, 0.05):
        for ps in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
            sim.cal["form_sd"], sim.cal["pace_sd_scale"] = float(fs), float(ps)
            sim.rng = np.random.default_rng(7)
            r = sim.simulate(exp_pts, exp_pts, n_sims=n, _tune=False)
            err = (r.margin.std() - target_margin_sd) ** 2 + (r.total.std() - target_total_sd) ** 2
            if best is None or err < best[0]:
                best = (err, fs, ps, r.margin.std(), r.total.std())
    sim.cal["form_sd"], sim.cal["pace_sd_scale"] = float(best[1]), float(best[2])
    return {"form_sd": best[1], "pace_sd_scale": best[2], "margin_sd": best[3], "total_sd": best[4]}


def fit_key_number_weights(sim: GameSimulator, games, max_margin: int = 40,
                           n_per_game: int = 400, smooth: float = 50.0) -> dict:
    """Learn how much more (or less) often real NFL games land on each final
    margin than the simulator predicts, e.g. real games finish on exactly 3
    or 7 points more often.

    For every historical game we simulate from the closing spread and total
    (the market's view, so this corrects only the *shape* of the distribution),
    then compare real vs simulated frequencies of each margin relative to the
    favourite. The ratio is used as an importance weight on simulated games.
    """
    g = games.dropna(subset=["result", "spread_line", "total_line"])
    size = 2 * max_margin + 1
    sim_counts = np.zeros(size)
    real_counts = np.zeros(size)
    old = sim.use_key_numbers
    sim.use_key_numbers = False
    keys = np.round(g[["spread_line", "total_line"]].to_numpy() * 2) / 2
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    for i, (spread, total) in enumerate(uniq):
        k = int((inv == i).sum())
        r = sim.simulate((total + spread) / 2, (total - spread) / 2, n_sims=n_per_game * k)
        sgn = 1 if spread >= 0 else -1  # measure margins from the favourite's side
        m = np.clip(sgn * r.margin, -max_margin, max_margin) + max_margin
        sim_counts += np.bincount(m, minlength=size) / n_per_game
        real = g.iloc[np.flatnonzero(inv == i)]
        rm = np.clip(sgn * real["result"].to_numpy(int), -max_margin, max_margin) + max_margin
        real_counts += np.bincount(rm, minlength=size)
    sim.use_key_numbers = old
    # shrink toward 1 where data is thin
    w_fav = (real_counts + smooth * sim_counts / sim_counts.sum()) / \
            (sim_counts + smooth * sim_counts / sim_counts.sum())
    # simulator margins are home-based; key-number effects are symmetric, so average both sides
    w = (w_fav + w_fav[::-1]) / 2
    return {"max_margin": max_margin, "weights": w.tolist(),
            "real_counts": real_counts.tolist(), "sim_counts": sim_counts.tolist()}
