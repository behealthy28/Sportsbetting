"""
Monte Carlo match simulator for football.

Samples thousands of full-time scorelines from the Dixon-Coles-corrected
joint score distribution, then reports:

  • exact scorelines — how many times each one came up
  • total goals distribution
  • first-half goals distribution (goals split via a binomial draw)

The full-time distribution it samples from is the same one Dixon-Coles uses
for the win/draw/loss probabilities, so the simulation never contradicts the
headline numbers — it just shows the texture underneath them.
"""
from collections import Counter

import numpy as np
from scipy.stats import poisson

# Empirically ~45% of goals in a football match are scored before half-time.
FIRST_HALF_SHARE = 0.45


def _dc_score_matrix(lam_home: float, lam_away: float, rho: float, max_goals: int) -> np.ndarray:
    """Dixon-Coles-corrected joint P(home=h, away=a) matrix, normalised."""
    h = np.arange(max_goals + 1)
    p_home = poisson.pmf(h, lam_home)
    p_away = poisson.pmf(h, lam_away)
    matrix = np.outer(p_home, p_away)

    # Low-score correlation correction (τ)
    matrix[0, 0] *= 1 - lam_home * lam_away * rho
    matrix[1, 0] *= 1 + lam_away * rho
    matrix[0, 1] *= 1 + lam_home * rho
    matrix[1, 1] *= 1 - rho

    matrix = np.clip(matrix, 1e-12, None)
    return matrix / matrix.sum()


def _bucket(values: np.ndarray, n_sims: int, top: int = 5) -> list:
    """Group goal counts into 0,1,...,top-1, "top+". Returns [(label, count, pct)]."""
    out = []
    for g in range(top):
        count = int(np.count_nonzero(values == g))
        out.append((str(g), count, round(100 * count / n_sims, 1)))
    plus = int(np.count_nonzero(values >= top))
    out.append((f"{top}+", plus, round(100 * plus / n_sims, 1)))
    return out


def simulate(
    lam_home: float,
    lam_away: float,
    rho: float = -0.13,
    n_sims: int = 10000,
    max_goals: int = 10,
    first_half_share: float = FIRST_HALF_SHARE,
    top_scorelines: int = 8,
    seed: int = 42,
) -> dict:
    """
    Run a Monte Carlo simulation of the match.

    Returns a dict with scoreline counts, total-goal and first-half-goal
    distributions, and a few derived market lines (over/under, BTTS).
    """
    rng = np.random.default_rng(seed)

    matrix = _dc_score_matrix(lam_home, lam_away, rho, max_goals)
    flat = matrix.ravel()
    idx = rng.choice(flat.size, size=n_sims, p=flat)
    home_goals = idx // (max_goals + 1)
    away_goals = idx % (max_goals + 1)
    total_goals = home_goals + away_goals

    # Split each team's goals into the first half with a binomial draw.
    fh_home = rng.binomial(home_goals, first_half_share)
    fh_away = rng.binomial(away_goals, first_half_share)
    fh_total = fh_home + fh_away

    # Exact scorelines — how many times each came up.
    score_counter = Counter(zip(home_goals.tolist(), away_goals.tolist()))
    scorelines = [
        {
            "home": h,
            "away": a,
            "score": f"{h}-{a}",
            "count": c,
            "pct": round(100 * c / n_sims, 1),
        }
        for (h, a), c in score_counter.most_common(top_scorelines)
    ]

    home_wins = int(np.count_nonzero(home_goals > away_goals))
    draws = int(np.count_nonzero(home_goals == away_goals))
    away_wins = int(np.count_nonzero(home_goals < away_goals))
    btts = int(np.count_nonzero((home_goals > 0) & (away_goals > 0)))

    over_under = {
        line: round(100 * np.count_nonzero(total_goals > line) / n_sims, 1)
        for line in (1.5, 2.5, 3.5)
    }

    return {
        "n_sims": n_sims,
        "lam_home": round(float(lam_home), 2),
        "lam_away": round(float(lam_away), 2),
        "scorelines": scorelines,
        "total_goals": _bucket(total_goals, n_sims),
        "first_half_goals": _bucket(fh_total, n_sims, top=4),
        "avg_total_goals": round(float(total_goals.mean()), 2),
        "avg_first_half_goals": round(float(fh_total.mean()), 2),
        "outcome_counts": {
            "home_win": home_wins,
            "draw": draws,
            "away_win": away_wins,
        },
        "over_under": over_under,
        "btts_pct": round(100 * btts / n_sims, 1),
    }
