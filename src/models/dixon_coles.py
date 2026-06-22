"""
Dixon-Coles Poisson model for football/cricket outcome prediction.
Gold standard for score-based sports.
"""
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from typing import Optional


def _tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles low-score correction factor (τ)."""
    if x == 0 and y == 0:
        return 1 - lam_h * lam_a * rho
    elif x == 1 and y == 0:
        return 1 + lam_a * rho
    elif x == 0 and y == 1:
        return 1 + lam_h * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _score_probability(
    score_home: int,
    score_away: int,
    lam_home: float,
    lam_away: float,
    rho: float,
) -> float:
    """P(home scores X, away scores Y) with Dixon-Coles correction."""
    tau = _tau(score_home, score_away, lam_home, lam_away, rho)
    p = (
        tau
        * poisson.pmf(score_home, lam_home)
        * poisson.pmf(score_away, lam_away)
    )
    return max(p, 1e-10)


def _outcome_probs(
    lam_home: float,
    lam_away: float,
    rho: float = -0.13,
    max_goals: int = 10,
) -> dict:
    """
    Integrate over score matrix to get home win / draw / away win probabilities.
    rho ≈ -0.13 is the empirically derived Dixon-Coles correlation parameter.
    """
    home_win = draw = away_win = 0.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = _score_probability(h, a, lam_home, lam_away, rho)
            if h > a:
                home_win += p
            elif h == a:
                draw += p
            else:
                away_win += p

    total = home_win + draw + away_win
    return {
        "home_win": round(home_win / total, 4),
        "draw": round(draw / total, 4),
        "away_win": round(away_win / total, 4),
    }


class DixonColesModel:
    """
    Dixon-Coles model estimating team attack/defense strengths
    via Maximum Likelihood Estimation on historical match data.
    """

    HOME_ADVANTAGE = 0.25  # additive log-scale advantage for home team

    def __init__(self):
        self.attack: dict = {}
        self.defense: dict = {}
        self.rho: float = -0.13
        self.is_fitted: bool = False

    def _lambda(self, attack: float, defense: float, home: bool = False) -> float:
        ha = self.HOME_ADVANTAGE if home else 0.0
        return math.exp(attack + defense + ha)

    def fit(self, matches: list, xi: float = 0.001) -> None:
        """
        Fit model on match history.
        matches: list of {home_team, away_team, home_goals, away_goals, weight?}
        xi: time-decay weight (higher = faster decay)
        """
        teams = sorted(set(
            [m["home_team"] for m in matches] + [m["away_team"] for m in matches]
        ))
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        def neg_log_likelihood(params):
            attack = {t: params[i] for i, t in enumerate(teams)}
            defense = {t: params[n + i] for i, t in enumerate(teams)}
            rho = params[2 * n]

            ll = 0.0
            for m in matches:
                ht, at = m["home_team"], m["away_team"]
                hg, ag = int(m["home_goals"]), int(m["away_goals"])
                w = float(m.get("weight", 1.0))

                lam_h = math.exp(attack[ht] + defense[at] + self.HOME_ADVANTAGE)
                lam_a = math.exp(attack[at] + defense[ht])
                lam_h = max(lam_h, 0.01)
                lam_a = max(lam_a, 0.01)

                tau = _tau(hg, ag, lam_h, lam_a, rho)
                if tau <= 0:
                    tau = 1e-8

                ll += w * (
                    math.log(tau)
                    + hg * math.log(lam_h) - lam_h - _log_factorial(hg)
                    + ag * math.log(lam_a) - lam_a - _log_factorial(ag)
                )
            return -ll

        x0 = np.zeros(2 * n + 1)
        x0[-1] = -0.13  # rho initial

        # Constraint: sum of attack = 0 (identifiability)
        constraints = [{"type": "eq", "fun": lambda p: sum(p[:n])}]

        try:
            result = minimize(
                neg_log_likelihood,
                x0,
                method="L-BFGS-B",
                bounds=[(None, None)] * (2 * n) + [(-0.5, 0.5)],
                options={"maxiter": 500, "ftol": 1e-9},
            )
            params = result.x
        except Exception:
            params = x0

        for i, t in enumerate(teams):
            self.attack[t] = float(params[i])
            self.defense[t] = float(params[n + i])
        self.rho = float(params[2 * n])
        self.is_fitted = True

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
    ) -> dict:
        """Predict outcome probabilities for a match."""
        if not self.is_fitted:
            return self._fallback_predict(home_team, away_team)

        ha = self.attack.get(home_team, 0.0)
        hd = self.defense.get(home_team, 0.0)
        aa = self.attack.get(away_team, 0.0)
        ad = self.defense.get(away_team, 0.0)

        home_adv = 0.0 if neutral else self.HOME_ADVANTAGE
        lam_home = math.exp(ha + ad + home_adv)
        lam_away = math.exp(aa + hd)

        lam_home = max(lam_home, 0.1)
        lam_away = max(lam_away, 0.1)

        return _outcome_probs(lam_home, lam_away, self.rho)

    def predict_from_ratings(
        self,
        home_attack: float,
        home_defense: float,
        away_attack: float,
        away_defense: float,
        neutral: bool = False,
    ) -> dict:
        """Predict using pre-computed attack/defense ratings (0-3 scale)."""
        home_adv_factor = 1.0 if neutral else math.exp(self.HOME_ADVANTAGE)
        lam_home = home_attack * away_defense * home_adv_factor
        lam_away = away_attack * home_defense

        lam_home = max(lam_home, 0.1)
        lam_away = max(lam_away, 0.1)

        return _outcome_probs(lam_home, lam_away, self.rho)

    def _fallback_predict(self, home_team: str, away_team: str) -> dict:
        """Fallback: equal-strength prediction with home advantage."""
        lam_home = math.exp(self.HOME_ADVANTAGE) * 1.3
        lam_away = 1.1
        return _outcome_probs(lam_home, lam_away, self.rho)


def quick_lambdas(
    home_goals_avg: float,
    away_goals_avg: float,
    home_conceded_avg: float,
    away_conceded_avg: float,
    league_avg_goals: float = 1.35,
    neutral: bool = False,
    injury_adj_home: float = 1.0,
    injury_adj_away: float = 1.0,
) -> tuple:
    """
    Compute the expected-goals rates (λ_home, λ_away) from team averages.
    Shared by quick_predict and the Monte Carlo simulator so both agree.
    """
    # λ_home = (home attack) × (away defensive weakness) × league_avg × home_advantage
    # λ_away = (away attack) × (home defensive weakness) × league_avg
    home_attack_str = home_goals_avg / league_avg_goals
    away_def_weakness = away_conceded_avg / league_avg_goals  # higher = weaker defense
    away_attack_str = away_goals_avg / league_avg_goals
    home_def_weakness = home_conceded_avg / league_avg_goals  # higher = weaker defense

    home_adv = math.exp(0.25) if not neutral else 1.0

    lam_home = home_attack_str * away_def_weakness * league_avg_goals * home_adv * injury_adj_home
    lam_away = away_attack_str * home_def_weakness * league_avg_goals * injury_adj_away

    return max(lam_home, 0.1), max(lam_away, 0.1)


def quick_predict(
    home_goals_avg: float,
    away_goals_avg: float,
    home_conceded_avg: float,
    away_conceded_avg: float,
    league_avg_goals: float = 1.35,
    neutral: bool = False,
    injury_adj_home: float = 1.0,
    injury_adj_away: float = 1.0,
) -> dict:
    """
    Fast prediction using team averages without fitting.
    Uses attack/defense ratings derived from goals data.
    """
    lam_home, lam_away = quick_lambdas(
        home_goals_avg, away_goals_avg, home_conceded_avg, away_conceded_avg,
        league_avg_goals, neutral, injury_adj_home, injury_adj_away,
    )
    return _outcome_probs(lam_home, lam_away, rho=-0.13)


def _log_factorial(n: int) -> float:
    return sum(math.log(i) for i in range(1, n + 1)) if n > 0 else 0.0
