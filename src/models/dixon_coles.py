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
        Fit model on match history via vectorised MLE (L-BFGS-B).

        Vectorised over all matches using NumPy — ~100× faster than a Python loop.
        matches: list of {home_team, away_team, home_goals, away_goals, weight?}
        """
        teams = sorted(set(
            [m["home_team"] for m in matches] + [m["away_team"] for m in matches]
        ))
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        # Build integer index arrays — constructed once, used in every gradient step
        home_idx  = np.array([idx[m["home_team"]] for m in matches], dtype=np.int32)
        away_idx  = np.array([idx[m["away_team"]] for m in matches], dtype=np.int32)
        home_goals = np.array([int(m["home_goals"]) for m in matches], dtype=np.float64)
        away_goals = np.array([int(m["away_goals"]) for m in matches], dtype=np.float64)
        weights    = np.array([float(m.get("weight", 1.0)) for m in matches], dtype=np.float64)

        # Precompute log-factorials for all possible goal counts
        max_g = int(max(home_goals.max(), away_goals.max())) + 1
        log_fact = np.zeros(max_g + 1)
        for g in range(1, max_g + 1):
            log_fact[g] = log_fact[g - 1] + math.log(g)
        lf_h = log_fact[home_goals.astype(int)]
        lf_a = log_fact[away_goals.astype(int)]

        ha = self.HOME_ADVANTAGE

        def neg_ll_and_grad(params: np.ndarray):
            """Return (negative log-likelihood, gradient) jointly — avoids finite-diff overhead."""
            att = params[:n]
            def_ = params[n:2 * n]
            rho  = params[2 * n]

            lam_h = np.exp(att[home_idx] + def_[away_idx] + ha)
            lam_a = np.exp(att[away_idx] + def_[home_idx])
            lam_h = np.maximum(lam_h, 0.01)
            lam_a = np.maximum(lam_a, 0.01)

            # τ correction and its partial derivatives w.r.t. λ_h, λ_a, ρ
            tau  = np.ones(len(matches))
            dtdlh = np.zeros(len(matches))  # ∂τ/∂λ_h
            dtdla = np.zeros(len(matches))  # ∂τ/∂λ_a
            dtdrho = np.zeros(len(matches)) # ∂τ/∂ρ
            m00 = (home_goals == 0) & (away_goals == 0)
            m10 = (home_goals == 1) & (away_goals == 0)
            m01 = (home_goals == 0) & (away_goals == 1)
            m11 = (home_goals == 1) & (away_goals == 1)

            tau[m00]    = 1 - lam_h[m00] * lam_a[m00] * rho
            dtdlh[m00]  = -lam_a[m00] * rho
            dtdla[m00]  = -lam_h[m00] * rho
            dtdrho[m00] = -lam_h[m00] * lam_a[m00]

            tau[m10]    = 1 + lam_a[m10] * rho
            dtdla[m10]  = rho
            dtdrho[m10] = lam_a[m10]

            tau[m01]    = 1 + lam_h[m01] * rho
            dtdlh[m01]  = rho
            dtdrho[m01] = lam_h[m01]

            tau[m11]    = 1 - rho
            dtdrho[m11] = -1.0

            tau = np.maximum(tau, 1e-8)

            # Log-likelihood per match
            ll = weights * (
                np.log(tau)
                + home_goals * np.log(lam_h) - lam_h - lf_h
                + away_goals * np.log(lam_a) - lam_a - lf_a
            )
            nll = -ll.sum()

            # ∂NLL/∂λ_h and ∂NLL/∂λ_a per match
            w_dlh = weights * (dtdlh / tau + home_goals / lam_h - 1)
            w_dla = weights * (dtdla / tau + away_goals / lam_a - 1)

            # Accumulate gradients for attack[k] and defense[k]
            # att[k]: appears as lam_h for home matches (∂lam_h/∂att[k] = lam_h)
            #                and as lam_a for away matches (∂lam_a/∂att[k] = lam_a)
            grad = np.zeros(2 * n + 1)
            np.add.at(grad,       home_idx, -(w_dlh * lam_h))  # ∂att from home
            np.add.at(grad,       away_idx, -(w_dla * lam_a))  # ∂att from away
            np.add.at(grad, n +   away_idx, -(w_dlh * lam_h))  # ∂def from away when home
            np.add.at(grad, n +   home_idx, -(w_dla * lam_a))  # ∂def from home when away
            grad[2 * n] = -np.sum(weights * dtdrho / tau)

            return nll, grad

        x0 = np.zeros(2 * n + 1)
        x0[-1] = -0.13  # rho initial

        try:
            result = minimize(
                neg_ll_and_grad,
                x0,
                method="L-BFGS-B",
                jac=True,  # function returns (f, grad) jointly — no finite differences
                bounds=[(None, None)] * (2 * n) + [(-0.5, 0.5)],
                options={"maxiter": 300, "ftol": 1e-7},
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
    # λ_home = (home attack) × (away defensive weakness) × league_avg × home_advantage
    # λ_away = (away attack) × (home defensive weakness) × league_avg
    home_attack_str = home_goals_avg / league_avg_goals
    away_def_weakness = away_conceded_avg / league_avg_goals  # higher = weaker defense
    away_attack_str = away_goals_avg / league_avg_goals
    home_def_weakness = home_conceded_avg / league_avg_goals  # higher = weaker defense

    home_adv = math.exp(0.25) if not neutral else 1.0

    lam_home = home_attack_str * away_def_weakness * league_avg_goals * home_adv * injury_adj_home
    lam_away = away_attack_str * home_def_weakness * league_avg_goals * injury_adj_away

    lam_home = max(lam_home, 0.1)
    lam_away = max(lam_away, 0.1)

    return _outcome_probs(lam_home, lam_away, rho=-0.13)


def _log_factorial(n: int) -> float:
    return sum(math.log(i) for i in range(1, n + 1)) if n > 0 else 0.0
