"""Unit tests for core prediction logic."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np


# ── ELO Tests ─────────────────────────────────────────────────────────────
class TestElo:
    def test_equal_teams_50_percent(self):
        from src.models.elo import win_probability
        assert abs(win_probability(1500, 1500) - 0.5) < 1e-9

    def test_higher_elo_wins_more(self):
        from src.models.elo import win_probability
        p = win_probability(1800, 1500)
        assert p > 0.8

    def test_elo_update_winner_gains(self):
        from src.models.elo import update_elo
        new_a, new_b = update_elo(1500, 1500, result=1.0, k=32)
        assert new_a > 1500
        assert new_b < 1500

    def test_draw_splits_evenly_for_equal(self):
        from src.models.elo import update_elo
        new_a, new_b = update_elo(1500, 1500, result=0.5, k=32)
        assert abs(new_a - 1500) < 1e-6
        assert abs(new_b - 1500) < 1e-6

    def test_surface_elo_adjustment(self):
        from src.models.elo import surface_elo
        # Player who wins 70% on clay vs 50% overall → clay ELO boost
        adj = surface_elo(1500, "clay", 0.70, 0.50)
        assert adj > 1500

    def test_draw_probability_peaks_at_equal(self):
        from src.models.elo import _estimate_draw_prob
        d_equal = _estimate_draw_prob(1500, 1500)
        d_unequal = _estimate_draw_prob(2000, 1500)
        assert d_equal > d_unequal
        assert 0.20 < d_equal < 0.35

    def test_predictor_1v1_sum_to_one(self):
        from src.models.elo import EloPredictor
        ep = EloPredictor()
        ep.set("player_a", 1600)
        ep.set("player_b", 1500)
        p = ep.predict_1v1("player_a", "player_b")
        assert 0.5 < p < 1.0

    def test_predictor_3way_sums_to_one(self):
        from src.models.elo import EloPredictor
        ep = EloPredictor()
        ep.set("team_a", 1800)
        ep.set("team_b", 1700)
        result = ep.predict("team_a", "team_b")
        total = result["a_win"] + result["draw"] + result["b_win"]
        assert abs(total - 1.0) < 0.001


# ── Dixon-Coles Tests ─────────────────────────────────────────────────────
class TestDixonColes:
    def test_probabilities_sum_to_one(self):
        from src.models.dixon_coles import quick_predict
        result = quick_predict(1.5, 1.2, 1.1, 1.3)
        total = result["home_win"] + result["draw"] + result["away_win"]
        assert abs(total - 1.0) < 0.01

    def test_home_advantage_increases_home_win(self):
        from src.models.dixon_coles import quick_predict
        home_adv = quick_predict(1.4, 1.4, 1.4, 1.4, neutral=False)
        neutral = quick_predict(1.4, 1.4, 1.4, 1.4, neutral=True)
        assert home_adv["home_win"] > neutral["home_win"]

    def test_strong_team_wins_more(self):
        from src.models.dixon_coles import quick_predict
        # Team with 2.5 goals/game vs 0.8 conceded should dominate
        result = quick_predict(2.5, 0.8, 0.6, 2.0)
        assert result["home_win"] > 0.70

    def test_tau_correction_not_negative(self):
        from src.models.dixon_coles import _tau
        # τ should remain positive for all valid score combos
        for x in range(3):
            for y in range(3):
                t = _tau(x, y, 1.2, 1.1, -0.13)
                assert t > 0

    def test_score_probability_sums_close_to_one(self):
        from src.models.dixon_coles import _score_probability
        total = sum(
            _score_probability(h, a, 1.35, 1.10, -0.13)
            for h in range(10) for a in range(10)
        )
        assert 0.95 < total < 1.05  # not exact due to truncation at 10

    def test_quick_lambdas_match_quick_predict(self):
        from src.models.dixon_coles import quick_lambdas, _outcome_probs, quick_predict
        lam_h, lam_a = quick_lambdas(1.5, 1.2, 1.1, 1.3)
        assert lam_h > lam_a > 0  # home edge from goals + home advantage
        # Probs derived from the lambdas must equal quick_predict's output.
        assert _outcome_probs(lam_h, lam_a, -0.13) == quick_predict(1.5, 1.2, 1.1, 1.3)


# ── Monte Carlo Tests ──────────────────────────────────────────────────────
class TestMonteCarlo:
    def test_scoreline_percentages_sane(self):
        from src.models.monte_carlo import simulate
        sim = simulate(1.6, 1.0, n_sims=5000)
        assert sim["n_sims"] == 5000
        # Every scoreline count is positive and percentages are bounded.
        assert all(0 < s["count"] <= 5000 for s in sim["scorelines"])
        assert all(0 < s["pct"] <= 100 for s in sim["scorelines"])

    def test_outcome_counts_sum_to_n_sims(self):
        from src.models.monte_carlo import simulate
        sim = simulate(1.4, 1.1, n_sims=4000)
        oc = sim["outcome_counts"]
        assert oc["home_win"] + oc["draw"] + oc["away_win"] == 4000

    def test_first_half_goals_below_total(self):
        from src.models.monte_carlo import simulate
        sim = simulate(1.8, 1.2, n_sims=6000)
        # ~45% of goals fall in the first half, so the average must be lower.
        assert sim["avg_first_half_goals"] < sim["avg_total_goals"]
        assert sim["avg_first_half_goals"] > 0

    def test_aligns_with_dixon_coles(self):
        from src.models.monte_carlo import simulate
        from src.models.dixon_coles import _outcome_probs
        lam_h, lam_a = 1.7, 0.9
        sim = simulate(lam_h, lam_a, n_sims=20000)
        dc = _outcome_probs(lam_h, lam_a, -0.13)
        sim_home = sim["outcome_counts"]["home_win"] / sim["n_sims"]
        # Monte Carlo home-win rate should track the analytic Dixon-Coles value.
        assert abs(sim_home - dc["home_win"]) < 0.03

    def test_deterministic_with_seed(self):
        from src.models.monte_carlo import simulate
        a = simulate(1.5, 1.1, n_sims=3000, seed=7)
        b = simulate(1.5, 1.1, n_sims=3000, seed=7)
        assert a["scorelines"] == b["scorelines"]


# ── Odds + Kelly Tests ────────────────────────────────────────────────────
class TestOdds:
    def test_american_to_prob_favorite(self):
        from src.market.odds import american_to_prob
        p = american_to_prob(-110)
        assert abs(p - (110 / 210)) < 0.001

    def test_american_to_prob_underdog(self):
        from src.market.odds import american_to_prob
        p = american_to_prob(150)
        assert abs(p - (100 / 250)) < 0.001

    def test_decimal_to_prob(self):
        from src.market.odds import decimal_to_prob
        assert abs(decimal_to_prob(2.0) - 0.5) < 0.001

    def test_remove_vig_sums_to_one(self):
        from src.market.odds import remove_vig
        raw = {"home": 0.55, "draw": 0.35, "away": 0.25}  # 1.15 total (15% vig)
        normed = remove_vig(raw)
        assert abs(sum(normed.values()) - 1.0) < 0.001

    def test_prob_to_decimal_inverse(self):
        from src.market.odds import decimal_to_prob, prob_to_decimal
        p = 0.65
        d = prob_to_decimal(p)
        assert abs(decimal_to_prob(d) - p) < 0.01


class TestKelly:
    def test_positive_edge_gives_positive_stake(self):
        from src.market.kelly import kelly_fraction
        # 60% win prob at 2.0 decimal (evens) → positive edge
        stake = kelly_fraction(0.60, 2.0)
        assert stake > 0

    def test_negative_edge_gives_zero_stake(self):
        from src.market.kelly import kelly_fraction
        # 40% win prob at 1.5 decimal → negative edge
        stake = kelly_fraction(0.40, 1.5)
        assert stake == 0.0

    def test_quarter_kelly_smaller_than_full(self):
        from src.market.kelly import kelly_fraction
        full = kelly_fraction(0.65, 2.0, fraction=1.0)
        quarter = kelly_fraction(0.65, 2.0, fraction=0.25)
        assert quarter < full
        assert abs(quarter - full * 0.25) < 0.01

    def test_expected_value_positive_when_edge(self):
        from src.market.kelly import expected_value
        ev = expected_value(0.60, 2.0)
        assert ev > 0

    def test_expected_value_negative_no_edge(self):
        from src.market.kelly import expected_value
        ev = expected_value(0.40, 1.5)
        assert ev < 0


# ── Calibrator Tests ──────────────────────────────────────────────────────
class TestCalibrator:
    def test_normalize_sums_to_one(self):
        from src.models.calibrator import normalize
        probs = {"a": 0.3, "b": 0.4, "c": 0.5}
        normed = normalize(probs)
        assert abs(sum(normed.values()) - 1.0) < 0.001

    def test_blend_weighted_average(self):
        from src.models.calibrator import blend
        p1 = {"win": 0.6, "loss": 0.4}
        p2 = {"win": 0.4, "loss": 0.6}
        result = blend([p1, p2], [0.5, 0.5], ["win", "loss"])
        assert abs(result["win"] - 0.5) < 0.001

    def test_confidence_high_for_strong_favorite(self):
        from src.models.calibrator import confidence_score
        probs = {"home_win": 0.85, "draw": 0.08, "away_win": 0.07}
        score = confidence_score(probs)
        assert score > 70

    def test_confidence_low_for_even_match(self):
        from src.models.calibrator import confidence_score
        probs = {"p1_win": 0.50, "p2_win": 0.50}
        score = confidence_score(probs)
        assert score < 10


# ── Edge Calculator Tests ─────────────────────────────────────────────────
class TestEdge:
    def test_positive_edge_detected(self):
        from src.market.edge import calculate_edge
        model = {"home_win": 0.55, "draw": 0.25, "away_win": 0.20}
        market = {"home_win": 0.45, "draw": 0.28, "away_win": 0.27}
        edges = calculate_edge(model, market)
        assert edges["home_win"]["edge"] > 0
        assert edges["home_win"]["rating"] in ("STRONG", "MODERATE")

    def test_negative_edge_flagged(self):
        from src.market.edge import calculate_edge
        model = {"home_win": 0.30, "draw": 0.30, "away_win": 0.40}
        market = {"home_win": 0.50, "draw": 0.25, "away_win": 0.25}
        edges = calculate_edge(model, market)
        assert edges["home_win"]["rating"] == "NEGATIVE"

    def test_best_bet_returns_highest_edge(self):
        from src.market.edge import calculate_edge, best_bet
        model = {"home_win": 0.60, "draw": 0.20, "away_win": 0.20}
        market = {"home_win": 0.40, "draw": 0.30, "away_win": 0.30}
        edges = calculate_edge(model, market)
        outcome, info = best_bet(edges)
        assert outcome == "home_win"


# ── Parser Tests ──────────────────────────────────────────────────────────
class TestParser:
    def test_basic_vs_parse(self):
        from src.parser import parse
        result = parse("Portugal vs Spain")
        assert result.get("entity1") is not None
        assert result.get("entity2") is not None

    def test_detects_football(self):
        from src.parser import parse
        result = parse("Portugal vs Spain Nations League")
        assert result.get("sport") in ("football", "soccer")

    def test_detects_tennis(self):
        from src.parser import parse
        result = parse("Djokovic vs Alcaraz Wimbledon")
        assert result.get("sport") == "tennis"

    def test_detects_ufc(self):
        from src.parser import parse
        result = parse("Jon Jones vs Stipe Miocic UFC")
        assert result.get("sport") in ("ufc", "mma")

    def test_detects_cricket(self):
        from src.parser import parse
        result = parse("India vs Australia T20 World Cup")
        assert result.get("sport") == "cricket"

    def test_show_sports_command(self):
        from src.parser import parse
        result = parse("show sports")
        assert result.get("command") == "show_sports"

    def test_date_tomorrow(self):
        from src.parser import parse
        from datetime import datetime, timedelta
        result = parse("Portugal vs Spain tomorrow")
        expected = str((datetime.now().date() + timedelta(days=1)))
        assert result.get("date") == expected

    def test_competition_extracted(self):
        from src.parser import parse
        result = parse("Portugal vs Spain Nations League final")
        assert "nations league" in result.get("competition", "").lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
