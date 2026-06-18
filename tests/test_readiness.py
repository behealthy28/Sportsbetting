"""Tests for the squad-readiness factor and backtest harness."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestReadiness:
    def test_neutral_for_unknown_team_fresh(self):
        from src.data import readiness
        r = readiness.get_readiness("Atlantis", days_rest=7)
        assert abs(r["multiplier"] - 1.0) < 1e-9

    def test_congestion_penalises(self):
        from src.data import readiness
        fresh = readiness.get_readiness("Atlantis", days_rest=7)["multiplier"]
        tired = readiness.get_readiness("Atlantis", days_rest=2)["multiplier"]
        assert tired < fresh

    def test_multiplier_never_inflates_above_one(self):
        from src.data import readiness
        assert readiness.readiness_to_multiplier(1.0) <= 1.0
        assert readiness.readiness_to_multiplier(2.0) <= 1.0

    def test_key_player_out_drops_availability(self):
        from src.data import readiness
        news = {"headlines": ["Luka Modric ruled out with hamstring injury"]}
        cro = readiness.get_squad_availability("Croatia", news)
        assert cro["availability"] < 1.0
        assert any(f["player"] == "Luka Modric" for f in cro["flagged"])

    def test_availability_neutral_without_roster(self):
        from src.data import readiness
        out = readiness.get_squad_availability("Atlantis", {"headlines": ["anything"]})
        assert out["availability"] == 1.0

    def test_manual_out_marks_player(self):
        from src.data import readiness
        out = readiness.get_squad_availability("England", None, manual_out=["Harry Kane"])
        assert out["availability"] < 1.0


class TestBacktest:
    def _perfect(self):
        return [
            {"probs": {"a": 1.0, "b": 0.0}, "outcome": "a"},
            {"probs": {"a": 0.0, "b": 1.0}, "outcome": "b"},
        ]

    def test_brier_perfect_is_zero(self):
        from src.models import backtest
        assert backtest.brier_score(self._perfect()) == 0.0

    def test_brier_worse_than_perfect(self):
        from src.models import backtest
        recs = [{"probs": {"a": 0.5, "b": 0.5}, "outcome": "a"}]
        assert backtest.brier_score(recs) > 0.0

    def test_ece_perfect_is_zero(self):
        from src.models import backtest
        assert backtest.expected_calibration_error(self._perfect()) == 0.0

    def test_evaluate_reports_all_metrics(self):
        from src.models import backtest
        rep = backtest.evaluate(self._perfect())
        assert {"n", "brier", "log_loss", "ece", "reliability"} <= set(rep)
