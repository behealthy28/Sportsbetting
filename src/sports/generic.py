"""Generic ELO predictor for team sports without a dedicated model.

Covers baseball, basketball, ice hockey, American football, rugby, handball,
volleyball, etc. Uses each team's season win% (parsed from the ESPN scoreboard
record, passed in via context) to derive an ELO-style rating, then a standard
logistic win probability with home advantage. When records are unavailable it
degrades to a near-even prediction rather than failing.
"""
from src.sports.base import AbstractSport, PredictionResult
from src.data import news, market
from src.models import elo as elo_module, calibrator
from src.market import edge as edge_mod, kelly as kelly_mod, odds as odds_mod

# Win% (0..1) maps to a rating spread around 1500. A .700 team sits ~1660,
# a .300 team ~1340 — a ~320-pt gap, i.e. the strong side ~87% before home edge.
_RATING_SPREAD = 800.0
_BASE_RATING = 1500.0
_HOME_ADVANTAGE = 55.0  # ELO points; dropped when the venue is neutral


def _rating(win_pct) -> float:
    if win_pct is None:
        return _BASE_RATING
    win_pct = max(0.0, min(1.0, float(win_pct)))
    return _BASE_RATING + (win_pct - 0.5) * _RATING_SPREAD


class GenericPredictor(AbstractSport):
    """Records-driven ELO handler for any two-outcome team sport."""

    def __init__(self, sport_label: str = "Match"):
        self._label = sport_label

    sport_name = "generic"
    sport_keywords: list = []

    def predict(self, entity1: str, entity2: str, date: str, context: dict) -> PredictionResult:
        p1_wp = context.get("p1_winpct")
        p2_wp = context.get("p2_winpct")
        is_neutral = context.get("is_neutral", False)

        r1 = _rating(p1_wp) + (0.0 if is_neutral else _HOME_ADVANTAGE)
        r2 = _rating(p2_wp)
        p1_win = elo_module.win_probability(r1, r2)

        # News sentiment nudge (same treatment as the other handlers)
        n1 = news.get_sentiment(entity1)
        n2 = news.get_sentiment(entity2)
        flags = n1.get("flags", []) + n2.get("flags", [])
        p1_win = max(0.05, min(0.95, p1_win + (n1.get("score", 0) - n2.get("score", 0)) * 0.05))

        probs = {"p1_win": round(p1_win, 4), "p2_win": round(1 - p1_win, 4)}

        mkt = market.get_market_odds(entity1, entity2, self._label.lower())
        market_mapped = None
        if mkt:
            market_mapped = {"p1_win": mkt.get("home_win"), "p2_win": mkt.get("away_win")}

        edges = edge_mod.calculate_edge(probs, market_mapped or {})
        best_outcome, best_info = edge_mod.best_bet(edges)
        kelly_pct = 0.0
        if best_info and best_info["edge"] > 0:
            decimal = odds_mod.prob_to_decimal(best_info["market_prob"])
            kelly_pct = kelly_mod.kelly_fraction(best_info["model_prob"], decimal)

        factors = []
        if p1_wp is not None or p2_wp is not None:
            f1 = f"{p1_wp*100:.0f}%" if p1_wp is not None else "n/a"
            f2 = f"{p2_wp*100:.0f}%" if p2_wp is not None else "n/a"
            factors.append(f"Season win%: {entity1} {f1} vs {entity2} {f2}")
        else:
            factors.append("No season records available — near-even estimate")
        factors.append(f"ELO: {entity1} ({r1:.0f}) vs {entity2} ({r2:.0f})")
        if not is_neutral:
            factors.append(f"Home advantage applied to {entity1}")

        return PredictionResult(
            sport=self._label,
            entity1=entity1, entity2=entity2, date=date,
            probabilities=probs, market_probs=market_mapped, edges=edges,
            best_bet=best_outcome,
            best_edge_pct=best_info["edge_pct"] if best_info else 0.0,
            kelly_stake_pct=kelly_pct,
            confidence=calibrator.confidence_score(probs),
            key_factors=factors,
            news_flags=flags[:3],
            data_sources=["ESPN season records + ELO"],
            model_breakdown={"ELO": {"p1_win": p1_win}},
            competition=context.get("competition", self._label),
            is_neutral=is_neutral,
            venue=context.get("venue", "Unknown") or "Unknown",
        )
