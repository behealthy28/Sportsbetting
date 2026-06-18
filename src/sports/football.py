"""Football/Soccer prediction handler."""
from src.sports.base import AbstractSport, PredictionResult
from src.data.scrapers import fbref
from src.data import news, market, readiness, momentum
from src.models import dixon_coles, elo as elo_module, calibrator, ml_ensemble
from src.market import edge as edge_mod, kelly as kelly_mod, odds as odds_mod
import json
import os
import numpy as np

_RECENT_ELO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "historical", "elo_ratings.json",
)
_RECENT_ELO_CACHE = None


def _recent_elo() -> dict:
    """Data-derived Elo from the last ~5 years (built by src.models.train)."""
    global _RECENT_ELO_CACHE
    if _RECENT_ELO_CACHE is None:
        try:
            with open(_RECENT_ELO_PATH, encoding="utf-8") as f:
                _RECENT_ELO_CACHE = {k.lower(): v for k, v in json.load(f).items()}
        except (FileNotFoundError, json.JSONDecodeError):
            _RECENT_ELO_CACHE = {}
    return _RECENT_ELO_CACHE


COMPETITION_WEIGHTS = {
    "world cup": 1.0, "euro": 0.95, "copa america": 0.95, "afcon": 0.90,
    "champions league": 0.95, "nations league": 0.85, "premier league": 0.85,
    "la liga": 0.85, "serie a": 0.83, "bundesliga": 0.83, "ligue 1": 0.80,
    "friendly": 0.50, "friendlies": 0.50,
}

HOME_ADVANTAGE_ELO = 65  # Elo points for playing at home


class FootballPredictor(AbstractSport):

    sport_name = "football"
    sport_keywords = [
        "football", "soccer", "fc", "united", "city", "athletic",
        "premier league", "la liga", "bundesliga", "serie a", "ligue 1",
        "champions league", "world cup", "euro", "copa", "nations league",
        "afcon", "friendly",
    ]

    # National team ELO seeds (FIFA/ELO-based)
    TEAM_ELO = {
        "argentina": 2085, "france": 2052, "brazil": 2035, "spain": 2015,
        "portugal": 1987, "england": 1963, "netherlands": 1941, "croatia": 1921,
        "italy": 1875, "morocco": 1831, "germany": 1907, "belgium": 1879,
        "manchester city": 1920, "real madrid": 1930, "barcelona": 1895,
        "liverpool": 1870, "arsenal": 1855, "manchester united": 1820,
        "bayern munich": 1900, "borussia dortmund": 1850, "psg": 1870,
        "juventus": 1840, "inter milan": 1860, "ac milan": 1845,
        "atletico madrid": 1855, "chelsea": 1830, "tottenham": 1815,
    }

    def predict(self, entity1: str, entity2: str, date: str, context: dict) -> PredictionResult:
        competition = context.get("competition", "")
        is_neutral = context.get("is_neutral", False)
        comp_weight = _get_comp_weight(competition)

        # 1. Fetch team data
        home_data = fbref.get_team_data(entity1)
        away_data = fbref.get_team_data(entity2)
        h2h = fbref.get_h2h(entity1, entity2)
        sources = ["FBRef/Understat", "H2H records"]

        # 2. News sentiment
        home_news = news.get_sentiment(entity1)
        away_news = news.get_sentiment(entity2)
        all_flags = home_news.get("flags", []) + away_news.get("flags", [])
        if home_news.get("flags") or away_news.get("flags"):
            sources.append("Google News RSS")

        # 3. ELO prediction
        elo_predictor = elo_module.EloPredictor(default_elo=1700)
        for name, elo in self.TEAM_ELO.items():
            elo_predictor.set(name, elo)
        # Static-seed / fbref override first...
        if home_data.get("elo"):
            elo_predictor.set(entity1.lower(), home_data["elo"])
        if away_data.get("elo"):
            elo_predictor.set(entity2.lower(), away_data["elo"])
        # ...then overlay data-derived recent (last ~5y) Elo so it WINS.
        recent_elo = _recent_elo()
        for name, elo in recent_elo.items():
            elo_predictor.set(name, elo)
        if entity1.lower() in recent_elo or entity2.lower() in recent_elo:
            sources.append("Recent Elo (last 5y, data-derived)")

        home_adv = 0 if is_neutral else HOME_ADVANTAGE_ELO
        elo_probs = elo_predictor.predict(entity1.lower(), entity2.lower(), home_advantage=home_adv)
        elo_result = {"home_win": elo_probs["a_win"], "draw": elo_probs["draw"], "away_win": elo_probs["b_win"]}

        # 3b. Squad readiness / recovery factor (privacy-safe — see readiness.py).
        #     Maps days-rest + optional manual estimates to a strength multiplier.
        #     Explicit context values still win, so callers can override.
        home_ready = readiness.get_readiness(
            entity1,
            days_rest=context.get("home_days_rest", 7),
            news_sentiment=home_news,
        )
        away_ready = readiness.get_readiness(
            entity2,
            days_rest=context.get("away_days_rest", 7),
            news_sentiment=away_news,
        )
        if "home_key_players" not in context:
            context["home_key_players"] = home_ready["multiplier"]
        if "away_key_players" not in context:
            context["away_key_players"] = away_ready["multiplier"]
        if "Squad readiness model" not in sources:
            sources.append("Squad readiness model")

        # 4. Dixon-Coles prediction
        injury_adj_home = context.get("home_key_players", 1.0)
        injury_adj_away = context.get("away_key_players", 1.0)

        dc_result = dixon_coles.quick_predict(
            home_goals_avg=home_data.get("avg_goals", 1.35),
            away_goals_avg=away_data.get("avg_goals", 1.35),
            home_conceded_avg=home_data.get("avg_conceded", 1.20),
            away_conceded_avg=away_data.get("avg_conceded", 1.20),
            league_avg_goals=1.35,
            neutral=is_neutral,
            injury_adj_home=injury_adj_home,
            injury_adj_away=injury_adj_away,
        )

        # 4b. Momentum: winning/losing streaks + response to dropped points.
        home_mom = momentum.get_momentum(entity1)
        away_mom = momentum.get_momentum(entity2)
        if entity1.lower() in momentum._load() or entity2.lower() in momentum._load():
            sources.append("Momentum (streaks / bounce-back)")

        # 5. ML ensemble
        ctx_dict = {
            "h2h_home_win_rate": h2h.get("team1_win_rate", 0.35),
            "home_key_players": injury_adj_home,
            "away_key_players": injury_adj_away,
            "home_days_rest": context.get("home_days_rest", 7),
            "away_days_rest": context.get("away_days_rest", 7),
            "competition_weight": comp_weight,
            "is_neutral": int(is_neutral),
            "home_streak_norm": home_mom["streak_norm"],
            "away_streak_norm": away_mom["streak_norm"],
            "home_bounceback": home_mom["bounceback"],
            "away_bounceback": away_mom["bounceback"],
        }
        features = ml_ensemble.build_football_features(home_data, away_data, ctx_dict)
        ml_model = ml_ensemble.MLEnsemble(sport="football", n_classes=3)
        ml_loaded = ml_model.load()
        if ml_loaded:
            ml_dict = ml_model.predict_dict(features, ["away_win", "draw", "home_win"])
            ml_result = {"home_win": ml_dict["home_win"], "draw": ml_dict["draw"], "away_win": ml_dict["away_win"]}
        else:
            ml_result = None

        # 6. Ensemble blend
        probs_list = [dc_result, elo_result]
        weights = [0.50, 0.30]
        if ml_result:
            probs_list.append(ml_result)
            weights.append(0.20)

        blended = calibrator.blend(probs_list, weights, ["home_win", "draw", "away_win"])

        # 7. News adjustments
        blended = calibrator.apply_news_sentiment(blended, home_news, "home_win")
        blended = calibrator.apply_news_sentiment(blended, away_news, "away_win")

        # 8. Market odds
        mkt = market.get_market_odds(entity1, entity2, "football")
        sources.append("Polymarket/Kalshi" if mkt else "No market found")

        # Remap market keys
        market_mapped = None
        if mkt:
            market_mapped = {
                "home_win": mkt.get("home_win"),
                "draw": mkt.get("draw"),
                "away_win": mkt.get("away_win"),
            }

        # 9. Edge
        edges = edge_mod.calculate_edge(blended, market_mapped or {})
        best_outcome, best_info = edge_mod.best_bet(edges)
        best_edge = best_info["edge_pct"] if best_info else 0.0

        # 10. Kelly
        kelly_pct = 0.0
        if best_info and best_info["edge"] > 0:
            decimal_odds = odds_mod.prob_to_decimal(best_info["market_prob"])
            kelly_pct = kelly_mod.kelly_fraction(best_info["model_prob"], decimal_odds)

        # 11. Key factors
        factors = _build_factors(home_data, away_data, h2h, entity1, entity2, competition)
        factors.extend(_readiness_factors(entity1, home_ready, entity2, away_ready))
        factors.extend(_momentum_factors(entity1, home_mom, entity2, away_mom))

        return PredictionResult(
            sport="Football",
            entity1=entity1,
            entity2=entity2,
            date=date,
            probabilities=blended,
            market_probs=market_mapped,
            edges=edges,
            best_bet=best_outcome,
            best_edge_pct=best_edge,
            kelly_stake_pct=kelly_pct,
            confidence=calibrator.confidence_score(blended),
            key_factors=factors,
            news_flags=all_flags[:5],
            data_sources=sources,
            model_breakdown={
                "Dixon-Coles": dc_result,
                "ELO": elo_result,
                "ML Ensemble": ml_result,
            },
            venue="Neutral" if is_neutral else f"{entity1} home",
            competition=competition,
            is_neutral=is_neutral,
        )


def _get_comp_weight(comp: str) -> float:
    comp_lower = comp.lower()
    for k, v in COMPETITION_WEIGHTS.items():
        if k in comp_lower:
            return v
    return 0.80


def _momentum_factors(team1: str, m1: dict, team2: str, m2: dict) -> list:
    """Describe streaks and bounce-back, only when notable."""
    out = []
    for team, m in ((team1, m1), (team2, m2)):
        s = m.get("streak", 0)
        if s >= 3:
            out.append(f"{team} on a {s}-game winning streak")
        elif s <= -3:
            out.append(f"{team} on a {abs(s)}-game losing streak")
        bb = m.get("bounceback", 0.5)
        if bb >= 0.6:
            out.append(f"{team} responds well after dropped points ({bb*100:.0f}% bounce-back)")
        elif bb <= 0.3 and bb > 0:
            out.append(f"{team} tends to spiral after dropped points ({bb*100:.0f}% bounce-back)")
    return out


def _readiness_factors(team1: str, r1: dict, team2: str, r2: dict) -> list:
    """Describe the squad-readiness adjustment, only when it is non-neutral."""
    out = []
    for team, r in ((team1, r1), (team2, r2)):
        mult = r.get("multiplier", 1.0)
        # Surface whenever readiness carries information (estimate/news/penalty).
        if abs(mult - 1.0) < 0.001 and not r.get("notes"):
            continue
        direction = "boost" if mult > 1.0 else "penalty"
        detail = f" ({'; '.join(r['notes'])})" if r.get("notes") else ""
        out.append(
            f"{team} squad readiness {direction}: x{mult:.3f}{detail}"
        )
    if out:
        out.append("Readiness = days-rest + manual estimates only (no biometric data harvested)")
    return out


def _build_factors(home: dict, away: dict, h2h: dict, team1: str, team2: str, comp: str) -> list:
    factors = []
    if home.get("form", 0.5) > 0.7:
        factors.append(f"{team1} in strong form ({home['form']*100:.0f}% recent win rate)")
    if away.get("form", 0.5) > 0.7:
        factors.append(f"{team2} in strong form ({away['form']*100:.0f}% recent win rate)")
    if home.get("avg_xg") and home.get("avg_goals"):
        if home["avg_xg"] > home["avg_goals"] + 0.2:
            factors.append(f"{team1} underperforming xG — regression to mean expected")
    h2h_total = h2h.get("total", 0)
    if h2h_total > 5:
        t1w = h2h.get("team1_wins", 0)
        factors.append(f"H2H: {team1} leads {t1w}-{h2h.get('team2_wins',0)} in last {h2h_total} meetings")
    if comp:
        factors.append(f"Competition: {comp}")
    return factors
