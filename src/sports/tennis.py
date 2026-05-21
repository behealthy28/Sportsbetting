"""Tennis prediction handler."""
import json
from pathlib import Path
from src.sports.base import AbstractSport, PredictionResult
from src.data.scrapers import sackmann
from src.data import news, market
from src.models import elo as elo_module, calibrator, ml_ensemble
from src.market import edge as edge_mod, kelly as kelly_mod, odds as odds_mod

PLAYERS_FILE = Path(__file__).parent.parent.parent / "data" / "mappings" / "players.json"

TOURNAMENT_WEIGHTS = {
    "grand slam": 1.0, "wimbledon": 1.0, "us open": 1.0, "french open": 1.0,
    "roland garros": 1.0, "australian open": 1.0,
    "atp finals": 0.95, "wta finals": 0.95,
    "masters": 0.90, "1000": 0.88, "500": 0.80, "250": 0.70,
    "challenger": 0.60, "itf": 0.50,
}

SURFACE_KEYWORDS = {
    "clay": ["clay", "roland garros", "french open", "madrid", "rome", "barcelona"],
    "grass": ["grass", "wimbledon", "queen's", "halle", "newport"],
    "hard": ["hard", "us open", "australian open", "miami", "indian wells"],
    "carpet": ["carpet", "indoor"],
}


def _detect_surface(context: dict) -> str:
    venue = (context.get("venue", "") + " " + context.get("competition", "")).lower()
    for surface, keywords in SURFACE_KEYWORDS.items():
        if any(kw in venue for kw in keywords):
            return surface
    return "hard"


def _tournament_weight(context: dict) -> float:
    comp = (context.get("competition", "") + " " + context.get("venue", "")).lower()
    for k, v in TOURNAMENT_WEIGHTS.items():
        if k in comp:
            return v
    return 0.80


def _load_player_mapping() -> dict:
    try:
        return json.loads(PLAYERS_FILE.read_text()).get("tennis", {})
    except Exception:
        return {}


class TennisPredictor(AbstractSport):

    sport_name = "tennis"
    sport_keywords = [
        "tennis", "atp", "wta", "wimbledon", "us open", "french open",
        "australian open", "roland garros", "grand slam", "masters", "open",
        "djokovic", "alcaraz", "sinner", "swiatek", "sabalenka", "nadal", "federer",
    ]

    PLAYER_ELO = {}

    def __init__(self):
        mapping = _load_player_mapping()
        self.PLAYER_ELO = {name: info["elo"] for name, info in mapping.items() if "elo" in info}

    def predict(self, entity1: str, entity2: str, date: str, context: dict) -> PredictionResult:
        surface = _detect_surface(context)
        t_weight = _tournament_weight(context)
        sources = []

        # 1. Fetch player stats from Sackmann
        p1_stats = sackmann.get_player_stats(entity1, months=12)
        p2_stats = sackmann.get_player_stats(entity2, months=12)
        if p1_stats["total_matches"] > 0 or p2_stats["total_matches"] > 0:
            sources.append("Sackmann ATP/WTA dataset")

        # 2. ELO ratings
        elo_predictor = elo_module.EloPredictor(default_elo=1500)
        mapping = _load_player_mapping()
        for name, info in mapping.items():
            if "elo" in info:
                elo_predictor.set(name, info["elo"])
        # Inject from stats if available
        if p1_stats.get("elo"):
            elo_predictor.set(entity1.lower(), float(p1_stats["elo"]))
        if p2_stats.get("elo"):
            elo_predictor.set(entity2.lower(), float(p2_stats["elo"]))

        # Surface-adjusted ELO
        p1_surf = p1_stats.get("surface_stats", {}).get(surface, {})
        p2_surf = p2_stats.get("surface_stats", {}).get(surface, {})
        elo_p1 = elo_predictor.get(entity1.lower())
        elo_p2 = elo_predictor.get(entity2.lower())

        if p1_surf.get("win_rate") and p1_stats.get("overall_win_rate"):
            elo_p1 = elo_module.surface_elo(elo_p1, surface, p1_surf["win_rate"], p1_stats["overall_win_rate"])
        if p2_surf.get("win_rate") and p2_stats.get("overall_win_rate"):
            elo_p2 = elo_module.surface_elo(elo_p2, surface, p2_surf["win_rate"], p2_stats["overall_win_rate"])

        elo_p1_win = elo_module.win_probability(elo_p1, elo_p2)
        elo_result = {"p1_win": round(elo_p1_win, 4), "p2_win": round(1 - elo_p1_win, 4)}

        # 3. H2H
        h2h = sackmann.get_h2h(entity1, entity2)
        if h2h["total"] > 0:
            sources.append("H2H historical data")

        h2h_p1_rate = h2h.get("p1_win_rate", 0.5)
        # Blend H2H and ELO: weight H2H more if we have >5 matches
        h2h_weight = min(0.30, h2h["total"] * 0.03)
        elo_weight = 1.0 - h2h_weight
        blended_p1 = elo_weight * elo_p1_win + h2h_weight * h2h_p1_rate

        # 4. Surface form adjustment
        if p1_surf.get("matches", 0) >= 3 and p2_surf.get("matches", 0) >= 3:
            surf_p1 = p1_surf["win_rate"]
            surf_p2 = p2_surf["win_rate"]
            surf_total = surf_p1 + surf_p2
            surf_p1_norm = surf_p1 / surf_total if surf_total > 0 else 0.5
            blended_p1 = 0.65 * blended_p1 + 0.35 * surf_p1_norm
            sources.append(f"Surface ({surface}) form stats")

        # 5. News
        p1_news = news.get_sentiment(entity1)
        p2_news = news.get_sentiment(entity2)
        all_flags = p1_news.get("flags", []) + p2_news.get("flags", [])

        # Apply sentiment
        p1_sentiment = p1_news.get("score", 0.0)
        p2_sentiment = p2_news.get("score", 0.0)
        blended_p1 += (p1_sentiment - p2_sentiment) * 0.08

        blended_p1 = max(0.05, min(0.95, blended_p1))
        probs = {"p1_win": round(blended_p1, 4), "p2_win": round(1 - blended_p1, 4)}

        # 6. ML ensemble
        ctx_ml = {
            "surface": surface,
            "h2h_win_rate": h2h_p1_rate,
            "tournament_importance": t_weight,
            "injury_p1": 1 if any("injury" in f.lower() or "doubt" in f.lower() for f in p1_news.get("flags", [])) else 0,
            "injury_p2": 1 if any("injury" in f.lower() or "doubt" in f.lower() for f in p2_news.get("flags", [])) else 0,
        }
        features = ml_ensemble.build_tennis_features(p1_stats, p2_stats, ctx_ml)
        ml_model = ml_ensemble.MLEnsemble(sport="tennis", n_classes=2)
        if ml_model.load():
            ml_dict = ml_model.predict_dict(features, ["p2_win", "p1_win"])
            probs = calibrator.blend(
                [probs, {"p1_win": ml_dict["p1_win"], "p2_win": ml_dict["p2_win"]}],
                [0.70, 0.30],
                ["p1_win", "p2_win"],
            )

        # 6b. Agent debate — optional swarm-intelligence layer (requires ANTHROPIC_API_KEY)
        debate_result = None
        try:
            from src.models.agent_debate import run_debate, blend_with_ml
            debate_ctx = {
                "date": date,
                "competition": context.get("competition", ""),
                "surface": surface,
                "ml_probability_a": probs.get("p1_win", 0.5),
                "elo_a": elo_p1,
                "elo_b": elo_p2,
                "form_a": p1_stats.get("overall_win_rate", 0.5),
                "form_b": p2_stats.get("overall_win_rate", 0.5),
                "h2h_wins_a": h2h.get("p1_wins", 0),
                "h2h_wins_b": h2h.get("p2_wins", 0),
                "avg_goals_a": p1_stats.get("ace_rate", 0.0),
                "avg_goals_b": p2_stats.get("ace_rate", 0.0),
                "news_flags": all_flags[:4],
                "key_factors": factors,
            }
            debate_result = run_debate(entity1, entity2, "tennis", debate_ctx)
            if debate_result:
                blended_p1_new = blend_with_ml(probs["p1_win"], debate_result, ml_weight=0.72)
                probs = {
                    "p1_win": round(blended_p1_new, 4),
                    "p2_win": round(max(0.05, 1 - blended_p1_new), 4),
                }
        except Exception:
            pass

        # 7. Market
        mkt = market.get_market_odds(entity1, entity2, "tennis")
        market_mapped = None
        if mkt:
            market_mapped = {
                "p1_win": mkt.get("home_win"),
                "p2_win": mkt.get("away_win"),
            }

        # 8. Edge + Kelly
        edges = edge_mod.calculate_edge(probs, market_mapped or {})
        best_outcome, best_info = edge_mod.best_bet(edges)
        best_edge = best_info["edge_pct"] if best_info else 0.0
        kelly_pct = 0.0
        if best_info and best_info["edge"] > 0:
            decimal = odds_mod.prob_to_decimal(best_info["market_prob"])
            kelly_pct = kelly_mod.kelly_fraction(best_info["model_prob"], decimal)

        factors = [
            f"Surface: {surface.capitalize()} — {'favors ' + entity1 if p1_surf.get('win_rate', 0.5) > 0.6 else 'neutral'}",
            f"ELO: {entity1} {elo_p1:.0f} vs {entity2} {elo_p2:.0f}",
        ]
        if h2h["total"] > 0:
            factors.append(f"H2H: {entity1} {h2h['p1_wins']}-{h2h['p2_wins']} {entity2}")

        return PredictionResult(
            sport="Tennis",
            entity1=entity1,
            entity2=entity2,
            date=date,
            probabilities=probs,
            market_probs=market_mapped,
            edges=edges,
            best_bet=best_outcome,
            best_edge_pct=best_edge,
            kelly_stake_pct=kelly_pct,
            confidence=calibrator.confidence_score(probs),
            key_factors=factors,
            news_flags=all_flags[:5],
            data_sources=sources or ["ELO seeded ratings"],
            model_breakdown={
                "ELO + Surface": elo_result,
                "H2H Adjusted": probs,
                **({
                    "Agent Debate": {
                        "p1_win": round(debate_result.consensus_probability, 4),
                        "summary": debate_result.summary,
                        "agents": {e.agent: {"p": e.probability, "conf": e.confidence, "reason": e.reasoning}
                                   for e in debate_result.agent_estimates},
                    }
                } if debate_result else {}),
            },
            venue=context.get("venue", ""),
            competition=context.get("competition", ""),
        )
