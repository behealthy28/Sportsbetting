"""UFC/MMA prediction handler."""
from src.sports.base import AbstractSport, PredictionResult
from src.data.scrapers import ufcstats
from src.data import news, market
from src.models import elo as elo_module, calibrator, ml_ensemble
from src.market import edge as edge_mod, kelly as kelly_mod, odds as odds_mod
import json
from pathlib import Path

PLAYERS_FILE = Path(__file__).parent.parent.parent / "data" / "mappings" / "players.json"


def _load_ufc_seeds() -> dict:
    try:
        return json.loads(PLAYERS_FILE.read_text()).get("ufc", {})
    except Exception:
        return {}


class UFCPredictor(AbstractSport):

    sport_name = "ufc"
    sport_keywords = [
        "ufc", "mma", "fight night", "ppv", "bellator", "one fc",
        "heavyweight", "lightweight", "welterweight", "middleweight",
        "featherweight", "bantamweight", "flyweight",
        "jones", "ngannou", "makhachev", "pereira", "volkanovski",
        "adesanya", "mcgregor", "poirier", "holloway",
    ]

    def predict(self, entity1: str, entity2: str, date: str, context: dict) -> PredictionResult:
        sources = []

        # 1. Fighter stats
        f1_stats = ufcstats.get_fighter_stats(entity1)
        f2_stats = ufcstats.get_fighter_stats(entity2)
        if f1_stats.get("slpm") or f2_stats.get("slpm"):
            sources.append("UFCStats.com")

        # 2. ELO
        elo_predictor = elo_module.EloPredictor(default_elo=1550)
        seeds = _load_ufc_seeds()
        for name, info in seeds.items():
            if "elo" in info:
                elo_predictor.set(name, info["elo"])
        # Fighter win_rate modifies ELO seed
        f1_elo = elo_predictor.get(entity1.lower())
        f2_elo = elo_predictor.get(entity2.lower())
        elo_p1 = elo_module.win_probability(f1_elo, f2_elo)

        # 3. Statistical prediction based on fighting stats
        stat_p1 = _stat_model(f1_stats, f2_stats)

        # 4. H2H
        h2h = ufcstats.get_h2h(entity1, entity2)
        h2h_p1 = h2h.get("f1_win_rate", 0.5)
        h2h_w = min(0.20, h2h.get("total", 0) * 0.10)

        # Blend: ELO 40%, stats 40%, H2H 20%
        blended_p1 = (0.40 - h2h_w / 2) * elo_p1 + (0.40 - h2h_w / 2) * stat_p1 + h2h_w * h2h_p1
        blended_p1 = max(0.05, min(0.95, blended_p1))

        # 5. News (injuries, layoffs)
        f1_news = news.get_sentiment(entity1)
        f2_news = news.get_sentiment(entity2)
        all_flags = f1_news.get("flags", []) + f2_news.get("flags", [])

        sentiment_adj = (f1_news.get("score", 0) - f2_news.get("score", 0)) * 0.10
        blended_p1 = max(0.05, min(0.95, blended_p1 + sentiment_adj))

        # 6. ML ensemble
        ctx_ml = {"h2h_result": h2h_p1 - 0.5}
        features = ml_ensemble.build_ufc_features(f1_stats, f2_stats, ctx_ml)
        ml_model = ml_ensemble.MLEnsemble(sport="ufc", n_classes=2)
        if ml_model.load():
            ml_dict = ml_model.predict_dict(features, ["f2_win", "f1_win"])
            blended_p1 = 0.70 * blended_p1 + 0.30 * ml_dict["f1_win"]

        probs = {"p1_win": round(blended_p1, 4), "p2_win": round(1 - blended_p1, 4)}

        # 6b. Agent debate — optional swarm-intelligence layer (requires ANTHROPIC_API_KEY)
        debate_result = None
        try:
            from src.models.agent_debate import run_debate, blend_with_ml
            debate_ctx = {
                "date": date,
                "competition": context.get("competition", "UFC"),
                "ml_probability_a": probs.get("p1_win", 0.5),
                "elo_a": f1_elo,
                "elo_b": f2_elo,
                "form_a": f1_stats.get("win_rate", 0.5),
                "form_b": f2_stats.get("win_rate", 0.5),
                "avg_goals_a": f1_stats.get("slpm", 3.5),
                "avg_goals_b": f2_stats.get("slpm", 3.5),
                "avg_conceded_a": f1_stats.get("sapm", 3.0),
                "avg_conceded_b": f2_stats.get("sapm", 3.0),
                "news_flags": all_flags[:4],
                "key_factors": factors,
            }
            debate_result = run_debate(entity1, entity2, "mma", debate_ctx)
            if debate_result:
                blended_p1_new = blend_with_ml(probs["p1_win"], debate_result, ml_weight=0.72)
                probs = {
                    "p1_win": round(blended_p1_new, 4),
                    "p2_win": round(max(0.05, 1 - blended_p1_new), 4),
                }
        except Exception:
            pass

        # 7. Market
        mkt = market.get_market_odds(entity1, entity2, "ufc")
        market_mapped = None
        if mkt:
            market_mapped = {"p1_win": mkt.get("home_win"), "p2_win": mkt.get("away_win")}

        # 8. Edge + Kelly
        edges = edge_mod.calculate_edge(probs, market_mapped or {})
        best_outcome, best_info = edge_mod.best_bet(edges)
        best_edge = best_info["edge_pct"] if best_info else 0.0
        kelly_pct = 0.0
        if best_info and best_info["edge"] > 0:
            decimal = odds_mod.prob_to_decimal(best_info["market_prob"])
            kelly_pct = kelly_mod.kelly_fraction(best_info["model_prob"], decimal)

        factors = _build_factors(f1_stats, f2_stats, entity1, entity2)

        return PredictionResult(
            sport="UFC/MMA",
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
            data_sources=sources or ["Seeded fighter stats"],
            model_breakdown={
                "ELO": {"p1_win": elo_p1},
                "Statistical": {"p1_win": stat_p1},
                **({
                    "Agent Debate": {
                        "p1_win": round(debate_result.consensus_probability, 4),
                        "summary": debate_result.summary,
                        "agents": {e.agent: {"p": e.probability, "conf": e.confidence, "reason": e.reasoning}
                                   for e in debate_result.agent_estimates},
                    }
                } if debate_result else {}),
            },
            competition=context.get("competition", "UFC"),
        )


def _stat_model(f1: dict, f2: dict) -> float:
    """Simple striking + grappling statistical model."""
    score = 0.0

    # Striking advantage: SLpM * accuracy vs SApM * defense
    f1_strike = f1.get("slpm", 3.5) * f1.get("str_acc", 0.45)
    f2_strike = f2.get("slpm", 3.5) * f2.get("str_acc", 0.45)
    f1_defense = f1.get("str_def", 0.55)
    f2_defense = f2.get("str_def", 0.55)

    strike_edge = (f1_strike * f2_defense) - (f2_strike * f1_defense)
    score += strike_edge * 0.5

    # Grappling advantage
    f1_td = f1.get("td_avg", 1.5) * f1.get("td_acc", 0.40)
    f2_td = f2.get("td_avg", 1.5) * f2.get("td_acc", 0.40)
    td_edge = f1_td * (1 - f2.get("td_def", 0.70)) - f2_td * (1 - f1.get("td_def", 0.70))
    score += td_edge * 0.3

    # Win rate
    wr_diff = f1.get("win_rate", 0.75) - f2.get("win_rate", 0.75)
    score += wr_diff * 0.2

    # Convert score to probability
    import math
    prob = 1 / (1 + math.exp(-score * 2))
    return round(max(0.10, min(0.90, prob)), 4)


def _build_factors(f1: dict, f2: dict, name1: str, name2: str) -> list:
    factors = []
    reach_diff = f1.get("reach_cm", 183) - f2.get("reach_cm", 183)
    if abs(reach_diff) > 5:
        leader = name1 if reach_diff > 0 else name2
        factors.append(f"{leader} has significant reach advantage ({abs(reach_diff):.0f}cm)")

    f1_finish = f1.get("finish_rate", 0.5)
    f2_finish = f2.get("finish_rate", 0.5)
    if f1_finish > 0.7:
        factors.append(f"{name1} high finish rate ({f1_finish*100:.0f}%) — dangerous finisher")
    if f2_finish > 0.7:
        factors.append(f"{name2} high finish rate ({f2_finish*100:.0f}%) — dangerous finisher")

    age_diff = f1.get("age", 30) - f2.get("age", 30)
    if abs(age_diff) > 5:
        younger = name1 if age_diff < 0 else name2
        factors.append(f"{younger} significantly younger ({abs(age_diff):.0f} yrs) — stamina edge")

    if f1.get("sub_avg", 0) > 1.5:
        factors.append(f"{name1} elite submission threat ({f1['sub_avg']:.1f} avg)")
    if f2.get("sub_avg", 0) > 1.5:
        factors.append(f"{name2} elite submission threat ({f2['sub_avg']:.1f} avg)")

    return factors
