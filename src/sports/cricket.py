"""Cricket prediction handler."""
import math
import numpy as np
from src.sports.base import AbstractSport, PredictionResult
from src.data.scrapers import cricsheet
from src.data import news, market
from src.models import elo as elo_module, calibrator, ml_ensemble
from src.market import edge as edge_mod, kelly as kelly_mod, odds as odds_mod


FORMAT_KEYWORDS = {
    "test": ["test", "test match", "ashes"],
    "t20": ["t20", "twenty20", "ipl", "big bash", "cpl", "t20i", "world t20"],
    "odi": ["odi", "one day", "world cup", "cricket world cup", "50 over"],
}


def _detect_format(context: dict) -> str:
    comp = (context.get("competition", "") + " " + context.get("venue", "")).lower()
    for fmt, keywords in FORMAT_KEYWORDS.items():
        if any(kw in comp for kw in keywords):
            return fmt
    return "odi"  # default


class CricketPredictor(AbstractSport):

    sport_name = "cricket"
    sport_keywords = [
        "cricket", "test match", "odi", "t20", "ipl", "ashes",
        "big bash", "bcci", "ecb", "batting", "bowling", "wicket",
        "india", "australia cricket", "england cricket",
    ]

    def predict(self, entity1: str, entity2: str, date: str, context: dict) -> PredictionResult:
        fmt = _detect_format(context)
        sources = ["CricSheet seeded stats"]

        # 1. Team stats
        t1_stats = cricsheet.get_team_stats(entity1, fmt)
        t2_stats = cricsheet.get_team_stats(entity2, fmt)
        h2h = cricsheet.get_h2h(entity1, entity2)

        # 2. ELO
        elo_predictor = elo_module.EloPredictor(default_elo=1700)
        elo_predictor.set(entity1.lower(), t1_stats.get("elo", 1700))
        elo_predictor.set(entity2.lower(), t2_stats.get("elo", 1700))

        is_neutral = context.get("is_neutral", False)
        home_adv = 0 if is_neutral else 60  # home advantage in cricket is significant
        elo_probs = elo_predictor.predict(entity1.lower(), entity2.lower(), home_advantage=home_adv)

        # 3. Batting/bowling statistical model
        stat_p1 = _batting_bowling_model(t1_stats, t2_stats)

        # 4. H2H weighting
        h2h_total = h2h.get("total", 0)
        h2h_p1 = h2h.get("team1_wins", 0) / max(h2h_total, 1)
        h2h_w = min(0.20, h2h_total * 0.005)

        # 5. Format-specific adjustment (T20 more variance, Test more skill-based)
        variance_factor = {"test": 0.85, "odi": 1.0, "t20": 1.15}.get(fmt, 1.0)

        # Blend
        base_p1 = (1 - h2h_w) * (0.55 * elo_probs["a_win"] + 0.45 * stat_p1) + h2h_w * h2h_p1

        # Test matches: more regression to ELO (skill dominates)
        if fmt == "test":
            base_p1 = 0.7 * base_p1 + 0.3 * elo_probs["a_win"]

        base_p1 = max(0.05, min(0.95, base_p1))

        # 6. ML ensemble (if trained)
        ml_model = ml_ensemble.MLEnsemble(sport="cricket", n_classes=2)
        if ml_model.load():
            elo1 = t1_stats.get("elo", 1700)
            elo2 = t2_stats.get("elo", 1700)
            fmt_enc = [int(fmt == "odi"), int(fmt == "t20"), int(fmt == "test")]
            ml_feat = np.array([
                (elo1 - elo2) / 400.0,
                elo1 / (elo2 + 1e-9),
                t1_stats.get("win_rate", 0.5) - t2_stats.get("win_rate", 0.5),
                t1_stats.get("batting_avg", 28) - t2_stats.get("batting_avg", 28),
                t2_stats.get("bowling_avg", 29) - t1_stats.get("bowling_avg", 29),
                t1_stats.get("run_rate", 5) - t2_stats.get("run_rate", 5),
                *fmt_enc,
                base_p1,
            ], dtype=np.float32)
            ml_dict = ml_model.predict_dict(ml_feat, ["p2_win", "p1_win"])
            base_p1 = 0.70 * base_p1 + 0.30 * ml_dict["p1_win"]
            base_p1 = max(0.05, min(0.95, base_p1))
            sources.append("ML Ensemble (cricket)")

        # 7. Draw probability (Test only)
        if fmt == "test":
            draw_prob = 0.22 * (1 - abs(base_p1 - 0.5) * 2)  # more draws for close matches
            draw_prob = max(0.10, min(0.30, draw_prob))
            p1_win = base_p1 * (1 - draw_prob)
            p2_win = (1 - base_p1) * (1 - draw_prob)
            probs = {
                "p1_win": round(p1_win, 4),
                "draw": round(draw_prob, 4),
                "p2_win": round(p2_win, 4),
            }
        else:
            probs = {"p1_win": round(base_p1, 4), "p2_win": round(1 - base_p1, 4)}

        # 8. News
        t1_news = news.get_sentiment(entity1 + " cricket")
        t2_news = news.get_sentiment(entity2 + " cricket")
        all_flags = t1_news.get("flags", []) + t2_news.get("flags", [])

        sent_adj = (t1_news.get("score", 0) - t2_news.get("score", 0)) * 0.06
        if "p1_win" in probs:
            probs["p1_win"] = max(0.01, probs["p1_win"] + sent_adj)
            if "p2_win" in probs:
                probs["p2_win"] = max(0.01, 1.0 - probs["p1_win"] - probs.get("draw", 0))

        probs = calibrator.normalize(probs)

        # 9. Market
        mkt = market.get_market_odds(entity1, entity2, "cricket")
        market_mapped = None
        if mkt:
            keys = list(probs.keys())
            market_mapped = {k: mkt.get("home_win" if i == 0 else "away_win") for i, k in enumerate(keys)}

        # 10. Edge + Kelly
        edges = edge_mod.calculate_edge(probs, market_mapped or {})
        best_outcome, best_info = edge_mod.best_bet(edges)
        best_edge = best_info["edge_pct"] if best_info else 0.0
        kelly_pct = 0.0
        if best_info and best_info["edge"] > 0:
            decimal = odds_mod.prob_to_decimal(best_info["market_prob"])
            kelly_pct = kelly_mod.kelly_fraction(best_info["model_prob"], decimal)

        factors = [
            f"Format: {fmt.upper()} — {'High variance format' if fmt == 't20' else 'Skill-dominant format'}",
            f"ELO Rankings: {entity1} #{t1_stats.get('ranking', '?')} vs {entity2} #{t2_stats.get('ranking', '?')}",
        ]
        if h2h_total > 5:
            factors.append(f"H2H: {entity1} {h2h.get('team1_wins',0)}-{h2h.get('team2_wins',0)} {entity2}")
        if not is_neutral:
            factors.append(f"Home advantage: {entity1} playing at home")

        return PredictionResult(
            sport="Cricket",
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
            data_sources=sources,
            model_breakdown={"ELO": {"p1_win": elo_probs["a_win"]}, "Batting/Bowling": {"p1_win": stat_p1}},
            competition=context.get("competition", f"{fmt.upper()} Match"),
            is_neutral=is_neutral,
        )


def _batting_bowling_model(t1: dict, t2: dict) -> float:
    """Score-based model using batting/bowling averages and run rates."""
    # Effective run rate difference (attack vs defense)
    t1_run_rate = t1.get("run_rate", 5.0)
    t2_run_rate = t2.get("run_rate", 5.0)
    t1_bat = t1.get("batting_avg", 28.0)
    t2_bat = t2.get("batting_avg", 28.0)
    t1_bowl = t2.get("bowling_avg", 30.0)  # opponent's bowling avg (lower = better bowling)
    t2_bowl = t1.get("bowling_avg", 30.0)

    # Combined strength score
    t1_score = (t1_bat / max(t1_bowl, 10)) * t1_run_rate
    t2_score = (t2_bat / max(t2_bowl, 10)) * t2_run_rate

    total = t1_score + t2_score
    p1 = t1_score / total if total > 0 else 0.5
    return round(max(0.15, min(0.85, p1)), 4)
