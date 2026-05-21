"""Football/Soccer prediction handler."""
from src.sports.base import AbstractSport, PredictionResult
from src.data.scrapers import fbref
from src.data import news, market
from src.models import dixon_coles, elo as elo_module, calibrator, ml_ensemble
from src.market import edge as edge_mod, kelly as kelly_mod, odds as odds_mod
import json
import numpy as np
from pathlib import Path

_DC_MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "models" / "football_dc.json"
_fitted_dc: "dixon_coles.DixonColesModel | None" = None


def _load_fitted_dc() -> "dixon_coles.DixonColesModel | None":
    """Load the MLE-fitted Dixon-Coles model from disk (cached in module scope)."""
    global _fitted_dc
    if _fitted_dc is not None:
        return _fitted_dc
    if not _DC_MODEL_PATH.exists():
        return None
    try:
        params = json.loads(_DC_MODEL_PATH.read_text())
        m = dixon_coles.DixonColesModel()
        m.attack = params["attack"]
        m.defense = params["defense"]
        m.rho = float(params["rho"])
        m.is_fitted = True
        _fitted_dc = m
        return _fitted_dc
    except Exception:
        return None


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

    def predict(self, entity1: str, entity2: str, date: str, context: dict) -> PredictionResult:
        competition = context.get("competition", "")
        is_neutral = context.get("is_neutral", False)
        comp_weight = _get_comp_weight(competition)

        # 1. Fetch team data — Understat (xG) → ESPN → ELO-derived
        home_data = fbref.get_team_data(entity1)
        away_data = fbref.get_team_data(entity2)
        h2h = fbref.get_h2h(entity1, entity2)
        sources = list(set(
            home_data.get("data_sources", []) + away_data.get("data_sources", [])
        )) or ["FBRef/ESPN"]

        # 2. News sentiment
        home_news = news.get_sentiment(entity1)
        away_news = news.get_sentiment(entity2)
        all_flags = home_news.get("flags", []) + away_news.get("flags", [])
        if all_flags:
            sources.append("Google News RSS")

        # 3. ELO — live ratings from ClubElo/eloratings.net via home_data["elo"]
        elo_predictor = elo_module.EloPredictor(default_elo=1700)
        if home_data.get("elo"):
            elo_predictor.set(entity1.lower(), home_data["elo"])
        if away_data.get("elo"):
            elo_predictor.set(entity2.lower(), away_data["elo"])

        home_adv = 0 if is_neutral else HOME_ADVANTAGE_ELO
        elo_probs = elo_predictor.predict(entity1.lower(), entity2.lower(), home_advantage=home_adv)
        elo_result = {
            "home_win": elo_probs["a_win"],
            "draw": elo_probs["draw"],
            "away_win": elo_probs["b_win"],
        }

        # 4. Dixon-Coles — prefer xG (expected goals) over actual goals when available.
        # xG is more predictive than goals scored because it removes luck from finishing.
        injury_adj_home = context.get("home_key_players", 1.0)
        injury_adj_away = context.get("away_key_players", 1.0)

        home_attack_rate = home_data.get("avg_xg") or home_data.get("avg_goals", 1.35)
        away_attack_rate = away_data.get("avg_xg") or away_data.get("avg_goals", 1.35)
        home_concede_rate = home_data.get("avg_xga") or home_data.get("avg_conceded", 1.20)
        away_concede_rate = away_data.get("avg_xga") or away_data.get("avg_conceded", 1.20)
        using_xg = bool(home_data.get("avg_xg") or away_data.get("avg_xg"))

        fitted_dc = _load_fitted_dc()
        if fitted_dc is not None and fitted_dc.attack.get(entity1.lower()) and fitted_dc.attack.get(entity2.lower()):
            # MLE-fitted model knows this team pair — use their learned strengths
            dc_result = fitted_dc.predict(entity1.lower(), entity2.lower(), neutral=is_neutral)
            # Apply injury adjustments as a post-hoc scaling on λ via probability redistribution
            if injury_adj_home != 1.0 or injury_adj_away != 1.0:
                dc_result = calibrator.apply_injury_adjustment(
                    dc_result, injury_adj_home, injury_adj_away
                )
        else:
            # Fall back to approximation for teams not in the fitted model (e.g. national teams)
            dc_result = dixon_coles.quick_predict(
                home_goals_avg=home_attack_rate,
                away_goals_avg=away_attack_rate,
                home_conceded_avg=home_concede_rate,
                away_conceded_avg=away_concede_rate,
                league_avg_goals=1.35,
                neutral=is_neutral,
                injury_adj_home=injury_adj_home,
                injury_adj_away=injury_adj_away,
            )

        # 5. ML ensemble (only used when pre-trained models exist in data/models/)
        ctx_dict = {
            "h2h_home_win_rate": h2h.get("team1_win_rate", 0.35),
            "home_key_players": injury_adj_home,
            "away_key_players": injury_adj_away,
            "home_days_rest": context.get("home_days_rest", 7),
            "away_days_rest": context.get("away_days_rest", 7),
            "competition_weight": comp_weight,
            "is_neutral": int(is_neutral),
        }
        features = ml_ensemble.build_football_features(home_data, away_data, ctx_dict)
        ml_model = ml_ensemble.MLEnsemble(sport="football", n_classes=3)
        ml_loaded = ml_model.load()
        ml_result = None
        if ml_loaded:
            ml_dict = ml_model.predict_dict(features, ["away_win", "draw", "home_win"])
            ml_result = {
                "home_win": ml_dict["home_win"],
                "draw": ml_dict["draw"],
                "away_win": ml_dict["away_win"],
            }

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

        # 7b. Agent debate — optional swarm-intelligence layer (requires ANTHROPIC_API_KEY)
        debate_result = None
        try:
            from src.models.agent_debate import run_debate, blend_with_ml
            debate_ctx = {
                "date": date,
                "competition": competition,
                "ml_probability_a": blended.get("home_win", 0.4),
                "elo_a": home_data.get("elo", 1500),
                "elo_b": away_data.get("elo", 1500),
                "form_a": home_data.get("form", 0.5),
                "form_b": away_data.get("form", 0.5),
                "avg_goals_a": home_data.get("avg_goals", 1.4),
                "avg_goals_b": away_data.get("avg_goals", 1.4),
                "avg_conceded_a": home_data.get("avg_conceded", 1.2),
                "avg_conceded_b": away_data.get("avg_conceded", 1.2),
                "home_advantage": entity1 if not is_neutral else "neutral",
                "news_flags": all_flags[:4],
                "key_factors": factors if "factors" in dir() else [],
            }
            debate_result = run_debate(entity1, entity2, "football", debate_ctx)
            if debate_result:
                from src.models.agent_debate import blend_with_ml
                blended_home = blend_with_ml(blended["home_win"], debate_result, ml_weight=0.72)
                shift = blended_home - blended["home_win"]
                # Distribute shift proportionally to away/draw
                away_draw_total = blended["away_win"] + blended["draw"]
                if away_draw_total > 0:
                    blended["home_win"] = blended_home
                    blended["away_win"] = round(blended["away_win"] - shift * blended["away_win"] / away_draw_total, 4)
                    blended["draw"]     = round(max(0.01, 1 - blended["home_win"] - blended["away_win"]), 4)
        except Exception:
            pass

        # 8. Market odds
        mkt = market.get_market_odds(entity1, entity2, "football")
        if mkt:
            sources.append("Polymarket/Kalshi")
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

        fitted_dc = _load_fitted_dc()
        dc_mode_label = "MLE-fitted" if (fitted_dc and fitted_dc.attack.get(entity1.lower())) else "approx"
        dc_label = f"Dixon-Coles ({', '.join(filter(None, [('xG' if using_xg else None), dc_mode_label]))})"
        breakdown = {dc_label: dc_result, "ELO (ClubElo/eloratings)": elo_result}
        if ml_result:
            breakdown["ML Ensemble"] = ml_result
        if debate_result:
            breakdown["Agent Debate"] = {
                "home_win": round(debate_result.consensus_probability, 4),
                "summary": debate_result.summary,
                "agents": {e.agent: {"p": e.probability, "conf": e.confidence, "reason": e.reasoning}
                           for e in debate_result.agent_estimates},
            }

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
            data_sources=list(dict.fromkeys(sources)),
            model_breakdown=breakdown,
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
