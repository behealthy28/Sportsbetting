"""Darts and Badminton/Table Tennis prediction handler."""
import math
import numpy as np
from src.sports.base import AbstractSport, PredictionResult
from src.data import news, market
from src.models import elo as elo_module, calibrator, ml_ensemble
from src.market import edge as edge_mod, kelly as kelly_mod, odds as odds_mod

# Darts player ELO seeds — top 40 PDC players (2024-25 world rankings + prize money).
# Scale: 1950=all-time great, 1900=slam winner, 1800=top-16, 1700=tour card holder.
DARTS_ELO = {
    # PDC Top 10
    "luke humphries": 1930, "cool hand luke": 1930,
    "luke littler": 1920, "the nuke": 1920,
    "michael van gerwen": 1910, "mvg": 1910,
    "michael smith": 1890, "bully boy": 1890,
    "stephen bunting": 1860, "bullet": 1860,
    "peter wright": 1850, "snakebite": 1850,
    "chris dobey": 1840, "hollywood": 1840,
    "gerwyn price": 1830, "iceman": 1830,
    "gary anderson": 1820, "the flying scotsman": 1820,
    "jonny clayton": 1810, "ferret": 1810,
    # PDC 11–25
    "rob cross": 1800, "voltage": 1800,
    "jose de sousa": 1800, "special one": 1800,
    "damon heta": 1790, "the heat": 1790,
    "dimitri van den bergh": 1790, "the dreammaker": 1790,
    "dave chisnall": 1790, "chizzy": 1790,
    "danny noppert": 1780, "freeze": 1780,
    "callan rydz": 1780,
    "martin schindler": 1770,
    "mike de decker": 1770,
    "brendan dolan": 1770, "the history maker": 1770,
    "josh rock": 1780, "the northern irish hammer": 1780,
    "ryan searle": 1770, "heavy metal": 1770,
    "florian hempel": 1760,
    "nathan aspinall": 1800, "the asp": 1800,
    "james wade": 1800, "the machine": 1800,
    "ian white": 1780, "diamond white": 1780,
    "daryl gurney": 1770, "superchín": 1770,
    "raymond van barneveld": 1790, "barney": 1790,
    "phil taylor": 2000, "the power": 2000,
    "andy hamilton": 1750, "the hammer": 1750,
    "glen durrant": 1780, "duzza": 1780,
}

# Badminton player ELO seeds — top 40 BWF world rankings (2024-25).
# Scale: 1950=world #1, 1900=top-5, 1800=top-20, 1700=top-50.
BADMINTON_ELO = {
    # Men's Singles top 20
    "viktor axelsen": 1940, "axelsen": 1940,
    "shi yuqi": 1900, "lee zii jia": 1870,
    "kunlavut vitidsarn": 1880,
    "jonatan christie": 1850,
    "chou tien chen": 1840,
    "lakshya sen": 1830,
    "ng ka long angus": 1820,
    "priyanshu rajawat": 1810,
    "weng hong yang": 1800,
    "loh kean yew": 1810,
    "thomas rouxel": 1790,
    "mark caljouw": 1790,
    "nhat nguyen": 1780,
    "lu guangzu": 1820,
    "kodai naraoka": 1830,
    # Women's Singles top 20
    "an se-young": 1940, "an se young": 1940,
    "carolina marin": 1900, "marin": 1900,
    "akane yamaguchi": 1880, "yamaguchi": 1880,
    "chen yu fei": 1890,
    "wang zhi yi": 1870,
    "tai tzu ying": 1860, "tai tzu-ying": 1860,
    "pusarla v sindhu": 1840, "pv sindhu": 1840,
    "ratchanok intanon": 1830,
    "han yue": 1820,
    "busanan ongbamrungphan": 1800,
    "line christophersen": 1790,
    "gregoria mariska tunjung": 1800,
    "nozomi okuhara": 1810,
    "aya ohori": 1790,
    "evgeniya kosetskaya": 1780,
}

# Table Tennis player ELO seeds — ITTF world rankings 2024-25.
TABLE_TENNIS_ELO = {
    # Men's
    "fan zhendong": 1980, "wang chuqin": 1950,
    "ma long": 1960, "liang jingkun": 1920,
    "truls moregard": 1900,
    "felix lebrun": 1890, "alexis lebrun": 1870,
    "lin yun-ju": 1880, "tomokazu harimoto": 1870,
    "anders lind": 1860, "simon gauzy": 1850,
    "quadri aruna": 1830, "darko jorgic": 1820,
    "benedikt duda": 1830, "patrick franziska": 1820,
    "dang qiu": 1840, "chuang chih-yuan": 1830,
    # Women's
    "sun yingsha": 1960, "chen meng": 1950,
    "wang manyu": 1930, "wang yidi": 1910,
    "mima ito": 1890, "hina hayata": 1870,
    "bernadette szocs": 1850, "wu yue": 1840,
    "bruna takahashi": 1820, "ni xia lian": 1820,
}


class DartsPredictor(AbstractSport):

    sport_name = "darts"
    sport_keywords = [
        "darts", "pdc", "bdo", "world darts", "premier league darts",
        "van gerwen", "mvg", "wright", "price", "smith", "littler", "humphries",
        "world championship darts", "masters darts", "uk open",
    ]

    def predict(self, entity1: str, entity2: str, date: str, context: dict) -> PredictionResult:
        elo_predictor = elo_module.EloPredictor(default_elo=1700)
        for name, elo in DARTS_ELO.items():
            elo_predictor.set(name, elo)

        f1_elo = elo_predictor.get(entity1.lower())
        f2_elo = elo_predictor.get(entity2.lower())
        p1_win = elo_module.win_probability(f1_elo, f2_elo)

        f1_news = news.get_sentiment(entity1 + " darts")
        f2_news = news.get_sentiment(entity2 + " darts")
        all_flags = f1_news.get("flags", []) + f2_news.get("flags", [])
        sent_adj = (f1_news.get("score", 0) - f2_news.get("score", 0)) * 0.08
        p1_win = max(0.05, min(0.95, p1_win + sent_adj))

        ml_model = ml_ensemble.MLEnsemble(sport="darts", n_classes=2)
        if ml_model.load():
            ml_feat = np.array([(float(f1_elo) - float(f2_elo)) / 400.0,
                                 float(f1_elo) / float(f2_elo), p1_win, 0.0, 0.0],
                               dtype=np.float32)
            ml_dict = ml_model.predict_dict(ml_feat, ["p2_win", "p1_win"])
            p1_win = 0.70 * p1_win + 0.30 * ml_dict["p1_win"]
            p1_win = max(0.05, min(0.95, p1_win))

        probs = {"p1_win": round(p1_win, 4), "p2_win": round(1 - p1_win, 4)}

        # Agent debate — optional swarm-intelligence layer (requires ANTHROPIC_API_KEY)
        debate_result = None
        try:
            from src.models.agent_debate import run_debate, blend_with_ml
            debate_ctx = {
                "date": date,
                "competition": context.get("competition", "Darts"),
                "ml_probability_a": probs["p1_win"],
                "elo_a": f1_elo,
                "elo_b": f2_elo,
                "form_a": probs["p1_win"],
                "news_flags": all_flags[:4],
            }
            debate_result = run_debate(entity1, entity2, "darts", debate_ctx)
            if debate_result:
                p1_new = blend_with_ml(probs["p1_win"], debate_result, ml_weight=0.72)
                probs = {"p1_win": round(p1_new, 4), "p2_win": round(max(0.05, 1 - p1_new), 4)}
        except Exception:
            pass

        mkt = market.get_market_odds(entity1, entity2, "darts")
        market_mapped = None
        if mkt:
            market_mapped = {"p1_win": mkt.get("home_win"), "p2_win": mkt.get("away_win")}

        edges = edge_mod.calculate_edge(probs, market_mapped or {})
        best_outcome, best_info = edge_mod.best_bet(edges)
        kelly_pct = 0.0
        if best_info and best_info["edge"] > 0:
            decimal = odds_mod.prob_to_decimal(best_info["market_prob"])
            kelly_pct = kelly_mod.kelly_fraction(best_info["model_prob"], decimal)

        return PredictionResult(
            sport="Darts",
            entity1=entity1, entity2=entity2, date=date,
            probabilities=probs, market_probs=market_mapped, edges=edges,
            best_bet=best_outcome,
            best_edge_pct=best_info["edge_pct"] if best_info else 0.0,
            kelly_stake_pct=kelly_pct,
            confidence=calibrator.confidence_score(probs),
            key_factors=[f"ELO: {entity1} ({f1_elo:.0f}) vs {entity2} ({f2_elo:.0f})"],
            news_flags=all_flags[:3],
            data_sources=["PDC ELO ratings"],
            model_breakdown={"ELO": {"p1_win": p1_win}},
            competition=context.get("competition", "Darts"),
        )


class BadmintonPredictor(AbstractSport):

    sport_name = "badminton"
    sport_keywords = [
        "badminton", "bwf", "all england", "thomas cup", "uber cup",
        "axelsen", "marin", "carolina", "shi yuqi", "vitidsarn",
        "table tennis", "ittf", "tt", "ping pong",
        "fan zhendong", "ma long", "wang chuqin", "chen meng",
    ]

    def predict(self, entity1: str, entity2: str, date: str, context: dict) -> PredictionResult:
        sport_label = "Table Tennis" if _is_table_tennis(entity1 + " " + entity2 + " " + context.get("competition", "")) else "Badminton"
        elo_seeds = TABLE_TENNIS_ELO if sport_label == "Table Tennis" else BADMINTON_ELO

        elo_predictor = elo_module.EloPredictor(default_elo=1700)
        for name, elo in elo_seeds.items():
            elo_predictor.set(name, elo)

        f1_elo = elo_predictor.get(entity1.lower())
        f2_elo = elo_predictor.get(entity2.lower())
        p1_win = elo_module.win_probability(f1_elo, f2_elo)

        f1_news = news.get_sentiment(entity1)
        f2_news = news.get_sentiment(entity2)
        all_flags = f1_news.get("flags", []) + f2_news.get("flags", [])
        sent_adj = (f1_news.get("score", 0) - f2_news.get("score", 0)) * 0.06
        p1_win = max(0.05, min(0.95, p1_win + sent_adj))

        sport_key = "badminton" if sport_label == "Badminton" else "table_tennis"
        ml_model = ml_ensemble.MLEnsemble(sport=sport_key, n_classes=2)
        if ml_model.load():
            ml_feat = np.array([(float(f1_elo) - float(f2_elo)) / 400.0,
                                 float(f1_elo) / float(f2_elo), p1_win, 0.0, 0.0],
                               dtype=np.float32)
            ml_dict = ml_model.predict_dict(ml_feat, ["p2_win", "p1_win"])
            p1_win = 0.70 * p1_win + 0.30 * ml_dict["p1_win"]
            p1_win = max(0.05, min(0.95, p1_win))

        probs = {"p1_win": round(p1_win, 4), "p2_win": round(1 - p1_win, 4)}

        # Agent debate — optional swarm-intelligence layer (requires ANTHROPIC_API_KEY)
        debate_result = None
        try:
            from src.models.agent_debate import run_debate, blend_with_ml
            debate_ctx = {
                "date": date,
                "competition": context.get("competition", sport_label),
                "ml_probability_a": probs["p1_win"],
                "elo_a": f1_elo,
                "elo_b": f2_elo,
                "form_a": probs["p1_win"],
                "news_flags": all_flags[:4],
            }
            debate_result = run_debate(entity1, entity2, sport_label.lower(), debate_ctx)
            if debate_result:
                p1_new = blend_with_ml(probs["p1_win"], debate_result, ml_weight=0.72)
                probs = {"p1_win": round(p1_new, 4), "p2_win": round(max(0.05, 1 - p1_new), 4)}
        except Exception:
            pass

        mkt = market.get_market_odds(entity1, entity2, sport_label.lower())
        market_mapped = None
        if mkt:
            market_mapped = {"p1_win": mkt.get("home_win"), "p2_win": mkt.get("away_win")}

        edges = edge_mod.calculate_edge(probs, market_mapped or {})
        best_outcome, best_info = edge_mod.best_bet(edges)
        kelly_pct = 0.0
        if best_info and best_info["edge"] > 0:
            decimal = odds_mod.prob_to_decimal(best_info["market_prob"])
            kelly_pct = kelly_mod.kelly_fraction(best_info["model_prob"], decimal)

        return PredictionResult(
            sport=sport_label,
            entity1=entity1, entity2=entity2, date=date,
            probabilities=probs, market_probs=market_mapped, edges=edges,
            best_bet=best_outcome,
            best_edge_pct=best_info["edge_pct"] if best_info else 0.0,
            kelly_stake_pct=kelly_pct,
            confidence=calibrator.confidence_score(probs),
            key_factors=[f"ELO: {entity1} ({f1_elo:.0f}) vs {entity2} ({f2_elo:.0f})"],
            news_flags=all_flags[:3],
            data_sources=["BWF/ITTF ELO ratings"],
            model_breakdown={"ELO": {"p1_win": p1_win}},
            competition=context.get("competition", sport_label),
        )


def _is_table_tennis(text: str) -> bool:
    tt_keywords = ["table tennis", "ittf", "ping pong", "fan zhendong", "ma long", "wang chuqin", "tt "]
    return any(kw in text.lower() for kw in tt_keywords)
