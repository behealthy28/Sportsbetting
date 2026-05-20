"""Boxing prediction handler."""
import json
import math
import numpy as np
from pathlib import Path
from src.sports.base import AbstractSport, PredictionResult
from src.data import news, market
from src.models import elo as elo_module, calibrator, ml_ensemble
from src.market import edge as edge_mod, kelly as kelly_mod, odds as odds_mod

PLAYERS_FILE = Path(__file__).parent.parent.parent / "data" / "mappings" / "players.json"

# ELO ratings seeded from recent performance (2024-25 records) — all 17 weight divisions.
# Scale: 2000=all-time great, 1900=elite champion, 1800=solid top-contender, 1700=journeyman.
BOXING_ELO = {
    # ── Heavyweight (200+ lb) ─────────────────────────────────────────────────
    "oleksandr usyk":    1950,
    "tyson fury":        1910,
    "anthony joshua":    1830,
    "deontay wilder":    1820,
    "daniel dubois":     1810,
    "joe joyce":         1760,
    "zhilei zhang":      1800,
    "frank sanchez":     1790,
    "joseph parker":     1800,
    "otto wallin":       1760,
    # ── Cruiserweight (200 lb) ────────────────────────────────────────────────
    "jai opetaia":       1860,
    "mairis briedis":    1840,
    "ilunga makabu":     1810,
    "badou jack":        1790,
    "yuniel dorticos":   1800,
    # ── Light Heavyweight (175 lb) ────────────────────────────────────────────
    "artur beterbiev":   1900,
    "dmitry bivol":      1890,
    "joe smith":         1790,
    "anthony yarde":     1800,
    "callum smith":      1800,
    "christian plant":   1790,
    # ── Super Middleweight (168 lb) ───────────────────────────────────────────
    "canelo alvarez":    1930,
    "david benavidez":   1860,
    "edgar berlanga":    1820,
    "caleb plant":       1820,
    "chris eubank jr":   1810,
    "william scull":     1800,
    # ── Middleweight (160 lb) ─────────────────────────────────────────────────
    "jermall charlo":    1820,
    "carlos adames":     1800,
    "erislandy lara":    1810,
    "janibek alimkhanuly": 1850,
    "michael zerafa":    1780,
    # ── Super Welterweight (154 lb) ───────────────────────────────────────────
    "terence crawford":  1900,
    "errol spence":      1870,
    "tim tszyu":         1860,
    "sebastian fundora": 1830,
    "brian castano":     1810,
    "tony harrison":     1790,
    # ── Welterweight (147 lb) ─────────────────────────────────────────────────
    "jaron ennis":       1880,
    "keith thurman":     1810,
    "vergil ortiz":      1820,
    "eimantas stanionis": 1800,
    "cody crowley":      1790,
    # ── Super Lightweight (140 lb) ────────────────────────────────────────────
    "jose zepeda":       1830,
    "regis prograis":    1820,
    "jack catterall":    1800,
    "subriel matias":    1790,
    "jose pedraza":      1790,
    # ── Lightweight (135 lb) ─────────────────────────────────────────────────
    "vasiliy lomachenko": 1880,
    "gervonta davis":    1890,
    "devin haney":       1870,
    "george kambosos":   1830,
    "shakur stevenson":  1860,
    "frank martin":      1820,
    # ── Super Featherweight (130 lb) ──────────────────────────────────────────
    "o'shaquie foster":  1840,
    "robson conceicao":  1820,
    "lamont roach":      1800,
    "hector garcia":     1810,
    # ── Featherweight (126 lb) ────────────────────────────────────────────────
    "leo santa cruz":    1820,
    "brandon figueroa":  1810,
    "isaac dogboe":      1800,
    "naoya inoue":       1920,
    "rey vargas":        1820,
    "mark magsayo":      1800,
    # ── Super Bantamweight (122 lb) ───────────────────────────────────────────
    "murodjon akhmadaliev": 1850,
    "marlon tapales":    1820,
    "roman gonzalez":    1830,
    "sor rungvisai":     1810,
    # ── Bantamweight (118 lb) ─────────────────────────────────────────────────
    "john riel casimero": 1810,
    "nonito donaire":    1820,
    "Emmanuel Rodriguez": 1800,
    # ── Super Flyweight (115 lb) ─────────────────────────────────────────────
    "juan francisco estrada": 1860,
    "srisaket sor rungvisai": 1840,
    "julio cesar martinez": 1820,
    "elwin soto":        1800,
    # ── Flyweight (112 lb) ───────────────────────────────────────────────────
    "julio cesar martinez flyweight": 1820,
    "sunny edwards":     1840,
    "moruti mthalane":   1800,
    # ── Strawweight (105 lb) ─────────────────────────────────────────────────
    "wanheng menayothin": 1850,
    "panya pradabsri":   1820,
    "knockout cp freshmart": 1800,
}


def _load_boxing_seeds() -> dict:
    try:
        return json.loads(PLAYERS_FILE.read_text()).get("boxing", {})
    except Exception:
        return {}


class BoxingPredictor(AbstractSport):

    sport_name = "boxing"
    sport_keywords = [
        "boxing", "fight", "bout", "wbc", "wba", "ibf", "wbo",
        "heavyweight boxing", "middleweight boxing", "welterweight boxing",
        "fury", "usyk", "joshua", "canelo", "crawford", "wilder",
        "ko", "knockout", "title fight", "world champion",
    ]

    def predict(self, entity1: str, entity2: str, date: str, context: dict) -> PredictionResult:
        sources = ["Seeded boxing ELO ratings"]

        seeds = _load_boxing_seeds()

        # 1. ELO
        elo_predictor = elo_module.EloPredictor(default_elo=1650)
        for name, elo in BOXING_ELO.items():
            elo_predictor.set(name, elo)
        for name, info in seeds.items():
            if "elo" in info:
                elo_predictor.set(name, info["elo"])

        f1_elo = elo_predictor.get(entity1.lower())
        f2_elo = elo_predictor.get(entity2.lower())
        elo_p1 = elo_module.win_probability(f1_elo, f2_elo)

        # 2. Style matchup heuristics (southpaw vs orthodox, reach, experience)
        style_adj = context.get("style_adjustment", 0.0)
        blended_p1 = max(0.05, min(0.95, elo_p1 + style_adj))

        # 3. News (injury, contract disputes, weight miss)
        f1_news = news.get_sentiment(entity1 + " boxing")
        f2_news = news.get_sentiment(entity2 + " boxing")
        all_flags = f1_news.get("flags", []) + f2_news.get("flags", [])

        sentiment_adj = (f1_news.get("score", 0) - f2_news.get("score", 0)) * 0.10
        blended_p1 = max(0.05, min(0.95, blended_p1 + sentiment_adj))

        # 4. ML ensemble (if trained)
        ml_model = ml_ensemble.MLEnsemble(sport="boxing", n_classes=2)
        if ml_model.load():
            elo_diff = (float(f1_elo) - float(f2_elo)) / 400.0
            ml_feat = np.array([elo_diff, float(f1_elo) / float(f2_elo),
                                0.0, 0.0, 0.0, elo_p1], dtype=np.float32)
            ml_dict = ml_model.predict_dict(ml_feat, ["p2_win", "p1_win"])
            blended_p1 = 0.70 * blended_p1 + 0.30 * ml_dict["p1_win"]
            sources.append("ML Ensemble (boxing)")

        blended_p1 = max(0.05, min(0.95, blended_p1))
        probs = {"p1_win": round(blended_p1, 4), "p2_win": round(1 - blended_p1, 4)}

        # 5. Market
        mkt = market.get_market_odds(entity1, entity2, "boxing")
        market_mapped = None
        if mkt:
            market_mapped = {"p1_win": mkt.get("home_win"), "p2_win": mkt.get("away_win")}

        # 6. Edge + Kelly
        edges = edge_mod.calculate_edge(probs, market_mapped or {})
        best_outcome, best_info = edge_mod.best_bet(edges)
        best_edge = best_info["edge_pct"] if best_info else 0.0
        kelly_pct = 0.0
        if best_info and best_info["edge"] > 0:
            decimal = odds_mod.prob_to_decimal(best_info["market_prob"])
            kelly_pct = kelly_mod.kelly_fraction(best_info["model_prob"], decimal)

        factors = [
            f"ELO: {entity1} ({f1_elo:.0f}) vs {entity2} ({f2_elo:.0f})",
            f"Model probability: {entity1} {blended_p1*100:.1f}%",
        ]

        return PredictionResult(
            sport="Boxing",
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
            model_breakdown={"ELO": {"p1_win": elo_p1}},
            competition=context.get("competition", "Boxing"),
        )
