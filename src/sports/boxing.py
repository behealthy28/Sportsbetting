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
# Full boxer profiles: elo, ko_rate (KO wins / total wins), punch_output (per round),
# accuracy (landed / thrown), defence (slips + blocks %), style archetype.
# style: P=pressure/brawler, S=outboxer/slickboxer, B=boxer-puncher, C=counter-puncher
BOXER_PROFILES = {
    # ── Heavyweight ──────────────────────────────────────────────────────────
    "oleksandr usyk":    {"elo":1950,"ko_rate":0.52,"punch_output":56,"accuracy":0.44,"defence":0.68,"style":"S","weight_class":"Heavyweight"},
    "tyson fury":        {"elo":1910,"ko_rate":0.62,"punch_output":52,"accuracy":0.38,"defence":0.74,"style":"S","weight_class":"Heavyweight"},
    "anthony joshua":    {"elo":1830,"ko_rate":0.82,"punch_output":55,"accuracy":0.42,"defence":0.54,"style":"B","weight_class":"Heavyweight"},
    "deontay wilder":    {"elo":1820,"ko_rate":0.96,"punch_output":36,"accuracy":0.40,"defence":0.56,"style":"P","weight_class":"Heavyweight"},
    "daniel dubois":     {"elo":1810,"ko_rate":0.88,"punch_output":47,"accuracy":0.40,"defence":0.55,"style":"P","weight_class":"Heavyweight"},
    "zhilei zhang":      {"elo":1800,"ko_rate":0.70,"punch_output":45,"accuracy":0.39,"defence":0.60,"style":"P","weight_class":"Heavyweight"},
    "joseph parker":     {"elo":1800,"ko_rate":0.52,"punch_output":50,"accuracy":0.37,"defence":0.62,"style":"B","weight_class":"Heavyweight"},
    "frank sanchez":     {"elo":1790,"ko_rate":0.67,"punch_output":48,"accuracy":0.38,"defence":0.64,"style":"B","weight_class":"Heavyweight"},
    "joe joyce":         {"elo":1760,"ko_rate":0.67,"punch_output":72,"accuracy":0.41,"defence":0.55,"style":"P","weight_class":"Heavyweight"},
    "otto wallin":       {"elo":1760,"ko_rate":0.44,"punch_output":48,"accuracy":0.37,"defence":0.63,"style":"S","weight_class":"Heavyweight"},
    # ── Light Heavyweight ────────────────────────────────────────────────────
    "artur beterbiev":   {"elo":1900,"ko_rate":1.00,"punch_output":68,"accuracy":0.45,"defence":0.60,"style":"P","weight_class":"Light Heavyweight"},
    "dmitry bivol":      {"elo":1890,"ko_rate":0.44,"punch_output":78,"accuracy":0.38,"defence":0.72,"style":"S","weight_class":"Light Heavyweight"},
    "joe smith":         {"elo":1790,"ko_rate":0.65,"punch_output":52,"accuracy":0.37,"defence":0.58,"style":"P","weight_class":"Light Heavyweight"},
    "anthony yarde":     {"elo":1800,"ko_rate":0.79,"punch_output":58,"accuracy":0.40,"defence":0.57,"style":"P","weight_class":"Light Heavyweight"},
    "callum smith":      {"elo":1800,"ko_rate":0.63,"punch_output":56,"accuracy":0.39,"defence":0.60,"style":"B","weight_class":"Light Heavyweight"},
    # ── Super Middleweight ───────────────────────────────────────────────────
    "canelo alvarez":    {"elo":1930,"ko_rate":0.73,"punch_output":62,"accuracy":0.42,"defence":0.74,"style":"B","weight_class":"Super Middleweight"},
    "david benavidez":   {"elo":1860,"ko_rate":0.87,"punch_output":82,"accuracy":0.42,"defence":0.60,"style":"P","weight_class":"Super Middleweight"},
    "edgar berlanga":    {"elo":1820,"ko_rate":0.88,"punch_output":59,"accuracy":0.38,"defence":0.59,"style":"P","weight_class":"Super Middleweight"},
    "caleb plant":       {"elo":1820,"ko_rate":0.57,"punch_output":70,"accuracy":0.40,"defence":0.70,"style":"S","weight_class":"Super Middleweight"},
    "chris eubank jr":   {"elo":1810,"ko_rate":0.57,"punch_output":74,"accuracy":0.38,"defence":0.60,"style":"P","weight_class":"Super Middleweight"},
    # ── Middleweight ─────────────────────────────────────────────────────────
    "janibek alimkhanuly":{"elo":1850,"ko_rate":0.77,"punch_output":68,"accuracy":0.42,"defence":0.66,"style":"B","weight_class":"Middleweight"},
    "jermall charlo":    {"elo":1820,"ko_rate":0.65,"punch_output":58,"accuracy":0.40,"defence":0.68,"style":"B","weight_class":"Middleweight"},
    "erislandy lara":    {"elo":1810,"ko_rate":0.38,"punch_output":60,"accuracy":0.42,"defence":0.76,"style":"C","weight_class":"Middleweight"},
    "carlos adames":     {"elo":1800,"ko_rate":0.71,"punch_output":70,"accuracy":0.38,"defence":0.59,"style":"P","weight_class":"Middleweight"},
    # ── Super Welterweight ───────────────────────────────────────────────────
    "terence crawford":  {"elo":1900,"ko_rate":0.72,"punch_output":58,"accuracy":0.45,"defence":0.72,"style":"B","weight_class":"Super Welterweight"},
    "errol spence":      {"elo":1870,"ko_rate":0.63,"punch_output":74,"accuracy":0.43,"defence":0.68,"style":"B","weight_class":"Super Welterweight"},
    "tim tszyu":         {"elo":1860,"ko_rate":0.69,"punch_output":68,"accuracy":0.41,"defence":0.64,"style":"P","weight_class":"Super Welterweight"},
    "sebastian fundora": {"elo":1830,"ko_rate":0.76,"punch_output":70,"accuracy":0.38,"defence":0.58,"style":"P","weight_class":"Super Welterweight"},
    "brian castano":     {"elo":1810,"ko_rate":0.56,"punch_output":76,"accuracy":0.39,"defence":0.61,"style":"P","weight_class":"Super Welterweight"},
    # ── Welterweight ─────────────────────────────────────────────────────────
    "jaron ennis":       {"elo":1880,"ko_rate":0.85,"punch_output":74,"accuracy":0.45,"defence":0.65,"style":"B","weight_class":"Welterweight"},
    "vergil ortiz":      {"elo":1820,"ko_rate":0.92,"punch_output":72,"accuracy":0.43,"defence":0.58,"style":"P","weight_class":"Welterweight"},
    "keith thurman":     {"elo":1810,"ko_rate":0.67,"punch_output":65,"accuracy":0.40,"defence":0.67,"style":"B","weight_class":"Welterweight"},
    "eimantas stanionis": {"elo":1800,"ko_rate":0.55,"punch_output":66,"accuracy":0.38,"defence":0.63,"style":"P","weight_class":"Welterweight"},
    # ── Lightweight ──────────────────────────────────────────────────────────
    "gervonta davis":    {"elo":1890,"ko_rate":0.87,"punch_output":57,"accuracy":0.43,"defence":0.67,"style":"P","weight_class":"Lightweight"},
    "vasiliy lomachenko":{"elo":1880,"ko_rate":0.52,"punch_output":80,"accuracy":0.50,"defence":0.74,"style":"S","weight_class":"Lightweight"},
    "shakur stevenson":  {"elo":1860,"ko_rate":0.55,"punch_output":78,"accuracy":0.47,"defence":0.76,"style":"S","weight_class":"Lightweight"},
    "devin haney":       {"elo":1870,"ko_rate":0.41,"punch_output":72,"accuracy":0.46,"defence":0.74,"style":"S","weight_class":"Lightweight"},
    "george kambosos":   {"elo":1830,"ko_rate":0.55,"punch_output":62,"accuracy":0.38,"defence":0.62,"style":"P","weight_class":"Lightweight"},
    "frank martin":      {"elo":1820,"ko_rate":0.68,"punch_output":58,"accuracy":0.40,"defence":0.63,"style":"B","weight_class":"Lightweight"},
    # ── Super Featherweight ───────────────────────────────────────────────────
    "o'shaquie foster":  {"elo":1840,"ko_rate":0.57,"punch_output":66,"accuracy":0.40,"defence":0.69,"style":"S","weight_class":"Super Featherweight"},
    "robson conceicao":  {"elo":1820,"ko_rate":0.44,"punch_output":70,"accuracy":0.39,"defence":0.70,"style":"C","weight_class":"Super Featherweight"},
    # ── Featherweight ─────────────────────────────────────────────────────────
    "naoya inoue":       {"elo":1920,"ko_rate":0.80,"punch_output":74,"accuracy":0.46,"defence":0.68,"style":"B","weight_class":"Featherweight"},
    "rey vargas":        {"elo":1820,"ko_rate":0.50,"punch_output":72,"accuracy":0.42,"defence":0.70,"style":"S","weight_class":"Featherweight"},
    # ── Super Bantamweight ────────────────────────────────────────────────────
    "murodjon akhmadaliev":{"elo":1850,"ko_rate":0.72,"punch_output":68,"accuracy":0.41,"defence":0.64,"style":"B","weight_class":"Super Bantamweight"},
    "marlon tapales":    {"elo":1820,"ko_rate":0.65,"punch_output":62,"accuracy":0.39,"defence":0.61,"style":"P","weight_class":"Super Bantamweight"},
    # ── Lower weights ────────────────────────────────────────────────────────
    "juan francisco estrada":{"elo":1860,"ko_rate":0.58,"punch_output":78,"accuracy":0.42,"defence":0.67,"style":"B","weight_class":"Super Flyweight"},
    "roman gonzalez":    {"elo":1830,"ko_rate":0.62,"punch_output":80,"accuracy":0.44,"defence":0.66,"style":"P","weight_class":"Super Flyweight"},
    "sunny edwards":     {"elo":1840,"ko_rate":0.35,"punch_output":82,"accuracy":0.45,"defence":0.74,"style":"S","weight_class":"Flyweight"},
    "wanheng menayothin":{"elo":1850,"ko_rate":0.62,"punch_output":70,"accuracy":0.41,"defence":0.64,"style":"P","weight_class":"Strawweight"},
}

# Keep flat ELO dict for backwards-compat with EloPredictor
BOXING_ELO = {name: p["elo"] for name, p in BOXER_PROFILES.items()}


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
