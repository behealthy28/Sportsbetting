"""Train the RF + XGBoost ensemble on the REAL historical dataset.

Walks matches in chronological order, maintaining rolling form, rolling
goal stats, a running Elo, and head-to-head records. Each training row's
features are computed ONLY from matches that happened before it — no leakage.
Labels: 0=away_win, 1=draw, 2=home_win (matches MLEnsemble's convention).

By default only the most recent RECENCY_YEARS of matches are used, so ratings
reflect the CURRENT team rather than ancient history. Elo is warm-started from
the seed table (elo_db) so the recent window is not a cold start.

Run:  python -m src.models.train   (after build_dataset)
"""
import json
import os
from collections import defaultdict, deque
from datetime import date

import numpy as np

from src.models.ml_ensemble import MLEnsemble, build_football_features, ML_AVAILABLE
from src.models import backtest
from src.data.momentum import streak_features
from src.data.scrapers.elo_db import NATIONAL_TEAM_ELO, get_national_elo

HIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "historical",
)
COMPETITION_WEIGHTS = {
    "world cup": 1.0, "copa america": 0.95, "euro": 0.95, "afcon": 0.90,
    "asian cup": 0.88, "nations league": 0.85, "world cup qualifier": 0.85,
    "international": 0.75, "friendly": 0.50,
}
LABEL = {"away_win": 0, "draw": 1, "home_win": 2}
MIN_HISTORY = 5         # games each team needs before we use a match
RECENCY_YEARS = 6       # only train/rate on the last N years (current squad)
ELO_K = 24
ELO_HA = 65


# Dataset spellings -> seed-table keys.
_ALIAS = {
    "united states": "usa", "czechia": "czech republic",
    "china pr": "china", "korea republic": "south korea",
    "bosnia and herzegovina": "bosnia", "cape verde islands": "cape verde",
    "côte d'ivoire": "ivory coast", "türkiye": "turkey",
}
DEFAULT_INTL = 1700.0   # neutral start for an international team with no seed


def _seed_elo(team: str) -> float:
    key = _ALIAS.get(team.lower(), team.lower())
    if key in NATIONAL_TEAM_ELO:
        return float(NATIONAL_TEAM_ELO[key])
    fuzzy = get_national_elo(key)
    return float(fuzzy) if fuzzy is not None else DEFAULT_INTL


def _expected(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def _elo_only_probs(eh, ea, neutral):
    """Simple Elo -> {away_win, draw, home_win} for a baseline comparison."""
    diff = eh + (0 if neutral else ELO_HA) - ea
    p_home_excl = 1.0 / (1.0 + 10 ** (-diff / 400.0))
    draw = max(0.18, min(0.30, 0.29 - abs(diff) / 5000.0))
    return {"away_win": (1 - p_home_excl) * (1 - draw),
            "draw": draw,
            "home_win": p_home_excl * (1 - draw)}


def _team_features(hist, elo, team):
    """Rolling features for a team from its recent history deque."""
    games = list(hist[team])
    n = len(games)
    if n == 0:
        return {"form": 0.5, "avg_goals": 1.35, "avg_conceded": 1.2,
                "clean_sheet_rate": 0.3, "elo": elo[team], "ranking": 50}
    wins = sum(g["w"] for g in games)
    gf = sum(g["gf"] for g in games)
    ga = sum(g["ga"] for g in games)
    cs = sum(1 for g in games if g["ga"] == 0)
    last5 = games[-5:]
    form = sum(g["w"] for g in last5) / len(last5)
    return {
        "form": round(form, 3),
        "avg_goals": round(gf / n, 3),
        "avg_conceded": round(ga / n, 3),
        "clean_sheet_rate": round(cs / n, 3),
        "elo": round(elo[team], 1),
        "ranking": 50,
    }


def build_training_matrix(matches):
    hist = defaultdict(lambda: deque(maxlen=15))
    elo = defaultdict(lambda: DEFAULT_INTL)
    # Warm-start every team that appears, from the seed table.
    for m in matches:
        for t in (m["home"], m["away"]):
            if t not in elo:
                elo[t] = _seed_elo(t)
    h2h = defaultdict(lambda: deque(maxlen=10))   # key: frozenset(pair) -> home win flags
    X, y = [], []
    eval_records = []

    for m in matches:
        home, away = m["home"], m["away"]
        neutral = bool(m.get("neutral"))   # real venue flag from the dataset
        comp_w = COMPETITION_WEIGHTS.get(m["competition"], 0.75)

        hd = _team_features(hist, elo, home)
        ad = _team_features(hist, elo, away)

        pair = frozenset((home, away))
        h2h_games = list(h2h[pair])
        h2h_rate = (sum(h2h_games) / len(h2h_games)) if h2h_games else 0.33

        hm = streak_features(list(hist[home]))
        am = streak_features(list(hist[away]))
        ctx = {
            "h2h_home_win_rate": h2h_rate,
            "home_key_players": 1.0, "away_key_players": 1.0,
            "home_days_rest": 7, "away_days_rest": 7,
            "competition_weight": comp_w,
            "is_neutral": int(neutral),
            "home_streak_norm": hm["streak_norm"], "away_streak_norm": am["streak_norm"],
            "home_bounceback": hm["bounceback"], "away_bounceback": am["bounceback"],
        }

        if len(hist[home]) >= MIN_HISTORY and len(hist[away]) >= MIN_HISTORY:
            feats = build_football_features(hd, ad, ctx)
            X.append(feats)
            y.append(LABEL[m["result"]])
            # Hold out a chronological tail for honest evaluation; keep Elo
            # inputs so we can compare ML vs an Elo-only baseline.
            eval_records.append({"feats": feats, "result": m["result"],
                                 "eh": elo[home], "ea": elo[away], "neutral": neutral})

        # ---- update state AFTER using the match (no leakage) ----
        hs, as_ = m["hs"], m["as"]
        hist[home].append({"w": 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0),
                           "gf": hs, "ga": as_})
        hist[away].append({"w": 1.0 if as_ > hs else (0.5 if hs == as_ else 0.0),
                           "gf": as_, "ga": hs})
        ea = _expected(elo[home] + (0 if neutral else ELO_HA), elo[away])
        sa = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        elo[home] += ELO_K * (sa - ea)
        elo[away] += ELO_K * ((1 - sa) - (1 - ea))
        h2h[pair].append(1 if hs > as_ else 0)

    return np.array(X, dtype=np.float32), np.array(y), eval_records, dict(elo)


def main(train_ml: bool = False):
    """Compute the recent Elo ratings (always) and optionally evaluate ML.

    The shipped model is Dixon-Coles + Elo: on international data Elo alone
    beats the XGBoost+RF ensemble, so ML training is off by default. Pass
    train_ml=True (or `--ml`) to fit/persist the ensemble for experimentation.
    """
    with open(os.path.join(HIST_DIR, "matches.json"), encoding="utf-8") as f:
        matches = json.load(f)
    matches.sort(key=lambda x: x["date"])

    cutoff = date.today().replace(year=date.today().year - RECENCY_YEARS).isoformat()
    recent = [m for m in matches if m["date"] >= cutoff]
    print(f"Loaded {len(matches)} matches; using {len(recent)} from last "
          f"{RECENCY_YEARS}y (since {cutoff})")

    X, y, eval_records, final_elo = build_training_matrix(recent)
    print(f"Training rows: {len(X)} (features={X.shape[1] if len(X) else 0})")

    split = int(len(X) * 0.85)
    labels = ["away_win", "draw", "home_win"]

    # Elo-only out-of-sample baseline — always available, no ML deps needed.
    base = [{"probs": _elo_only_probs(r["eh"], r["ea"], r["neutral"]),
             "outcome": r["result"]} for r in eval_records[split:]]
    brep = backtest.evaluate(base)
    print(f"\nOut-of-sample ({len(base)} matches):")
    print(f"  Elo-only    : Brier={brep['brier']} LogLoss={brep['log_loss']} ECE={brep['ece']}")

    if train_ml and ML_AVAILABLE and len(X):
        model = MLEnsemble(sport="football", n_classes=3)
        model.fit(X[:split], y[:split])
        X_eval = np.array([r["feats"] for r in eval_records[split:]], dtype=np.float32)
        Xs = model.scaler.transform(X_eval)
        proba = 0.55 * model.xgb_model.predict_proba(Xs) + 0.45 * model.rf.predict_proba(Xs)
        proba /= proba.sum(axis=1, keepdims=True)
        recs = [{"probs": {labels[i]: float(p[i]) for i in range(3)}, "outcome": r["result"]}
                for r, p in zip(eval_records[split:], proba)]
        report = backtest.evaluate(recs)
        d_brier = brep["brier"] - report["brier"]
        verdict = ("ML helps" if d_brier > 0.002 else
                   "ML ~ Elo (marginal)" if abs(d_brier) <= 0.002 else "Elo alone is better")
        print(f"  ML ensemble : Brier={report['brier']} LogLoss={report['log_loss']} ECE={report['ece']}")
        print(f"  Verdict     : {verdict} (Brier delta {d_brier:+.4f})")
    elif train_ml and not ML_AVAILABLE:
        print("  (ML requested but xgboost/sklearn unavailable — skipped)")

    # Persist final running Elo as a real, data-derived rating table.
    elo_sorted = dict(sorted(final_elo.items(), key=lambda kv: -kv[1]))
    with open(os.path.join(HIST_DIR, "elo_ratings.json"), "w", encoding="utf-8") as f:
        json.dump({k: round(v, 1) for k, v in elo_sorted.items()}, f, indent=1)
    print(f"Saved {len(elo_sorted)} data-derived Elo ratings to data/historical/elo_ratings.json")


if __name__ == "__main__":
    import sys
    main(train_ml="--ml" in sys.argv)
