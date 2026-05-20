"""
Train ML ensembles and fit Dixon-Coles on real historical match data.

Data sources used automatically (no API keys):
  Football : openfootball GitHub JSON (14 seasons, 10 leagues, ~25k matches)
             StatsBomb open-data xG index (2400+ matches with shot-level xG)
  Tennis   : Jeff Sackmann ATP/WTA CSVs (2015-present)
  UFC      : Kaggle mdabbert/ultimate-ufc-dataset (if Kaggle credentials configured)

Usage:
  python main.py train              # all sports
  python main.py train football
  python main.py train tennis
  python main.py train ufc

After training, models are saved to data/models/ and predictions automatically
use the trained models. Run `python main.py backtest` to see accuracy metrics.
"""
import sys
import json
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent / "data" / "models"
BACKTEST_DIR = Path(__file__).parent.parent.parent / "data" / "backtest"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

MIN_TEAM_GAMES = 5
MIN_PLAYER_GAMES = 10

# ── Rolling stats helpers ─────────────────────────────────────────────────────

def _rolling_team_stats(team_history: list, n: int = 10) -> dict:
    """Compute rolling stats from a team's last N match entries."""
    if not team_history:
        return {}
    recent = team_history[-n:]
    n_g = len(recent)
    # xg fallback: use goals when xg not available
    xgf_vals = [m.get("xgf") if m.get("xgf") is not None else m["gf"] for m in recent]
    xga_vals = [m.get("xga") if m.get("xga") is not None else m["ga"] for m in recent]
    return {
        "form":            sum(m["win"] for m in recent) / n_g,
        "avg_goals":       sum(m["gf"] for m in recent) / n_g,
        "avg_conceded":    sum(m["ga"] for m in recent) / n_g,
        "avg_xg":          sum(xgf_vals) / n_g,
        "avg_xga":         sum(xga_vals) / n_g,
        "clean_sheet_rate":sum(1 for m in recent if m["ga"] == 0) / n_g,
        "elo":             team_history[-1].get("elo_after", 1500),
        "ranking":         50,
    }


def _rolling_player_stats(player_history: list, surface: str, n: int = 20) -> dict:
    """Compute rolling tennis player stats on a given surface."""
    if not player_history:
        return {}
    recent_all = player_history[-n:]
    n_all = len(recent_all)
    overall_wr = sum(m["won"] for m in recent_all) / n_all

    surf_lower = surface.lower()
    surf_matches = [m for m in player_history if m.get("surface", "").lower() == surf_lower]
    recent_surf = surf_matches[-n:] if surf_matches else []
    n_surf = len(recent_surf)

    if n_surf > 0:
        surf_wr   = sum(m["won"] for m in recent_surf) / n_surf
        fs_pct    = sum(m.get("first_serve_in_pct", 0.62) for m in recent_surf) / n_surf
        bp_save   = sum(m.get("bp_save_rate", 0.62) for m in recent_surf) / n_surf
        ace_rate  = sum(m.get("ace_rate", 0.05) for m in recent_surf) / n_surf
    else:
        surf_wr  = overall_wr
        fs_pct   = sum(m.get("first_serve_in_pct", 0.62) for m in recent_all) / max(n_all, 1)
        bp_save  = sum(m.get("bp_save_rate", 0.62) for m in recent_all) / max(n_all, 1)
        ace_rate = sum(m.get("ace_rate", 0.05) for m in recent_all) / max(n_all, 1)

    return {
        "overall_win_rate": round(overall_wr, 4),
        "elo":              player_history[-1].get("elo_after", 1500),
        "recent_rank":      player_history[-1].get("rank", 100),
        "ranking_trend":    0.0,
        "surface_stats": {
            surf_lower: {
                "win_rate":        round(surf_wr, 4),
                "first_serve_pct": round(fs_pct, 4),
                "bp_save_rate":    round(bp_save, 4),
                "ace_rate":        round(ace_rate, 4),
            }
        },
    }


def _calibration_data(probs: np.ndarray, one_hot: np.ndarray, n_bins: int = 10) -> list:
    """Return calibration curve as list of {pred, actual, count} dicts."""
    bins = []
    edges = np.linspace(0, 1, n_bins + 1)
    flat_p = probs.flatten()
    flat_a = one_hot.flatten()
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (flat_p >= lo) & (flat_p < hi)
        count = int(mask.sum())
        if count > 0:
            bins.append({
                "pred":   round(float(flat_p[mask].mean()), 3),
                "actual": round(float(flat_a[mask].mean()), 3),
                "count":  count,
            })
        else:
            bins.append({"pred": round((lo + hi) / 2, 3), "actual": 0.0, "count": 0})
    return bins


# ── Football training ─────────────────────────────────────────────────────────

def train_football(verbose: bool = True) -> dict:
    """
    Fetch openfootball match history (25k+ matches across 10 leagues, 2010-present),
    optionally enrich with StatsBomb xG, fit Dixon-Coles via MLE, train ML ensemble,
    evaluate on held-out 20% test set, and save all artefacts.
    """
    from src.data.scrapers.historical import fetch_football_history, get_data_summary
    from src.models.dixon_coles import DixonColesModel
    from src.models.ml_ensemble import MLEnsemble, build_football_features
    from src.models.elo import EloPredictor, update_elo

    if verbose:
        summary = get_data_summary()
        cached_files = summary["football_files"]
        print(f"\n[train:football] Data status: {cached_files} cached league-season files, "
              f"{summary['statsbomb_xg_matches']} StatsBomb xG matches")
        print("[train:football] Downloading match history (openfootball + StatsBomb)…")

    matches = fetch_football_history(
        min_season_year=2010,
        include_xg=True,
        verbose=verbose,
    )
    # Also fold in Kaggle Soccer DB if available
    from src.data.scrapers.historical import fetch_kaggle_soccer_db, _kaggle_available
    if _kaggle_available():
        kaggle_ms = fetch_kaggle_soccer_db(verbose=verbose)
        if kaggle_ms:
            matches.extend(kaggle_ms)
            if verbose:
                print(f"[train:football] Added {len(kaggle_ms)} Kaggle Soccer DB matches.")

    if not matches:
        print("[train:football] ERROR: No match data fetched. Check network connectivity.")
        return {}

    # Sort chronologically
    def _dt(m):
        try:
            return datetime.strptime(m["date"][:10], "%Y-%m-%d")
        except Exception:
            return datetime(2000, 1, 1)

    matches.sort(key=_dt)

    if verbose:
        xg_count = sum(1 for m in matches if m.get("home_xg") is not None)
        print(f"[train:football] {len(matches):,} total matches loaded  |  "
              f"{xg_count:,} with xG  |  "
              f"date range: {matches[0]['date'][:10]} → {matches[-1]['date'][:10]}")

    split_idx = int(len(matches) * 0.8)
    train_matches = matches[:split_idx]

    # ── Fit Dixon-Coles MLE on training set ──────────────────────────────────
    if verbose:
        print(f"[train:football] Fitting Dixon-Coles MLE on {len(train_matches):,} matches…")

    ref_dt = _dt(train_matches[-1])
    dc_data = []
    for m in train_matches:
        days_ago = (ref_dt - _dt(m)).days
        weight = float(np.exp(-0.0005 * days_ago))  # half-life ~1386 days (~4 seasons)
        dc_data.append({
            "home_team":  m["home_team"].lower(),
            "away_team":  m["away_team"].lower(),
            "home_goals": m["home_goals"],
            "away_goals": m["away_goals"],
            "weight":     weight,
        })

    dc_model = DixonColesModel()
    dc_model.fit(dc_data)

    dc_params = {
        "attack":       dc_model.attack,
        "defense":      dc_model.defense,
        "rho":          dc_model.rho,
        "trained_at":   datetime.now().isoformat(),
        "n_matches":    len(train_matches),
        "n_teams":      len(dc_model.attack),
    }
    (MODELS_DIR / "football_dc.json").write_text(json.dumps(dc_params, indent=2))

    if verbose:
        print(f"[train:football] Dixon-Coles fitted: rho={dc_model.rho:.4f}, "
              f"{len(dc_model.attack)} teams. Saved to data/models/football_dc.json")

    # ── Build ML feature vectors chronologically ─────────────────────────────
    if verbose:
        print("[train:football] Building rolling feature vectors (no lookahead)…")

    elo_pred = EloPredictor(default_elo=1500)
    team_hist = defaultdict(list)
    h2h_hist  = defaultdict(list)

    X_train, y_train = [], []
    X_test,  y_test  = [], []
    test_info = []

    for i, m in enumerate(matches):
        ht = m["home_team"]
        at = m["away_team"]
        hg = m["home_goals"]
        ag = m["away_goals"]
        h_xg = m.get("home_xg")  # may be None — handled in rolling stats
        a_xg = m.get("away_xg")
        is_train = (i < split_idx)

        if (len(team_hist[ht]) >= MIN_TEAM_GAMES and
                len(team_hist[at]) >= MIN_TEAM_GAMES):
            hs = _rolling_team_stats(team_hist[ht])
            hs["elo"] = elo_pred.get(ht.lower())
            as_ = _rolling_team_stats(team_hist[at])
            as_["elo"] = elo_pred.get(at.lower())

            h2h_key  = tuple(sorted([ht, at]))
            h2h_games = h2h_hist[h2h_key]
            h2h_wr   = (sum(1 for r in h2h_games if r == ht) / len(h2h_games)
                        if h2h_games else 0.33)

            ctx = {
                "h2h_home_win_rate":  h2h_wr,
                "competition_weight": 0.85,
                "is_neutral":         0,
            }
            feat  = build_football_features(hs, as_, ctx)
            label = 2 if hg > ag else (1 if hg == ag else 0)

            if is_train:
                X_train.append(feat)
                y_train.append(label)
            else:
                X_test.append(feat)
                y_test.append(label)
                test_info.append({
                    "home": ht, "away": at,
                    "home_goals": hg, "away_goals": ag,
                    "date": m.get("date", ""),
                })

        # Update state AFTER feature extraction — no lookahead
        elo_a = elo_pred.get(ht.lower())
        elo_b = elo_pred.get(at.lower())
        res   = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        na, nb = update_elo(elo_a, elo_b, res, k=20.0)
        elo_pred.set(ht.lower(), na)
        elo_pred.set(at.lower(), nb)

        team_hist[ht].append({
            "gf": hg, "ga": ag,
            "xgf": h_xg, "xga": a_xg,
            "win": 1 if hg > ag else 0,
            "elo_after": na,
        })
        team_hist[at].append({
            "gf": ag, "ga": hg,
            "xgf": a_xg, "xga": h_xg,
            "win": 1 if ag > hg else 0,
            "elo_after": nb,
        })
        h2h_hist[tuple(sorted([ht, at]))].append(
            ht if hg > ag else (at if ag > hg else None)
        )

    if not X_train:
        print("[train:football] ERROR: Insufficient samples for ML training.")
        return {}

    X_tr = np.array(X_train, dtype=np.float32)
    y_tr = np.array(y_train, dtype=np.int32)

    if verbose:
        vc  = np.bincount(y_tr, minlength=3)
        pct = vc / len(y_tr) * 100
        print(f"[train:football] Training ML ensemble on {len(X_tr):,} samples | "
              f"away {pct[0]:.0f}%  draw {pct[1]:.0f}%  home {pct[2]:.0f}%")

    ml_model = MLEnsemble(sport="football", n_classes=3)
    ml_model.fit(X_tr, y_tr)

    if verbose:
        print("[train:football] RF+XGBoost ensemble trained. Saved to data/models/football_*.pkl")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    metrics = {
        "sport": "football",
        "n_train": len(X_tr),
        "n_test": 0,
        "ml_accuracy": None,
        "ml_brier": None,
        "dc_accuracy": None,
        "dc_brier": None,
        "naive_brier": 0.667,
        "calibration": [],
        "generated_at": datetime.now().isoformat(),
        "data_sources": ["openfootball/football.json", "StatsBomb open-data"],
    }

    if X_test:
        X_te = np.array(X_test, dtype=np.float32)
        y_te = np.array(y_test, dtype=np.int32)
        n_te = len(X_te)

        # ML predictions
        probs_arr = np.array([ml_model.predict_proba(x) for x in X_te])
        predicted = np.argmax(probs_arr, axis=1)
        accuracy  = float((predicted == y_te).mean())
        one_hot   = np.zeros((n_te, 3))
        for j, lbl in enumerate(y_te):
            one_hot[j, lbl] = 1.0
        brier     = float(np.mean(np.sum((probs_arr - one_hot) ** 2, axis=1)))

        # DC predictions on same test set
        dc_preds = []
        for info in test_info:
            p = dc_model.predict(info["home"].lower(), info["away"].lower())
            dc_preds.append([p["away_win"], p["draw"], p["home_win"]])
        dc_arr     = np.array(dc_preds)
        dc_pred_lbl= np.argmax(dc_arr, axis=1)
        dc_acc     = float((dc_pred_lbl == y_te).mean())
        dc_brier   = float(np.mean(np.sum((dc_arr - one_hot) ** 2, axis=1)))

        calibration = _calibration_data(probs_arr, one_hot)

        metrics.update({
            "n_test":       n_te,
            "ml_accuracy":  round(accuracy, 4),
            "ml_brier":     round(brier, 4),
            "dc_accuracy":  round(dc_acc, 4),
            "dc_brier":     round(dc_brier, 4),
            "calibration":  calibration,
        })

        if verbose:
            print(f"\n[train:football] ── Test set results ({n_te:,} matches) ──")
            print(f"  ML  accuracy={accuracy:.1%}  Brier={brier:.4f}  "
                  f"(naive baseline=0.667)")
            print(f"  DC  accuracy={dc_acc:.1%}   Brier={dc_brier:.4f}")
            bss_ml = round(1 - brier / 0.667, 4)
            bss_dc = round(1 - dc_brier / 0.667, 4)
            print(f"  Brier Skill: ML={bss_ml:+.3f}  DC={bss_dc:+.3f}  "
                  f"(+ve = better than random)")

    (BACKTEST_DIR / "football_results.json").write_text(json.dumps(metrics, indent=2))

    if verbose:
        print("[train:football] Results saved to data/backtest/football_results.json")

    return metrics


# ── Tennis training ───────────────────────────────────────────────────────────

def train_tennis(verbose: bool = True) -> dict:
    """
    Fetch Sackmann ATP + WTA CSVs (2010-present), build per-match rolling stats,
    train binary RF+XGBoost ensemble, evaluate on held-out 20%.

    Data: ~400k+ matches (ATP main + WTA main, 15+ years).
    Challengers can be included by passing tours=["atp","wta","atp_chall"] to
    fetch_tennis_history but they lack many serve stat columns.
    """
    from src.data.scrapers.historical import fetch_tennis_history
    from src.models.ml_ensemble import MLEnsemble, build_tennis_features
    from src.models.elo import EloPredictor, update_elo

    if verbose:
        print("\n[train:tennis] Downloading Jeff Sackmann ATP + WTA CSVs (2010-present)…")

    matches = fetch_tennis_history(tours=["atp", "wta"], verbose=verbose)
    if not matches:
        print("[train:tennis] ERROR: No data fetched.")
        return {}

    if verbose:
        atp_count = sum(1 for m in matches if m.get("tour") == "atp")
        wta_count = sum(1 for m in matches if m.get("tour") == "wta")
        print(f"[train:tennis] {len(matches):,} total matches loaded  |  "
              f"ATP={atp_count:,}  WTA={wta_count:,}")

    split_idx = int(len(matches) * 0.8)

    elo_pred   = EloPredictor(default_elo=1500)
    player_hist = defaultdict(list)
    rng = np.random.default_rng(42)

    X_train, y_train = [], []
    X_test,  y_test  = [], []

    for i, m in enumerate(matches):
        winner  = m["winner"]
        loser   = m["loser"]
        surface = m["surface"].lower()
        is_train = (i < split_idx)

        w_hist = player_hist[winner]
        l_hist = player_hist[loser]

        if len(w_hist) >= MIN_PLAYER_GAMES and len(l_hist) >= MIN_PLAYER_GAMES:
            # Randomly assign p1/p2 to eliminate winner-position bias in the label
            if rng.random() > 0.5:
                p1, p2, label = winner, loser, 1
            else:
                p1, p2, label = loser, winner, 0

            p1_stats = _rolling_player_stats(player_hist[p1], surface)
            p1_stats["elo"]         = elo_pred.get(p1.lower(), surface)
            p1_stats["recent_rank"] = player_hist[p1][-1].get("rank", 100)

            p2_stats = _rolling_player_stats(player_hist[p2], surface)
            p2_stats["elo"]         = elo_pred.get(p2.lower(), surface)
            p2_stats["recent_rank"] = player_hist[p2][-1].get("rank", 100)

            ctx  = {"surface": surface, "h2h_win_rate": 0.5, "tournament_importance": 0.85}
            feat = build_tennis_features(p1_stats, p2_stats, ctx)

            if is_train:
                X_train.append(feat)
                y_train.append(label)
            else:
                X_test.append(feat)
                y_test.append(label)

        # Update state AFTER feature extraction
        elo_w = elo_pred.get(winner.lower(), surface)
        elo_l = elo_pred.get(loser.lower(), surface)
        new_w, new_l = update_elo(elo_w, elo_l, 1.0, k=32.0)
        elo_pred.set(winner.lower(), new_w, surface)
        elo_pred.set(loser.lower(), new_l, surface)
        elo_pred.set(winner.lower(), new_w)
        elo_pred.set(loser.lower(), new_l)

        def _serve(row, prefix):
            svpt = float(row.get(f"{prefix}_svpt") or 1) or 1
            return {
                "first_serve_in_pct": float(row.get(f"{prefix}_1stIn") or 0) / svpt,
                "ace_rate":           float(row.get(f"{prefix}_ace") or 0) / svpt,
                "bp_save_rate":       float(row.get(f"{prefix}_bpSaved") or 0) /
                                      max(float(row.get(f"{prefix}_bpFaced") or 0), 1),
            }

        w_serve = _serve(m, "w")
        l_serve = _serve(m, "l")

        player_hist[winner].append({"won": 1, "surface": surface,
                                    "rank": m.get("winner_rank", 100),
                                    "elo_after": new_w, **w_serve})
        player_hist[loser].append({"won": 0, "surface": surface,
                                   "rank": m.get("loser_rank", 100),
                                   "elo_after": new_l, **l_serve})

    if not X_train:
        print("[train:tennis] Insufficient data for training.")
        return {}

    X_tr = np.array(X_train, dtype=np.float32)
    y_tr = np.array(y_train, dtype=np.int32)

    if verbose:
        vc = np.bincount(y_tr, minlength=2)
        print(f"[train:tennis] Training ML ensemble on {len(X_tr):,} samples | "
              f"p1_loss {vc[0]}  p1_win {vc[1]}")

    ml_model = MLEnsemble(sport="tennis", n_classes=2)
    ml_model.fit(X_tr, y_tr)

    if verbose:
        print("[train:tennis] RF+XGBoost trained. Saved to data/models/tennis_*.pkl")

    metrics = {
        "sport": "tennis",
        "n_train": len(X_tr),
        "n_test": 0,
        "ml_accuracy": None,
        "ml_brier": None,
        "naive_brier": 0.5,
        "calibration": [],
        "generated_at": datetime.now().isoformat(),
        "data_sources": ["Jeff Sackmann ATP/WTA CSVs"],
    }

    if X_test:
        X_te  = np.array(X_test, dtype=np.float32)
        y_te  = np.array(y_test, dtype=np.int32)
        n_te  = len(X_te)
        probs = np.array([ml_model.predict_proba(x) for x in X_te])
        preds = np.argmax(probs, axis=1)
        acc   = float((preds == y_te).mean())
        oh    = np.zeros((n_te, 2))
        for j, lbl in enumerate(y_te):
            oh[j, lbl] = 1.0
        brier = float(np.mean(np.sum((probs - oh) ** 2, axis=1)))
        cal   = _calibration_data(probs, oh)

        metrics.update({
            "n_test":      n_te,
            "ml_accuracy": round(acc, 4),
            "ml_brier":    round(brier, 4),
            "calibration": cal,
        })

        if verbose:
            bss = round(1 - brier / 0.5, 4)
            print(f"\n[train:tennis] ── Test set results ({n_te:,} matches) ──")
            print(f"  accuracy={acc:.1%}  Brier={brier:.4f}  "
                  f"Skill={bss:+.3f}  (naive baseline=0.500)")

    (BACKTEST_DIR / "tennis_results.json").write_text(json.dumps(metrics, indent=2))
    return metrics


# ── Boxing training (ELO-seed simulation) ────────────────────────────────────

def train_boxing(verbose: bool = True) -> dict:
    """
    Train boxing ML ensemble using synthetic fights generated from BOXING_ELO seeds.
    Creates ~8k simulated fights by pairing seed fighters with noise-injected outcomes.
    When real fight data becomes available the same pipeline applies directly.
    """
    from src.sports.boxing import BOXING_ELO
    from src.models.ml_ensemble import MLEnsemble
    from src.models.elo import win_probability

    if verbose:
        print("\n[train:boxing] Building synthetic fight dataset from seeded ELO ratings…")

    fighters = list(BOXING_ELO.items())
    rng = np.random.default_rng(7)

    X, y = [], []
    fighter_names = [f for f, elo in fighters if elo >= 1700]

    for _ in range(8000):
        i, j = rng.choice(len(fighter_names), size=2, replace=False)
        f1_name, f1_elo = fighter_names[i], BOXING_ELO[fighter_names[i]]
        f2_name, f2_elo = fighter_names[j], BOXING_ELO[fighter_names[j]]

        # True win probability from ELO
        true_p1 = win_probability(float(f1_elo), float(f2_elo))

        # Sample a noisy label from this probability
        label = 1 if rng.random() < true_p1 else 0

        # Features: elo diff, elo ratio, age proxy (add noise), style random
        elo_diff = (float(f1_elo) - float(f2_elo)) / 400.0
        elo_ratio = float(f1_elo) / float(f2_elo)
        age_diff = rng.normal(0, 4)          # synthetic: unknown, add noise
        reach_diff = rng.normal(0, 6)        # synthetic reach difference in cm
        # Experience proxy: higher elo = more experienced on average
        exp_diff = (float(f1_elo) - 1700) / 200.0 - (float(f2_elo) - 1700) / 200.0

        feat = [elo_diff, elo_ratio, age_diff / 10.0, reach_diff / 20.0, exp_diff, true_p1]
        X.append(feat)
        y.append(label)

    if len(X) < 100:
        if verbose:
            print("[train:boxing] Insufficient samples.")
        return {}

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)
    split = int(len(X) * 0.8)

    ml = MLEnsemble(sport="boxing", n_classes=2)
    ml.fit(X_arr[:split], y_arr[:split])

    metrics = {
        "sport": "boxing",
        "n_train": split,
        "n_test": len(X) - split,
        "data_sources": ["Synthetic from BOXING_ELO seeds"],
        "generated_at": datetime.now().isoformat(),
    }

    if len(X) > split:
        te_probs = np.array([ml.predict_proba(x) for x in X_arr[split:]])
        te_preds = np.argmax(te_probs, axis=1)
        acc = float((te_preds == y_arr[split:]).mean())
        oh = np.zeros((len(X) - split, 2))
        for j, lbl in enumerate(y_arr[split:]):
            oh[j, lbl] = 1.0
        brier = float(np.mean(np.sum((te_probs - oh) ** 2, axis=1)))
        metrics.update({
            "ml_accuracy": round(acc, 4),
            "ml_brier":    round(brier, 4),
            "naive_brier": 0.5,
        })
        if verbose:
            bss = round(1 - brier / 0.5, 4)
            print(f"[train:boxing] accuracy={acc:.1%}  Brier={brier:.4f}  Skill={bss:+.3f}")

    (BACKTEST_DIR / "boxing_results.json").write_text(json.dumps(metrics, indent=2))
    if metrics.get("ml_accuracy", 0) <= 0.52:
        # Synthetic-only model doesn't beat chance — don't deploy it
        for ext in ("_rf.pkl", "_xgb.pkl", "_scaler.pkl"):
            p = MODELS_DIR / f"boxing{ext}"
            if p.exists():
                p.unlink()
        if verbose:
            print("[train:boxing] Synthetic accuracy at chance — ML layer disabled. "
                  "Provide real fight data (Kaggle) for meaningful ML predictions.")
    return metrics


# ── Darts / Badminton / Table Tennis training (ELO-seed simulation) ───────────

def train_darts(verbose: bool = True) -> dict:
    """Train darts ML ensemble from synthetic matches based on PDC ELO seeds."""
    from src.sports.darts import DARTS_ELO
    from src.models.ml_ensemble import MLEnsemble
    from src.models.elo import win_probability

    if verbose:
        print("\n[train:darts] Building synthetic darts dataset from PDC ELO seeds…")

    players = [(n, e) for n, e in DARTS_ELO.items() if e >= 1700]
    # deduplicate by ELO value (nicknames share the same ELO)
    seen_elos = set()
    unique_players = []
    for name, elo in players:
        if elo not in seen_elos:
            seen_elos.add(elo)
            unique_players.append((name, elo))

    rng = np.random.default_rng(11)
    X, y = [], []

    for _ in range(5000):
        i, j = rng.choice(len(unique_players), size=2, replace=False)
        _, elo1 = unique_players[i]
        _, elo2 = unique_players[j]
        true_p1 = win_probability(float(elo1), float(elo2))
        label = 1 if rng.random() < true_p1 else 0
        elo_diff = (float(elo1) - float(elo2)) / 400.0
        feat = [elo_diff, float(elo1) / float(elo2), true_p1,
                rng.normal(0, 0.05), rng.normal(0, 0.03)]
        X.append(feat)
        y.append(label)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)
    split = int(len(X) * 0.8)

    ml = MLEnsemble(sport="darts", n_classes=2)
    ml.fit(X_arr[:split], y_arr[:split])

    metrics = {
        "sport": "darts",
        "n_train": split,
        "n_test": len(X) - split,
        "data_sources": ["Synthetic from PDC ELO seeds"],
        "generated_at": datetime.now().isoformat(),
    }
    if len(X) > split:
        probs = np.array([ml.predict_proba(x) for x in X_arr[split:]])
        acc = float((np.argmax(probs, 1) == y_arr[split:]).mean())
        metrics["ml_accuracy"] = round(acc, 4)
        if verbose:
            print(f"[train:darts] accuracy={acc:.1%}")

    (BACKTEST_DIR / "darts_results.json").write_text(json.dumps(metrics, indent=2))
    if metrics.get("ml_accuracy", 0) <= 0.52:
        for ext in ("_rf.pkl", "_xgb.pkl", "_scaler.pkl"):
            p = MODELS_DIR / f"darts{ext}"
            if p.exists():
                p.unlink()
        if verbose:
            print("[train:darts] Synthetic accuracy at chance — ML layer disabled.")
    return metrics


def train_badminton(verbose: bool = True) -> dict:
    """Train badminton ML ensemble from synthetic matches based on BWF ELO seeds."""
    from src.sports.darts import BADMINTON_ELO
    from src.models.ml_ensemble import MLEnsemble
    from src.models.elo import win_probability

    if verbose:
        print("\n[train:badminton] Building synthetic badminton dataset from BWF ELO seeds…")

    players = [(n, e) for n, e in BADMINTON_ELO.items() if e >= 1700]
    seen_elos, unique_players = set(), []
    for name, elo in players:
        if elo not in seen_elos:
            seen_elos.add(elo)
            unique_players.append((name, elo))

    rng = np.random.default_rng(13)
    X, y = [], []

    for _ in range(5000):
        i, j = rng.choice(len(unique_players), size=2, replace=False)
        _, elo1 = unique_players[i]
        _, elo2 = unique_players[j]
        true_p1 = win_probability(float(elo1), float(elo2))
        label = 1 if rng.random() < true_p1 else 0
        elo_diff = (float(elo1) - float(elo2)) / 400.0
        feat = [elo_diff, float(elo1) / float(elo2), true_p1,
                rng.normal(0, 0.05), rng.normal(0, 0.03)]
        X.append(feat)
        y.append(label)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)
    split = int(len(X) * 0.8)

    ml = MLEnsemble(sport="badminton", n_classes=2)
    ml.fit(X_arr[:split], y_arr[:split])

    metrics = {
        "sport": "badminton",
        "n_train": split,
        "n_test": len(X) - split,
        "data_sources": ["Synthetic from BWF ELO seeds"],
        "generated_at": datetime.now().isoformat(),
    }
    if len(X) > split:
        probs = np.array([ml.predict_proba(x) for x in X_arr[split:]])
        acc = float((np.argmax(probs, 1) == y_arr[split:]).mean())
        metrics["ml_accuracy"] = round(acc, 4)
        if verbose:
            print(f"[train:badminton] accuracy={acc:.1%}")

    (BACKTEST_DIR / "badminton_results.json").write_text(json.dumps(metrics, indent=2))
    if metrics.get("ml_accuracy", 0) <= 0.52:
        for ext in ("_rf.pkl", "_xgb.pkl", "_scaler.pkl"):
            p = MODELS_DIR / f"badminton{ext}"
            if p.exists():
                p.unlink()
        if verbose:
            print("[train:badminton] Synthetic accuracy at chance — ML layer disabled.")
    return metrics


# ── Cricket training (seed-based simulation) ─────────────────────────────────

def train_cricket(verbose: bool = True) -> dict:
    """
    Train cricket ML ensemble from synthetic matches based on TEAM_SEEDS.
    Simulates Test / ODI / T20 results using seeded win rates + ELO.
    """
    from src.data.scrapers.cricsheet import TEAM_SEEDS
    from src.models.ml_ensemble import MLEnsemble
    from src.models.elo import win_probability

    if verbose:
        print("\n[train:cricket] Building synthetic cricket dataset from TEAM_SEEDS…")

    teams = list(TEAM_SEEDS.items())
    rng = np.random.default_rng(17)
    X, y = [], []
    formats = ["odi", "t20", "test"]

    for _ in range(8000):
        i, j = rng.choice(len(teams), size=2, replace=False)
        t1_name, t1_data = teams[i]
        t2_name, t2_data = teams[j]
        fmt = formats[rng.integers(0, 3)]

        elo1 = float(t1_data.get("elo", 1700))
        elo2 = float(t2_data.get("elo", 1700))
        wr1  = float(t1_data.get(f"{fmt}_win_rate", 0.5))
        wr2  = float(t2_data.get(f"{fmt}_win_rate", 0.5))

        elo_p1   = win_probability(elo1, elo2)
        form_p1  = wr1 / (wr1 + wr2 + 1e-9)
        # Blend ELO 60% + format win rate 40%
        true_p1  = 0.6 * elo_p1 + 0.4 * form_p1
        # Home advantage: 5% bump for team1 (randomly assigned as home)
        if rng.random() < 0.5:
            true_p1 = min(0.95, true_p1 + 0.05)

        label = 1 if rng.random() < true_p1 else 0

        fmt_enc = [int(fmt == "odi"), int(fmt == "t20"), int(fmt == "test")]
        feat = [
            (elo1 - elo2) / 400.0,
            elo1 / (elo2 + 1e-9),
            wr1 - wr2,
            float(t1_data.get("batting_avg", 28)) - float(t2_data.get("batting_avg", 28)),
            float(t2_data.get("bowling_avg", 29)) - float(t1_data.get("bowling_avg", 29)),
            float(t1_data.get("run_rate", 5)) - float(t2_data.get("run_rate", 5)),
            *fmt_enc,
            true_p1,
        ]
        X.append(feat)
        y.append(label)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)
    split = int(len(X) * 0.8)

    ml = MLEnsemble(sport="cricket", n_classes=2)
    ml.fit(X_arr[:split], y_arr[:split])

    metrics = {
        "sport": "cricket",
        "n_train": split,
        "n_test": len(X) - split,
        "data_sources": ["Synthetic from TEAM_SEEDS"],
        "generated_at": datetime.now().isoformat(),
    }
    if len(X) > split:
        probs = np.array([ml.predict_proba(x) for x in X_arr[split:]])
        acc = float((np.argmax(probs, 1) == y_arr[split:]).mean())
        oh = np.zeros((len(X) - split, 2))
        for j, lbl in enumerate(y_arr[split:]):
            oh[j, lbl] = 1.0
        brier = float(np.mean(np.sum((probs - oh) ** 2, axis=1)))
        metrics.update({
            "ml_accuracy": round(acc, 4),
            "ml_brier":    round(brier, 4),
            "naive_brier": 0.5,
        })
        if verbose:
            bss = round(1 - brier / 0.5, 4)
            print(f"[train:cricket] accuracy={acc:.1%}  Brier={brier:.4f}  Skill={bss:+.3f}")

    (BACKTEST_DIR / "cricket_results.json").write_text(json.dumps(metrics, indent=2))
    if metrics.get("ml_accuracy", 0) <= 0.52:
        for ext in ("_rf.pkl", "_xgb.pkl", "_scaler.pkl"):
            p = MODELS_DIR / f"cricket{ext}"
            if p.exists():
                p.unlink()
        if verbose:
            print("[train:cricket] Synthetic accuracy at chance — ML layer disabled. "
                  "Provide Cricsheet ball-by-ball data for meaningful ML predictions.")
    return metrics


# ── UFC training (Kaggle only) ────────────────────────────────────────────────

def train_ufc(verbose: bool = True) -> dict:
    """Train UFC binary ML ensemble using Kaggle dataset (if credentials available)."""
    from src.data.scrapers.historical import fetch_kaggle_ufc, _kaggle_available
    from src.models.ml_ensemble import MLEnsemble, build_ufc_features

    if not _kaggle_available():
        if verbose:
            print("\n[train:ufc] Kaggle not configured. "
                  "Add ~/.kaggle/kaggle.json to enable UFC training.\n"
                  "  Dataset: kaggle datasets download -d mdabbert/ultimate-ufc-dataset")
        return {}

    if verbose:
        print("\n[train:ufc] Downloading Kaggle UFC dataset…")

    fights = fetch_kaggle_ufc(verbose=verbose)
    if not fights:
        if verbose:
            print("[train:ufc] No data available.")
        return {}

    rng = np.random.default_rng(0)
    X, y = [], []
    for fight in fights:
        r = fight.get("r_fighter", "")
        b = fight.get("b_fighter", "")
        winner = fight.get("winner", "")
        if not r or not b or not winner:
            continue

        r_stats = {
            "slpm": fight.get("r_slpm", 4.0), "str_acc": 0.45,
            "sapm": 3.5, "str_def": 0.55, "td_avg": fight.get("r_td_avg", 1.5),
            "td_acc": 0.4, "td_def": 0.7, "sub_avg": 0.3,
            "reach_cm": 183, "height_cm": 178, "age": 30,
            "win_rate": fight.get("r_win_rate", 0.6), "finish_rate": 0.5, "ko_wins": 5, "wins": 15,
        }
        b_stats = {
            "slpm": fight.get("b_slpm", 4.0), "str_acc": 0.45,
            "sapm": 3.5, "str_def": 0.55, "td_avg": fight.get("b_td_avg", 1.5),
            "td_acc": 0.4, "td_def": 0.7, "sub_avg": 0.3,
            "reach_cm": 183, "height_cm": 178, "age": 30,
            "win_rate": fight.get("b_win_rate", 0.6), "finish_rate": 0.5, "ko_wins": 5, "wins": 15,
        }

        if rng.random() > 0.5:
            feat  = build_ufc_features(r_stats, b_stats)
            label = 1 if "Red" in winner or r in winner else 0
        else:
            feat  = build_ufc_features(b_stats, r_stats)
            label = 1 if "Blue" in winner or b in winner else 0

        X.append(feat)
        y.append(label)

    if len(X) < 50:
        if verbose:
            print("[train:ufc] Insufficient samples.")
        return {}

    split = int(len(X) * 0.8)
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)

    ml = MLEnsemble(sport="ufc", n_classes=2)
    ml.fit(X_arr[:split], y_arr[:split])

    metrics = {"sport": "ufc", "n_train": split, "n_test": len(X) - split,
               "generated_at": datetime.now().isoformat()}

    if len(X) > split:
        te_probs = np.array([ml.predict_proba(x) for x in X_arr[split:]])
        te_preds = np.argmax(te_probs, axis=1)
        acc = float((te_preds == y_arr[split:]).mean())
        metrics["ml_accuracy"] = round(acc, 4)
        if verbose:
            print(f"[train:ufc] accuracy={acc:.1%} on {len(X)-split} test fights")

    (BACKTEST_DIR / "ufc_results.json").write_text(json.dumps(metrics, indent=2))
    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(sports: list = None) -> dict:
    default_sports = ["football", "tennis", "ufc", "boxing", "cricket", "darts", "badminton"]
    target = [s.lower() for s in (sports or default_sports)]
    results = {}

    for sport in target:
        if sport == "football":
            results["football"] = train_football(verbose=True)
        elif sport == "tennis":
            results["tennis"] = train_tennis(verbose=True)
        elif sport == "ufc" or sport == "mma":
            results["ufc"] = train_ufc(verbose=True)
        elif sport == "boxing":
            results["boxing"] = train_boxing(verbose=True)
        elif sport == "cricket":
            results["cricket"] = train_cricket(verbose=True)
        elif sport == "darts":
            results["darts"] = train_darts(verbose=True)
        elif sport == "badminton":
            results["badminton"] = train_badminton(verbose=True)
        else:
            print(f"[train] Unknown sport: {sport}")

    print("\n" + "=" * 60)
    print("[train] Complete.")
    for sport, m in results.items():
        if m and m.get("ml_accuracy") is not None:
            print(f"  {sport:12s}: accuracy={m['ml_accuracy']:.1%}  "
                  f"Brier={m.get('ml_brier', '?')}  "
                  f"n_test={m.get('n_test', '?'):,}")
    print()
    return results


if __name__ == "__main__":
    target = sys.argv[1:] if len(sys.argv) > 1 else None
    main(target)
