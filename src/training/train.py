"""
Train ML ensembles and fit Dixon-Coles on real historical data.

Usage:
  python -m src.training.train              # all sports
  python -m src.training.train football
  python -m src.training.train tennis
  python train.py football                  # if invoked via main.py dispatch
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

MIN_TEAM_GAMES = 5     # minimum matches before including in training set
MIN_PLAYER_GAMES = 10  # minimum matches for tennis player


# ── Rolling stats helpers ────────────────────────────────────────────────────

def _rolling_team_stats(team_history: list, n: int = 10) -> dict:
    """Compute rolling stats from team's last N entries."""
    if not team_history:
        return {}
    recent = team_history[-n:]
    n_g = len(recent)
    return {
        "form": sum(m["win"] for m in recent) / n_g,
        "avg_goals": sum(m["gf"] for m in recent) / n_g,
        "avg_conceded": sum(m["ga"] for m in recent) / n_g,
        "avg_xg": sum(m.get("xgf", m["gf"]) for m in recent) / n_g,
        "avg_xga": sum(m.get("xga", m["ga"]) for m in recent) / n_g,
        "clean_sheet_rate": sum(1 for m in recent if m["ga"] == 0) / n_g,
        "elo": team_history[-1].get("elo_after", 1500),
        "ranking": 50,
    }


def _rolling_player_stats(player_history: list, surface: str, n: int = 20) -> dict:
    """Compute rolling stats for a tennis player on a given surface."""
    if not player_history:
        return {}

    # Overall (all surfaces)
    recent_all = player_history[-n:]
    n_all = len(recent_all)
    overall_wr = sum(m["won"] for m in recent_all) / n_all if n_all else 0.5

    # Surface-specific
    surf_lower = surface.lower()
    surf_matches = [m for m in player_history if m.get("surface", "").lower() == surf_lower]
    recent_surf = surf_matches[-n:] if surf_matches else []
    n_surf = len(recent_surf)

    if n_surf > 0:
        surf_wr = sum(m["won"] for m in recent_surf) / n_surf
        # Serve stats from surface matches
        w_1st_pct = sum(m.get("first_serve_in_pct", 0.62) for m in recent_surf) / n_surf
        bp_save = sum(m.get("bp_save_rate", 0.62) for m in recent_surf) / n_surf
        ace_rate = sum(m.get("ace_rate", 0.05) for m in recent_surf) / n_surf
    else:
        surf_wr = overall_wr
        w_1st_pct = sum(m.get("first_serve_in_pct", 0.62) for m in recent_all) / max(n_all, 1)
        bp_save = sum(m.get("bp_save_rate", 0.62) for m in recent_all) / max(n_all, 1)
        ace_rate = sum(m.get("ace_rate", 0.05) for m in recent_all) / max(n_all, 1)

    return {
        "overall_win_rate": round(overall_wr, 4),
        "elo": player_history[-1].get("elo_after", 1500),
        "recent_rank": player_history[-1].get("rank", 100),
        "ranking_trend": 0.0,  # simplified
        "surface_stats": {
            surf_lower: {
                "win_rate": round(surf_wr, 4),
                "first_serve_pct": round(w_1st_pct, 4),
                "bp_save_rate": round(bp_save, 4),
                "ace_rate": round(ace_rate, 4),
            }
        },
    }


# ── Football training ─────────────────────────────────────────────────────────

def train_football(verbose: bool = True) -> dict:
    """
    1. Fetch Understat match history (EPL/La Liga/Bundesliga/Serie A/Ligue 1, 2014–now)
    2. Build rolling feature vectors, split 80/20 by match date
    3. Fit DixonColesModel via MLE on training matches
    4. Train RF+XGBoost MLEnsemble on training feature vectors
    5. Evaluate on held-out 20% and save backtest results
    Returns dict of evaluation metrics.
    """
    from src.data.scrapers.historical import fetch_football_history
    from src.models.dixon_coles import DixonColesModel
    from src.models.ml_ensemble import MLEnsemble, build_football_features
    from src.models.elo import EloPredictor, update_elo

    if verbose:
        print("[train:football] Fetching historical match data from Understat...")

    matches = fetch_football_history(verbose=verbose)
    if not matches:
        print("[train:football] No data fetched. Check network connectivity.")
        return {}

    if verbose:
        print(f"[train:football] {len(matches)} matches loaded across all leagues/seasons.")

    split_idx = int(len(matches) * 0.8)
    train_matches_raw = matches[:split_idx]

    # ── Fit Dixon-Coles on training set with recency weighting ───────────────
    if verbose:
        print(f"[train:football] Fitting Dixon-Coles MLE on {len(train_matches_raw)} matches...")

    def _parse_dt(m):
        try:
            return datetime.strptime(m["date"][:10], "%Y-%m-%d")
        except Exception:
            return datetime(2000, 1, 1)

    ref_dt = _parse_dt(train_matches_raw[-1]) if train_matches_raw else datetime.now()

    dc_data = []
    for m in train_matches_raw:
        days_ago = (ref_dt - _parse_dt(m)).days
        weight = float(np.exp(-0.0008 * days_ago))  # half-life ~867 days
        dc_data.append({
            "home_team": m["home_team"].lower(),
            "away_team": m["away_team"].lower(),
            "home_goals": m["home_goals"],
            "away_goals": m["away_goals"],
            "weight": weight,
        })

    dc_model = DixonColesModel()
    dc_model.fit(dc_data)

    dc_params = {
        "attack": dc_model.attack,
        "defense": dc_model.defense,
        "rho": dc_model.rho,
        "trained_at": datetime.now().isoformat(),
        "n_matches": len(train_matches_raw),
    }
    (MODELS_DIR / "football_dc.json").write_text(json.dumps(dc_params, indent=2))
    if verbose:
        print(f"[train:football] Dixon-Coles fitted. rho={dc_model.rho:.4f}, {len(dc_model.attack)} teams.")

    # ── Build ML feature vectors chronologically ──────────────────────────────
    if verbose:
        print("[train:football] Building rolling feature vectors...")

    elo_pred = EloPredictor(default_elo=1500)
    team_hist = defaultdict(list)
    h2h_hist = defaultdict(list)

    X_train, y_train = [], []
    X_test, y_test = [], []
    test_info = []  # match metadata for backtest report

    for i, m in enumerate(matches):
        ht = m["home_team"]
        at = m["away_team"]
        hg = m["home_goals"]
        ag = m["away_goals"]
        h_xg = m.get("home_xg", float(hg))
        a_xg = m.get("away_xg", float(ag))

        is_train = i < split_idx

        # Only include when both teams have baseline history
        if (len(team_hist[ht]) >= MIN_TEAM_GAMES and len(team_hist[at]) >= MIN_TEAM_GAMES):
            home_stats = _rolling_team_stats(team_hist[ht])
            home_stats["elo"] = elo_pred.get(ht.lower())
            away_stats = _rolling_team_stats(team_hist[at])
            away_stats["elo"] = elo_pred.get(at.lower())

            h2h_key = tuple(sorted([ht, at]))
            h2h_games = h2h_hist[h2h_key]
            if h2h_games:
                h2h_home_wins = sum(1 for r in h2h_games if r == ht)
                h2h_home_wr = h2h_home_wins / len(h2h_games)
            else:
                h2h_home_wr = 0.33

            ctx = {
                "h2h_home_win_rate": h2h_home_wr,
                "competition_weight": 0.85,
                "is_neutral": 0,
            }
            feat = build_football_features(home_stats, away_stats, ctx)
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
                    "league": m.get("league", ""),
                })

        # Update rolling state AFTER building features (no lookahead)
        elo_a = elo_pred.get(ht.lower())
        elo_b = elo_pred.get(at.lower())
        result_val = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        new_a, new_b = update_elo(elo_a, elo_b, result_val, k=20.0)
        elo_pred.set(ht.lower(), new_a)
        elo_pred.set(at.lower(), new_b)

        team_hist[ht].append({
            "gf": hg, "ga": ag, "xgf": h_xg, "xga": a_xg,
            "win": 1 if hg > ag else 0,
            "elo_after": new_a,
        })
        team_hist[at].append({
            "gf": ag, "ga": hg, "xgf": a_xg, "xga": h_xg,
            "win": 1 if ag > hg else 0,
            "elo_after": new_b,
        })

        h2h_key = tuple(sorted([ht, at]))
        h2h_hist[h2h_key].append(ht if hg > ag else (at if ag > hg else None))

    if not X_train:
        print("[train:football] Insufficient data for ML training.")
        return {}

    X_tr = np.array(X_train, dtype=np.float32)
    y_tr = np.array(y_train, dtype=np.int32)
    if verbose:
        vc = np.bincount(y_tr, minlength=3)
        pct = vc / len(y_tr) * 100
        print(
            f"[train:football] Training on {len(X_tr)} samples | "
            f"away {pct[0]:.0f}%  draw {pct[1]:.0f}%  home {pct[2]:.0f}%"
        )

    ml_model = MLEnsemble(sport="football", n_classes=3)
    ml_model.fit(X_tr, y_tr)
    if verbose:
        print(f"[train:football] ML ensemble trained. Models saved to data/models/")

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    metrics = {}
    if X_test:
        X_te = np.array(X_test, dtype=np.float32)
        y_te = np.array(y_test, dtype=np.int32)

        probs_list = [ml_model.predict_proba(x) for x in X_te]
        probs_arr = np.array(probs_list)

        predicted = np.argmax(probs_arr, axis=1)
        accuracy = float((predicted == y_te).mean())

        # Brier score: mean over all classes
        n_classes = 3
        one_hot = np.zeros((len(y_te), n_classes))
        for j, label in enumerate(y_te):
            one_hot[j, label] = 1.0
        brier = float(np.mean(np.sum((probs_arr - one_hot) ** 2, axis=1)))

        # Also evaluate DC model predictions on test set
        dc_probs_list = []
        for info in test_info:
            p = dc_model.predict(info["home"].lower(), info["away"].lower())
            dc_probs_list.append([p["away_win"], p["draw"], p["home_win"]])
        dc_probs_arr = np.array(dc_probs_list) if dc_probs_list else None

        dc_accuracy = dc_brier = None
        if dc_probs_arr is not None and len(dc_probs_arr) == len(y_te):
            dc_pred = np.argmax(dc_probs_arr, axis=1)
            dc_accuracy = float((dc_pred == y_te).mean())
            dc_brier = float(np.mean(np.sum((dc_probs_arr - one_hot) ** 2, axis=1)))

        # Calibration data (10 bins)
        calibration = _calibration_data(probs_arr, one_hot)

        metrics = {
            "sport": "football",
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "ml_accuracy": round(accuracy, 4),
            "ml_brier": round(brier, 4),
            "dc_accuracy": round(dc_accuracy, 4) if dc_accuracy is not None else None,
            "dc_brier": round(dc_brier, 4) if dc_brier is not None else None,
            "naive_brier": 0.667,  # baseline: uniform 1/3 over 3 classes → BS = 2*(1/3)^2 + (2/3)^2 = 0.667
            "calibration": calibration,
            "generated_at": datetime.now().isoformat(),
        }

        (BACKTEST_DIR / "football_results.json").write_text(json.dumps(metrics, indent=2))

        if verbose:
            print(
                f"[train:football] Test set ({len(X_te)} matches): "
                f"ML accuracy={accuracy:.1%}  Brier={brier:.3f}  "
                f"(naive baseline={0.667:.3f})"
            )
            if dc_accuracy is not None:
                print(
                    f"[train:football] Dixon-Coles: accuracy={dc_accuracy:.1%}  "
                    f"Brier={dc_brier:.3f}"
                )

    return metrics


# ── Tennis training ───────────────────────────────────────────────────────────

def train_tennis(verbose: bool = True) -> dict:
    """
    Fetch Jeff Sackmann ATP CSVs (2015–present), build rolling player stats,
    train binary MLEnsemble (0=loser wins, 1=winner wins... randomised for balance).
    """
    from src.data.scrapers.historical import fetch_tennis_history
    from src.models.ml_ensemble import MLEnsemble, build_tennis_features
    from src.models.elo import EloPredictor, update_elo

    if verbose:
        print("[train:tennis] Fetching ATP match history (Sackmann CSVs)...")

    matches = fetch_tennis_history(verbose=verbose)
    if not matches:
        print("[train:tennis] No data fetched.")
        return {}

    if verbose:
        print(f"[train:tennis] {len(matches)} ATP matches loaded.")

    split_idx = int(len(matches) * 0.8)

    elo_pred = EloPredictor(default_elo=1500)
    player_hist = defaultdict(list)

    X_train, y_train = [], []
    X_test, y_test = [], []
    test_info = []

    rng = np.random.default_rng(42)

    for i, m in enumerate(matches):
        winner = m["winner"]
        loser = m["loser"]
        surface = m["surface"].lower()
        is_train = i < split_idx

        w_hist = player_hist[winner]
        l_hist = player_hist[loser]

        if len(w_hist) >= MIN_PLAYER_GAMES and len(l_hist) >= MIN_PLAYER_GAMES:
            # Randomly assign p1/p2 to avoid winner-position bias
            if rng.random() > 0.5:
                p1, p2 = winner, loser
                label = 1  # p1 wins
            else:
                p1, p2 = loser, winner
                label = 0  # p1 loses (p2 wins)

            p1_stats = _rolling_player_stats(player_hist[p1], surface)
            p1_stats["elo"] = elo_pred.get(p1.lower(), surface)
            p1_stats["recent_rank"] = player_hist[p1][-1].get("rank", 100)

            p2_stats = _rolling_player_stats(player_hist[p2], surface)
            p2_stats["elo"] = elo_pred.get(p2.lower(), surface)
            p2_stats["recent_rank"] = player_hist[p2][-1].get("rank", 100)

            ctx = {
                "surface": surface,
                "h2h_win_rate": 0.5,  # simplified
                "tournament_importance": 0.85,
            }
            feat = build_tennis_features(p1_stats, p2_stats, ctx)

            if is_train:
                X_train.append(feat)
                y_train.append(label)
            else:
                X_test.append(feat)
                y_test.append(label)
                test_info.append({
                    "winner": winner, "loser": loser,
                    "surface": surface,
                    "date": m.get("tourney_date", ""),
                })

        # Update state AFTER features
        elo_w = elo_pred.get(winner.lower(), surface)
        elo_l = elo_pred.get(loser.lower(), surface)
        new_w, new_l = update_elo(elo_w, elo_l, 1.0, k=32.0)
        elo_pred.set(winner.lower(), new_w, surface)
        elo_pred.set(loser.lower(), new_l, surface)
        elo_pred.set(winner.lower(), new_w)  # also update flat rating
        elo_pred.set(loser.lower(), new_l)

        def _serve_stats(row, prefix):
            svpt = float(row.get(f"{prefix}_svpt") or 1) or 1
            first_in = float(row.get(f"{prefix}_1stIn") or 0)
            ace = float(row.get(f"{prefix}_ace") or 0)
            bp_faced = float(row.get(f"{prefix}_bpFaced") or 0)
            bp_saved = float(row.get(f"{prefix}_bpSaved") or 0)
            return {
                "first_serve_in_pct": first_in / svpt,
                "ace_rate": ace / svpt,
                "bp_save_rate": bp_saved / max(bp_faced, 1),
            }

        w_serve = _serve_stats(m, "w")
        l_serve = _serve_stats(m, "l")

        player_hist[winner].append({
            "won": 1, "surface": surface,
            "rank": m.get("winner_rank", 100), "elo_after": new_w,
            **w_serve,
        })
        player_hist[loser].append({
            "won": 0, "surface": surface,
            "rank": m.get("loser_rank", 100), "elo_after": new_l,
            **l_serve,
        })

    if not X_train:
        print("[train:tennis] Insufficient data for training.")
        return {}

    X_tr = np.array(X_train, dtype=np.float32)
    y_tr = np.array(y_train, dtype=np.int32)

    if verbose:
        vc = np.bincount(y_tr, minlength=2)
        print(
            f"[train:tennis] Training on {len(X_tr)} samples | "
            f"p1_loss {vc[0]} / p1_win {vc[1]}"
        )

    ml_model = MLEnsemble(sport="tennis", n_classes=2)
    ml_model.fit(X_tr, y_tr)
    if verbose:
        print("[train:tennis] ML ensemble trained.")

    metrics = {}
    if X_test:
        X_te = np.array(X_test, dtype=np.float32)
        y_te = np.array(y_test, dtype=np.int32)

        probs_list = [ml_model.predict_proba(x) for x in X_te]
        probs_arr = np.array(probs_list)
        predicted = np.argmax(probs_arr, axis=1)
        accuracy = float((predicted == y_te).mean())

        one_hot = np.zeros((len(y_te), 2))
        for j, label in enumerate(y_te):
            one_hot[j, label] = 1.0
        brier = float(np.mean(np.sum((probs_arr - one_hot) ** 2, axis=1)))

        calibration = _calibration_data(probs_arr, one_hot)

        metrics = {
            "sport": "tennis",
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "ml_accuracy": round(accuracy, 4),
            "ml_brier": round(brier, 4),
            "naive_brier": 0.5,  # baseline for binary: (0.5)^2*2 = 0.5
            "calibration": calibration,
            "generated_at": datetime.now().isoformat(),
        }
        (BACKTEST_DIR / "tennis_results.json").write_text(json.dumps(metrics, indent=2))

        if verbose:
            print(
                f"[train:tennis] Test set ({len(X_te)} matches): "
                f"accuracy={accuracy:.1%}  Brier={brier:.3f}  (naive={0.5:.3f})"
            )

    return metrics


# ── Calibration helper ────────────────────────────────────────────────────────

def _calibration_data(probs: np.ndarray, one_hot: np.ndarray, n_bins: int = 10) -> list:
    """
    Compute calibration curve data.
    For each probability bin, returns (mean_predicted_prob, actual_frequency, count).
    """
    bins = []
    bin_edges = np.linspace(0, 1, n_bins + 1)

    # Flatten: each (sample, class) pair is one prediction
    flat_probs = probs.flatten()
    flat_actual = one_hot.flatten()

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (flat_probs >= lo) & (flat_probs < hi)
        count = int(mask.sum())
        if count > 0:
            mean_pred = float(flat_probs[mask].mean())
            actual_freq = float(flat_actual[mask].mean())
        else:
            mean_pred = (lo + hi) / 2
            actual_freq = 0.0
        bins.append({"pred": round(mean_pred, 3), "actual": round(actual_freq, 3), "count": count})

    return bins


# ── CLI entry point ───────────────────────────────────────────────────────────

def main(sports: list = None):
    if sports is None:
        sports = ["football", "tennis"]

    results = {}
    for sport in sports:
        print(f"\n{'='*60}")
        if sport == "football":
            results["football"] = train_football(verbose=True)
        elif sport == "tennis":
            results["tennis"] = train_tennis(verbose=True)
        else:
            print(f"[train] Unknown sport: {sport}")

    print("\n" + "="*60)
    print("[train] Done.")
    for sport, m in results.items():
        if m:
            acc = m.get("ml_accuracy")
            brier = m.get("ml_brier")
            n_test = m.get("n_test")
            if acc is not None:
                print(f"  {sport}: accuracy={acc:.1%}  Brier={brier:.3f}  (n_test={n_test})")

    return results


if __name__ == "__main__":
    target = sys.argv[1:] if len(sys.argv) > 1 else None
    if target:
        main([t.lower() for t in target])
    else:
        main()
