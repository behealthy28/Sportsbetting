"""
Random Forest + XGBoost soft-voting ensemble for sports prediction.
Trains on historical feature data; falls back gracefully with no data.
"""
import numpy as np
import joblib
from pathlib import Path
from typing import Optional

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

MODELS_DIR = Path(__file__).parent.parent.parent / "data" / "models"


class MLEnsemble:
    """
    Random Forest + XGBoost ensemble.
    For 1v1 sports: binary classification (0=player_b, 1=player_a).
    For team sports: 3-class (0=away, 1=draw, 2=home).
    """

    def __init__(self, sport: str = "football", n_classes: int = 3):
        self.sport = sport
        self.n_classes = n_classes
        self.rf: Optional[object] = None
        self.xgb_model: Optional[object] = None
        self.scaler: Optional[object] = None
        self.is_fitted = False
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def _model_path(self, name: str) -> Path:
        return MODELS_DIR / f"{self.sport}_{name}"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train RF + XGBoost on feature matrix X and labels y."""
        if not ML_AVAILABLE:
            return

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Random Forest with isotonic calibration
        rf_base = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.rf = CalibratedClassifierCV(rf_base, method="isotonic", cv=3)
        self.rf.fit(X_scaled, y)

        # XGBoost
        obj = "binary:logistic" if self.n_classes == 2 else "multi:softprob"
        params = {
            "objective": obj,
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "logloss" if self.n_classes == 2 else "mlogloss",
        }
        if self.n_classes > 2:
            params["num_class"] = self.n_classes

        self.xgb_model = xgb.XGBClassifier(**params)
        self.xgb_model.fit(X_scaled, y)
        self.is_fitted = True

        # Persist
        joblib.dump(self.rf, self._model_path("rf.pkl"))
        joblib.dump(self.xgb_model, self._model_path("xgb.pkl"))
        joblib.dump(self.scaler, self._model_path("scaler.pkl"))

    def load(self) -> bool:
        """Load persisted models."""
        if not ML_AVAILABLE:
            return False
        try:
            self.rf = joblib.load(self._model_path("rf.pkl"))
            self.xgb_model = joblib.load(self._model_path("xgb.pkl"))
            self.scaler = joblib.load(self._model_path("scaler.pkl"))
            self.is_fitted = True
            return True
        except Exception:
            return False

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for a single feature vector.
        Returns array of shape (n_classes,).
        """
        if not self.is_fitted or not ML_AVAILABLE:
            return np.full(self.n_classes, 1.0 / self.n_classes)

        X = features.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        rf_probs = self.rf.predict_proba(X_scaled)[0]
        xgb_probs = self.xgb_model.predict_proba(X_scaled)[0]

        # Soft vote: 55% XGBoost, 45% RF
        ensemble = 0.55 * xgb_probs + 0.45 * rf_probs
        ensemble /= ensemble.sum()
        return ensemble

    def predict_dict(self, features: np.ndarray, labels: list = None) -> dict:
        """Return probabilities as a labeled dict."""
        probs = self.predict_proba(features)
        if labels is None:
            if self.n_classes == 3:
                labels = ["away_win", "draw", "home_win"]
            else:
                labels = ["b_win", "a_win"]
        return {labels[i]: round(float(probs[i]), 4) for i in range(len(labels))}


def build_football_features(home_data: dict, away_data: dict, context: dict = None) -> np.ndarray:
    """Build 20-feature vector for football prediction."""
    ctx = context or {}

    def safe(d, key, default=0.5):
        v = d.get(key, default)
        return float(v) if v is not None else default

    elo_home = safe(home_data, "elo", 1500)
    elo_away = safe(away_data, "elo", 1500)

    features = np.array([
        safe(home_data, "form", 0.5),                   # 0: home form (5 games)
        safe(away_data, "form", 0.5),                   # 1: away form
        safe(home_data, "avg_goals", 1.4),              # 2: home goals/game
        safe(away_data, "avg_goals", 1.4),              # 3: away goals/game
        safe(home_data, "avg_xg", safe(home_data, "avg_goals", 1.4)),  # 4: home xG
        safe(away_data, "avg_xg", safe(away_data, "avg_goals", 1.4)),  # 5: away xG
        safe(home_data, "avg_conceded", 1.2),           # 6: home goals conceded
        safe(away_data, "avg_conceded", 1.2),           # 7: away goals conceded
        safe(ctx, "h2h_home_win_rate", 0.33),           # 8: H2H home win rate
        (elo_home - elo_away) / 400.0,                  # 9: normalized ELO diff
        safe(home_data, "clean_sheet_rate", 0.3),       # 10: home clean sheet %
        safe(away_data, "clean_sheet_rate", 0.3),       # 11: away clean sheet %
        safe(ctx, "home_key_players", 1.0),             # 12: home player availability
        safe(ctx, "away_key_players", 1.0),             # 13: away player availability
        safe(ctx, "home_days_rest", 7) / 7.0,          # 14: home rest (normalized)
        safe(ctx, "away_days_rest", 7) / 7.0,          # 15: away rest
        safe(ctx, "competition_weight", 0.85),          # 16: competition importance
        safe(home_data, "ranking", 50) / 100.0,         # 17: home ranking (normalized)
        safe(away_data, "ranking", 50) / 100.0,         # 18: away ranking
        float(ctx.get("is_neutral", 0)),                # 19: neutral venue
        safe(ctx, "home_streak_norm", 0.0),             # 20: home win/loss streak (signed)
        safe(ctx, "away_streak_norm", 0.0),             # 21: away win/loss streak
        safe(ctx, "home_bounceback", 0.5),              # 22: home win rate after dropped pts
        safe(ctx, "away_bounceback", 0.5),              # 23: away win rate after dropped pts
    ], dtype=np.float32)

    return features


def build_tennis_features(p1_data: dict, p2_data: dict, context: dict = None) -> np.ndarray:
    """Build 18-feature vector for tennis prediction."""
    ctx = context or {}
    surface = ctx.get("surface", "hard")

    def safe(d, key, default=0.5):
        v = d.get(key, default)
        return float(v) if v is not None else default

    p1_surf = p1_data.get("surface_stats", {}).get(surface, {})
    p2_surf = p2_data.get("surface_stats", {}).get(surface, {})

    elo_p1 = safe(p1_data, "elo", 1500)
    elo_p2 = safe(p2_data, "elo", 1500)

    features = np.array([
        (elo_p1 - elo_p2) / 400.0,                                    # 0: ELO diff
        safe(p1_surf, "win_rate", 0.5) - safe(p2_surf, "win_rate", 0.5),  # 1: surface win rate diff
        safe(ctx, "h2h_win_rate", 0.5),                               # 2: H2H win rate
        safe(p1_surf, "win_rate", 0.5) - safe(p2_surf, "win_rate", 0.5),  # 3: surface H2H (approx)
        safe(p1_data, "overall_win_rate", 0.5),                       # 4: p1 overall form
        safe(p2_data, "overall_win_rate", 0.5),                       # 5: p2 overall form
        (safe(p1_data, "recent_rank", 50) - safe(p2_data, "recent_rank", 50)) / 100.0,  # 6: rank diff
        safe(p1_surf, "first_serve_pct", 0.62) - safe(p2_surf, "first_serve_pct", 0.62),  # 7: serve diff
        0.0,                                                           # 8: break pt save diff (approx)
        safe(p1_surf, "ace_rate", 0.05) - safe(p2_surf, "ace_rate", 0.05),  # 9: ace rate diff
        safe(ctx, "recent_result_p1", 0.5),                           # 10: recent tourney result
        safe(ctx, "days_rest_diff", 0) / 7.0,                        # 11: rest diff
        safe(ctx, "age_diff", 0) / 10.0,                             # 12: age diff
        float(ctx.get("injury_p1", 0)),                               # 13: p1 injury flag
        float(ctx.get("injury_p2", 0)),                               # 14: p2 injury flag
        0.0,                                                           # 15: ranking trend
        safe(p1_data, "overall_win_rate", 0.5) - safe(p2_data, "overall_win_rate", 0.5),  # 16: career diff
        safe(ctx, "tournament_importance", 0.85),                     # 17: tournament weight
    ], dtype=np.float32)

    return features


def build_ufc_features(f1_data: dict, f2_data: dict, context: dict = None) -> np.ndarray:
    """Build 16-feature vector for UFC/MMA prediction."""
    ctx = context or {}

    def diff(key, default=0.0):
        v1 = float(f1_data.get(key, default))
        v2 = float(f2_data.get(key, default))
        return v1 - v2

    features = np.array([
        diff("slpm"),                                    # 0: strikes landed/min diff
        diff("str_acc"),                                 # 1: striking accuracy diff
        diff("sapm"),                                    # 2: strikes absorbed diff
        diff("str_def"),                                 # 3: striking defense diff
        diff("td_avg"),                                  # 4: takedowns/match diff
        diff("td_acc"),                                  # 5: TD accuracy diff
        diff("td_def"),                                  # 6: TD defense diff
        diff("sub_avg"),                                 # 7: submission avg diff
        (float(f1_data.get("reach_cm", 183)) - float(f2_data.get("reach_cm", 183))) / 20.0,  # 8
        (float(f1_data.get("height_cm", 178)) - float(f2_data.get("height_cm", 178))) / 10.0,  # 9
        (float(f1_data.get("age", 30)) - float(f2_data.get("age", 30))) / 5.0,  # 10
        diff("win_rate"),                                # 11: win rate diff
        diff("finish_rate"),                             # 12: finish rate diff
        (safe_ctx(ctx, "days_since_fight_f1") - safe_ctx(ctx, "days_since_fight_f2")) / 180.0,  # 13
        float(ctx.get("h2h_result", 0)),                # 14: H2H result (-1/0/1)
        diff("ko_wins") / max(float(f1_data.get("wins", 1)), 1),  # 15: KO rate diff
    ], dtype=np.float32)

    return features


def safe_ctx(ctx, key, default=90.0):
    v = ctx.get(key, default)
    return float(v) if v is not None else default
