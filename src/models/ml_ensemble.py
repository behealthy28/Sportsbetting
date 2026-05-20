"""
RF + XGBoost + LightGBM soft-voting ensemble for sports prediction.
Trains on historical feature data; falls back gracefully with no data.
"""
import hashlib
import json
import numpy as np
import joblib
from pathlib import Path
from typing import Optional

VALID_SPORTS = {"football", "tennis", "ufc", "boxing", "darts", "badminton", "table_tennis", "cricket"}

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

MODELS_DIR = Path(__file__).parent.parent.parent / "data" / "models"


class MLEnsemble:
    """
    RF + XGBoost + LightGBM soft-voting ensemble.
    For 1v1 sports: binary classification (0=player_b, 1=player_a).
    For team sports: 3-class (0=away, 1=draw, 2=home).
    """

    def __init__(self, sport: str = "football", n_classes: int = 3):
        if sport not in VALID_SPORTS:
            raise ValueError(f"Unknown sport {sport!r}. Valid: {sorted(VALID_SPORTS)}")
        self.sport = sport
        self.n_classes = n_classes
        self.rf: Optional[object] = None
        self.xgb_model: Optional[object] = None
        self.lgb_model: Optional[object] = None
        self.scaler: Optional[object] = None
        self.is_fitted = False
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def _model_path(self, name: str) -> Path:
        return MODELS_DIR / f"{self.sport}_{name}"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train RF + XGBoost + LightGBM on feature matrix X and labels y."""
        if not ML_AVAILABLE:
            return

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Random Forest — Platt-scaled for calibrated probabilities
        rf_base = RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.rf = CalibratedClassifierCV(rf_base, method="isotonic", cv=3)
        self.rf.fit(X_scaled, y)

        # XGBoost — tuned params that consistently beat defaults for sports data
        obj = "binary:logistic" if self.n_classes == 2 else "multi:softprob"
        xgb_params = {
            "objective": obj,
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss" if self.n_classes == 2 else "mlogloss",
        }
        if self.n_classes > 2:
            xgb_params["num_class"] = self.n_classes

        self.xgb_model = xgb.XGBClassifier(**xgb_params)
        self.xgb_model.fit(X_scaled, y)

        # LightGBM — often fastest and matches/beats XGBoost on tabular sports data
        if LGB_AVAILABLE:
            lgb_obj = "binary" if self.n_classes == 2 else "multiclass"
            lgb_params = {
                "objective": lgb_obj,
                "n_estimators": 500,
                "num_leaves": 63,
                "learning_rate": 0.03,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_child_samples": 20,
                "reg_alpha": 0.05,
                "reg_lambda": 0.1,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
            if self.n_classes > 2:
                lgb_params["num_class"] = self.n_classes
            self.lgb_model = lgb.LGBMClassifier(**lgb_params)
            self.lgb_model.fit(X_scaled, y)

        self.is_fitted = True

        joblib.dump(self.rf, self._model_path("rf.pkl"))
        joblib.dump(self.xgb_model, self._model_path("xgb.pkl"))
        joblib.dump(self.scaler, self._model_path("scaler.pkl"))
        if self.lgb_model is not None:
            joblib.dump(self.lgb_model, self._model_path("lgb.pkl"))
        self._write_hashes()

    def _sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    def _write_hashes(self) -> None:
        manifest = {}
        for name in ("rf.pkl", "xgb.pkl", "scaler.pkl", "lgb.pkl"):
            p = self._model_path(name)
            if p.exists():
                manifest[name] = self._sha256(p)
        (self._model_path("manifest.json")).write_text(json.dumps(manifest, indent=2))

    def _verify_hashes(self) -> bool:
        manifest_path = self._model_path("manifest.json")
        if not manifest_path.exists():
            return True  # no manifest — legacy models, skip check
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest.items():
            p = self._model_path(name)
            if p.exists() and self._sha256(p) != expected:
                return False
        return True

    def load(self) -> bool:
        """Load persisted models (RF + XGBoost required; LightGBM optional)."""
        if not ML_AVAILABLE:
            return False
        try:
            if not self._verify_hashes():
                print(f"[MLEnsemble] WARNING: model hash mismatch for {self.sport} — skipping load")
                return False
            self.rf = joblib.load(self._model_path("rf.pkl"))
            self.xgb_model = joblib.load(self._model_path("xgb.pkl"))
            self.scaler = joblib.load(self._model_path("scaler.pkl"))
            lgb_path = self._model_path("lgb.pkl")
            if LGB_AVAILABLE and lgb_path.exists():
                self.lgb_model = joblib.load(lgb_path)
            self.is_fitted = True
            return True
        except Exception:
            return False

    def _ensemble_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """Weighted soft-vote across available models. Returns (n_samples, n_classes)."""
        import warnings
        rf_p   = self.rf.predict_proba(X_scaled)
        xgb_p  = self.xgb_model.predict_proba(X_scaled)
        if self.lgb_model is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                lgb_p = self.lgb_model.predict_proba(X_scaled)
            # RF 25% | XGBoost 37.5% | LightGBM 37.5%
            ensemble = 0.25 * rf_p + 0.375 * xgb_p + 0.375 * lgb_p
        else:
            # RF 45% | XGBoost 55%
            ensemble = 0.45 * rf_p + 0.55 * xgb_p
        ensemble /= ensemble.sum(axis=1, keepdims=True)
        return ensemble

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Single-sample prediction. Returns (n_classes,)."""
        if not self.is_fitted or not ML_AVAILABLE:
            return np.full(self.n_classes, 1.0 / self.n_classes)
        X_scaled = self.scaler.transform(features.reshape(1, -1))
        return self._ensemble_proba(X_scaled)[0]

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorised batch prediction. Returns (n_samples, n_classes)."""
        if not self.is_fitted or not ML_AVAILABLE:
            return np.full((len(X), self.n_classes), 1.0 / self.n_classes)
        X_scaled = self.scaler.transform(X)
        return self._ensemble_proba(X_scaled)

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
        (elo_p1 - elo_p2) / 400.0,                                          # 0: overall ELO diff
        safe(ctx, "surf_elo_diff", (elo_p1 - elo_p2) / 400.0),              # 1: surface-specific ELO diff
        safe(ctx, "h2h_win_rate", 0.5),                                      # 2: real H2H win rate
        safe(p1_surf, "win_rate", 0.5) - safe(p2_surf, "win_rate", 0.5),    # 3: surface win rate diff
        safe(p1_data, "overall_win_rate", 0.5),                              # 4: p1 long-term form
        safe(p2_data, "overall_win_rate", 0.5),                              # 5: p2 long-term form
        safe(p1_data, "short_form", safe(p1_data, "overall_win_rate", 0.5)),  # 6: p1 short-term form (last 5)
        safe(p2_data, "short_form", safe(p2_data, "overall_win_rate", 0.5)),  # 7: p2 short-term form
        (safe(p1_data, "recent_rank", 50) - safe(p2_data, "recent_rank", 50)) / 100.0,  # 8: rank diff
        safe(p1_surf, "first_serve_pct", 0.62) - safe(p2_surf, "first_serve_pct", 0.62),  # 9: serve diff
        safe(p1_surf, "bp_save_rate", 0.62) - safe(p2_surf, "bp_save_rate", 0.62),      # 10: break pt save diff
        safe(p1_surf, "ace_rate", 0.05) - safe(p2_surf, "ace_rate", 0.05),              # 11: ace rate diff
        safe(ctx, "days_rest_diff", 0) / 7.0,                               # 12: rest diff
        safe(ctx, "age_diff", 0) / 10.0,                                    # 13: age diff
        float(ctx.get("injury_p1", 0)),                                      # 14: p1 injury flag
        float(ctx.get("injury_p2", 0)),                                      # 15: p2 injury flag
        safe(p1_data, "overall_win_rate", 0.5) - safe(p2_data, "overall_win_rate", 0.5),  # 16: career win rate diff
        safe(ctx, "tournament_importance", 0.85),                            # 17: tournament weight
        safe(ctx, "p1_short_form", safe(p1_data, "overall_win_rate", 0.5)) -
        safe(ctx, "p2_short_form", safe(p2_data, "overall_win_rate", 0.5)), # 18: short form diff
        safe(p1_surf, "win_rate", safe(p1_data, "overall_win_rate", 0.5)),  # 19: p1 surface abs win rate
        safe(p2_surf, "win_rate", safe(p2_data, "overall_win_rate", 0.5)),  # 20: p2 surface abs win rate
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
