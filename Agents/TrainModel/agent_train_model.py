import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import json
from pathlib import Path

from sklearn.metrics import roc_auc_score
from Agents.BaseAgent.base_agent import BaseAgent
from logger import AgentLogger
from config import Config


# Marker column used in pre-split mode (set by AutoMLPipeline._build_combined_input).
# When present in self.df, _tool_split_data uses it to reconstruct exact
# user-defined splits instead of running the temporal / random auto-split.
_SPLIT_MARKER = "_split_"


def _smallest_int_dtype(n_classes: int) -> np.dtype:
    """Smallest signed int dtype that fits LabelEncoder output range [0, n-1].

    Used to keep encoded categorical columns at 1-4 bytes/row instead of int64's 8.
    A 100k×1500 DataFrame at int8 is 150 MB; at int64 it's 1.2 GB — the difference
    decides whether FLAML's internal X.copy() OOMs on Windows.
    """
    if n_classes <= 127:
        return np.int8
    if n_classes <= 32_767:
        return np.int16
    return np.int32


class TrainModelAgent(BaseAgent):
    """
    Advanced model training agent: FLAML → Optuna → RFE → PSI → Stability → Final Model.

    Pipeline steps:
      1. Split    : OOT temporal split (fallback to simple split if no date col)
      2. FLAML    : AutoML selects best estimator + base hyperparams
      3. Optuna   : TPE fine-tunes hyperparams on valid-temporal (no leakage)
      4. RFE      : Recursive feature elimination → MAX_FINAL_FEATURES features
      5. PSI      : Drop features with high distribution drift (train vs OOT)
      6. Stability: Drop features with unstable monthly Gini
      7. SHAP+PSI : Iterative pruning — remove low-importance/high-drift features
                    until validation AUC stops improving
      8. Final    : Train + evaluate on CV / valid-temporal / valid-random / OOT
    """

    def __init__(
        self,
        logger: AgentLogger,
        col_descriptions: Optional[Dict[str, str]] = None,
        domain: str = "generic",
    ):
        super().__init__(name="TrainModel", role="Advanced ML Pipeline Engineer", logger=logger)
        self.df: pd.DataFrame = None
        self.target_column: str = None
        self.date_col: Optional[str] = None
        self.id_col: Optional[str] = None
        # Temporal frequency controls the granularity of every temporal slice.
        # temporal_freq ∈ {"auto","weekly","monthly"} (default from Config); when
        # "weekly", week_closing_day picks the W-anchor (auto from data if blank).
        # Resolved into period_freq (e.g. "M" / "W-FRI") + period_unit at split time.
        self.temporal_freq: str = Config.TEMPORAL_FREQ
        self.week_closing_day: str = Config.WEEK_CLOSING_DAY
        self.period_freq: str = "M"
        self.period_unit: str = "month"
        # Temporal cadence detected from date_col during _tool_split_data.
        # cadence ∈ {"monthly_snapshot","intra_month","weekly_snapshot","weekly",
        # "non_temporal"}; temporal_meta records how many WHOLE PERIODS (months or
        # weeks) each split spans so the report makes the granularity explicit.
        self.date_cadence: Optional[str] = None
        self.temporal_meta: Dict[str, object] = {"cadence": "non_temporal"}
        self.splits: Dict[str, pd.DataFrame] = {}
        self.feature_cols: List[str] = []
        self._cat_encoders: Dict[str, Any] = {}
        self.model: Any = None
        # Multi-seed bagging ensemble + isotonic/sigmoid calibrator. Populated
        # by _tool_train_final_model. ensemble_models[0] == self.model
        # (the primary fit used by SHAP). calibrator is None when disabled
        # or when no usable calibration split was found.
        self.ensemble_models: List[Any] = []
        self.calibrator: Any = None
        # Resolved calibration on/off for THIS run. Defaults to the global
        # Config toggle; process()/process_splits() may override per-run via the
        # `calibration=` arg (None → keep Config default). Read in
        # _tool_train_final_model so the overfit-retry path honours it too.
        self.calibration_enabled: bool = Config.CALIBRATION_ENABLED
        # Forwarded from FeatureEngineer / Pipeline so the post-training SHAP
        # explain step can ground the LLM's narrative in real column semantics.
        self._col_descriptions: Dict[str, str] = col_descriptions or {}
        self._domain: str = domain

    # ── Column auto-detection ─────────────────────────────────────────────────

    def _auto_detect_id_col(self, df: pd.DataFrame) -> Optional[str]:
        """Simple fallback — used only when entity_id_col is absent from the previous report."""
        for col in df.columns:
            lc = col.lower()
            if lc in ("customer_id", "user_id", "id", "cif") or lc.endswith("_id"):
                return col
        return None

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        exclude = {self.target_column}
        if self.date_col:
            exclude.add(self.date_col)
        if self.id_col:
            exclude.add(self.id_col)
        # Pre-split marker is metadata, never a feature
        exclude.add(_SPLIT_MARKER)
        return [c for c in df.columns if c not in exclude]

    def _fit_prepare_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit encoders on train data and transform. Call once on X_train.

        Builds output column-by-column with smallest-fitting dtypes (int8/16/32 for
        encoded categoricals, float32 for numerics) so the resulting DataFrame uses
        ~4-8x less RAM than the naive float64/int64 default. Critical for FLAML
        which calls X.copy() internally and can OOM on wide DataFrames.
        """
        from sklearn.preprocessing import LabelEncoder
        self._cat_encoders = {}
        new_cols: Dict[str, np.ndarray] = {}
        for col in df.columns:
            s = df[col]
            if s.dtype == object or hasattr(s.dtype, "categories"):
                le = LabelEncoder()
                series = s.fillna("__NA__").astype(str)
                # Always include "__NA__" so valid/OOT NaN values can be encoded
                # even when train has no NaN for this column
                le.fit(sorted(set(series) | {"__NA__"}))
                dtype = _smallest_int_dtype(len(le.classes_))
                new_cols[col] = le.transform(series).astype(dtype, copy=False)
                self._cat_encoders[col] = le
            elif pd.api.types.is_float_dtype(s):
                new_cols[col] = s.fillna(-999).to_numpy(dtype=np.float32, copy=False)
            elif pd.api.types.is_integer_dtype(s):
                new_cols[col] = s.fillna(-999).to_numpy(dtype=np.int32, copy=False)
            elif pd.api.types.is_bool_dtype(s):
                new_cols[col] = s.to_numpy(dtype=np.int8, copy=False)
            else:
                new_cols[col] = pd.to_numeric(s, errors="coerce") \
                    .fillna(-999).to_numpy(dtype=np.float32, copy=False)
        return pd.DataFrame(new_cols, index=df.index)

    def _transform_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using encoders fitted on train. Use for valid/oot splits.

        Mirrors the dtype choices made in _fit_prepare_X so X_train / X_valid / X_oot
        share an identical schema (FLAML refuses validation data with dtype drift).
        """
        new_cols: Dict[str, np.ndarray] = {}
        for col in df.columns:
            s = df[col]
            if col in self._cat_encoders:
                le = self._cat_encoders[col]
                known = set(le.classes_)
                vals = s.fillna("__NA__").astype(str)
                # Unseen labels → "__NA__" (must exist in le.classes_ from fit)
                mapped = vals.where(vals.isin(known), "__NA__")
                dtype = _smallest_int_dtype(len(le.classes_))
                new_cols[col] = le.transform(mapped).astype(dtype, copy=False)
            elif pd.api.types.is_float_dtype(s):
                new_cols[col] = s.fillna(-999).to_numpy(dtype=np.float32, copy=False)
            elif pd.api.types.is_integer_dtype(s):
                new_cols[col] = s.fillna(-999).to_numpy(dtype=np.int32, copy=False)
            elif pd.api.types.is_bool_dtype(s):
                new_cols[col] = s.to_numpy(dtype=np.int8, copy=False)
            else:
                new_cols[col] = pd.to_numeric(s, errors="coerce") \
                    .fillna(-999).to_numpy(dtype=np.float32, copy=False)
        return pd.DataFrame(new_cols, index=df.index)

    # ── Model class lookup ────────────────────────────────────────────────────

    def _get_n_estimators_key(self, estimator_name: str) -> str:
        """CatBoost uses 'iterations', all others use 'n_estimators'."""
        return "iterations" if estimator_name in ("catboost", "CatBoost") else "n_estimators"

    def _extract_best_iteration(self, model, estimator_name: str) -> Optional[int]:
        """Return the early-stopping-selected tree count as an n_estimators value,
        or None for libs/configs without ES (RF / ExtraTrees / DART / failure).

        lightgbm exposes 0-based-safe `best_iteration_`; xgboost/catboost expose a
        0-indexed best iteration → +1 to convert to a tree count.
        """
        try:
            if estimator_name in ("lgbm", "LightGBM"):
                bi = getattr(model, "best_iteration_", None)
                return int(bi) if bi else None
            if estimator_name in ("xgboost", "XGBoost"):
                bi = getattr(model, "best_iteration", None)
                return int(bi) + 1 if bi is not None else None
            if estimator_name in ("catboost", "CatBoost"):
                bi = model.get_best_iteration()
                return int(bi) + 1 if bi is not None else None
        except Exception:
            return None
        return None

    def _get_model_class(self, estimator_name: str):
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        mapping = {
            "rf": RandomForestClassifier,
            "RandomForest": RandomForestClassifier,
            "extra_tree": ExtraTreesClassifier,
            "ExtraTrees": ExtraTreesClassifier,
        }
        try:
            import lightgbm as lgb
            mapping["lgbm"] = lgb.LGBMClassifier
            mapping["LightGBM"] = lgb.LGBMClassifier
        except ImportError:
            pass
        try:
            import xgboost as xgb
            mapping["xgboost"] = xgb.XGBClassifier
            mapping["XGBoost"] = xgb.XGBClassifier
        except ImportError:
            pass
        try:
            from catboost import CatBoostClassifier
            mapping["catboost"] = CatBoostClassifier
            mapping["CatBoost"] = CatBoostClassifier
        except ImportError:
            pass
        return mapping.get(estimator_name, mapping.get("lgbm", RandomForestClassifier))

    @staticmethod
    def _safe_params(estimator_name: str, params: Dict) -> Dict:
        """Filter params to only those valid for estimator_name.

        Prevents 'Unknown parameter' errors when best_params originates from a
        different model type — e.g. Optuna times out and falls back to FLAML's
        best_config, which may contain RF-specific keys (min_samples_split, …)
        that LightGBM / XGBoost / CatBoost do not accept.
        """
        _lgbm = {
            "n_estimators", "learning_rate", "num_leaves", "max_depth",
            "min_child_samples", "subsample", "subsample_freq", "colsample_bytree",
            "reg_alpha", "reg_lambda", "min_split_gain", "bagging_freq",
            "random_state", "n_jobs", "verbose", "device",
            "class_weight", "importance_type",
            # DART-mode hyperparameters (Optuna-tuned when boosting_type='dart')
            "boosting_type", "drop_rate", "max_drop", "skip_drop",
        }
        _xgb = {
            "n_estimators", "learning_rate", "max_depth", "min_child_weight",
            "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda",
            "eval_metric", "random_state", "n_jobs", "verbosity",
            "device", "tree_method", "predictor", "scale_pos_weight",
        }
        _cb = {
            "iterations", "learning_rate", "depth", "l2_leaf_reg",
            "bagging_temperature", "random_strength", "random_seed",
            "verbose", "task_type", "scale_pos_weight",
        }
        _rf = {
            "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
            "max_features", "random_state", "n_jobs", "class_weight",
            "bootstrap", "min_impurity_decrease", "max_leaf_nodes",
            "min_weight_fraction_leaf", "max_samples",
        }
        whitelist = {
            "lgbm": _lgbm, "LightGBM": _lgbm,
            "xgboost": _xgb, "XGBoost": _xgb,
            "catboost": _cb, "CatBoost": _cb,
            "rf": _rf, "RandomForest": _rf,
            "extra_tree": _rf, "ExtraTrees": _rf,
        }.get(estimator_name)
        if whitelist is None:
            return params
        return {k: v for k, v in params.items() if k in whitelist}

    # ── AUC boosters: class weight + early stopping ──────────────────────────

    @staticmethod
    def _compute_imbalance_ratio(y: pd.Series) -> float:
        """Return majority/minority count ratio. 1.0 when balanced or non-binary."""
        try:
            counts = pd.Series(y).dropna().value_counts()
        except Exception:
            return 1.0
        if len(counts) != 2:
            return 1.0
        majority = int(counts.max())
        minority = int(counts.min())
        return (majority / minority) if minority > 0 else 1.0

    @staticmethod
    def _compute_class_weight_params(y: pd.Series, estimator_name: str) -> Dict[str, Any]:
        """Auto-derive class weighting from train imbalance.

        Returns {} when imbalance is mild (caller can blindly **-unpack).

        The minority-class weight is picked by `Config.CLASS_WEIGHT_STRATEGY`:
            auto     → sqrt(ratio) when ratio >= CLASS_WEIGHT_SEVERE_THRESHOLD,
                       else ratio.   Recommended default for credit_risk.
            balanced → ratio (full sklearn-style balance — can over-weight).
            sqrt     → sqrt(ratio) (softer; best empirical AUC on banking 95:5).
            half     → ratio/2.
            <float>  → float * ratio (custom multiplier).

        On severe imbalance (banking 95:5, ratio=19), boosters are already
        adaptive and full `balanced` weight (19x) tends to overshoot — the
        model becomes too sensitive to minority and ranking degrades.
        sqrt(19)≈4.4 is empirically the AUC sweet spot.

        Returned kwarg shape:
          xgboost / catboost           → {'scale_pos_weight': <weight>}
          lgbm / rf / extra_tree       → {'class_weight': {0: 1.0, 1: <weight>}}
        """
        try:
            counts = pd.Series(y).dropna().value_counts()
        except Exception:
            return {}
        if len(counts) != 2:
            return {}
        majority = int(counts.max())
        minority = int(counts.min())
        if minority == 0:
            return {}
        ratio = majority / minority
        if ratio < Config.AUTO_CLASS_WEIGHT_THRESHOLD:
            return {}

        # Identify which class is minority — `class_weight` dict needs the right key
        cls_min = counts.idxmin()

        # Resolve weight per strategy
        strategy = str(Config.CLASS_WEIGHT_STRATEGY).lower().strip()
        severe   = ratio >= Config.CLASS_WEIGHT_SEVERE_THRESHOLD
        try:
            if strategy == "auto":
                weight = float(np.sqrt(ratio)) if severe else float(ratio)
            elif strategy == "balanced":
                weight = float(ratio)
            elif strategy == "sqrt":
                weight = float(np.sqrt(ratio))
            elif strategy == "half":
                weight = float(ratio) / 2.0
            else:
                # Numeric multiplier (e.g. "0.3" → 30% of balanced)
                weight = float(strategy) * float(ratio)
        except (TypeError, ValueError):
            weight = float(ratio)   # fallback to balanced on bad strategy string

        weight = max(1.0, round(weight, 2))

        if estimator_name in ("xgboost", "XGBoost", "catboost", "CatBoost"):
            return {"scale_pos_weight": weight}
        if estimator_name in ("lgbm", "LightGBM", "rf", "RandomForest", "extra_tree", "ExtraTrees"):
            # Dict form so the EXACT weight (sqrt/half/etc.) is applied; the
            # string 'balanced' would silently override with sklearn's ratio.
            cls_maj = counts.idxmax()
            return {"class_weight": {cls_maj: 1.0, cls_min: weight}}
        return {}

    def _fit_with_early_stopping(
        self,
        model: Any,
        estimator_name: str,
        X_tr: pd.DataFrame, y_tr: pd.Series,
        X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
        rounds: Optional[int] = None,
    ) -> Any:
        """Fit `model` with early stopping when a validation set is supplied.

        Falls back to plain fit when:
          - No valid set (or empty)
          - Estimator without ES support (RF / ExtraTrees / unknown)
          - ES API call raises (logged + plain fit)

        Each lib has a different ES API:
          - lightgbm : callbacks=[lgb.early_stopping(rounds, verbose=False)]
          - xgboost  : early_stopping_rounds (2.x: ctor param; 1.x: fit param)
          - catboost : early_stopping_rounds in fit(eval_set=...)
        """
        rounds = rounds if rounds is not None else Config.EARLY_STOPPING_ROUNDS
        no_val = X_val is None or len(X_val) == 0
        rf_like = estimator_name in ("rf", "RandomForest", "extra_tree", "ExtraTrees")
        # DART boosting rebuilds dropped trees each round → ES gives no signal
        # (every validation eval can be arbitrarily worse than the previous
        # round). LightGBM raises at fit time if both are set together.
        lgbm_dart = (
            estimator_name in ("lgbm", "LightGBM")
            and getattr(model, "boosting_type", None) == "dart"
        )
        if no_val or rf_like or lgbm_dart:
            model.fit(X_tr, y_tr)
            return model
        try:
            if estimator_name in ("lgbm", "LightGBM"):
                import lightgbm as lgb
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric="auc",
                    callbacks=[lgb.early_stopping(rounds, verbose=False),
                               lgb.log_evaluation(0)],
                )
            elif estimator_name in ("xgboost", "XGBoost"):
                import xgboost as xgb
                if int(xgb.__version__.split(".")[0]) >= 2:
                    # XGBoost 2.x takes early_stopping_rounds on the ctor — but
                    # this param PERSISTS on the fitted model, so any later
                    # cross_val_score / sklearn refit clones the model with ES
                    # still active but no eval_set, raising
                    # "Must have at least 1 validation dataset for early stopping"
                    # and failing every CV fold. Always reset it after the fit.
                    model.set_params(early_stopping_rounds=rounds)
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                    try:
                        model.set_params(early_stopping_rounds=None)
                    except Exception:
                        pass
                else:
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                              early_stopping_rounds=rounds, verbose=False)
            elif estimator_name in ("catboost", "CatBoost"):
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                          early_stopping_rounds=rounds, verbose=False)
            else:
                model.fit(X_tr, y_tr)
        except Exception as e:
            self.logger.log(self.name, "EarlyStop WARN",
                f"ES fit failed for {estimator_name} ({e}) → plain fit")
            try:
                # Reset any ES state that might have been left on the model
                model.set_params(early_stopping_rounds=None)
            except Exception:
                pass
            model.fit(X_tr, y_tr)
        return model

    # ── GPU detection ─────────────────────────────────────────────────────────

    _GPU_AVAILABLE: Optional[bool] = None  # class-level cache; None = not yet checked

    @classmethod
    def _has_gpu(cls) -> bool:
        """Return True if an NVIDIA GPU is accessible (result cached after first call)."""
        if cls._GPU_AVAILABLE is None:
            import subprocess
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, timeout=5,
                )
                cls._GPU_AVAILABLE = r.returncode == 0 and bool(r.stdout.strip())
            except Exception:
                cls._GPU_AVAILABLE = False
        return cls._GPU_AVAILABLE

    @classmethod
    def _gpu_params(cls, estimator_name: str) -> Dict:
        """Return GPU-specific constructor kwargs if a GPU is available, else {}."""
        if not cls._has_gpu():
            return {}
        if estimator_name in ("lgbm", "LightGBM"):
            return {"device": "gpu"}
        if estimator_name in ("xgboost", "XGBoost"):
            try:
                import xgboost as xgb
                # XGBoost 2.x unified `device` param; 1.x used `tree_method="gpu_hist"`
                if int(xgb.__version__.split(".")[0]) >= 2:
                    return {"device": "cuda"}
                return {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}
            except Exception:
                return {"device": "cuda"}
        if estimator_name in ("catboost", "CatBoost"):
            return {"task_type": "GPU"}
        return {}  # RF / ExtraTrees: no sklearn GPU backend

    # ── Dynamic time budget ───────────────────────────────────────────────────

    @staticmethod
    def _compute_time_budget(n_rows: int, n_cols: int) -> Tuple[int, int]:
        """Return (flaml_time_budget_s, optuna_timeout_s) scaled to dataset size.

        The factor (0 → 1) scales the user-configured `Config.FLAML_TIME_BUDGET`
        and `Config.OPTUNA_TIMEOUT` down for small data and up to the full budget
        for large data. Previous version had hard-coded 1000 s cap that silently
        dropped user-set budgets of 1800-3600 s.

        Factor tiers:
          Small  (≤ 90 k rows AND <100 cols)        → 0.10
          Medium (≤ 200 k rows AND ≤400 cols)       → 0.10 – 0.30
          Large  (above medium, capped at 1M × 2k)  → 0.30 – 1.00

        A 120 s floor protects against degenerate budgets when the user
        accidentally sets FLAML_TIME_BUDGET very low.
        """
        if n_rows <= 90_000 and n_cols < 100:
            factor = 0.10
        elif n_rows <= 200_000 and n_cols <= 400:
            row_ratio = max(0.0, (n_rows - 90_000) / (200_000 - 90_000))
            col_ratio = max(0.0, (n_cols - 100) / (400 - 100))
            factor = 0.10 + max(row_ratio, col_ratio) * (0.30 - 0.10)
        else:
            row_ratio = min(1.0, max(0.0, (n_rows - 200_000) / 800_000))
            col_ratio = min(1.0, max(0.0, (n_cols - 400) / 1_600))
            factor = 0.30 + max(row_ratio, col_ratio) * (1.00 - 0.30)
        flaml_t  = max(120, int(Config.FLAML_TIME_BUDGET * factor))
        optuna_t = max(120, int(Config.OPTUNA_TIMEOUT     * factor))
        return flaml_t, optuna_t

    # ── Step 1: Data split ────────────────────────────────────────────────────

    @staticmethod
    def _stratified_split(df: pd.DataFrame, target: str, test_size: float, rs: int):
        """Stratified split with automatic fallback to unstratified when a class is too small."""
        from sklearn.model_selection import train_test_split as _split
        try:
            return _split(df, test_size=test_size, stratify=df[target], random_state=rs)
        except ValueError:
            return _split(df, test_size=test_size, stratify=None, random_state=rs)

    def _resolve_period_freq(self, dt_series: pd.Series) -> Tuple[str, str]:
        """Resolve (pandas period freq, human unit) from temporal_freq.

        "weekly" → ("W-<ANCHOR>", "week"); the anchor is the snapshot/closing
        weekday — taken from week_closing_day, else auto-detected as the most
        common weekday present in date_col. Everything else → ("M", "month").
        Weekly is opt-in only (never auto-inferred) to avoid mis-detection.
        """
        mode = (self.temporal_freq or "auto").lower()
        if mode != "weekly":
            return "M", "month"
        valid = {"MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"}
        anchor = (self.week_closing_day or "").strip().upper()
        if anchor not in valid:
            wd = dt_series.dropna().dt.day_name().str[:3].str.upper()
            anchor = wd.mode().iloc[0] if not wd.empty else "SUN"
            self.logger.log(self.name, "Temporal freq",
                f"weekly cadence — week_closing_day auto-detected = {anchor} "
                "(most common weekday in date_col; set week_closing_day to override)")
        return f"W-{anchor}", "week"

    def _tool_split_data(
        self, df: pd.DataFrame, provided_oot: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        target = self.target_column
        rs = Config.RANDOM_STATE
        total = len(df)

        # ── Pre-split mode: reconstruct train/valid/oot from `_split_` marker ──
        # AutoMLPipeline._build_combined_input writes the marker whenever the
        # caller supplies valid_path or oot_path. Agents 1+2 protected the
        # column so it survived cleaning + feature engineering. Honour it
        # exactly. Three sub-cases by which marker values are present:
        #   {train, valid, oot} → use all three as marked.
        #   {train, valid}      → no OOT; use train/valid as marked.
        #   {train, oot}        → split train 80/20 into train+valid; oot as marked.
        if _SPLIT_MARKER in df.columns:
            marker = df[_SPLIT_MARKER].astype(str)
            feat_df = df.drop(columns=[_SPLIT_MARKER])
            empty = pd.DataFrame(columns=feat_df.columns)

            train_marked = feat_df[marker == "train"].reset_index(drop=True)
            valid_marked = feat_df[marker == "valid"].reset_index(drop=True)
            oot_marked   = feat_df[marker == "oot"].reset_index(drop=True)

            has_explicit_valid = len(valid_marked) > 0
            if has_explicit_valid:
                train_df = train_marked
                valid_df = valid_marked
                self.logger.log(self.name, "Split (pre-split marker)",
                    f"train={len(train_df)} | valid={len(valid_df)} | "
                    f"oot={len(oot_marked)} (all from marker)")
            else:
                # valid missing — auto-split 80/20 from the train portion
                train_df, valid_df = self._stratified_split(train_marked, target, 0.20, rs)
                train_df = train_df.reset_index(drop=True)
                valid_df = valid_df.reset_index(drop=True)
                self.logger.log(self.name, "Split (pre-split marker, auto-valid)",
                    f"train={len(train_df)} | valid={len(valid_df)} (auto 20% of train) | "
                    f"oot={len(oot_marked)} (from marker)")

            return {
                "train":          train_df,
                "valid_temporal": empty.copy(),
                "valid_random":   empty.copy(),
                "valid":          valid_df,
                "oot":            oot_marked,
                "test":           empty.copy(),
            }

        empty = pd.DataFrame(columns=df.columns)

        # ── Fast path: OOT already supplied — split pool 80/20 only ─────────
        if provided_oot is not None:
            oot_df = provided_oot.reset_index(drop=True)
            train_df, valid_df = self._stratified_split(df, target, 0.20, rs)
            self.logger.log(self.name, "Split (pre-OOT)",
                f"train={len(train_df)} | valid={len(valid_df)} | "
                f"oot={len(oot_df)} (provided externally)")
            return {
                "train":          train_df.reset_index(drop=True),
                "valid_temporal": empty.copy(),
                "valid_random":   empty.copy(),
                "valid":          valid_df.reset_index(drop=True),
                "oot":            oot_df,
                "test":           empty.copy(),
            }

        if self.date_col and self.date_col in df.columns:
            # Bucket each row into a period (month or week) as a standalone Series —
            # avoids df.copy() and the later .drop(columns=["_ym"]) which each
            # allocate a full copy of the frame. Frequency is resolved from
            # temporal_freq: monthly by default, weekly only when opted in.
            _dt = pd.to_datetime(df[self.date_col], errors="coerce")
            self.period_freq, self.period_unit = self._resolve_period_freq(_dt)
            ym  = _dt.dt.to_period(self.period_freq)

            nat_count = int(_dt.isna().sum())
            if nat_count:
                self.logger.log(self.name, "WARN",
                    f"{nat_count} rows have unparseable dates in '{self.date_col}' — "
                    "treated as non-OOT (assigned to train pool)")
            sorted_months = sorted(ym.dropna().unique())   # periods (month or week)
            valid_rows = total - nat_count

            # ── Detect temporal cadence on the FULL series (authoritative) ──────
            # "*_snapshot": exactly one distinct calendar date per period (e.g.
            # every row stamped 2023-01-31, or every Friday) → the model operates
            # at that granularity, so every temporal slice (OOT, valid_temporal,
            # stability) must move in whole-period steps. Otherwise "intra_month" /
            # "weekly" (multiple dates per period). Recorded so the report is
            # explicit about the granularity rather than a blind row-count split.
            n_dates_total = int(_dt.dropna().nunique())
            n_periods_total = len(sorted_months)
            is_snapshot = (n_dates_total == n_periods_total)
            if self.period_unit == "week":
                self.date_cadence = "weekly_snapshot" if is_snapshot else "weekly"
            else:
                self.date_cadence = "monthly_snapshot" if is_snapshot else "intra_month"
            self.temporal_meta = {
                "cadence": self.date_cadence,
                "period_unit": self.period_unit,
                "period_freq": self.period_freq,
                "date_col": self.date_col,
                "n_periods_total": n_periods_total,
                "n_distinct_dates": n_dates_total,
                "first_period": str(sorted_months[0]) if sorted_months else None,
                "last_period": str(sorted_months[-1]) if sorted_months else None,
            }
            if n_periods_total >= 2:
                runs_at = self.period_unit.upper()
                self.logger.log(self.name, "Temporal cadence",
                    f"{self.date_cadence} | {n_periods_total} {self.period_unit}s "
                    f"[{self.temporal_meta['first_period']}..{self.temporal_meta['last_period']}] "
                    f"| {n_dates_total} distinct dates"
                    + (f" — model runs {runs_at}LY; all temporal splits snap to whole {self.period_unit}s"
                       if is_snapshot else ""))

            # B2 fix: need >= 2 distinct periods to create a meaningful OOT set
            if len(sorted_months) < 2:
                self.logger.log(self.name, "WARN",
                    f"Only {len(sorted_months)} distinct {self.period_unit}(s) in '{self.date_col}' — "
                    "cannot create OOT split, falling back to simple split")
                # fall through to simple split below
                ym = None
            else:
                # Step 1: find minimum trailing months to reach OOT_MIN_RATIO (start from 1)
                n_oot = 1
                while n_oot < len(sorted_months):
                    if ym.isin(sorted_months[-n_oot:]).sum() / valid_rows >= Config.OOT_MIN_RATIO:
                        break
                    n_oot += 1

                # Step 2: floor — always include at least OOT_INIT_MONTHS
                n_oot = max(n_oot, Config.OOT_INIT_MONTHS)

                # Clamp to leave at least 1 month for pool
                n_oot = min(n_oot, len(sorted_months) - 1)

                # Step 3: ceiling — shrink if OOT exceeds OOT_MAX_RATIO
                oot_ratio = ym.isin(sorted_months[-n_oot:]).sum() / valid_rows
                if oot_ratio > Config.OOT_MAX_RATIO:
                    while n_oot > 1:
                        candidate_ratio = ym.isin(sorted_months[-(n_oot - 1):]).sum() / valid_rows
                        if candidate_ratio <= Config.OOT_MAX_RATIO:
                            n_oot -= 1
                            break
                        n_oot -= 1
                    oot_ratio = ym.isin(sorted_months[-n_oot:]).sum() / valid_rows
                    self.logger.log(self.name, "WARN",
                        f"OOT exceeded cap {Config.OOT_MAX_RATIO:.0%} — "
                        f"reduced to {n_oot} {self.period_unit}(s) = {oot_ratio:.1%}. "
                        f"Data may have coarse {self.period_unit} granularity or heavy recency bias.")

                mask_oot  = ym.isin(set(sorted_months[-n_oot:])).values
                mask_pool = ~mask_oot
                pool_size = int(mask_pool.sum())

                # B2 fix: guard against empty pool after OOT split
                if pool_size == 0:
                    self.logger.log(self.name, "WARN",
                        "Pool is empty after OOT split — falling back to simple split")
                    ym = None
                else:
                    # Materialise oot_df first (smaller subset)
                    oot_df = df[mask_oot].reset_index(drop=True)

                    # Valid-temporal: pool rows closest to the OOT boundary.
                    # Selection is snapped to WHOLE PERIOD boundaries (month or
                    # week), not a raw row-count slice. Without snapping, snapshot
                    # data (all rows in a period share the same date, e.g.
                    # 2023-01-31 or a Friday) would have valid_temporal and train
                    # contain rows from the SAME snapshot period — no real temporal
                    # separation. Whole-period snapping is safe for any cadence: it
                    # picks the minimum trailing pool periods whose combined row
                    # count covers n_temporal, keeping period boundaries intact.
                    valid_size = max(1, int(round(pool_size * Config.TRAIN_TEST_SPLIT_SIZE)))
                    n_temporal = max(1, int(round(valid_size * Config.VALID_TEMPORAL_RATIO)))

                    pool_ym_s   = ym[mask_pool]          # Series: pool rows → Period
                    pool_months = sorted(pool_ym_s.dropna().unique())

                    # Cadence already detected on the full series above. For a
                    # snapshot cadence, snapping valid_temporal to whole periods is
                    # what prevents train + valid_temporal sharing the same snapshot
                    # period (within-period leakage).
                    if self.date_cadence in ("monthly_snapshot", "weekly_snapshot"):
                        self.logger.log(self.name, "Split temporal",
                            f"{self.date_cadence} — snapping valid_temporal to "
                            f"whole {self.period_unit}s to avoid within-{self.period_unit} leakage")

                    n_val_months = 1
                    while n_val_months < len(pool_months):
                        covered = int(pool_ym_s.isin(pool_months[-n_val_months:]).sum())
                        if covered >= n_temporal:
                            break
                        n_val_months += 1
                    # Always leave at least 1 pool period for train
                    n_val_months = min(n_val_months, max(1, len(pool_months) - 1))

                    temporal_orig_idx = set(
                        pool_ym_s[pool_ym_s.isin(set(pool_months[-n_val_months:]))].index
                    )
                    self.logger.log(self.name, "Split temporal",
                        f"valid_temporal = {n_val_months} whole {self.period_unit}(s) | "
                        f"rows={len(temporal_orig_idx)} | target_rows={n_temporal}")
                    del _dt, ym  # free temporary Series before creating large DataFrames

                    mask_temp_orig = pd.Series(False, index=df.index)
                    mask_temp_orig[list(temporal_orig_idx)] = True

                    # Materialise each split from the original df using boolean masks
                    valid_temp_df = df[mask_temp_orig.values].reset_index(drop=True)
                    mask_remain   = mask_pool & ~mask_temp_orig.values
                    remain_df     = df[mask_remain].reset_index(drop=True)

                    # Valid-random: stratified from remaining pool (B3 fix: fallback)
                    n_rand     = valid_size - len(valid_temp_df)
                    rand_ratio = n_rand / len(remain_df) if len(remain_df) > 0 else 0.0
                    if 0 < rand_ratio < 1:
                        train_df, valid_rand_df = self._stratified_split(remain_df, target, rand_ratio, rs)
                        train_df      = train_df.reset_index(drop=True)
                        valid_rand_df = valid_rand_df.reset_index(drop=True)
                    else:
                        train_df      = remain_df
                        valid_rand_df = empty.copy()

                    valid_df = pd.concat([valid_temp_df, valid_rand_df], ignore_index=True)

                    # Degenerate-split warning — usually means date_col is a
                    # mislabelled feature (e.g. days_birth coerced as
                    # nanoseconds-since-epoch). The downstream eval_set / CV
                    # paths will fall back to `valid` or skip ES entirely,
                    # but the user almost certainly wants to know.
                    target_in = lambda d: target in d.columns and len(d) > 0
                    bad_temp = (target_in(valid_temp_df)
                                and pd.to_numeric(valid_temp_df[target], errors="coerce").nunique() < 2)
                    bad_oot  = (target_in(oot_df)
                                and pd.to_numeric(oot_df[target], errors="coerce").nunique() < 2)
                    if bad_temp or bad_oot or len(valid_temp_df) < 10 or len(oot_df) < 10:
                        self.logger.log(self.name, "SPLIT WARN",
                            f"Degenerate temporal split — valid_temp={len(valid_temp_df)} rows "
                            f"(bad_classes={bad_temp}), oot={len(oot_df)} rows "
                            f"(bad_classes={bad_oot}). date_col='{self.date_col}' may not be a "
                            "real snapshot column; downstream will fall back to `valid` for ES.")

                    # Record how many whole periods each temporal split spans so
                    # the report states the cadence (monthly/weekly) explicitly.
                    self.temporal_meta.update({
                        "oot_periods": int(n_oot),
                        "valid_temporal_periods": int(n_val_months),
                        "train_periods": int(len(pool_months) - n_val_months),
                    })

                    self.logger.log(self.name, "Split (OOT)",
                        f"oot_{self.period_unit}s={n_oot} | oot_ratio={len(oot_df)/total:.1%} | "
                        f"train={len(train_df)} | valid_temp={len(valid_temp_df)} | "
                        f"valid_rand={len(valid_rand_df)} | oot={len(oot_df)}")
                    return {
                        "train": train_df, "valid_temporal": valid_temp_df,
                        "valid_random": valid_rand_df, "valid": valid_df,
                        "oot": oot_df, "test": empty.copy(),
                    }

        # Fallback: no date col or fell through from OOT path. No temporal OOT was
        # produced, so the splits are NOT month-based — mark cadence accordingly
        # (overrides any cadence detected before the fall-through).
        self.temporal_meta = {"cadence": "non_temporal"}
        self.date_cadence = "non_temporal"
        # 3-way stratified split: 60% train / 20% valid / 20% test (no temporal OOT)
        train_valid_df, test_df = self._stratified_split(df, target, 0.20, rs)
        train_df, valid_df      = self._stratified_split(train_valid_df, target, 0.25, rs)
        self.logger.log(self.name, "Split (simple)",
            f"train={len(train_df)} | valid={len(valid_df)} | test={len(test_df)} | no OOT")
        return {
            "train":          train_df.reset_index(drop=True),
            "valid_temporal": empty.copy(),
            "valid_random":   empty.copy(),
            "valid":          valid_df.reset_index(drop=True),
            "oot":            empty.copy(),
            "test":           test_df.reset_index(drop=True),
        }

    # ── Step 2: FLAML AutoML ──────────────────────────────────────────────────

    def _tool_run_flaml(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_valid: pd.DataFrame, y_valid: pd.Series,
        time_budget: int = None,
    ) -> Tuple[str, Dict]:
        try:
            from flaml import AutoML
        except ImportError:
            self.logger.log(self.name, "FLAML", "flaml not installed → fallback to lgbm. Run: pip install flaml")
            return "lgbm", {}

        automl = AutoML()
        estimators = [e.strip() for e in Config.FLAML_ESTIMATORS.split(",")]
        has_valid = len(X_valid) > 0
        settings: Dict[str, Any] = {
            "time_budget": time_budget if time_budget is not None else Config.FLAML_TIME_BUDGET,
            "metric": "roc_auc",
            "task": "classification",
            "estimator_list": estimators,
            "seed": Config.RANDOM_STATE,
            "verbose": 0,
            "log_training_metric": False,
        }
        if has_valid:
            # Custom validation data requires eval_method='holdout', not 'cv'
            settings["eval_method"] = "holdout"
            settings["X_val"] = X_valid
            settings["y_val"] = y_valid
        else:
            settings["eval_method"] = "cv"
            settings["n_splits"] = Config.FLAML_N_SPLITS

        # FLAML internally calls X.copy() for pandas block consolidation, which can OOM
        # on large DataFrames. Sampling to FLAML_MAX_ROWS keeps the copy manageable;
        # FLAML's job here is model-type selection, not final accuracy.
        if len(X_train) > Config.FLAML_MAX_ROWS:
            sample_idx = np.random.RandomState(Config.RANDOM_STATE).choice(
                len(X_train), Config.FLAML_MAX_ROWS, replace=False
            )
            X_fit = X_train.iloc[sample_idx]
            y_fit = y_train.iloc[sample_idx]
        else:
            X_fit = X_train
            y_fit = y_train
        automl.fit(X_fit, y_fit, **settings)

        best_estimator = automl.best_estimator
        best_config = automl.best_config
        valid_auc = float("nan")
        if len(X_valid) > 0:
            try:
                valid_auc = roc_auc_score(y_valid, automl.predict_proba(X_valid)[:, 1])
            except Exception:
                pass

        self.logger.log(self.name, "FLAML",
            f"best={best_estimator} | CV_loss={automl.best_loss:.4f} | valid_auc={valid_auc:.4f}")
        return best_estimator, best_config

    # ── Step 3: Optuna fine-tuning ────────────────────────────────────────────

    def _build_optuna_params(
        self,
        estimator_name: str,
        trial,
        pos_weight_range: Optional[Tuple[float, float]] = None,
    ) -> Dict:
        """Build Optuna search-space params for one estimator.

        When `pos_weight_range=(lo, hi)` is supplied (set by `_tool_run_optuna`
        on imbalanced data), xgb/catboost get `scale_pos_weight` tuned in
        [lo, hi] on log scale — this lets Optuna pick the imbalance correction
        instead of using a static sqrt(ratio) / ratio heuristic. Empirically
        +0.3-0.6% AUC on credit_risk / fraud.

        lgbm / rf / extra_tree don't need it — their class_weight dict is
        applied separately at fit time.
        """
        gpu = self._gpu_params(estimator_name)
        if estimator_name in ("lgbm", "LightGBM"):
            # DART (Dropouts meet Multiple Additive Regression Trees) — dropout
            # applied to existing trees during boosting. Slower (3-5x) but
            # often +0.3-1% AUC on credit data because it regularises tail
            # bins where gbdt overfits. ES is incompatible with DART (it
            # rebuilds dropped trees each round) — handled in _fit_with_early_stopping.
            boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])
            params = {
                "boosting_type": boosting_type,
                "n_estimators": trial.suggest_int("n_estimators", Config.N_ESTIMATORS_MIN, Config.N_ESTIMATORS_MAX, step=50),
                "learning_rate": trial.suggest_float("learning_rate", Config.LR_MIN, Config.LR_MAX, log=True),
                "num_leaves": trial.suggest_int("num_leaves", Config.NUM_LEAVES_MIN, Config.NUM_LEAVES_MAX),
                "max_depth": trial.suggest_int("max_depth", Config.MAX_DEPTH_MIN, Config.MAX_DEPTH_MAX),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
                "subsample": trial.suggest_float("subsample", Config.SUBSAMPLE_MIN, Config.SUBSAMPLE_MAX),
                "colsample_bytree": trial.suggest_float("colsample_bytree", Config.SUBSAMPLE_MIN, Config.SUBSAMPLE_MAX),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": Config.RANDOM_STATE, "n_jobs": -1, "verbose": -1,
                **gpu,
            }
            if boosting_type == "dart":
                # DART-specific drop hyperparams. Bounds picked from LightGBM
                # paper § 4 + bank credit-risk benchmarking sweet spot.
                params["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.3)
                params["max_drop"] = trial.suggest_int("max_drop", 30, 100)
                params["skip_drop"] = trial.suggest_float("skip_drop", 0.3, 0.7)
            return params
        if estimator_name in ("xgboost", "XGBoost"):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", Config.N_ESTIMATORS_MIN, Config.N_ESTIMATORS_MAX, step=50),
                "learning_rate": trial.suggest_float("learning_rate", Config.LR_MIN, Config.LR_MAX, log=True),
                "max_depth": trial.suggest_int("max_depth", Config.MAX_DEPTH_MIN, Config.MAX_DEPTH_MAX),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
                "subsample": trial.suggest_float("subsample", Config.SUBSAMPLE_MIN, Config.SUBSAMPLE_MAX),
                "colsample_bytree": trial.suggest_float("colsample_bytree", Config.SUBSAMPLE_MIN, Config.SUBSAMPLE_MAX),
                "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "eval_metric": "logloss",
                "random_state": Config.RANDOM_STATE, "n_jobs": -1, "verbosity": 0,
                **gpu,
            }
            if pos_weight_range is not None:
                lo, hi = pos_weight_range
                params["scale_pos_weight"] = trial.suggest_float(
                    "scale_pos_weight", max(1.0, lo), max(1.01, hi), log=True
                )
            return params
        if estimator_name in ("catboost", "CatBoost"):
            params = {
                "iterations": trial.suggest_int("iterations", Config.N_ESTIMATORS_MIN, Config.N_ESTIMATORS_MAX, step=50),
                "learning_rate": trial.suggest_float("learning_rate", Config.LR_MIN, Config.LR_MAX, log=True),
                "depth": trial.suggest_int("depth", Config.CB_DEPTH_MIN, Config.CB_DEPTH_MAX),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
                "random_seed": Config.RANDOM_STATE, "verbose": 0,
                **gpu,
            }
            if pos_weight_range is not None:
                lo, hi = pos_weight_range
                params["scale_pos_weight"] = trial.suggest_float(
                    "scale_pos_weight", max(1.0, lo), max(1.01, hi), log=True
                )
            return params
        # rf / extra_tree / RandomForest / ExtraTrees — no GPU support
        return {
            "n_estimators": trial.suggest_int("n_estimators", Config.N_ESTIMATORS_MIN, Config.N_ESTIMATORS_MAX, step=50),
            "max_depth": trial.suggest_int("max_depth", Config.MAX_DEPTH_MIN, Config.MAX_DEPTH_MAX),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
            "random_state": Config.RANDOM_STATE, "n_jobs": -1,
        }

    def _tool_run_optuna(
        self,
        estimator_name: str, base_params: Dict,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_valid: pd.DataFrame, y_valid: pd.Series,
        timeout: int = None,
    ) -> Dict:
        try:
            import optuna
            from optuna.samplers import TPESampler
            from optuna.trial import FixedTrial
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            self.logger.log(self.name, "Optuna", "optuna not installed → skip. Run: pip install optuna")
            return base_params

        if len(X_valid) == 0:
            self.logger.log(self.name, "Optuna", "No validation set → skip")
            return base_params

        ModelClass = self._get_model_class(estimator_name)
        _estimator_name = estimator_name
        _self = self
        # Auto class weighting + early stopping: both safe defaults that boost
        # AUC on imbalanced data without changing the search space dimensions.
        class_weight_params = self._compute_class_weight_params(y_train, estimator_name)
        if class_weight_params:
            self.logger.log(self.name, "Optuna class weight",
                f"applying {class_weight_params} (auto from train imbalance)")

        # For xgb/cb on imbalanced data, push scale_pos_weight into Optuna's
        # search space instead of fixing it via class_weight_params. Optuna
        # finds the AUC-optimal weight; static sqrt/balanced often misses it
        # by 1-3 ratio points.
        pos_weight_range: Optional[Tuple[float, float]] = None
        if estimator_name in ("xgboost", "XGBoost", "catboost", "CatBoost"):
            ratio = self._compute_imbalance_ratio(y_train)
            if ratio >= Config.AUTO_CLASS_WEIGHT_THRESHOLD:
                # Search log-uniform in [1, ratio]. Optuna typically converges
                # near sqrt(ratio) for severe imbalance but can pick higher
                # when valid set has different distribution.
                pos_weight_range = (1.0, float(ratio))
                # Static scale_pos_weight from class_weight_params would override
                # Optuna's tuned value — strip it so search wins.
                class_weight_params.pop("scale_pos_weight", None)
                self.logger.log(self.name, "Optuna scale_pos_weight",
                    f"searching in [1.0, {ratio:.1f}] (log-uniform) — overrides static {Config.CLASS_WEIGHT_STRATEGY}")

        def objective(trial):
            params = _self._build_optuna_params(_estimator_name, trial, pos_weight_range)
            params.update(class_weight_params)
            model = ModelClass(**params)
            _self._fit_with_early_stopping(
                model, _estimator_name, X_train, y_train, X_valid, y_valid
            )
            return roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=Config.RANDOM_STATE),
        )
        study.optimize(
            objective,
            n_trials=Config.OPTUNA_N_TRIALS,
            timeout=timeout if timeout is not None else Config.OPTUNA_TIMEOUT,
            show_progress_bar=False,
        )
        # Replay _build_optuna_params with FixedTrial to recover fixed params
        # (random_state, n_jobs, verbose, etc.) that study.best_params doesn't contain
        try:
            full_best_params = self._build_optuna_params(
                estimator_name, FixedTrial(study.best_params), pos_weight_range
            )
            self.logger.log(self.name, "Optuna",
                f"best_AUC={study.best_value:.4f} | params={json.dumps(study.best_params)[:200]}")
            return full_best_params
        except Exception as e:
            self.logger.log(self.name, "Optuna", f"No completed trials ({e}) → fallback to base_params")
            return base_params

    # ── Step 4: RFE ───────────────────────────────────────────────────────────

    def _tool_run_rfe(
        self,
        estimator_name: str, best_params: Dict,
        X_train: pd.DataFrame, y_train: pd.Series,
    ) -> List[str]:
        from sklearn.feature_selection import RFE, RFECV
        from sklearn.model_selection import StratifiedKFold

        ModelClass = self._get_model_class(estimator_name)
        # RFE keeps RFE_TARGET_FEATURES (not MAX_FINAL_FEATURES) so PSI / Stability /
        # SHAP+PSI prune downstream have more candidates to work with. Floor at
        # MAX_FINAL_FEATURES so RFE never drops below the final cap. Cap at n_cols
        # for narrow datasets where the requested target exceeds available features.
        n_target = min(
            max(Config.RFE_TARGET_FEATURES, Config.MAX_FINAL_FEATURES),
            X_train.shape[1],
        )
        n_est_key = self._get_n_estimators_key(estimator_name)
        base_model = ModelClass(**{**self._safe_params(estimator_name, best_params), n_est_key: Config.RFE_N_ESTIMATORS, **self._gpu_params(estimator_name)})

        if Config.ENABLE_RFECV:
            cv = StratifiedKFold(n_splits=Config.RFE_CV_SPLITS, shuffle=True,
                                 random_state=Config.RANDOM_STATE)
            selector = RFECV(
                estimator=base_model, step=Config.RFE_STEP, cv=cv,
                scoring="roc_auc", min_features_to_select=n_target, n_jobs=-1,
            )
        else:
            selector = RFE(
                estimator=base_model,
                n_features_to_select=n_target,
                step=Config.RFE_STEP,
            )

        selector.fit(X_train, y_train)
        selected = list(X_train.columns[selector.support_])
        self.logger.log(self.name, "RFE", f"{X_train.shape[1]} → {len(selected)} features")
        return selected

    # ── Step 5: PSI filter ────────────────────────────────────────────────────

    @staticmethod
    def _calc_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
        eps = 1e-8
        try:
            cuts = pd.qcut(expected, q=bins, duplicates="drop", retbins=True)[1]
            cuts[0], cuts[-1] = -np.inf, np.inf
            e_pct = pd.cut(expected, bins=cuts).value_counts(normalize=True).sort_index() + eps
            a_pct = pd.cut(actual, bins=cuts).value_counts(normalize=True).sort_index() + eps
            e_pct, a_pct = e_pct.align(a_pct, fill_value=eps)
            return float(((a_pct - e_pct) * np.log(a_pct / e_pct)).sum())
        except Exception:
            return 999.0

    def _tool_run_psi_filter(
        self,
        feature_cols: List[str],
        train_df: pd.DataFrame,
        oot_df: pd.DataFrame,
    ) -> Tuple[List[str], pd.DataFrame]:
        if len(oot_df) == 0:
            self.logger.log(self.name, "PSI", "No OOT data → skip")
            return feature_cols, pd.DataFrame()

        records = []
        for col in feature_cols:
            if col not in train_df.columns or col not in oot_df.columns:
                continue
            tr_col = train_df[col]
            ot_col = oot_df[col]
            # PSI is only meaningful for numeric features; skip object/category columns
            if tr_col.dtype == object or hasattr(tr_col.dtype, "categories"):
                continue
            psi_val = self._calc_psi(tr_col.dropna(), ot_col.dropna(), Config.PSI_BINS)
            records.append({"feature": col, "psi": round(psi_val, 4)})

        if not records:
            self.logger.log(self.name, "PSI", "No features found in both train and OOT → skip")
            return feature_cols, pd.DataFrame()

        psi_df = pd.DataFrame(records).sort_values("psi", ascending=False)
        psi_df["flag"] = psi_df["psi"].apply(
            lambda x: f"DROP (PSI={x:.3f})" if x > Config.PSI_THRESHOLD else "KEEP"
        )
        kept = psi_df.loc[psi_df["flag"] == "KEEP", "feature"].tolist()
        dropped_n = (psi_df["flag"] != "KEEP").sum()
        self.logger.log(self.name, "PSI", f"KEEP={len(kept)} | DROP={dropped_n}")
        return kept, psi_df

    # ── Step 6: Stability check ───────────────────────────────────────────────

    @staticmethod
    def _calc_gini(df: pd.DataFrame, feature: str, target: str) -> float:
        try:
            sub = df[[feature, target]].dropna()
            if sub[target].nunique() < 2 or sub[feature].nunique() < 2:
                return 0.0
            return abs(2 * roc_auc_score(sub[target], sub[feature]) - 1)
        except Exception:
            return 0.0

    def _tool_run_stability_check(
        self,
        feature_cols: List[str],
        train_df: pd.DataFrame,
    ) -> Tuple[List[str], pd.DataFrame]:
        if not self.date_col or self.date_col not in train_df.columns:
            self.logger.log(self.name, "Stability", "No date col → skip")
            return feature_cols, pd.DataFrame()

        tmp = train_df.copy()
        tmp[self.date_col] = pd.to_datetime(tmp[self.date_col], errors="coerce")
        # In pre-split mode _tool_split_data's date branch is skipped, so
        # period_freq may still be the default "M" — honour an explicit weekly
        # request here too so split-mode weekly models get weekly stability.
        if self.temporal_freq == "weekly" and self.period_freq == "M":
            self.period_freq, self.period_unit = self._resolve_period_freq(tmp[self.date_col])
        # Group Gini by the SAME period as the split (month or week) so a weekly
        # model gets weekly stability; STABILITY_MIN_MONTHS is read as a min count
        # of periods regardless of unit.
        tmp["_ym"] = tmp[self.date_col].dt.to_period(self.period_freq)
        periods = sorted(tmp["_ym"].dropna().unique())

        if len(periods) < Config.STABILITY_MIN_MONTHS:
            self.logger.log(self.name, "Stability",
                f"Only {len(periods)} {self.period_unit}s < min={Config.STABILITY_MIN_MONTHS} → skip")
            return feature_cols, pd.DataFrame()

        records = []
        for col in feature_cols:
            ginis = [self._calc_gini(tmp[tmp["_ym"] == p], col, self.target_column) for p in periods]
            mean_g = float(np.mean(ginis))
            std_g = float(np.std(ginis))
            flag = "KEEP"
            if std_g > Config.STABILITY_GINI_STD_THRESHOLD:
                flag = f"DROP (std_gini={std_g:.3f})"
            elif mean_g < Config.STABILITY_MIN_GINI:
                flag = f"DROP (mean_gini={mean_g:.3f})"
            records.append({
                "feature": col,
                "mean_gini": round(mean_g, 4),
                "std_gini": round(std_g, 4),
                "flag": flag,
            })

        if not records:
            return feature_cols, pd.DataFrame()
        stab_df = pd.DataFrame(records).sort_values("mean_gini", ascending=False)
        kept = stab_df.loc[stab_df["flag"] == "KEEP", "feature"].tolist()
        dropped_n = (stab_df["flag"] != "KEEP").sum()
        self.logger.log(self.name, "Stability", f"KEEP={len(kept)} | DROP={dropped_n}")
        return kept, stab_df

    # ── Step 7: Iterative SHAP + PSI feature pruning ─────────────────────────

    def _tool_shap_psi_prune(
        self,
        estimator_name: str,
        best_params: Dict,
        feature_cols: List[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        psi_df: pd.DataFrame,
    ) -> Tuple[List[str], pd.DataFrame]:
        """Iteratively remove features ranked worst on combined SHAP importance + PSI drift
        until validation AUC no longer improves.

        Returns (best_features, pruning_log_df).
        """
        ModelClass = self._get_model_class(estimator_name)
        n_est_key = self._get_n_estimators_key(estimator_name)
        gpu = self._gpu_params(estimator_name)
        min_features = max(
            Config.SHAP_PSI_MIN_FEATURES_FLOOR,
            int(Config.MAX_FINAL_FEATURES * Config.SHAP_PSI_MIN_FEATURES_RATIO),
        )
        max_no_improve = Config.SHAP_PSI_MAX_NO_IMPROVE
        # Policy: 'auc_first' = early-stop fires immediately when AUC stops
        # improving (final count may exceed MAX_FINAL_FEATURES);
        # 'cap_first' = ignore early-stop until we cross MAX_FINAL_FEATURES,
        # then re-enable it for fine-tuning down toward the floor.
        cap_policy = Config.FEATURE_CAP_POLICY
        if cap_policy not in ("auc_first", "cap_first"):
            self.logger.log(self.name, "SHAP+PSI WARN",
                f"Unknown FEATURE_CAP_POLICY '{cap_policy}' — defaulting to 'auc_first'")
            cap_policy = "auc_first"
        feature_cap = Config.MAX_FINAL_FEATURES

        has_valid = len(X_valid) > 0 and y_valid.nunique() >= 2
        if not has_valid:
            self.logger.log(self.name, "SHAP+PSI Prune", "No validation set → skip pruning")
            return list(feature_cols), pd.DataFrame()

        # PSI lookup: feature → psi_value (0.0 if not available)
        psi_lookup: Dict[str, float] = {}
        if not psi_df.empty and "feature" in psi_df.columns and "psi" in psi_df.columns:
            psi_lookup = dict(zip(psi_df["feature"], psi_df["psi"]))

        # columns guaranteed to exist in both matrices (X_train and X_valid were built
        # from the same feat_avail set, so they should always match — guard for safety)
        class_weight_params = self._compute_class_weight_params(y_train, estimator_name)
        # Optuna may have tuned scale_pos_weight already (xgb/cb path). Don't
        # let the static heuristic override that — keep best_params' choice.
        if "scale_pos_weight" in best_params:
            class_weight_params.pop("scale_pos_weight", None)
        def _fit_model(features: List[str]):
            avail = [c for c in features if c in X_train.columns and c in X_valid.columns]
            if not avail:
                return None, avail
            m = ModelClass(**{
                **self._safe_params(estimator_name, best_params),
                n_est_key: Config.SHAP_N_ESTIMATORS,
                **class_weight_params,
                **gpu,
            })
            self._fit_with_early_stopping(
                m, estimator_name,
                X_train[avail], y_train,
                X_valid[avail] if len(X_valid) > 0 else None,
                y_valid if len(X_valid) > 0 else None,
            )
            return m, avail

        def _fast_auc(features: List[str]) -> float:
            m, avail = _fit_model(features)
            if m is None:
                return 0.0
            try:
                return float(roc_auc_score(y_valid, m.predict_proba(X_valid[avail])[:, 1]))
            except Exception:
                return 0.0

        def _shap_importance(features: List[str]) -> pd.Series:
            m, avail = _fit_model(features)
            if m is None:
                return pd.Series(dtype=float)
            try:
                import shap
                sample_n = min(Config.SHAP_SAMPLE_SIZE, len(X_train))
                rng = np.random.default_rng(Config.RANDOM_STATE)
                idx = rng.choice(len(X_train), sample_n, replace=False)
                explainer = shap.TreeExplainer(m)
                sv = explainer.shap_values(X_train[avail].iloc[idx])
                # shap may return: list[array], 2D array, or 3D array [n, f, classes]
                if isinstance(sv, list):
                    sv = sv[1]
                elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                    sv = sv[:, :, 1]
                return pd.Series(np.abs(sv).mean(axis=0), index=avail)
            except Exception:
                imp = getattr(m, "feature_importances_", None)
                if imp is not None:
                    return pd.Series(imp, index=avail)
                return pd.Series(1.0, index=pd.Index(avail))

        current_features = list(feature_cols)
        best_features = list(current_features)
        best_auc = _fast_auc(current_features)
        no_improve_streak = 0
        records = []

        self.logger.log(self.name, "SHAP+PSI Prune",
            f"start={len(current_features)} | baseline_valid_auc={best_auc:.4f} | "
            f"min_features={min_features} | cap={feature_cap} | "
            f"policy={cap_policy} | psi_available={bool(psi_lookup)}")

        step = 0
        while len(current_features) > min_features:
            step += 1
            shap_imp = _shap_importance(current_features)

            if shap_imp.empty:
                self.logger.log(self.name, "SHAP+PSI Prune", "SHAP importance empty — stopping pruning")
                break

            # Normalize SHAP: 0 = least important → candidate to remove
            s_min, s_max = shap_imp.min(), shap_imp.max()
            shap_norm = (shap_imp - s_min) / (s_max - s_min + 1e-12)

            # Normalize PSI: 0 = stable, 1 = most drifting → candidate to remove
            psi_s = pd.Series({f: psi_lookup.get(f, 0.0) for f in current_features})
            p_min, p_max = psi_s.min(), psi_s.max()
            psi_norm = (psi_s - p_min) / (p_max - p_min + 1e-12)

            # Combined removal score — high = remove first
            # If no PSI data available fall back to pure SHAP
            psi_weight = 0.5 if psi_lookup else 0.0
            removal_score = (1.0 - psi_weight) * (1 - shap_norm) + psi_weight * psi_norm
            worst_feat = removal_score.idxmax()

            candidate = [f for f in current_features if f != worst_feat]
            candidate_auc = _fast_auc(candidate)
            improved = candidate_auc >= best_auc

            records.append({
                "step": step,
                "removed_feature": worst_feat,
                "n_remaining": len(candidate),
                "candidate_auc": round(candidate_auc, 4),
                "prev_best_auc": round(best_auc, 4),
                "improved": improved,
                "shap_importance": round(float(shap_imp.get(worst_feat, 0.0)), 6),
                "psi": round(float(psi_s.get(worst_feat, 0.0)), 4),
                "removal_score": round(float(removal_score.get(worst_feat, 0.0)), 4),
            })

            current_features = candidate  # always advance current (greedy exploration)

            # Forced advance: in 'cap_first' policy, while still above OR at the
            # cap we override the AUC-improvement rule to guarantee the cap is
            # met. Using >= (not >) ensures the boundary step (cap+1 → cap) also
            # snapshots, so best_features ends ≤ feature_cap exactly. Once we
            # drop below the cap, the normal AUC-driven snapshot logic resumes
            # and best_features can only improve from that point on.
            force_advance = (cap_policy == "cap_first"
                             and len(current_features) >= feature_cap)

            if improved:
                best_features = list(current_features)
                best_auc = candidate_auc
                no_improve_streak = 0
                self.logger.log(self.name, "SHAP+PSI",
                    f"step={step} removed '{worst_feat}' | "
                    f"n={len(current_features)} | auc={candidate_auc:.4f}")
            elif force_advance:
                # AUC tụt nhưng đang trên cap → vẫn snapshot để cap chắc chắn đạt.
                # Reset streak: khi cross cap vào Phase 2, early-stop có ngân sách full.
                best_features = list(current_features)
                no_improve_streak = 0
                self.logger.log(self.name, "SHAP+PSI",
                    f"step={step} forced remove '{worst_feat}' "
                    f"(above cap {feature_cap}) → n={len(current_features)} | "
                    f"auc={candidate_auc:.4f} (Δ={candidate_auc - best_auc:+.4f})")
            else:
                no_improve_streak += 1
                self.logger.log(self.name, "SHAP+PSI",
                    f"step={step} remove '{worst_feat}' → auc dropped "
                    f"{best_auc:.4f}→{candidate_auc:.4f} "
                    f"[no_improve={no_improve_streak}/{max_no_improve}]")
                if no_improve_streak >= max_no_improve:
                    break

        self.logger.log(self.name, "SHAP+PSI Done",
            f"{len(feature_cols)} → {len(best_features)} features | "
            f"best_valid_auc={best_auc:.4f} | steps={step}")

        return best_features, pd.DataFrame(records)

    # ── Step 8: Final model training & evaluation ─────────────────────────────

    def _tool_train_final_model(
        self,
        estimator_name: str,
        best_params: Dict,
        feature_cols: List[str],
    ) -> Dict[str, Any]:
        """Train the production-ready final model.

        Refit-on-train+valid flow (banking AUC stack). Feature selection and
        hyperparameter tuning happen earlier on train (with valid as the ES /
        scoring holdout); here the DEPLOYED model is refit on the full in-time
        data so it sees every available row before facing OOT/test.

          0. Merge — X_final = train + valid_temporal + valid_random (or `valid`
             in simple mode). OOT (and `test` in simple mode) are held out.
          1. K-fold CV on X_final — each fold fits with early stopping to give a
             stable best_iteration (median across folds) AND collects out-of-fold
             raw probabilities. cv_auc_mean (OOF) is the honest in-time estimate.
          2. Multi-seed bagging — refit `Config.MULTI_SEED_N` copies of the
             Optuna-best config on the FULL X_final with the tree count fixed to
             best_iteration (no ES, since valid is now in-sample). Mean of base
             raw probas. MULTI_SEED_N=1 reverts to single-model behaviour.
          3. Leakage-free calibration — isotonic/sigmoid fit on the pooled CV
             out-of-fold predictions (CalibratedClassifierCV-style). Required for
             IFRS9 / Basel PD — AUC neutral, big Brier improvement.
          4. Per-split AUC + Brier eval — valid_* are in-sample after the merge
             (reported for transparency); OOT/test are the clean final holdouts.

        State written:
          self.model            — primary single-seed fit (used by SHAP)
          self.ensemble_models  — list[BaseEstimator] of all fitted seeds
          self.calibrator       — IsotonicRegression / _SigmoidCalibrator or None
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import brier_score_loss

        target = self.target_column
        ModelClass = self._get_model_class(estimator_name)

        def _prep(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
            avail = [c for c in feature_cols if c in df.columns]
            return self._transform_X(df[avail]), pd.to_numeric(df[target], errors="coerce")

        # ── Build the FINAL training set = train + ALL valid ─────────────
        # Banking rule: once features + hyperparams are locked during model
        # SELECTION, the DEPLOYED model is refit on every in-time row
        # (train + valid_temporal + valid_random, or `valid` in simple mode).
        # OOT (and `test` in simple mode) are NEVER merged in → they stay clean
        # final holdouts for the out-of-time / hold-out acceptance report.
        train_df_only = self.splits["train"]
        final_parts   = [train_df_only]
        merged_labels: List[str] = []
        for s in ("valid_temporal", "valid_random"):
            sdf = self.splits.get(s, pd.DataFrame())
            if len(sdf) > 0:
                final_parts.append(sdf); merged_labels.append(s)
        if len(final_parts) == 1:                       # simple (non-temporal) mode
            vdf = self.splits.get("valid", pd.DataFrame())
            if len(vdf) > 0:
                final_parts.append(vdf); merged_labels.append("valid")
        final_df = (pd.concat(final_parts, ignore_index=True)
                    if len(final_parts) > 1 else train_df_only)
        X_fin, y_fin = _prep(final_df)
        self.logger.log(self.name, "Final training set",
            f"train+valid merged → rows={len(X_fin)} "
            f"(train={len(train_df_only)} + [{', '.join(merged_labels) or 'none'}]); "
            "OOT/test held out as final acceptance set")

        # Auto class weighting from the MERGED labels — but respect Optuna's
        # tuned scale_pos_weight if it's already in best_params (xgb/cb path).
        class_weight_params = self._compute_class_weight_params(y_fin, estimator_name)
        if "scale_pos_weight" in best_params:
            class_weight_params.pop("scale_pos_weight", None)
        if class_weight_params:
            self.logger.log(self.name, "Final class weight",
                f"applying {class_weight_params}")

        seed_param = self._random_state_key(estimator_name)
        n_est_key  = self._get_n_estimators_key(estimator_name)

        def _base_params(seed: int) -> Dict[str, Any]:
            return {
                **self._safe_params(estimator_name, best_params),
                **class_weight_params,
                **self._gpu_params(estimator_name),
                seed_param: seed,
            }

        # ── 1. K-fold CV on train+valid → stable best_iteration + OOF preds ──
        # One CV pass over the merged set yields BOTH:
        #   (a) a robust early-stopping tree count (median best_iteration across
        #       folds) to refit with on the full merged set, and
        #   (b) out-of-fold raw probabilities for LEAKAGE-FREE calibration (each
        #       row scored by a fold model that never trained on it).
        # Preferred over a single ES holdout because `valid` is now inside the
        # training data — there is no clean held split left to early-stop on.
        oof_raw   = np.full(len(y_fin), np.nan)
        best_iters: List[int] = []
        fold_aucs:  List[float] = []
        cv_ok = False
        try:
            n_splits = max(2, min(Config.CV_N_SPLITS, int(y_fin.value_counts().min())))
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=Config.RANDOM_STATE)
            for f_tr, f_va in skf.split(X_fin, y_fin):
                fm = ModelClass(**_base_params(Config.RANDOM_STATE))
                self._fit_with_early_stopping(
                    fm, estimator_name,
                    X_fin.iloc[f_tr], y_fin.iloc[f_tr],
                    X_fin.iloc[f_va], y_fin.iloc[f_va])
                bi = self._extract_best_iteration(fm, estimator_name)
                if bi:
                    best_iters.append(bi)
                p = fm.predict_proba(X_fin.iloc[f_va])[:, 1]
                oof_raw[f_va] = p
                if y_fin.iloc[f_va].nunique() >= 2:
                    fold_aucs.append(float(roc_auc_score(y_fin.iloc[f_va], p)))
            cv_ok = True
        except Exception as e:
            self.logger.log(self.name, "CV WARN",
                f"K-fold on train+valid failed ({e}) — refit without fixed "
                "iteration and skip OOF calibration")

        best_iteration = int(np.median(best_iters)) if best_iters else None
        self.logger.log(self.name, "Best iteration (CV on train+valid)",
            f"folds={best_iters or 'n/a (no ES)'} → median={best_iteration} "
            f"({n_est_key}) | cv_folds={len(fold_aucs)}")

        # ── 2. Multi-seed refit on the FULL merged set (fixed iteration, no ES) ──
        # Disabled flag → single-seed legacy path; keep ensemble_models a list so
        # the artifact / replay shape stays uniform.
        if Config.MULTI_SEED_ENABLED:
            n_seeds = max(1, int(Config.MULTI_SEED_N))
        else:
            n_seeds = 1
        seeds: List[int] = [Config.RANDOM_STATE]
        for i in range(1, n_seeds):
            seeds.append(Config.RANDOM_STATE + Config.MULTI_SEED_BASE_OFFSET * i)
        self.logger.log(self.name, "Multi-seed bagging",
            f"enabled={Config.MULTI_SEED_ENABLED} | n_seeds={n_seeds} | seeds={seeds} | "
            f"refit on train+valid (rows={len(X_fin)}) | fixed {n_est_key}={best_iteration}")

        ensemble_models: List[Any] = []
        for idx, seed in enumerate(seeds):
            params = _base_params(seed)
            if best_iteration is not None:
                params[n_est_key] = best_iteration       # lock ES-derived tree count
            m = ModelClass(**params)
            m.fit(X_fin, y_fin)                            # no ES — iteration fixed
            ensemble_models.append(m)
            self.logger.log(self.name, f"Final fit (seed {idx+1}/{n_seeds})",
                f"seed={seed} | done")
        self.ensemble_models = ensemble_models
        self.model = ensemble_models[0]  # primary for SHAP / single-model ops

        def _ensemble_raw_proba(X) -> np.ndarray:
            """Mean of base models' P(class=1) — defined here so the calibrator
            fit + metric eval share the exact same prediction logic."""
            return np.mean([m.predict_proba(X)[:, 1] for m in ensemble_models], axis=0)

        # ── 3. Leakage-free calibration on the CV out-of-fold predictions ──
        # `valid` is inside the training data now, so calibrate on the pooled
        # OOF probabilities instead (CalibratedClassifierCV-style, no leakage).
        self.calibrator = None
        calibration_split_used = None
        if self.calibration_enabled and cv_ok:
            mask  = ~np.isnan(oof_raw)
            y_oof = y_fin.values[mask]
            if mask.sum() >= 100 and pd.Series(y_oof).nunique() >= 2:
                try:
                    self.calibrator = self._fit_calibrator(
                        oof_raw[mask], y_oof, method=Config.CALIBRATION_METHOD)
                    calibration_split_used = "cv_oof(train+valid)"
                    self.logger.log(self.name, "Calibration fit",
                        f"method={Config.CALIBRATION_METHOD} | source=CV out-of-fold | "
                        f"n_rows={int(mask.sum())} (leakage-free)")
                except Exception as e:
                    self.logger.log(self.name, "Calibration WARN",
                        f"OOF calibration failed: {e}")
            else:
                self.logger.log(self.name, "Calibration",
                    f"insufficient OOF rows/classes ({int(mask.sum())}) — skipped")
        elif not self.calibration_enabled:
            self.logger.log(self.name, "Calibration",
                "disabled for this run (calibration=False or CALIBRATION_ENABLED=false)")
        else:
            self.logger.log(self.name, "Calibration",
                "no CV OOF available (CV failed) — calibration skipped")

        # ── 4. Metrics ────────────────────────────────────────────────────
        metrics: Dict[str, Any] = {
            "best_model":           estimator_name,
            "best_params":          best_params,
            "n_seeds":              n_seeds,
            "final_trained_on":     "train+valid",
            "final_train_rows":     int(len(X_fin)),
            "best_iteration":       best_iteration,
            "calibration_method":   Config.CALIBRATION_METHOD if self.calibrator is not None else None,
            "calibration_split":    calibration_split_used,
        }
        # CV AUC is now the honest in-time estimate: out-of-fold on train+valid.
        if fold_aucs:
            metrics["cv_auc_mean"] = round(float(np.mean(fold_aucs)), 4)
            metrics["cv_auc_std"]  = round(float(np.std(fold_aucs)), 4)
            self.logger.log(self.name, "CV AUC (train+valid OOF)",
                f"{metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")

        # ── 5. Per-split AUC + Brier. NOTE: valid_* are IN-SAMPLE now (merged
        #       into training) — kept for transparency; OOT/test are the clean
        #       holdouts and the cv_auc above is the honest generalization proxy.

        for split_name in ("valid_temporal", "valid_random", "valid", "oot", "test"):
            split_df = self.splits.get(split_name, pd.DataFrame())
            if len(split_df) == 0:
                continue
            X_s, y_s = _prep(split_df)
            if y_s.nunique() < 2:
                continue
            try:
                raw_proba = _ensemble_raw_proba(X_s)
                cal_proba = self._apply_calibrator(raw_proba) if self.calibrator is not None else raw_proba
                # AUC is rank-based → calibration is monotonic → same number, but report both for audit
                auc_raw = float(roc_auc_score(y_s, raw_proba))
                auc_cal = float(roc_auc_score(y_s, cal_proba))
                brier_raw = float(brier_score_loss(y_s, raw_proba))
                brier_cal = float(brier_score_loss(y_s, cal_proba))
                metrics[f"{split_name}_auc"]       = round(auc_cal, 4)
                metrics[f"{split_name}_auc_raw"]   = round(auc_raw, 4)
                metrics[f"{split_name}_brier"]     = round(brier_cal, 4)
                metrics[f"{split_name}_brier_raw"] = round(brier_raw, 4)
                self.logger.log(self.name, f"{split_name} metrics",
                    f"AUC raw={auc_raw:.4f} cal={auc_cal:.4f} | "
                    f"Brier raw={brier_raw:.4f} cal={brier_cal:.4f}")
            except Exception as e:
                self.logger.log(self.name, f"{split_name} eval error", str(e))

        return metrics

    @staticmethod
    def _random_state_key(estimator_name: str) -> str:
        """Per-library random-state ctor kwarg. Used by multi-seed bagging."""
        if estimator_name in ("catboost", "CatBoost"):
            return "random_seed"
        return "random_state"

    @staticmethod
    def _fit_calibrator(raw_proba: np.ndarray, y: np.ndarray, method: str):
        """Fit a probability calibrator (raw P(positive) → calibrated P).

        Returns an object with a `.predict(np.ndarray) -> np.ndarray` method
        so the artifact stays self-contained. Both branches are pure sklearn
        classes — load anywhere sklearn is installed without custom modules.
        """
        if method == "sigmoid":
            # Platt scaling — fit logistic on raw probas
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=1e6, solver="lbfgs")
            clf.fit(raw_proba.reshape(-1, 1), y)
            # Wrap into a uniform interface
            class _SigmoidCalibrator:
                def __init__(self, clf): self.clf = clf
                def predict(self, x):
                    return self.clf.predict_proba(np.asarray(x).reshape(-1, 1))[:, 1]
            return _SigmoidCalibrator(clf)
        # Default: isotonic — banking standard for non-monotonic calibration drift
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(raw_proba, y)
        return iso

    def _apply_calibrator(self, raw_proba: np.ndarray) -> np.ndarray:
        """Apply self.calibrator to raw P(positive). Returns calibrated probs.

        IsotonicRegression uses .predict; the sigmoid wrapper also exposes .predict.
        """
        if self.calibrator is None:
            return raw_proba
        cal = self.calibrator.predict(raw_proba)
        return np.clip(cal, 1e-15, 1 - 1e-15)

    # ── Overfitting detection & remediation ──────────────────────────────────

    @staticmethod
    def _check_overfitting(metrics: Dict, threshold: float = 0.12) -> Dict:
        """Return overfitting diagnosis.

        Gap = (ref_auc - holdout_auc) / ref_auc.  detected=True when any holdout
        gap exceeds threshold.

        Reference = cv_auc_mean (out-of-fold on train+valid) — the honest in-time
        generalization estimate. The final model is refit on train+valid, so the
        per-split valid_auc is in-sample (optimistic) and would inflate the gap;
        cv_auc avoids that false trigger. Falls back to valid_auc only if CV AUC
        is unavailable (e.g. CV failed on tiny data).
        """
        ref_auc = (metrics.get("cv_auc_mean")
                   or metrics.get("valid_auc")
                   or metrics.get("valid_temporal_auc") or 0.0)
        gaps: Dict[str, float] = {}
        if ref_auc > 0:
            for key in ("test_auc", "oot_auc"):
                val = metrics.get(key)
                if isinstance(val, float):
                    gap = (ref_auc - val) / ref_auc
                    if gap > threshold:
                        gaps[key] = round(gap, 4)
        return {"detected": bool(gaps), "gaps": gaps, "ref_auc": round(ref_auc, 4)}

    def _apply_conservative_regularization(self, estimator_name: str, params: Dict) -> Dict:
        """Fallback: manually strengthen regularization when LLM parsing fails."""
        if estimator_name in ("lgbm", "LightGBM"):
            overrides = {
                "num_leaves":        max(15, int(params.get("num_leaves", 31) * 0.6)),
                "max_depth":         max(3,  int(params.get("max_depth", 6) * 0.7)),
                "min_child_samples": max(50, int(params.get("min_child_samples", 20) * 2)),
                "reg_alpha":         max(0.1, params.get("reg_alpha", 0.0) * 10 + 0.1),
                "reg_lambda":        max(1.0, params.get("reg_lambda", 0.0) * 10 + 1.0),
                "subsample":         min(0.7, params.get("subsample", 1.0)),
                "colsample_bytree":  min(0.7, params.get("colsample_bytree", 1.0)),
            }
        elif estimator_name in ("xgboost", "XGBoost"):
            overrides = {
                "max_depth":         max(3,  int(params.get("max_depth", 6) * 0.7)),
                "min_child_weight":  max(10, int(params.get("min_child_weight", 1) * 3)),
                "reg_alpha":         max(0.1, params.get("reg_alpha", 0.0) * 10 + 0.1),
                "reg_lambda":        max(1.0, params.get("reg_lambda", 1.0) * 3),
                "subsample":         min(0.7, params.get("subsample", 1.0)),
                "colsample_bytree":  min(0.7, params.get("colsample_bytree", 1.0)),
            }
        elif estimator_name in ("catboost", "CatBoost"):
            overrides = {
                "depth":              max(4, int(params.get("depth", 6) * 0.7)),
                "l2_leaf_reg":        max(5.0, params.get("l2_leaf_reg", 1.0) * 5),
                "bagging_temperature": 0.5,
                "random_strength":    max(1.0, params.get("random_strength", 1.0) * 2),
            }
        else:  # RF / ExtraTrees
            overrides = {
                "max_depth":         max(5, int(params.get("max_depth", 10) * 0.6)) if params.get("max_depth") else 8,
                "min_samples_leaf":  max(10, int(params.get("min_samples_leaf", 1) * 5)),
                "min_samples_split": max(10, int(params.get("min_samples_split", 2) * 3)),
                "max_features":      "sqrt",
            }
        return {**params, **overrides}

    def _get_antioverfitting_params(
        self,
        estimator_name: str,
        best_params: Dict,
        metrics: Dict,
        overfit_gaps: Dict,
    ) -> Dict:
        """Call LLM to suggest regularization hyperparameter overrides that reduce overfitting.

        Falls back to `_apply_conservative_regularization` if LLM response cannot be parsed.
        """
        import re

        gap_desc = " | ".join(
            f"{k.replace('_auc', '')}_gap={v:.1%}" for k, v in overfit_gaps.items()
        )
        params_json = json.dumps(
            {k: v for k, v in best_params.items() if not callable(v)},
            indent=2, default=str,
        )
        metrics_json = json.dumps(
            {k: v for k, v in metrics.items() if isinstance(v, float)}, indent=2
        )

        system = (
            "You are an ML hyperparameter expert. "
            "Return ONLY a valid JSON object — no markdown fences, no explanation text."
        )
        user = (
            f"Model: {estimator_name}\n"
            f"Overfitting detected: {gap_desc}\n"
            f"Current hyperparameters:\n{params_json}\n"
            f"Current metrics:\n{metrics_json}\n\n"
            f"Return a JSON object with 4–6 hyperparameter overrides that strengthen "
            f"regularization to reduce overfitting. Focus on: increasing reg_alpha / "
            f"reg_lambda / l2_leaf_reg, reducing max_depth / num_leaves / n_estimators, "
            f"increasing min_child_samples / min_samples_leaf / min_child_weight, "
            f"reducing subsample / colsample_bytree. "
            f"Only include parameter names valid for {estimator_name}."
        )

        raw = self.call_llm(user, system)
        self.logger.log(self.name, "Overfit LLM Response", raw[:300])

        try:
            match = re.search(r'\{[\s\S]*?\}', raw)
            overrides = json.loads(match.group() if match else raw.strip())
            overrides = self._safe_params(estimator_name, overrides)
            self.logger.log(self.name, "Overfit Fix Params", json.dumps(overrides))
            return {**best_params, **overrides}
        except Exception as e:
            self.logger.log(self.name, "Overfit Fix WARN",
                f"Could not parse LLM params ({e}) → using conservative regularization defaults")
            return self._apply_conservative_regularization(estimator_name, best_params)

    # ── Model persistence ─────────────────────────────────────────────────────

    @staticmethod
    def _collect_runtime_versions(estimator_class_name: str) -> Dict[str, str]:
        """Snapshot the library versions used at training time.

        Stored in the joblib artifact so the inference environment can warn
        when it differs — a silent dtype/encoding change between sklearn
        versions can give wrong predictions without any traceback.
        """
        import sys, sklearn, pandas, numpy, joblib as _joblib
        from datetime import datetime
        versions = {
            "python":       sys.version.split()[0],
            "sklearn":      sklearn.__version__,
            "pandas":       pandas.__version__,
            "numpy":        numpy.__version__,
            "joblib":       _joblib.__version__,
            "trained_at":   datetime.now().isoformat(timespec="seconds"),
        }
        name_lower = estimator_class_name.lower()
        for lib, attr in (("lightgbm", "lgb"), ("xgboost", "xgb"),
                          ("catboost", "cat")):
            if lib in name_lower or attr in name_lower:
                try:
                    mod = __import__(lib)
                    versions[lib] = getattr(mod, "__version__", "unknown")
                except ImportError:
                    pass
        return versions

    def _save_model(self, feature_cols: List[str]) -> None:
        """Persist artifact with ensemble + calibrator (banking AUC stack).

        Schema:
          model            primary single-seed fit (back-compat — old replay code
                           that reads `artifact['model']` keeps working at the
                           cost of skipping the ensemble averaging).
          ensemble_models  list of all fitted base models (len == MULTI_SEED_N).
                           predict_proba should mean across these for production.
          calibrator       fitted IsotonicRegression / sigmoid wrapper or None.
                           Applied after ensemble averaging to map raw → calibrated PD.
        """
        import joblib
        models = self.ensemble_models or [self.model]
        artifact = {
            "model":             self.model,                # primary, back-compat
            "ensemble_models":   models,
            "calibrator":        self.calibrator,
            "calibration_method": Config.CALIBRATION_METHOD if self.calibrator is not None else None,
            "n_seeds":           len(models),
            "feature_cols":      feature_cols,
            "cat_encoders":      self._cat_encoders,
            "target_column":     self.target_column,
            "estimator_name":    self.model.__class__.__name__,
            "versions":          self._collect_runtime_versions(self.model.__class__.__name__),
        }
        Path(Config.RUN_DIR).mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, Config.FINAL_MODEL_PATH)
        self.logger.log(self.name, "Model Saved",
            f"{Config.FINAL_MODEL_PATH} | n_seeds={len(models)} | "
            f"calibrator={'yes (' + Config.CALIBRATION_METHOD + ')' if self.calibrator is not None else 'no'} | "
            f"versions={artifact['versions']}")

    def _generate_model_code(self, estimator_name: str, best_params: Dict,
                              feature_cols: List[str]) -> None:
        from datetime import datetime
        params_repr = json.dumps(best_params, indent=4, default=str)
        features_repr = json.dumps(feature_cols, indent=4)
        cat_cols = json.dumps(sorted(self._cat_encoders.keys()), indent=4)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        code = f'''"""
Auto-generated by TrainModelAgent  |  {ts}
Best model : {estimator_name}
Features   : {len(feature_cols)}
Target     : {self.target_column}
"""

import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ── Load artifact ─────────────────────────────────────────────────────────────
_ARTIFACT_PATH = Path(__file__).parent / "final_model.pkl"
_artifact      = joblib.load(_ARTIFACT_PATH)

# `model` is the primary single-seed fit (back-compat). Production scoring uses
# `ensemble_models` (multi-seed mean) + `calibrator` (isotonic / sigmoid map).
model            = _artifact["model"]
ensemble_models  = _artifact.get("ensemble_models", [model])
calibrator       = _artifact.get("calibrator", None)
feature_cols     = _artifact["feature_cols"]
cat_encoders     = _artifact["cat_encoders"]
target_column    = _artifact["target_column"]

# ── Version compatibility check ──────────────────────────────────────────────
# Warn (do not raise) when the deployment env differs from the training env —
# pickled sklearn / lightgbm objects can change semantics across versions and
# silently produce wrong predictions. Loud warning gives ops a chance to pin.
_TRAINED_VERSIONS = _artifact.get("versions", {{}})
if _TRAINED_VERSIONS:
    import sklearn as _sklearn_runtime
    _CURRENT = {{
        "python":  sys.version.split()[0],
        "sklearn": _sklearn_runtime.__version__,
        "pandas":  pd.__version__,
        "numpy":   np.__version__,
    }}
    _mismatches = [
        f"{{k}}: trained={{_TRAINED_VERSIONS[k]}}  runtime={{_CURRENT[k]}}"
        for k in ("python", "sklearn", "pandas", "numpy")
        if k in _TRAINED_VERSIONS and _TRAINED_VERSIONS[k] != _CURRENT[k]
    ]
    if _mismatches:
        warnings.warn(
            "final_model.pkl was trained with different library versions:\\n  "
            + "\\n  ".join(_mismatches)
            + "\\nPredictions may differ from training-time behavior. Pin requirements.txt to match.",
            stacklevel=2,
        )

# ── Feature list (recorded at training time) ──────────────────────────────────
FEATURES = {features_repr}

# Categorical columns encoded via LabelEncoder
CAT_COLS = {cat_cols}

# Best hyperparameters
BEST_PARAMS = {params_repr}


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same encoding + imputation used during training.

    Categorical replay uses vectorized Series.where(isin) — about 10x faster
    than the equivalent .apply(lambda) on big batches and matches what the
    training-time _transform_X does, so behavior is consistent end-to-end.
    """
    avail = [c for c in FEATURES if c in df.columns]
    out = df[avail].copy()
    for col in out.columns:
        if col in cat_encoders:
            le    = cat_encoders[col]
            known = set(le.classes_)
            vals  = out[col].fillna("__NA__").astype(str)
            vals  = vals.where(vals.isin(known), "__NA__")
            out[col] = le.transform(vals)
        else:
            out[col] = out[col].fillna(-999)
    return out


# ── Prediction helpers ────────────────────────────────────────────────────────
def predict_proba(df: pd.DataFrame) -> np.ndarray:
    """Return calibrated probability of the positive class (shape: n_samples,).

    Two-stage banking PD: ensemble-mean raw probability across all seeds, then
    isotonic / sigmoid calibration mapping to a probability that obeys
    P(default) ≈ observed default rate per score band. Required for IFRS9 ECL.
    """
    X = preprocess(df)
    # Mean of base models' P(positive). 1-model artifact still works (mean of 1).
    raw = np.mean([m.predict_proba(X)[:, 1] for m in ensemble_models], axis=0)
    if calibrator is not None:
        cal = calibrator.predict(raw)
        return np.clip(cal, 1e-15, 1 - 1e-15)
    return raw


def predict_proba_raw(df: pd.DataFrame) -> np.ndarray:
    """Uncalibrated ensemble probability — useful for diagnostics / debugging.

    Same rank order as predict_proba (calibration is monotonic) so AUC is
    identical; differs only in scale + matches observed-vs-predicted PD.
    """
    X = preprocess(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in ensemble_models], axis=0)


def predict(df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    """Return binary predictions (0/1) at a given probability threshold."""
    return (predict_proba(df) >= threshold).astype(int)


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = pd.read_csv(Path(__file__).parent / "engineered_data.csv")
    scores = predict_proba(sample)
    preds  = predict(sample, threshold=0.5)
    print(f"Rows scored   : {{len(scores)}}")
    print(f"Score range   : {{scores.min():.4f}} – {{scores.max():.4f}}")
    print(f"Positive rate : {{preds.mean():.2%}}")
    print(f"Sample scores : {{scores[:5]}}")
'''
        Path(Config.FINAL_MODEL_CODE_PATH).write_text(code, encoding="utf-8")
        self.logger.log(self.name, "Model Code Saved", Config.FINAL_MODEL_CODE_PATH)

        # Per-agent replay script — same content as final_model_code.py since
        # Agent 3's "process" is encoded entirely in the trained artifact.
        # Surfacing it under the pipeline_process_* naming keeps the trio of
        # replay files visible together for users.
        Path(Config.PIPELINE_PROCESS_TM_PATH).write_text(code, encoding="utf-8")
        self.logger.log(self.name, "Process Script Saved", Config.PIPELINE_PROCESS_TM_PATH)

    # ── Step 9: SHAP visual + LLM-explained top features ─────────────────────

    def _tool_shap_final_explain(
        self,
        estimator_name: str,
        feature_cols: List[str],
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Compute SHAP on the FINAL model, save a bar-plot of importances,
        ask the LLM to explain the top N features, and persist a CSV the
        final report can render.

        Returns a DataFrame (empty if SHAP failed or shap is missing) with:
            rank, feature, shap_importance, description, meaning, why_matters
        """
        if self.model is None:
            self.logger.log(self.name, "SHAP Explain", "No final model → skip")
            return pd.DataFrame()

        train_df = self.splits.get("train", pd.DataFrame())
        if len(train_df) == 0:
            self.logger.log(self.name, "SHAP Explain", "Empty train split → skip")
            return pd.DataFrame()

        avail = [c for c in feature_cols if c in train_df.columns]
        if not avail:
            self.logger.log(self.name, "SHAP Explain", "No final features in train → skip")
            return pd.DataFrame()

        X_tr = self._transform_X(train_df[avail])

        sample_n = min(Config.SHAP_SAMPLE_SIZE, len(X_tr))
        rng = np.random.default_rng(Config.RANDOM_STATE)
        idx = rng.choice(len(X_tr), sample_n, replace=False)
        X_sample = X_tr.iloc[idx]

        # ── 1. Compute SHAP values ────────────────────────────────────────────
        importance: Optional[pd.Series] = None
        importance_source = "shap"
        # Keep the raw 2D sample×feature SHAP matrix — needed for the beeswarm plot
        # (bar plot only needs the aggregated importance series).
        raw_shap_values: Optional[np.ndarray] = None
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            sv = explainer.shap_values(X_sample)
            if isinstance(sv, list):
                sv = sv[1]                          # binary classification → positive class
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]
            raw_shap_values = np.asarray(sv)
            importance = pd.Series(
                np.abs(raw_shap_values).mean(axis=0), index=avail
            ).sort_values(ascending=False)
            self.logger.log(self.name, "SHAP Explain",
                f"computed on {sample_n} train rows × {len(avail)} features")
        except Exception as e:
            self.logger.log(self.name, "SHAP Explain WARN",
                f"shap.TreeExplainer failed ({e}) → falling back to model.feature_importances_")
            imp = getattr(self.model, "feature_importances_", None)
            if imp is None:
                self.logger.log(self.name, "SHAP Explain", "No fallback importance available → skip")
                return pd.DataFrame()
            importance = pd.Series(imp, index=avail).sort_values(ascending=False)
            importance_source = "feature_importances_"

        # ── 2a. Bar plot (top 2N features so the distribution context is visible) ─
        plot_n = min(len(importance), max(top_n * 2, top_n))
        bar_saved = False
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            top_plot = importance.head(plot_n).iloc[::-1]   # reverse → biggest on top
            fig_h = max(5, len(top_plot) * 0.28)
            fig, ax = plt.subplots(figsize=(10, fig_h))
            ax.barh(top_plot.index, top_plot.values, color="#3498db")
            ax.set_xlabel("Mean |SHAP value|" if importance_source == "shap" else "Feature importance")
            ax.set_title(f"Top {plot_n} features — {estimator_name} (final model)")
            ax.grid(axis="x", linestyle=":", alpha=0.4)
            plt.tight_layout()
            Path(Config.SHAP_PLOT_PATH).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(Config.SHAP_PLOT_PATH, dpi=120, bbox_inches="tight")
            plt.close(fig)
            bar_saved = True
            self.logger.log(self.name, "SHAP Plot Saved", Config.SHAP_PLOT_PATH)
        except Exception as e:
            self.logger.log(self.name, "SHAP Plot WARN", f"matplotlib failed: {e}")

        # ── 2b. Beeswarm — direction + magnitude per sample for top N ────────
        # Bar plot answers "which features matter?" (magnitude only). The
        # beeswarm answers "in which direction?" (positive SHAP = pushes
        # prediction up, negative = pushes down) AND "is the effect monotonic
        # in the feature value?" (color gradient).
        beeswarm_saved = False
        if raw_shap_values is not None and importance_source == "shap":
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import shap as _shap_lib

                # Reorder columns to match importance ranking — shap.summary_plot
                # already sorts internally but matching upfront keeps colors stable.
                top_feats_beeswarm = importance.head(top_n).index.tolist()
                col_idx = [avail.index(c) for c in top_feats_beeswarm]
                sv_top = raw_shap_values[:, col_idx]
                X_top  = X_sample[top_feats_beeswarm]

                fig_h = max(6, top_n * 0.35)
                plt.figure(figsize=(11, fig_h))
                _shap_lib.summary_plot(
                    sv_top, X_top,
                    feature_names=top_feats_beeswarm,
                    plot_type="dot",          # beeswarm
                    max_display=top_n,
                    show=False,
                    sort=False,               # already pre-sorted by importance
                )
                plt.title(f"SHAP impact on output — top {top_n} features ({estimator_name})",
                          fontsize=12, pad=12)
                plt.tight_layout()
                Path(Config.SHAP_BEESWARM_PATH).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(Config.SHAP_BEESWARM_PATH, dpi=120, bbox_inches="tight")
                plt.close()
                beeswarm_saved = True
                self.logger.log(self.name, "SHAP Beeswarm Saved", Config.SHAP_BEESWARM_PATH)
            except Exception as e:
                self.logger.log(self.name, "SHAP Beeswarm WARN",
                    f"shap.summary_plot failed: {e} — bar plot only")

        # ── 3. LLM-explain top N ──────────────────────────────────────────────
        top_features = importance.head(top_n).index.tolist()
        descriptions = self._col_descriptions or {}
        top_with_desc = [
            {
                "feature": f,
                "shap_importance": round(float(importance[f]), 6),
                "description": descriptions.get(f, ""),
            }
            for f in top_features
        ]

        system = (
            "You are an ML explainability expert. Given a list of features ranked by "
            "SHAP importance from a trained model, explain each feature concisely. "
            "Return ONLY a valid JSON object — no markdown fences, no extra text."
        )
        user = (
            f"Domain : {self._domain}\n"
            f"Estimator : {estimator_name}\n"
            f"Target   : {self.target_column}\n\n"
            f"Top {len(top_with_desc)} features by mean |SHAP value| (high = drives prediction more):\n"
            f"{json.dumps(top_with_desc, indent=2, ensure_ascii=False)}\n\n"
            'Return JSON in format:\n'
            '{"explanations": [\n'
            '    {"feature": "<name>", "meaning": "<1 sentence>", "why_matters": "<1-2 sentences>"},\n'
            '    ...\n'
            ']}\n'
            "- meaning     : what the feature represents in plain language. "
            "If a description is provided, refine and expand it; otherwise infer from name + domain.\n"
            "- why_matters : why this feature drives predictions for the target in this domain.\n"
            "Match the language of the descriptions when present (vd. Vietnamese descriptions → reply in Vietnamese)."
        )

        exp_map: Dict[str, Dict[str, str]] = {}
        try:
            raw = self.call_llm(user, system, json_mode=True, max_tokens=Config.LLM_MAX_TOKENS_LARGE)
            parsed = json.loads(self._extract_json(raw))
            for entry in parsed.get("explanations", []):
                feat = entry.get("feature")
                if feat:
                    exp_map[feat] = {
                        "meaning":     entry.get("meaning", ""),
                        "why_matters": entry.get("why_matters", ""),
                    }
            self.logger.log(self.name, "SHAP LLM Explain",
                f"received explanations for {len(exp_map)}/{len(top_features)} features")
        except Exception as e:
            self.logger.log(self.name, "SHAP LLM WARN",
                f"LLM explain failed ({e}) — saving CSV without narratives")

        # ── 4. Persist CSV ────────────────────────────────────────────────────
        records = []
        for rank, feat in enumerate(top_features, 1):
            exp = exp_map.get(feat, {})
            records.append({
                "rank":            rank,
                "feature":         feat,
                "shap_importance": round(float(importance[feat]), 6),
                "description":     descriptions.get(feat, ""),
                "meaning":         exp.get("meaning", ""),
                "why_matters":     exp.get("why_matters", ""),
            })
        report_df = pd.DataFrame(records)
        Path(Config.SHAP_FEATURE_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(Config.SHAP_FEATURE_REPORT_PATH, index=False, encoding="utf-8")
        self.logger.log(self.name, "SHAP Feature Report Saved", Config.SHAP_FEATURE_REPORT_PATH)

        # ── 5. Dedicated final-model SHAP markdown report ────────────────────
        # One self-contained file that combines: bar plot (importance ranking),
        # beeswarm plot (impact direction + magnitude per sample), and the
        # LLM-narrated top-N table. Lives next to final_model.pkl so analysts
        # can pick up the entire explainability bundle in one place.
        try:
            self._write_shap_markdown_report(
                report_df=report_df,
                estimator_name=estimator_name,
                importance_source=importance_source,
                sample_n=sample_n,
                bar_saved=bar_saved,
                beeswarm_saved=beeswarm_saved,
            )
        except Exception as e:
            self.logger.log(self.name, "SHAP MD Report WARN",
                f"could not assemble combined markdown report: {e}")
        return report_df

    def _write_shap_markdown_report(
        self,
        report_df: pd.DataFrame,
        estimator_name: str,
        importance_source: str,
        sample_n: int,
        bar_saved: bool,
        beeswarm_saved: bool,
    ) -> None:
        """Assemble bar + beeswarm + LLM narrative into final_model_shap_report.md.

        Image paths are written as bare filenames so the report renders correctly
        when the run_NN/ directory is moved, zipped, or served from a static host.
        """
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        bar_name      = Path(Config.SHAP_PLOT_PATH).name
        beeswarm_name = Path(Config.SHAP_BEESWARM_PATH).name
        csv_name      = Path(Config.SHAP_FEATURE_REPORT_PATH).name
        model_name    = Path(Config.FINAL_MODEL_PATH).name

        n_final = len(self.feature_cols) if self.feature_cols else len(report_df)
        importance_metric = "Mean |SHAP value|" if importance_source == "shap" else "Feature importance (fallback)"

        lines: List[str] = []
        lines.append("# Final Model — SHAP Explainability Report\n\n")
        lines.append(
            "| Field | Value |\n"
            "|---|---|\n"
            f"| Generated      | {ts} |\n"
            f"| Estimator      | `{estimator_name}` |\n"
            f"| Target         | `{self.target_column}` |\n"
            f"| Domain         | `{self._domain}` |\n"
            f"| Features in final model | {n_final} |\n"
            f"| SHAP sample size | {sample_n} rows from train |\n"
            f"| Importance metric | {importance_metric} |\n"
            f"| Model artifact | `{model_name}` |\n\n"
            "---\n\n"
        )

        # 1. Bar plot — importance magnitude
        lines.append("## 1. Feature importance — magnitude\n\n")
        if bar_saved:
            lines.append(f"![SHAP feature importance]({bar_name})\n\n")
        else:
            lines.append("_Bar plot was not generated (matplotlib unavailable)._\n\n")
        lines.append(
            "Mỗi bar = trung bình giá trị tuyệt đối của SHAP của một feature trên "
            f"sample SHAP ({sample_n} rows train). Đây là answer cho câu hỏi: "
            "**\"feature nào quan trọng nhất với prediction trung bình?\"**.\n\n"
            "---\n\n"
        )

        # 2. Beeswarm — impact direction + sample distribution
        lines.append("## 2. Impact on model output — beeswarm (direction + magnitude)\n\n")
        if beeswarm_saved:
            lines.append(f"![SHAP beeswarm]({beeswarm_name})\n\n")
            lines.append(
                "Mỗi dot = một sample (1 row train). Đọc plot:\n\n"
                "- **Trục X** : SHAP value của sample đó. SHAP > 0 = feature *đẩy probability dự đoán lên*; "
                "SHAP < 0 = *đẩy xuống*. Khoảng cách từ 0 = độ lớn ảnh hưởng.\n"
                "- **Trục Y** : features xếp theo mean |SHAP| giảm dần (giống bar plot).\n"
                "- **Màu**   : giá trị thực của feature ở sample đó. Đỏ = giá trị cao, xanh = giá trị thấp.\n"
                "- **Mật độ dot** : phần phình to = nhiều sample có cùng mức ảnh hưởng → effect nhất quán; "
                "rải rác = effect phụ thuộc context (interaction với feature khác).\n\n"
                "**Cách diễn giải nhanh**:\n"
                "- Cluster **đỏ phía phải** → feature giá trị cao → tăng prediction (positive class)\n"
                "- Cluster **đỏ phía trái** → feature giá trị cao → giảm prediction (negative class, inverse relationship)\n"
                "- Hai cụm tách biệt 2 đầu → feature có effect mạnh và monotonic (đáng quan tâm trong scorecard)\n"
                "- Spread rộng giữa các sample cùng màu → effect biến đổi theo interactions\n\n"
            )
        else:
            lines.append(
                "_Beeswarm plot was not generated. Common reasons: `shap` library not installed, "
                "or the fallback `feature_importances_` was used (no per-sample SHAP matrix available)._\n\n"
            )
        lines.append("---\n\n")

        # 3. LLM-narrated table
        lines.append(f"## 3. Top {len(report_df)} features — LLM-narrated meaning\n\n")
        if not report_df.empty:
            lines.append(
                "| Rank | Feature | SHAP importance | Description | Meaning | Why it matters |\n"
                "|---:|---|---:|---|---|---|\n"
            )
            for rec in report_df.to_dict(orient="records"):
                feat = rec.get("feature", "")
                imp  = float(rec.get("shap_importance", 0.0))
                desc = (rec.get("description") or "").replace("|", "\\|").replace("\n", " ")
                mean = (rec.get("meaning") or "").replace("|", "\\|").replace("\n", " ")
                why  = (rec.get("why_matters") or "").replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| {rec.get('rank', '')} | `{feat}` | {imp:.4f} | {desc} | {mean} | {why} |\n"
                )
            lines.append("\n")
        else:
            lines.append("_No top features available (SHAP step skipped)._\n\n")

        # 4. Footer + sibling files
        lines.append(
            "---\n\n"
            "## Sibling files\n\n"
            f"- `{bar_name}` — bar plot (magnitude only)\n"
            f"- `{beeswarm_name}` — beeswarm (direction + per-sample distribution)\n"
            f"- `{csv_name}` — full CSV (rank, feature, importance, description, meaning, why_matters)\n"
            f"- `{model_name}` — joblib artifact (model + encoders + versions)\n"
            f"- `{Path(Config.FINAL_MODEL_CODE_PATH).name}` — standalone inference script\n\n"
            "*SHAP values computed on the FINAL model after overfit-aware retrain — "
            "reflects exactly what gets deployed.*\n"
        )

        Path(Config.SHAP_FINAL_MODEL_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(Config.SHAP_FINAL_MODEL_REPORT_PATH).write_text("".join(lines), encoding="utf-8")
        self.logger.log(self.name, "SHAP MD Report Saved", Config.SHAP_FINAL_MODEL_REPORT_PATH)

    # ── Step 10: Model diagnostic charts ─────────────────────────────────────

    def _tool_generate_model_charts(
        self,
        feature_cols: List[str],
        estimator_name: str,
    ) -> Dict[str, str]:
        """Generate 9 model diagnostic charts and save to {RUN_DIR}/charts/.

        Uses the best available eval split (OOT > valid > test).
        Each chart is wrapped in try/except — failures are logged and skipped.
        Returns dict {chart_key: saved_path} for charts that succeeded.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        chart_dir = Path(Config.CHART_DIR)
        chart_dir.mkdir(parents=True, exist_ok=True)
        saved: Dict[str, str] = {}

        # Pick best eval split — prefer the clean final holdouts (oot, then test
        # in simple mode). `valid` is in-sample after the train+valid refit, so it
        # is the last resort only.
        X_eval, y_eval, split_name = None, None, None
        for sname in ("oot", "test", "valid"):
            sdf = self.splits.get(sname, pd.DataFrame())
            if len(sdf) < 50:
                continue
            avail = [c for c in feature_cols if c in sdf.columns]
            if not avail:
                continue
            X_s = self._transform_X(sdf[avail])
            y_s = pd.to_numeric(sdf[self.target_column], errors="coerce")
            if y_s.nunique() < 2:
                continue
            X_eval, y_eval, split_name = X_s, y_s, sname
            break

        if X_eval is None:
            self.logger.log(self.name, "Charts WARN", "No usable eval split (≥50 rows, 2 classes) — skipping all charts")
            return saved

        # Ensemble predict → calibrate
        y_score = np.mean([m.predict_proba(X_eval)[:, 1] for m in self.ensemble_models], axis=0)
        if self.calibrator is not None:
            y_score = np.clip(self.calibrator.predict(y_score), 1e-15, 1 - 1e-15)
        y_true = y_eval.values

        self.logger.log(self.name, "Charts", f"generating on split='{split_name}' | n={len(y_true)}")

        STYLE = {"dpi": 120, "bbox_inches": "tight"}
        BLUE, RED, GREEN = "#2196F3", "#F44336", "#4CAF50"

        def _decile_table(y_true, y_score, n=Config.CHART_N_DECILES):
            df = pd.DataFrame({"score": y_score, "label": y_true})
            df["decile"] = pd.qcut(-df["score"], q=n, labels=False, duplicates="drop") + 1
            grp = df.groupby("decile").agg(
                n_total=("label", "count"),
                n_bad=("label", "sum"),
                avg_score=("score", "mean"),
            ).reset_index().sort_values("decile")
            total_bad = max(y_true.sum(), 1)
            grp["bad_rate"] = grp["n_bad"] / grp["n_total"]
            grp["cum_bad"] = grp["n_bad"].cumsum()
            grp["cum_total"] = grp["n_total"].cumsum()
            grp["cum_bad_pct"] = grp["cum_bad"] / total_bad
            return grp

        # ── 1. ROC Curve ──────────────────────────────────────────────────
        try:
            from sklearn.metrics import roc_curve, roc_auc_score
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"AUC = {auc:.4f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve — {estimator_name} ({split_name})")
            ax.legend(loc="lower right"); ax.grid(linestyle=":", alpha=0.4)
            plt.tight_layout()
            plt.savefig(Config.CHART_ROC_PATH, **STYLE); plt.close(fig)
            saved["roc"] = Config.CHART_ROC_PATH
        except Exception as e:
            self.logger.log(self.name, "Chart WARN", f"ROC failed: {e}")

        # ── 2. Precision-Recall Curve ─────────────────────────────────────
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(rec, prec, color=BLUE, lw=2, label=f"AP = {ap:.4f}")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_title(f"Precision-Recall Curve — {estimator_name} ({split_name})")
            ax.legend(); ax.grid(linestyle=":", alpha=0.4)
            plt.tight_layout()
            plt.savefig(Config.CHART_PR_PATH, **STYLE); plt.close(fig)
            saved["pr"] = Config.CHART_PR_PATH
        except Exception as e:
            self.logger.log(self.name, "Chart WARN", f"PR Curve failed: {e}")

        # ── 3. KS Curve ───────────────────────────────────────────────────
        try:
            ks_df = (pd.DataFrame({"score": y_score, "label": y_true})
                     .sort_values("score", ascending=False)
                     .reset_index(drop=True))
            total_bad  = max(int(y_true.sum()), 1)
            total_good = max(int(len(y_true) - y_true.sum()), 1)
            ks_df["cum_bad_rate"]  = ks_df["label"].cumsum() / total_bad
            ks_df["cum_good_rate"] = (1 - ks_df["label"]).cumsum() / total_good
            ks_df["ks"]            = ks_df["cum_bad_rate"] - ks_df["cum_good_rate"]
            ks_val = ks_df["ks"].max()
            ks_idx = ks_df["ks"].idxmax()
            x_axis = np.linspace(0, 1, len(ks_df))
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(x_axis, ks_df["cum_bad_rate"].values,  color=RED,   lw=2, label="Cumulative Bad")
            ax.plot(x_axis, ks_df["cum_good_rate"].values, color=GREEN, lw=2, label="Cumulative Good")
            ax.axvline(x=x_axis[ks_idx], color="gray", linestyle="--", lw=1)
            ax.annotate(f"KS={ks_val:.4f}",
                        xy=(x_axis[ks_idx], ks_df.loc[ks_idx, "cum_bad_rate"]),
                        xytext=(5, -15), textcoords="offset points", fontsize=10, color="gray")
            ax.set_xlabel("Population (sorted by score desc)"); ax.set_ylabel("Cumulative Rate")
            ax.set_title(f"KS Curve — {estimator_name} ({split_name})")
            ax.legend(); ax.grid(linestyle=":", alpha=0.4)
            plt.tight_layout()
            plt.savefig(Config.CHART_KS_PATH, **STYLE); plt.close(fig)
            saved["ks"] = Config.CHART_KS_PATH
            self.logger.log(self.name, "Chart KS", f"KS={ks_val:.4f}")
        except Exception as e:
            self.logger.log(self.name, "Chart WARN", f"KS Curve failed: {e}")

        # ── 4-7. Decile-based charts (one pass) ───────────────────────────
        try:
            dec = _decile_table(y_true, y_score)
            avg_bad_rate = float(y_true.mean())
            dec_labels = dec["decile"].astype(str).tolist()
            # Stay accurate when CHART_N_DECILES != 10 (no longer "deciles").
            _nb = Config.CHART_N_DECILES
            bucket_word = "Decile" if _nb == 10 else f"Bucket (of {_nb})"

            # 4. Lift Chart
            lift_vals = (dec["bad_rate"] / avg_bad_rate).tolist()
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(dec_labels, lift_vals, color=BLUE)
            ax.axhline(1.0, color=RED, linestyle="--", lw=1.5, label="Baseline (lift=1)")
            for bar, val in zip(bars, lift_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.2f}x", ha="center", va="bottom", fontsize=8)
            ax.set_xlabel(f"{bucket_word} (1=highest score)"); ax.set_ylabel("Lift")
            ax.set_title(f"Lift Chart — {estimator_name} ({split_name})")
            ax.legend(); ax.grid(axis="y", linestyle=":", alpha=0.4)
            plt.tight_layout()
            plt.savefig(Config.CHART_LIFT_PATH, **STYLE); plt.close(fig)
            saved["lift"] = Config.CHART_LIFT_PATH

            # 5. Gain Chart
            x_pct = (dec["cum_total"] / dec["cum_total"].iloc[-1] * 100).tolist()
            y_pct = (dec["cum_bad_pct"] * 100).tolist()
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot([0] + x_pct, [0] + y_pct, color=BLUE, lw=2, marker="o", ms=4, label="Model")
            ax.plot([0, 100], [0, 100], "k--", lw=1, label="Random")
            ax.set_xlabel("% Population"); ax.set_ylabel("% Bad Captured")
            ax.set_title(f"Gain Chart — {estimator_name} ({split_name})")
            ax.legend(); ax.grid(linestyle=":", alpha=0.4)
            plt.tight_layout()
            plt.savefig(Config.CHART_GAIN_PATH, **STYLE); plt.close(fig)
            saved["gain"] = Config.CHART_GAIN_PATH

            # 6. Bad Rate by Decile
            bad_rates = (dec["bad_rate"] * 100).tolist()
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(dec_labels, bad_rates, color=BLUE)
            ax.axhline(avg_bad_rate * 100, color=RED, linestyle="--", lw=1.5,
                       label=f"Overall={avg_bad_rate:.1%}")
            for bar, val in zip(bars, dec["bad_rate"].tolist()):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{val:.1%}", ha="center", va="bottom", fontsize=8)
            ax.set_xlabel(f"{bucket_word} (1=highest score)"); ax.set_ylabel("Bad Rate (%)")
            ax.set_title(f"Bad Rate by {bucket_word} — {estimator_name} ({split_name})")
            ax.legend(); ax.grid(axis="y", linestyle=":", alpha=0.4)
            plt.tight_layout()
            plt.savefig(Config.CHART_BAD_RATE_DECILE_PATH, **STYLE); plt.close(fig)
            saved["bad_rate_decile"] = Config.CHART_BAD_RATE_DECILE_PATH

            # 7. Average Score by Decile
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(dec_labels, dec["avg_score"].tolist(), color=BLUE, lw=2, marker="o", ms=5)
            for x, y in zip(dec_labels, dec["avg_score"].tolist()):
                ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 6),
                            textcoords="offset points", ha="center", fontsize=8)
            ax.set_xlabel(f"{bucket_word} (1=highest score)"); ax.set_ylabel("Average Score")
            ax.set_title(f"Average Score by {bucket_word} — {estimator_name} ({split_name})")
            ax.grid(linestyle=":", alpha=0.4)
            plt.tight_layout()
            plt.savefig(Config.CHART_AVG_SCORE_DECILE_PATH, **STYLE); plt.close(fig)
            saved["avg_score_decile"] = Config.CHART_AVG_SCORE_DECILE_PATH

            self.logger.log(self.name, "Charts Saved", "Lift + Gain + Bad Rate + Avg Score by Decile")
        except Exception as e:
            self.logger.log(self.name, "Chart WARN", f"Decile charts failed: {e}")

        # ── 8. Calibration Plot ───────────────────────────────────────────
        try:
            from sklearn.calibration import calibration_curve
            frac_pos, mean_pred = calibration_curve(
                y_true, y_score, n_bins=Config.CHART_N_DECILES, strategy="quantile")
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(mean_pred, frac_pos, color=BLUE, lw=2, marker="o", ms=5, label="Model")
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
            ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
            ax.set_title(f"Calibration Plot — {estimator_name} ({split_name})")
            ax.legend(); ax.grid(linestyle=":", alpha=0.4)
            plt.tight_layout()
            plt.savefig(Config.CHART_CALIBRATION_PATH, **STYLE); plt.close(fig)
            saved["calibration"] = Config.CHART_CALIBRATION_PATH
        except Exception as e:
            self.logger.log(self.name, "Chart WARN", f"Calibration Plot failed: {e}")

        # ── 9. Score Distribution (Good vs Bad) ───────────────────────────
        try:
            scores_good = y_score[y_true == 0]
            scores_bad  = y_score[y_true == 1]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(scores_good, bins=50, alpha=0.6, color=GREEN, label="Good (0)", density=True)
            ax.hist(scores_bad,  bins=50, alpha=0.6, color=RED,   label="Bad (1)",  density=True)
            ax.set_xlabel("Score"); ax.set_ylabel("Density")
            ax.set_title(f"Score Distribution (Good vs Bad) — {estimator_name} ({split_name})")
            ax.legend(); ax.grid(linestyle=":", alpha=0.4)
            plt.tight_layout()
            plt.savefig(Config.CHART_SCORE_DIST_PATH, **STYLE); plt.close(fig)
            saved["score_dist"] = Config.CHART_SCORE_DIST_PATH
        except Exception as e:
            self.logger.log(self.name, "Chart WARN", f"Score Distribution failed: {e}")

        self.logger.log(self.name, "Charts Complete",
            f"{len(saved)}/9 charts saved to {chart_dir}")
        return saved

    # ── Split-mode entry point ────────────────────────────────────────────────

    def process_splits(
        self,
        train_path: str,
        previous_report: Dict[str, Any],
        target_column: str,
        valid_path: Optional[str] = None,
        oot_path: Optional[str] = None,
        date_col: Optional[str] = None,
        id_col: Optional[str] = None,
        calibration: Optional[bool] = None,
        temporal_freq: Optional[str] = None,
        week_closing_day: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load 3 pre-engineered partitions, add `_split_` marker, dispatch to `process`.

        This is the only place in the pipeline where train + valid + oot live in
        the same DataFrame — concat happens here because the downstream feature-
        selection / training steps (FLAML / Optuna / CV / PSI / Stability) need
        all three slices to compute their metrics.

        The marker drives `_tool_split_data`'s pre-split branch to reconstruct
        the user-supplied splits exactly as provided.
        """
        import gc

        self.logger.log(self.name, "process_splits start",
            f"train={train_path} | valid={valid_path} | oot={oot_path}")

        frames: List[pd.DataFrame] = []
        sizes: Dict[str, int] = {}
        for tag, path in [("train", train_path), ("valid", valid_path), ("oot", oot_path)]:
            if path is None:
                continue
            sub = self.load_dataframe(path)
            sub[_SPLIT_MARKER] = tag
            sizes[tag] = len(sub)
            frames.append(sub)
            self.logger.log(self.name, f"Loaded {tag}", f"shape={sub.shape} | path={path}")

        if not frames:
            raise ValueError("process_splits called with no input paths")

        df = pd.concat(frames, ignore_index=True)
        del frames
        gc.collect()
        self.logger.log(self.name, "Concat done",
            f"combined shape={df.shape} | sizes={sizes}")

        return self.process(
            df=df,
            previous_report=previous_report,
            target_column=target_column,
            date_col=date_col,
            id_col=id_col,
            oot_df=None,  # OOT comes through the marker — never via oot_df arg
            calibration=calibration,
            temporal_freq=temporal_freq,
            week_closing_day=week_closing_day,
        )

    # ── Main process ──────────────────────────────────────────────────────────

    def process(
        self,
        df: pd.DataFrame,
        previous_report: Dict[str, Any],
        target_column: str,
        date_col: Optional[str] = None,
        id_col: Optional[str] = None,
        oot_df: Optional[pd.DataFrame] = None,
        calibration: Optional[bool] = None,
        temporal_freq: Optional[str] = None,
        week_closing_day: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.logger.log(self.name, "Process Start", f"Shape={df.shape}")
        # Per-run calibration override. None → keep the global Config default set
        # in __init__; True/False → override CALIBRATION_ENABLED for this run only.
        if calibration is not None:
            self.calibration_enabled = bool(calibration)
            self.logger.log(self.name, "Calibration override",
                f"calibration={self.calibration_enabled} (per-run arg overrides "
                f"CALIBRATION_ENABLED={Config.CALIBRATION_ENABLED})")
        # Per-run temporal-frequency override. None → keep Config default; weekly
        # is opt-in here. week_closing_day picks the W-anchor (auto from data if
        # omitted). Both feed _resolve_period_freq during _tool_split_data.
        if temporal_freq is not None:
            self.temporal_freq = str(temporal_freq).lower()
            self.logger.log(self.name, "Temporal freq override",
                f"temporal_freq={self.temporal_freq} (per-run arg overrides "
                f"TEMPORAL_FREQ={Config.TEMPORAL_FREQ})")
        if week_closing_day is not None:
            self.week_closing_day = str(week_closing_day).strip().upper()
        self.logger.log(self.name, "Previous Agent Summary",
            previous_report.get("summary", "No summary"))

        self.df = df.copy()
        self.target_column = self._resolve_target_column(self.df, target_column)
        target_column = self.target_column  # sync local var to resolved name

        # Prefer key columns from DataCleaner/FeatureEngineer report over auto-detection
        _report_id = previous_report.get("entity_id_col")
        _ck_cols = previous_report.get("composite_key_cols", [])
        _report_date = next(
            (c for c in _ck_cols if c != _report_id and c in self.df.columns),
            None,
        )
        self.id_col = id_col or (_report_id if _report_id in self.df.columns else None) or self._auto_detect_id_col(df)
        self.date_col = date_col or _report_date or self._auto_detect_date_col(df)
        self.feature_cols = self._get_feature_cols(self.df)

        self.logger.log(self.name, "Config",
            f"target={self.target_column} | date_col={self.date_col} | "
            f"id_col={self.id_col} | n_features={len(self.feature_cols)} | "
            f"gpu={'YES' if self._has_gpu() else 'NO (CPU only)'}")

        if self.date_col is None:
            self.logger.log(self.name, "WARN",
                "No date column detected — OOT split will be skipped, PSI filter and "
                "Stability check will be SKIPPED. Pass date_col= to process() or ensure "
                "a date/time/snapshot column exists in the data.")

        # ── 1. Split ──────────────────────────────────────────────────────────
        self.splits = self._tool_split_data(self.df, provided_oot=oot_df)
        train_df = self.splits["train"]
        valid_temp_df = self.splits["valid_temporal"]
        valid_df = self.splits["valid"]
        oot_df = self.splits["oot"]

        feat_avail = [c for c in self.feature_cols if c in train_df.columns]
        X_train = self._fit_prepare_X(train_df[feat_avail])
        y_train = pd.to_numeric(train_df[target_column], errors="coerce")

        # Use valid_temporal for Optuna (no leakage); fall back to valid if empty
        if len(valid_temp_df) > 0:
            X_valid = self._transform_X(valid_temp_df[feat_avail])
            y_valid = pd.to_numeric(valid_temp_df[target_column], errors="coerce")
        elif len(valid_df) > 0:
            X_valid = self._transform_X(valid_df[feat_avail])
            y_valid = pd.to_numeric(valid_df[target_column], errors="coerce")
        else:
            X_valid = pd.DataFrame(columns=feat_avail)
            y_valid = pd.Series(dtype=float)

        # ── Dynamic time budget ───────────────────────────────────────────────
        n_rows_total = len(self.df)
        n_cols_total = len(self.feature_cols)
        flaml_budget, optuna_budget = self._compute_time_budget(n_rows_total, n_cols_total)
        self.logger.log(self.name, "Time Budget",
            f"rows={n_rows_total} | cols={n_cols_total} | "
            f"flaml={flaml_budget}s | optuna={optuna_budget}s")

        # ── 2. FLAML ──────────────────────────────────────────────────────────
        self.logger.log(self.name, "FLAML Start",
            f"time_budget={flaml_budget}s | estimators={Config.FLAML_ESTIMATORS}")
        best_estimator, flaml_params = self._tool_run_flaml(
            X_train, y_train, X_valid, y_valid, time_budget=flaml_budget
        )

        # ── 3. Optuna ─────────────────────────────────────────────────────────
        self.logger.log(self.name, "Optuna Start",
            f"n_trials={Config.OPTUNA_N_TRIALS} | timeout={optuna_budget}s")
        best_params = self._tool_run_optuna(
            best_estimator, flaml_params, X_train, y_train, X_valid, y_valid,
            timeout=optuna_budget,
        )

        # ── 4. RFE ────────────────────────────────────────────────────────────
        self.logger.log(self.name, "RFE Start",
            f"target={Config.MAX_FINAL_FEATURES} | rfecv={Config.ENABLE_RFECV}")
        rfe_features = self._tool_run_rfe(best_estimator, best_params, X_train, y_train)

        # ── 5. PSI ────────────────────────────────────────────────────────────
        # PSI measures temporal distribution drift — only meaningful when a date
        # column exists (i.e. OOT is a genuine time-based holdout).
        # Without a date key, PSI is skipped to avoid spurious feature drops.
        psi_train_raw = train_df[[c for c in rfe_features if c in train_df.columns]]
        if self.date_col and len(oot_df) > 0:
            psi_oot_raw = oot_df[[c for c in rfe_features if c in oot_df.columns]]
        else:
            psi_oot_raw = pd.DataFrame()
        psi_features, psi_df = self._tool_run_psi_filter(
            rfe_features, psi_train_raw, psi_oot_raw,
        )

        # ── 6. Stability ──────────────────────────────────────────────────────
        # Build encoded train slice with date + target for stability Gini calc
        train_for_stability = self._transform_X(
            train_df[[c for c in psi_features if c in train_df.columns]]
        )
        train_for_stability[self.target_column] = pd.to_numeric(train_df[self.target_column], errors="coerce").values
        if self.date_col and self.date_col in train_df.columns:
            train_for_stability[self.date_col] = train_df[self.date_col].values
        stable_features, stab_df = self._tool_run_stability_check(psi_features, train_for_stability)

        # ── 7. SHAP+PSI iterative pruning ─────────────────────────────────────
        X_tr_stable = X_train[[c for c in stable_features if c in X_train.columns]]
        X_vl_stable = (
            X_valid[[c for c in stable_features if c in X_valid.columns]]
            if len(X_valid) > 0 else pd.DataFrame()
        )
        final_features, prune_df = self._tool_shap_psi_prune(
            best_estimator, best_params,
            stable_features, X_tr_stable, y_train,
            X_vl_stable, y_valid,
            psi_df,
        )

        self.logger.log(self.name, "Feature Pipeline",
            f"init={len(feat_avail)} → RFE={len(rfe_features)} → "
            f"PSI={len(psi_features)} → Stability={len(stable_features)} → "
            f"SHAP+PSI={len(final_features)}")

        if not final_features:
            raise ValueError(
                "Feature pipeline eliminated all features. "
                "Lower PSI_THRESHOLD / STABILITY_GINI_STD_THRESHOLD or reduce MAX_FINAL_FEATURES."
            )

        # ── 7b. Re-tune Optuna on the final (settled) feature set ────────────
        # The initial Optuna study optimised hyperparams against feat_avail
        # (~1500 cols for wide data). After RFE → PSI → Stability → SHAP+PSI
        # the model usually sees 50-100 features at inference. Hyperparams
        # like num_leaves / max_depth / min_child_samples that were optimal
        # at 1500 cols are typically too aggressive at 100 cols → suboptimal
        # AUC on the FINAL model.
        #
        # We re-run a shorter Optuna study restricted to the final feature
        # set when the pipeline meaningfully shrank the feature count.
        n_init = len(feat_avail)
        reduction_ratio = len(final_features) / n_init if n_init else 1.0
        if (Config.ENABLE_RETUNE_AFTER_PRUNE
                and reduction_ratio <= Config.RETUNE_FEATURE_REDUCTION_TRIGGER):
            self.logger.log(self.name, "Optuna Re-tune Start",
                f"feature count {n_init} → {len(final_features)} "
                f"({reduction_ratio:.0%} kept ≤ trigger {Config.RETUNE_FEATURE_REDUCTION_TRIGGER:.0%}) "
                f"→ re-tuning hyperparams on final features")
            X_tr_final = X_train[[c for c in final_features if c in X_train.columns]]
            X_vl_final = (
                X_valid[[c for c in final_features if c in X_valid.columns]]
                if len(X_valid) > 0 else pd.DataFrame()
            )
            retune_budget = max(60, int(optuna_budget * Config.RETUNE_TIMEOUT_RATIO))
            new_best_params = self._tool_run_optuna(
                best_estimator, best_params,
                X_tr_final, y_train, X_vl_final, y_valid,
                timeout=retune_budget,
            )
            # _tool_run_optuna falls back to base_params when no trial completes
            # within the budget — only adopt when something actually came back.
            if new_best_params is not best_params:
                best_params = new_best_params
                self.logger.log(self.name, "Optuna Re-tune Adopted",
                    "best_params replaced with re-tuned hyperparams")
        elif Config.ENABLE_RETUNE_AFTER_PRUNE:
            self.logger.log(self.name, "Optuna Re-tune Skipped",
                f"feature kept ratio {reduction_ratio:.0%} > trigger "
                f"{Config.RETUNE_FEATURE_REDUCTION_TRIGGER:.0%} (not enough reduction)")

        # ── 8. Final model ────────────────────────────────────────────────────
        self.feature_cols = final_features
        metrics = self._tool_train_final_model(best_estimator, best_params, final_features)

        # ── 8b. Overfitting detection & optional LLM-guided retrain ──────────
        overfit_info = self._check_overfitting(metrics, Config.OVERFIT_THRESHOLD)
        if overfit_info["detected"]:
            gap_log = " | ".join(
                f"{k.replace('_auc', '')}_gap={v:.1%}" for k, v in overfit_info["gaps"].items()
            )
            self.logger.log(self.name, "OVERFIT DETECTED",
                f"ref_auc(cv_oof)={overfit_info['ref_auc']:.4f} | {gap_log} "
                f"(threshold={Config.OVERFIT_THRESHOLD:.0%}) → requesting LLM-guided retrain")

            import io as _io
            import joblib as _jl
            # Snapshot the FULL training state, not just self.model: production
            # scoring (and the diagnostic charts) read self.ensemble_models +
            # self.calibrator, which _tool_train_final_model overwrites on the
            # retry below. If we only restored self.model on revert, the saved
            # artifact would mix the original primary fit with the *discarded*
            # retry ensemble/calibrator → inconsistent deployed model.
            _buf = _io.BytesIO()
            _jl.dump(
                {
                    "model":           self.model,
                    "ensemble_models": self.ensemble_models,
                    "calibrator":      self.calibrator,
                },
                _buf,
            )
            _orig_state_bytes = _buf.getvalue()
            orig_best_params = best_params

            retrain_params = self._get_antioverfitting_params(
                best_estimator, best_params, metrics, overfit_info["gaps"]
            )
            metrics_retry = self._tool_train_final_model(
                best_estimator, retrain_params, final_features
            )
            overfit_retry = self._check_overfitting(metrics_retry, Config.OVERFIT_THRESHOLD)

            _holdout_key = "oot_auc" if metrics.get("oot_auc") is not None else "test_auc"
            retry_holdout = metrics_retry.get(_holdout_key) or 0.0
            orig_holdout  = metrics.get(_holdout_key) or 0.0

            if retry_holdout >= orig_holdout:
                best_params = retrain_params
                metrics = metrics_retry
                _status = "resolved" if not overfit_retry["detected"] else "reduced"
                self.logger.log(self.name, f"Overfit {_status.title()}",
                    f"Retrain kept | {_holdout_key}={retry_holdout:.4f} "
                    f"(was {orig_holdout:.4f}) | still_overfit={overfit_retry['detected']}")
                overfit_info["retrain"] = _status
            else:
                _orig_state = _jl.load(_io.BytesIO(_orig_state_bytes))
                self.model           = _orig_state["model"]
                self.ensemble_models = _orig_state["ensemble_models"]
                self.calibrator      = _orig_state["calibrator"]
                best_params = orig_best_params
                self.logger.log(self.name, "Overfit Retry Reverted",
                    f"Retrain degraded {_holdout_key}: "
                    f"{retry_holdout:.4f} < {orig_holdout:.4f} → original model "
                    f"(+ensemble +calibrator) restored")
                overfit_info["retrain"] = "reverted"
        else:
            overfit_info["retrain"] = "not_needed"

        # ── Save model artifact + inference code ──────────────────────────────
        self._save_model(final_features)
        self._generate_model_code(best_estimator, best_params, final_features)

        # ── 9. Model diagnostic charts ────────────────────────────────────────
        chart_paths: Dict[str, str] = {}
        try:
            chart_paths = self._tool_generate_model_charts(final_features, best_estimator)
        except Exception as e:
            self.logger.log(self.name, "Charts ERROR",
                f"chart generation failed: {e} — continuing without charts")

        # ── 10. SHAP visual + LLM-explained top features ─────────────────────
        # Runs on the *final* model so importances reflect what actually ships.
        # Gracefully degrades: missing shap → feature_importances_; LLM failure
        # → CSV without narrative columns. Toggle with SHAP_FINAL_EXPLAIN_ENABLED.
        shap_explain_df = pd.DataFrame()
        if Config.SHAP_FINAL_EXPLAIN_ENABLED:
            try:
                shap_explain_df = self._tool_shap_final_explain(
                    best_estimator, final_features,
                    top_n=Config.SHAP_FINAL_EXPLAIN_TOP_N,
                )
            except Exception as e:
                self.logger.log(self.name, "SHAP Explain ERROR",
                    f"final-model SHAP explain failed: {e} — continuing without it")

        # ── 11. LLM summary ──────────────────────────────────────────────────
        summary = self._generate_llm_summary(
            best_estimator, best_params, metrics,
            len(feat_avail), len(rfe_features), len(psi_features),
            len(stable_features), len(final_features),
            psi_df, stab_df, prune_df,
            overfit_info=overfit_info,
        )

        # ── 12. Save report ──────────────────────────────────────────────────
        report = {
            "agent": self.name,
            "best_estimator": best_estimator,
            "best_params": best_params,
            "chart_paths": chart_paths,
            "feature_pipeline": {
                "n_init": len(feat_avail),
                "n_after_rfe": len(rfe_features),
                "n_after_psi": len(psi_features),
                "n_after_stability": len(stable_features),
                "n_after_shap_psi": len(final_features),
                "n_final": len(final_features),
                "final_features": final_features,
            },
            "shap_top_features": (
                shap_explain_df.to_dict(orient="records")
                if not shap_explain_df.empty else []
            ),
            "splits": {k: len(v) for k, v in self.splits.items()
                       if isinstance(v, pd.DataFrame)},
            "temporal": self.temporal_meta,
            "metrics": metrics,
            "summary": summary,
        }
        self.save_report(report, Config.MODEL_TRAINER_REPORT_PATH)

        Path(Config.RUN_DIR).mkdir(parents=True, exist_ok=True)
        if not psi_df.empty:
            psi_df.to_csv(Config.PSI_REPORT_PATH, index=False)
        if not stab_df.empty:
            stab_df.to_csv(Config.STABILITY_REPORT_PATH, index=False)
        if not prune_df.empty:
            prune_df.to_csv(Config.SHAP_PSI_PRUNE_LOG_PATH, index=False)

        self.logger.log(self.name, "Process Complete",
            f"best={best_estimator} | features={len(final_features)} | "
            f"oot_auc={metrics.get('oot_auc', 'N/A')} | "
            f"valid_auc={metrics.get('valid_auc', 'N/A')}")
        return metrics, report

    # ── LLM summary ───────────────────────────────────────────────────────────

    def _generate_llm_summary(
        self,
        estimator_name: str,
        best_params: Dict,
        metrics: Dict,
        n_init: int, n_rfe: int, n_psi: int, n_stable: int, n_final: int,
        psi_df: pd.DataFrame,
        stab_df: pd.DataFrame,
        prune_df: pd.DataFrame = None,
        overfit_info: Dict = None,
    ) -> str:
        psi_dropped = int((psi_df["flag"] != "KEEP").sum()) if not psi_df.empty else 0
        stab_dropped = int((stab_df["flag"] != "KEEP").sum()) if not stab_df.empty else 0
        shap_psi_dropped = n_stable - n_final
        prune_steps = len(prune_df) if prune_df is not None and not prune_df.empty else 0

        if overfit_info and overfit_info.get("detected"):
            gap_parts = ", ".join(
                f"{k.replace('_auc', '')}_gap={v:.1%}"
                for k, v in overfit_info.get("gaps", {}).items()
            )
            retrain = overfit_info.get("retrain", "unknown")
            overfit_str = f"DETECTED ({gap_parts}) | retrain={retrain}"
        else:
            overfit_str = "none"

        prompt = self._load_prompt(
            self._PROMPTS_ROOT / "TrainModel" / "prompts" / "llm_summary_user.txt",
            N_INIT=n_init,
            N_RFE=n_rfe,
            PSI_DROPPED=psi_dropped,
            N_PSI=n_psi,
            STAB_DROPPED=stab_dropped,
            N_STABLE=n_stable,
            PRUNE_STEPS=prune_steps,
            SHAP_PSI_DROPPED=shap_psi_dropped,
            N_FINAL=n_final,
            ESTIMATOR_NAME=estimator_name,
            METRICS_JSON=json.dumps({k: v for k, v in metrics.items() if k not in ("best_params",)}, indent=2),
            HYPERPARAMS_JSON=json.dumps({k: v for k, v in list(best_params.items())[:6]}, indent=2),
            OVERFIT_INFO=overfit_str,
        )
        system_prompt = self._load_prompt(
            self._PROMPTS_ROOT / "TrainModel" / "prompts" / "llm_summary_system.txt",
        )
        return self.call_llm(prompt, system_prompt)
