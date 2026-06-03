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

    def __init__(self, logger: AgentLogger):
        super().__init__(name="TrainModel", role="Advanced ML Pipeline Engineer", logger=logger)
        self.df: pd.DataFrame = None
        self.target_column: str = None
        self.date_col: Optional[str] = None
        self.id_col: Optional[str] = None
        self.splits: Dict[str, pd.DataFrame] = {}
        self.feature_cols: List[str] = []
        self._cat_encoders: Dict[str, Any] = {}
        self.model: Any = None

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
        """Fit encoders on train data and transform. Call once on X_train."""
        from sklearn.preprocessing import LabelEncoder
        self._cat_encoders = {}
        out = df.copy()
        for col in out.columns:
            if out[col].dtype == object or hasattr(out[col].dtype, "categories"):
                le = LabelEncoder()
                series = out[col].fillna("__NA__").astype(str)
                # Always include "__NA__" so valid/OOT NaN values can be encoded
                # even when train has no NaN for this column
                le.fit(sorted(set(series.tolist()) | {"__NA__"}))
                out[col] = le.transform(series)
                self._cat_encoders[col] = le
            else:
                out[col] = out[col].fillna(-999)
        return out

    def _transform_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using encoders fitted on train. Use for valid/oot splits."""
        out = df.copy()
        for col in out.columns:
            if col in self._cat_encoders:
                le = self._cat_encoders[col]
                known = set(le.classes_)
                vals = out[col].fillna("__NA__").astype(str)
                # Unseen labels → "__NA__" (must exist in le.classes_ from fit)
                out[col] = le.transform(vals.apply(lambda x: x if x in known else "__NA__"))
            else:
                out[col] = out[col].fillna(-999)
        return out

    # ── Model class lookup ────────────────────────────────────────────────────

    def _get_n_estimators_key(self, estimator_name: str) -> str:
        """CatBoost uses 'iterations', all others use 'n_estimators'."""
        return "iterations" if estimator_name in ("catboost", "CatBoost") else "n_estimators"

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

        Tiers:
          Small  : rows ≤ 90 000 AND cols < 100  → 120 s
          Medium : rows ≤ 200 000 AND cols ≤ 400 → 120 – 360 s (interpolated)
          Large  : above medium                  → 360 – 1 000 s (interpolated)
        """
        if n_rows <= 90_000 and n_cols < 100:
            t = 120
        elif n_rows <= 200_000 and n_cols <= 400:
            row_ratio = max(0.0, (n_rows - 90_000) / (200_000 - 90_000))
            col_ratio = max(0.0, (n_cols - 100) / (400 - 100))
            t = int(120 + max(row_ratio, col_ratio) * (360 - 120))
        else:
            row_ratio = min(1.0, max(0.0, (n_rows - 200_000) / 800_000))
            col_ratio = min(1.0, max(0.0, (n_cols - 400) / 1_600))
            t = int(360 + max(row_ratio, col_ratio) * (1_000 - 360))
        return t, t

    # ── Step 1: Data split ────────────────────────────────────────────────────

    @staticmethod
    def _stratified_split(df: pd.DataFrame, target: str, test_size: float, rs: int):
        """Stratified split with automatic fallback to unstratified when a class is too small."""
        from sklearn.model_selection import train_test_split as _split
        try:
            return _split(df, test_size=test_size, stratify=df[target], random_state=rs)
        except ValueError:
            return _split(df, test_size=test_size, stratify=None, random_state=rs)

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
            # Compute year-month as a standalone Series — avoids df.copy() and the
            # later .drop(columns=["_ym"]) which each allocate a full copy of the frame.
            _dt = pd.to_datetime(df[self.date_col], errors="coerce")
            ym  = _dt.dt.to_period("M")

            nat_count = int(_dt.isna().sum())
            if nat_count:
                self.logger.log(self.name, "WARN",
                    f"{nat_count} rows have unparseable dates in '{self.date_col}' — "
                    "treated as non-OOT (assigned to train pool)")
            sorted_months = sorted(ym.dropna().unique())
            valid_rows = total - nat_count

            # B2 fix: need >= 2 distinct months to create a meaningful OOT set
            if len(sorted_months) < 2:
                self.logger.log(self.name, "WARN",
                    f"Only {len(sorted_months)} distinct month(s) in '{self.date_col}' — "
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
                        f"reduced to {n_oot} month(s) = {oot_ratio:.1%}. "
                        "Data may have coarse month granularity or heavy recency bias.")

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

                    # Valid-temporal: indices of pool rows closest to OOT boundary by date
                    valid_size = max(1, int(round(pool_size * Config.TRAIN_TEST_SPLIT_SIZE)))
                    n_temporal = max(1, int(round(valid_size * Config.VALID_TEMPORAL_RATIO)))
                    pool_orig_idx = df.index[mask_pool]
                    temporal_orig_idx = set(
                        _dt.loc[mask_pool].sort_values(ascending=False).iloc[:n_temporal].index
                    )
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

                    self.logger.log(self.name, "Split (OOT)",
                        f"oot_months={n_oot} | oot_ratio={len(oot_df)/total:.1%} | "
                        f"train={len(train_df)} | valid_temp={len(valid_temp_df)} | "
                        f"valid_rand={len(valid_rand_df)} | oot={len(oot_df)}")
                    return {
                        "train": train_df, "valid_temporal": valid_temp_df,
                        "valid_random": valid_rand_df, "valid": valid_df,
                        "oot": oot_df, "test": empty.copy(),
                    }

        # Fallback: no date col or fell through from OOT path
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

    def _build_optuna_params(self, estimator_name: str, trial) -> Dict:
        gpu = self._gpu_params(estimator_name)
        if estimator_name in ("lgbm", "LightGBM"):
            return {
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
        if estimator_name in ("xgboost", "XGBoost"):
            return {
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
        if estimator_name in ("catboost", "CatBoost"):
            return {
                "iterations": trial.suggest_int("iterations", Config.N_ESTIMATORS_MIN, Config.N_ESTIMATORS_MAX, step=50),
                "learning_rate": trial.suggest_float("learning_rate", Config.LR_MIN, Config.LR_MAX, log=True),
                "depth": trial.suggest_int("depth", Config.CB_DEPTH_MIN, Config.CB_DEPTH_MAX),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
                "random_seed": Config.RANDOM_STATE, "verbose": 0,
                **gpu,
            }
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

        def objective(trial):
            params = _self._build_optuna_params(_estimator_name, trial)
            model = ModelClass(**params)
            model.fit(X_train, y_train)
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
            full_best_params = self._build_optuna_params(estimator_name, FixedTrial(study.best_params))
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
        n_target = min(Config.MAX_FINAL_FEATURES, X_train.shape[1])
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
        tmp["_ym"] = tmp[self.date_col].dt.to_period("M")
        periods = sorted(tmp["_ym"].dropna().unique())

        if len(periods) < Config.STABILITY_MIN_MONTHS:
            self.logger.log(self.name, "Stability",
                f"Only {len(periods)} months < min={Config.STABILITY_MIN_MONTHS} → skip")
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
        def _fit_model(features: List[str]):
            avail = [c for c in features if c in X_train.columns and c in X_valid.columns]
            if not avail:
                return None, avail
            m = ModelClass(**{**self._safe_params(estimator_name, best_params), n_est_key: Config.SHAP_N_ESTIMATORS, **gpu})
            m.fit(X_train[avail], y_train)
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
            f"min_features={min_features} | psi_available={bool(psi_lookup)}")

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

            if improved:
                best_features = list(current_features)
                best_auc = candidate_auc
                no_improve_streak = 0
                self.logger.log(self.name, "SHAP+PSI",
                    f"step={step} removed '{worst_feat}' | "
                    f"n={len(current_features)} | auc={candidate_auc:.4f}")
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
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        target = self.target_column
        ModelClass = self._get_model_class(estimator_name)

        def _prep(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
            avail = [c for c in feature_cols if c in df.columns]
            return self._transform_X(df[avail]), pd.to_numeric(df[target], errors="coerce")

        X_tr, y_tr = _prep(self.splits["train"])
        model = ModelClass(**{**self._safe_params(estimator_name, best_params), **self._gpu_params(estimator_name)})
        model.fit(X_tr, y_tr)
        self.model = model

        metrics: Dict[str, Any] = {"best_model": estimator_name, "best_params": best_params}

        cv = StratifiedKFold(n_splits=Config.CV_N_SPLITS, shuffle=True, random_state=Config.RANDOM_STATE)
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        metrics["cv_auc_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cv_auc_std"] = round(float(cv_scores.std()), 4)
        self.logger.log(self.name, "CV AUC",
            f"{metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")

        for split_name in ("valid_temporal", "valid_random", "valid", "oot", "test"):
            split_df = self.splits.get(split_name, pd.DataFrame())
            if len(split_df) == 0:
                continue
            X_s, y_s = _prep(split_df)
            if y_s.nunique() < 2:
                continue
            try:
                auc = roc_auc_score(y_s, model.predict_proba(X_s)[:, 1])
                metrics[f"{split_name}_auc"] = round(float(auc), 4)
                self.logger.log(self.name, f"{split_name} AUC", f"{auc:.4f}")
            except Exception as e:
                self.logger.log(self.name, f"{split_name} AUC error", str(e))

        return metrics

    # ── Overfitting detection & remediation ──────────────────────────────────

    @staticmethod
    def _check_overfitting(metrics: Dict, threshold: float = 0.12) -> Dict:
        """Return overfitting diagnosis.

        Gap = (valid_auc - holdout_auc) / valid_auc.  detected=True when any
        holdout gap exceeds threshold.
        """
        valid_auc = metrics.get("valid_auc") or metrics.get("valid_temporal_auc") or 0.0
        gaps: Dict[str, float] = {}
        if valid_auc > 0:
            for key in ("test_auc", "oot_auc"):
                val = metrics.get(key)
                if isinstance(val, float):
                    gap = (valid_auc - val) / valid_auc
                    if gap > threshold:
                        gaps[key] = round(gap, 4)
        return {"detected": bool(gaps), "gaps": gaps, "valid_auc": round(valid_auc, 4)}

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

    def _save_model(self, feature_cols: List[str]) -> None:
        import joblib
        artifact = {
            "model":          self.model,
            "feature_cols":   feature_cols,
            "cat_encoders":   self._cat_encoders,
            "target_column":  self.target_column,
            "estimator_name": self.model.__class__.__name__,
        }
        Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)
        joblib.dump(artifact, Config.FINAL_MODEL_PATH)
        self.logger.log(self.name, "Model Saved", Config.FINAL_MODEL_PATH)

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

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ── Load artifact ─────────────────────────────────────────────────────────────
_ARTIFACT_PATH = Path(__file__).parent / "final_model.pkl"
_artifact      = joblib.load(_ARTIFACT_PATH)

model         = _artifact["model"]
feature_cols  = _artifact["feature_cols"]
cat_encoders  = _artifact["cat_encoders"]
target_column = _artifact["target_column"]

# ── Feature list (recorded at training time) ──────────────────────────────────
FEATURES = {features_repr}

# Categorical columns encoded via LabelEncoder
CAT_COLS = {cat_cols}

# Best hyperparameters
BEST_PARAMS = {params_repr}


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same encoding + imputation used during training."""
    avail = [c for c in FEATURES if c in df.columns]
    out = df[avail].copy()
    for col in out.columns:
        if col in cat_encoders:
            le    = cat_encoders[col]
            known = set(le.classes_)
            vals  = out[col].fillna("__NA__").astype(str)
            out[col] = le.transform(vals.apply(lambda x: x if x in known else "__NA__"))
        else:
            out[col] = out[col].fillna(-999)
    return out


# ── Prediction helpers ────────────────────────────────────────────────────────
def predict_proba(df: pd.DataFrame) -> np.ndarray:
    """Return probability of the positive class (shape: n_samples,)."""
    return model.predict_proba(preprocess(df))[:, 1]


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
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.logger.log(self.name, "Process Start", f"Shape={df.shape}")
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
                f"valid_auc={overfit_info['valid_auc']:.4f} | {gap_log} "
                f"(threshold={Config.OVERFIT_THRESHOLD:.0%}) → requesting LLM-guided retrain")

            import io as _io
            import joblib as _jl
            _buf = _io.BytesIO()
            _jl.dump(self.model, _buf)
            _orig_model_bytes = _buf.getvalue()
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
                self.model = _jl.load(_io.BytesIO(_orig_model_bytes))
                best_params = orig_best_params
                self.logger.log(self.name, "Overfit Retry Reverted",
                    f"Retrain degraded {_holdout_key}: "
                    f"{retry_holdout:.4f} < {orig_holdout:.4f} → original model restored")
                overfit_info["retrain"] = "reverted"
        else:
            overfit_info["retrain"] = "not_needed"

        # ── Save model artifact + inference code ──────────────────────────────
        self._save_model(final_features)
        self._generate_model_code(best_estimator, best_params, final_features)

        # ── 9. LLM summary ────────────────────────────────────────────────────
        summary = self._generate_llm_summary(
            best_estimator, best_params, metrics,
            len(feat_avail), len(rfe_features), len(psi_features),
            len(stable_features), len(final_features),
            psi_df, stab_df, prune_df,
            overfit_info=overfit_info,
        )

        # ── 10. Save report ───────────────────────────────────────────────────
        report = {
            "agent": self.name,
            "best_estimator": best_estimator,
            "best_params": best_params,
            "feature_pipeline": {
                "n_init": len(feat_avail),
                "n_after_rfe": len(rfe_features),
                "n_after_psi": len(psi_features),
                "n_after_stability": len(stable_features),
                "n_after_shap_psi": len(final_features),
                "n_final": len(final_features),
                "final_features": final_features,
            },
            "splits": {k: len(v) for k, v in self.splits.items()
                       if isinstance(v, pd.DataFrame)},
            "metrics": metrics,
            "summary": summary,
        }
        self.save_report(report, Config.MODEL_TRAINER_REPORT_PATH)

        Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)
        if not psi_df.empty:
            psi_df.to_csv(f"{Config.OUTPUT_DIR}/psi_report.csv", index=False)
        if not stab_df.empty:
            stab_df.to_csv(f"{Config.OUTPUT_DIR}/stability_report.csv", index=False)
        if not prune_df.empty:
            prune_df.to_csv(f"{Config.OUTPUT_DIR}/shap_psi_prune_log.csv", index=False)

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
