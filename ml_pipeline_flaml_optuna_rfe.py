"""
================================================================================
ML PIPELINE: FLAML → Optuna → RFE + PSI + Stability Check
================================================================================
Luồng chính:
  0.  Config
  1.  Load toàn bộ data từ Spark (1 table duy nhất)
  2.  Split 3 tập: OOT / Valid-Temporal / Train+Valid-Random
        ┌─────────────────────────────────────────────────────────┐
        │  Toàn bộ data (sort theo date ↑)                        │
        │                                                          │
        │  ├──── Train+Valid pool (~80%) ──┤── OOT (~≥20%) ──┤   │
        │         ↓                                                │
        │  ├─ Train (80% pool) ─┤ Valid (20% pool)               │
        │                              ↓                          │
        │                   ├ Valid-Random (80% of valid) ┤       │
        │                   ├ Valid-Temporal (20% of valid)┤      │
        │                     (gần OOT nhất, không lẫn train)     │
        └─────────────────────────────────────────────────────────┘
  3.  FLAML AutoML     → best_estimator + best_params
  4.  Optuna           → fine-tune hyperparams
  5.  RFE              → loại feature yếu (~150 còn lại)
  6.  PSI              → loại feature drift mạnh (Train vs OOT)
  7.  Stability Check  → loại feature Gini không ổn định theo tháng
  8.  Final cut        → top-100 bằng importance nếu vẫn còn nhiều
  9.  Final Model      → train + đánh giá CV / Valid / OOT
  10. Save outputs
================================================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, Dict, List

# ── PySpark ──────────────────────────────────────────────────────────────────
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# ── ML ───────────────────────────────────────────────────────────────────────
from flaml import AutoML
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import roc_auc_score

# ── Optuna ───────────────────────────────────────────────────────────────────
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Models ───────────────────────────────────────────────────────────────────
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# ── Logging ──────────────────────────────────────────────────────────────────
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  0. CONFIG                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

CFG: Dict = {
    # ── Data ─────────────────────────────────────────────────────────────
    "source_table"       : "db.full_data_table",  # 1 Spark table chứa toàn bộ data
    "target_col"         : "label",               # cột nhãn (0/1)
    "date_col"           : "snap_dt",             # cột ngày (dùng để split)
    "id_col"             : "customer_id",         # cột ID (loại trước khi train)
    "exclude_cols"       : [],                    # cột khác cần loại thủ công

    # ── Split strategy ───────────────────────────────────────────────────
    #   OOT  : lấy N tháng cuối sao cho >= oot_min_ratio tổng data
    #   Valid : 20% của pool (train+valid)
    #     └─ valid_temporal_ratio: phần trong valid lấy theo thời gian (gần OOT nhất)
    #     └─ phần còn lại của valid: stratified random từ pool còn lại
    "oot_init_months"    : 2,        # bắt đầu thử 2 tháng cuối cho OOT
    "oot_min_ratio"      : 0.20,     # OOT phải chiếm ít nhất 20% tổng data
    "valid_ratio"        : 0.20,     # valid = 20% của pool (sau khi bỏ OOT)
    "valid_temporal_ratio": 0.20,    # 20% của valid lấy theo thời gian (không random)

    # ── FLAML ────────────────────────────────────────────────────────────
    "flaml_time_budget"  : 600,
    "flaml_estimators"   : ["lgbm", "xgboost", "rf", "extra_tree"],
    "flaml_metric"       : "roc_auc",
    "flaml_task"         : "classification",
    "flaml_n_splits"     : 5,

    # ── Optuna ───────────────────────────────────────────────────────────
    "optuna_n_trials"    : 100,
    "optuna_timeout"     : 300,
    "optuna_cv_splits"   : 5,
    "optuna_metric"      : "roc_auc",

    # ── RFE ──────────────────────────────────────────────────────────────
    "rfe_target_features": 150,
    "rfe_step"           : 0.05,
    "rfe_cv_splits"      : 3,
    "enable_rfecv"       : False,

    # ── PSI ──────────────────────────────────────────────────────────────
    "psi_threshold"      : 0.25,
    "psi_bins"           : 10,

    # ── Stability ────────────────────────────────────────────────────────
    "stability_min_months"          : 6,
    "stability_gini_std_threshold"  : 0.10,
    "stability_min_gini"            : 0.02,

    # ── Final ────────────────────────────────────────────────────────────
    "max_final_features" : 100,
    "output_dir"         : "./outputs",
    "random_state"       : 42,
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1. SPARK UTILS                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("ML_Pipeline_FLAML_Optuna_RFE")
        .config("spark.sql.shuffle.partitions", "400")
        .config("spark.driver.memory", "16g")
        .enableHiveSupport()
        .getOrCreate()
    )


def load_spark_to_pandas(spark: SparkSession, table: str, cfg: Dict) -> pd.DataFrame:
    """Load Spark table → Pandas. Giữ lại id_col & date_col để dùng khi split."""
    sdf = spark.table(table)
    # Chỉ drop exclude_cols (giữ id_col/date_col để split rồi drop sau)
    drop_now = [c for c in cfg["exclude_cols"] if c in sdf.columns]
    if drop_now:
        sdf = sdf.drop(*drop_now)
    # Cast numeric → double
    for col_name, dtype in sdf.dtypes:
        if dtype in ("bigint", "int", "float", "decimal"):
            sdf = sdf.withColumn(col_name, F.col(col_name).cast(DoubleType()))
    pdf = sdf.toPandas()
    pdf[cfg["date_col"]] = pd.to_datetime(pdf[cfg["date_col"]])
    return pdf


def get_feature_cols(df: pd.DataFrame, cfg: Dict) -> List[str]:
    non_feat = {cfg["target_col"], cfg["date_col"], cfg["id_col"]} | set(cfg["exclude_cols"])
    return [c for c in df.columns if c not in non_feat]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2. DATA SPLIT: OOT / Valid-Temporal / Train / Valid-Random              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
#  Sơ đồ chi tiết:
#
#  ┌────────── Toàn bộ data, sort theo date ──────────────────────────────┐
#  │  Tháng T-n  ...  T-3  │  T-2  │  T-1  │  T (latest)               │
#  └──────────────────────────────────────────────────────────────────────┘
#       ↑                    ↑
#       Pool (train+valid)   OOT bắt đầu từ đây
#                            (mở rộng thêm tháng nếu < 20% tổng)
#
#  Pool (sau khi bỏ OOT):
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  [------------ Train (80%) ----------] [------ Valid (20%) ------]  │
#  │                                         ├─ Temporal (20% of valid) ┤│
#  │                                         │  (N tháng gần OOT nhất)  ││
#  │                                         ├─ Random   (80% of valid) ┤│
#  │                                         │  (stratified từ pool)    ││
#  └─────────────────────────────────────────────────────────────────────┘
#
#  Quy tắc quan trọng:
#  - Valid-Temporal: lấy theo thời gian (gần OOT nhất), KHÔNG lẫn vào train
#  - Valid-Random  : stratified random từ phần pool còn lại sau khi tách Temporal
#  - Train         : toàn bộ pool còn lại sau khi bỏ Valid-Temporal + Valid-Random
# ──────────────────────────────────────────────────────────────────────────────

def _get_oot_cutoff_date(
    df: pd.DataFrame,
    date_col: str,
    oot_init_months: int,
    oot_min_ratio: float,
) -> pd.Timestamp:
    """
    Xác định ngày bắt đầu OOT:
      - Bắt đầu với N tháng cuối (oot_init_months)
      - Nếu số record OOT < oot_min_ratio * total → mở rộng thêm 1 tháng
      - Lặp cho đến khi đủ tỉ lệ
    """
    df["_ym"] = df[date_col].dt.to_period("M")
    sorted_months = sorted(df["_ym"].unique())  # tăng dần
    total = len(df)

    n_oot_months = oot_init_months
    while n_oot_months < len(sorted_months):
        oot_months   = sorted_months[-n_oot_months:]
        oot_size     = df[df["_ym"].isin(oot_months)].shape[0]
        oot_ratio    = oot_size / total
        if oot_ratio >= oot_min_ratio:
            break
        n_oot_months += 1

    cutoff_period = sorted_months[-n_oot_months]
    cutoff_date   = cutoff_period.to_timestamp()  # ngày đầu tháng OOT đầu tiên

    oot_size  = df[df["_ym"].isin(sorted_months[-n_oot_months:])].shape[0]
    log.info(
        "OOT: %d tháng cuối (%s → %s) | %d records | %.1f%% tổng data",
        n_oot_months,
        str(sorted_months[-n_oot_months]),
        str(sorted_months[-1]),
        oot_size,
        oot_size / total * 100,
    )
    df.drop(columns=["_ym"], inplace=True)
    return cutoff_date


def _get_valid_temporal_cutoff(
    pool_df: pd.DataFrame,
    date_col: str,
    valid_size: int,
    valid_temporal_ratio: float,
) -> pd.Timestamp:
    """
    Tính ngày cutoff để lấy valid_temporal = valid_temporal_ratio * valid_size
    record gần OOT nhất từ pool.
    Trả về ngày bắt đầu của valid-temporal (lấy từ đây đến hết pool).
    """
    n_temporal = max(1, int(round(valid_size * valid_temporal_ratio)))
    # Sắp xếp theo date desc, lấy n_temporal record cuối cùng của pool
    pool_sorted = pool_df.sort_values(date_col, ascending=False)
    cutoff_date = pool_sorted.iloc[n_temporal - 1][date_col]  # ngày nhỏ nhất trong temporal

    log.info(
        "Valid-Temporal: %d records (%.0f%% of valid) | từ %s trở đi",
        n_temporal,
        valid_temporal_ratio * 100,
        str(cutoff_date.date()),
    )
    return cutoff_date


def split_data(df: pd.DataFrame, cfg: Dict) -> Dict[str, pd.DataFrame]:
    """
    Tách data thành 4 tập:
      - oot           : N tháng cuối, ≥ 20% tổng
      - valid_temporal: gần OOT nhất, không random, không lẫn train
      - valid_random  : stratified random từ pool còn lại
      - train         : phần còn lại của pool

    Trả về dict với keys: train / valid_temporal / valid_random / valid / oot
    (valid = valid_temporal ∪ valid_random để đánh giá tổng thể)
    """
    log.info("=" * 60)
    log.info("STEP 0 – Data Split (tổng: %d records)", len(df))
    log.info("=" * 60)

    date_col  = cfg["date_col"]
    target    = cfg["target_col"]
    rs        = cfg["random_state"]

    df = df.copy().reset_index(drop=True)
    df[date_col] = pd.to_datetime(df[date_col])

    # ── 1. Tách OOT ──────────────────────────────────────────────────────
    oot_cutoff = _get_oot_cutoff_date(
        df, date_col,
        cfg["oot_init_months"],
        cfg["oot_min_ratio"],
    )
    mask_oot = df[date_col] >= oot_cutoff
    oot_df   = df[mask_oot].reset_index(drop=True)
    pool_df  = df[~mask_oot].reset_index(drop=True)

    pool_total = len(pool_df)
    valid_size = max(1, int(round(pool_total * cfg["valid_ratio"])))

    # ── 2. Tách Valid-Temporal từ pool ───────────────────────────────────
    temporal_cutoff = _get_valid_temporal_cutoff(
        pool_df, date_col, valid_size, cfg["valid_temporal_ratio"]
    )
    # Lấy tất cả record trong pool có date >= temporal_cutoff
    # nhưng chỉ lấy đúng n_temporal record gần nhất (tránh lấy dư nếu nhiều record cùng ngày)
    n_temporal = max(1, int(round(valid_size * cfg["valid_temporal_ratio"])))
    pool_sorted   = pool_df.sort_values(date_col, ascending=False)
    temporal_idx  = pool_sorted.iloc[:n_temporal].index
    mask_temporal = pool_df.index.isin(temporal_idx)

    valid_temp_df = pool_df[mask_temporal].reset_index(drop=True)
    remain_df     = pool_df[~mask_temporal].reset_index(drop=True)

    # ── 3. Stratified random split: Train vs Valid-Random ────────────────
    # n_valid_random = valid_size - n_temporal
    n_valid_random = valid_size - len(valid_temp_df)
    valid_random_ratio = n_valid_random / len(remain_df) if len(remain_df) > 0 else 0.0

    if valid_random_ratio <= 0 or valid_random_ratio >= 1:
        log.warning(
            "valid_random_ratio=%.4f nằm ngoài (0,1), bỏ qua Valid-Random split.",
            valid_random_ratio,
        )
        train_df       = remain_df
        valid_random_df = pd.DataFrame(columns=remain_df.columns)
    else:
        train_df, valid_random_df = train_test_split(
            remain_df,
            test_size=valid_random_ratio,
            stratify=remain_df[target],
            random_state=rs,
        )
        train_df        = train_df.reset_index(drop=True)
        valid_random_df = valid_random_df.reset_index(drop=True)

    # ── 4. Ghép valid = temporal + random ────────────────────────────────
    valid_df = pd.concat([valid_temp_df, valid_random_df], ignore_index=True)

    # ── 5. Log phân phối ─────────────────────────────────────────────────
    total = len(df)
    for name, sub in [
        ("Train",          train_df),
        ("Valid-Random",   valid_random_df),
        ("Valid-Temporal", valid_temp_df),
        ("Valid (total)",  valid_df),
        ("OOT",            oot_df),
    ]:
        if len(sub) == 0:
            continue
        ev_rate = sub[target].mean() * 100 if target in sub.columns else float("nan")
        date_min = sub[date_col].min().date() if date_col in sub.columns else "?"
        date_max = sub[date_col].max().date() if date_col in sub.columns else "?"
        log.info(
            "  %-18s : %6d rows (%5.1f%%) | event_rate=%5.2f%% | %s → %s",
            name, len(sub), len(sub) / total * 100,
            ev_rate, date_min, date_max,
        )

    return {
        "train"          : train_df,
        "valid_temporal" : valid_temp_df,
        "valid_random"   : valid_random_df,
        "valid"          : valid_df,
        "oot"            : oot_df,
        "oot_cutoff_date": oot_cutoff,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3. FLAML AUTO-ML                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_flaml(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cfg: Dict,
) -> Tuple[str, Dict, AutoML]:
    log.info("=" * 60)
    log.info("STEP 1 – FLAML AutoML (time_budget=%ss)", cfg["flaml_time_budget"])
    log.info("=" * 60)

    automl   = AutoML()
    settings = {
        "time_budget"        : cfg["flaml_time_budget"],
        "metric"             : cfg["flaml_metric"],
        "task"               : cfg["flaml_task"],
        "estimator_list"     : cfg["flaml_estimators"],
        "n_splits"           : cfg["flaml_n_splits"],
        "seed"               : cfg["random_state"],
        "verbose"            : 1,
        "eval_method"        : "cv",       # dùng CV nội bộ trong FLAML
        "log_training_metric": True,
        # Truyền valid để FLAML đánh giá thêm (không dùng cho tuning)
        "X_val"              : X_valid,
        "y_val"              : y_valid,
    }

    automl.fit(X_train, y_train, **settings)

    best_estimator = automl.best_estimator
    best_config    = automl.best_config
    best_loss      = automl.best_loss

    # Đánh giá trên valid
    valid_pred = automl.predict_proba(X_valid)[:, 1]
    valid_auc  = roc_auc_score(y_valid, valid_pred)

    log.info("✔ FLAML best estimator  : %s", best_estimator)
    log.info("✔ FLAML best CV loss    : %.6f  (metric=%s)", best_loss, cfg["flaml_metric"])
    log.info("✔ FLAML Valid AUC       : %.4f", valid_auc)
    log.info("✔ FLAML best params     : %s", best_config)

    return best_estimator, best_config, automl


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4. OPTUNA FINE-TUNING                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _build_lgbm_trial(trial: optuna.Trial, base: Dict) -> Dict:
    return {
        "n_estimators"     : trial.suggest_int("n_estimators", 100, 2000, step=50),
        "learning_rate"    : trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "num_leaves"       : trial.suggest_int("num_leaves",
                               max(8, base.get("num_leaves", 31) // 2),
                               min(base.get("num_leaves", 31) * 4, 512)),
        "max_depth"        : trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 300),
        "subsample"        : trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state"     : 42,
        "n_jobs"           : -1,
        "verbose"          : -1,
    }


def _build_xgb_trial(trial: optuna.Trial, base: Dict) -> Dict:
    return {
        "n_estimators"    : trial.suggest_int("n_estimators", 100, 2000, step=50),
        "learning_rate"   : trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "max_depth"       : trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "subsample"       : trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma"           : trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "use_label_encoder": False,
        "eval_metric"     : "logloss",
        "random_state"    : 42,
        "n_jobs"          : -1,
    }


def _build_rf_trial(trial: optuna.Trial, base: Dict) -> Dict:
    return {
        "n_estimators"    : trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth"       : trial.suggest_int("max_depth", 3, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features"    : trial.suggest_categorical("max_features", ["sqrt", "log2", 0.4, 0.6, 0.8]),
        "random_state"    : 42,
        "n_jobs"          : -1,
    }


TRIAL_BUILDERS: Dict = {
    "lgbm"      : (_build_lgbm_trial, lgb.LGBMClassifier),
    "xgboost"   : (_build_xgb_trial,  xgb.XGBClassifier),
    "rf"        : (_build_rf_trial,    RandomForestClassifier),
    "extra_tree": (_build_rf_trial,    RandomForestClassifier),
}


def run_optuna(
    estimator_name: str,
    base_params: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cfg: Dict,
) -> Tuple[Dict, Optional[optuna.Study]]:
    """
    Fine-tune hyperparams bằng Optuna.
    Objective = AUC trên valid_temporal (không lẫn train) để tránh leakage.
    """
    log.info("=" * 60)
    log.info("STEP 2 – Optuna fine-tuning (%s trials, estimator=%s)",
             cfg["optuna_n_trials"], estimator_name)
    log.info("=" * 60)

    if estimator_name not in TRIAL_BUILDERS:
        log.warning("Estimator '%s' không có Optuna builder → giữ FLAML params.", estimator_name)
        return base_params, None

    builder_fn, ModelClass = TRIAL_BUILDERS[estimator_name]

    def objective(trial: optuna.Trial) -> float:
        params = builder_fn(trial, base_params)
        model  = ModelClass(**params)
        model.fit(X_train, y_train)
        preds  = model.predict_proba(X_valid)[:, 1]
        return roc_auc_score(y_valid, preds)

    sampler = TPESampler(seed=cfg["random_state"])
    study   = optuna.create_study(direction="maximize", sampler=sampler)

    # Warm-start từ FLAML best params
    try:
        study.enqueue_trial({
            k: v for k, v in base_params.items()
            if k in builder_fn(
                optuna.trial.create_trial(
                    params={}, distributions={}, value=0.0
                ),
                base_params,
            )
        })
    except Exception:
        pass

    study.optimize(
        objective,
        n_trials=cfg["optuna_n_trials"],
        timeout=cfg["optuna_timeout"],
        show_progress_bar=True,
    )

    best = study.best_params
    log.info("✔ Optuna best Valid AUC : %.6f", study.best_value)
    log.info("✔ Optuna best params    : %s", best)

    return best, study


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5. RFE – Recursive Feature Elimination                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_rfe(
    estimator_name: str,
    best_params: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: Dict,
) -> List[str]:
    log.info("=" * 60)
    log.info("STEP 3 – RFE (target=%d features, mode=%s)",
             cfg["rfe_target_features"], "RFECV" if cfg["enable_rfecv"] else "RFE")
    log.info("=" * 60)

    if estimator_name not in TRIAL_BUILDERS:
        log.warning("Không có estimator cho RFE → giữ toàn bộ features.")
        return list(X_train.columns)

    _, ModelClass  = TRIAL_BUILDERS[estimator_name]
    base_model     = ModelClass(**{**best_params, "n_estimators": 200})
    n_target       = min(cfg["rfe_target_features"], X_train.shape[1])

    if cfg["enable_rfecv"]:
        cv       = StratifiedKFold(n_splits=cfg["rfe_cv_splits"], shuffle=True,
                                   random_state=cfg["random_state"])
        selector = RFECV(
            estimator=base_model, step=cfg["rfe_step"], cv=cv,
            scoring="roc_auc",
            min_features_to_select=n_target,
            n_jobs=-1,
        )
    else:
        selector = RFE(
            estimator=base_model,
            n_features_to_select=n_target,
            step=cfg["rfe_step"],
        )

    selector.fit(X_train, y_train)
    selected = list(X_train.columns[selector.support_])
    log.info("✔ RFE: %d → %d features", X_train.shape[1], len(selected))

    # Log top-10 importance
    try:
        imp = pd.Series(
            selector.estimator_.feature_importances_,
            index=X_train.columns[selector.support_],
        ).sort_values(ascending=False)
        log.info("Top-10 RFE features:\n%s", imp.head(10).to_string())
    except AttributeError:
        pass

    return selected


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6. PSI – Population Stability Index                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _calc_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    eps  = 1e-8
    cuts = pd.qcut(expected, q=bins, duplicates="drop", retbins=True)[1]
    cuts[0], cuts[-1] = -np.inf, np.inf
    e_pct = pd.cut(expected, bins=cuts).value_counts(normalize=True).sort_index() + eps
    a_pct = pd.cut(actual,   bins=cuts).value_counts(normalize=True).sort_index() + eps
    e_pct, a_pct = e_pct.align(a_pct, fill_value=eps)
    return ((a_pct - e_pct) * np.log(a_pct / e_pct)).sum()


def run_psi_filter(
    feature_cols: List[str],
    train_df: pd.DataFrame,
    oot_df: pd.DataFrame,
    cfg: Dict,
) -> Tuple[List[str], pd.DataFrame]:
    log.info("=" * 60)
    log.info("STEP 4 – PSI Filter (threshold=%.2f)", cfg["psi_threshold"])
    log.info("=" * 60)

    records = []
    for col in feature_cols:
        if col not in train_df.columns or col not in oot_df.columns:
            continue
        try:
            psi_val = _calc_psi(
                train_df[col].dropna(),
                oot_df[col].dropna(),
                bins=cfg["psi_bins"],
            )
        except Exception:
            psi_val = 999.0
        records.append({"feature": col, "psi": psi_val})

    psi_df          = pd.DataFrame(records).sort_values("psi", ascending=False)
    threshold       = cfg["psi_threshold"]
    psi_df["flag"]  = psi_df["psi"].apply(
        lambda x: f"DROP (PSI={x:.3f}>{threshold})" if x > threshold else "KEEP"
    )

    kept    = psi_df.loc[psi_df["flag"] == "KEEP",  "feature"].tolist()
    dropped = psi_df.loc[psi_df["flag"] != "KEEP",  "feature"].tolist()
    log.info("✔ PSI: %d KEEP, %d DROP", len(kept), len(dropped))
    if dropped:
        log.info("   Dropped (top-10): %s", dropped[:10])

    return kept, psi_df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7. STABILITY CHECK (Gini / IV theo tháng)                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _calc_gini(df: pd.DataFrame, feature: str, target: str) -> float:
    try:
        sub = df[[feature, target]].dropna()
        if sub[target].nunique() < 2 or sub[feature].nunique() < 2:
            return 0.0
        auc = roc_auc_score(sub[target], sub[feature])
        return abs(2 * auc - 1)
    except Exception:
        return 0.0


def run_stability_check(
    feature_cols: List[str],
    train_df: pd.DataFrame,
    cfg: Dict,
) -> Tuple[List[str], pd.DataFrame]:
    log.info("=" * 60)
    log.info("STEP 5 – Stability Check (std_thr=%.2f, min_gini=%.2f)",
             cfg["stability_gini_std_threshold"], cfg["stability_min_gini"])
    log.info("=" * 60)

    target = cfg["target_col"]
    date_c = cfg["date_col"]

    if date_c not in train_df.columns:
        log.warning("Không có cột date '%s' → skip Stability Check.", date_c)
        return feature_cols, pd.DataFrame()

    tmp           = train_df.copy()
    tmp[date_c]   = pd.to_datetime(tmp[date_c])
    tmp["_ym"]    = tmp[date_c].dt.to_period("M")
    periods       = sorted(tmp["_ym"].unique())

    if len(periods) < cfg["stability_min_months"]:
        log.warning(
            "Chỉ có %d tháng trong pool (train) < min=%d → skip Stability Check.",
            len(periods), cfg["stability_min_months"],
        )
        return feature_cols, pd.DataFrame()

    records = []
    for col in feature_cols:
        ginis = [_calc_gini(tmp[tmp["_ym"] == p], col, target) for p in periods]
        mean_g, std_g = float(np.mean(ginis)), float(np.std(ginis))
        flag = "KEEP"
        if std_g > cfg["stability_gini_std_threshold"]:
            flag = f"DROP (std_Gini={std_g:.3f})"
        elif mean_g < cfg["stability_min_gini"]:
            flag = f"DROP (mean_Gini={mean_g:.3f})"
        records.append({
            "feature"   : col,
            "mean_gini" : round(mean_g, 4),
            "std_gini"  : round(std_g,  4),
            "flag"      : flag,
            **{str(p): round(g, 4) for p, g in zip(periods, ginis)},
        })

    stab_df  = pd.DataFrame(records).sort_values("mean_gini", ascending=False)
    kept     = stab_df.loc[stab_df["flag"] == "KEEP", "feature"].tolist()
    dropped  = stab_df.loc[stab_df["flag"] != "KEEP", "feature"].tolist()
    log.info("✔ Stability: %d KEEP, %d DROP", len(kept), len(dropped))
    if dropped:
        log.info("   Dropped (top-10): %s", dropped[:10])

    return kept, stab_df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  8. FINAL CUT: Top-N bằng importance                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def pick_top_n_by_importance(
    estimator_name: str,
    best_params: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n: int,
) -> List[str]:
    if X_train.shape[1] <= n:
        return list(X_train.columns)

    log.info("Còn %d features > %d → cắt bằng importance...", X_train.shape[1], n)
    _, ModelClass = TRIAL_BUILDERS.get(estimator_name, TRIAL_BUILDERS["lgbm"])
    m = ModelClass(**{**best_params, "n_estimators": 300})
    m.fit(X_train, y_train)
    imp = pd.Series(m.feature_importances_, index=X_train.columns)
    top = imp.nlargest(n).index.tolist()
    log.info("✔ Giữ lại top-%d features.", n)
    return top


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  9. FINAL MODEL & EVALUATION                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train_final_model(
    estimator_name: str,
    best_params: Dict,
    splits: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    cfg: Dict,
):
    log.info("=" * 60)
    log.info("STEP 6 – Train Final Model  (n_features=%d)", len(feature_cols))
    log.info("=" * 60)

    target = cfg["target_col"]

    def _prep(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        feat_avail = [c for c in feature_cols if c in df.columns]
        return df[feat_avail].fillna(-999), df[target]

    X_tr,  y_tr  = _prep(splits["train"])
    X_vt,  y_vt  = _prep(splits["valid_temporal"])
    X_vr,  y_vr  = _prep(splits["valid_random"])
    X_v,   y_v   = _prep(splits["valid"])
    X_oot, y_oot = _prep(splits["oot"])

    _, ModelClass = TRIAL_BUILDERS.get(estimator_name, TRIAL_BUILDERS["lgbm"])
    model = ModelClass(**best_params)
    model.fit(X_tr, y_tr)

    # ── Evaluate ─────────────────────────────────────────────────────────
    def _auc(X, y, label):
        if len(y.unique()) < 2:
            log.warning("  %s: chỉ có 1 class → skip AUC", label)
            return float("nan")
        score = roc_auc_score(y, model.predict_proba(X)[:, 1])
        log.info("  %-25s AUC = %.4f  (n=%d)", label, score, len(y))
        return score

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["random_state"])
    cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
    log.info("  %-25s AUC = %.4f ± %.4f", "CV (train)", cv_auc.mean(), cv_auc.std())

    metrics = {
        "cv_auc_mean"       : round(cv_auc.mean(), 4),
        "cv_auc_std"        : round(cv_auc.std(),  4),
        "valid_temporal_auc": _auc(X_vt,  y_vt,  "Valid-Temporal"),
        "valid_random_auc"  : _auc(X_vr,  y_vr,  "Valid-Random"),
        "valid_auc"         : _auc(X_v,   y_v,   "Valid (total)"),
        "oot_auc"           : _auc(X_oot, y_oot, "OOT"),
    }

    return model, metrics


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  10. SAVE OUTPUTS                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def save_outputs(
    psi_df: pd.DataFrame,
    stab_df: pd.DataFrame,
    final_features: List[str],
    metrics: Dict,
    split_summary: Dict,
    output_dir: str,
):
    import joblib
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not psi_df.empty:
        p = f"{output_dir}/psi_report_{ts}.csv"
        psi_df.to_csv(p, index=False)
        log.info("Saved → %s", p)

    if not stab_df.empty:
        p = f"{output_dir}/stability_report_{ts}.csv"
        stab_df.to_csv(p, index=False)
        log.info("Saved → %s", p)

    p = f"{output_dir}/final_features_{ts}.csv"
    pd.Series(final_features, name="feature").to_csv(p, index=False)
    log.info("Saved %d final features → %s", len(final_features), p)

    p = f"{output_dir}/metrics_{ts}.csv"
    pd.DataFrame([metrics]).to_csv(p, index=False)
    log.info("Saved metrics → %s", p)

    p = f"{output_dir}/split_summary_{ts}.csv"
    pd.DataFrame([split_summary]).to_csv(p, index=False)
    log.info("Saved split summary → %s", p)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main(cfg: Dict = CFG) -> Dict:
    log.info("▶ Pipeline bắt đầu: %s", datetime.now())

    # ── Spark ─────────────────────────────────────────────────────────────
    spark = get_spark()

    # ── Load toàn bộ data ─────────────────────────────────────────────────
    log.info("Loading data từ %s ...", cfg["source_table"])
    full_df      = load_spark_to_pandas(spark, cfg["source_table"], cfg)
    feature_cols = get_feature_cols(full_df, cfg)
    log.info("Tổng: %d rows | %d features", len(full_df), len(feature_cols))

    # ── Split 3 tập ───────────────────────────────────────────────────────
    splits = split_data(full_df, cfg)

    # Chuẩn bị X/y cho train (feature only)
    train_df = splits["train"]
    X_train  = train_df[feature_cols].fillna(-999)
    y_train  = train_df[cfg["target_col"]]

    # Valid-Temporal dùng cho Optuna (không lẫn train)
    vt_df   = splits["valid_temporal"]
    X_vt    = vt_df[[c for c in feature_cols if c in vt_df.columns]].fillna(-999)
    y_vt    = vt_df[cfg["target_col"]]

    # Valid tổng dùng cho FLAML
    v_df    = splits["valid"]
    X_v     = v_df[[c for c in feature_cols if c in v_df.columns]].fillna(-999)
    y_v     = v_df[cfg["target_col"]]

    # ── FLAML ─────────────────────────────────────────────────────────────
    best_estimator, flaml_params, automl_obj = run_flaml(
        X_train, y_train, X_v, y_v, cfg
    )

    # ── Optuna (dùng valid_temporal làm hold-out, không leak với train) ───
    optuna_params, study = run_optuna(
        best_estimator, flaml_params,
        X_train, y_train,
        X_vt, y_vt,        # ← valid_temporal: gần OOT, không lẫn train
        cfg,
    )
    final_params = {**flaml_params, **optuna_params}

    # ── RFE (chỉ train trên X_train) ──────────────────────────────────────
    rfe_features = run_rfe(best_estimator, final_params, X_train, y_train, cfg)

    # ── PSI (train vs OOT) ────────────────────────────────────────────────
    psi_kept, psi_df = run_psi_filter(
        rfe_features,
        train_df[rfe_features].fillna(-999),
        splits["oot"][[c for c in rfe_features if c in splits["oot"].columns]].fillna(-999),
        cfg,
    )

    # ── Stability (chỉ trên train pool theo tháng) ────────────────────────
    stab_kept, stab_df = run_stability_check(psi_kept, train_df, cfg)

    log.info("Features sau lọc RFE+PSI+Stability: %d", len(stab_kept))

    # ── Cắt xuống ≤ max_final_features ────────────────────────────────────
    final_features = pick_top_n_by_importance(
        best_estimator, final_params,
        X_train[[c for c in stab_kept if c in X_train.columns]], y_train,
        n=cfg["max_final_features"],
    )
    log.info("✔ FINAL features: %d", len(final_features))

    # ── Final model & evaluation ───────────────────────────────────────────
    final_model, metrics = train_final_model(
        best_estimator, final_params,
        splits, final_features, cfg,
    )

    # ── Split summary ──────────────────────────────────────────────────────
    split_summary = {
        "n_train"         : len(splits["train"]),
        "n_valid_temporal": len(splits["valid_temporal"]),
        "n_valid_random"  : len(splits["valid_random"]),
        "n_valid_total"   : len(splits["valid"]),
        "n_oot"           : len(splits["oot"]),
        "oot_cutoff_date" : str(splits["oot_cutoff_date"].date()),
        "n_features_init" : len(feature_cols),
        "n_features_rfe"  : len(rfe_features),
        "n_features_psi"  : len(psi_kept),
        "n_features_stab" : len(stab_kept),
        "n_features_final": len(final_features),
    }

    # ── Save ──────────────────────────────────────────────────────────────
    save_outputs(psi_df, stab_df, final_features, metrics, split_summary, cfg["output_dir"])

    log.info("▶ Pipeline hoàn tất: %s", datetime.now())

    return {
        "final_model"    : final_model,
        "final_features" : final_features,
        "best_estimator" : best_estimator,
        "final_params"   : final_params,
        "splits"         : splits,
        "metrics"        : metrics,
        "split_summary"  : split_summary,
        "psi_df"         : psi_df,
        "stability_df"   : stab_df,
        "optuna_study"   : study,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ENTRYPOINT                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    results = main(CFG)

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    ss = results["split_summary"]
    print(f"  OOT cutoff        : {ss['oot_cutoff_date']}")
    print(f"  Train             : {ss['n_train']:,} rows")
    print(f"  Valid-Random      : {ss['n_valid_random']:,} rows")
    print(f"  Valid-Temporal    : {ss['n_valid_temporal']:,} rows")
    print(f"  OOT               : {ss['n_oot']:,} rows")
    print(f"  Features: init={ss['n_features_init']} → RFE={ss['n_features_rfe']}"
          f" → PSI={ss['n_features_psi']} → Stability={ss['n_features_stab']}"
          f" → Final={ss['n_features_final']}")
    print()
    m = results["metrics"]
    print(f"  CV AUC            : {m['cv_auc_mean']:.4f} ± {m['cv_auc_std']:.4f}")
    print(f"  Valid-Temporal AUC: {m['valid_temporal_auc']:.4f}")
    print(f"  Valid-Random AUC  : {m['valid_random_auc']:.4f}")
    print(f"  Valid (total) AUC : {m['valid_auc']:.4f}")
    print(f"  OOT AUC           : {m['oot_auc']:.4f}")
    print("=" * 60)
