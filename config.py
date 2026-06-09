import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Redirect joblib's memory-map staging dir off /dev/shm — that filesystem is tiny
# (default 64 MB inside Docker) and silently truncates the .pkl files joblib hands
# to workers, surfacing as "BrokenProcessPool / FileNotFoundError" mid-CV. The
# system tempdir is disk-backed and effectively unbounded.
# Must be set BEFORE any sklearn / joblib / FLAML import — kept here because
# config.py is the first module every entry point imports. Respects user override
# via .env (setdefault leaves an existing value alone).
os.environ.setdefault("JOBLIB_TEMP_FOLDER", tempfile.gettempdir())


class Config:
    # Backend marker — overridden by GatewayConfig when LLM_BACKEND=gateway.
    BACKEND: str = "legacy"

    # ── LLM Endpoints ────────────────────────────────────────────────────────
    LITELLM_URL: str = os.getenv("LITELLM_URL", "http://localhost:4000")
    API_KEY: str = os.getenv("API_KEY", "anything")
    LOCAL_MODEL: str = os.getenv("LOCAL_MODEL", "local-model")
    CLOUD_MODEL: str = os.getenv("CLOUD_MODEL", "cloud-model")
    TIMEOUT: int = int(os.getenv("TIMEOUT", 60))

    # ── LLM Behaviour ────────────────────────────────────────────────────────
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.2))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 2000))
    # top_p: nucleus sampling — 0.95 keeps 95% probability mass, filters low-prob tokens
    LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", 0.95))
    # frequency_penalty: reduces word repetition in output (0.0–2.0)
    LLM_FREQUENCY_PENALTY: float = float(os.getenv("LLM_FREQUENCY_PENALTY", 0.1))
    # presence_penalty: encourages introducing new topics (0.0–2.0)
    LLM_PRESENCE_PENALTY: float = float(os.getenv("LLM_PRESENCE_PENALTY", 0.0))
    # seed: fixed seed for reproducible LLM outputs; None = disabled
    LLM_SEED: Optional[int] = int(os.getenv("LLM_SEED")) if os.getenv("LLM_SEED") else None
    # Max retries on rate-limit (429) or server errors (5xx) before giving up
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", 3))
    # Higher token budget for tasks that produce long output (FE decisions, model code generation)
    LLM_MAX_TOKENS_LARGE: int = int(os.getenv("LLM_MAX_TOKENS_LARGE", 4000))
    # Base delay in seconds for exponential backoff between retries
    LLM_RETRY_DELAY: float = float(os.getenv("LLM_RETRY_DELAY", 2.0))
    # Prompts longer than this (chars) are routed to the cloud model
    MODEL_ROUTING_THRESHOLD: int = int(os.getenv("MODEL_ROUTING_THRESHOLD", 6000))

    # ── Output Paths ─────────────────────────────────────────────────────────
    # Output layout (set by init_run() at pipeline construction):
    #   OUTPUT_DIR/<YYYY-MM-DD>/run_<NN>/         ← RUN_DIR (persisted)
    #     ├── agent_execution.log
    #     ├── data_cleaner_report.json
    #     ├── feature_engineer_report.json
    #     ├── model_trainer_report.json
    #     ├── final_report.md
    #     ├── final_model.pkl + final_model_code.py
    #     ├── psi_report.csv + stability_report.csv + shap_psi_prune_log.csv
    #     └── pipeline_process_<agent>.py            ← per-agent replay scripts
    #   <system tempdir>/automl_pipeline_<rand>/   ← TMP_DIR (cleaned at end)
    #     └── clean_*.parquet + engineered_*.parquet (intermediate handoffs)
    #
    # Set KEEP_INTERMEDIATES=true to redirect TMP_DIR → RUN_DIR (debugging, tests).
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "outputs")
    KEEP_INTERMEDIATES: bool = os.getenv("KEEP_INTERMEDIATES", "false").lower() == "true"
    # Minimum free disk space (GB) on OUTPUT_DIR's filesystem at pipeline start.
    # Pipeline raises if below this — prevents mid-run write failures on production
    # batch jobs. Default 2 GB is enough for a typical run; bump for very wide
    # datasets where engineered_*.parquet + final reports exceed that.
    MIN_DISK_FREE_GB: float = float(os.getenv("MIN_DISK_FREE_GB", 2.0))

    # Populated by init_run() — defaults so import-time access does not crash.
    RUN_DIR: str = OUTPUT_DIR
    TMP_DIR: str = OUTPUT_DIR

    # Persisted artifacts (RUN_DIR)
    FINAL_MODEL_CODE_PATH: str = f"{OUTPUT_DIR}/final_model_code.py"
    FINAL_MODEL_PATH: str = f"{OUTPUT_DIR}/final_model.pkl"
    FINAL_REPORT_PATH: str = f"{OUTPUT_DIR}/final_report.md"
    EXECUTION_LOG_PATH: str = f"{OUTPUT_DIR}/agent_execution.log"
    DATA_CLEANER_REPORT_PATH: str = f"{OUTPUT_DIR}/data_cleaner_report.json"
    FEATURE_ENGINEER_REPORT_PATH: str = f"{OUTPUT_DIR}/feature_engineer_report.json"
    MODEL_TRAINER_REPORT_PATH: str = f"{OUTPUT_DIR}/model_trainer_report.json"
    PSI_REPORT_PATH: str = f"{OUTPUT_DIR}/psi_report.csv"
    STABILITY_REPORT_PATH: str = f"{OUTPUT_DIR}/stability_report.csv"
    SHAP_PSI_PRUNE_LOG_PATH: str = f"{OUTPUT_DIR}/shap_psi_prune_log.csv"
    # Final-model SHAP importance visual + LLM-generated explanations for top features
    SHAP_PLOT_PATH: str = f"{OUTPUT_DIR}/shap_summary.png"
    SHAP_FEATURE_REPORT_PATH: str = f"{OUTPUT_DIR}/shap_feature_explanations.csv"
    # Per-agent replay scripts (new — let user re-run each agent's transforms on
    # fresh input without re-doing the LLM analysis).
    PIPELINE_PROCESS_DC_PATH: str = f"{OUTPUT_DIR}/pipeline_process_data_cleaner.py"
    PIPELINE_PROCESS_FE_PATH: str = f"{OUTPUT_DIR}/pipeline_process_feature_engineer.py"
    PIPELINE_PROCESS_FE_SPEC_PATH: str = f"{OUTPUT_DIR}/feature_spec.pkl"
    PIPELINE_PROCESS_TM_PATH: str = f"{OUTPUT_DIR}/pipeline_process_train_model.py"

    # Intermediate handoff files (TMP_DIR by default — deleted after run)
    CLEAN_DATA_PATH: str = f"{OUTPUT_DIR}/clean_data.parquet"
    ENGINEERED_DATA_PATH: str = f"{OUTPUT_DIR}/engineered_data.parquet"
    CLEAN_TRAIN_PATH: str = f"{OUTPUT_DIR}/clean_train.parquet"
    CLEAN_VALID_PATH: str = f"{OUTPUT_DIR}/clean_valid.parquet"
    CLEAN_OOT_PATH:   str = f"{OUTPUT_DIR}/clean_oot.parquet"
    ENGINEERED_TRAIN_PATH: str = f"{OUTPUT_DIR}/engineered_train.parquet"
    ENGINEERED_VALID_PATH: str = f"{OUTPUT_DIR}/engineered_valid.parquet"
    ENGINEERED_OOT_PATH:   str = f"{OUTPUT_DIR}/engineered_oot.parquet"

    @classmethod
    def init_run(cls) -> str:
        """Allocate today's next run dir + tempdir for intermediate files.

        Folder layout (counter resets when day rolls over because new YYYY-MM-DD
        starts with no run_* siblings):
            outputs/2026-06-07/run_01/...
            outputs/2026-06-07/run_02/...
            outputs/2026-06-08/run_01/...   ← new day → counter resets

        Updates every path attribute in-place so existing
        `Config.FINAL_MODEL_PATH` etc. references pick up the new run dir
        without changing callsites.

        Returns the absolute RUN_DIR path.
        """
        base = Path(cls.OUTPUT_DIR) / datetime.now().strftime("%Y-%m-%d")
        base.mkdir(parents=True, exist_ok=True)
        existing = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")]
        # max+1 (not len+1) so a deleted middle run does not cause collisions
        used_nums = []
        for d in existing:
            try:
                used_nums.append(int(d.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        candidate_n = (max(used_nums) + 1) if used_nums else 1
        # Atomic claim via mkdir(exist_ok=False): if a parallel process picked the
        # same number first we get FileExistsError and probe the next slot. Caps
        # at 256 attempts to avoid pathological loops; in practice resolves on
        # attempt 0 or 1 even under heavy concurrent batch automation.
        run_dir = None
        for offset in range(256):
            candidate = base / f"run_{candidate_n + offset:02d}"
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                run_dir = candidate
                break
            except FileExistsError:
                continue
        if run_dir is None:
            raise RuntimeError(
                f"Could not allocate a fresh run_NN under {base} after 256 attempts "
                "— filesystem race or stale dirs blocking allocation"
            )
        cls.RUN_DIR = str(run_dir)

        if cls.KEEP_INTERMEDIATES:
            cls.TMP_DIR = cls.RUN_DIR
        else:
            cls.TMP_DIR = tempfile.mkdtemp(prefix="automl_pipeline_")

        # Persisted artifacts → RUN_DIR
        rd = cls.RUN_DIR
        cls.FINAL_MODEL_CODE_PATH         = f"{rd}/final_model_code.py"
        cls.FINAL_MODEL_PATH              = f"{rd}/final_model.pkl"
        cls.FINAL_REPORT_PATH             = f"{rd}/final_report.md"
        cls.EXECUTION_LOG_PATH            = f"{rd}/agent_execution.log"
        cls.DATA_CLEANER_REPORT_PATH      = f"{rd}/data_cleaner_report.json"
        cls.FEATURE_ENGINEER_REPORT_PATH  = f"{rd}/feature_engineer_report.json"
        cls.MODEL_TRAINER_REPORT_PATH     = f"{rd}/model_trainer_report.json"
        cls.PSI_REPORT_PATH               = f"{rd}/psi_report.csv"
        cls.STABILITY_REPORT_PATH         = f"{rd}/stability_report.csv"
        cls.SHAP_PSI_PRUNE_LOG_PATH       = f"{rd}/shap_psi_prune_log.csv"
        cls.SHAP_PLOT_PATH                = f"{rd}/shap_summary.png"
        cls.SHAP_FEATURE_REPORT_PATH      = f"{rd}/shap_feature_explanations.csv"
        cls.PIPELINE_PROCESS_DC_PATH      = f"{rd}/pipeline_process_data_cleaner.py"
        cls.PIPELINE_PROCESS_FE_PATH      = f"{rd}/pipeline_process_feature_engineer.py"
        cls.PIPELINE_PROCESS_FE_SPEC_PATH = f"{rd}/feature_spec.pkl"
        cls.PIPELINE_PROCESS_TM_PATH      = f"{rd}/pipeline_process_train_model.py"

        # Intermediate handoff files → TMP_DIR
        td = cls.TMP_DIR
        cls.CLEAN_DATA_PATH        = f"{td}/clean_data.parquet"
        cls.ENGINEERED_DATA_PATH   = f"{td}/engineered_data.parquet"
        cls.CLEAN_TRAIN_PATH       = f"{td}/clean_train.parquet"
        cls.CLEAN_VALID_PATH       = f"{td}/clean_valid.parquet"
        cls.CLEAN_OOT_PATH         = f"{td}/clean_oot.parquet"
        cls.ENGINEERED_TRAIN_PATH  = f"{td}/engineered_train.parquet"
        cls.ENGINEERED_VALID_PATH  = f"{td}/engineered_valid.parquet"
        cls.ENGINEERED_OOT_PATH    = f"{td}/engineered_oot.parquet"

        return cls.RUN_DIR

    @classmethod
    def cleanup_run(cls) -> None:
        """Remove the intermediate tempdir at the end of a pipeline run.

        Safe to call multiple times. No-op when KEEP_INTERMEDIATES=true
        (TMP_DIR == RUN_DIR — never delete the run dir).
        """
        if cls.KEEP_INTERMEDIATES:
            return
        tmp = cls.TMP_DIR
        if not tmp or tmp == cls.RUN_DIR or not Path(tmp).exists():
            return
        shutil.rmtree(tmp, ignore_errors=True)

    # ── S3 / S3-compatible storage (MinIO, Wasabi, ...) ───────────────────────
    # Used by BaseAgent.load_dataframe when path starts with s3://
    # Leave keys empty to fall back to AWS default credential chain (boto3-style)
    S3_ACCESS_KEY: str = os.getenv("S3_ACCESS_KEY", "")
    S3_SECRET_KEY: str = os.getenv("S3_SECRET_KEY", "")
    # Override for S3-compatible endpoints (MinIO etc.); leave empty for AWS S3
    S3_ENDPOINT_URL: str = os.getenv("S3_ENDPOINT_URL", "")
    # Network timeouts in seconds — bumped to handle large parquet datasets
    S3_CONNECT_TIMEOUT: int = int(os.getenv("S3_CONNECT_TIMEOUT", 300))
    S3_REQUEST_TIMEOUT: int = int(os.getenv("S3_REQUEST_TIMEOUT", 3600))

    # ── Logging ───────────────────────────────────────────────────────────────
    # Max chars of a tool result shown in logs
    LOG_RESULT_PREVIEW_CHARS: int = int(os.getenv("LOG_RESULT_PREVIEW_CHARS", 200))
    # How many actions to include in the human-readable summary string
    SUMMARY_ACTION_PREVIEW: int = int(os.getenv("SUMMARY_ACTION_PREVIEW", 3))

    # ── Data Cleaning ─────────────────────────────────────────────────────────
    # Drop columns whose null ratio exceeds this threshold
    NULL_DROP_THRESHOLD: float = float(os.getenv("NULL_DROP_THRESHOLD", 0.9))
    # Columns with more unique values than this skip value_counts display
    HIGH_CARDINALITY_THRESHOLD: int = int(os.getenv("HIGH_CARDINALITY_THRESHOLD", 50))
    # How many columns' stats to include in the analysis sample
    SAMPLE_STATS_PREVIEW: int = int(os.getenv("SAMPLE_STATS_PREVIEW", 3))
    # Duplicate row percentage above this triggers a drop suggestion
    DUPLICATE_PCT_THRESHOLD: float = float(os.getenv("DUPLICATE_PCT_THRESHOLD", 1.0))
    # Composite key violation percentage above this triggers deduplicate_by_key suggestion
    PK_VIOLATION_PCT_THRESHOLD: float = float(os.getenv("PK_VIOLATION_PCT_THRESHOLD", 1.0))
    # Orphan entity percentage above this is flagged (no identity anchor at all)
    ENTITY_ORPHAN_PCT_THRESHOLD: float = float(os.getenv("ENTITY_ORPHAN_PCT_THRESHOLD", 5.0))
    # Ambiguous identity values percentage above this is flagged as potential fraud ring
    AMBIGUOUS_IDENTITY_PCT_THRESHOLD: float = float(os.getenv("AMBIGUOUS_IDENTITY_PCT_THRESHOLD", 5.0))
    # Soft duplicate percentage above this triggers deduplicate_by_key suggestion
    SOFT_DUP_PCT_THRESHOLD: float = float(os.getenv("SOFT_DUP_PCT_THRESHOLD", 0.5))
    # Duplicated application entities percentage above this triggers deduplicate_by_key
    APP_DUP_PCT_THRESHOLD: float = float(os.getenv("APP_DUP_PCT_THRESHOLD", 10.0))
    # Outlier percentage (IQR 3x) above this triggers a clip suggestion
    OUTLIER_PCT_THRESHOLD: float = float(os.getenv("OUTLIER_PCT_THRESHOLD", 3.0))
    # Class imbalance ratio (max/min class count) above this is flagged
    IMBALANCE_RATIO_THRESHOLD: float = float(os.getenv("IMBALANCE_RATIO_THRESHOLD", 20.0))
    # Max numeric columns to run outlier detection on (for performance)
    OUTLIER_NUMERIC_COLS_LIMIT: int = int(os.getenv("OUTLIER_NUMERIC_COLS_LIMIT", 10))

    # ── Feature Engineering ───────────────────────────────────────────────────
    HIGH_CORRELATION_THRESHOLD: float = float(os.getenv("HIGH_CORRELATION_THRESHOLD", 0.8))
    LOW_CORRELATION_THRESHOLD: float = float(os.getenv("LOW_CORRELATION_THRESHOLD", 0.04))
    # Features with |correlation| below this are flagged for removal
    MIN_CORRELATION_THRESHOLD: float = float(os.getenv("MIN_CORRELATION_THRESHOLD", 0.001))
    # Ceiling for FeatureEngineerAgent.select_top_features (overridable via env var)
    TOP_K_FEATURES_CAP: int = int(os.getenv("TOP_K_FEATURES_CAP", 800))
    # Fraction of engineerable features kept by select_top_features (0 < ratio <= 1)
    TOP_K_RATIO: float = float(os.getenv("TOP_K_RATIO", 0.70))
    # Fraction of numeric columns to include in LLM prompt metadata (0 < ratio <= 1)
    FEATURE_META_NUMERIC_RATIO: float = float(os.getenv("FEATURE_META_NUMERIC_RATIO", 0.6))
    # Absolute ceiling for numeric columns in LLM prompt (safety net for very wide datasets)
    FEATURE_META_MAX_NUMERIC_COLS: int = int(os.getenv("FEATURE_META_MAX_NUMERIC_COLS", 300))
    # Max categorical columns to include full per-column metadata in the LLM prompt
    FEATURE_META_MAX_CATEGORICAL_COLS: int = int(os.getenv("FEATURE_META_MAX_CATEGORICAL_COLS", 80))
    # Max description entries shown per group in the LLM prompt column-description block
    MAX_DESC_PER_GROUP: int = int(os.getenv("MAX_DESC_PER_GROUP", 50))

    # ── Model Training ────────────────────────────────────────────────────────
    TRAIN_TEST_SPLIT_SIZE: float = float(os.getenv("TRAIN_TEST_SPLIT_SIZE", 0.2))
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", 42))
    MAX_TRAINING_ITERATIONS: int = int(os.getenv("MAX_TRAINING_ITERATIONS", 3))
    MODELS_TO_COMPARE: str = os.getenv("MODELS_TO_COMPARE", "XGBoost,RandomForest,ExtraTrees,LightGBM,CatBoost")
    CLASSIFICATION_UNIQUE_THRESHOLD: int = int(os.getenv("CLASSIFICATION_UNIQUE_THRESHOLD", 10))
    # n_splits for the final cross-validation evaluation of the trained model
    CV_N_SPLITS: int = int(os.getenv("CV_N_SPLITS", 5))
    TARGET_ROC_AUC: float = float(os.getenv("TARGET_ROC_AUC", 0.85))
    TARGET_F1: float = float(os.getenv("TARGET_F1", 0.80))
    TARGET_R2: float = float(os.getenv("TARGET_R2", 0.75))
    TARGET_ACCURACY: float = float(os.getenv("TARGET_ACCURACY", 0.85))

    # ── Optuna Hyperparameter Search Space ────────────────────────────────────
    # Shared across lgbm / xgboost / rf / extra_tree
    LR_MIN: float = float(os.getenv("LR_MIN", 0.01))
    LR_MAX: float = float(os.getenv("LR_MAX", 0.3))
    MAX_DEPTH_MIN: int = int(os.getenv("MAX_DEPTH_MIN", 3))
    MAX_DEPTH_MAX: int = int(os.getenv("MAX_DEPTH_MAX", 10))
    N_ESTIMATORS_MIN: int = int(os.getenv("N_ESTIMATORS_MIN", 100))
    N_ESTIMATORS_MAX: int = int(os.getenv("N_ESTIMATORS_MAX", 1000))
    SUBSAMPLE_MIN: float = float(os.getenv("SUBSAMPLE_MIN", 0.4))
    SUBSAMPLE_MAX: float = float(os.getenv("SUBSAMPLE_MAX", 1.0))
    # LightGBM-specific
    NUM_LEAVES_MIN: int = int(os.getenv("NUM_LEAVES_MIN", 15))
    NUM_LEAVES_MAX: int = int(os.getenv("NUM_LEAVES_MAX", 256))
    # CatBoost-specific (depth range differs from XGBoost/LGBM)
    CB_DEPTH_MIN: int = int(os.getenv("CB_DEPTH_MIN", 4))
    CB_DEPTH_MAX: int = int(os.getenv("CB_DEPTH_MAX", 10))

    # ── OOT / Temporal Split ─────────────────────────────────────────────────────
    OOT_INIT_MONTHS: int = int(os.getenv("OOT_INIT_MONTHS", 2))   # floor: always include >= this many months
    OOT_MIN_RATIO: float = float(os.getenv("OOT_MIN_RATIO", 0.15))  # expand until OOT >= this ratio
    OOT_MAX_RATIO: float = float(os.getenv("OOT_MAX_RATIO", 0.2))  # hard cap: shrink if OOT exceeds this
    VALID_TEMPORAL_RATIO: float = float(os.getenv("VALID_TEMPORAL_RATIO", 0.20))

    # ── FLAML AutoML ──────────────────────────────────────────────────────────
    FLAML_TIME_BUDGET: int = int(os.getenv("FLAML_TIME_BUDGET", 1200))
    FLAML_ESTIMATORS: str = os.getenv("FLAML_ESTIMATORS", "xgboost,lgbm,catboost,rf,extra_tree")
    FLAML_N_SPLITS: int = int(os.getenv("FLAML_N_SPLITS", 5))
    # Max training rows passed to FLAML — FLAML internally copies the DataFrame for block
    # consolidation which can OOM on large datasets. Sampling here keeps the copy small
    # while still giving FLAML enough signal to select the best estimator type.
    FLAML_MAX_ROWS: int = int(os.getenv("FLAML_MAX_ROWS", 500_000))

    # ── Optuna ────────────────────────────────────────────────────────────────
    OPTUNA_N_TRIALS: int = int(os.getenv("OPTUNA_N_TRIALS", 50))
    OPTUNA_TIMEOUT: int = int(os.getenv("OPTUNA_TIMEOUT", 1200))

    # ── RFE ───────────────────────────────────────────────────────────────────
    RFE_TARGET_FEATURES: int = int(os.getenv("RFE_TARGET_FEATURES", 150))
    RFE_STEP: float = float(os.getenv("RFE_STEP", 0.05))
    RFE_CV_SPLITS: int = int(os.getenv("RFE_CV_SPLITS", 3))
    ENABLE_RFECV: bool = os.getenv("ENABLE_RFECV", "false").lower() == "true"
    # n_estimators used for the base model fitted during RFE selection
    RFE_N_ESTIMATORS: int = int(os.getenv("RFE_N_ESTIMATORS", 200))

    # ── PSI ───────────────────────────────────────────────────────────────────
    PSI_THRESHOLD: float = float(os.getenv("PSI_THRESHOLD", 0.3))
    PSI_BINS: int = int(os.getenv("PSI_BINS", 100))

    # ── Stability ─────────────────────────────────────────────────────────────
    STABILITY_MIN_MONTHS: int = int(os.getenv("STABILITY_MIN_MONTHS", 6))
    STABILITY_GINI_STD_THRESHOLD: float = float(os.getenv("STABILITY_GINI_STD_THRESHOLD", 0.15))
    STABILITY_MIN_GINI: float = float(os.getenv("STABILITY_MIN_GINI", 0.01))

    # ── Final Feature Cut ─────────────────────────────────────────────────────
    MAX_FINAL_FEATURES: int = int(os.getenv("MAX_FINAL_FEATURES", 100))

    # ── SHAP + PSI Iterative Pruning ──────────────────────────────────────────
    # n_estimators for the quick model fitted at each SHAP pruning step
    SHAP_N_ESTIMATORS: int = int(os.getenv("SHAP_N_ESTIMATORS", 100))
    # Max training rows sampled for SHAP value computation (speed vs accuracy)
    SHAP_SAMPLE_SIZE: int = int(os.getenv("SHAP_SAMPLE_SIZE", 2000))
    # Stop pruning after this many consecutive steps without AUC improvement
    SHAP_PSI_MAX_NO_IMPROVE: int = int(os.getenv("SHAP_PSI_MAX_NO_IMPROVE", 2))
    # Absolute floor on minimum features kept after SHAP+PSI pruning
    SHAP_PSI_MIN_FEATURES_FLOOR: int = int(os.getenv("SHAP_PSI_MIN_FEATURES_FLOOR", 5))
    # Relative floor: keep at least this fraction of MAX_FINAL_FEATURES
    SHAP_PSI_MIN_FEATURES_RATIO: float = float(os.getenv("SHAP_PSI_MIN_FEATURES_RATIO", 0.10))

    # ── Final SHAP visual + LLM-generated explanations ────────────────────────
    # Generate shap_summary.png + LLM explanation of top features after the
    # final model is trained. Set to "false" to skip (saves SHAP compute time
    # and one LLM call when running purely automated batches).
    SHAP_FINAL_EXPLAIN_ENABLED: bool = os.getenv("SHAP_FINAL_EXPLAIN_ENABLED", "true").lower() == "true"
    # How many top features to ask the LLM to explain. Plot shows up to 2x this
    # so users can see the broader importance distribution.
    SHAP_FINAL_EXPLAIN_TOP_N: int = int(os.getenv("SHAP_FINAL_EXPLAIN_TOP_N", 20))

    # ── Pipeline Stage-0 distribution drift check ────────────────────────────
    # Sample size per partition for the PSI-based train↔valid / train↔oot
    # comparison. ~50K rows × n_cols stays well under 1 GB even for wide data.
    DRIFT_SAMPLE_N: int = int(os.getenv("DRIFT_SAMPLE_N", 100_000))
    # PSI value above which a column is flagged as drifting between partitions.
    # Reusing PSI_THRESHOLD (default 0.3, finance-standard) means the pipeline
    # warning matches Agent 3's feature-drop threshold — what gets flagged here
    # is what would later be dropped.
    DRIFT_PSI_THRESHOLD: float = float(os.getenv("DRIFT_PSI_THRESHOLD", 0.25))
    # How many top-drifting columns to list in the warning log
    DRIFT_TOP_N_REPORT: int = int(os.getenv("DRIFT_TOP_N_REPORT", 10))

    # ── Split-mode memory savers (column pre-filter + stratified sampling) ────
    # Applied INSIDE Agent 1's fit_transform on the TRAIN partition only, to
    # shrink very wide datasets that would otherwise OOM during cleaning.
    # The same column drops + clip bounds are captured into CleaningSpec and
    # replayed on valid / oot (zero leakage in the prefilter decisions).
    # Pre-filter drops columns with null ratio above this threshold.
    PREFILTER_MAX_NULL_RATIO: float = float(os.getenv("PREFILTER_MAX_NULL_RATIO", 0.95))
    # Pre-filter drops columns where one value dominates above this fraction
    # of non-null rows (catches near-constant cols). Cols with nunique > 1000
    # skip this check (high-cardinality => not near-constant).
    PREFILTER_MAX_DOMINANT_RATIO: float = float(os.getenv("PREFILTER_MAX_DOMINANT_RATIO", 0.99))

    # ── Overfitting Detection ─────────────────────────────────────────────────
    # Relative gap (valid_auc - holdout_auc) / valid_auc above which overfitting
    # is flagged and an LLM-guided retrain is triggered
    OVERFIT_THRESHOLD: float = float(os.getenv("OVERFIT_THRESHOLD", 0.12))

    # ── Direct Cloud Fallback — order: LiteLLM proxy → OpenAI → Claude ─────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_DIRECT_MODEL: str = os.getenv("OPENAI_DIRECT_MODEL", "gpt-4.1-mini")

    # Claude (Anthropic) — last-resort fallback; pip install anthropic
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    CLAUDE_DIRECT_MODEL: str = os.getenv("CLAUDE_DIRECT_MODEL", "claude-sonnet-4-6")

    @classmethod
    def validate(cls):
        """Validate that at least one LLM endpoint is configured.

        Pipeline tolerates the LiteLLM proxy being unreachable at runtime
        (it falls back to OpenAI direct → Claude direct), but at least one
        of the three paths must be configured upfront. Also sanity-checks
        the LITELLM_URL format.
        """
        if cls.LITELLM_URL and not cls.LITELLM_URL.startswith(("http://", "https://")):
            raise ValueError(
                f"Invalid LITELLM_URL '{cls.LITELLM_URL}'. "
                "Expected a URL starting with http:// or https://"
            )
        has_proxy = bool(cls.LITELLM_URL)
        has_openai = bool(cls.OPENAI_API_KEY)
        has_claude = bool(cls.ANTHROPIC_API_KEY)
        if not (has_proxy or has_openai or has_claude):
            raise ValueError(
                "No LLM endpoint configured. Set at least one of:\n"
                "  - LITELLM_URL  (proxy)\n"
                "  - OPENAI_API_KEY  (direct OpenAI fallback)\n"
                "  - ANTHROPIC_API_KEY  (direct Claude fallback)"
            )
        if has_proxy and not (has_openai or has_claude):
            print(
                f"WARN: LITELLM_URL is set but no fallback API key configured. "
                f"If the proxy at {cls.LITELLM_URL} becomes unreachable the pipeline will fail."
            )

    @classmethod
    def is_proxy_reachable(cls) -> bool:
        import urllib.request
        try:
            urllib.request.urlopen(f"{cls.LITELLM_URL}/health", timeout=3)
            return True
        except Exception:
            return False

    @classmethod
    def get_client(cls):
        from openai import OpenAI
        cls.validate()
        return OpenAI(base_url=cls.LITELLM_URL, api_key=cls.API_KEY)

    @classmethod
    def get_direct_client(cls):
        from openai import OpenAI
        if not cls.OPENAI_API_KEY:
            return None
        # Same defensive pattern as get_claude_client — explicit base_url stops
        # any stray OPENAI_BASE_URL env var from rerouting the public-OpenAI
        # fallback to a local/internal endpoint that may not be running.
        return OpenAI(
            api_key=cls.OPENAI_API_KEY,
            base_url="https://api.openai.com/v1",
        )

    @classmethod
    def get_claude_client(cls):
        if not cls.ANTHROPIC_API_KEY:
            return None
        try:
            import anthropic
            # Explicit base_url overrides ANTHROPIC_BASE_URL env var. That env var
            # is meant for LLM_BACKEND=gateway only — if it leaks into legacy mode
            # (common when user copies .env.example which sets it to
            # http://localhost:4000), the anthropic SDK routes calls to localhost
            # instead of api.anthropic.com → WinError 10061 / Connection refused
            # when no LiteLLM proxy is running.
            return anthropic.Anthropic(
                api_key=cls.ANTHROPIC_API_KEY,
                base_url="https://api.anthropic.com",
            )
        except ImportError:
            return None


# ── Backend dispatcher ───────────────────────────────────────────────────────
# When LLM_BACKEND=gateway, re-export GatewayConfig as Config so every
# `from config import Config` transparently picks up the gateway-aware class.
# Legacy behavior is the default — no change unless the env var is set.
if os.getenv("LLM_BACKEND", "legacy").lower() == "gateway":
    from config_gateway import GatewayConfig as Config   # noqa: F811
