import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
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
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "outputs")
    CLEAN_DATA_PATH: str = f"{os.getenv('OUTPUT_DIR', 'outputs')}/clean_data.csv"
    ENGINEERED_DATA_PATH: str = f"{os.getenv('OUTPUT_DIR', 'outputs')}/engineered_data.csv"
    FINAL_MODEL_CODE_PATH: str = f"{os.getenv('OUTPUT_DIR', 'outputs')}/final_model_code.py"
    FINAL_MODEL_PATH: str = f"{os.getenv('OUTPUT_DIR', 'outputs')}/final_model.pkl"
    FINAL_REPORT_PATH: str = f"{os.getenv('OUTPUT_DIR', 'outputs')}/final_report.md"
    EXECUTION_LOG_PATH: str = f"{os.getenv('OUTPUT_DIR', 'outputs')}/agent_execution.log"
    DATA_CLEANER_REPORT_PATH: str = f"{os.getenv('OUTPUT_DIR', 'outputs')}/data_cleaner_report.json"
    FEATURE_ENGINEER_REPORT_PATH: str = f"{os.getenv('OUTPUT_DIR', 'outputs')}/feature_engineer_report.json"
    MODEL_TRAINER_REPORT_PATH: str = f"{os.getenv('OUTPUT_DIR', 'outputs')}/model_trainer_report.json"

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
    NULL_DROP_THRESHOLD: float = float(os.getenv("NULL_DROP_THRESHOLD", 0.8))
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
    OUTLIER_PCT_THRESHOLD: float = float(os.getenv("OUTLIER_PCT_THRESHOLD", 5.0))
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
    TOP_K_FEATURES_CAP: int = int(os.getenv("TOP_K_FEATURES_CAP", 350))
    # Fraction of engineerable features kept by select_top_features (0 < ratio <= 1)
    TOP_K_RATIO: float = float(os.getenv("TOP_K_RATIO", 0.70))
    # Fraction of numeric columns to include in LLM prompt metadata (0 < ratio <= 1)
    FEATURE_META_NUMERIC_RATIO: float = float(os.getenv("FEATURE_META_NUMERIC_RATIO", 0.6))
    # Absolute ceiling for numeric columns in LLM prompt (safety net for very wide datasets)
    FEATURE_META_MAX_NUMERIC_COLS: int = int(os.getenv("FEATURE_META_MAX_NUMERIC_COLS", 200))
    # Max categorical columns to include full per-column metadata in the LLM prompt
    FEATURE_META_MAX_CATEGORICAL_COLS: int = int(os.getenv("FEATURE_META_MAX_CATEGORICAL_COLS", 40))
    # Max description entries shown per group in the LLM prompt column-description block
    MAX_DESC_PER_GROUP: int = int(os.getenv("MAX_DESC_PER_GROUP", 30))

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
    FLAML_TIME_BUDGET: int = int(os.getenv("FLAML_TIME_BUDGET", 600))
    FLAML_ESTIMATORS: str = os.getenv("FLAML_ESTIMATORS", "xgboost,lgbm,catboost,rf,extra_tree")
    FLAML_N_SPLITS: int = int(os.getenv("FLAML_N_SPLITS", 5))
    # Max training rows passed to FLAML — FLAML internally copies the DataFrame for block
    # consolidation which can OOM on large datasets. Sampling here keeps the copy small
    # while still giving FLAML enough signal to select the best estimator type.
    FLAML_MAX_ROWS: int = int(os.getenv("FLAML_MAX_ROWS", 100_000))

    # ── Optuna ────────────────────────────────────────────────────────────────
    OPTUNA_N_TRIALS: int = int(os.getenv("OPTUNA_N_TRIALS", 50))
    OPTUNA_TIMEOUT: int = int(os.getenv("OPTUNA_TIMEOUT", 600))

    # ── RFE ───────────────────────────────────────────────────────────────────
    RFE_TARGET_FEATURES: int = int(os.getenv("RFE_TARGET_FEATURES", 50))
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
    STABILITY_MIN_GINI: float = float(os.getenv("STABILITY_MIN_GINI", 0.02))

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
        return OpenAI(api_key=cls.OPENAI_API_KEY)

    @classmethod
    def get_claude_client(cls):
        if not cls.ANTHROPIC_API_KEY:
            return None
        try:
            import anthropic
            return anthropic.Anthropic(api_key=cls.ANTHROPIC_API_KEY)
        except ImportError:
            return None
