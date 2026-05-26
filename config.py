import os
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
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.7))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 4000))
    # Prompts longer than this (chars) are routed to the cloud model
    MODEL_ROUTING_THRESHOLD: int = int(os.getenv("MODEL_ROUTING_THRESHOLD", 3000))

    # ── Output Paths ─────────────────────────────────────────────────────────
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "outputs")
    CLEAN_DATA_PATH: str = f"{os.getenv('OUTPUT_DIR', 'outputs')}/clean_data.csv"
    ENGINEERED_DATA_PATH: str = f"{os.getenv('OUTPUT_DIR', 'outputs')}/engineered_data.csv"
    FINAL_MODEL_CODE_PATH: str = f"{os.getenv('OUTPUT_DIR', 'outputs')}/final_model_code.py"
    FINAL_REPORT_PATH: str = f"{os.getenv('OUTPUT_DIR', 'outputs')}/final_report.md"
    EXECUTION_LOG_PATH: str = f"{os.getenv('OUTPUT_DIR', 'outputs')}/agent_execution.log"
    DATA_CLEANER_REPORT_PATH: str = f"{os.getenv('OUTPUT_DIR', 'outputs')}/data_cleaner_report.json"
    FEATURE_ENGINEER_REPORT_PATH: str = f"{os.getenv('OUTPUT_DIR', 'outputs')}/feature_engineer_report.json"
    MODEL_TRAINER_REPORT_PATH: str = f"{os.getenv('OUTPUT_DIR', 'outputs')}/model_trainer_report.json"

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
    DEFAULT_TOP_K_FEATURES: int = int(os.getenv("DEFAULT_TOP_K_FEATURES", 350))
    TARGET_FEATURE_COUNT_MIN: int = int(os.getenv("TARGET_FEATURE_COUNT_MIN", 10))
    TARGET_FEATURE_COUNT_MAX: int = int(os.getenv("TARGET_FEATURE_COUNT_MAX", 1000))
    # Max columns to include full per-column metadata in the LLM prompt
    FEATURE_META_MAX_COLS: int = int(os.getenv("FEATURE_META_MAX_COLS", 50))

    # ── Model Training ────────────────────────────────────────────────────────
    MAX_TRAINING_ITERATIONS: int = int(os.getenv("MAX_TRAINING_ITERATIONS", 800))
    # Target column with fewer unique values than this is treated as classification
    CLASSIFICATION_UNIQUE_THRESHOLD: int = int(os.getenv("CLASSIFICATION_UNIQUE_THRESHOLD", 5))
    TRAIN_TEST_SPLIT_SIZE: float = float(os.getenv("TRAIN_TEST_SPLIT_SIZE", 0.2))
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", 42))
    # Comma-separated list of models to train and compare each iteration
    MODELS_TO_COMPARE: str = os.getenv("MODELS_TO_COMPARE", "XGBoost,RandomForest,LightGBM,CatBoost,ExtraTrees")

    # ── Stopping Criteria ─────────────────────────────────────────────────────
    TARGET_ROC_AUC: float = float(os.getenv("TARGET_ROC_AUC", 0.70))
    TARGET_F1: float = float(os.getenv("TARGET_F1", 0.75))
    TARGET_R2: float = float(os.getenv("TARGET_R2", 0.1))
    TARGET_ACCURACY: float = float(os.getenv("TARGET_ACCURACY", 0.75))

    # ── Shared Hyperparameter Search Space ────────────────────────────────────
    LR_MIN: float = float(os.getenv("LR_MIN", 0.01))
    LR_MAX: float = float(os.getenv("LR_MAX", 0.3))
    MAX_DEPTH_MIN: int = int(os.getenv("MAX_DEPTH_MIN", 3))
    MAX_DEPTH_MAX: int = int(os.getenv("MAX_DEPTH_MAX", 8))
    N_ESTIMATORS_MIN: int = int(os.getenv("N_ESTIMATORS_MIN", 50))
    N_ESTIMATORS_MAX: int = int(os.getenv("N_ESTIMATORS_MAX", 300))
    SUBSAMPLE_MIN: float = float(os.getenv("SUBSAMPLE_MIN", 0.6))
    SUBSAMPLE_MAX: float = float(os.getenv("SUBSAMPLE_MAX", 1.0))
    # LightGBM-specific
    NUM_LEAVES_MIN: int = int(os.getenv("NUM_LEAVES_MIN", 15))
    NUM_LEAVES_MAX: int = int(os.getenv("NUM_LEAVES_MAX", 63))
    # CatBoost-specific (depth range differs from XGBoost)
    CB_DEPTH_MIN: int = int(os.getenv("CB_DEPTH_MIN", 4))
    CB_DEPTH_MAX: int = int(os.getenv("CB_DEPTH_MAX", 10))

    # ── OOT / Temporal Split ─────────────────────────────────────────────────────
    OOT_INIT_MONTHS: int = int(os.getenv("OOT_INIT_MONTHS", 2))
    OOT_MIN_RATIO: float = float(os.getenv("OOT_MIN_RATIO", 0.20))
    VALID_TEMPORAL_RATIO: float = float(os.getenv("VALID_TEMPORAL_RATIO", 0.20))

    # ── FLAML AutoML ──────────────────────────────────────────────────────────
    FLAML_TIME_BUDGET: int = int(os.getenv("FLAML_TIME_BUDGET", 300))
    FLAML_ESTIMATORS: str = os.getenv("FLAML_ESTIMATORS", "lgbm,xgboost,rf,extra_tree")
    FLAML_N_SPLITS: int = int(os.getenv("FLAML_N_SPLITS", 5))

    # ── Optuna ────────────────────────────────────────────────────────────────
    OPTUNA_N_TRIALS: int = int(os.getenv("OPTUNA_N_TRIALS", 50))
    OPTUNA_TIMEOUT: int = int(os.getenv("OPTUNA_TIMEOUT", 180))

    # ── RFE ───────────────────────────────────────────────────────────────────
    RFE_TARGET_FEATURES: int = int(os.getenv("RFE_TARGET_FEATURES", 50))
    RFE_STEP: float = float(os.getenv("RFE_STEP", 0.05))
    RFE_CV_SPLITS: int = int(os.getenv("RFE_CV_SPLITS", 3))
    ENABLE_RFECV: bool = os.getenv("ENABLE_RFECV", "false").lower() == "true"

    # ── PSI ───────────────────────────────────────────────────────────────────
    PSI_THRESHOLD: float = float(os.getenv("PSI_THRESHOLD", 0.25))
    PSI_BINS: int = int(os.getenv("PSI_BINS", 10))

    # ── Stability ─────────────────────────────────────────────────────────────
    STABILITY_MIN_MONTHS: int = int(os.getenv("STABILITY_MIN_MONTHS", 6))
    STABILITY_GINI_STD_THRESHOLD: float = float(os.getenv("STABILITY_GINI_STD_THRESHOLD", 0.10))
    STABILITY_MIN_GINI: float = float(os.getenv("STABILITY_MIN_GINI", 0.02))

    # ── Final Feature Cut ─────────────────────────────────────────────────────
    MAX_FINAL_FEATURES: int = int(os.getenv("MAX_FINAL_FEATURES", 100))

    # ── Direct Cloud Fallback (used when LiteLLM proxy is unreachable) ─────────
    # If OPENAI_API_KEY is set, the pipeline falls back to this model directly
    # instead of failing completely when the local LiteLLM proxy is down.
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_DIRECT_MODEL: str = os.getenv("OPENAI_DIRECT_MODEL", "gpt-4o-mini")

    @classmethod
    def validate(cls):
        if not cls.LITELLM_URL:
            raise ValueError("Missing LITELLM_URL. Set it in .env or as an environment variable.")

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
