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
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.3))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 4000))
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
    SHAP_BEESWARM_PATH: str = f"{OUTPUT_DIR}/shap_beeswarm.png"
    SHAP_FEATURE_REPORT_PATH: str = f"{OUTPUT_DIR}/shap_feature_explanations.csv"
    SHAP_FINAL_MODEL_REPORT_PATH: str = f"{OUTPUT_DIR}/final_model_shap_report.md"
    # Model diagnostic charts (generated by _tool_generate_model_charts)
    CHART_DIR: str = f"{OUTPUT_DIR}/charts"
    CHART_ROC_PATH: str = f"{OUTPUT_DIR}/charts/roc_curve.png"
    CHART_PR_PATH: str = f"{OUTPUT_DIR}/charts/pr_curve.png"
    CHART_KS_PATH: str = f"{OUTPUT_DIR}/charts/ks_curve.png"
    CHART_LIFT_PATH: str = f"{OUTPUT_DIR}/charts/lift_chart.png"
    CHART_GAIN_PATH: str = f"{OUTPUT_DIR}/charts/gain_chart.png"
    CHART_BAD_RATE_DECILE_PATH: str = f"{OUTPUT_DIR}/charts/bad_rate_decile.png"
    CHART_AVG_SCORE_DECILE_PATH: str = f"{OUTPUT_DIR}/charts/avg_score_decile.png"
    CHART_CALIBRATION_PATH: str = f"{OUTPUT_DIR}/charts/calibration_plot.png"
    CHART_SCORE_DIST_PATH: str = f"{OUTPUT_DIR}/charts/score_distribution.png"
    # Number of quantile buckets for the decile-based charts (Lift / Gain /
    # Bad-rate / Avg-score) and the calibration reliability plot. Default 10
    # = classic deciles; raise (e.g. 20) for finer granularity on large eval
    # splits, lower (e.g. 5) for small ones to avoid sparse/empty buckets.
    CHART_N_DECILES: int = int(os.getenv("CHART_N_DECILES", 10))
    # Per-agent replay scripts (new — let user re-run each agent's transforms on
    # fresh input without re-doing the LLM analysis).
    PIPELINE_PROCESS_DC_PATH: str = f"{OUTPUT_DIR}/pipeline_process_data_cleaner.py"
    PIPELINE_PROCESS_FE_PATH: str = f"{OUTPUT_DIR}/pipeline_process_feature_engineer.py"
    PIPELINE_PROCESS_FE_SPEC_PATH: str = f"{OUTPUT_DIR}/feature_spec.pkl"
    PIPELINE_PROCESS_TM_PATH: str = f"{OUTPUT_DIR}/pipeline_process_train_model.py"
    PIPELINE_PROCESS_DC_SPEC_PATH: str = f"{OUTPUT_DIR}/cleaning_spec.pkl"
    NULL_PROCESSOR_PATH: str = f"{OUTPUT_DIR}/null_processor.json"
    # End-to-end replay bundle: re-run this exact run (split -> clean -> FE ->
    # retrain / score) in another environment. See docs/replay.md.
    REPLAY_DRIVER_PATH: str = f"{OUTPUT_DIR}/replay_pipeline.py"
    REPLAY_MANIFEST_PATH: str = f"{OUTPUT_DIR}/replay_manifest.json"
    SPLIT_ASSIGNMENT_PATH: str = f"{OUTPUT_DIR}/split_assignment.parquet"

    # Intermediate handoff files (TMP_DIR by default — deleted after run)
    CLEAN_DATA_PATH: str = f"{OUTPUT_DIR}/clean_data.parquet"
    ENGINEERED_DATA_PATH: str = f"{OUTPUT_DIR}/engineered_data.parquet"
    CLEAN_TRAIN_PATH: str = f"{OUTPUT_DIR}/clean_train.parquet"
    CLEAN_VALID_PATH: str = f"{OUTPUT_DIR}/clean_valid.parquet"
    CLEAN_OOT_PATH:   str = f"{OUTPUT_DIR}/clean_oot.parquet"
    CLEAN_TEST_PATH:  str = f"{OUTPUT_DIR}/clean_test.parquet"
    ENGINEERED_TRAIN_PATH: str = f"{OUTPUT_DIR}/engineered_train.parquet"
    ENGINEERED_VALID_PATH: str = f"{OUTPUT_DIR}/engineered_valid.parquet"
    ENGINEERED_OOT_PATH:   str = f"{OUTPUT_DIR}/engineered_oot.parquet"
    ENGINEERED_TEST_PATH:  str = f"{OUTPUT_DIR}/engineered_test.parquet"
    # Single-file mode: partitions cut from the raw input before Agent 1 runs,
    # so cleaning + feature engineering fit on TRAIN only (no holdout leakage).
    AUTOSPLIT_TRAIN_PATH: str = f"{OUTPUT_DIR}/autosplit_train.parquet"
    AUTOSPLIT_VALID_PATH: str = f"{OUTPUT_DIR}/autosplit_valid.parquet"
    AUTOSPLIT_OOT_PATH:   str = f"{OUTPUT_DIR}/autosplit_oot.parquet"
    AUTOSPLIT_TEST_PATH:  str = f"{OUTPUT_DIR}/autosplit_test.parquet"

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
        cls.SHAP_BEESWARM_PATH            = f"{rd}/shap_beeswarm.png"
        cls.SHAP_FEATURE_REPORT_PATH      = f"{rd}/shap_feature_explanations.csv"
        cls.SHAP_FINAL_MODEL_REPORT_PATH  = f"{rd}/final_model_shap_report.md"
        cls.CHART_DIR                     = f"{rd}/charts"
        cls.CHART_ROC_PATH                = f"{rd}/charts/roc_curve.png"
        cls.CHART_PR_PATH                 = f"{rd}/charts/pr_curve.png"
        cls.CHART_KS_PATH                 = f"{rd}/charts/ks_curve.png"
        cls.CHART_LIFT_PATH               = f"{rd}/charts/lift_chart.png"
        cls.CHART_GAIN_PATH               = f"{rd}/charts/gain_chart.png"
        cls.CHART_BAD_RATE_DECILE_PATH    = f"{rd}/charts/bad_rate_decile.png"
        cls.CHART_AVG_SCORE_DECILE_PATH   = f"{rd}/charts/avg_score_decile.png"
        cls.CHART_CALIBRATION_PATH        = f"{rd}/charts/calibration_plot.png"
        cls.CHART_SCORE_DIST_PATH         = f"{rd}/charts/score_distribution.png"
        cls.PIPELINE_PROCESS_DC_PATH      = f"{rd}/pipeline_process_data_cleaner.py"
        cls.PIPELINE_PROCESS_FE_PATH      = f"{rd}/pipeline_process_feature_engineer.py"
        cls.PIPELINE_PROCESS_FE_SPEC_PATH = f"{rd}/feature_spec.pkl"
        cls.PIPELINE_PROCESS_TM_PATH      = f"{rd}/pipeline_process_train_model.py"
        cls.PIPELINE_PROCESS_DC_SPEC_PATH = f"{rd}/cleaning_spec.pkl"
        cls.NULL_PROCESSOR_PATH           = f"{rd}/null_processor.json"
        cls.REPLAY_DRIVER_PATH            = f"{rd}/replay_pipeline.py"
        cls.REPLAY_MANIFEST_PATH          = f"{rd}/replay_manifest.json"
        cls.SPLIT_ASSIGNMENT_PATH         = f"{rd}/split_assignment.parquet"

        # Intermediate handoff files → TMP_DIR
        td = cls.TMP_DIR
        cls.CLEAN_DATA_PATH        = f"{td}/clean_data.parquet"
        cls.ENGINEERED_DATA_PATH   = f"{td}/engineered_data.parquet"
        cls.CLEAN_TRAIN_PATH       = f"{td}/clean_train.parquet"
        cls.CLEAN_VALID_PATH       = f"{td}/clean_valid.parquet"
        cls.CLEAN_OOT_PATH         = f"{td}/clean_oot.parquet"
        cls.CLEAN_TEST_PATH        = f"{td}/clean_test.parquet"
        cls.ENGINEERED_TRAIN_PATH  = f"{td}/engineered_train.parquet"
        cls.ENGINEERED_VALID_PATH  = f"{td}/engineered_valid.parquet"
        cls.ENGINEERED_OOT_PATH    = f"{td}/engineered_oot.parquet"
        cls.ENGINEERED_TEST_PATH   = f"{td}/engineered_test.parquet"
        cls.AUTOSPLIT_TRAIN_PATH   = f"{td}/autosplit_train.parquet"
        cls.AUTOSPLIT_VALID_PATH   = f"{td}/autosplit_valid.parquet"
        cls.AUTOSPLIT_OOT_PATH     = f"{td}/autosplit_oot.parquet"
        cls.AUTOSPLIT_TEST_PATH    = f"{td}/autosplit_test.parquet"

        return cls.RUN_DIR

    @classmethod
    def chart_files(cls):
        """Ordered (path, label) for the 9 model diagnostic charts (Agent 3).

        Read at call time so the run-dir rebinding done in init_run() is
        reflected. Single source of truth — main.py / test_full_pipeline.py /
        pipeline._render_charts_section all iterate this so the chart list never
        drifts out of sync across callers.
        """
        return [
            (cls.CHART_ROC_PATH,              "ROC curve                   — Agent 3"),
            (cls.CHART_PR_PATH,               "Precision-Recall curve      — Agent 3"),
            (cls.CHART_KS_PATH,               "KS curve                    — Agent 3"),
            (cls.CHART_LIFT_PATH,             "Lift chart                  — Agent 3"),
            (cls.CHART_GAIN_PATH,             "Gain chart                  — Agent 3"),
            (cls.CHART_BAD_RATE_DECILE_PATH,  "Bad rate by decile          — Agent 3"),
            (cls.CHART_AVG_SCORE_DECILE_PATH, "Average score by decile     — Agent 3"),
            (cls.CHART_CALIBRATION_PATH,      "Calibration plot            — Agent 3"),
            (cls.CHART_SCORE_DIST_PATH,       "Score distribution Good/Bad — Agent 3"),
        ]

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

    # ── Null processing (Stage 1b, between Agent 1 and Agent 2) ──────────────
    # Fits a per-feature missing-value policy on TRAIN, replays it on every
    # other partition and in the replay bundle (see preprocessing/). Fully
    # automatic: rules come from the LLM (guard-railed) or a heuristic, and are
    # recorded in null_processor.json. false → skip the stage (legacy behaviour).
    NULL_PROCESSOR_ENABLED: bool = os.getenv("NULL_PROCESSOR_ENABLED", "true").lower() == "true"
    # false → heuristic rules only (no LLM call for this stage)
    NULL_PROCESSOR_USE_LLM: bool = os.getenv("NULL_PROCESSOR_USE_LLM", "true").lower() == "true"
    # Warn when a partition's missing rate differs from train by more than this (absolute)
    NULL_DRIFT_WARN_THRESHOLD: float = float(os.getenv("NULL_DRIFT_WARN_THRESHOLD", 0.10))
    # Scale (unit-change) guard: batch magnitude vs train, either direction.
    # >= WARN warns, >= FAIL raises; FAIL <= 0 disables. Batches under 100 rows are not judged.
    NULL_SCALE_WARN_RATIO: float = float(os.getenv("NULL_SCALE_WARN_RATIO", 3.0))
    NULL_SCALE_FAIL_RATIO: float = float(os.getenv("NULL_SCALE_FAIL_RATIO", 10.0))

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
    # Target number of new interaction features the LLM should generate per run.
    # Used by the Agent 2 system prompt to size FAMILY A–H distribution.
    # Scaled by dataset width: small datasets need fewer, wide datasets benefit
    # from more interactions. Override per-product via env var if needed.
    TARGET_NEW_FEATURE_COUNT: int = int(os.getenv("TARGET_NEW_FEATURE_COUNT", 25))
    # Master switch for the interaction-feature step. When false, Agent 2 skips
    # every `create_interaction` action the LLM proposes (the raw + encoded
    # features still flow through IV/WoE/selection). Use it when you only want
    # the model to learn from original columns, or to debug feature drift coming
    # from engineered ratios. Can also be overridden per-run via
    # FeatureEngineerAgent.process(..., create_interactions=False).
    FE_CREATE_INTERACTIONS_ENABLED: bool = os.getenv("FE_CREATE_INTERACTIONS_ENABLED", "true").lower() == "true"

    # ── IV / WoE feature selection (credit-risk gold standard) ───────────────
    # Default number of quantile bins used by compute_iv. 10 is the Siddiqi
    # convention; raise to 20 for very dense distributions, lower to 5 if
    # most features have < 50 unique values.
    IV_BINS_DEFAULT: int = int(os.getenv("IV_BINS_DEFAULT", 10))
    # IV at or above this is flagged as leakage-suspect (Siddiqi). compute_iv
    # surfaces these in the JSON summary; LLM is asked to audit before keeping.
    # Also serves as the upper bound of the Siddiqi "strong" band in the
    # FeatureEngineer system prompt — bands are: useless < 0.02 ≤ weak < 0.10
    # ≤ medium < 0.30 ≤ strong < IV_LEAKAGE_THRESHOLD ≤ leakage_suspect. So
    # raising this value widens the "strong" band and narrows the suspect zone.
    IV_LEAKAGE_THRESHOLD: float = float(os.getenv("IV_LEAKAGE_THRESHOLD", 0.50))
    # |Pearson| above which select_top_features (criterion='iv') drops the
    # lower-IV feature in a multicollinear pair. 0.85 matches the prompt's
    # explicit guidance; tighten to 0.75 for strict scorecard pipelines.
    MULTICOLLINEARITY_THRESHOLD: float = float(os.getenv("MULTICOLLINEARITY_THRESHOLD", 0.85))
    # Pre-filter: skip IV compute for columns with this much missing — at this
    # level the IV is dominated by the NaN-bin which is uninformative on its own.
    IV_MAX_NULL_RATIO: float = float(os.getenv("IV_MAX_NULL_RATIO", 0.50))
    # Minimum IV for a feature to be WoE-transformed by apply_woe_transform.
    # Below this (Siddiqi 'useless' band), WoE transformation adds noise more
    # than it linearises — better to leave the raw value for GBM to split on.
    # 0.02 matches the Siddiqi useless/weak boundary used elsewhere in the agent.
    WOE_MIN_IV: float = float(os.getenv("WOE_MIN_IV", 0.02))

    # ── IV stability (train ↔ valid/oot, diagnostic only) ────────────────────
    # After Agent 2 transforms a holdout partition it re-computes IV there using
    # the bin edges FROZEN from train, then compares against the train IV. This
    # catches what PSI cannot: a feature whose X-distribution is perfectly stable
    # but whose relationship to the target weakens or reverses out of time.
    # Diagnostic only — nothing is dropped here; Agent 3's PSI / SHAP+PSI prune
    # remain the only places features are removed.
    FE_IV_STABILITY_ENABLED: bool = os.getenv("FE_IV_STABILITY_ENABLED", "true").lower() == "true"
    # Relative IV loss vs train above which a feature is flagged UNSTABLE.
    # 0.30 = "IV on OOT fell by more than 30%" — the common scorecard rule of thumb.
    FE_IV_STABILITY_MAX_DROP: float = float(os.getenv("FE_IV_STABILITY_MAX_DROP", 0.30))
    # Pearson corr between the train WoE vector and the holdout WoE vector (per
    # bin). Below this the bin-level log-odds pattern no longer agrees with train
    # — the "relationship flipped" signal. Negative corr = outright reversal.
    FE_IV_STABILITY_MIN_WOE_CORR: float = float(os.getenv("FE_IV_STABILITY_MIN_WOE_CORR", 0.50))
    # Bins with fewer rows than this in the holdout are ignored for sign-flip and
    # WoE-correlation purposes — tail bins are too noisy to judge.
    FE_IV_STABILITY_MIN_BIN_COUNT: int = int(os.getenv("FE_IV_STABILITY_MIN_BIN_COUNT", 30))
    # How many worst-offending features to name in the WARN log line.
    FE_IV_STABILITY_TOP_N: int = int(os.getenv("FE_IV_STABILITY_TOP_N", 20))

    # ── Model Training ────────────────────────────────────────────────────────
    TRAIN_TEST_SPLIT_SIZE: float = float(os.getenv("TRAIN_TEST_SPLIT_SIZE", 0.2))
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", 42))
    CLASSIFICATION_UNIQUE_THRESHOLD: int = int(os.getenv("CLASSIFICATION_UNIQUE_THRESHOLD", 10))
    # n_splits for the final cross-validation evaluation of the trained model
    CV_N_SPLITS: int = int(os.getenv("CV_N_SPLITS", 5))

    # ── Optuna Hyperparameter Search Space ────────────────────────────────────
    # Shared across lgbm / xgboost / rf / extra_tree
    # Widened bounds (lr down to 0.005, depth up to 12, leaves up to 512) so
    # Optuna can explore the regimes that benefit big credit_risk datasets.
    # With early stopping enabled, large n_estimators upper bound is harmless —
    # actual trees grown stops at ES patience.
    LR_MIN: float = float(os.getenv("LR_MIN", 0.005))
    LR_MAX: float = float(os.getenv("LR_MAX", 0.3))
    MAX_DEPTH_MIN: int = int(os.getenv("MAX_DEPTH_MIN", 3))
    MAX_DEPTH_MAX: int = int(os.getenv("MAX_DEPTH_MAX", 12))
    N_ESTIMATORS_MIN: int = int(os.getenv("N_ESTIMATORS_MIN", 100))
    N_ESTIMATORS_MAX: int = int(os.getenv("N_ESTIMATORS_MAX", 2000))
    SUBSAMPLE_MIN: float = float(os.getenv("SUBSAMPLE_MIN", 0.4))
    SUBSAMPLE_MAX: float = float(os.getenv("SUBSAMPLE_MAX", 1.0))
    # LightGBM-specific
    NUM_LEAVES_MIN: int = int(os.getenv("NUM_LEAVES_MIN", 15))
    NUM_LEAVES_MAX: int = int(os.getenv("NUM_LEAVES_MAX", 512))
    # CatBoost-specific (depth range differs from XGBoost/LGBM)
    CB_DEPTH_MIN: int = int(os.getenv("CB_DEPTH_MIN", 4))
    CB_DEPTH_MAX: int = int(os.getenv("CB_DEPTH_MAX", 12))

    # ── OOT / Temporal Split ─────────────────────────────────────────────────────
    # Granularity of every temporal slice (OOT, valid_temporal, stability).
    #   "auto"    → monthly buckets (to_period("M")); detects monthly_snapshot vs
    #               intra_month but NEVER infers weekly (avoids mis-detection).
    #   "weekly"  → weekly buckets; user opts in explicitly for a weekly model.
    #   "monthly" → force monthly buckets.
    # For weekly, the week-closing weekday (snapshot/cutoff day) is auto-detected
    # as the most common weekday in date_col unless WEEK_CLOSING_DAY overrides it.
    # OOT_INIT_MONTHS / STABILITY_MIN_MONTHS are interpreted as counts of PERIODS
    # (months or weeks) according to the active frequency.
    TEMPORAL_FREQ: str = os.getenv("TEMPORAL_FREQ", "auto").lower()  # auto | weekly | monthly
    WEEK_CLOSING_DAY: str = os.getenv("WEEK_CLOSING_DAY", "").strip().upper()  # MON..SUN, "" = auto
    OOT_INIT_MONTHS: int = int(os.getenv("OOT_INIT_MONTHS", 2))   # floor: always include >= this many periods
    OOT_MIN_RATIO: float = float(os.getenv("OOT_MIN_RATIO", 0.15))  # expand until OOT >= this ratio
    OOT_MAX_RATIO: float = float(os.getenv("OOT_MAX_RATIO", 0.2))  # hard cap: shrink if OOT exceeds this
    VALID_TEMPORAL_RATIO: float = float(os.getenv("VALID_TEMPORAL_RATIO", 0.20))

    # ── AUC boosters ─────────────────────────────────────────────────────────────
    # Auto-enable class weighting when train majority/minority ratio exceeds this.
    # Set to a high value to disable. Boost mainly visible on imbalanced
    # credit_risk / fraud datasets.
    AUTO_CLASS_WEIGHT_THRESHOLD: float = float(os.getenv("AUTO_CLASS_WEIGHT_THRESHOLD", 3.0))
    # Strategy for the weight magnitude applied to the minority class.
    #   'auto'     → sqrt(ratio) when ratio >= CLASS_WEIGHT_SEVERE_THRESHOLD
    #                else ratio (sklearn 'balanced' equivalent). Recommended default.
    #   'balanced' → weight = ratio (full sklearn 'balanced' behaviour).
    #                Can over-weight minority when ratio is very high (banking 95:5).
    #   'sqrt'     → weight = sqrt(ratio). Softer; consistently best on credit_risk.
    #   'half'     → weight = ratio / 2. Middle ground between 'sqrt' and 'balanced'.
    #   <float>    → weight = float * ratio. Custom multiplier (e.g. 0.3 = 30% of balanced).
    CLASS_WEIGHT_STRATEGY: str = os.getenv("CLASS_WEIGHT_STRATEGY", "auto")
    # In 'auto' mode, ratios at or above this switch from 'balanced' to 'sqrt'.
    # 10.0 = banking-style severe imbalance (≤ 9% positive class) gets softer weighting.
    CLASS_WEIGHT_SEVERE_THRESHOLD: float = float(os.getenv("CLASS_WEIGHT_SEVERE_THRESHOLD", 10.0))
    # Early-stopping patience (rounds without valid AUC improvement) used in
    # Optuna trials + final train for lgbm/xgb/catboost. Speeds up training and
    # often improves AUC by avoiding over-training. RF/ExtraTrees ignore this.
    EARLY_STOPPING_ROUNDS: int = int(os.getenv("EARLY_STOPPING_ROUNDS", 200))
    # When feature pipeline (RFE → PSI → Stability → SHAP+PSI) shrinks the
    # feature set by >= this fraction of the initial, re-run a (smaller) Optuna
    # study on the FINAL feature set so hyperparams are not stale.
    ENABLE_RETUNE_AFTER_PRUNE: bool = os.getenv("ENABLE_RETUNE_AFTER_PRUNE", "true").lower() == "true"
    # Trigger threshold: re-tune only if n_final / n_init <= this. 0.7 means
    # "re-tune when >= 30% of features were pruned".
    RETUNE_FEATURE_REDUCTION_TRIGGER: float = float(os.getenv("RETUNE_FEATURE_REDUCTION_TRIGGER", 0.8))
    # Fraction of the original OPTUNA_TIMEOUT used by the re-tune study.
    RETUNE_TIMEOUT_RATIO: float = float(os.getenv("RETUNE_TIMEOUT_RATIO", 0.5))

    # ── FLAML AutoML ──────────────────────────────────────────────────────────
    FLAML_TIME_BUDGET: int = int(os.getenv("FLAML_TIME_BUDGET", 3600))
    FLAML_ESTIMATORS: str = os.getenv("FLAML_ESTIMATORS", "xgboost,lgbm,catboost,rf,extra_tree")
    FLAML_N_SPLITS: int = int(os.getenv("FLAML_N_SPLITS", 5))
    # Max training rows passed to FLAML — FLAML internally copies the DataFrame for block
    # consolidation which can OOM on large datasets. Sampling here keeps the copy small
    # while still giving FLAML enough signal to select the best estimator type.
    FLAML_MAX_ROWS: int = int(os.getenv("FLAML_MAX_ROWS", 500_000))

    # Final-model CV: early stopping picks best_iteration, but the tree ceiling comes from the tuned
    # n_estimators, so a fold can stop AT the ceiling (ES never fired). CV folds get this multiple of
    # the tuned tree count as headroom (boosters only; 1 = legacy behaviour, needed to reproduce
    # bundles written before this option existed).
    FINAL_ES_TREE_HEADROOM: float = float(os.getenv("FINAL_ES_TREE_HEADROOM", 2.0))

    # ── Optuna ────────────────────────────────────────────────────────────────
    # Trials raised 150 → 300 + timeout 30min → 2h to give the search room when
    # boosting_type=dart is included in the LightGBM space (DART is ~3-5x slower
    # per trial, needs more exploration budget). Compute-tolerant environments
    # only — drop these back if Optuna time is a constraint.
    OPTUNA_N_TRIALS: int = int(os.getenv("OPTUNA_N_TRIALS", 300))
    OPTUNA_TIMEOUT: int = int(os.getenv("OPTUNA_TIMEOUT", 7200))
    # Optuna's `timeout` is only checked BETWEEN trials, so one slow trial (DART, a 2000-tree
    # forest) can overrun the whole budget by hours. Each trial now gets its own wall-clock
    # cap (LightGBM / XGBoost: checked every boosting round, then the trial is pruned).
    # 0 = auto: max(120 s, study timeout / 8).
    OPTUNA_TRIAL_TIMEOUT: int = int(os.getenv("OPTUNA_TRIAL_TIMEOUT", 0))
    # DART has no early stopping and is 3-5x slower per trial; off by default.
    OPTUNA_ENABLE_DART: bool = os.getenv("OPTUNA_ENABLE_DART", "false").lower() == "true"
    # rf / extra_tree cannot be interrupted mid-fit, so bound their tree count instead.
    TREE_ENSEMBLE_MAX_ESTIMATORS: int = int(os.getenv("TREE_ENSEMBLE_MAX_ESTIMATORS", 500))
    # The post-prune re-tune replaces the tuned params only if it beats them (re-scored on the
    # final feature set) by at least this much AUC.
    RETUNE_MIN_GAIN: float = float(os.getenv("RETUNE_MIN_GAIN", 0.0005))

    # ── RFE ───────────────────────────────────────────────────────────────────
    RFE_TARGET_FEATURES: int = int(os.getenv("RFE_TARGET_FEATURES", 150))
    RFE_STEP: float = float(os.getenv("RFE_STEP", 0.01))
    RFE_CV_SPLITS: int = int(os.getenv("RFE_CV_SPLITS", 3))
    ENABLE_RFECV: bool = os.getenv("ENABLE_RFECV", "false").lower() == "true"
    # n_estimators used for the base model fitted during RFE selection
    RFE_N_ESTIMATORS: int = int(os.getenv("RFE_N_ESTIMATORS", 200))

    # ── Feature ranking / pruning strategy (Agent 3) ──────────────────────────
    # Every knob in this block picks HOW the candidate set is cut down; the legacy
    # behaviour of each is still available.
    #
    # FEATURE_RANK_METHOD — how RFE_TARGET_FEATURES candidates are chosen from the input set:
    #   "importance" (default) one model fit, rank by importance, drop near-duplicates.
    #                Minutes instead of hours.
    #   "rfe"        legacy sklearn RFE (refits many times; RFE_STEP controls the pace).
    FEATURE_RANK_METHOD: str = os.getenv("FEATURE_RANK_METHOD", "importance").lower()
    # Importance used for ranking AND for the batch prune below:
    #   "gain" (default, free from the fitted booster) | "shap" (mean |SHAP|, slower).
    FEATURE_RANK_IMPORTANCE: str = os.getenv("FEATURE_RANK_IMPORTANCE", "gain").lower()
    # Ranking-time near-duplicate filter: of two features with |corr| above this, keep the more
    # important one. 0 disables. (Applied on a row sample; -999 sentinel is treated as missing.)
    FEATURE_RANK_CORR_THRESHOLD: float = float(os.getenv("FEATURE_RANK_CORR_THRESHOLD", 0.90))
    # Trees / learning-rate floor of the ranking & prune models (early stopping on valid).
    FEATURE_RANK_N_ESTIMATORS: int = int(os.getenv("FEATURE_RANK_N_ESTIMATORS", 500))
    FEATURE_RANK_LEARNING_RATE: float = float(os.getenv("FEATURE_RANK_LEARNING_RATE", 0.05))
    # PRUNE_METHOD — final cut after PSI / Stability:
    #   "batch_tolerance" (default) remove PRUNE_BATCH_FRACTION of the weakest features per round,
    #                    record valid AUC per size, then pick the SMALLEST set whose AUC is within
    #                    PRUNE_AUC_TOLERANCE of the best (cap policy below can force <= MAX_FINAL_FEATURES).
    #   "one_by_one"     legacy SHAP+PSI loop (1 feature per step, stops after SHAP_PSI_MAX_NO_IMPROVE).
    PRUNE_METHOD: str = os.getenv("PRUNE_METHOD", "batch_tolerance").lower()
    PRUNE_BATCH_FRACTION: float = float(os.getenv("PRUNE_BATCH_FRACTION", 0.10))
    # Absolute valid-AUC loss accepted in exchange for a smaller feature set. 0 = never trade AUC.
    # (CV std on this kind of data is ~0.003, so 0.001 is inside the noise.)
    PRUNE_AUC_TOLERANCE: float = float(os.getenv("PRUNE_AUC_TOLERANCE", 0.001))
    # Stop shrinking once AUC is this far below the best seen: smaller sets are clearly worse.
    # The AUC-vs-size curve is noisy on small valid sets, and "smallest set within tolerance of the
    # peak" would latch onto a lucky spike. The rule is therefore applied to a moving average of
    # PRUNE_SMOOTH_WINDOW neighbouring sizes (odd; 1 = raw curve), and the tolerance can be widened to
    # PRUNE_SE_MULTIPLIER x the Hanley-McNeil standard error of the valid AUC (0 = off, default:
    # on Home Credit 0.5 cut 98 -> 52 features and cost ~0.003 test AUC; use 0.5 only for
    # small / noisy validation sets).
    PRUNE_SMOOTH_WINDOW: int = int(os.getenv("PRUNE_SMOOTH_WINDOW", 3))
    PRUNE_SE_MULTIPLIER: float = float(os.getenv("PRUNE_SE_MULTIPLIER", 0.0))
    PRUNE_STOP_DROP: float = float(os.getenv("PRUNE_STOP_DROP", 0.010))
    PRUNE_MAX_ROUNDS: int = int(os.getenv("PRUNE_MAX_ROUNDS", 40))
    # Optuna placement:
    #   true (default)  FLAML -> rank/prune with FLAML's params -> Optuna ONCE on the final features
    #                   (no re-tune step).
    #   false           legacy: Optuna on all features first, prune, then optional re-tune.
    OPTUNA_AFTER_SELECTION: bool = os.getenv("OPTUNA_AFTER_SELECTION", "true").lower() == "true"

    # ── PSI ───────────────────────────────────────────────────────────────────
    # 0.30 = finance standard "feature drift" cut-off. Raised to 0.35 because
    # the SHAP+PSI prune step already weighs PSI vs SHAP importance — keeping
    # moderately-drifting features lets prune pick the best combination
    # instead of dropping them upfront. +0.2-0.5% AUC on credit_risk.
    PSI_THRESHOLD: float = float(os.getenv("PSI_THRESHOLD", 0.35))
    PSI_BINS: int = int(os.getenv("PSI_BINS", 100))

    # ── Stability ─────────────────────────────────────────────────────────────
    STABILITY_MIN_MONTHS: int = int(os.getenv("STABILITY_MIN_MONTHS", 6))
    # Raised from 0.15 to 0.20: gives prune step more candidates to balance
    # against SHAP importance. Drops only features genuinely unstable across
    # monthly snapshots.
    STABILITY_GINI_STD_THRESHOLD: float = float(os.getenv("STABILITY_GINI_STD_THRESHOLD", 0.20))
    STABILITY_MIN_GINI: float = float(os.getenv("STABILITY_MIN_GINI", 0.01))

    # ── Final Feature Cut ─────────────────────────────────────────────────────
    # Soft target for the final feature count (FEATURE_CAP_POLICY=auc_first
    # treats it as a target, not a hard cap, and protects against picking up
    # noise). Raise toward 200 for banking models on 1.5M+ rows that benefit
    # from a wider feature set; lower when interpretability budget is tight.
    MAX_FINAL_FEATURES: int = int(os.getenv("MAX_FINAL_FEATURES", 100))

    # ── SHAP + PSI Iterative Pruning ──────────────────────────────────────────
    # n_estimators for the quick model fitted at each SHAP pruning step
    SHAP_N_ESTIMATORS: int = int(os.getenv("SHAP_N_ESTIMATORS", 200))
    # Max training rows sampled for SHAP value computation (speed vs accuracy).
    # Raised 5000 → 20000 — at 1.5M+ rows, 5K sample produces noisy importance
    # ranks that propagate into prune decisions. 20K reduces variance enough
    # that the prune step picks consistent features run-to-run.
    SHAP_SAMPLE_SIZE: int = int(os.getenv("SHAP_SAMPLE_SIZE", 20000))
    # Stop pruning after this many consecutive steps without AUC improvement.
    # Raised from 2 to 3 so the greedy prune can explore past a local plateau —
    # often the AUC dips one step then recovers on the next. +0.1-0.3% AUC.
    SHAP_PSI_MAX_NO_IMPROVE: int = int(os.getenv("SHAP_PSI_MAX_NO_IMPROVE", 3))
    # Absolute floor on minimum features kept after SHAP+PSI pruning
    SHAP_PSI_MIN_FEATURES_FLOOR: int = int(os.getenv("SHAP_PSI_MIN_FEATURES_FLOOR", 5))
    # Relative floor: keep at least this fraction of MAX_FINAL_FEATURES
    SHAP_PSI_MIN_FEATURES_RATIO: float = float(os.getenv("SHAP_PSI_MIN_FEATURES_RATIO", 0.10))
    # ── Feature-count policy: AUC-first vs Cap-first ──────────────────────────
    # Controls what SHAP+PSI prune does when it hits the early-stop streak
    # while the feature count is still above MAX_FINAL_FEATURES.
    #
    #   "auc_first" (default): respect SHAP_PSI_MAX_NO_IMPROVE — stop pruning
    #       the moment AUC stops improving, even if final count > MAX_FINAL_FEATURES.
    #       Prioritises model performance; the cap acts as a soft target.
    #
    #   "cap_first":  ignore the early-stop streak while above the cap; keep
    #       removing worst features until count ≤ MAX_FINAL_FEATURES, THEN
    #       re-enable early-stop for fine-tuning down to the floor.
    #       Prioritises hitting the feature-budget; AUC may drop a few bp.
    #
    # Pick "cap_first" when you have a hard deployment / interpretability
    # budget; pick "auc_first" when you care most about AUC.
    #   (batch_tolerance prune: "cap_first" picks the best-AUC set with <= MAX_FINAL_FEATURES when the
    #    tolerance rule alone would leave more; "auc_first" keeps the smallest set within tolerance.)
    FEATURE_CAP_POLICY: str = os.getenv("FEATURE_CAP_POLICY", "cap_first").lower()

    # ── Probability Calibration (IFRS9 / scorecard requirement) ──────────────
    # Wraps the final model with isotonic / sigmoid calibration fitted on a
    # held-out valid split. Required for IFRS9 Expected Credit Loss (PD must
    # be a calibrated probability, not just a ranking score). AUC-neutral to
    # slightly positive (+0-0.3%); big improvement on Brier score / log-loss.
    CALIBRATION_ENABLED: bool = os.getenv("CALIBRATION_ENABLED", "true").lower() == "true"
    # Calibration method:
    #   'isotonic' — non-parametric, more flexible, needs ≥ 1000 calibration rows
    #                (banking default; safe at 1.5M-row scale)
    #   'sigmoid'  — Platt scaling; cheaper, OK for very small calibration sets
    CALIBRATION_METHOD: str = os.getenv("CALIBRATION_METHOD", "isotonic")
    # PD floor / cap applied to calibrated probabilities. Isotonic can output exactly
    # 0 (no defaults in the lowest score band), which is not a usable PD for IFRS9 /
    # Basel (retail PD floor 0.03%). Stored in the model artifact, so scoring uses the
    # same bounds. Set PD_FLOOR=0 / PD_CAP=1 to disable.
    PD_FLOOR: float = float(os.getenv("PD_FLOOR", 0.0003))
    PD_CAP: float = float(os.getenv("PD_CAP", 0.9999))

    # ── Multi-seed bagging for final model ────────────────────────────────────
    # Trains N copies of the best-Optuna config with different random_state
    # seeds and averages predict_proba. Reduces variance; +0.1-0.3% AUC + much
    # more stable PD scores across retraining cycles.
    #
    # Off → single-model (legacy behaviour, only `RANDOM_STATE` is used).
    # On  → MULTI_SEED_N copies trained sequentially.
    MULTI_SEED_ENABLED: bool = os.getenv("MULTI_SEED_ENABLED", "true").lower() == "true"
    # Number of seeds when MULTI_SEED_ENABLED=true. Ignored when disabled.
    # 5 is the sweet spot — variance flattens, marginal AUC gain past 5 < 0.05%.
    MULTI_SEED_N: int = int(os.getenv("MULTI_SEED_N", 5))
    # Seed sequence: first one is RANDOM_STATE itself for reproducibility of
    # the legacy single-model path; extra seeds = RANDOM_STATE + k * offset.
    MULTI_SEED_BASE_OFFSET: int = int(os.getenv("MULTI_SEED_BASE_OFFSET", 1000))

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
    # PSI value above which a column is flagged as drifting between partitions
    # in the Stage-0 distribution check. SEPARATE from Agent 3's feature-drop
    # PSI_THRESHOLD (default 0.35) — kept lower (0.25) so Stage 0 acts as an
    # *early warning* about features that may later be borderline-OK but worth
    # investigating before they reach the drop threshold. Stage 0 only warns,
    # never drops.
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
    # Relative gap (ref_auc - holdout_auc) / ref_auc above which overfitting is
    # flagged and an LLM-guided retrain is triggered. ref_auc = cv_auc_mean (OOF
    # in-time; fallback valid_auc → valid_temporal_auc); holdout = oot / test.
    # OOF reference replaced in-sample valid after the refit-on-train+valid change
    # (valid is now in-sample) to avoid false overfit triggers.
    OVERFIT_THRESHOLD: float = float(os.getenv("OVERFIT_THRESHOLD", 0.12))

    # true → never try the LiteLLM proxy (no local gateway hosted): go straight to the
    # direct providers (OpenAI if a key is set, else Claude). Saves the ~14 s connection
    # retry that every LLM call otherwise burns before falling back.
    LLM_SKIP_PROXY: bool = os.getenv("LLM_SKIP_PROXY", "false").lower() == "true"

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
