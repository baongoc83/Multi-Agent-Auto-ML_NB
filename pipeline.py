import gc
import json
import numpy as np
import pandas as pd
from logger import AgentLogger
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from config import Config
import splitting
from Agents.BaseAgent.base_agent import BaseAgent
from Agents.DataCleaner.agent_data_cleaner import DataCleanerAgent
from Agents.FeatureEngineer.agent_feature_engineer import FeatureEngineerAgent
from Agents.TrainModel.agent_train_model import TrainModelAgent, _SPLIT_MARKER as _SPLIT_MARKER_COL

# Position of each row WITHIN its partition, as the original run ordered it.
_SPLIT_POS_COL = "_split_pos_"

# Occurrence rank of a row within its key group, numbered over the RAW input in
# file order. Zero everywhere when the key is unique, so the replay join stays a
# plain key join; only duplicate-key rows get 1, 2, ... to tell them apart.
# Needed because a run that deduplicates rows leaves the raw input holding
# several rows per key by definition — without a tiebreak the replay join fans
# out and cannot say which copy was train.
_KEY_OCC_COL = "_key_occ_"


def _dtype_kind(dtype_str: str) -> str:
    """Bucket a pyarrow / pandas dtype string into a coarse compatibility class.

    Cross-class mismatch (e.g. numeric vs object) is a real problem — it usually
    means one file stored a column as text and another as numeric, which will
    silently corrupt downstream encoders. Within-class mismatch (int32 vs int64,
    decimal vs float64) is normalised by Agent 1 / pandas itself.
    """
    d = dtype_str.lower()
    if any(x in d for x in ("int", "float", "double", "decimal", "number")):
        return "numeric"
    if any(x in d for x in ("date", "time", "timestamp")):
        return "datetime"
    if "bool" in d:
        return "bool"
    if any(x in d for x in ("string", "object", "utf8", "binary")):
        return "object"
    return "other"


class AutoMLPipeline:
    """Orchestrates the three-agent pipeline.

    Two modes:

    1. Split mode (valid_path or oot_path supplied):
       - Agent 1 fits CleaningSpec on TRAIN, writes clean_train.parquet,
         then replays the spec on valid/oot one at a time → clean_{valid,oot}.parquet
       - Agent 2 same with FeatureSpec → engineered_{train,valid,oot}.parquet
       - Agent 3 reads all three engineered files, adds `_split_` marker,
         concatenates ONCE (the only point where all 3 partitions live in one df),
         and runs training. The user-supplied splits are preserved exactly.

       Memory profile: agents 1+2 only ever load ONE partition at a time, so
       peak RAM ≈ size of the largest partition (typically train). The previous
       concat-then-process design loaded all three into pandas simultaneously
       which OOMed on 1M-row × 3000-col datasets.

    2. Single-file mode (no valid/oot supplied):
       - Stage 0 cuts the raw input into train / valid / (oot | test) via
         `splitting.auto_split` — the same function Agent 3 uses — and then
         delegates to split mode above.
       - A temporal split (date column present) yields an OOT tail; without one
         the fallback yields a random `test` holdout, kept under that name so
         PSI and the OOT metrics never imply a temporal guarantee.

    Single-file mode used to let Agent 3 split at the end, which meant Agents 1
    and 2 fitted imputation medians, IV, WoE bin edges and feature selection
    over rows that later became the holdout. Splitting upfront is what makes
    the reported holdout numbers honest; both modes now run the same chain.
    """

    def __init__(self):
        # Allocate run dir BEFORE the logger is constructed — AgentLogger reads
        # Config.EXECUTION_LOG_PATH at __init__ time, so init_run() must update
        # the Config path attrs first or the log goes to the stale default.
        Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        run_dir = Config.init_run()
        self.logger = AgentLogger()
        # Columns that identify a row in split_assignment.parquet. Set when the
        # split is frozen; read when the replay manifest is written.
        self._split_key_cols: List[str] = []
        self._split_key_positional: bool = False
        # Data-quality findings of every stage, as warnings rather than stops:
        # thresholds are domain-specific, so the run carries on and the user judges.
        # Written to data_quality_report.json and the top of final_report.md.
        self._dq_issues: List[Dict[str, Any]] = []
        self.logger.log("PIPELINE", "Run dir",
            f"persisted → {run_dir} | intermediates → "
            f"{'(same dir, KEEP_INTERMEDIATES=true)' if Config.KEEP_INTERMEDIATES else Config.TMP_DIR}")
        self._check_disk_space()

    def _check_disk_space(self) -> None:
        """Fail fast if free space on OUTPUT_DIR's filesystem is below the
        configured floor. Catches the "ran out of disk mid-CV" failure mode
        before any compute is burned. The check is local to OUTPUT_DIR — TMP_DIR
        usually lives on the same filesystem; if not, joblib will surface its
        own error which is still better than a half-written final_model.pkl.
        """
        import shutil
        try:
            usage = shutil.disk_usage(Config.OUTPUT_DIR)
        except OSError as e:
            self.logger.log("PIPELINE", "Disk check WARN",
                f"shutil.disk_usage({Config.OUTPUT_DIR}) failed: {e} — skipping pre-check")
            return
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        floor = Config.MIN_DISK_FREE_GB
        self.logger.log("PIPELINE", "Disk check",
            f"OUTPUT_DIR={Config.OUTPUT_DIR}  free={free_gb:.1f} GB / total={total_gb:.1f} GB  "
            f"(floor={floor:.1f} GB)")
        if free_gb < floor:
            raise RuntimeError(
                f"Insufficient disk space on {Config.OUTPUT_DIR}: "
                f"{free_gb:.1f} GB free < {floor:.1f} GB required.\n"
                f"Free up space or lower MIN_DISK_FREE_GB in .env if you accept the risk."
            )

    # ── Schema validation (split mode) ──────────────────────────────────────

    def _read_schema(self, path: str) -> Dict[str, str]:
        """Return {column_name: dtype_string} for `path` without loading full data.

        Parquet: pyarrow.parquet.read_schema → one disk seek, no row scan.
        CSV/TSV: pd.read_csv with nrows=1000, encoding-fallback chain.
        Other (Excel/Feather/ORC/JSON/S3): BaseAgent.load_dataframe fallback —
            for those formats the schema-only optimisation is less critical
            (Excel files are usually small; S3 parquet can route through
            pyarrow.dataset).
        """
        from pathlib import PurePosixPath
        # Recognise outer compression suffixes so train.csv.gz reads as csv
        inner_suffix, _ = BaseAgent._split_compression(path)
        suffix = inner_suffix or PurePosixPath(path).suffix.lower()

        # Local parquet — read schema only
        if suffix == ".parquet" and not path.startswith(BaseAgent._S3_SCHEMES):
            import pyarrow.parquet as pq
            schema = pq.read_schema(path)
            return {f.name: str(f.type) for f in schema}

        # Local CSV / TSV — sniff first 1000 rows for dtype inference
        if suffix in (".csv", ".tsv") and not path.startswith(BaseAgent._S3_SCHEMES):
            import pandas as pd
            sep = "\t" if suffix == ".tsv" else ","
            for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
                try:
                    head_df = pd.read_csv(path, sep=sep, nrows=1000, encoding=enc)
                    return {c: str(head_df[c].dtype) for c in head_df.columns}
                except UnicodeDecodeError:
                    continue
            head_df = pd.read_csv(path, sep=sep, nrows=1000, encoding="latin-1")
            return {c: str(head_df[c].dtype) for c in head_df.columns}

        # Fallback — small / remote formats: full load is acceptable for a
        # schema check, which only touches the first row of metadata anyway.
        df = BaseAgent.load_dataframe(path)
        return {c: str(df[c].dtype) for c in df.columns}

    def _validate_split_schema(
        self,
        train_path: str,
        valid_path: Optional[str],
        oot_path: Optional[str],
        target_column: str,
        entity_id_col: Optional[str],
        composite_key_cols: Optional[List[str]],
    ) -> None:
        """Fail-fast schema check across train/valid/oot.

        Hard errors (raise ValueError) — these break downstream agents:
          - target_column missing in any partition
          - entity_id_col / composite_key_cols missing in any partition

        Warnings (logged, do not raise) — handled gracefully later but worth
        surfacing to the user:
          - train has cols missing in valid/oot → spec replay skips them silently
          - valid/oot have extra cols → dropped during feature selection
          - dtype "kind" mismatch on a shared column (numeric vs object, etc.)
            → likely silent NaN / encoding errors during Agent 2 transform
        """
        schemas: Dict[str, Dict[str, str]] = {"train": self._read_schema(train_path)}
        if valid_path is not None:
            schemas["valid"] = self._read_schema(valid_path)
        if oot_path is not None:
            schemas["oot"] = self._read_schema(oot_path)

        sizes_msg = " | ".join(f"{p}={len(s)} cols" for p, s in schemas.items())
        self.logger.log("PIPELINE", "Schema scan", sizes_msg)

        # ── 1. HARD: required columns must exist in every partition ───────
        required: List[tuple] = []
        if target_column:
            required.append(("target", target_column))
        if entity_id_col:
            required.append(("entity_id", entity_id_col))
        for c in (composite_key_cols or []):
            required.append(("composite_key", c))

        missing_errors: List[str] = []
        for tag, col in required:
            for part, schema in schemas.items():
                if col not in schema:
                    missing_errors.append(f"{tag} column '{col}' missing in {part} ({part}_path)")
        if missing_errors:
            raise ValueError(
                "Schema validation FAILED — required columns missing:\n  - "
                + "\n  - ".join(missing_errors)
                + "\nCheck that train, valid and oot all use the same column names "
                "for target / entity_id / composite_key. Pass --keys=<cols> if your "
                "composite key columns differ from the entity id alone."
            )

        # ── 2. SOFT: column-set overlap between train and valid/oot ───────
        train_cols = set(schemas["train"].keys())
        for part in ("valid", "oot"):
            if part not in schemas:
                continue
            part_cols = set(schemas[part].keys())
            missing_in_part = train_cols - part_cols
            extra_in_part   = part_cols - train_cols
            if missing_in_part:
                examples = sorted(missing_in_part)[:5]
                self.logger.log("PIPELINE", "Schema WARN",
                    f"{part} is missing {len(missing_in_part)} cols present in train. "
                    f"These cols will be dropped by spec replay (silent). "
                    f"Examples: {examples}")
            if extra_in_part:
                examples = sorted(extra_in_part)[:5]
                self.logger.log("PIPELINE", "Schema info",
                    f"{part} has {len(extra_in_part)} extra cols not in train "
                    f"(will be dropped during feature selection). "
                    f"Examples: {examples}")

        # ── 3. SOFT: dtype-kind compatibility for shared columns ──────────
        dtype_warnings: List[str] = []
        for part in ("valid", "oot"):
            if part not in schemas:
                continue
            shared = train_cols & set(schemas[part].keys())
            for col in shared:
                t_kind = _dtype_kind(schemas["train"][col])
                p_kind = _dtype_kind(schemas[part][col])
                if t_kind != p_kind:
                    dtype_warnings.append(
                        f"'{col}': train={schemas['train'][col]} ({t_kind}) "
                        f"vs {part}={schemas[part][col]} ({p_kind})"
                    )
        if dtype_warnings:
            self.logger.log("PIPELINE", "Schema WARN",
                f"{len(dtype_warnings)} cols have dtype-kind mismatch across partitions "
                f"— may cause silent NaN / encoding errors during transform. "
                f"Examples: {dtype_warnings[:5]}")

        self.logger.log("PIPELINE", "Schema OK",
            f"all {len(required)} required cols present across {len(schemas)} partition(s)")

    # ── Distribution drift check (split mode) ───────────────────────────────

    @staticmethod
    def _calc_psi(expected, actual, bins: int = 10) -> float:
        """Population Stability Index between two numeric series.

        PSI = sum((actual% - expected%) * ln(actual% / expected%)) over `bins`
        quantile buckets fit to `expected`. The infinity guards (-inf / +inf
        boundary cuts; eps floor on counts) mirror Agent 3's implementation
        so the two stages report the same metric.
        """
        import numpy as np
        import pandas as pd
        eps = 1e-8
        try:
            cuts = pd.qcut(expected, q=bins, duplicates="drop", retbins=True)[1]
            cuts[0], cuts[-1] = -np.inf, np.inf
            e_pct = pd.cut(expected, bins=cuts).value_counts(normalize=True).sort_index() + eps
            a_pct = pd.cut(actual,   bins=cuts).value_counts(normalize=True).sort_index() + eps
            e_pct, a_pct = e_pct.align(a_pct, fill_value=eps)
            return float(((a_pct - e_pct) * np.log(a_pct / e_pct)).sum())
        except Exception:
            return 0.0

    def _load_sample(self, path: str, n: int):
        """Load the first ~n rows of `path` for distribution comparison.

        First-N-rows sampling (vs random sample) is intentional: it costs
        one sequential read and avoids loading the full file. The trade-off
        is that PSI may be biased if the file is sorted by date — but for a
        rough "is distribution very different?" warning that bias is
        acceptable. Agent 3's PSI step runs on full data later.
        """
        import pandas as pd
        from pathlib import PurePosixPath
        inner_suffix, _ = BaseAgent._split_compression(path)
        suffix = inner_suffix or PurePosixPath(path).suffix.lower()

        if suffix == ".parquet" and not path.startswith(BaseAgent._S3_SCHEMES):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            if pf.metadata.num_rows <= n:
                return pf.read().to_pandas()
            # iter_batches lets us stop after we've read enough rows
            chunks, rows = [], 0
            for batch in pf.iter_batches(batch_size=min(n, 20_000)):
                chunks.append(batch.to_pandas())
                rows += len(chunks[-1])
                if rows >= n:
                    break
            df = pd.concat(chunks, ignore_index=True)
            return df.iloc[:n] if len(df) > n else df

        if suffix in (".csv", ".tsv") and not path.startswith(BaseAgent._S3_SCHEMES):
            sep = "\t" if suffix == ".tsv" else ","
            for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
                try:
                    return pd.read_csv(path, sep=sep, nrows=n, encoding=enc)
                except UnicodeDecodeError:
                    continue
            return pd.read_csv(path, sep=sep, nrows=n, encoding="latin-1")

        # Other formats (Excel, Feather, ORC, JSON, S3) — full load. These are
        # usually small or pyarrow-streamed under the hood.
        df = BaseAgent.load_dataframe(path)
        if len(df) > n:
            return df.sample(n, random_state=Config.RANDOM_STATE).reset_index(drop=True)
        return df

    def _compare_distributions(
        self,
        train_path: str,
        valid_path: Optional[str],
        oot_path: Optional[str],
        target_column: str,
        entity_id_col: Optional[str],
        composite_key_cols: Optional[List[str]],
    ) -> None:
        """PSI-based drift check on a sample of numeric features.

        Skips: target, entity_id, composite keys, date-named cols.
        Per-partition behaviour: train↔valid and train↔oot are scored
        independently — drift in valid does not contaminate the oot stats.

        Only LOGS warnings — never raises. Distribution drift is rarely a
        reason to abort the pipeline (the user may want to model the shift)
        but they should know it's there before burning compute.
        """
        import numpy as np
        sample_n = Config.DRIFT_SAMPLE_N
        threshold = Config.DRIFT_PSI_THRESHOLD
        top_n = Config.DRIFT_TOP_N_REPORT

        self.logger.log("PIPELINE", "Distribution check start",
            f"sample={sample_n} rows/partition | PSI threshold={threshold:.2f}")

        train_sample = self._load_sample(train_path, sample_n)
        samples = {"train": train_sample}
        if valid_path is not None:
            samples["valid"] = self._load_sample(valid_path, sample_n)
        if oot_path is not None:
            samples["oot"] = self._load_sample(oot_path, sample_n)

        # Cols to skip — keys and the target carry no useful drift signal
        skip = set()
        if target_column:
            skip.add(target_column)
        if entity_id_col:
            skip.add(entity_id_col)
        for c in (composite_key_cols or []):
            skip.add(c)

        # Date-named columns — drift is expected (recency bias) and not
        # actionable here; Agent 3's temporal split handles it explicitly.
        _DATE_KW = ("date", "time", "timestamp", "dt", "snap", "period", "month", "year", "week", "day")
        date_like = [
            c for c in train_sample.columns
            if any(kw in c.lower() for kw in _DATE_KW)
        ]
        skip.update(date_like)

        numeric_cols = [
            c for c in train_sample.select_dtypes(include=[np.number]).columns
            if c not in skip
        ]
        if not numeric_cols:
            self.logger.log("PIPELINE", "Distribution check",
                "No numeric feature cols to compare — skipped")
            return

        # Score each non-train partition against train
        for part in ("valid", "oot"):
            if part not in samples:
                continue
            scores: Dict[str, float] = {}
            for col in numeric_cols:
                if col not in samples[part].columns:
                    continue
                tr = train_sample[col].dropna()
                ot = samples[part][col].dropna()
                # Need enough rows and >1 unique value in train for binning
                if len(tr) < 100 or len(ot) < 100 or tr.nunique() < 2:
                    continue
                psi = self._calc_psi(tr, ot)
                # PSI can return inf when expected has zero mass in a bin
                if np.isnan(psi) or np.isinf(psi):
                    continue
                scores[col] = round(psi, 4)

            if not scores:
                self.logger.log("PIPELINE", f"Distribution train↔{part}",
                    "No usable numeric cols to compare")
                continue

            sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            high_drift = [(c, s) for c, s in sorted_scores if s > threshold]
            max_col, max_psi = sorted_scores[0]

            self.logger.log("PIPELINE", f"Distribution train↔{part}",
                f"checked {len(scores)} numeric cols | "
                f"max PSI={max_psi:.3f} on '{max_col}' | "
                f"≥{threshold:.2f} count={len(high_drift)}")
            if high_drift:
                topk = high_drift[:top_n]
                top_str = ", ".join(f"{c}={s:.2f}" for c, s in topk)
                more = f" (+{len(high_drift) - len(topk)} more)" if len(high_drift) > len(topk) else ""
                self.logger.log("PIPELINE", f"Distribution WARN train↔{part}",
                    f"{len(high_drift)} cols with PSI > {threshold:.2f} between train and {part}. "
                    f"Top {len(topk)}: {top_str}{more}. "
                    "Agent 3's PSI step will likely drop these features.")
                self._record_dq(
                    "Stage 0b distribution", "psi_drift", "warn", part,
                    f"{len(high_drift)} column(s) with PSI > {threshold:.2f} vs train "
                    f"(max {max_psi:.2f} on '{max_col}'). Agent 3's PSI step will likely drop them.",
                    {c: s for c, s in high_drift})

    def run(
        self,
        input_path: str,
        target_column: str,
        col_descriptions_path: str = None,
        col_descriptions_kwargs: dict = None,
        entity_id_col: str = None,
        composite_key_cols: list = None,
        valid_path: str = None,
        oot_path: str = None,
        train_sample_ratio: float = None,
        prefilter: bool = True,
        check_distribution: bool = True,
        domain: str = "generic",
        model_type: str = "binary_classification",
        product_type: str = "generic",
        calibration: bool = None,
        temporal_freq: str = None,
        week_closing_day: str = None,
        create_interactions: bool = None,
    ):
        """Execute the full three-agent pipeline.

        Args:
            input_path:         TRAIN dataset path (any format supported by BaseAgent.load_dataframe).
            target_column:      Name of the binary target / label column.
            valid_path:         Optional path to a pre-split VALID dataset.
            oot_path:           Optional path to a pre-split OOT dataset.
            train_sample_ratio: Stratified sample fraction (0 < r < 1) applied to TRAIN only.
                                Useful for very wide datasets that OOM during cleaning.
            prefilter:          Drop near-constant / mostly-null cols from train before LLM analysis.
                                Same drops captured in CleaningSpec for replay on valid/oot.
            check_distribution: Run Stage-0 PSI drift check (default on). Set False to skip
                                — useful when intermediate files are already known stable
                                (e.g. re-runs with the same input data).
            domain:             Feature engineering domain (credit_risk | propensity | fraud | generic).
            model_type:         ML problem type (binary_classification | regression | multiclass).
            product_type:       Lending-product line for Agent 2's product-specific guidance
                                (consumer_unsecured | credit_card | mortgage | auto | overdraft
                                 | bnpl | sme | generic).
            calibration:        Per-run override for probability calibration (Agent 3).
                                None (default) → use Config.CALIBRATION_ENABLED;
                                True/False → force on/off for this run only, no env change.
            temporal_freq:      Per-run override for the temporal-split granularity (Agent 3).
                                None (default) → use Config.TEMPORAL_FREQ ("auto" = monthly);
                                "weekly" → OOT / valid_temporal / stability move in whole
                                weeks. Weekly is opt-in only (never auto-inferred).
            week_closing_day:   Snapshot/cutoff weekday for weekly runs (MON..SUN). None →
                                auto-detect the most common weekday in the date column.
            create_interactions: Per-run override for Agent 2's interaction-feature step.
                                None (default) → use Config.FE_CREATE_INTERACTIONS_ENABLED;
                                False → skip all create_interaction actions this run.
        """
        split_mode = (valid_path is not None) or (oot_path is not None)
        if split_mode:
            parts_label = ["train"]
            if valid_path: parts_label.append("valid")
            if oot_path:   parts_label.append("oot")
            mode_label = f"split ({'+'.join(parts_label)})"
        else:
            mode_label = "single-file (auto-split upfront, then split mode)"
        self.logger.log("PIPELINE", "Starting",
            f"Input: {input_path} | Target: {target_column} | Domain: {domain} | "
            f"Product: {product_type} | Model: {model_type} | Mode: {mode_label}")

        try:
            if split_mode:
                return self._run_split_mode(
                    train_path=input_path, valid_path=valid_path, oot_path=oot_path,
                    target_column=target_column,
                    col_descriptions_path=col_descriptions_path,
                    col_descriptions_kwargs=col_descriptions_kwargs,
                    entity_id_col=entity_id_col, composite_key_cols=composite_key_cols,
                    train_sample_ratio=train_sample_ratio, prefilter=prefilter,
                    check_distribution=check_distribution,
                    domain=domain, model_type=model_type, product_type=product_type,
                    calibration=calibration,
                    temporal_freq=temporal_freq, week_closing_day=week_closing_day,
                    create_interactions=create_interactions,
                )
            return self._run_single_mode(
                input_path=input_path, target_column=target_column,
                col_descriptions_path=col_descriptions_path,
                col_descriptions_kwargs=col_descriptions_kwargs,
                entity_id_col=entity_id_col, composite_key_cols=composite_key_cols,
                domain=domain, model_type=model_type, product_type=product_type,
                calibration=calibration,
                temporal_freq=temporal_freq, week_closing_day=week_closing_day,
                create_interactions=create_interactions,
                train_sample_ratio=train_sample_ratio, prefilter=prefilter,
                check_distribution=check_distribution,
            )
        finally:
            # Persist the log even when the run dies (hours-long runs: OOM / kill / error).
            try:
                self.logger.save()
            except Exception:
                pass
            # Remove the intermediate tempdir even when the pipeline raises —
            # /tmp/automl_pipeline_* gigabytes lying around defeat the point of
            # writing them out of RUN_DIR in the first place.
            Config.cleanup_run()

    # ── Mode 1: split (train + valid + oot) ─────────────────────────────────

    def _run_split_mode(
        self,
        train_path: str,
        valid_path: str,
        oot_path: str,
        target_column: str,
        col_descriptions_path: str,
        col_descriptions_kwargs: dict,
        entity_id_col: str,
        composite_key_cols: list,
        train_sample_ratio: float,
        prefilter: bool,
        check_distribution: bool,
        domain: str,
        model_type: str,
        product_type: str = "generic",
        calibration: bool = None,
        temporal_freq: str = None,
        week_closing_day: str = None,
        create_interactions: bool = None,
        test_path: str = None,
        temporal_meta: dict = None,
        skip_schema_validation: bool = False,
        source_input: str = None,
    ):
        """Fit Agents 1+2 on TRAIN, replay their transforms on every holdout, train on all.

        test_path / temporal_meta / skip_schema_validation are set by
        `_run_single_mode` when the partitions came from `_auto_split_input`;
        callers who supplied their own --valid / --oot leave them at the
        defaults and get exactly the previous behaviour.
        """
        # ── Stage 0a: schema validation (fail-fast before any agent runs) ─
        if skip_schema_validation:
            self.logger.log("PIPELINE", "Stage 0a",
                "Skipped — partitions were cut from a single frame, schemas match by construction")
        else:
            self.logger.log("PIPELINE", "Stage 0a", "Validating schema across train / valid / oot")
            self._validate_split_schema(
                train_path=train_path,
                valid_path=valid_path,
                oot_path=oot_path,
                target_column=target_column,
                entity_id_col=entity_id_col,
                composite_key_cols=composite_key_cols,
            )

        # ── Stage 0b: distribution drift check (warning only) ────────────
        if check_distribution:
            self.logger.log("PIPELINE", "Stage 0b", "Checking distribution drift between partitions")
            try:
                self._compare_distributions(
                    train_path=train_path,
                    valid_path=valid_path,
                    oot_path=oot_path,
                    target_column=target_column,
                    entity_id_col=entity_id_col,
                    composite_key_cols=composite_key_cols,
                )
            except Exception as e:
                # Drift check is advisory — never block the pipeline on its failure
                self.logger.log("PIPELINE", "Distribution check ERROR",
                    f"Drift check failed ({type(e).__name__}: {e}) — continuing anyway")

        # ── Stage 1: DataCleaner ────────────────────────────────────────
        self.logger.log("PIPELINE", "Stage 1", "Initializing Data Cleaner Agent (split mode)")
        agent1 = DataCleanerAgent(
            self.logger,
            entity_id_col=entity_id_col,
            composite_key_cols=composite_key_cols,
            target_column=target_column,
        )
        # Freeze the split for the replay bundle. In single-file mode
        # _auto_split_input already did this; here the partitions came from the
        # user, so record them as given.
        if not self._split_key_cols:
            self._freeze_supplied_split(
                {"train": train_path, "valid": valid_path,
                 "oot": oot_path, "test": test_path},
                entity_id_col, composite_key_cols)

        clean_paths, report1 = agent1.process_splits(
            train_path=train_path,
            valid_path=valid_path,
            oot_path=oot_path,
            test_path=test_path,
            prefilter=prefilter,
            train_sample_ratio=train_sample_ratio,
        )
        del agent1   # release agent 1's state (incl. last df) before agent 2

        # ── Stage 1b: Null processing (fit on TRAIN, replay everywhere) ──
        self._null_info = self._apply_null_processing(
            clean_paths, report1.get("target_column") or target_column,
            protected={report1.get("entity_id_col"), *(report1.get("composite_key_cols") or [])})

        # ── Stage 2: FeatureEngineer ────────────────────────────────────
        self.logger.log("PIPELINE", "Stage 2", "Initializing Feature Engineer Agent (split mode)")
        agent2 = FeatureEngineerAgent(
            self.logger,
            col_descriptions_path=col_descriptions_path,
            domain=domain,
            model_type=model_type,
            product_type=product_type,
            **(col_descriptions_kwargs or {}),
        )
        # Capture the loaded descriptions before agent2 is released so Agent 3
        # can reuse them for the SHAP-explain step without reloading.
        col_descriptions_loaded = dict(agent2._col_descriptions) if agent2._col_descriptions else {}
        agent2.null_context = getattr(self, "_null_context", None)
        eng_paths, report2 = agent2.process_splits(
            train_path=clean_paths["train"],
            previous_report=report1,
            target_column=target_column,
            valid_path=clean_paths.get("valid"),
            oot_path=clean_paths.get("oot"),
            test_path=clean_paths.get("test"),
            create_interactions=create_interactions,
        )
        del agent2

        # ── Stage 3: TrainModel ─────────────────────────────────────────
        self.logger.log("PIPELINE", "Stage 3", "Initializing Train Model Agent (split mode)")
        agent3 = TrainModelAgent(
            self.logger,
            col_descriptions=col_descriptions_loaded,
            domain=domain,
        )
        final_metrics, report3 = agent3.process_splits(
            train_path=eng_paths["train"],
            previous_report=report2,
            target_column=target_column,
            valid_path=eng_paths.get("valid"),
            oot_path=eng_paths.get("oot"),
            test_path=eng_paths.get("test"),
            calibration=calibration,
            temporal_freq=temporal_freq,
            week_closing_day=week_closing_day,
            temporal_meta=temporal_meta,
        )

        try:
            self._write_replay_bundle(
                report1, report2, report3, final_metrics,
                # In single-file mode train_path points at the temporary
                # autosplit parquet, which is deleted at the end of the run.
                # The replay has to be pointed at the file the user actually has.
                input_path=source_input or train_path,
                mode="single-file (auto-split)" if source_input else "split",
                target_column=target_column,
                entity_id_col=entity_id_col,
                composite_key_cols=composite_key_cols,
                temporal_meta=temporal_meta,
                calibration=calibration,
            )
        except Exception as e:
            # A missing replay bundle must never cost the user a finished run.
            self.logger.log("PIPELINE", "Replay bundle ERROR",
                f"could not write the replay bundle ({type(e).__name__}: {e}) — "
                "the model and reports above are unaffected")

        self._generate_final_report(report1, report2, report3, final_metrics)
        self.logger.log("PIPELINE", "Complete", "All agents finished successfully")
        self.logger.save()
        return final_metrics

    # ── Stage 1b: null processing ───────────────────────────────────────────

    def _apply_null_processing(self, clean_paths: Dict[str, Optional[str]],
                               target_column: str,
                               protected: Optional[set] = None) -> Optional[Dict[str, Any]]:
        """Fit a NullProcessor on clean TRAIN, rewrite every clean partition.

        Rules are proposed automatically (LLM behind deterministic guardrails,
        heuristic fallback) and frozen into null_processor.json. Only one
        partition is in memory at a time. Key / target columns are untouched.
        """
        if not Config.NULL_PROCESSOR_ENABLED:
            self.logger.log("PIPELINE", "Stage 1b", "Null processing disabled (NULL_PROCESSOR_ENABLED=false)")
            return None
        from preprocessing import NullProcessor, suggest_rules

        self.logger.log("PIPELINE", "Stage 1b", "Fitting null processor on TRAIN")
        train = BaseAgent.load_dataframe(clean_paths["train"])
        protected = {c for c in (protected or set()) if c} | {
            target_column, *self._split_key_cols, _SPLIT_MARKER_COL}
        cols = [c for c in train.columns if c not in protected]

        llm_call = None
        if Config.NULL_PROCESSOR_USE_LLM:
            advisor = BaseAgent("NullProcessor", "Missing-value policy advisor", self.logger)
            llm_call = lambda prompt, system: advisor.call_llm(
                prompt, system, json_mode=True, max_tokens=Config.LLM_MAX_TOKENS_LARGE)
        rules = suggest_rules(train[cols], llm_call=llm_call)["features"]

        proc = NullProcessor({"features": rules}, on_unconfigured="ignore",
                             drift_warn_threshold=Config.NULL_DRIFT_WARN_THRESHOLD,
                             scale_warn_ratio=Config.NULL_SCALE_WARN_RATIO,
                             scale_fail_ratio=Config.NULL_SCALE_FAIL_RATIO,
                             guard_action="raise" if Config.DATA_GUARD_ACTION in ("fail", "raise") else "warn")
        proc.fit(train[cols], train[target_column] if target_column in train.columns else None)
        proc.passthrough_columns_ = [c for c in train.columns if c not in cols]
        sha = proc.save(Config.NULL_PROCESSOR_PATH)

        n_src = {}
        for r in rules.values():
            n_src[r["source"]] = n_src.get(r["source"], 0) + 1
        strat = {}
        for r in rules.values():
            strat[r["missing_strategy"]] = strat.get(r["missing_strategy"], 0) + 1
        self.logger.log("PIPELINE", "Null processor fitted",
            f"{len(rules)} rules | source={n_src} | strategy={strat} | "
            f"indicators={len(proc.indicator_columns_)} | sha256={sha[:12]}…")

        for tag in ("train", "valid", "oot", "test"):
            path = clean_paths.get(tag)
            if not path:
                continue
            df = train if tag == "train" else BaseAgent.load_dataframe(path)
            out = proc.transform(df)
            out.to_parquet(path, compression="snappy", index=False)
            issues = proc.report_.get("issues", [])
            self.logger.log("PIPELINE", f"Null processor {tag}",
                f"shape {df.shape} -> {out.shape}" + (f" | {len(issues)} data-quality issue(s)" if issues else ""))
            self._record_null_issues(tag, issues)
            del out
            if tag != "train":
                del df
            gc.collect()
        del train
        # Handed to Agent 2 so it knows which NULLs are gone and which indicators exist.
        self._null_context = {
            "imputed": {c: r.missing_strategy for c, r in proc.rules_.items()
                        if r.missing_strategy not in ("none", "missing_indicator_only")},
            "kept_nan": [c for c, r in proc.rules_.items()
                         if r.missing_strategy in ("none", "missing_indicator_only")],
            "indicators": list(proc.indicator_columns_),
        }
        self._save_dq_report()                      # step report, rewritten at the end of the run
        return {"file": Path(Config.NULL_PROCESSOR_PATH).name, "sha256": sha,
                "n_rules": len(rules), "n_indicators": len(proc.indicator_columns_),
                "llm": llm_call is not None}

    # ── Data-quality warnings (all stages) ──────────────────────────────────

    _DQ_HINTS = {
        "psi_drift": "Population / feature-pipeline change between periods? Check with the data owner; "
                     "tune DRIFT_PSI_THRESHOLD / PSI_THRESHOLD for this domain.",
        "scale_unit_change": "Median AND p95 moved together: unit / definition change likely "
                             "(e.g. VND vs thousand VND). Verify upstream; tune NULL_SCALE_FAIL_RATIO.",
        "scale_distribution_shift": "One quantile moved (heavy tail / mix change), not a unit change. "
                                    "Tune NULL_SCALE_WARN_RATIO if expected for this domain.",
        "missing_rate_drift": "NULL share differs from train: upstream join / coverage change? "
                              "Tune NULL_DRIFT_WARN_THRESHOLD.",
        "unparseable": "Non-numeric values in a numeric column were treated as NULL.",
    }

    def _record_dq(self, stage: str, check: str, severity: str, partition: Optional[str],
                   message: str, columns: Optional[Dict[str, Any]] = None) -> None:
        cols = dict(list((columns or {}).items())[:300])        # keep the report readable
        self.__dict__.setdefault("_dq_issues", []).append({
            "stage": stage, "check": check, "severity": severity, "partition": partition,
            "message": message, "n_columns": len(columns or {}), "columns": cols,
            "hint": self._DQ_HINTS.get(check, ""),
        })
        self.logger.log("PIPELINE", f"DATA QUALITY {severity.upper()} [{stage}{'/' + partition if partition else ''}]",
                        message)

    def _record_null_issues(self, partition: str, issues: List[Dict[str, Any]]) -> None:
        """Fold per-column NullProcessor issues into one entry per (check, severity)."""
        groups: Dict[Tuple[str, str], Dict[str, str]] = {}
        for it in issues:
            groups.setdefault((it["check"], it["severity"]), {})[it["column"]] = it["message"]
        labels = {"scale_unit_change": "changed scale (median and p95 together) beyond "
                                       f"x{Config.NULL_SCALE_FAIL_RATIO:g}",
                  "scale_distribution_shift": f"shifted median or p95 beyond x{Config.NULL_SCALE_WARN_RATIO:g}",
                  "missing_rate_drift": f"missing rate differs from train by > {Config.NULL_DRIFT_WARN_THRESHOLD:.0%}",
                  "unparseable": "had unparseable values (treated as NULL)"}
        for (check, sev), cols in sorted(groups.items(), key=lambda kv: kv[0][1] != "critical"):
            first = next(iter(cols.values()))
            self._record_dq("Stage 1b null processing", check, sev, partition,
                            f"{len(cols)} column(s) {labels.get(check, check)}. e.g. {first}", cols)

    def _save_dq_report(self) -> None:
        sev = [i["severity"] for i in getattr(self, "_dq_issues", [])]
        body = {"guard_action": Config.DATA_GUARD_ACTION,
                "summary": {"critical": sev.count("critical"), "warn": sev.count("warn")},
                "issues": getattr(self, "_dq_issues", [])}
        try:
            Path(Config.DATA_QUALITY_REPORT_PATH).write_text(
                json.dumps(body, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        except Exception as e:      # a report must never cost the run
            self.logger.log("PIPELINE", "Data quality report ERROR", f"{type(e).__name__}: {e}")

    def _collect_log_warnings(self) -> List[Tuple[str, str, str]]:
        """WARN / SKIP lines logged by any stage that are not already structured."""
        seen, out = set(), []
        for e in self.logger.logs:
            act = e.get("action", "")
            if act.startswith(("Distribution WARN", "DATA QUALITY")):
                continue
            if "WARN" in act or act.startswith("SKIP"):
                key = (e["agent"], act, str(e["details"])[:300])
                if key not in seen:
                    seen.add(key)
                    out.append(key)
        return out

    def _render_data_quality_section(self) -> str:
        crit = [i for i in getattr(self, "_dq_issues", []) if i["severity"] == "critical"]
        warn = [i for i in getattr(self, "_dq_issues", []) if i["severity"] == "warn"]
        other = self._collect_log_warnings()
        if not (crit or warn or other):
            return "# Data quality\n\nNo data-quality warnings were raised.\n\n"
        out = ["# Data quality warnings\n\n",
               f"**{len(crit)} critical · {len(warn)} warning(s) · {len(other)} other notice(s)** "
               f"(DATA_GUARD_ACTION=`{Config.DATA_GUARD_ACTION}`: the run continued; review before go-live). "
               f"Full column lists: `{Path(Config.DATA_QUALITY_REPORT_PATH).name}`.\n\n"]
        if crit or warn:
            out += ["| Severity | Stage | Partition | Check | Finding | What to check |\n",
                    "|---|---|---|---|---|---|\n"]
            for i in crit + warn:
                msg = i["message"].replace("|", "\\|")
                out.append(f"| {'🔴 critical' if i['severity'] == 'critical' else '🟡 warn'} | "
                           f"{i['stage']} | {i['partition'] or '-'} | `{i['check']}` | {msg} | {i['hint']} |\n")
            out.append("\n")
        if other:
            out.append("**Other notices logged by the agents:**\n\n")
            for agent, act, det in other[:40]:
                out.append(f"- {agent} · {act}: {det.replace(chr(10), ' ')[:300]}\n")
            if len(other) > 40:
                out.append(f"- … {len(other) - 40} more in agent_execution.log\n")
            out.append("\n")
        return "".join(out)

    # ── Replay bundle: manifest + driver ────────────────────────────────────

    def _write_replay_bundle(
        self,
        report1: Dict[str, Any],
        report2: Dict[str, Any],
        report3: Dict[str, Any],
        metrics: Dict[str, Any],
        *,
        input_path: str,
        mode: str,
        target_column: str,
        entity_id_col: Optional[str],
        composite_key_cols: Optional[List[str]],
        temporal_meta: Optional[Dict[str, Any]],
        calibration: Optional[bool],
    ) -> None:
        """Write replay_manifest.json + replay_pipeline.py into RUN_DIR.

        Together with cleaning_spec.pkl, feature_spec.pkl, final_model.pkl and
        split_assignment.parquet, these let another environment re-run this
        exact run end to end — see docs/replay.md.
        """
        fp = report3.get("feature_pipeline", {}) or {}
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Relative to the repo root when inside it, so the bundle carries no machine path.
            "run_dir": (lambda p: (str(p.relative_to(Path(__file__).resolve().parent)).replace("\\", "/")
                                   if p.is_relative_to(Path(__file__).resolve().parent) else str(p)))(
                Path(Config.RUN_DIR).resolve()),
            "source_input": str(input_path),
            "pipeline_mode": mode,
            "target_column": target_column,
            "entity_id_col": entity_id_col,
            "composite_key_cols": list(composite_key_cols or []),

            "split": {
                "assignment_file": Path(Config.SPLIT_ASSIGNMENT_PATH).name
                                   if self._split_key_cols else None,
                "key_cols": list(self._split_key_cols),
                "positional_fallback": self._split_key_positional,
                "temporal": temporal_meta or report3.get("temporal", {}),
                "sizes": report3.get("splits", {}),
            },
            "cleaning": {
                "spec_file": Path(Config.PIPELINE_PROCESS_DC_SPEC_PATH).name,
                # Train-only row ops. apply() does NOT replay these; the driver
                # applies them to the train partition and nothing else.
                "row_ops": (report1.get("cleaning_spec", {}) or {}).get("row_ops", []),
            },
            # Applied after cleaning (and train-only row ops), before the feature
            # spec. None when the stage was disabled for this run.
            "null_processing": getattr(self, "_null_info", None),
            "data_quality": {
                "report": Path(Config.DATA_QUALITY_REPORT_PATH).name,
                "guard_action": Config.DATA_GUARD_ACTION,
                "critical": sum(i["severity"] == "critical" for i in getattr(self, "_dq_issues", [])),
                "warn": sum(i["severity"] == "warn" for i in getattr(self, "_dq_issues", [])),
            },
            "feature_engineering": {
                "spec_file": Path(Config.PIPELINE_PROCESS_FE_SPEC_PATH).name,
                "n_final_features": len(report2.get("final_features", []) or []),
            },
            "training": {
                "estimator": report3.get("best_estimator"),
                "best_params": report3.get("best_params", {}),
                "final_features": fp.get("final_features", []),
                "n_final_features": fp.get("n_final"),
                "model_file": Path(Config.FINAL_MODEL_PATH).name,
                "calibration": calibration,
                "random_state": Config.RANDOM_STATE,
                "multi_seed_n": Config.MULTI_SEED_N,
                "cv_n_splits": Config.CV_N_SPLITS,
            },
            # Recorded so a replay can state plainly whether it reproduced the
            # original numbers instead of leaving the user to eyeball them.
            "expected_metrics": metrics,
        }
        try:
            import provenance
            manifest["provenance"] = provenance.collect(input_path, Config)
        except Exception as e:      # never cost the user a finished run
            self.logger.log("PIPELINE", "Provenance WARN", f"{type(e).__name__}: {e}")
        # SHA-256 of every artifact the replay will unpickle / read. The replay
        # driver refuses to joblib.load a file whose hash differs (pickle = code).
        import hashlib
        manifest["artifacts"] = {}
        for p in (Config.PIPELINE_PROCESS_DC_SPEC_PATH, Config.PIPELINE_PROCESS_FE_SPEC_PATH,
                  Config.FINAL_MODEL_PATH, Config.NULL_PROCESSOR_PATH, Config.SPLIT_ASSIGNMENT_PATH):
            fp_ = Path(p)
            if fp_.exists():
                h = hashlib.sha256()
                with open(fp_, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                manifest["artifacts"][fp_.name] = h.hexdigest()
        Path(Config.REPLAY_MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(Config.REPLAY_MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)

        # Copied rather than generated from a template string: replay_driver.py
        # is a real module in the repo, so it gets linted, imported and tested
        # like any other file instead of only being syntax-checked at runtime.
        driver_src = Path(__file__).resolve().parent / "replay_driver.py"
        Path(Config.REPLAY_DRIVER_PATH).write_text(
            driver_src.read_text(encoding="utf-8"), encoding="utf-8")

        self.logger.log("PIPELINE", "Replay bundle",
            f"manifest={Path(Config.REPLAY_MANIFEST_PATH).name} | "
            f"driver={Path(Config.REPLAY_DRIVER_PATH).name} | "
            f"split_key={self._split_key_cols or 'n/a'} | "
            f"features={fp.get('n_final')}")
        print(f"Replay bundle saved to: {Config.RUN_DIR}")

    # ── Mode 2: single file (auto-split upfront, then split mode) ───────────

    # ── Replay bundle: freeze the split row-by-row ──────────────────────────

    def _freeze_supplied_split(
        self,
        paths: Dict[str, Optional[str]],
        entity_id_col: Optional[str],
        composite_key_cols: Optional[List[str]],
    ) -> None:
        """Record the split when the user supplied the partition files.

        Reads only the key columns from each file, so this stays cheap even on
        very wide inputs. Positional fallback is refused here: separate files
        have no shared row ordering to fall back on, so a key is required —
        without one the bundle just records nothing and says why.
        """
        first = next((p for p in paths.values() if p), None)
        if first is None:
            return
        cols = list(BaseAgent.load_dataframe(first).head(0).columns)
        key_cols, is_positional = self._resolve_split_key(
            cols, entity_id_col, composite_key_cols)
        if is_positional:
            self.logger.log("PIPELINE", "Split assignment skipped",
                "pre-split mode needs an entity id or composite key to record which "
                "row went where; none found. replay_pipeline.py will fall back to "
                "reading the partition files directly.")
            return

        parts: Dict[str, Optional[pd.DataFrame]] = {}
        for tag, path in paths.items():
            if not path:
                continue
            df = BaseAgent.load_dataframe(path)
            parts[tag] = df[[c for c in key_cols if c in df.columns]].copy()
            del df
        gc.collect()
        self._write_split_assignment(parts, key_cols, False)
        self._split_key_cols = key_cols
        self._split_key_positional = False


    _ROW_IDX_COL = "_replay_row_idx_"

    def _resolve_split_key(
        self,
        columns: List[str],
        entity_id_col: Optional[str],
        composite_key_cols: Optional[List[str]],
    ) -> Tuple[List[str], bool]:
        """Pick the columns that identify a row across runs.

        Returns (key_cols, is_positional). Preference order:
          1. composite key  — the only thing that is unique on panel data
             (same customer appears once per snapshot).
          2. entity id alone.
          3. positional row index — a synthetic fallback that is only valid if
             the replay input has the SAME rows in the SAME order. Flagged so
             the manifest and the driver can warn about it.
        """
        comp = [c for c in (composite_key_cols or []) if c in columns]
        if len(comp) >= 1:
            return comp, False
        if entity_id_col and entity_id_col in columns:
            return [entity_id_col], False
        return [self._ROW_IDX_COL], True

    def _write_split_assignment(
        self,
        parts: Dict[str, Optional[pd.DataFrame]],
        key_cols: List[str],
        is_positional: bool,
    ) -> Dict[str, int]:
        """Record which partition every row landed in, keyed by `key_cols`.

        This is the authoritative split record for a replay: re-deriving the
        split from thresholds would silently drift the moment the input data or
        an OOT_* config value changes, and the whole point of the bundle is that
        it cannot drift.
        """
        frames = []
        sizes: Dict[str, int] = {}
        for tag, part in parts.items():
            if part is None or len(part) == 0:
                continue
            sizes[tag] = len(part)
            cols = [c for c in key_cols if c in part.columns]
            if _KEY_OCC_COL in part.columns:
                cols = cols + [_KEY_OCC_COL]
            sub = part[cols].copy()
            sub[_SPLIT_MARKER_COL] = tag
            # Row ORDER, not just membership. auto_split hands back a train
            # partition shuffled by train_test_split, and `valid` as
            # valid_temporal followed by valid_random — neither is in input
            # order. StratifiedKFold then assigns folds by position, so a replay
            # that rebuilt the same rows in input order would train on different
            # folds and land on a different best_iteration.
            sub[_SPLIT_POS_COL] = np.arange(len(part), dtype=np.int64)
            frames.append(sub)
        assignment = pd.concat(frames, ignore_index=True)

        Path(Config.SPLIT_ASSIGNMENT_PATH).parent.mkdir(parents=True, exist_ok=True)
        assignment.to_parquet(Config.SPLIT_ASSIGNMENT_PATH, compression="snappy", index=False)

        join_cols = key_cols + ([_KEY_OCC_COL] if _KEY_OCC_COL in assignment.columns else [])
        dup = int(assignment.duplicated(subset=join_cols).sum())
        self.logger.log("PIPELINE", "Split assignment saved",
            f"{len(assignment)} rows | join={join_cols} | sizes={sizes} | "
            f"path={Config.SPLIT_ASSIGNMENT_PATH}")
        if dup:
            self.logger.log("PIPELINE", "Split assignment WARN",
                f"{dup} assignment row(s) are still ambiguous under {join_cols} — "
                "a replay cannot match those rows to a partition.")
        if is_positional:
            self.logger.log("PIPELINE", "Split assignment WARN",
                "No entity/composite key available — split frozen by ROW POSITION. "
                "A replay is only exact if the input file has the same rows in the "
                "same order.")
        return sizes

    def _auto_split_input(
        self,
        input_path: str,
        target_column: str,
        entity_id_col: Optional[str],
        composite_key_cols: Optional[List[str]],
        temporal_freq: Optional[str],
        week_closing_day: Optional[str],
    ) -> Tuple[Dict[str, Optional[str]], Dict[str, Any]]:
        """Cut the raw input into train / valid / (oot | test) before Agent 1 runs.

        Single-file mode used to hand the whole file to Agents 1 and 2 and let
        Agent 3 split afterwards. That meant Agent 2 computed IV, WoE bin edges
        and feature selection over rows that later became the holdout — the
        reported holdout numbers were measured on data that had already
        influenced which features existed. Splitting here removes that: every
        statistic downstream is fitted on TRAIN only, and Agent 3 inherits
        exactly these partitions instead of re-cutting its own.

        Returns ({"train": path, "valid": path|None, "oot": path|None,
        "test": path|None}, temporal_meta).
        """
        df = BaseAgent.load_dataframe(input_path)

        # Same rule Agent 2 uses for _date_col: the composite-key member that
        # is not the entity id.
        date_col = next(
            (c for c in (composite_key_cols or [])
             if c != entity_id_col and c in df.columns),
            None,
        )
        key_cols, is_positional = self._resolve_split_key(
            list(df.columns), entity_id_col, composite_key_cols)
        if is_positional:
            # No usable identity key — stamp the original row position so the
            # split can still be frozen and replayed (order-dependent).
            df[self._ROW_IDX_COL] = np.arange(len(df), dtype=np.int64)
        # Numbered on the raw frame, before the split, so a key that appears in
        # two partitions still gets distinct ranks.
        df[_KEY_OCC_COL] = df.groupby(key_cols, sort=False, dropna=False).cumcount()
        n_dup_keys = int((df[_KEY_OCC_COL] > 0).sum())
        if n_dup_keys:
            self.logger.log("PIPELINE", "Split key not unique",
                f"{n_dup_keys} row(s) share a key under {key_cols}; they are "
                "distinguished by occurrence order within the raw file. Exact for "
                "fully duplicate rows; if these rows differ, pass a composite key "
                "that is unique per row.")

        self.logger.log("PIPELINE", "Stage 0 auto-split",
            f"shape={df.shape} | date_col={date_col or '-'} "
            f"| split key={key_cols}{' (positional fallback)' if is_positional else ''} "
            f"| temporal_freq={temporal_freq or Config.TEMPORAL_FREQ}")

        res = splitting.auto_split(
            df,
            target_column=target_column,
            date_col=date_col,
            temporal_freq=temporal_freq if temporal_freq is not None else Config.TEMPORAL_FREQ,
            week_closing_day=(week_closing_day if week_closing_day is not None
                              else Config.WEEK_CLOSING_DAY),
            log=lambda stage, msg: self.logger.log("PIPELINE", f"Auto-split {stage}", msg),
        )
        del df
        gc.collect()

        parts = {"train": res.train, "valid": res.valid,
                 "oot": res.oot, "test": res.test}
        self._write_split_assignment(parts, key_cols, is_positional)
        self._split_key_cols = key_cols
        self._split_key_positional = is_positional

        Path(Config.AUTOSPLIT_TRAIN_PATH).parent.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Optional[str]] = {
            "train": None, "valid": None, "oot": None, "test": None,
        }
        for tag, out_path in [
            ("train", Config.AUTOSPLIT_TRAIN_PATH),
            ("valid", Config.AUTOSPLIT_VALID_PATH),
            ("oot",   Config.AUTOSPLIT_OOT_PATH),
            ("test",  Config.AUTOSPLIT_TEST_PATH),
        ]:
            part = parts[tag]
            if part is None or len(part) == 0:
                continue
            # The synthetic index was only ever for freezing the split — it must
            # not reach the agents as a feature.
            helper = [c for c in (self._ROW_IDX_COL, _KEY_OCC_COL) if c in part.columns]
            if helper:
                part = part.drop(columns=helper)
            part.to_parquet(out_path, compression="snappy", index=False)
            paths[tag] = out_path

        if paths["train"] is None:
            raise ValueError(
                f"Auto-split produced an empty TRAIN partition from {input_path} — "
                "check the target column and date column before re-running"
            )

        self.logger.log("PIPELINE", "Stage 0 auto-split done",
            " | ".join(f"{t}={len(p)}" for t, p in
                       [("train", res.train), ("valid", res.valid),
                        ("oot", res.oot), ("test", res.test)] if len(p))
            + f" | cadence={res.date_cadence}")
        return paths, res.temporal_meta

    def _run_single_mode(
        self,
        input_path: str,
        target_column: str,
        col_descriptions_path: str,
        col_descriptions_kwargs: dict,
        entity_id_col: str,
        composite_key_cols: list,
        domain: str,
        model_type: str,
        product_type: str = "generic",
        calibration: bool = None,
        temporal_freq: str = None,
        week_closing_day: str = None,
        create_interactions: bool = None,
        train_sample_ratio: float = None,
        prefilter: bool = True,
        check_distribution: bool = True,
    ):
        """Split the input upfront, then run the ordinary split-mode pipeline.

        There is deliberately no separate agent chain for single-file mode any
        more — one code path means Agent 2 can never be fitting on a different
        notion of "train" than Agent 3 evaluates against.
        """
        paths, temporal_meta = self._auto_split_input(
            input_path=input_path,
            target_column=target_column,
            entity_id_col=entity_id_col,
            composite_key_cols=composite_key_cols,
            temporal_freq=temporal_freq,
            week_closing_day=week_closing_day,
        )
        return self._run_split_mode(
            train_path=paths["train"],
            valid_path=paths["valid"],
            oot_path=paths["oot"],
            test_path=paths["test"],
            target_column=target_column,
            col_descriptions_path=col_descriptions_path,
            col_descriptions_kwargs=col_descriptions_kwargs,
            entity_id_col=entity_id_col, composite_key_cols=composite_key_cols,
            train_sample_ratio=train_sample_ratio, prefilter=prefilter,
            check_distribution=check_distribution,
            domain=domain, model_type=model_type, product_type=product_type,
            calibration=calibration,
            temporal_freq=temporal_freq, week_closing_day=week_closing_day,
            create_interactions=create_interactions,
            temporal_meta=temporal_meta,
            # Partitions were cut from one frame moments ago — their schemas
            # cannot disagree, so re-reading all of them to prove it is waste.
            skip_schema_validation=True,
            source_input=input_path,
        )

    def _generate_final_report(self, report1, report2, report3, metrics):
        self._save_dq_report()
        markdown = self.logger.get_markdown_report()
        # Data-quality warnings go first so nobody has to dig through the log for them
        cut = markdown.find("## [")
        cut = len(markdown) if cut < 0 else cut
        markdown = markdown[:cut] + self._render_data_quality_section() + markdown[cut:]
        markdown += "\n# Final Summary\n\n"
        markdown += "## Agent 1: Data Cleaner\n"
        markdown += f"- Actions: {report1.get('summary', 'N/A')}\n\n"
        markdown += "## Agent 2: Feature Engineer\n"
        markdown += f"- Strategy: {report2.get('summary', 'N/A')}\n\n"
        markdown += "## Agent 3: Model Trainer\n"
        markdown += f"- Final Metrics: {metrics}\n\n"
        markdown += self._render_temporal_section(report3)
        markdown += self._render_charts_section(report3)
        markdown += self._render_shap_section(report3)
        markdown += self._render_token_section()

        with open(Config.FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"Final Report saved to: {Config.FINAL_REPORT_PATH}")

    def _render_temporal_section(self, report3: Dict) -> str:
        """State the temporal cadence so the report is explicit that a snapshot
        dataset produces a MONTHLY/WEEKLY model (OOT / valid_temporal / stability
        all move in whole-period steps). Skipped for non-temporal runs."""
        t = (report3 or {}).get("temporal", {}) or {}
        cadence = t.get("cadence", "non_temporal")
        if cadence == "non_temporal":
            return ""
        unit = t.get("period_unit", "month")
        labels = {
            "monthly_snapshot": "Monthly snapshot — model operates at MONTHLY granularity",
            "intra_month":      "Intra-month feed — bucketed to whole months",
            "weekly_snapshot":  "Weekly snapshot — model operates at WEEKLY granularity",
            "weekly":           "Weekly feed — bucketed to whole weeks",
        }
        out: List[str] = [
            "## Temporal cadence\n\n",
            f"- **Cadence**: {labels.get(cadence, cadence)}"
            + (f" (period freq `{t['period_freq']}`)" if t.get("period_freq") else "") + "\n",
            f"- **History**: {t.get('n_periods_total', '?')} {unit}s "
            f"[{t.get('first_period', '?')} .. {t.get('last_period', '?')}] "
            f"({t.get('n_distinct_dates', '?')} distinct dates in `{t.get('date_col', 'date')}`)\n",
        ]
        if "oot_periods" in t:
            out.append(
                f"- **Split (whole {unit}s)**: train={t.get('train_periods', '?')} | "
                f"valid_temporal={t.get('valid_temporal_periods', '?')} | "
                f"oot={t.get('oot_periods', '?')}\n"
            )
        out.append("\n")
        return "".join(out)

    def _render_token_section(self) -> str:
        """LLM token attribution per-model + grand total. Helps users compare
        token cost between pipeline runs and between LLM model tiers."""
        usage = getattr(self.logger, "token_usage", {}) or {}
        if not usage:
            return ""
        total = self.logger.token_summary()
        out: List[str] = [
            "## LLM token usage\n",
            "| Model | Calls | Input | Output | Total |\n",
            "|---|---:|---:|---:|---:|\n",
        ]
        for model, b in sorted(usage.items(), key=lambda kv: -kv[1].get("total", 0)):
            out.append(
                f"| `{model}` | {b.get('calls', 0)} | "
                f"{b.get('input', 0):,} | {b.get('output', 0):,} | {b.get('total', 0):,} |\n"
            )
        out.append(
            f"| **Total** | **{total['calls']}** | "
            f"**{total['input']:,}** | **{total['output']:,}** | **{total['total']:,}** |\n\n"
            "*Multiply by your provider's per-token price to get $ cost. "
            "Token counts cover every LLM call in this run (Agent 1+2+3 decisions, "
            "SHAP explain, overfit reg suggest).*\n\n"
        )
        return "".join(out)

    def _render_charts_section(self, report3: Dict) -> str:
        """Embed the 9 model diagnostic charts (Agent 3) into the final report.

        Charts live in `{run_dir}/charts/` so each image is referenced with a
        `charts/<name>.png` relative path — renders correctly when the run dir
        is zipped or served statically. Only charts that were actually saved
        (file exists on disk) are embedded; the rest are silently skipped so a
        partially-failed chart step doesn't leave broken image links.
        """
        from pathlib import Path as _P
        rendered = []
        for path, label in Config.chart_files():
            p = _P(path)
            if not p.exists():
                continue
            # Strip the "— Agent 3" suffix for the in-report caption
            caption = label.split("—")[0].strip()
            rel = f"{p.parent.name}/{p.name}"
            rendered.append((caption, rel))

        if not rendered:
            return ""

        out: List[str] = ["## Model diagnostic charts\n\n"]
        for caption, rel in rendered:
            out.append(f"**{caption}**\n\n")
            out.append(f"![{caption}]({rel})\n\n")
        out.append(
            "_Charts computed on the best available eval split (OOT > valid > test) "
            "using the calibrated ensemble — reflect deployed scoring behaviour._\n\n"
        )
        return "".join(out)

    def _render_shap_section(self, report3: Dict) -> str:
        """Render the SHAP visual + top-N feature explanations into the
        final-report markdown. Empty string when the SHAP step was disabled
        or produced no records (so the report stays clean instead of showing
        an "N/A" placeholder)."""
        top = report3.get("shap_top_features", []) if isinstance(report3, dict) else []
        if not top:
            return ""

        from pathlib import Path as _P
        out: List[str] = ["## SHAP — Top features driving the final model\n\n"]

        # Bar plot (importance magnitude)
        bar_path = _P(Config.SHAP_PLOT_PATH)
        if bar_path.exists():
            out.append("**Importance magnitude (bar):**\n\n")
            out.append(f"![SHAP summary]({bar_path.name})\n\n")

        # Beeswarm (impact direction + per-sample distribution)
        beeswarm_path = _P(Config.SHAP_BEESWARM_PATH)
        if beeswarm_path.exists():
            out.append("**Impact on model output (beeswarm — direction + magnitude per sample):**\n\n")
            out.append(f"![SHAP beeswarm]({beeswarm_path.name})\n\n")
            out.append(
                "_Dot = sample. X = SHAP value (right → pushes prediction up). "
                "Color = feature value (red = high, blue = low). "
                "Cluster red-right = high value drives positive class._\n\n"
            )

        out.append(
            "| Rank | Feature | SHAP importance | Meaning | Why it matters |\n"
            "|---:|---|---:|---|---|\n"
        )
        for rec in top:
            feat   = rec.get("feature", "")
            imp    = rec.get("shap_importance", 0.0)
            mean_  = (rec.get("meaning") or rec.get("description") or "").replace("|", "\\|").replace("\n", " ")
            why    = (rec.get("why_matters") or "").replace("|", "\\|").replace("\n", " ")
            out.append(f"| {rec.get('rank', '')} | `{feat}` | {imp:.4f} | {mean_} | {why} |\n")

        out.append(
            f"\n*Full bundle: `{_P(Config.SHAP_FINAL_MODEL_REPORT_PATH).name}` "
            f"(dedicated SHAP report) + `{_P(Config.SHAP_FEATURE_REPORT_PATH).name}` (CSV). "
            "SHAP computed on the FINAL model after overfitting handling — "
            "values reflect what actually drives the deployed predictions.*\n\n"
        )
        return "".join(out)
