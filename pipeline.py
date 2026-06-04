from logger import AgentLogger
from handoff import Handoff
from pathlib import Path
from typing import Dict, List, Optional
from config import Config
from Agents.BaseAgent.base_agent import BaseAgent
from Agents.DataCleaner.agent_data_cleaner import DataCleanerAgent
from Agents.FeatureEngineer.agent_feature_engineer import FeatureEngineerAgent
from Agents.TrainModel.agent_train_model import TrainModelAgent


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
       - Agent 1 cleans the one input file → clean_data.parquet
       - Agent 2 engineers features → engineered_data.parquet
       - Agent 3 does its own temporal-OOT or 60/20/20 auto-split

    Both modes share the same agent code; only the pipeline orchestration
    differs.
    """

    def __init__(self):
        self.logger = AgentLogger()
        self.handoff = Handoff(self.logger)
        Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)

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
        """
        split_mode = (valid_path is not None) or (oot_path is not None)
        if split_mode:
            parts_label = ["train"]
            if valid_path: parts_label.append("valid")
            if oot_path:   parts_label.append("oot")
            mode_label = f"split ({'+'.join(parts_label)})"
        else:
            mode_label = "single-file (auto-split in Agent 3)"
        self.logger.log("PIPELINE", "Starting",
            f"Input: {input_path} | Target: {target_column} | Domain: {domain} | "
            f"Model: {model_type} | Mode: {mode_label}")

        if split_mode:
            return self._run_split_mode(
                train_path=input_path, valid_path=valid_path, oot_path=oot_path,
                target_column=target_column,
                col_descriptions_path=col_descriptions_path,
                col_descriptions_kwargs=col_descriptions_kwargs,
                entity_id_col=entity_id_col, composite_key_cols=composite_key_cols,
                train_sample_ratio=train_sample_ratio, prefilter=prefilter,
                check_distribution=check_distribution,
                domain=domain, model_type=model_type,
            )
        return self._run_single_mode(
            input_path=input_path, target_column=target_column,
            col_descriptions_path=col_descriptions_path,
            col_descriptions_kwargs=col_descriptions_kwargs,
            entity_id_col=entity_id_col, composite_key_cols=composite_key_cols,
            domain=domain, model_type=model_type,
        )

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
    ):
        # ── Stage 0a: schema validation (fail-fast before any agent runs) ─
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
        clean_paths, report1 = agent1.process_splits(
            train_path=train_path,
            valid_path=valid_path,
            oot_path=oot_path,
            prefilter=prefilter,
            train_sample_ratio=train_sample_ratio,
        )
        del agent1   # release agent 1's state (incl. last df) before agent 2

        # ── Stage 2: FeatureEngineer ────────────────────────────────────
        self.logger.log("PIPELINE", "Stage 2", "Initializing Feature Engineer Agent (split mode)")
        agent2 = FeatureEngineerAgent(
            self.logger,
            col_descriptions_path=col_descriptions_path,
            domain=domain,
            model_type=model_type,
            **(col_descriptions_kwargs or {}),
        )
        eng_paths, report2 = agent2.process_splits(
            train_path=clean_paths["train"],
            previous_report=report1,
            target_column=target_column,
            valid_path=clean_paths.get("valid"),
            oot_path=clean_paths.get("oot"),
        )
        del agent2

        # ── Stage 3: TrainModel ─────────────────────────────────────────
        self.logger.log("PIPELINE", "Stage 3", "Initializing Train Model Agent (split mode)")
        agent3 = TrainModelAgent(self.logger)
        final_metrics, report3 = agent3.process_splits(
            train_path=eng_paths["train"],
            previous_report=report2,
            target_column=target_column,
            valid_path=eng_paths.get("valid"),
            oot_path=eng_paths.get("oot"),
        )

        self._generate_final_report(report1, report2, report3, final_metrics)
        self.logger.log("PIPELINE", "Complete", "All agents finished successfully")
        self.logger.save()
        return final_metrics

    # ── Mode 2: single file (auto-split in Agent 3) ─────────────────────────

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
    ):
        self.logger.log("PIPELINE", "Stage 1", "Initializing Data Cleaner Agent (single-file mode)")
        agent1 = DataCleanerAgent(
            self.logger,
            entity_id_col=entity_id_col,
            composite_key_cols=composite_key_cols,
            target_column=target_column,
        )
        clean_data_path, report1 = agent1.process(input_path)
        self.handoff.set_data(clean_data_path, report1, "DataCleaner")
        del agent1

        self.logger.log("PIPELINE", "Stage 2", "Initializing Feature Engineer Agent (single-file mode)")
        agent2 = FeatureEngineerAgent(
            self.logger,
            col_descriptions_path=col_descriptions_path,
            domain=domain,
            model_type=model_type,
            **(col_descriptions_kwargs or {}),
        )
        engineered_data_path, report2 = agent2.process(
            self.handoff.get_data(),
            self.handoff.get_report(),
            target_column,
        )
        self.handoff.set_data(engineered_data_path, report2, "FeatureEngineer")
        del agent2

        self.logger.log("PIPELINE", "Stage 3", "Initializing Train Model Agent (single-file mode)")
        agent3 = TrainModelAgent(self.logger)
        final_metrics, report3 = agent3.process(
            self.handoff.get_data(),
            self.handoff.get_report(),
            target_column,
            oot_df=None,
        )

        self._generate_final_report(report1, report2, report3, final_metrics)
        self.logger.log("PIPELINE", "Complete", "All agents finished successfully")
        self.logger.save()
        return final_metrics

    def _generate_final_report(self, report1, report2, report3, metrics):
        markdown = self.logger.get_markdown_report()
        markdown += "\n# Final Summary\n\n"
        markdown += "## Agent 1: Data Cleaner\n"
        markdown += f"- Actions: {report1.get('summary', 'N/A')}\n\n"
        markdown += "## Agent 2: Feature Engineer\n"
        markdown += f"- Strategy: {report2.get('summary', 'N/A')}\n\n"
        markdown += "## Agent 3: Model Trainer\n"
        markdown += f"- Final Metrics: {metrics}\n\n"

        with open(Config.FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"Final Report saved to: {Config.FINAL_REPORT_PATH}")
