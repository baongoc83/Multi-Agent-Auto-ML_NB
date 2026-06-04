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
        domain: str,
        model_type: str,
    ):
        # ── Stage 0: schema validation (fail-fast before any agent runs) ─
        self.logger.log("PIPELINE", "Stage 0", "Validating schema across train / valid / oot")
        self._validate_split_schema(
            train_path=train_path,
            valid_path=valid_path,
            oot_path=oot_path,
            target_column=target_column,
            entity_id_col=entity_id_col,
            composite_key_cols=composite_key_cols,
        )

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
