from logger import AgentLogger
from handoff import Handoff
from pathlib import Path
import numpy as np
import pandas as pd
from config import Config
from Agents.BaseAgent.base_agent import BaseAgent
from Agents.DataCleaner.agent_data_cleaner import DataCleanerAgent
from Agents.FeatureEngineer.agent_feature_engineer import FeatureEngineerAgent
from Agents.TrainModel.agent_train_model import TrainModelAgent


# Marker column used to preserve user-specified splits through Agents 1+2.
# Agents protect this column from drop/encode/interaction; Agent 3 reads it
# to reconstruct train/valid/oot instead of auto-splitting.
SPLIT_MARKER = "_split_"


class AutoMLPipeline:
    """Orchestrates the three-agent pipeline."""

    def __init__(self):
        self.logger = AgentLogger()
        self.handoff = Handoff(self.logger)
        Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)

    def _scan_useful_columns(
        self,
        df: "pd.DataFrame",
        target_column: str = None,
        entity_id_col: str = None,
        composite_key_cols: list = None,
    ) -> list:
        """Identify columns worth keeping before combining files. Drops:
          - All-null cols
          - High-null cols   (null ratio > PREFILTER_MAX_NULL_RATIO)
          - Constant cols    (nunique <= 1)
          - Near-constant    (dominant value ratio > PREFILTER_MAX_DOMINANT_RATIO,
                              only checked when nunique <= 1000)

        Always keeps: target_column, entity_id_col, composite_key_cols,
                      SPLIT_MARKER (if present).
        """
        null_threshold = Config.PREFILTER_MAX_NULL_RATIO
        dominant_threshold = Config.PREFILTER_MAX_DOMINANT_RATIO

        keep_mandatory = {SPLIT_MARKER}
        if target_column:
            keep_mandatory.add(target_column)
        if entity_id_col:
            keep_mandatory.add(entity_id_col)
        if composite_key_cols:
            keep_mandatory.update(composite_key_cols)

        cols_keep: list = []
        n_dropped = {"null": 0, "constant": 0, "dominant": 0}
        n_total = len(df) if len(df) > 0 else 1

        for col in df.columns:
            if col in keep_mandatory:
                cols_keep.append(col)
                continue

            series = df[col]
            null_ratio = series.isnull().sum() / n_total
            if null_ratio > null_threshold:
                n_dropped["null"] += 1
                continue

            non_null = series.dropna()
            nunique = non_null.nunique()
            if nunique <= 1:
                n_dropped["constant"] += 1
                continue

            # Near-constant check (skip high-cardinality cols)
            if nunique <= 1000:
                try:
                    top_count = non_null.value_counts(dropna=True).iloc[0]
                    dominant_ratio = top_count / len(non_null)
                    if dominant_ratio > dominant_threshold:
                        n_dropped["dominant"] += 1
                        continue
                except Exception:
                    pass

            cols_keep.append(col)

        self.logger.log("PIPELINE", "Pre-filter scan",
            f"keep={len(cols_keep)} / drop={sum(n_dropped.values())} "
            f"(null>{null_threshold:.0%}={n_dropped['null']} | "
            f"constant={n_dropped['constant']} | "
            f"dominant>{dominant_threshold:.0%}={n_dropped['dominant']})")
        return cols_keep

    def _stratified_sample(
        self,
        df: "pd.DataFrame",
        target_col: str,
        ratio: float,
    ) -> "pd.DataFrame":
        """Sample `ratio` fraction of rows stratified by target — INDEX-based
        to keep peak memory bounded.

        pandas `df.groupby(target).apply(lambda g: g.sample(...))` first
        materialises each group as a full sub-DataFrame, peaking at
        ~2× memory and silently killing the kernel on 1M-row × 1000-col
        datasets. This implementation instead:
          1. Pulls target as a 1-D numpy array (cheap).
          2. Computes per-class row positions then samples positions
             (small integer arrays, MB-scale).
          3. Does ONE final df.iloc[positions] copy of size ratio × N.

        Peak overhead ≈ size of the SAMPLED result, not the whole frame.
        """
        rs = Config.RANDOM_STATE
        rng = np.random.default_rng(rs)
        n_rows = len(df)

        self.logger.log("PIPELINE", "Pre-split sample starting",
            f"input rows={n_rows} cols={df.shape[1]} target_ratio={ratio:.2f}")

        # Fallback: random sample if no usable target
        if (target_col is None
                or target_col not in df.columns
                or df[target_col].nunique() < 2):
            n_sample = max(1, int(n_rows * ratio))
            positions = rng.choice(n_rows, n_sample, replace=False)
            positions.sort()
            result = df.iloc[positions].reset_index(drop=True)
            self.logger.log("PIPELINE", "Pre-split sample done (random)",
                f"output rows={len(result)}")
            return result

        # Stratified: per-class position indices
        target_arr = df[target_col].to_numpy()
        sampled_positions = []
        class_summary = {}
        for cls in np.unique(target_arr):
            cls_pos = np.where(target_arr == cls)[0]
            n_cls = max(1, int(len(cls_pos) * ratio))
            chosen = rng.choice(cls_pos, n_cls, replace=False)
            sampled_positions.append(chosen)
            class_summary[str(cls)] = (len(cls_pos), n_cls)
        positions = np.concatenate(sampled_positions)
        positions.sort()
        del target_arr, sampled_positions

        self.logger.log("PIPELINE", "Pre-split sample indices",
            f"per-class kept = {class_summary} | total to materialise={len(positions)}")

        # One vectorized .iloc — pandas does a single take operation.
        result = df.iloc[positions].reset_index(drop=True)
        self.logger.log("PIPELINE", "Pre-split sample done (stratified)",
            f"output rows={len(result)} cols={result.shape[1]}")
        return result

    @staticmethod
    def _normalize_numeric_types(table):
        """Cast Decimal columns to float64 so train/valid/oot can share one
        writer schema.

        Decimal columns store a fixed (precision, scale) in their pyarrow
        type. If train's value range fits in Decimal(4, 2) but valid has a
        larger value that needs Decimal(6, 2), the cast train_schema(valid)
        fails with "Decimal value does not fit in precision N". Promoting
        every Decimal column to float64 avoids the precision negotiation
        entirely and matches what downstream ML libraries expect.
        """
        import pyarrow as pa
        new_fields = []
        needs_cast = False
        for field in table.schema:
            if pa.types.is_decimal(field.type):
                new_fields.append(pa.field(field.name, pa.float64(), field.nullable))
                needs_cast = True
            else:
                new_fields.append(field)
        if needs_cast:
            return table.cast(pa.schema(new_fields))
        return table

    def _build_combined_input(
        self,
        input_path: str,
        valid_path: str = None,
        oot_path: str = None,
        target_column: str = None,
        entity_id_col: str = None,
        composite_key_cols: list = None,
        train_sample_ratio: float = None,
        prefilter: bool = True,
    ) -> str:
        """Stream train + (optional) valid + (optional) oot to a single parquet
        file with a `_split_` marker column.

        Memory-efficient design for large datasets:
          - Train is loaded ONCE: scanned for useful columns, filtered,
            optionally stratified-sampled, then written to parquet and freed.
          - Valid / oot loaded one at a time, filtered to train's column set,
            then written and freed.
          - Peak memory ≈ size of the largest single partition (not the sum).
          - Output is parquet+snappy (typically 5–10× smaller than CSV) so
            Agent 1 reads it back quickly with low overhead.

        Agents 1+2 see the combined data with the marker preserved, so all
        partitions get identical cleaning + feature engineering. Agent 3
        reconstructs the exact user-defined splits from the marker.

        Args:
            train_sample_ratio: When 0 < ratio < 1, apply stratified sampling
                                to TRAIN only (valid/oot kept intact).
            prefilter:          When True, drop useless columns from train (high
                                null / constant / near-constant). Same filter
                                applied to valid/oot for schema alignment.

        Returns the combined file path: OUTPUT_DIR/combined_input.parquet
        """
        import gc
        import pyarrow as pa
        import pyarrow.parquet as pq

        combined_path = f"{Config.OUTPUT_DIR}/combined_input.parquet"
        sizes: dict = {}
        cols_keep: list = None
        writer: pq.ParquetWriter | None = None
        writer_schema: pa.Schema | None = None
        chunk_rows = max(1, Config.CONCAT_CHUNK_ROWS)

        # ── Pass 1: TRAIN — scan, filter (in-place), sample, chunk-write ──
        train_df = BaseAgent.load_dataframe(input_path)
        n_train_raw, n_cols_raw = train_df.shape
        self.logger.log("PIPELINE", "Pre-split load",
            f"train loaded rows={n_train_raw} cols={n_cols_raw}")

        if prefilter:
            cols_keep = self._scan_useful_columns(
                train_df,
                target_column=target_column,
                entity_id_col=entity_id_col,
                composite_key_cols=composite_key_cols,
            )
            n_dropped = n_cols_raw - len(cols_keep)
            self.logger.log("PIPELINE", "Pre-filter scan complete",
                f"will project {len(cols_keep)} kept cols at write time "
                f"({n_dropped} dropped — NO pandas drop to avoid OOM)")
            # ── CRITICAL DESIGN ────────────────────────────────────────
            # We do NOT drop columns from the pandas train_df here.
            # pandas `df.drop(columns=..., inplace=True)` still allocates
            # a full copy internally (BlockManager rebuild), peaking at
            # 2× memory which silently kills the kernel on very wide
            # datasets. Instead we PROJECT the kept columns at chunk
            # write time below — the chunk slice handles only what the
            # writer needs.
            if n_dropped > 100 and (train_sample_ratio is None or train_sample_ratio >= 0.5):
                self.logger.log("PIPELINE", "WARN",
                    f"{n_dropped} cols to drop but no aggressive sampling. "
                    f"If the kernel still dies during write, set "
                    f"train_sample_ratio=0.3 (or smaller) in pipeline.run().")
        else:
            cols_keep = list(train_df.columns)

        if train_sample_ratio is not None and 0 < train_sample_ratio < 1:
            # Two-step rebind frees the original 24 GB train_df immediately
            # after the sampled copy is materialised, instead of waiting for
            # natural GC. Critical when memory is tight.
            sampled_df = self._stratified_sample(train_df, target_column, train_sample_ratio)
            del train_df
            gc.collect()
            train_df = sampled_df
            del sampled_df
            gc.collect()
            self.logger.log("PIPELINE", "Pre-split sample",
                f"train sampled {n_train_raw} -> {len(train_df)} "
                f"(ratio={train_sample_ratio:.2f}, stratified by {target_column!r})")

        train_df[SPLIT_MARKER] = "train"
        n_train = len(train_df)
        sizes["train"] = n_train

        # Cols to materialise each chunk: kept cols + marker
        cols_to_write = list(cols_keep)
        if SPLIT_MARKER not in cols_to_write:
            cols_to_write.append(SPLIT_MARKER)
        # Filter to cols actually present (defensive)
        cols_to_write = [c for c in cols_to_write if c in train_df.columns]

        # Chunked write: peak RAM = chunk_rows × n_cols_kept × 8 bytes (per chunk),
        # NOT the full train_df + arrow table at once. The chunk slice projects
        # only kept cols so dropped cols never get converted to Arrow.
        n_chunks = (n_train + chunk_rows - 1) // chunk_rows
        self.logger.log("PIPELINE", "Pre-split write start",
            f"train rows={n_train} cols_to_write={len(cols_to_write)} "
            f"(of {train_df.shape[1]} in df) | chunk_rows={chunk_rows} | n_chunks={n_chunks}")
        for chunk_idx, start in enumerate(range(0, n_train, chunk_rows), 1):
            end = min(start + chunk_rows, n_train)
            # Row-slice + col-project in one expression. iloc returns a view
            # in pandas 2.x with copy-on-write; the [cols_to_write] selection
            # materialises only the kept columns (smaller copy).
            chunk_df = train_df.iloc[start:end][cols_to_write]
            chunk_table = pa.Table.from_pandas(chunk_df, preserve_index=False)
            del chunk_df
            # Normalize Decimal -> float64 (so valid/oot precision differences
            # don't break the writer schema later)
            chunk_table = self._normalize_numeric_types(chunk_table)
            if writer is None:
                writer_schema = chunk_table.schema
                writer = pq.ParquetWriter(combined_path, writer_schema,
                                          compression="snappy")
            else:
                # All train chunks share schema with the first chunk
                chunk_table = chunk_table.cast(writer_schema)
            writer.write_table(chunk_table)
            del chunk_table
            gc.collect()
            if chunk_idx == 1 or chunk_idx == n_chunks or chunk_idx % 5 == 0:
                self.logger.log("PIPELINE", "Pre-split write progress",
                    f"train chunk {chunk_idx}/{n_chunks} done ({end} rows)")

        del train_df
        gc.collect()
        self.logger.log("PIPELINE", "Pre-split write done", f"train fully written ({n_train} rows)")

        # ── Pass 2/3: VALID and OOT — same chunked write, filter only ─────
        for path, marker in [(valid_path, "valid"), (oot_path, "oot")]:
            if path is None:
                continue

            df = BaseAgent.load_dataframe(path)
            self.logger.log("PIPELINE", "Pre-split load",
                f"{marker} loaded rows={len(df)} cols={df.shape[1]}")

            # Same memory-aware approach: do NOT drop cols from pandas; project
            # at chunk write time instead.
            df[SPLIT_MARKER] = marker
            n_part = len(df)
            sizes[marker] = n_part

            cols_to_write_p = list(cols_keep) if cols_keep is not None else list(df.columns)
            if SPLIT_MARKER not in cols_to_write_p:
                cols_to_write_p.append(SPLIT_MARKER)
            cols_to_write_p = [c for c in cols_to_write_p if c in df.columns]

            n_chunks_p = (n_part + chunk_rows - 1) // chunk_rows
            self.logger.log("PIPELINE", "Pre-split write start",
                f"{marker} rows={n_part} cols_to_write={len(cols_to_write_p)} | n_chunks={n_chunks_p}")
            for chunk_idx, start in enumerate(range(0, n_part, chunk_rows), 1):
                end = min(start + chunk_rows, n_part)
                chunk_df = df.iloc[start:end][cols_to_write_p]
                chunk_table = pa.Table.from_pandas(chunk_df, preserve_index=False)
                del chunk_df
                chunk_table = self._normalize_numeric_types(chunk_table)
                try:
                    chunk_table = chunk_table.cast(writer_schema)
                except Exception as e:
                    writer.close()
                    raise RuntimeError(
                        f"Schema mismatch in '{marker}' partition (chunk {chunk_idx}) "
                        f"cannot be reconciled with train schema: {e}. Re-export "
                        f"inputs so train, valid, and oot share identical column "
                        f"types and order."
                    ) from e
                writer.write_table(chunk_table)
                del chunk_table
                gc.collect()
                if chunk_idx == 1 or chunk_idx == n_chunks_p or chunk_idx % 5 == 0:
                    self.logger.log("PIPELINE", "Pre-split write progress",
                        f"{marker} chunk {chunk_idx}/{n_chunks_p} done ({end} rows)")

            del df
            gc.collect()
            self.logger.log("PIPELINE", "Pre-split write done",
                f"{marker} fully written ({n_part} rows)")

        if writer is not None:
            writer.close()

        self.logger.log("PIPELINE", "Pre-split concat (streaming parquet)",
            f"Combined sizes={sizes} | total={sum(sizes.values())} | saved={combined_path}")
        return combined_path

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
            input_path: Path to the input dataset (always interpreted as
                        TRAIN when valid_path or oot_path is also provided).
                        Any format supported by BaseAgent.load_dataframe.
            valid_path: Optional path to a pre-split VALID dataset.
            oot_path:   Optional path to a pre-split OOT dataset.

            Pre-split mode is enabled whenever valid_path OR oot_path is set.
            In that mode, the three (or two) files are concatenated with a
            `_split_` marker column so Agents 1+2 apply identical cleaning +
            feature engineering to every partition. Agent 3 then reconstructs
            the splits from the marker:
              - train+valid+oot supplied → use all three exactly as marked.
              - train+valid only        → no OOT; use train/valid as marked.
              - train+oot only          → train auto-splits 20% into valid;
                                          OOT used as marked.
              - train only              → 60/20/20 auto-split (no marker).

            domain:     Feature engineering domain context. Options:
                        credit_risk, propensity, fraud, generic (default).
            model_type: ML problem type passed to Agent 2 prompt.
                        Options: binary_classification (default), regression, multiclass.
        """
        pre_split = (valid_path is not None) or (oot_path is not None)
        mode_label = "auto-split"
        if pre_split:
            parts_label = ["train"]
            if valid_path: parts_label.append("valid")
            if oot_path: parts_label.append("oot")
            mode_label = f"pre-split ({'+'.join(parts_label)})"
        self.logger.log("PIPELINE", "Starting",
            f"Input: {input_path} | Target: {target_column} | Domain: {domain} | "
            f"Model: {model_type} | Mode: {mode_label}")

        # Pre-split mode: concat train + (valid) + (oot) with marker → use as input.
        # This makes Agents 1+2 process the SAME schema for every partition, so
        # column drops / engineered features from Agent 2 apply consistently.
        if pre_split:
            input_path = self._build_combined_input(
                input_path,
                valid_path=valid_path,
                oot_path=oot_path,
                target_column=target_column,
                entity_id_col=entity_id_col,
                composite_key_cols=composite_key_cols,
                train_sample_ratio=train_sample_ratio,
                prefilter=prefilter,
            )

        self.logger.log("PIPELINE", "Stage 1", "Initializing Data Cleaner Agent")
        agent1 = DataCleanerAgent(
            self.logger,
            entity_id_col=entity_id_col,
            composite_key_cols=composite_key_cols,
            target_column=target_column,
        )
        clean_data_path, report1 = agent1.process(input_path)
        self.handoff.set_data(clean_data_path, report1, "DataCleaner")

        self.logger.log("PIPELINE", "Stage 2", "Initializing Feature Engineer Agent")
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

        # OOT routing: when pre_split is on, OOT is already in the combined input
        # via the marker (went through Agents 1+2), so we don't pass it separately
        # to Agent 3 here. The auto-split branch (pre_split=False) means no OOT at all.
        oot_df = None

        self.logger.log("PIPELINE", "Stage 3", "Initializing Train Model Agent")
        agent3 = TrainModelAgent(self.logger)
        final_metrics, report3 = agent3.process(
            self.handoff.get_data(),
            self.handoff.get_report(),
            target_column,
            oot_df=oot_df,
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
