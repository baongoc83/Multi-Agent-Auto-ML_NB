import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import json
from base_agent import BaseAgent, ToolRegistry
from logger import AgentLogger
from config import Config


class DataCleanerAgent(BaseAgent):

    def __init__(self, logger: AgentLogger):
        super().__init__(name="DataCleaner", role="Data Quality Auditor", logger=logger)
        self.df: pd.DataFrame = None
        self.tool_registry = ToolRegistry()
        self._entity_id_col: Optional[str] = None
        self._composite_key_cols: List[str] = []
        self._register_tools()

    def _register_tools(self):
        self.tool_registry.register(
            "inspect_metadata",
            "Returns dataset shape, column data types, null counts, and duplicate stats",
            {"df": "The dataframe to inspect"},
        )
        self.tool_registry.register(
            "get_column_stats",
            "Returns distribution statistics or unique values for a column",
            {"df": "The dataframe", "col": "Column name to analyze"},
        )
        self.tool_registry.register(
            "impute_missing",
            "Fills missing values in a column using specified strategy",
            {
                "df": "The dataframe",
                "col": "Column name",
                "strategy": "One of: 'mean', 'median', 'mode', 'zero', 'forward_fill'",
            },
        )
        self.tool_registry.register(
            "drop_column",
            "Removes a column from the dataset",
            {"df": "The dataframe", "col": "Column name to drop"},
        )
        self.tool_registry.register(
            "detect_outliers",
            "Detects outliers in a numeric column using IQR 3x method",
            {"df": "The dataframe", "col": "Numeric column name to check"},
        )
        self.tool_registry.register(
            "check_label_quality",
            "Analyzes label distribution, class balance, and null labels",
            {"df": "The dataframe", "label_col": "Name of the label/target column"},
        )
        self.tool_registry.register(
            "check_temporal",
            "Checks temporal integrity: date range, future dates, sort order",
            {"df": "The dataframe", "date_col": "Name of the date/timestamp column"},
        )
        self.tool_registry.register(
            "drop_duplicates",
            "Removes duplicate rows from the dataset",
            {"df": "The dataframe"},
        )
        self.tool_registry.register(
            "clip_outliers",
            "Clips outlier values in a numeric column to IQR 3x bounds",
            {"df": "The dataframe", "col": "Column name", "factor": "IQR multiplier, default 3.0"},
        )
        self.tool_registry.register(
            "check_pk_uniqueness",
            "Checks PK/uniqueness: exact/soft/app duplicates, composite key violations, entity resolution",
            {
                "df": "The dataframe",
                "entity_id_col": "Primary entity identifier column (e.g. customer_id)",
                "composite_key_cols": "List of columns that together must be unique",
                "label_col": "Optional target column for label contamination check",
                "identity_cols": "Optional dict {label: col_name} for entity resolution (cif, phone, device, etc.)",
                "app_dup_max_allowed": "Max rows per entity before flagging (default 1)",
            },
        )
        self.tool_registry.register(
            "deduplicate_by_key",
            "Removes soft duplicate rows keeping first/last occurrence per composite key",
            {
                "df": "The dataframe",
                "key_cols": "List of columns forming the composite key",
                "keep": "Which duplicate to keep: 'first' or 'last' (default 'first')",
            },
        )

    def _tool_inspect_metadata(self, df: pd.DataFrame) -> str:
        duplicate_count = int(df.duplicated().sum())
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "null_percentages": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "duplicate_rows": duplicate_count,
            "duplicate_percentage": round(duplicate_count / len(df) * 100, 2),
        }
        return json.dumps(info, indent=2)

    def _tool_get_column_stats(self, df: pd.DataFrame, col: str) -> str:
        if col not in df.columns:
            return f"Error: Column '{col}' not found in dataset"

        col_data = df[col]
        stats = {
            "column": col,
            "dtype": str(col_data.dtype),
            "non_null_count": int(col_data.count()),
            "null_count": int(col_data.isnull().sum()),
            "unique_count": int(col_data.nunique()),
        }

        if pd.api.types.is_numeric_dtype(col_data):
            stats.update({
                "mean": float(col_data.mean()) if not col_data.empty else None,
                "median": float(col_data.median()) if not col_data.empty else None,
                "std": float(col_data.std()) if not col_data.empty else None,
                "min": float(col_data.min()) if not col_data.empty else None,
                "max": float(col_data.max()) if not col_data.empty else None,
            })

        if col_data.nunique() < Config.HIGH_CARDINALITY_THRESHOLD:
            stats["value_counts"] = col_data.value_counts().head(10).to_dict()
        else:
            stats["note"] = f"High cardinality: {col_data.nunique()} unique values"

        return json.dumps(stats, indent=2)

    def _tool_impute_missing(self, df: pd.DataFrame, col: str, strategy: str) -> pd.DataFrame:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")

        if strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            mode_value = df[col].mode()[0] if not df[col].mode().empty else 0
            df[col] = df[col].fillna(mode_value)
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
        elif strategy == "forward_fill":
            df[col] = df[col].ffill()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return df

    def _tool_drop_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")
        df.drop(columns=[col], inplace=True)
        return df

    def _tool_detect_outliers(self, df: pd.DataFrame, col: str) -> str:
        if col not in df.columns:
            return json.dumps({"error": f"Column '{col}' not found"})
        if not pd.api.types.is_numeric_dtype(df[col]):
            return json.dumps({"column": col, "note": "Not numeric, skipped"})
        series = df[col].dropna()
        q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        outlier_count = int(((df[col] < lower) | (df[col] > upper)).sum())
        return json.dumps({
            "column": col,
            "q1": q1, "q3": q3, "iqr": round(iqr, 4),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "outlier_count": outlier_count,
            "outlier_percentage": round(outlier_count / len(df) * 100, 2),
        }, indent=2)

    def _tool_check_label_quality(self, df: pd.DataFrame, label_col: str) -> str:
        if label_col not in df.columns:
            return json.dumps({"error": f"Column '{label_col}' not found"})
        total = len(df)
        null_labels = int(df[label_col].isnull().sum())
        counts = {str(k): int(v) for k, v in df[label_col].value_counts().items()}
        values = list(counts.values())
        imbalance_ratio = round(max(values) / min(values), 2) if len(values) > 1 else 1.0
        return json.dumps({
            "label_col": label_col,
            "total_rows": total,
            "num_classes": len(counts),
            "class_distribution": counts,
            "null_labels": null_labels,
            "null_label_percentage": round(null_labels / total * 100, 2),
            "imbalance_ratio": imbalance_ratio,
            "note": "Imbalance ratio = max_class_count / min_class_count. "
                    f">{Config.IMBALANCE_RATIO_THRESHOLD} is typically problematic.",
        }, indent=2)

    def _tool_check_temporal(self, df: pd.DataFrame, date_col: str) -> str:
        if date_col not in df.columns:
            return json.dumps({"error": f"Column '{date_col}' not found"})
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        null_dates = int(parsed.isnull().sum())
        valid = parsed.dropna()
        future_count = int((valid > pd.Timestamp.now()).sum())
        sorted_check = valid.tolist()
        is_sorted = sorted_check == sorted(sorted_check)
        date_range_days = int((valid.max() - valid.min()).days) if len(valid) > 1 else None
        leakage_suspects = [
            c for c in df.columns
            if any(kw in c.lower() for kw in ("future", "next", "forward", "ahead", "leakage"))
        ]
        return json.dumps({
            "date_col": date_col,
            "min_date": str(valid.min()) if len(valid) > 0 else None,
            "max_date": str(valid.max()) if len(valid) > 0 else None,
            "date_range_days": date_range_days,
            "distinct_dates": int(valid.nunique()),
            "null_dates": null_dates,
            "rows_with_future_dates": future_count,
            "is_sorted_by_date": is_sorted,
            "potential_leakage_columns": leakage_suspects,
        }, indent=2)

    def _tool_drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        self.logger.log(self.name, "drop_duplicates", f"Removed {before - len(df)} duplicate rows")
        return df

    def _tool_clip_outliers(self, df: pd.DataFrame, col: str, factor: float = 3.0) -> pd.DataFrame:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' is not numeric")
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - factor * iqr, upper=q3 + factor * iqr)
        return df

    def _tool_check_pk_uniqueness(
        self,
        df: pd.DataFrame,
        entity_id_col: str,
        composite_key_cols: List[str],
        label_col: Optional[str] = None,
        identity_cols: Optional[Dict[str, str]] = None,
        app_dup_max_allowed: int = 1,
    ) -> str:
        total = len(df)

        # ── 2.1a Exact duplicates ─────────────────────────────
        exact_dup_rows = int(df.duplicated().sum())
        dup_stats: Dict[str, Any] = {
            "2_1a_exact_duplicates": {
                "exact_duplicate_rows": exact_dup_rows,
                "exact_duplicate_pct": round(exact_dup_rows / total * 100, 2),
                "unique_rows_after_dedup": total - exact_dup_rows,
            }
        }

        # ── 2.1b Soft duplicates ──────────────────────────────
        valid_ck = [c for c in composite_key_cols if c in df.columns]
        if valid_ck:
            key_counts = df.groupby(valid_ck).size()
            soft_dup_keys = int((key_counts > 1).sum())
            soft_dup_rows = int(key_counts[key_counts > 1].sum())
            top5 = (
                key_counts[key_counts > 1]
                .nlargest(5)
                .reset_index()
                .rename(columns={0: "count"})
                .to_dict(orient="records")
            )
            dup_stats["2_1b_soft_duplicates"] = {
                "soft_dup_subset_cols": valid_ck,
                "soft_duplicate_key_combos": soft_dup_keys,
                "soft_duplicate_affected_rows": soft_dup_rows,
                "soft_duplicate_affected_pct": round(soft_dup_rows / total * 100, 2),
                "top5_offending_keys": top5,
            }
        else:
            dup_stats["2_1b_soft_duplicates"] = {"note": "skipped — no valid composite key cols"}

        # ── 2.1c Duplicated application ──────────────────────
        if entity_id_col in df.columns:
            entity_counts = df.groupby(entity_id_col).size()
            total_entities = len(entity_counts)
            dup_ent = entity_counts[entity_counts > app_dup_max_allowed]
            dup_entity_count = len(dup_ent)
            dist: Dict[str, int] = {}
            if dup_entity_count > 0:
                dist = {
                    "2x": int((dup_ent == 2).sum()),
                    "3-5x": int(dup_ent.between(3, 5).sum()),
                    "6-10x": int(dup_ent.between(6, 10).sum()),
                    ">10x": int((dup_ent > 10).sum()),
                }
            dup_stats["2_1c_duplicated_application"] = {
                "total_distinct_entities": total_entities,
                "duplicated_application_entities": dup_entity_count,
                "duplicated_application_pct": round(dup_entity_count / total_entities * 100, 2) if total_entities else 0,
                "app_dup_max_allowed": app_dup_max_allowed,
                "row_count_distribution": dist,
            }
        else:
            dup_stats["2_1c_duplicated_application"] = {"note": f"skipped — '{entity_id_col}' not found"}

        # ── 2.2 Composite key validation ─────────────────────
        if valid_ck:
            sizes = df.groupby(valid_ck).size().reset_index(name="row_count")
            violations = sizes[sizes["row_count"] > 1]
            viol_count = len(violations)

            if viol_count == 0:
                ck_stats: Dict[str, Any] = {
                    "composite_key": valid_ck,
                    "violation_count": 0,
                    "verdict_note": "Composite key is UNIQUE — no violations found.",
                }
            else:
                affected_rows = int(violations["row_count"].sum())
                label_contamination = 0
                if label_col and label_col in df.columns:
                    label_div = df.groupby(valid_ck)[label_col].nunique()
                    label_contamination = int((label_div > 1).sum())
                top5_viol = violations.nlargest(5, "row_count").to_dict(orient="records")
                ck_stats = {
                    "composite_key": valid_ck,
                    "violation_count": viol_count,
                    "affected_rows": affected_rows,
                    "affected_rows_pct": round(affected_rows / total * 100, 2),
                    "max_rows_per_key": int(violations["row_count"].max()),
                    "avg_rows_per_key": round(float(violations["row_count"].mean()), 2),
                    "label_contamination_count": label_contamination,
                    "label_contamination_note": (
                        "label_contamination_count > 0 means same (entity, date) has conflicting "
                        "labels → direct training signal corruption."
                    ),
                    "double_count_risk_pct": round(affected_rows / total * 100, 2),
                    "top5_violation_samples": top5_viol,
                }
        else:
            ck_stats = {"note": "skipped — no valid composite key cols"}

        # ── 2.3 Entity resolution ─────────────────────────────
        if identity_cols and entity_id_col in df.columns:
            valid_identity = {k: v for k, v in identity_cols.items() if v in df.columns}
            if valid_identity:
                total_entities = df[entity_id_col].nunique()
                col_stats: Dict[str, Any] = {}
                for label, col in valid_identity.items():
                    null_pct = round(df[col].isnull().mean() * 100, 2)
                    total_distinct = int(df[col].nunique())
                    non_null = df[df[col].notna()]

                    fwd = non_null.groupby(entity_id_col)[col].nunique()
                    multi_id = int((fwd > 1).sum())
                    fwd_dist = {str(k): int(v) for k, v in fwd.value_counts().sort_index().head(6).to_dict().items()}

                    rev = non_null.groupby(col)[entity_id_col].nunique()
                    ambiguous = int((rev > 1).sum())

                    col_stats[label] = {
                        "null_pct": null_pct,
                        "total_distinct_values": total_distinct,
                        "customers_with_multiple_values": multi_id,
                        "multi_value_pct_of_customers": round(multi_id / total_entities * 100, 2) if total_entities else 0,
                        "id_count_distribution": fwd_dist,
                        "ambiguous_values_multi_customer": ambiguous,
                        "ambiguous_pct_of_distinct_values": round(ambiguous / total_distinct * 100, 2) if total_distinct else 0,
                        "graph_integrity_note": (
                            "ambiguous > 0 means same identifier links to multiple customers. "
                            "For phone/device this can be fraud signal or data error."
                        ),
                    }

                all_null = df[[c for c in valid_identity.values()]].isnull().all(axis=1)
                orphan_entities = int(df[all_null][entity_id_col].nunique())
                er_stats: Dict[str, Any] = {
                    "total_distinct_entities": total_entities,
                    "identity_column_stats": col_stats,
                    "orphan_entities": orphan_entities,
                    "orphan_pct": round(orphan_entities / total_entities * 100, 2) if total_entities else 0,
                    "orphan_note": (
                        "Orphan = customer with NULL in ALL identity columns. "
                        "Cannot be linked across systems; high fraud risk if orphan rate > 5%."
                    ),
                }
            else:
                er_stats = {"note": f"skipped — none of {list(identity_cols.values())} found in DataFrame"}
        else:
            er_stats = {"entity_resolution_check": "skipped — no identity_cols configured"}

        return json.dumps({
            "domain_context": (
                "Banking / fraud ML dataset. Identity integrity is critical: "
                "composite key violations → double-count labels; "
                "shared device/phone → fraud ring leakage."
            ),
            "total_rows": total,
            "check_2_1_duplicates": dup_stats,
            "check_2_2_composite_key": ck_stats,
            "check_2_3_entity_resolution": er_stats,
        }, indent=2, default=str)

    def _tool_deduplicate_by_key(
        self,
        df: pd.DataFrame,
        key_cols: List[str],
        keep: str = "first",
    ) -> pd.DataFrame:
        valid_cols = [c for c in key_cols if c in df.columns]
        if not valid_cols:
            raise ValueError(f"None of the key columns {key_cols} found in dataframe")
        before = len(df)
        df = df.drop_duplicates(subset=valid_cols, keep=keep).reset_index(drop=True)
        self.logger.log(self.name, "deduplicate_by_key",
                        f"Removed {before - len(df)} soft duplicate rows by {valid_cols} (keep={keep})")
        return df

    def process(self, input_csv: str) -> Tuple[str, Dict[str, Any]]:
        """Returns (path_to_clean_data, report_dict)."""
        self.logger.log(self.name, "Process Start", f"Loading data from {input_csv}")

        self.df = pd.read_csv(input_csv)
        original_shape = self.df.shape

        metadata = self.execute_tool("inspect_metadata", df=self.df)
        outlier_stats = self._collect_outlier_stats()
        label_stats = self._collect_label_stats()
        temporal_stats = self._collect_temporal_stats()
        pk_stats = self._collect_pk_stats()

        llm_response = self.call_llm(
            self._build_analysis_prompt(metadata, outlier_stats, label_stats, temporal_stats, pk_stats),
            self._get_system_prompt(),
        )
        actions_taken = self._execute_llm_decisions(llm_response)

        Path(Config.CLEAN_DATA_PATH).parent.mkdir(exist_ok=True)
        self.df.to_csv(Config.CLEAN_DATA_PATH, index=False)
        self.logger.log(self.name, "Data Saved", f"Cleaned data saved to {Config.CLEAN_DATA_PATH}")

        report = {
            "agent": self.name,
            "original_shape": original_shape,
            "final_shape": self.df.shape,
            "actions_taken": actions_taken,
            "summary": self._generate_summary(actions_taken),
            "columns_remaining": list(self.df.columns),
            "entity_id_col": self._entity_id_col,
            "composite_key_cols": self._composite_key_cols,
        }
        self.save_report(report, Config.DATA_CLEANER_REPORT_PATH)

        self.logger.log(self.name, "Process Complete", f"Shape: {original_shape} -> {self.df.shape}")
        return Config.CLEAN_DATA_PATH, report

    def _collect_outlier_stats(self) -> str:
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        results = {}
        for col in numeric_cols[:Config.OUTLIER_NUMERIC_COLS_LIMIT]:
            results[col] = json.loads(self.execute_tool("detect_outliers", df=self.df, col=col))
        return json.dumps(results, indent=2)

    def _collect_label_stats(self) -> Optional[str]:
        candidates = [c for c in self.df.columns
                      if c.lower() in ("target", "label", "y", "class", "output")]
        if not candidates:
            candidates = [self.df.columns[-1]]
        return self.execute_tool("check_label_quality", df=self.df, label_col=candidates[0])

    def _collect_temporal_stats(self) -> Optional[str]:
        date_cols = [c for c in self.df.columns
                     if any(kw in c.lower() for kw in ("date", "time", "timestamp", "dt"))]
        if not date_cols:
            return None
        return self.execute_tool("check_temporal", df=self.df, date_col=date_cols[0])

    def _collect_pk_stats(self) -> Optional[str]:
        # Auto-detect entity ID column
        entity_id_col = next(
            (c for c in self.df.columns
             if c.lower() in ("customer_id", "entity_id", "user_id", "client_id", "account_id")),
            next((c for c in self.df.columns if c.lower().endswith("_id")), None),
        )
        if entity_id_col is None:
            return None

        # Build composite key: entity_id + first date col (if any)
        date_cols = [c for c in self.df.columns
                     if any(kw in c.lower() for kw in ("date", "time", "timestamp", "dt"))]
        composite_key_cols = [entity_id_col] + date_cols[:1]

        # Store for report so downstream agents can skip these cols
        self._entity_id_col = entity_id_col
        self._composite_key_cols = composite_key_cols

        # Auto-detect label col
        label_candidates = [c for c in self.df.columns
                            if c.lower() in ("target", "label", "y", "class", "output", "default_flag", "fraud")]
        label_col = label_candidates[0] if label_candidates else None

        # Auto-detect identity cols (cif, phone, device, national_id)
        _identity_map = {
            "cif":         ("cif", "cif_id"),
            "phone":       ("phone", "phone_number", "mobile"),
            "device":      ("device", "device_id"),
            "national_id": ("national_id", "nid", "id_number"),
        }
        identity_cols = {
            label: col
            for label, kws in _identity_map.items()
            for col in self.df.columns
            if col.lower() in kws
        }

        return self.execute_tool(
            "check_pk_uniqueness",
            df=self.df,
            entity_id_col=entity_id_col,
            composite_key_cols=composite_key_cols,
            label_col=label_col,
            identity_cols=identity_cols if identity_cols else None,
        )
# - For numeric columns with missing values, prefer median imputation
# - For categorical columns with missing values, prefer mode imputation
    def _get_system_prompt(self) -> str:
        return f"""You are the Data Cleaner Agent, an expert data quality auditor.

Your role: Inspect the raw dataset and make decisions about cleaning actions.

Available Tools:
{self.tool_registry.get_tool_descriptions()}

Your task:
1. Analyze all statistics provided: metadata, outliers, label quality, temporal integrity
2. Decide which columns need cleaning or should be dropped
3. For each issue found, decide the best action
4. Output your decisions in a structured format

Guidelines:
- Drop columns with >{Config.NULL_DROP_THRESHOLD * 100:.0f}% missing values (unless they seem important)
- Drop ID columns or columns with all unique values (no predictive power)
- Drop duplicate rows if exact_duplicate_pct > {Config.DUPLICATE_PCT_THRESHOLD:.1f}%
- Clip outliers in a numeric column if outlier_percentage > {Config.OUTLIER_PCT_THRESHOLD:.1f}%
- Flag (but do not drop) label column if imbalance_ratio > {Config.IMBALANCE_RATIO_THRESHOLD:.0f}
- Flag future dates or potential leakage columns found in temporal integrity check

PK & Uniqueness (Banking / Fraud, Credit, Propensity domains — apply stricter thresholds):
- Any label_contamination_count > 0                      → CRITICAL: deduplicate_by_key (keep='first')
- Composite key violation > {Config.PK_VIOLATION_PCT_THRESHOLD:.1f}%                     → deduplicate_by_key
- Ambiguous identity values > {Config.AMBIGUOUS_IDENTITY_PCT_THRESHOLD:.1f}% per identity column → flag as potential fraud ring (do not drop)
- Orphan customers > {Config.ENTITY_ORPHAN_PCT_THRESHOLD:.1f}%                          → flag for investigation (do not drop)
- Soft duplicates > 0.5%                                 → deduplicate_by_key
- Duplicated application entities > 10%                  → deduplicate_by_key

Output Format (JSON):
{{
  "reasoning": "Your analysis of all data quality issues",
  "actions": [
    {{"action": "drop_column", "column": "id", "reason": "Unique identifier with no predictive value"}},
    {{"action": "impute_missing", "column": "age", "strategy": "median", "reason": "20% missing numeric values"}},
    {{"action": "impute_missing", "column": "category", "strategy": "mode", "reason": "15% missing categorical values"}},
    {{"action": "drop_duplicates", "reason": "3.5% exact duplicate rows detected"}},
    {{"action": "clip_outliers", "column": "income", "factor": 3.0, "reason": "8% outliers via IQR"}},
    {{"action": "deduplicate_by_key", "columns": ["customer_id", "snapshot_date"], "keep": "first", "reason": "2% composite key violations with label contamination"}}
  ]
}}

Be decisive but explain your reasoning clearly."""

    def _build_analysis_prompt(
        self,
        metadata: str,
        outlier_stats: str,
        label_stats: Optional[str],
        temporal_stats: Optional[str],
        pk_stats: Optional[str],
    ) -> str:
        sections = [
            f"DATASET METADATA:\n{metadata}",
            f"OUTLIER ANALYSIS (IQR 3x per numeric column):\n{outlier_stats}",
        ]
        if label_stats:
            sections.append(f"LABEL QUALITY:\n{label_stats}")
        if temporal_stats:
            sections.append(f"TEMPORAL INTEGRITY:\n{temporal_stats}")
        if pk_stats:
            sections.append(f"PK & UNIQUENESS CHECK (exact/soft/app duplicates, composite key, entity resolution):\n{pk_stats}")
        sections.append(
            "Based on all statistics above, what cleaning actions should be performed?\n"
            "Provide your response in the JSON format specified."
        )
        return "\n\n".join(sections)

    def _execute_llm_decisions(self, llm_response: str) -> List[str]:
        self.logger.log(self.name, "LLM Decision", "Parsing decisions from LLM response")
        actions_taken = []

        try:
            response_text = llm_response.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            decisions = json.loads(response_text)
            self.logger.log(self.name, "LLM Reasoning", decisions.get("reasoning", "No reasoning provided"))

            for action_spec in decisions.get("actions", []):
                action_type = action_spec.get("action")
                column = action_spec.get("column")
                reason = action_spec.get("reason", "No reason provided")

                self.logger.log(self.name, f"Action: {action_type}", f"Column: {column}, Reason: {reason}")

                if action_type == "drop_column":
                    self.df = self.execute_tool("drop_column", df=self.df, col=column)
                    actions_taken.append(f"Dropped column '{column}': {reason}")

                elif action_type == "impute_missing":
                    strategy = action_spec.get("strategy", "median")
                    self.df = self.execute_tool("impute_missing", df=self.df, col=column, strategy=strategy)
                    actions_taken.append(f"Imputed '{column}' with {strategy}: {reason}")

                elif action_type == "drop_duplicates":
                    self.df = self.execute_tool("drop_duplicates", df=self.df)
                    actions_taken.append(f"Dropped duplicate rows: {reason}")

                elif action_type == "clip_outliers":
                    factor = float(action_spec.get("factor", 3.0))
                    self.df = self.execute_tool("clip_outliers", df=self.df, col=column, factor=factor)
                    actions_taken.append(f"Clipped outliers in '{column}' (factor={factor}): {reason}")

                elif action_type == "deduplicate_by_key":
                    key_cols = action_spec.get("columns", [])
                    keep = action_spec.get("keep", "first")
                    self.df = self.execute_tool("deduplicate_by_key", df=self.df, key_cols=key_cols, keep=keep)
                    actions_taken.append(f"Deduplicated by key {key_cols} (keep={keep}): {reason}")

        except json.JSONDecodeError as e:
            self.logger.log(self.name, "ERROR", f"Failed to parse LLM response as JSON: {e}")
            self.logger.log(self.name, "Raw Response", llm_response[:500])
            actions_taken.append("ERROR: Could not parse LLM decisions, performed basic cleaning")
            self._fallback_cleaning()

        return actions_taken

    def _fallback_cleaning(self):
        for col in list(self.df.columns):
            if self.df[col].isnull().sum() / len(self.df) > Config.NULL_DROP_THRESHOLD:
                self.df.drop(columns=[col], inplace=True)
                self.logger.log(self.name, "Fallback", f"Dropped {col} (>{Config.NULL_DROP_THRESHOLD * 100:.0f}% missing)")

    def _generate_summary(self, actions: List[str]) -> str:
        if not actions:
            return "No cleaning actions were necessary. Data quality is good."
        preview = actions[:Config.SUMMARY_ACTION_PREVIEW]
        summary = f"Performed {len(actions)} cleaning actions: " + "; ".join(preview)
        if len(actions) > Config.SUMMARY_ACTION_PREVIEW:
            summary += f"; and {len(actions) - Config.SUMMARY_ACTION_PREVIEW} more actions."
        return summary
