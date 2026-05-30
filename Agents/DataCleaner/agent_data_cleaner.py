import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import json
from Agents.BaseAgent.base_agent import BaseAgent, ToolRegistry
from logger import AgentLogger
from config import Config


class DataCleanerAgent(BaseAgent):

    def __init__(
        self,
        logger: AgentLogger,
        entity_id_col: Optional[str] = None,
        composite_key_cols: Optional[List[str]] = None,
        target_column: Optional[str] = None,
    ):
        super().__init__(name="DataCleaner", role="Data Quality Auditor", logger=logger)
        self.df: pd.DataFrame = None
        self.tool_registry = ToolRegistry()
        # Pre-supplied hints — override auto-detection when set
        self._entity_id_col: Optional[str] = entity_id_col
        self._composite_key_cols: List[str] = list(composite_key_cols) if composite_key_cols else []
        self._target_column: Optional[str] = target_column
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
        # self.tool_registry.register(
        #     "impute_missing",
        #     "Fills missing values in a column using specified strategy",
        #     {
        #         "df": "The dataframe",
        #         "col": "Column name",
        #         "strategy": "One of: 'mean', 'median', 'mode', 'zero', 'forward_fill'",
        #     },
        # )
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
        self.tool_registry.register(
            "check_column_formats",
            "Detects format issues: numeric stored as object, date stored as string, whitespace padding, mixed case, unexpected negatives",
            {"df": "The dataframe"},
        )
        self.tool_registry.register(
            "fix_column_dtype",
            "Fixes a column's format: cast to numeric/datetime, strip whitespace, or standardize case",
            {
                "df": "The dataframe",
                "col": "Column name to fix",
                "fix_type": "One of: 'cast_to_numeric', 'cast_to_datetime', 'strip_whitespace', 'standardize_case'",
            },
        )

    def _tool_inspect_metadata(self, df: pd.DataFrame) -> str:
        duplicate_count = int(df.duplicated().sum())
        # Constant columns: single unique non-null value (zero variance / no predictive power)
        constant_cols = [
            col for col in df.columns
            if df[col].dropna().nunique() == 1
        ]
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "null_percentages": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "duplicate_rows": duplicate_count,
            "duplicate_percentage": round(duplicate_count / len(df) * 100, 2),
            "constant_columns": constant_cols,
            "constant_columns_count": len(constant_cols),
        }
        return json.dumps(info, indent=2)

    def _tool_get_column_stats(self, df: pd.DataFrame, col: str) -> str:
        if col not in df.columns:
            return f"Error: Column '{col}' not found in dataset"

        col_data = df[col]
        non_null = col_data.dropna()
        stats = {
            "column": col,
            "dtype": str(col_data.dtype),
            "non_null_count": int(col_data.count()),
            "null_count": int(col_data.isnull().sum()),
            "unique_count": int(col_data.nunique()),
        }

        if pd.api.types.is_numeric_dtype(col_data) and len(non_null) > 0:
            stats.update({
                "mean": float(non_null.mean()),
                "median": float(non_null.median()),
                "std": float(non_null.std()),
                "min": float(non_null.min()),
                "max": float(non_null.max()),
            })

        if col_data.nunique() < Config.HIGH_CARDINALITY_THRESHOLD:
            # str(k) handles Timestamp/Period keys; int(v) ensures native int
            stats["value_counts"] = {str(k): int(v) for k, v in col_data.value_counts().head(10).items()}
        else:
            stats["note"] = f"High cardinality: {col_data.nunique()} unique values"

        return json.dumps(stats, indent=2)

    # def _tool_impute_missing(self, df: pd.DataFrame, col: str, strategy: str) -> pd.DataFrame:
    #     if col not in df.columns:
    #         raise ValueError(f"Column '{col}' not found")

    #     if strategy == "mean":
    #         df[col] = df[col].fillna(df[col].mean())
    #     elif strategy == "median":
    #         df[col] = df[col].fillna(df[col].median())
    #     elif strategy == "mode":
    #         mode_value = df[col].mode()[0] if not df[col].mode().empty else 0
    #         df[col] = df[col].fillna(mode_value)
    #     elif strategy == "zero":
    #         df[col] = df[col].fillna(0)
    #     elif strategy == "forward_fill":
    #         df[col] = df[col].ffill()
    #     else:
    #         raise ValueError(f"Unknown strategy: {strategy}")

    #     return df

    def _tool_drop_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")
        return df.drop(columns=[col])

    def _tool_detect_outliers(self, df: pd.DataFrame, col: str) -> str:
        if col not in df.columns:
            return json.dumps({"error": f"Column '{col}' not found"})
        if not pd.api.types.is_numeric_dtype(df[col]):
            return json.dumps({"column": col, "note": "Not numeric, skipped"})
        series = df[col].dropna()
        if series.empty:
            return json.dumps({"column": col, "note": "All values are null, skipped"})
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
        is_sorted = bool(valid.is_monotonic_increasing)
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

    _DATE_NAME_KEYWORDS = ("date", "dt", "time", "timestamp", "snap", "period", "month", "year", "week", "day")

    def _tool_check_column_formats(self, df: pd.DataFrame) -> str:
        # Columns excluded from format checks — PK, composite keys, and target.
        # Target is excluded because _execute_llm_decisions blocks fixes on it anyway;
        # reporting format issues for target would waste LLM tokens on unfixable items.
        protected: set = set(self._composite_key_cols)
        if self._entity_id_col:
            protected.add(self._entity_id_col)
        if self._target_column:
            protected.add(self._target_column)

        issues: Dict[str, list] = {}
        for col in df.columns:
            if col in protected:
                continue

            col_issues = []
            series = df[col]
            dtype = str(series.dtype)
            non_null = series.notna().sum()
            col_lower = col.lower()
            looks_like_date_col = any(kw in col_lower for kw in self._DATE_NAME_KEYWORDS)

            if dtype == "object" and non_null > 0:
                sample = series.dropna().head(100)

                # For date-named columns: check date parsability FIRST to avoid
                # YYYYMMDD-style strings being misclassified as numeric
                if looks_like_date_col:
                    date_parsed = pd.to_datetime(sample, errors="coerce", dayfirst=False).notna().sum()
                    if len(sample) > 0 and date_parsed / len(sample) > 0.9:
                        col_issues.append({
                            "issue": "date_stored_as_object",
                            "detail": f"{date_parsed}/{len(sample)} sampled values parseable as date",
                            "suggested_action": "cast_to_datetime",
                        })
                else:
                    # Numeric stored as object (e.g., "N/A", "?" masking numbers)
                    numeric_parsed = pd.to_numeric(series, errors="coerce").notna().sum()
                    if numeric_parsed / non_null > 0.9:
                        col_issues.append({
                            "issue": "numeric_stored_as_object",
                            "detail": f"{numeric_parsed}/{non_null} values parseable as numeric",
                            "suggested_action": "cast_to_numeric",
                        })
                    else:
                        # Date stored as object (non-date-named column)
                        date_parsed = pd.to_datetime(sample, errors="coerce", dayfirst=False).notna().sum()
                        if len(sample) > 0 and date_parsed / len(sample) > 0.9:
                            col_issues.append({
                                "issue": "date_stored_as_object",
                                "detail": f"{date_parsed}/{len(sample)} sampled values parseable as date",
                                "suggested_action": "cast_to_datetime",
                            })

                # Whitespace and case checks apply to all non-date object columns
                if not looks_like_date_col:
                    str_series = series.dropna().astype(str)
                    if str_series.str.contains(r"^\s|\s$").any():
                        col_issues.append({
                            "issue": "leading_trailing_whitespace",
                            "detail": "Some values have leading or trailing spaces",
                            "suggested_action": "strip_whitespace",
                        })
                    if series.nunique() < Config.HIGH_CARDINALITY_THRESHOLD:
                        lower_nunique = series.str.lower().nunique()
                        if lower_nunique < series.nunique():
                            col_issues.append({
                                "issue": "mixed_case_values",
                                "detail": (
                                    f"Case-folding reduces unique values "
                                    f"{series.nunique()} → {lower_nunique}"
                                ),
                                "suggested_action": "standardize_case",
                            })

            elif pd.api.types.is_numeric_dtype(series):
                if any(kw in col_lower for kw in ("age", "amount", "count", "qty", "quantity", "balance", "income", "salary", "price")):
                    neg_count = int((series < 0).sum())
                    if neg_count > 0:
                        col_issues.append({
                            "issue": "unexpected_negative_values",
                            "detail": f"{neg_count} negative values ({neg_count / len(df) * 100:.1f}%)",
                            "suggested_action": "clip_outliers or investigate",
                        })

            if col_issues:
                issues[col] = col_issues

        return json.dumps({
            "format_issues_found": len(issues),
            "columns_with_issues": issues,
        }, indent=2)

    def _tool_fix_column_dtype(self, df: pd.DataFrame, col: str, fix_type: str) -> pd.DataFrame:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")
        if fix_type == "cast_to_numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif fix_type == "cast_to_datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=False)
        elif fix_type == "strip_whitespace":
            if df[col].dtype != object:
                raise ValueError(f"strip_whitespace requires object dtype, got '{df[col].dtype}' for '{col}'")
            df[col] = df[col].str.strip()  # .str accessor preserves NaN natively
        elif fix_type == "standardize_case":
            if df[col].dtype != object:
                raise ValueError(f"standardize_case requires object dtype, got '{df[col].dtype}' for '{col}'")
            df[col] = df[col].str.lower()  # .str accessor preserves NaN natively
        else:
            raise ValueError(
                f"Unknown fix_type '{fix_type}'. "
                "Use: cast_to_numeric, cast_to_datetime, strip_whitespace, standardize_case"
            )
        return df

    def process(self, input_path: str) -> Tuple[str, Dict[str, Any]]:
        """Returns (path_to_clean_data, report_dict).

        Accepts any format supported by BaseAgent.load_dataframe:
        .csv, .tsv, .parquet, .orc, .feather, .xlsx, .xls, .xlsm, .json
        and remote paths (s3://, gs://, az://).
        """
        self.logger.log(self.name, "Process Start", f"Loading data from {input_path}")

        self.df = self.load_dataframe(input_path)
        original_shape = self.df.shape

        metadata = self.execute_tool("inspect_metadata", df=self.df)
        label_stats = self._collect_label_stats()    # resolves self._target_column first
        pk_stats = self._collect_pk_stats()          # uses self._target_column; sets self._entity_id_col / _composite_key_cols
        format_stats = self.execute_tool("check_column_formats", df=self.df)
        outlier_stats = self._collect_outlier_stats()
        temporal_stats = self._collect_temporal_stats()

        llm_response = self.call_llm(
            self._build_analysis_prompt(metadata, format_stats, outlier_stats, label_stats, temporal_stats, pk_stats),
            self._get_system_prompt(),
            json_mode=True,
            max_tokens=Config.LLM_MAX_TOKENS_LARGE,
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
            "target_column": self._target_column,
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

    _LABEL_NAMES = frozenset({
        "target", "label", "y", "class", "output",
        "default_flag", "fraud", "is_fraud", "bad_flag", "churn", "default",
    })

    def _collect_label_stats(self) -> Optional[str]:
        if self._target_column:
            if self._target_column not in self.df.columns:
                self.logger.log(self.name, "WARN",
                    f"Specified target_column '{self._target_column}' not found in columns — falling back to auto-detect")
                self._target_column = None   # reset so auto-detect runs below
            else:
                self.logger.log(self.name, "Label", f"Using supplied target_column='{self._target_column}'")
                return self.execute_tool("check_label_quality", df=self.df, label_col=self._target_column)

        # Auto-detect
        candidates = [c for c in self.df.columns if c.lower() in self._LABEL_NAMES]
        if not candidates:
            self.logger.log(self.name, "WARN",
                f"No recognised label column found — falling back to last column '{self.df.columns[-1]}'. "
                "Pass --target explicitly if this is wrong.")
            candidates = [self.df.columns[-1]]
        self._target_column = candidates[0]
        self.logger.log(self.name, "Label", f"Auto-detected target_column='{self._target_column}'")
        return self.execute_tool("check_label_quality", df=self.df, label_col=self._target_column)

    def _collect_temporal_stats(self) -> Optional[str]:
        date_cols = [c for c in self.df.columns
                     if any(kw in c.lower() for kw in ("date", "time", "timestamp", "dt"))]
        if not date_cols:
            return None
        combined: Dict[str, Any] = {}
        for dc in date_cols:
            combined[dc] = json.loads(self.execute_tool("check_temporal", df=self.df, date_col=dc))
        return json.dumps(combined, indent=2)

    @staticmethod
    def _is_entity_id_col(col: str) -> bool:
        """True if col name looks like a primary entity / surrogate key column.

        Tiers (checked in order):
          1. ends with _id           — customer_id, loan_id, contract_id
          2. "id" is a middle word   — sk_id_curr (["sk","id","curr"]), NOT id_number (["id","number"])
             Rule: "id" appears at parts[1..n-1], never as the very first word.
             This avoids false positives like id_number / id_card (national ID documents).

        Does NOT match:
          rapid, hidden, stride      — "id" is inside a longer word, not a _-segment
          id_number, id_card         — "id" is the first word → identity document, not entity key
        """
        parts = col.lower().split("_")
        ends_with_id = parts[-1] == "id"
        id_in_middle = "id" in parts[1:]   # skip index 0 to exclude id_number / id_card
        return ends_with_id or id_in_middle

    # Primary entity key names — used to detect entity_id_col
    _ENTITY_ID_EXACT = frozenset({
        # English / international
        "customer_id", "entity_id", "user_id", "client_id", "account_id",
        "cust_id", "applicant_id", "borrower_id",
        "loan_id", "application_id", "contract_id",
        # CIF / T24 core banking
        "cif", "cif_id", "cif_no", "so_cif", "ma_cif",
        "cust_no", "cust_t24", "customer_no", "customer_number", "customer_t24",
        "t24_number", "t24_num",
        "client_no", "account_no",
        # Vietnamese banking — mã khách hàng
        "ma_kh", "ma_khach", "ma_khach_hang", "ma_khach_hang_t24",
        "kh_code", "kh_ma", "id_kh",
        # Vietnamese banking — số hợp đồng / hồ sơ
        "ma_hd", "so_hd", "so_ho_so",
    })

    # Secondary identity columns used for entity resolution (fraud ring detection).
    # Each group maps a semantic label → tuple of column name variants.
    # NOTE: these may overlap with _ENTITY_ID_EXACT intentionally — whichever column
    # becomes entity_id_col is excluded from identity_cols at runtime.
    _IDENTITY_MAP = {
        "cif": (
            "cif", "cif_id", "cif_no", "so_cif", "ma_cif",
            "cust_no", "cust_t24", "customer_no", "customer_number", "customer_t24",
            "t24_number", "t24_num",
        ),
        "phone": ("phone", "phone_number", "mobile", "dien_thoai", "so_dt", "so_dien_thoai"),
        "device": ("device", "device_id", "thiet_bi"),
        "national_id": (
            "national_id", "nid", "id_number", "id_no", "idnum",
            "cccd", "cmt", "cmnd", "so_cmnd", "so_cccd",
        ),
    }

    def _collect_pk_stats(self) -> Optional[str]:
        # ── Step 1: resolve entity_id_col ────────────────────────────────────
        if self._entity_id_col is not None:
            # Caller supplied it explicitly — validate it exists in the data
            entity_id_col = self._entity_id_col
            if entity_id_col not in self.df.columns:
                self.logger.log(self.name, "WARN",
                    f"Specified entity_id_col '{entity_id_col}' not found in columns — skipping PK check")
                return None
            self.logger.log(self.name, "PK", f"Using supplied entity_id_col='{entity_id_col}'")
        else:
            # Auto-detect: Priority 1 — exact known names (case-insensitive)
            entity_id_col = next(
                (c for c in self.df.columns if c.lower() in self._ENTITY_ID_EXACT),
                # Priority 2 — generic heuristic: ends _id or "id" is a middle word
                next((c for c in self.df.columns if self._is_entity_id_col(c)), None),
            )
            if entity_id_col is None:
                return None
            self.logger.log(self.name, "PK", f"Auto-detected entity_id_col='{entity_id_col}'")

        # ── Step 2: resolve composite_key_cols ───────────────────────────────
        if self._composite_key_cols:
            # Caller supplied composite key — validate all cols exist
            missing = [c for c in self._composite_key_cols if c not in self.df.columns]
            if missing:
                self.logger.log(self.name, "WARN",
                    f"Supplied composite_key_cols contain unknown columns {missing} — they will be ignored")
            composite_key_cols = [c for c in self._composite_key_cols if c in self.df.columns]
            if not composite_key_cols:
                composite_key_cols = [entity_id_col]
            self.logger.log(self.name, "PK", f"Using supplied composite_key_cols={composite_key_cols}")
        else:
            # Auto-detect composite key partner.
            #
            # A candidate column qualifies when:
            #   1. No null values  (null → can't form a reliable key)
            #   2. (entity_id_col, candidate) uniquely identifies every row
            #      i.e. groupby ngroups == len(df)
            #
            # Search order:
            #   Priority 1 — date/time/period columns (matched by keyword in name)
            #   Priority 2 — any other column, but ONLY when entity_id is NOT
            #                already unique per row (avoids attaching arbitrary
            #                feature columns in one-row-per-entity tables).
            n_rows = len(self.df)
            n_entity_unique = self.df[entity_id_col].nunique()
            composite_key_cols = [entity_id_col]

            _DATE_KWS = ("date", "time", "timestamp", "dt", "month", "period", "year")
            _excl = {entity_id_col, self._target_column}   # never use target as key partner
            date_candidates = [
                c for c in self.df.columns
                if c not in _excl
                and any(kw in c.lower() for kw in _DATE_KWS)
            ]
            other_candidates = (
                [c for c in self.df.columns if c not in _excl and c not in date_candidates]
                if n_entity_unique < n_rows else []
            )

            for dc in date_candidates + other_candidates:
                if self.df[dc].isna().any():
                    continue
                n_groups = self.df.groupby([entity_id_col, dc], sort=False).ngroups
                if n_groups == n_rows:
                    composite_key_cols = [entity_id_col, dc]
                    break

        # ── Step 3: persist for downstream use ───────────────────────────────
        self._entity_id_col = entity_id_col
        self._composite_key_cols = composite_key_cols

        # Use pre-resolved target_column (set by _collect_label_stats which runs before this)
        label_col = self._target_column

        # Auto-detect secondary identity cols for entity resolution.
        # Exclude entity_id_col itself — no point checking a column against itself.
        identity_cols = {
            label: col
            for label, kws in self._IDENTITY_MAP.items()
            for col in self.df.columns
            if col.lower() in kws and col != entity_id_col
        }

        return self.execute_tool(
            "check_pk_uniqueness",
            df=self.df,
            entity_id_col=entity_id_col,
            composite_key_cols=composite_key_cols,
            label_col=label_col,
            identity_cols=identity_cols if identity_cols else None,
        )

    def _get_system_prompt(self) -> str:
        return self._load_prompt(
            self._PROMPTS_ROOT / "DataCleaner" / "prompts" / "system.txt",
            TOOL_DESCRIPTIONS=self.tool_registry.get_tool_descriptions(),
            NULL_DROP_THRESHOLD_PCT=f"{Config.NULL_DROP_THRESHOLD * 100:.0f}",
            DUPLICATE_PCT_THRESHOLD=f"{Config.DUPLICATE_PCT_THRESHOLD:.1f}",
            OUTLIER_PCT_THRESHOLD=f"{Config.OUTLIER_PCT_THRESHOLD:.1f}",
            IMBALANCE_RATIO_THRESHOLD=f"{Config.IMBALANCE_RATIO_THRESHOLD:.0f}",
            TARGET_COLUMN=self._target_column or "unknown",
            PK_VIOLATION_PCT_THRESHOLD=f"{Config.PK_VIOLATION_PCT_THRESHOLD:.1f}",
            AMBIGUOUS_IDENTITY_PCT_THRESHOLD=f"{Config.AMBIGUOUS_IDENTITY_PCT_THRESHOLD:.1f}",
            ENTITY_ORPHAN_PCT_THRESHOLD=f"{Config.ENTITY_ORPHAN_PCT_THRESHOLD:.1f}",
            SOFT_DUP_PCT_THRESHOLD=f"{Config.SOFT_DUP_PCT_THRESHOLD:.1f}",
            APP_DUP_PCT_THRESHOLD=f"{Config.APP_DUP_PCT_THRESHOLD:.1f}",
        )

    def _compress_metadata_for_llm(self, metadata_json: str) -> str:
        """Compress raw metadata to an actionable summary for the LLM prompt.

        With wide datasets (500+ columns), the raw JSON from inspect_metadata
        contains per-column dtypes, null_counts, and null_percentages for every
        column, easily reaching 80K+ chars and overflowing the LLM context.
        We keep only the fields needed to make cleaning decisions.
        """
        from collections import Counter
        meta = json.loads(metadata_json)
        n_rows, n_cols = meta["shape"]
        null_pcts: Dict[str, float] = meta.get("null_percentages", {})
        dtypes: Dict[str, str] = meta.get("dtypes", {})
        null_thresh_pct = Config.NULL_DROP_THRESHOLD * 100

        droppable_null = {
            k: round(v, 2) for k, v in null_pcts.items() if v > null_thresh_pct
        }
        moderate_null_count = sum(1 for v in null_pcts.values() if 20 < v <= null_thresh_pct)
        low_null_count = sum(1 for v in null_pcts.values() if 0 < v <= 20)
        zero_null_count = sum(1 for v in null_pcts.values() if v == 0)

        return json.dumps({
            "shape": [n_rows, n_cols],
            "total_columns": n_cols,
            "duplicate_rows": meta.get("duplicate_rows"),
            "duplicate_percentage": meta.get("duplicate_percentage"),
            "dtype_distribution": dict(Counter(dtypes.values())),
            "constant_columns": meta.get("constant_columns", []),
            "constant_columns_count": meta.get("constant_columns_count", 0),
            f"columns_null_above_{int(null_thresh_pct)}pct_DROPPABLE": droppable_null,
            f"columns_null_above_{int(null_thresh_pct)}pct_count": len(droppable_null),
            "columns_null_20_to_80pct_count": moderate_null_count,
            "columns_null_1_to_20pct_count": low_null_count,
            "columns_zero_null_count": zero_null_count,
            "action_note": (
                f"Only columns with null > {int(null_thresh_pct)}% are listed above "
                "(candidates for drop_column). All other column issues are covered by "
                "FORMAT_STATS and OUTLIER_STATS below."
            ),
        }, indent=2)

    def _build_analysis_prompt(
        self,
        metadata: str,
        format_stats: str,
        outlier_stats: str,
        label_stats: Optional[str],
        temporal_stats: Optional[str],
        pk_stats: Optional[str],
    ) -> str:
        optional_parts = []
        if label_stats:
            optional_parts.append(f"LABEL QUALITY:\n{label_stats}")
        if temporal_stats:
            optional_parts.append(f"TEMPORAL INTEGRITY:\n{temporal_stats}")
        if pk_stats:
            optional_parts.append(
                f"PK & UNIQUENESS CHECK (exact/soft/app duplicates, composite key, entity resolution):\n{pk_stats}"
            )
        optional_sections = ("\n\n" + "\n\n".join(optional_parts) + "\n\n") if optional_parts else "\n\n"
        return self._load_prompt(
            self._PROMPTS_ROOT / "DataCleaner" / "prompts" / "user.txt",
            METADATA=self._compress_metadata_for_llm(metadata),
            FORMAT_STATS=format_stats,
            OUTLIER_STATS=outlier_stats,
            OPTIONAL_SECTIONS=optional_sections,
        )

    def _execute_llm_decisions(self, llm_response: str) -> List[str]:
        self.logger.log(self.name, "LLM Decision", "Parsing decisions from LLM response")
        actions_taken = []

        try:
            decisions = json.loads(self._extract_json(llm_response))
            self.logger.log(self.name, "LLM Reasoning", decisions.get("reasoning", "No reasoning provided"))

            for action_spec in decisions.get("actions", []):
                action_type = action_spec.get("action")
                column = action_spec.get("column")
                reason = action_spec.get("reason", "No reason provided")

                self.logger.log(self.name, f"Action: {action_type}", f"Column: {column}, Reason: {reason}")

                try:
                    if action_type == "drop_column":
                        if column in self._pk_protected_cols:
                            self.logger.log(self.name, f"SKIP {action_type}",
                                f"'{column}' is a PK/composite-key column — drop blocked")
                            continue
                        # Hard guard: only allow drop when the column genuinely qualifies.
                        # Legitimate reasons: (a) null_rate > threshold, OR
                        #                     (b) constant (zero variance), OR
                        #                     (c) all-unique surrogate ID (row index / txn key)
                        # Everything else → block; feature selection belongs to Agent 2.
                        if column in self.df.columns:
                            null_rate = self.df[column].isnull().mean()
                            is_constant = self.df[column].dropna().nunique() <= 1
                            is_all_unique = (
                                self.df[column].nunique() == len(self.df)
                                and self.df[column].notna().all()
                            )
                            if not is_constant and not is_all_unique and null_rate <= Config.NULL_DROP_THRESHOLD:
                                self.logger.log(self.name, f"BLOCK {action_type}",
                                    f"'{column}' null_rate={null_rate:.1%} ≤ {Config.NULL_DROP_THRESHOLD:.0%} threshold, "
                                    f"not constant, not all-unique — drop blocked (reason from LLM: {reason})")
                                continue
                        self.df = self.execute_tool("drop_column", df=self.df, col=column)
                        actions_taken.append(f"Dropped column '{column}': {reason}")

                    # elif action_type == "impute_missing":  # disabled — impute_missing tool not active
                    #     strategy = action_spec.get("strategy", "median")
                    #     self.df = self.execute_tool("impute_missing", df=self.df, col=column, strategy=strategy)
                    #     actions_taken.append(f"Imputed '{column}' with {strategy}: {reason}")

                    elif action_type == "drop_duplicates":
                        self.df = self.execute_tool("drop_duplicates", df=self.df)
                        actions_taken.append(f"Dropped duplicate rows: {reason}")

                    elif action_type == "clip_outliers":
                        if column in self._pk_protected_cols:
                            self.logger.log(self.name, f"SKIP {action_type}",
                                f"'{column}' is a PK/composite-key column — clip blocked")
                            continue
                        factor = float(action_spec.get("factor", 3.0))
                        self.df = self.execute_tool("clip_outliers", df=self.df, col=column, factor=factor)
                        actions_taken.append(f"Clipped outliers in '{column}' (factor={factor}): {reason}")

                    elif action_type == "deduplicate_by_key":
                        key_cols = action_spec.get("columns", [])
                        keep = action_spec.get("keep", "first")
                        if self._target_column and self._target_column in key_cols:
                            self.logger.log(self.name, f"SKIP {action_type}",
                                f"Target column '{self._target_column}' must not be a dedup key — blocked")
                            continue
                        self.df = self.execute_tool("deduplicate_by_key", df=self.df, key_cols=key_cols, keep=keep)
                        actions_taken.append(f"Deduplicated by key {key_cols} (keep={keep}): {reason}")

                    elif action_type == "fix_column_dtype":
                        if column in self._pk_protected_cols:
                            self.logger.log(self.name, f"SKIP {action_type}",
                                f"'{column}' is a PK/composite-key column — format fix skipped")
                            continue
                        fix_type = action_spec.get("fix_type", "")
                        self.df = self.execute_tool("fix_column_dtype", df=self.df, col=column, fix_type=fix_type)
                        actions_taken.append(f"Fixed format of '{column}' ({fix_type}): {reason}")

                    else:
                        self.logger.log(self.name, "WARN",
                            f"Unknown or missing action_type '{action_type}' — skipped")

                except Exception as action_err:
                    self.logger.log(self.name, f"SKIP {action_type}",
                        f"Action failed, continuing with next action: {action_err}")

        except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
            self.logger.log(self.name, "ERROR", f"Failed to parse LLM response as JSON: {e}")
            self.logger.log(self.name, "Raw Response", llm_response[:500])
            actions_taken.append("ERROR: Could not parse LLM decisions, performed basic cleaning")
            self._fallback_cleaning()

        return actions_taken

    @property
    def _pk_protected_cols(self) -> set:
        """Set of columns that must never be dropped or coerced — entity ID, composite keys, and target."""
        protected = set(self._composite_key_cols)
        if self._entity_id_col:
            protected.add(self._entity_id_col)
        if self._target_column:
            protected.add(self._target_column)
        return protected

    def _fallback_cleaning(self):
        protected = self._pk_protected_cols
        for col in list(self.df.columns):
            if col in protected:
                continue
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
