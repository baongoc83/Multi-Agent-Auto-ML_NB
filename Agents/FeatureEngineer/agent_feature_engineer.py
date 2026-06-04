import re
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif, f_regression
from Agents.BaseAgent.base_agent import BaseAgent, ToolRegistry
from logger import AgentLogger
from config import Config


@dataclass
class FeatureSpec:
    """Captured transforms from TRAIN, replayed on VALID/OOT.

    - interactions: list of (new_col, expression, fill_value) where fill_value
      is the train-median used to fill NaN/inf so valid/oot get a deterministic
      imputation, not their own median (leakage-free).
    - label_encoders: fitted LabelEncoder per column. The fit includes an
      `__NA__` sentinel so unseen categories in valid/oot map to it cleanly.
    - onehot_columns: per-col list of dummy column names produced on train,
      used to reindex valid/oot dummies (extra cols dropped, missing cols
      back-filled with 0).
    - selected_features: final column subset chosen by select_top_features
      (target appended automatically at apply time).
    """
    interactions:      List[Tuple[str, str, float]] = field(default_factory=list)
    label_encoders:    Dict[str, Any]               = field(default_factory=dict)
    onehot_columns:    Dict[str, List[str]]         = field(default_factory=dict)
    selected_features: Optional[List[str]]          = None
    target_column:     Optional[str]                = None

    def apply(self, df: pd.DataFrame, logger=None, name: str = "") -> pd.DataFrame:
        """Replay every captured transform on `df` in fit-time order."""

        # 1. Re-create interactions (same expression; NaN/inf filled with train median)
        for new_col, expression, fill_value in self.interactions:
            safe_locals = {"df": df, "np": np}
            try:
                df[new_col] = eval(expression, {"__builtins__": {}}, safe_locals)  # noqa: S307
                df[new_col] = df[new_col].replace([np.inf, -np.inf], np.nan)
                df[new_col] = df[new_col].fillna(fill_value)
            except Exception as e:
                if logger is not None:
                    logger.log(name, "Transform WARN",
                        f"interaction '{new_col}' failed on transform set: {e} — filling with train median")
                df[new_col] = fill_value

        # 2. Label encoders — unseen categories map to __NA__ sentinel
        for col, le in self.label_encoders.items():
            if col not in df.columns:
                continue
            known = set(le.classes_)
            vals = df[col].astype(str).apply(lambda x: x if x in known else "__NA__")
            df[col] = le.transform(vals)

        # 3. One-hot — reindex against train's dummy column list
        for col, dummy_cols in self.onehot_columns.items():
            if col not in df.columns:
                continue
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            dummies = dummies.reindex(columns=dummy_cols, fill_value=0)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        # 4. Column selection — keep only the train-chosen features (+ target)
        if self.selected_features is not None:
            keep = [c for c in self.selected_features if c in df.columns]
            if self.target_column and self.target_column in df.columns and self.target_column not in keep:
                keep.append(self.target_column)
            df = df[keep]
        return df

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interactions": [[n, e, f] for (n, e, f) in self.interactions],
            "label_encoders": sorted(self.label_encoders.keys()),
            "onehot_columns": {c: list(v) for c, v in self.onehot_columns.items()},
            "selected_features": self.selected_features,
            "target_column": self.target_column,
        }


class FeatureEngineerAgent(BaseAgent):

    _DOMAIN_GUIDANCE: Dict[str, str] = {
        "credit_risk": """\
DOMAIN: Credit Risk ({model_type})
Target: Binary default flag — 1 = defaulted (DPD ≥ threshold) within observation window (e.g. MOB6/MOB12)
Key interaction patterns:
- Debt burden        : credit_balance / (income + 1), installment_amount / (income + 1)
- Credit utilization : credit_drawn / (credit_limit + 1)
- Repayment quality  : overdue_amount / (total_due + 1), overdue_days / (loan_term + 1)
- Bureau delinquency : bureau_dpd_count / (bureau_total_loans + 1)
- Behavioral trend   : compare recent (MOB1-3) vs historical aggregates (MOB1-12)
Avoid leakage: do not use post-origination data or outcomes after the observation cutoff.""",

        "propensity": """\
DOMAIN: Propensity Model ({model_type})
Target: Binary flag — 1 = positive outcome (purchased / retained / responded / interested)
Key interaction patterns:
- RFM               : days_since_last_action, event_count_30d / (event_count_90d + 1)
- Engagement rate   : interactions / (total_contacts + 1), clicks / (impressions + 1)
- Product affinity  : category_count / (total_purchase_count + 1)
- Value             : avg_order_value, cumulative_spend / (tenure_days + 1)
Avoid: features derived from the future or from post-event behavior not available at scoring time.""",

        "fraud": """\
DOMAIN: Fraud Detection ({model_type})
Target: Binary flag — 1 = fraudulent (extreme class imbalance expected: 0.1–5%)
Key interaction patterns:
- Velocity          : txn_count_1h / (txn_count_24h + 1), amount_1h / (amount_30d + 1)
- Amount anomaly    : current_amount / (entity_avg_amount + 1)
- Network signals   : shared_device_count, shared_phone_count across entities
- Time patterns     : hour_of_day, days_since_account_open, weekend_flag * amount
- Risk ratio        : high_value_txn_count / (total_txn_count + 1)
Note: extreme imbalance — prefer interactions that amplify differences between fraud and non-fraud behavior.""",

        "generic": """\
DOMAIN: General ML ({model_type})
Apply standard feature engineering: ratio features between semantically related columns, \
frequency encoding for high-cardinality categoricals, interaction terms between correlated predictors.""",
    }

    def __init__(
        self,
        logger: AgentLogger,
        col_descriptions_path: Optional[str] = None,
        col_name_field: str = "column_name",
        col_desc_field: str = "description",
        col_group_field: Optional[str] = None,
        domain: str = "generic",
        model_type: str = "binary_classification",
    ):
        super().__init__(name="FeatureEngineer", role="Feature Architect", logger=logger)
        self.df: pd.DataFrame = None
        self.target_column: str = None
        self._protected_cols: set = set()
        self._date_col: Optional[str] = None
        self.domain = domain.lower()
        self.model_type = model_type.lower()
        self._col_descriptions: Dict[str, str] = self._load_col_descriptions(
            col_descriptions_path, col_name_field, col_desc_field, col_group_field
        )
        # Capture buffers populated by tool methods and harvested by _execute_llm_decisions
        # into the FeatureSpec. Initialised to None so we never read stale state.
        self._last_label_encoder: Optional[Any] = None
        self._last_onehot_cols: Optional[List[str]] = None
        self._last_selected_features: Optional[List[str]] = None
        self._last_interaction_fill: float = 0.0
        self._batch_label_encoders: Dict[str, Any] = {}
        self._batch_onehot_cols: Dict[str, List[str]] = {}
        self.tool_registry = ToolRegistry()
        self._register_tools()

    def _get_domain_guidance(self) -> str:
        if self.domain not in self._DOMAIN_GUIDANCE:
            self.logger.log(self.name, "WARN",
                f"Unknown domain '{self.domain}' — using generic guidance. "
                f"Valid options: {list(self._DOMAIN_GUIDANCE.keys())}")
        template = self._DOMAIN_GUIDANCE.get(self.domain, self._DOMAIN_GUIDANCE["generic"])
        return template.format(model_type=self.model_type)

    def _load_col_descriptions(
        self,
        path: Optional[str],
        col_name_field: str = "column_name",
        col_desc_field: str = "description",
        col_group_field: Optional[str] = None,
    ) -> Dict[str, str]:
        """Load column descriptions from JSON, CSV, Parquet, or Excel.

        For tabular files (CSV/Parquet/Excel), specify which columns hold the
        column name, description, and optional group via the *_field params.

        Example — HomeCredit CSV (Table=group, Row=name, Description=desc):
            _load_col_descriptions(path, col_name_field="Row",
                                   col_desc_field="Description",
                                   col_group_field="Table")
        """
        if not path:
            return {}
        try:
            suffix = Path(path).suffix.lower()
            if suffix == ".json":
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    # Detect 3-level: {group: {col_name: desc, ...}, ...}
                    if data and isinstance(next(iter(data.values())), dict):
                        return self._parse_nested_json_descriptions(data, col_group_field)
                    # Flat 2-level: {col_name: desc, ...}
                    return {str(k): str(v) for k, v in data.items()}
                if isinstance(data, list):
                    # List of dicts — treat as tabular rows
                    return self._parse_tabular_descriptions(
                        pd.DataFrame(data), col_name_field, col_desc_field, col_group_field
                    )
                self.logger.log(self.name, "WARN",
                    f"JSON col_descriptions: expected dict or list, got {type(data).__name__}")
                return {}
            elif suffix == ".csv":
                for _enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
                    try:
                        df_desc = pd.read_csv(path, encoding=_enc)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise UnicodeDecodeError("csv", b"", 0, 1,
                        f"Could not decode '{path}' with utf-8 / cp1252 / latin-1")
            elif suffix in (".xls", ".xlsx", ".xlsm"):
                # _read_excel_smart sniffs file magic so a mislabelled .xls /
                # .xlsx pair is still readable. Surfaces a clearer error when
                # the file is encrypted/corrupt or when openpyxl/xlrd is
                # missing from the environment.
                df_desc = self._read_excel_smart(path)
            elif suffix == ".parquet":
                df_desc = pd.read_parquet(path)
            else:
                self.logger.log(self.name, "WARN",
                    f"Unsupported col_descriptions file type: '{suffix}'. "
                    "Supported: .json, .csv, .parquet, .xls, .xlsx")
                return {}
            return self._parse_tabular_descriptions(df_desc, col_name_field, col_desc_field, col_group_field)
        except Exception as e:
            self.logger.log(self.name, "WARN", f"Could not load col_descriptions from '{path}': {e}")
        return {}

    def _parse_tabular_descriptions(
        self,
        df_desc: pd.DataFrame,
        col_name_field: str,
        col_desc_field: str,
        col_group_field: Optional[str],
    ) -> Dict[str, str]:
        """Parse a tabular description file into {column_name: description} dict.

        Only col_name_field and col_desc_field are read; all other columns
        (e.g. Special, Notes) are silently ignored.
        """
        if col_name_field not in df_desc.columns:
            self.logger.log(self.name, "WARN",
                f"col_name_field '{col_name_field}' not found in description file. "
                f"Available columns: {list(df_desc.columns)}")
            return {}
        if col_desc_field not in df_desc.columns:
            self.logger.log(self.name, "WARN",
                f"col_desc_field '{col_desc_field}' not found in description file. "
                f"Available columns: {list(df_desc.columns)}")
            return {}
        has_group = bool(col_group_field and col_group_field in df_desc.columns)
        result: Dict[str, str] = {}
        for _, row in df_desc.iterrows():
            col_name = str(row[col_name_field]).strip()
            if not col_name or col_name == "nan":
                continue
            desc = str(row[col_desc_field]).strip() if pd.notna(row[col_desc_field]) else ""
            if has_group and pd.notna(row.get(col_group_field)):
                group = str(row[col_group_field]).strip()
                desc = f"[{group}] {desc}" if desc else f"[{group}]"
            result[col_name] = desc
        self.logger.log(self.name, "Col Descriptions Loaded",
            f"Loaded {len(result)} descriptions"
            + (f" | name_field='{col_name_field}' desc_field='{col_desc_field}'"
               f" group_field='{col_group_field}'" if has_group
               else f" | name_field='{col_name_field}' desc_field='{col_desc_field}'"))
        return result

    def _parse_nested_json_descriptions(
        self,
        data: dict,
        col_group_field: Optional[str],
    ) -> Dict[str, str]:
        """Parse 3-level JSON: {group: {col_name: desc, ...}, ...} → {col_name: desc}.

        If col_group_field is not None, prefixes each description with [group].
        If a column name appears in multiple groups, the last group wins.
        """
        include_group = col_group_field is not None
        result: Dict[str, str] = {}
        for group, cols in data.items():
            if not isinstance(cols, dict):
                continue
            for col_name, desc in cols.items():
                col_name = str(col_name).strip()
                desc = str(desc).strip() if desc is not None else ""
                result[col_name] = f"[{group}] {desc}" if include_group and desc else (
                    f"[{group}]" if include_group else desc
                )
        self.logger.log(self.name, "Col Descriptions Loaded",
            f"Loaded {len(result)} descriptions from nested JSON "
            f"({len(data)} groups)"
            + (" with group prefix" if include_group else ""))
        return result

    def _register_tools(self):
        self.tool_registry.register(
            "create_interaction",
            "Creates a new column using mathematical expressions between columns based on meaning of columns",
            {
                "df": "The dataframe",
                "new_col": "Name for the new column",
                "expression": "Python expression using df['col'] syntax, e.g. \"df['a'] / df['b']\"",
            },
        )
        self.tool_registry.register(
            "encode_categorical",
            "Encodes a single categorical column into numeric format",
            {
                "df": "The dataframe",
                "col": "Column name to encode",
                "method": "Either 'label' (ordinal) or 'onehot' (binary columns)",
            },
        )
        self.tool_registry.register(
            "encode_all_categorical",
            "Encodes ALL remaining object/categorical columns at once (preferred for wide datasets)",
            {
                "df": "The dataframe",
                "method": "Either 'label' (default) or 'onehot'",
            },
        )
        self.tool_registry.register(
            "correlation_analysis",
            "Analyzes correlation between features and target",
            {"df": "The dataframe", "target": "Target column name"},
        )
        self.tool_registry.register(
            "select_top_features",
            "Keeps only the k most predictive features",
            {
                "df": "The dataframe",
                "target": "Target column name",
                "k": "Number of top features to keep",
            },
        )

    def _tool_create_interaction(self, df: pd.DataFrame, new_col: str, expression: str) -> pd.DataFrame:
        # Restrict eval to df and np only — no builtins — to prevent code injection
        safe_locals = {"df": df, "np": np}
        try:
            df[new_col] = eval(expression, {"__builtins__": {}}, safe_locals)  # noqa: S307
            df[new_col] = df[new_col].replace([np.inf, -np.inf], np.nan)
            train_median = df[new_col].median()
            # Median can still be NaN if every value is NaN/inf — use 0.0 as a safe sentinel
            if pd.isna(train_median):
                train_median = 0.0
            df[new_col] = df[new_col].fillna(train_median)
            # Drop if constant or all-NaN — causes divide-by-zero in correlation/SelectKBest
            if df[new_col].isna().all() or df[new_col].std() == 0:
                df = df.drop(columns=[new_col])
                raise ValueError(f"Generated feature '{new_col}' is constant or all-NaN after fill — dropped")
            # Expose train median so _execute_llm_decisions can capture it for replay on valid/oot
            self._last_interaction_fill = float(train_median)
            return df
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Error creating interaction '{new_col}': {e}")

    def _tool_encode_categorical(self, df: pd.DataFrame, col: str, method: str) -> pd.DataFrame:
        """FIT-on-train encoding. Fitted encoders / column lists are exposed via
        `_last_label_encoder` and `_last_onehot_cols` so `_execute_llm_decisions`
        can capture them into a FeatureSpec for valid/oot replay.
        """
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")

        if method == "label":
            le = LabelEncoder()
            # Train values + __NA__ sentinel — guarantees transform never raises
            # on unseen valid/oot categories when we replay this encoder.
            train_vals = df[col].astype(str)
            le.fit(sorted(set(train_vals.tolist()) | {"__NA__"}))
            df[col] = le.transform(train_vals)
            self._last_label_encoder = le

        elif method == "onehot":
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            self._last_onehot_cols = list(dummies.columns)
        else:
            raise ValueError(f"Unknown encoding method: {method}")

        return df

    def _tool_encode_all_categorical(self, df: pd.DataFrame, method: str = "label") -> pd.DataFrame:
        """Encode every non-numeric column (except protected/target) using `method`.

        For each column, the fitted encoder is stored in `_batch_label_encoders` /
        `_batch_onehot_cols` so `_execute_llm_decisions` can transfer them
        into the FeatureSpec.
        """
        skip = self._protected_cols | {self.target_column}
        cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c not in skip]
        self._batch_label_encoders: Dict[str, Any] = {}
        self._batch_onehot_cols: Dict[str, List[str]] = {}
        encoded = 0
        for col in cat_cols:
            effective_method = method
            if method == "onehot":
                col_nu = df[col].nunique()
                if col_nu > 5:
                    effective_method = "label"
                    self.logger.log(self.name, "encode_all_categorical",
                        f"'{col}' nunique={col_nu} > 5 — falling back to label encoding")
            self._last_label_encoder = None
            self._last_onehot_cols = None
            df = self._tool_encode_categorical(df, col, effective_method)
            if effective_method == "label" and self._last_label_encoder is not None:
                self._batch_label_encoders[col] = self._last_label_encoder
            elif effective_method == "onehot" and self._last_onehot_cols is not None:
                self._batch_onehot_cols[col] = self._last_onehot_cols
            encoded += 1
        self.logger.log(self.name, "encode_all_categorical",
            f"Encoded {encoded} categorical columns (requested method='{method}')")
        return df

    def _tool_correlation_analysis(self, df: pd.DataFrame, target: str) -> str:
        """Correlation of each feature with the target. `df` is the TRAIN frame
        in split mode (caller passes self.df, which IS train), so the score is
        leakage-free.
        """
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        y = pd.to_numeric(df[target], errors="coerce")
        if y.isna().all():
            raise ValueError(
                f"Target '{target}' cannot be converted to numeric — correlation analysis skipped"
            )

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)

        correlations = {}
        unreliable = []
        for col in numeric_cols:
            null_pct = df[col].isnull().mean() * 100
            if df[col].std() == 0:
                correlations[col] = 0.0
                continue
            corr = df[col].corr(y)
            correlations[col] = round(corr, 4) if not pd.isna(corr) else 0.0
            if null_pct > 30:
                unreliable.append(f"{col} ({null_pct:.1f}% null)")

        sorted_corrs = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))

        result = {
            "correlations": sorted_corrs,
            "high_correlation": [k for k, v in sorted_corrs.items() if abs(v) > Config.HIGH_CORRELATION_THRESHOLD],
            "low_correlation": [k for k, v in sorted_corrs.items() if abs(v) < Config.LOW_CORRELATION_THRESHOLD],
            "unreliable_due_to_nulls": unreliable,
        }
        return json.dumps(result, indent=2)

    def _tool_select_top_features(self, df: pd.DataFrame, target: str, k: int) -> pd.DataFrame:
        """Score features on the TRAIN frame (caller passes self.df=train) and
        keep the top-k. The retained column list is exposed via
        `_last_selected_features` so it can be captured into FeatureSpec and
        applied to valid/oot during transform.
        """
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        feature_cols = [c for c in df.columns if c != target]
        non_numeric_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_features:
            return df

        y = pd.to_numeric(df[target], errors="coerce")
        valid_mask = y.notna()
        y_valid = y[valid_mask].values
        if len(y_valid) < 2 or pd.Series(y_valid).nunique() < 2:
            # Not enough training signal — return df unchanged
            return df

        is_classification = pd.Series(y_valid).nunique() < Config.CLASSIFICATION_UNIQUE_THRESHOLD
        score_func = f_classif if is_classification else f_regression

        score_idx = df.index[valid_mask]

        scores: Dict[str, float] = {}
        for col in numeric_features:
            x = df.loc[score_idx, col]
            med = x.median()
            if pd.isna(med):
                scores[col] = 0.0
                continue
            x = x.fillna(med).values
            if x.std() == 0:
                scores[col] = 0.0
                continue
            try:
                f_stat, _ = score_func(x.reshape(-1, 1), y_valid)
                scores[col] = float(f_stat[0]) if not np.isnan(f_stat[0]) else 0.0
            except Exception:
                scores[col] = 0.0

        eligible = [c for c, s in scores.items() if s > 0]
        k = min(k, len(eligible) if eligible else len(scores))
        selected_numeric = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
        final_cols = selected_numeric + non_numeric_cols + [target]
        # Exposed for capture into FeatureSpec
        self._last_selected_features = list(final_cols)
        return df[final_cols]

    def _setup_protected_cols(self, previous_report: Dict[str, Any]) -> None:
        """Compute self._protected_cols + self._date_col from prior agent's report.
        Idempotent — called by both fit_transform and process (single-file mode).
        """
        _entity_id = previous_report.get("entity_id_col")
        _composite = previous_report.get("composite_key_cols", [])
        self._protected_cols = set(_composite)
        if _entity_id:
            self._protected_cols.add(_entity_id)
        self._protected_cols.discard(self.target_column)

        # Infer date column from composite key (the non-entity partner, if any).
        self._date_col = next(
            (c for c in _composite if c != _entity_id and c in self.df.columns),
            None,
        )
        if self._date_col:
            self._protected_cols.add(self._date_col)

        if self._protected_cols:
            self.logger.log(self.name, "Protected Cols",
                f"Skipping feature engineering for: {sorted(self._protected_cols)}")
        if self._date_col:
            self.logger.log(self.name, "Date Col",
                f"'{self._date_col}' identified as temporal column — excluded from interactions")

    # ── FIT / TRANSFORM API ───────────────────────────────────────────────

    def fit_transform(
        self,
        train_path: str,
        previous_report: Dict[str, Any],
        target_column: str,
    ) -> Tuple[pd.DataFrame, FeatureSpec, List[str], Tuple[int, int]]:
        """Load TRAIN, run LLM feature-engineering, return (engineered_train, spec, actions, original_shape).

        The spec captures every column transform so valid/oot can be transformed
        without re-running the LLM. Encoders are fitted on train only.
        """
        self.logger.log(self.name, "fit_transform start", f"Loading train from {train_path}")
        self.df = self.load_dataframe(train_path)
        self.target_column = self._resolve_target_column(self.df, target_column)
        original_shape = self.df.shape

        self._setup_protected_cols(previous_report)
        self.logger.log(self.name, "Previous Agent Summary", previous_report.get("summary", "No summary"))

        analysis = self._analyze_features()
        llm_response = self.call_llm(
            self._build_engineering_prompt(analysis, previous_report),
            self._get_system_prompt(),
            json_mode=True,
            max_tokens=Config.LLM_MAX_TOKENS_LARGE,
        )

        spec = FeatureSpec(target_column=self.target_column)
        actions_taken = self._execute_llm_decisions(llm_response, spec=spec)
        return self.df, spec, actions_taken, original_shape

    def transform(self, df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
        """Apply a captured FeatureSpec to valid/oot. No LLM, no refit."""
        return spec.apply(df, logger=self.logger, name=self.name)

    def process_splits(
        self,
        train_path: str,
        previous_report: Dict[str, Any],
        target_column: str,
        valid_path: Optional[str] = None,
        oot_path: Optional[str] = None,
    ) -> Tuple[Dict[str, Optional[str]], Dict[str, Any]]:
        """Pre-split mode entry point. Fit on train, transform valid/oot one at a time.

        Returns ({"train": path, "valid": path|None, "oot": path|None}, report).
        """
        import gc

        train_df, spec, actions_taken, original_shape = self.fit_transform(
            train_path, previous_report, target_column)

        Path(Config.ENGINEERED_TRAIN_PATH).parent.mkdir(exist_ok=True)
        train_df.to_parquet(Config.ENGINEERED_TRAIN_PATH, compression="snappy", index=False)
        self.logger.log(self.name, "Train saved",
            f"shape={train_df.shape} | path={Config.ENGINEERED_TRAIN_PATH}")
        train_final_shape = train_df.shape
        train_columns = list(train_df.columns)
        del train_df, self.df
        self.df = None
        gc.collect()

        out_paths: Dict[str, Optional[str]] = {
            "train": Config.ENGINEERED_TRAIN_PATH, "valid": None, "oot": None,
        }
        for tag, in_path, out_path in [
            ("valid", valid_path, Config.ENGINEERED_VALID_PATH),
            ("oot",   oot_path,   Config.ENGINEERED_OOT_PATH),
        ]:
            if in_path is None:
                continue
            self.logger.log(self.name, f"Transform {tag} start", f"path={in_path}")
            df = self.load_dataframe(in_path)
            orig = df.shape
            df = self.transform(df, spec)
            df.to_parquet(out_path, compression="snappy", index=False)
            self.logger.log(self.name, f"Transform {tag} done",
                f"shape {orig} -> {df.shape} | saved={out_path}")
            out_paths[tag] = out_path
            del df
            gc.collect()

        report = {
            "agent": self.name,
            "mode": "split",
            "original_shape": list(original_shape),
            "final_shape": list(train_final_shape),
            "actions_taken": actions_taken,
            "summary": self._generate_summary(actions_taken),
            "final_features": train_columns,
            "entity_id_col": previous_report.get("entity_id_col"),
            "composite_key_cols": previous_report.get("composite_key_cols", []),
            "target_column": self.target_column,
            "out_paths": {k: v for k, v in out_paths.items() if v is not None},
            "feature_spec": spec.to_dict(),
        }
        self.save_report(report, Config.FEATURE_ENGINEER_REPORT_PATH)
        self.logger.log(self.name, "Process Complete",
            f"train={train_final_shape} | valid={'y' if out_paths['valid'] else '-'} "
            f"| oot={'y' if out_paths['oot'] else '-'}")
        return out_paths, report

    def process(
        self,
        df: pd.DataFrame,
        previous_report: Dict[str, Any],
        target_column: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Single-file mode. Used when no valid/oot was supplied to the pipeline."""
        self.logger.log(self.name, "Process Start", f"Received clean data with shape {df.shape}")

        self.df = df.copy()
        self.target_column = self._resolve_target_column(self.df, target_column)
        original_shape = self.df.shape

        self._setup_protected_cols(previous_report)
        self.logger.log(self.name, "Previous Agent Summary", previous_report.get("summary", "No summary"))

        analysis = self._analyze_features()
        llm_response = self.call_llm(
            self._build_engineering_prompt(analysis, previous_report),
            self._get_system_prompt(),
            json_mode=True,
            max_tokens=Config.LLM_MAX_TOKENS_LARGE,
        )
        spec = FeatureSpec(target_column=self.target_column)
        actions_taken = self._execute_llm_decisions(llm_response, spec=spec)

        self.df.to_parquet(Config.ENGINEERED_DATA_PATH, compression="snappy", index=False)
        self.logger.log(self.name, "Data Saved", f"Engineered data saved to {Config.ENGINEERED_DATA_PATH}")

        report = {
            "agent": self.name,
            "mode": "single",
            "original_shape": list(original_shape),
            "final_shape": list(self.df.shape),
            "actions_taken": actions_taken,
            "summary": self._generate_summary(actions_taken),
            "final_features": list(self.df.columns),
            "entity_id_col": previous_report.get("entity_id_col"),
            "composite_key_cols": previous_report.get("composite_key_cols", []),
            "target_column": self.target_column,
            "feature_spec": spec.to_dict(),
        }
        self.save_report(report, Config.FEATURE_ENGINEER_REPORT_PATH)
        self.logger.log(self.name, "Process Complete", f"Shape: {original_shape} -> {self.df.shape}")
        return Config.ENGINEERED_DATA_PATH, report

    def _analyze_features(self) -> Dict[str, Any]:
        """Build stats for the LLM prompt from self.df. In split mode self.df IS train."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

        exclude = self._protected_cols | {self.target_column}
        numeric_cols = [c for c in numeric_cols if c not in exclude]
        categorical_cols = [c for c in categorical_cols if c not in exclude]

        numeric_meta: Dict[str, Any] = {}
        numeric_limit = min(
            max(10, int(len(numeric_cols) * Config.FEATURE_META_NUMERIC_RATIO)),
            Config.FEATURE_META_MAX_NUMERIC_COLS,
        )
        for col in numeric_cols[:numeric_limit]:
            s = self.df[col]
            non_null = s.dropna()
            entry: Dict[str, Any] = {
                "null_pct": round(float(s.isnull().mean() * 100), 2),
                "nunique": int(s.nunique()),
                "mean":  round(float(non_null.mean()), 4)  if len(non_null) > 0 else None,
                "std":   round(float(non_null.std()),  4)  if len(non_null) > 1 else None,
                "min":   round(float(non_null.min()),  4)  if len(non_null) > 0 else None,
                "max":   round(float(non_null.max()),  4)  if len(non_null) > 0 else None,
                "skew":  round(float(non_null.skew()), 4)  if len(non_null) > 2 else None,
            }
            if col in self._col_descriptions:
                entry["description"] = self._col_descriptions[col]
            numeric_meta[col] = entry

        categorical_meta: Dict[str, Any] = {}
        for col in categorical_cols[:Config.FEATURE_META_MAX_CATEGORICAL_COLS]:
            s = self.df[col]
            top_vals = s.value_counts().head(5).to_dict()
            entry = {
                "null_pct": round(float(s.isnull().mean() * 100), 2),
                "nunique": int(s.nunique()),
                "top_values": {str(k): int(v) for k, v in top_vals.items()},
            }
            if col in self._col_descriptions:
                entry["description"] = self._col_descriptions[col]
            categorical_meta[col] = entry

        return {
            "shape": self.df.shape,
            "total_features": len(self.df.columns) - 1,
            "numeric_features_total": len(numeric_cols),
            "numeric_features_shown": len(numeric_meta),
            "categorical_features_total": len(categorical_cols),
            "categorical_features_shown": len(categorical_meta),
            "protected_cols": sorted(self._protected_cols),
            "numeric_features": numeric_meta,
            "categorical_features": categorical_meta,
        }

    def _build_col_desc_section(self) -> Optional[str]:
        """Return a group-organised column description block for the LLM prompt.

        Descriptions stored as "[group] text" are clustered by group so the LLM
        can easily identify which columns come from the same source table and
        therefore form natural interaction pairs.
        """
        if not self._col_descriptions:
            return None

        current_cols = set(self.df.columns) - self._protected_cols - {self.target_column}
        relevant = {c: d for c, d in self._col_descriptions.items() if c in current_cols}
        if not relevant:
            return None

        groups: Dict[str, List[str]] = {}
        ungrouped: List[str] = []
        for col, desc in sorted(relevant.items()):
            m = re.match(r"^\[([^\]]+)\]\s*(.*)", desc)
            if m:
                group_name, col_desc = m.group(1), m.group(2).strip()
                entry = f"  - {col}: {col_desc}" if col_desc else f"  - {col}"
                groups.setdefault(group_name, []).append(entry)
            else:
                ungrouped.append(f"  - {col}: {desc}" if desc else f"  - {col}")

        lines: List[str] = []
        for group_name, entries in sorted(groups.items()):
            lines.append(f"[{group_name}] — {len(entries)} columns:")
            lines.extend(entries[: Config.MAX_DESC_PER_GROUP])
            if len(entries) > Config.MAX_DESC_PER_GROUP:
                lines.append(f"  ... and {len(entries) - Config.MAX_DESC_PER_GROUP} more")
        if ungrouped:
            lines.append(f"(no group) — {len(ungrouped)} columns:")
            lines.extend(ungrouped[: Config.MAX_DESC_PER_GROUP])
            if len(ungrouped) > Config.MAX_DESC_PER_GROUP:
                lines.append(f"  ... and {len(ungrouped) - Config.MAX_DESC_PER_GROUP} more")

        return "\n".join(lines)

    def _suggest_top_k(self) -> int:
        exclude = self._protected_cols | ({self.target_column} if self.target_column else set())
        n = max(1, sum(1 for c in self.df.columns if c not in exclude))
        return max(10, min(int(n * Config.TOP_K_RATIO), Config.TOP_K_FEATURES_CAP))

    def _get_system_prompt(self) -> str:
        exclude = self._protected_cols | ({self.target_column} if self.target_column else set())
        n_engineerable = max(1, sum(1 for c in self.df.columns if c not in exclude))
        protected_display = str(sorted(self._protected_cols)) if self._protected_cols else "none"
        date_col_display = f"'{self._date_col}'" if self._date_col else "none"
        suggested_k = self._suggest_top_k()
        top_k_ratio_pct = int(Config.TOP_K_RATIO * 100)
        return self._load_prompt(
            self._PROMPTS_ROOT / "FeatureEngineer" / "prompts" / "system.txt",
            TOOL_DESCRIPTIONS=self.tool_registry.get_tool_descriptions(),
            MIN_CORRELATION_THRESHOLD=Config.MIN_CORRELATION_THRESHOLD,
            N_ENGINEERABLE_FEATURES=n_engineerable,
            TOP_K_RATIO_PCT=top_k_ratio_pct,
            TOP_K_FEATURES_CAP=Config.TOP_K_FEATURES_CAP,
            SUGGESTED_K=suggested_k,
            PROTECTED_COLS=protected_display,
            DATE_COL=date_col_display,
            DOMAIN_GUIDANCE=self._get_domain_guidance(),
        )

    def _build_engineering_prompt(self, analysis: Dict, previous_report: Dict) -> str:
        col_desc_block = self._build_col_desc_section()
        col_desc_section = (
            f"\nCOLUMN DESCRIPTIONS BY GROUP (format: [source_table] column_name: meaning):\n"
            f"{col_desc_block}\n"
            "Use the group names above to identify same-group vs cross-group interaction candidates.\n"
            if col_desc_block else ""
        )
        return self._load_prompt(
            self._PROMPTS_ROOT / "FeatureEngineer" / "prompts" / "user.txt",
            PREVIOUS_SUMMARY=previous_report.get("summary", "Data cleaning completed"),
            ANALYSIS_JSON=json.dumps(analysis, indent=2),
            COL_DESC_SECTION=col_desc_section,
            TARGET_COLUMN=self.target_column,
        )

    def _execute_llm_decisions(self, llm_response: str, spec: Optional["FeatureSpec"] = None) -> List[str]:
        """Apply LLM feature-engineering decisions AND populate `spec` for valid/oot replay.

        Captures into spec:
          - create_interaction → spec.interactions (new_col, expression, train_median fill)
          - encode_categorical (label) → spec.label_encoders[col]
          - encode_categorical (onehot) → spec.onehot_columns[col]
          - encode_all_categorical → spec.label_encoders / spec.onehot_columns (batch)
          - select_top_features → spec.selected_features
          - correlation_analysis: logged only, no transform recorded
        """
        if spec is None:
            spec = FeatureSpec(target_column=self.target_column)
        self.logger.log(self.name, "LLM Decision", "Parsing feature engineering decisions")
        actions_taken: List[str] = []

        try:
            decisions = json.loads(self._extract_json(llm_response))
            self.logger.log(self.name, "LLM Reasoning", decisions.get("reasoning", "No reasoning provided"))

            for action_spec in decisions.get("actions", []):
                action_type = action_spec.get("action")
                reason = action_spec.get("reason", "No reason provided")

                self.logger.log(self.name, f"Action: {action_type}", reason)

                try:
                    if action_type == "create_interaction":
                        new_col = action_spec.get("new_col")
                        expression = action_spec.get("expression", "")
                        # Block if expression references any protected column
                        refs_protected = any(f"'{c}'" in expression or f'"{c}"' in expression
                                            for c in self._protected_cols)
                        if refs_protected:
                            self.logger.log(self.name, f"SKIP {action_type}",
                                f"Expression references a protected column — skipped: {expression}")
                            continue
                        self._last_interaction_fill = 0.0
                        self.df = self.execute_tool("create_interaction", df=self.df,
                                                    new_col=new_col, expression=expression)
                        spec.interactions.append((new_col, expression, float(self._last_interaction_fill)))
                        actions_taken.append(f"Created feature '{new_col}': {reason}")

                    elif action_type == "encode_all_categorical":
                        method = action_spec.get("method", "label")
                        self._batch_label_encoders = {}
                        self._batch_onehot_cols = {}
                        self.df = self.execute_tool("encode_all_categorical", df=self.df, method=method)
                        spec.label_encoders.update(self._batch_label_encoders)
                        spec.onehot_columns.update(self._batch_onehot_cols)
                        actions_taken.append(f"Encoded all categorical columns with {method}: {reason}")

                    elif action_type == "encode_categorical":
                        column = action_spec.get("column")
                        if column in self._protected_cols:
                            self.logger.log(self.name, f"SKIP {action_type}",
                                f"'{column}' is a protected key column — skipped")
                            continue
                        method = action_spec.get("method", "label")
                        self._last_label_encoder = None
                        self._last_onehot_cols = None
                        self.df = self.execute_tool("encode_categorical", df=self.df,
                                                    col=column, method=method)
                        if method == "label" and self._last_label_encoder is not None:
                            spec.label_encoders[column] = self._last_label_encoder
                        elif method == "onehot" and self._last_onehot_cols is not None:
                            spec.onehot_columns[column] = self._last_onehot_cols
                        actions_taken.append(f"Encoded '{column}' with {method}: {reason}")

                    elif action_type == "correlation_analysis":
                        result = self.execute_tool("correlation_analysis", df=self.df, target=self.target_column)
                        actions_taken.append(f"Analyzed correlations: {reason}")
                        self.logger.log(self.name, "Correlation Results", result[:500])

                    elif action_type == "select_top_features":
                        k = int(action_spec.get("k", self._suggest_top_k()))
                        # Snapshot protected cols — SelectKBest drops them since they're non-numeric / not scored
                        protected_snapshot = {
                            col: self.df[col].copy()
                            for col in self._protected_cols
                            if col in self.df.columns
                        }
                        self._last_selected_features = None
                        self.df = self.execute_tool("select_top_features", df=self.df,
                                                    target=self.target_column, k=k)
                        # Re-add any protected cols that were removed by selection
                        restored: List[str] = []
                        for col, series in protected_snapshot.items():
                            if col not in self.df.columns:
                                self.df[col] = series.values
                                restored.append(col)
                                self.logger.log(self.name, "Protected Col Restored",
                                    f"Re-added '{col}' (composite key / entity ID) after feature selection")
                        # Capture the FINAL column list (including restored protected cols)
                        # so transform applies the same selection to valid/oot
                        spec.selected_features = list(self.df.columns)
                        actions_taken.append(f"Selected top {k} features: {reason}")

                    else:
                        self.logger.log(self.name, "WARN",
                            f"Unknown or missing action_type '{action_type}' — skipped")

                except Exception as action_err:
                    self.logger.log(self.name, f"SKIP {action_type}",
                        f"Action failed, continuing with next action: {action_err}")

        except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
            self.logger.log(self.name, "ERROR", f"Failed to parse LLM response: {e}")
            self.logger.log(self.name, "Raw Response", llm_response[:500])
            actions_taken.append("ERROR: Could not parse LLM decisions, performed basic encoding")
            self._fallback_engineering(spec)

        return actions_taken

    def _fallback_engineering(self, spec: Optional["FeatureSpec"] = None):
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        skip = self._protected_cols | {self.target_column}
        for col in categorical_cols:
            if col in skip:
                continue
            self._last_label_encoder = None
            self.df = self._tool_encode_categorical(self.df, col, "label")
            if spec is not None and self._last_label_encoder is not None:
                spec.label_encoders[col] = self._last_label_encoder
            self.logger.log(self.name, "Fallback", f"Label encoded {col}")

    def _generate_summary(self, actions: List[str]) -> str:
        if not actions:
            return "No feature engineering was necessary."
        preview = actions[:Config.SUMMARY_ACTION_PREVIEW]
        summary = f"Performed {len(actions)} feature engineering actions: " + "; ".join(preview)
        if len(actions) > Config.SUMMARY_ACTION_PREVIEW:
            summary += f"; and {len(actions) - Config.SUMMARY_ACTION_PREVIEW} more actions."
        return summary
