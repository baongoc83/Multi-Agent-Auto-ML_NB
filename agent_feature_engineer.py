import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, VarianceThreshold
from base_agent import BaseAgent, ToolRegistry
from logger import AgentLogger
from config import Config


class FeatureEngineerAgent(BaseAgent):

    def __init__(self, logger: AgentLogger, col_descriptions_path: Optional[str] = None):
        super().__init__(name="FeatureEngineer", role="Feature Architect", logger=logger)
        self.df: pd.DataFrame = None
        self.target_column: str = None
        self._protected_cols: set = set()
        self._col_descriptions: Dict[str, str] = self._load_col_descriptions(col_descriptions_path)
        self.tool_registry = ToolRegistry()
        self._register_tools()

    def _load_col_descriptions(self, path: Optional[str]) -> Dict[str, str]:
        if not path:
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
            self.logger.log(self.name, "WARN", f"col_descriptions file must be a JSON object, got {type(data).__name__}")
        except Exception as e:
            self.logger.log(self.name, "WARN", f"Could not load col_descriptions from '{path}': {e}")
        return {}

    def _register_tools(self):
        self.tool_registry.register(
            "create_interaction",
            "Creates a new column using mathematical expressions between columns",
            {
                "df": "The dataframe",
                "new_col": "Name for the new column",
                "expression": "Python expression using df['col'] syntax, e.g. \"df['a'] / df['b']\"",
            },
        )
        self.tool_registry.register(
            "encode_categorical",
            "Encodes categorical columns into numeric format",
            {
                "df": "The dataframe",
                "col": "Column name to encode",
                "method": "Either 'label' (ordinal) or 'onehot' (binary columns)",
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
            df[new_col] = df[new_col].fillna(df[new_col].median())
            # Drop if constant or all-NaN — causes divide-by-zero in correlation/SelectKBest
            if df[new_col].isna().all() or df[new_col].std() == 0:
                df = df.drop(columns=[new_col])
                raise ValueError(f"Generated feature '{new_col}' is constant or all-NaN after fill — dropped")
            return df
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Error creating interaction '{new_col}': {e}")

    def _tool_encode_categorical(self, df: pd.DataFrame, col: str, method: str) -> pd.DataFrame:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")

        if method == "label":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        elif method == "onehot":
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
        else:
            raise ValueError(f"Unknown encoding method: {method}")

        return df

    def _tool_correlation_analysis(self, df: pd.DataFrame, target: str) -> str:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)

        correlations = {}
        unreliable = []  # columns where null_pct > 30% — correlation may be biased
        for col in numeric_cols:
            null_pct = df[col].isnull().mean() * 100
            # Skip zero-variance columns — corr() divides by std=0 → RuntimeWarning + NaN
            if df[col].std() == 0:
                correlations[col] = 0.0
                continue
            # pandas corr() drops NaN pairwise — works but biased for high-null cols
            corr = df[col].corr(df[target])
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
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        X = df.drop(columns=[target])
        y = df[target]

        # Non-numeric columns cannot be scored by SelectKBest — always preserve them
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Drop rows where target is null — sklearn cannot handle NaN in y
        valid_mask = y.notna()
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        numeric_features = X_valid.select_dtypes(include=[np.number]).columns
        X_numeric = X_valid[numeric_features]

        if len(numeric_features) == 0:
            # No numeric features to select — keep everything
            return df

        # Impute NaN with column median for SelectKBest only — original df is unchanged
        X_for_selection = X_numeric.fillna(X_numeric.median())

        # Remove zero-variance columns before SelectKBest — they cause divide-by-zero warnings
        vt = VarianceThreshold(threshold=0.0)
        X_for_selection = pd.DataFrame(
            vt.fit_transform(X_for_selection),
            columns=X_for_selection.columns[vt.get_support()],
        )
        non_zero_var_features = X_for_selection.columns

        k = min(k, len(non_zero_var_features))
        if k == 0:
            # All numeric features had zero variance — keep non-numeric cols + target only
            final_cols = non_numeric_cols + [target]
            return df[final_cols]

        is_classification = y_valid.nunique() < Config.CLASSIFICATION_UNIQUE_THRESHOLD
        score_func = f_classif if is_classification else f_regression

        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X_for_selection, y_valid)

        selected_numeric = non_zero_var_features[selector.get_support()].tolist()
        # Preserve all non-numeric cols + selected numeric cols + target
        final_cols = selected_numeric + non_numeric_cols + [target]
        return df[final_cols]

    def process(
        self,
        df: pd.DataFrame,
        previous_report: Dict[str, Any],
        target_column: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Returns (path_to_engineered_data, report_dict)."""
        self.logger.log(self.name, "Process Start", f"Received clean data with shape {df.shape}")

        self.df = df.copy()
        self.target_column = self._resolve_target_column(self.df, target_column)
        original_shape = self.df.shape

        # Extract key columns flagged by DataCleaner — skip these in all FE steps
        self._protected_cols = set(
            previous_report.get("composite_key_cols", [])
        )
        if previous_report.get("entity_id_col"):
            self._protected_cols.add(previous_report["entity_id_col"])
        self._protected_cols.discard(self.target_column)

        if self._protected_cols:
            self.logger.log(self.name, "Protected Cols",
                f"Skipping feature engineering for: {sorted(self._protected_cols)}")

        self.logger.log(self.name, "Previous Agent Summary", previous_report.get("summary", "No summary"))

        analysis = self._analyze_features()
        llm_response = self.call_llm(
            self._build_engineering_prompt(analysis, previous_report),
            self._get_system_prompt(),
        )
        actions_taken = self._execute_llm_decisions(llm_response)

        self.df.to_csv(Config.ENGINEERED_DATA_PATH, index=False)
        self.logger.log(self.name, "Data Saved", f"Engineered data saved to {Config.ENGINEERED_DATA_PATH}")

        report = {
            "agent": self.name,
            "original_shape": original_shape,
            "final_shape": self.df.shape,
            "actions_taken": actions_taken,
            "summary": self._generate_summary(actions_taken),
            "final_features": list(self.df.columns),
            # Forward key column info from DataCleaner for downstream agents
            "entity_id_col": previous_report.get("entity_id_col"),
            "composite_key_cols": previous_report.get("composite_key_cols", []),
        }
        self.save_report(report, Config.FEATURE_ENGINEER_REPORT_PATH)

        self.logger.log(self.name, "Process Complete", f"Shape: {original_shape} -> {self.df.shape}")
        return Config.ENGINEERED_DATA_PATH, report

    def _resolve_target_column(self, df: pd.DataFrame, target_column: str) -> str:
        """Resolve target column name — handles case differences and common aliases."""
        if target_column in df.columns:
            return target_column

        # Case-insensitive match
        lower_map = {c.lower(): c for c in df.columns}
        if target_column.lower() in lower_map:
            resolved = lower_map[target_column.lower()]
            self.logger.log(self.name, "Target Resolved",
                f"'{target_column}' → '{resolved}' (case-insensitive match)")
            return resolved

        # Try common target column names
        _common = ["target", "label", "y", "class", "output",
                   "default_flag", "fraud", "is_fraud", "churn", "bad_flag"]
        for name in _common:
            if name in lower_map:
                resolved = lower_map[name]
                self.logger.log(self.name, "Target Resolved",
                    f"'{target_column}' not found → using '{resolved}' (common target name)")
                return resolved

        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    def _analyze_features(self) -> Dict[str, Any]:
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

        exclude = self._protected_cols | {self.target_column}
        numeric_cols = [c for c in numeric_cols if c not in exclude]
        categorical_cols = [c for c in categorical_cols if c not in exclude]

        numeric_meta: Dict[str, Any] = {}
        for col in numeric_cols[:Config.FEATURE_META_MAX_COLS]:
            s = self.df[col]
            entry: Dict[str, Any] = {
                "null_pct": round(float(s.isnull().mean() * 100), 2),
                "nunique": int(s.nunique()),
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "skew": round(float(s.skew()), 4),
            }
            if col in self._col_descriptions:
                entry["description"] = self._col_descriptions[col]
            numeric_meta[col] = entry

        categorical_meta: Dict[str, Any] = {}
        for col in categorical_cols[:Config.FEATURE_META_MAX_COLS]:
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
            "protected_cols": sorted(self._protected_cols),
            "numeric_features": numeric_meta,
            "categorical_features": categorical_meta,
        }

    def _get_system_prompt(self) -> str:
        return f"""You are the Feature Engineer Agent, an expert in creating predictive features.

Your role: Receive cleaned data and create new features to maximize model performance.

Available Tools:
{self.tool_registry.get_tool_descriptions()}

Your task:
1. Analyze the current features
2. Create new interaction features (ratios, products, etc.)
3. Encode categorical variables appropriately
4. Select the most predictive features
5. Output your decisions in a structured format

Guidelines:
Reading column metadata before creating interactions:
- Each numeric column has: null_pct, nunique, mean, std, min, max, skew
- Each categorical column has: null_pct, nunique, top_values

Use metadata to make smart decisions:
- Skip interactions if null_pct > 30% (too many missing values, result will be noisy)
- If |skew| > 2: apply log1p before using in ratio → expression: "np.log1p(df['col'].clip(lower=0))"
- If min >= 0 and the column is a count/amount: safe to use as denominator (add +1 to avoid div/0)
- If nunique == 2: column is binary — consider product interactions instead of ratios
- For categorical: use onehot if nunique <= 5 (low cardinality), label if nunique > 5

Interaction ideas based on domain:
- Ratios: debt/income, loan_amount/(income+1), overdue_count/(total_count+1)
- Products: amount * rate, months * monthly_payment
- Differences: age - account_age, limit - balance

Encoding:
- Use label encoding for ordinal categories (low/mid/high, grades)
- Use one-hot for nominal categories (region, product_type) with nunique <= 5

Filtering:
- Remove features with |correlation| to target < {Config.MIN_CORRELATION_THRESHOLD}
- Keep feature count reasonable (prefer {Config.TARGET_FEATURE_COUNT_MIN}-{Config.TARGET_FEATURE_COUNT_MAX} final features)
- PROTECTED COLUMNS (DO NOT engineer, encode, or use in interactions): {sorted(self._protected_cols) if self._protected_cols else "none"}
  These are composite key / entity ID columns identified by the DataCleaner agent.

Output Format (JSON):
{{
  "reasoning": "Your strategy and why these features will help",
  "actions": [
    {{"action": "create_interaction", "new_col": "income_per_age", "expression": "df['income'] / (df['age'] + 1)", "reason": "Normalize income by age"}},
    {{"action": "encode_categorical", "column": "category", "method": "onehot", "reason": "Nominal variable needs one-hot encoding"}},
    {{"action": "correlation_analysis", "reason": "Check which features correlate with target"}},
    {{"action": "select_top_features", "k": {Config.DEFAULT_TOP_K_FEATURES}, "reason": "Keep only most predictive features"}}
  ]
}}

Be creative but practical. Focus on features that make logical sense."""

    def _build_engineering_prompt(self, analysis: Dict, previous_report: Dict) -> str:
        return f"""You have received cleaned data from the Data Cleaner agent.

PREVIOUS AGENT'S WORK:
{previous_report.get('summary', 'Data cleaning completed')}

CURRENT FEATURE ANALYSIS:
{json.dumps(analysis, indent=2)}

TARGET COLUMN: {self.target_column}

Based on this information, what feature engineering actions should you perform?
Think about:
- What new features could be informative?
- Which categorical variables need encoding?
- Are there redundant features to remove?

Provide your response in the JSON format specified."""

    def _execute_llm_decisions(self, llm_response: str) -> List[str]:
        self.logger.log(self.name, "LLM Decision", "Parsing feature engineering decisions")
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
                reason = action_spec.get("reason", "No reason provided")

                self.logger.log(self.name, f"Action: {action_type}", reason)

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
                    self.df = self.execute_tool("create_interaction", df=self.df, new_col=new_col, expression=expression)
                    actions_taken.append(f"Created feature '{new_col}': {reason}")

                elif action_type == "encode_categorical":
                    column = action_spec.get("column")
                    if column in self._protected_cols:
                        self.logger.log(self.name, f"SKIP {action_type}",
                            f"'{column}' is a protected key column — skipped")
                        continue
                    method = action_spec.get("method", "label")
                    self.df = self.execute_tool("encode_categorical", df=self.df, col=column, method=method)
                    actions_taken.append(f"Encoded '{column}' with {method}: {reason}")

                elif action_type == "correlation_analysis":
                    result = self.execute_tool("correlation_analysis", df=self.df, target=self.target_column)
                    actions_taken.append(f"Analyzed correlations: {reason}")
                    self.logger.log(self.name, "Correlation Results", result[:500])

                elif action_type == "select_top_features":
                    k = action_spec.get("k", Config.DEFAULT_TOP_K_FEATURES)
                    # Snapshot protected cols — SelectKBest drops them since they're non-numeric / not scored
                    protected_snapshot = {
                        col: self.df[col].copy()
                        for col in self._protected_cols
                        if col in self.df.columns
                    }
                    self.df = self.execute_tool("select_top_features", df=self.df, target=self.target_column, k=k)
                    # Re-add any protected cols that were removed by selection
                    for col, series in protected_snapshot.items():
                        if col not in self.df.columns:
                            self.df[col] = series.values
                            self.logger.log(self.name, "Protected Col Restored",
                                f"Re-added '{col}' (composite key / entity ID) after feature selection")
                    actions_taken.append(f"Selected top {k} features: {reason}")

        except json.JSONDecodeError as e:
            self.logger.log(self.name, "ERROR", f"Failed to parse LLM response: {e}")
            self.logger.log(self.name, "Raw Response", llm_response[:500])
            actions_taken.append("ERROR: Could not parse LLM decisions, performed basic encoding")
            self._fallback_engineering()

        return actions_taken

    def _fallback_engineering(self):
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        skip = self._protected_cols | {self.target_column}
        for col in categorical_cols:
            if col in skip:
                continue
            self.df = self._tool_encode_categorical(self.df, col, "label")
            self.logger.log(self.name, "Fallback", f"Label encoded {col}")

    def _generate_summary(self, actions: List[str]) -> str:
        if not actions:
            return "No feature engineering was necessary."
        preview = actions[:Config.SUMMARY_ACTION_PREVIEW]
        summary = f"Performed {len(actions)} feature engineering actions: " + "; ".join(preview)
        if len(actions) > Config.SUMMARY_ACTION_PREVIEW:
            summary += f"; and {len(actions) - Config.SUMMARY_ACTION_PREVIEW} more actions."
        return summary
