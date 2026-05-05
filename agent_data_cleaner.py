import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
from base_agent import BaseAgent, ToolRegistry
from logger import AgentLogger
from config import Config


class DataCleanerAgent(BaseAgent):

    def __init__(self, logger: AgentLogger):
        super().__init__(name="DataCleaner", role="Data Quality Auditor", logger=logger)
        self.df: pd.DataFrame = None
        self.tool_registry = ToolRegistry()
        self._register_tools()

    def _register_tools(self):
        self.tool_registry.register(
            "inspect_metadata",
            "Returns dataset shape, column data types, and null counts",
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

    def _tool_inspect_metadata(self, df: pd.DataFrame) -> str:
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "null_percentages": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
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

    def process(self, input_csv: str) -> Tuple[str, Dict[str, Any]]:
        """Returns (path_to_clean_data, report_dict)."""
        self.logger.log(self.name, "Process Start", f"Loading data from {input_csv}")

        self.df = pd.read_csv(input_csv)
        original_shape = self.df.shape

        metadata = self.execute_tool("inspect_metadata", df=self.df)
        llm_response = self.call_llm(
            self._build_analysis_prompt(metadata),
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
        }
        self.save_report(report, Config.DATA_CLEANER_REPORT_PATH)

        self.logger.log(self.name, "Process Complete", f"Shape: {original_shape} -> {self.df.shape}")
        return Config.CLEAN_DATA_PATH, report

    def _get_system_prompt(self) -> str:
        return f"""You are the Data Cleaner Agent, an expert data quality auditor.

Your role: Inspect the raw dataset and make decisions about cleaning actions.

Available Tools:
{self.tool_registry.get_tool_descriptions()}

Your task:
1. Analyze the metadata provided
2. Decide which columns need cleaning or should be dropped
3. For each problematic column, decide the best action
4. Output your decisions in a structured format

Guidelines:
- Drop columns with >{Config.NULL_DROP_THRESHOLD * 100:.0f}% missing values (unless they seem important)
- Drop ID columns or columns with all unique values (no predictive power)
- For numeric columns with missing values, prefer median imputation
- For categorical columns with missing values, prefer mode imputation
- Consider data types: are they correct?

Output Format (JSON):
{{
  "reasoning": "Your analysis of the data quality issues",
  "actions": [
    {{"action": "drop_column", "column": "id", "reason": "Unique identifier with no predictive value"}},
    {{"action": "impute_missing", "column": "age", "strategy": "median", "reason": "20% missing numeric values"}},
    {{"action": "impute_missing", "column": "category", "strategy": "mode", "reason": "15% missing categorical values"}}
  ]
}}

Be decisive but explain your reasoning clearly."""

    def _build_analysis_prompt(self, metadata: str) -> str:
        return f"""Analyze this dataset and decide what cleaning actions to take:

DATASET METADATA:
{metadata}

Based on this information, what cleaning actions should be performed?
Provide your response in the JSON format specified."""

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
