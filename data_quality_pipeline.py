"""
data_quality_pipeline.py
========================
Pre-training data quality checks with LLM-powered decision gates.

Architecture:
  Each check function collects statistics → formats a structured prompt
  → calls LLM → parses a structured decision: PASS | WARN | FAIL + reason.

Usage (PySpark):
    from data_quality_pipeline import DataQualityPipeline

    pipeline = DataQualityPipeline(
        df=spark_df,
        feature_cols=["age", "income", "score"],
        label_col="target",
        date_col="observation_date",
        anthropic_api_key="sk-ant-...",   # or set env ANTHROPIC_API_KEY
    )
    report = pipeline.run_all()
    print(report.summary())

Requirements:
    pip install pyspark anthropic
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import anthropic
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType, TimestampType, DateType


# ─────────────────────────────────────────────────────────────
# Decision model
# ─────────────────────────────────────────────────────────────

@dataclass
class CheckDecision:
    step: str
    verdict: str          # "PASS" | "WARN" | "FAIL"
    reason: str
    recommendation: str
    stats: dict[str, Any] = field(default_factory=dict)

    def is_blocking(self) -> bool:
        return self.verdict == "FAIL"

    def __repr__(self) -> str:
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(self.verdict, "?")
        return f"{icon} [{self.verdict}] {self.step}: {self.reason}"


@dataclass
class QualityReport:
    decisions: list[CheckDecision] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def overall(self) -> str:
        if any(d.verdict == "FAIL" for d in self.decisions):
            return "FAIL"
        if any(d.verdict == "WARN" for d in self.decisions):
            return "WARN"
        return "PASS"

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"DATA QUALITY REPORT  —  {self.created_at}",
            f"Overall: {self.overall()}",
            "=" * 60,
        ]
        for d in self.decisions:
            lines.append(repr(d))
            lines.append(f"   Recommendation: {d.recommendation}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "overall": self.overall(),
            "created_at": self.created_at,
            "checks": [
                {
                    "step": d.step,
                    "verdict": d.verdict,
                    "reason": d.reason,
                    "recommendation": d.recommendation,
                    "stats": d.stats,
                }
                for d in self.decisions
            ],
        }


# ─────────────────────────────────────────────────────────────
# LLM tool (single responsibility: call Claude and parse JSON)
# ─────────────────────────────────────────────────────────────

class LLMDecisionTool:
    """
    Wraps the Anthropic Claude API.
    Every check passes a structured stats dict; LLM returns JSON:
        { "verdict": "PASS|WARN|FAIL", "reason": "...", "recommendation": "..." }
    """

    SYSTEM_PROMPT = textwrap.dedent("""
        You are a senior ML data engineer reviewing data quality statistics
        before model training. You must respond ONLY with a valid JSON object
        — no markdown, no preamble — with exactly these keys:
          "verdict"        : "PASS" | "WARN" | "FAIL"
          "reason"         : one-sentence explanation
          "recommendation" : one concrete action (max 20 words)

        Decision thresholds guidance:
          PASS  — data is clean, no action needed
          WARN  — issue exists but training can proceed with caution
          FAIL  — issue is severe enough to block training
    """).strip()

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
        )
        self.model = model

    def decide(self, step_name: str, stats: dict[str, Any]) -> CheckDecision:
        user_msg = (
            f"Check step: {step_name}\n\n"
            f"Statistics:\n{json.dumps(stats, indent=2, default=str)}"
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)

        return CheckDecision(
            step=step_name,
            verdict=parsed["verdict"].upper(),
            reason=parsed["reason"],
            recommendation=parsed["recommendation"],
            stats=stats,
        )


# ─────────────────────────────────────────────────────────────
# Individual check tools
# ─────────────────────────────────────────────────────────────

class SchemaCheckTool:
    """
    Check 1 — Schema & data types
    Validates column names, dtypes, and row count.
    """

    name = "schema_and_dtypes"

    def __init__(self, df: DataFrame, feature_cols: list[str], label_col: str, date_col: str):
        self.df = df
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.date_col = date_col

    def collect_stats(self) -> dict[str, Any]:
        all_expected = self.feature_cols + [self.label_col, self.date_col]
        actual_cols = self.df.columns
        missing_cols = [c for c in all_expected if c not in actual_cols]
        unexpected_cols = [c for c in actual_cols if c not in all_expected]

        dtype_map = {name: str(dtype) for name, dtype in self.df.dtypes}
        numeric_features = [
            c for c in self.feature_cols
            if isinstance(self.df.schema[c].dataType, NumericType)
        ]
        non_numeric_features = [
            c for c in self.feature_cols
            if c not in numeric_features
        ]
        date_type_ok = isinstance(
            self.df.schema[self.date_col].dataType, (TimestampType, DateType)
        ) if self.date_col in actual_cols else False

        return {
            "total_rows": self.df.count(),
            "total_columns": len(actual_cols),
            "missing_expected_columns": missing_cols,
            "unexpected_columns": unexpected_cols,
            "numeric_feature_count": len(numeric_features),
            "non_numeric_feature_count": len(non_numeric_features),
            "non_numeric_features": non_numeric_features,
            "date_col_type_ok": date_type_ok,
            "dtypes": dtype_map,
        }


class MissingValueCheckTool:
    """
    Check 2 — Missing values & duplicates
    """

    name = "missing_values_and_duplicates"

    def __init__(self, df: DataFrame, feature_cols: list[str], label_col: str):
        self.df = df
        self.feature_cols = feature_cols
        self.label_col = label_col

    def collect_stats(self) -> dict[str, Any]:
        total = self.df.count()
        all_cols = self.feature_cols + [self.label_col]

        null_counts = self.df.select(
            [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in all_cols]
        ).collect()[0].asDict()

        null_pct = {c: round(null_counts[c] / total * 100, 2) for c in all_cols}
        high_null_cols = {c: p for c, p in null_pct.items() if p > 20}

        duplicate_count = total - self.df.dropDuplicates().count()

        return {
            "total_rows": total,
            "null_percentage_per_column": null_pct,
            "columns_with_high_nulls_above_20pct": high_null_cols,
            "duplicate_rows": duplicate_count,
            "duplicate_percentage": round(duplicate_count / total * 100, 2),
        }


class LabelQualityCheckTool:
    """
    Check 3 — Label quality & class balance
    """

    name = "label_quality_and_class_balance"

    def __init__(self, df: DataFrame, label_col: str):
        self.df = df
        self.label_col = label_col

    def collect_stats(self) -> dict[str, Any]:
        total = self.df.count()
        label_counts = (
            self.df.groupBy(self.label_col)
            .count()
            .orderBy(F.desc("count"))
            .collect()
        )

        counts = {str(row[self.label_col]): row["count"] for row in label_counts}
        null_labels = self.df.filter(F.col(self.label_col).isNull()).count()

        values = list(counts.values())
        imbalance_ratio = round(max(values) / min(values), 2) if len(values) > 1 else 1.0
        num_classes = len(counts)

        return {
            "total_rows": total,
            "num_classes": num_classes,
            "class_distribution": counts,
            "null_labels": null_labels,
            "null_label_percentage": round(null_labels / total * 100, 2),
            "imbalance_ratio": imbalance_ratio,
            "note": (
                "Imbalance ratio = max_class_count / min_class_count. "
                ">10 is typically problematic."
            ),
        }


class TemporalIntegrityCheckTool:
    """
    Check 4 — Temporal integrity
    Detects date gaps, ordering issues, and potential future leakage.
    """

    name = "temporal_integrity"

    def __init__(self, df: DataFrame, date_col: str, label_col: str, feature_cols: list[str]):
        self.df = df
        self.date_col = date_col
        self.label_col = label_col
        self.feature_cols = feature_cols

    def collect_stats(self) -> dict[str, Any]:
        date_stats = self.df.select(
            F.min(self.date_col).alias("min_date"),
            F.max(self.date_col).alias("max_date"),
            F.countDistinct(self.date_col).alias("distinct_dates"),
            F.count("*").alias("total_rows"),
        ).collect()[0]

        # Future dates relative to today
        future_rows = self.df.filter(
            F.col(self.date_col) > F.current_date()
        ).count()

        # Check for potential feature-leakage columns (contain "future", "next", "target")
        leakage_suspects = [
            c for c in self.feature_cols
            if any(kw in c.lower() for kw in ["future", "next", "forward", "ahead", "leakage"])
        ]

        # Check temporal ordering (are rows roughly sorted by date?)
        sample = self.df.select(self.date_col).limit(1000).toPandas()
        dates_sorted = sample[self.date_col].dropna().tolist()
        is_sorted = dates_sorted == sorted(dates_sorted)

        return {
            "min_date": str(date_stats["min_date"]),
            "max_date": str(date_stats["max_date"]),
            "date_range_days": (
                (date_stats["max_date"] - date_stats["min_date"]).days
                if date_stats["min_date"] and date_stats["max_date"]
                else None
            ),
            "distinct_dates": date_stats["distinct_dates"],
            "total_rows": date_stats["total_rows"],
            "rows_with_future_dates": future_rows,
            "rows_sorted_by_date_in_sample": is_sorted,
            "potential_leakage_column_names": leakage_suspects,
        }


class FeatureDistributionCheckTool:
    """
    Check 5 — Feature distributions, outliers, skew, high cardinality
    """

    name = "feature_distributions"

    def __init__(self, df: DataFrame, feature_cols: list[str]):
        self.df = df
        self.feature_cols = feature_cols

    def collect_stats(self) -> dict[str, Any]:
        numeric_cols = [
            c for c in self.feature_cols
            if isinstance(self.df.schema[c].dataType, NumericType)
        ]
        string_cols = [
            c for c in self.feature_cols
            if isinstance(self.df.schema[c].dataType, StringType)
        ]

        # Numeric stats
        numeric_stats = {}
        if numeric_cols:
            stats_df = self.df.select(numeric_cols).describe().collect()
            for row in stats_df:
                numeric_stats[row["summary"]] = {
                    c: row[c] for c in numeric_cols
                }

        # High cardinality detection for string cols
        cardinality = {}
        for c in string_cols:
            cardinality[c] = self.df.select(c).distinct().count()

        high_cardinality = {c: v for c, v in cardinality.items() if v > 100}

        # Outlier detection via IQR on numeric cols (sampled)
        outlier_flags: dict[str, float] = {}
        total = self.df.count()
        for c in numeric_cols[:10]:   # cap at 10 for performance
            q = self.df.approxQuantile(c, [0.01, 0.25, 0.75, 0.99], 0.05)
            if len(q) == 4:
                iqr = q[2] - q[1]
                lower, upper = q[1] - 3 * iqr, q[2] + 3 * iqr
                outlier_count = self.df.filter(
                    (F.col(c) < lower) | (F.col(c) > upper)
                ).count()
                outlier_flags[c] = round(outlier_count / total * 100, 2)

        return {
            "numeric_column_count": len(numeric_cols),
            "string_column_count": len(string_cols),
            "numeric_summary_stats": numeric_stats,
            "string_cardinality": cardinality,
            "high_cardinality_columns_above_100": high_cardinality,
            "outlier_percentage_per_column_iqr_3x": outlier_flags,
        }


# ─────────────────────────────────────────────────────────────
# Main pipeline orchestrator
# ─────────────────────────────────────────────────────────────

class DataQualityPipeline:
    """
    Orchestrates all checks and calls the LLM after each step.
    After all checks, a final LLM call aggregates everything
    into a single PASS / WARN / FAIL verdict.

    Example
    -------
    >>> pipeline = DataQualityPipeline(
    ...     df=spark_df,
    ...     feature_cols=["age", "income", "score"],
    ...     label_col="target",
    ...     date_col="observation_date",
    ... )
    >>> report = pipeline.run_all()
    >>> print(report.summary())
    """

    def __init__(
        self,
        df: DataFrame,
        feature_cols: list[str],
        label_col: str,
        date_col: str,
        anthropic_api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        fail_fast: bool = False,
    ):
        self.df = df
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.date_col = date_col
        self.fail_fast = fail_fast

        self.llm = LLMDecisionTool(api_key=anthropic_api_key, model=model)
        self.report = QualityReport()

    # ── individual check runners ──────────────────────────────

    def check_schema(self) -> CheckDecision:
        tool = SchemaCheckTool(self.df, self.feature_cols, self.label_col, self.date_col)
        stats = tool.collect_stats()
        decision = self.llm.decide(tool.name, stats)
        self.report.decisions.append(decision)
        print(repr(decision))
        return decision

    def check_missing_values(self) -> CheckDecision:
        tool = MissingValueCheckTool(self.df, self.feature_cols, self.label_col)
        stats = tool.collect_stats()
        decision = self.llm.decide(tool.name, stats)
        self.report.decisions.append(decision)
        print(repr(decision))
        return decision

    def check_label_quality(self) -> CheckDecision:
        tool = LabelQualityCheckTool(self.df, self.label_col)
        stats = tool.collect_stats()
        decision = self.llm.decide(tool.name, stats)
        self.report.decisions.append(decision)
        print(repr(decision))
        return decision

    def check_temporal_integrity(self) -> CheckDecision:
        tool = TemporalIntegrityCheckTool(
            self.df, self.date_col, self.label_col, self.feature_cols
        )
        stats = tool.collect_stats()
        decision = self.llm.decide(tool.name, stats)
        self.report.decisions.append(decision)
        print(repr(decision))
        return decision

    def check_feature_distributions(self) -> CheckDecision:
        tool = FeatureDistributionCheckTool(self.df, self.feature_cols)
        stats = tool.collect_stats()
        decision = self.llm.decide(tool.name, stats)
        self.report.decisions.append(decision)
        print(repr(decision))
        return decision

    def _final_gate(self) -> CheckDecision:
        """
        Final LLM call: aggregates all check results and issues the
        overall go / no-go verdict for model training.
        """
        aggregate_stats = {
            "check_results": [
                {
                    "step": d.step,
                    "verdict": d.verdict,
                    "reason": d.reason,
                    "recommendation": d.recommendation,
                }
                for d in self.report.decisions
            ],
            "fail_count": sum(1 for d in self.report.decisions if d.verdict == "FAIL"),
            "warn_count": sum(1 for d in self.report.decisions if d.verdict == "WARN"),
            "pass_count": sum(1 for d in self.report.decisions if d.verdict == "PASS"),
            "question": (
                "Based on ALL checks above, should we proceed to model training? "
                "PASS = safe to train; WARN = train with caution; FAIL = block training."
            ),
        }
        decision = self.llm.decide("final_gate", aggregate_stats)
        self.report.decisions.append(decision)
        return decision

    # ── main runner ───────────────────────────────────────────

    def run_all(self) -> QualityReport:
        """Run all checks sequentially. Stop early on FAIL if fail_fast=True."""
        checks = [
            self.check_schema,
            self.check_missing_values,
            self.check_label_quality,
            self.check_temporal_integrity,
            self.check_feature_distributions,
        ]
        for check_fn in checks:
            decision = check_fn()
            if self.fail_fast and decision.is_blocking():
                print(f"\n⛔  fail_fast=True — stopping after blocking check: {decision.step}")
                break

        print("\n─── Running final LLM gate ───")
        self._final_gate()
        return self.report

    def run_check(self, name: str) -> CheckDecision:
        """Run a single check by name (useful for re-running after fixes)."""
        registry = {
            "schema": self.check_schema,
            "missing": self.check_missing_values,
            "label": self.check_label_quality,
            "temporal": self.check_temporal_integrity,
            "distribution": self.check_feature_distributions,
        }
        if name not in registry:
            raise ValueError(f"Unknown check: {name!r}. Choose from {list(registry)}")
        return registry[name]()


# ─────────────────────────────────────────────────────────────
# Quick-start example (local Spark for demo)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    import random
    from datetime import date, timedelta

    spark = SparkSession.builder.master("local").appName("dq_demo").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # ── generate a synthetic dataset ──
    random.seed(42)
    n = 2000
    rows = []
    base_date = date(2023, 1, 1)
    for i in range(n):
        obs_date = base_date + timedelta(days=random.randint(0, 365))
        rows.append({
            "age":              random.gauss(35, 12),
            "income":           random.gauss(50000, 20000),
            "credit_score":     random.randint(300, 850) if random.random() > 0.05 else None,
            "product_category": random.choice(["A", "B", "C", None]),
            "observation_date": obs_date,
            "target":           random.choice([0, 0, 0, 1]),   # 75/25 imbalance
        })

    df = spark.createDataFrame(rows)

    pipeline = DataQualityPipeline(
        df=df,
        feature_cols=["age", "income", "credit_score", "product_category"],
        label_col="target",
        date_col="observation_date",
        # anthropic_api_key="sk-ant-..."  # or set ANTHROPIC_API_KEY env var
        fail_fast=False,
    )

    report = pipeline.run_all()
    print("\n" + report.summary())

    # Save report as JSON
    import json
    with open("dq_report.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    print("\nReport saved to dq_report.json")
