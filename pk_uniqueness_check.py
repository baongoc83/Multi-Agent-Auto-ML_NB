"""
pk_uniqueness_check.py
======================
Check 2 (extended) — PRIMARY KEY & UNIQUENESS CHECK
Covers three sub-checks, critical for banking / fraud ML datasets:

  2.1  Duplicate records
       ├─ Exact duplicate  : byte-for-byte identical rows
       ├─ Soft duplicate   : same composite key, different feature values
       └─ Duplicated app   : same entity_id appearing N times (policy check)

  2.2  Composite key validation
       ├─ (entity_id + snapshot_date) must be unique
       ├─ Label contamination: same key → different labels → training signal noise
       └─ Double-count risk: same key → different feature values → aggregation error

  2.3  Entity resolution
       ├─ Forward  : one customer → many CIFs / phones / devices / national IDs
       ├─ Reverse  : one CIF / phone / device → many customers (fragmented graph)
       └─ Orphan   : customers with NO identity anchor across all identity columns

Designed for PySpark DataFrames. Works standalone or as part of DataQualityPipeline.

Usage
-----
    from pk_uniqueness_check import PKCheckConfig, PKUniquenessCheckTool

    config = PKCheckConfig(
        entity_id_col     = "customer_id",
        snapshot_date_col = "snapshot_date",
        composite_key_cols= ["customer_id", "snapshot_date"],
        label_col         = "default_flag",
        identity_cols     = {
            "cif"        : "cif_id",
            "phone"      : "phone_number",
            "device"     : "device_id",
            "national_id": "national_id",
        },
    )
    tool  = PKUniquenessCheckTool(df, config)
    stats = tool.collect_stats()       # pure PySpark, no LLM
    # → pass stats to LLMDecisionTool.decide(tool.name, stats)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window


# ─────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class PKCheckConfig:
    """
    All parameters for PK & uniqueness checks.

    Parameters
    ----------
    entity_id_col : str
        Primary entity identifier, e.g. ``"customer_id"``.
    snapshot_date_col : str
        Observation / snapshot date column, e.g. ``"snapshot_date"``.
    composite_key_cols : list[str]
        Columns that together must be unique, e.g.
        ``["customer_id", "snapshot_date"]``.
    label_col : str | None
        Target column — used to detect label contamination within duplicate keys.
    identity_cols : dict[str, str] | None
        Mapping of logical name → actual column name for entity resolution.
        Example: ``{"cif": "cif_id", "phone": "phone_number", "device": "device_id",
                    "national_id": "national_id"}``.
        Only columns that exist in the DataFrame will be checked.
    soft_dup_subset : list[str] | None
        Subset of columns for soft-duplicate detection.
        Defaults to ``composite_key_cols``.
    app_dup_max_allowed : int
        Maximum number of rows per entity before flagging as "duplicated application".
        Default 1 (each entity may appear only once).
    """
    entity_id_col     : str
    snapshot_date_col : str
    composite_key_cols: list[str]
    label_col         : str | None              = None
    identity_cols     : dict[str, str] | None   = None
    soft_dup_subset   : list[str] | None        = None
    app_dup_max_allowed: int                    = 1


# ─────────────────────────────────────────────────────────────
# 2.1  Duplicate record helpers
# ─────────────────────────────────────────────────────────────

class _DuplicateSubCheck:
    """
    Runs all three duplicate variants and returns a nested stats dict.
    Equivalent SQL shown as comments for auditability.
    """

    def __init__(self, df: DataFrame, config: PKCheckConfig):
        self.df     = df
        self.config = config
        self._total = df.count()

    # ── 2.1a  Exact duplicates ────────────────────────────────

    def _exact_dup_stats(self) -> dict[str, Any]:
        """
        Equivalent SQL:
            SELECT COUNT(*) - COUNT(DISTINCT *) AS exact_dup_rows FROM tbl;
        Spark has no COUNT(DISTINCT *) so we use dropDuplicates().
        """
        unique_count   = self.df.dropDuplicates().count()
        exact_dup_rows = self._total - unique_count
        return {
            "exact_duplicate_rows"     : exact_dup_rows,
            "exact_duplicate_pct"      : _pct(exact_dup_rows, self._total),
            "unique_rows_after_dedup"  : unique_count,
        }

    # ── 2.1b  Soft duplicates ─────────────────────────────────

    def _soft_dup_stats(self) -> dict[str, Any]:
        """
        Rows sharing the same composite key but differing in at least one
        feature column — signals ETL merge errors or multiple extracts.

        Equivalent SQL:
            SELECT <composite_key_cols>, COUNT(*) AS n
            FROM tbl
            GROUP BY <composite_key_cols>
            HAVING COUNT(*) > 1;
        """
        subset = self.config.soft_dup_subset or self.config.composite_key_cols

        # Guard: only check columns that exist
        subset = [c for c in subset if c in self.df.columns]
        if not subset:
            return {"soft_duplicate_check": "skipped — no valid subset columns"}

        dup_keys = (
            self.df
            .groupBy(subset)
            .count()
            .filter(F.col("count") > 1)
        )
        dup_key_count = dup_keys.count()
        dup_row_sum   = (
            dup_keys.agg(F.sum("count").alias("s")).collect()[0]["s"] or 0
            if dup_key_count > 0 else 0
        )
        top_offenders = (
            dup_keys.orderBy(F.desc("count")).limit(5).collect()
            if dup_key_count > 0 else []
        )

        return {
            "soft_dup_subset_cols"          : subset,
            "soft_duplicate_key_combos"     : dup_key_count,
            "soft_duplicate_affected_rows"  : int(dup_row_sum),
            "soft_duplicate_affected_pct"   : _pct(int(dup_row_sum), self._total),
            "top5_offending_keys"           : [row.asDict() for row in top_offenders],
        }

    # ── 2.1c  Duplicated application ─────────────────────────

    def _app_dup_stats(self) -> dict[str, Any]:
        """
        How many entity_id values appear more than `app_dup_max_allowed` times?
        Flags cases where the same customer has multiple loan applications, for
        example, and the pipeline hasn't deduplicated to one row per entity.

        Equivalent SQL:
            SELECT customer_id, COUNT(*) AS n
            FROM tbl
            GROUP BY customer_id
            HAVING COUNT(*) > <app_dup_max_allowed>;
        """
        eid = self.config.entity_id_col
        if eid not in self.df.columns:
            return {"app_dup_check": f"skipped — column '{eid}' not found"}

        max_allowed = self.config.app_dup_max_allowed
        entity_counts = (
            self.df
            .groupBy(eid)
            .count()
        )
        total_entities  = entity_counts.count()
        dup_entities_df = entity_counts.filter(F.col("count") > max_allowed)
        dup_entity_count = dup_entities_df.count()

        # Distribution: 2×, 3-5×, 6-10×, >10×
        dist = {}
        if dup_entity_count > 0:
            dist = {
                "2x"    : dup_entities_df.filter(F.col("count") == 2).count(),
                "3-5x"  : dup_entities_df.filter(F.col("count").between(3, 5)).count(),
                "6-10x" : dup_entities_df.filter(F.col("count").between(6, 10)).count(),
                ">10x"  : dup_entities_df.filter(F.col("count") > 10).count(),
            }

        return {
            "total_distinct_entities"         : total_entities,
            "duplicated_application_entities" : dup_entity_count,
            "duplicated_application_pct"      : _pct(dup_entity_count, total_entities),
            "app_dup_max_allowed"             : max_allowed,
            "row_count_distribution"          : dist,
        }

    def collect(self) -> dict[str, Any]:
        return {
            "2_1a_exact_duplicates"       : self._exact_dup_stats(),
            "2_1b_soft_duplicates"        : self._soft_dup_stats(),
            "2_1c_duplicated_application" : self._app_dup_stats(),
        }


# ─────────────────────────────────────────────────────────────
# 2.2  Composite key validation
# ─────────────────────────────────────────────────────────────

class _CompositeKeySubCheck:
    """
    Validates that composite_key_cols form a unique key.
    Also checks label contamination and double-count risk.
    """

    def __init__(self, df: DataFrame, config: PKCheckConfig):
        self.df     = df
        self.config = config
        self._total = df.count()

    def collect(self) -> dict[str, Any]:
        ck    = [c for c in self.config.composite_key_cols if c in self.df.columns]
        label = self.config.label_col

        if not ck:
            return {"composite_key_check": "skipped — no valid composite key columns"}

        # ── base violation query ──────────────────────────────
        # SQL equivalent:
        #   SELECT <ck_cols>, COUNT(*) AS row_count
        #   FROM tbl
        #   GROUP BY <ck_cols>
        #   HAVING COUNT(*) > 1
        agg_exprs = [F.count("*").alias("row_count")]
        if label and label in self.df.columns:
            agg_exprs.append(F.countDistinct(F.col(label)).alias("distinct_labels"))
            # Feature variance proxy: any non-key, non-label numeric column
            feature_probe = _first_numeric_non_key(self.df, ck + [label])
            if feature_probe:
                agg_exprs.append(
                    (F.max(feature_probe) - F.min(feature_probe)).alias("feature_range_proxy")
                )

        violations = (
            self.df
            .groupBy(ck)
            .agg(*agg_exprs)
            .filter(F.col("row_count") > 1)
        )

        viol_count = violations.count()

        if viol_count == 0:
            return {
                "composite_key"          : ck,
                "violation_count"        : 0,
                "verdict_note"           : "Composite key is UNIQUE — no violations found.",
            }

        # ── aggregate violation stats ─────────────────────────
        agg_result = violations.agg(
            F.sum("row_count").alias("affected_rows"),
            F.max("row_count").alias("max_rows_per_key"),
            F.avg("row_count").alias("avg_rows_per_key"),
        ).collect()[0]

        affected_rows     = int(agg_result["affected_rows"] or 0)
        max_rows_per_key  = int(agg_result["max_rows_per_key"] or 0)
        avg_rows_per_key  = round(float(agg_result["avg_rows_per_key"] or 0), 2)

        # Label contamination: violations where >1 distinct label exists
        label_contamination = 0
        if label and label in self.df.columns and "distinct_labels" in violations.columns:
            label_contamination = violations.filter(F.col("distinct_labels") > 1).count()

        # Sample of worst offenders
        sample = violations.orderBy(F.desc("row_count")).limit(5).collect()

        return {
            "composite_key"              : ck,
            "violation_count"            : viol_count,
            "affected_rows"              : affected_rows,
            "affected_rows_pct"          : _pct(affected_rows, self._total),
            "max_rows_per_key"           : max_rows_per_key,
            "avg_rows_per_key"           : avg_rows_per_key,
            "label_contamination_count"  : label_contamination,
            "label_contamination_note"   : (
                "label_contamination_count > 0 means same (entity, date) has conflicting "
                "labels → direct training signal corruption."
            ),
            "double_count_risk_pct"      : _pct(affected_rows, self._total),
            "top5_violation_samples"     : [r.asDict() for r in sample],
        }


# ─────────────────────────────────────────────────────────────
# 2.3  Entity resolution check
# ─────────────────────────────────────────────────────────────

class _EntityResolutionSubCheck:
    """
    For each identity column (CIF, phone, device, national_id):
      - Forward  : customers with > 1 distinct value of this identity type
      - Reverse  : values linking to > 1 distinct customer (graph fragmentation)
    Also computes orphan customer count (no identity anchor at all).

    Critical for fraud and banking: a shared device_id or phone_number
    across customers is a strong fraud signal and can leak between train/test.
    """

    def __init__(self, df: DataFrame, config: PKCheckConfig):
        self.df     = df
        self.config = config

    def collect(self) -> dict[str, Any]:
        identity_cols = self.config.identity_cols
        eid           = self.config.entity_id_col

        if not identity_cols:
            return {"entity_resolution_check": "skipped — no identity_cols configured"}
        if eid not in self.df.columns:
            return {"entity_resolution_check": f"skipped — entity_id_col '{eid}' not found"}

        # Only check identity columns that actually exist
        valid_identity = {
            label: col
            for label, col in identity_cols.items()
            if col in self.df.columns
        }
        if not valid_identity:
            return {
                "entity_resolution_check" : "skipped",
                "reason"                  : f"None of {list(identity_cols.values())} found in DataFrame",
            }

        total_entities = self.df.select(eid).distinct().count()
        col_stats: dict[str, dict] = {}

        for label, col in valid_identity.items():
            col_stats[label] = self._single_id_col_stats(eid, col, total_entities)

        # Orphan customers: null across ALL identity columns
        orphan_filter = F.lit(True)
        for col in valid_identity.values():
            orphan_filter = orphan_filter & F.col(col).isNull()

        orphan_count = self.df.filter(orphan_filter).select(eid).distinct().count()

        return {
            "total_distinct_entities"   : total_entities,
            "identity_column_stats"     : col_stats,
            "orphan_entities"           : orphan_count,
            "orphan_pct"                : _pct(orphan_count, total_entities),
            "orphan_note"               : (
                "Orphan = customer with NULL in ALL identity columns. "
                "Cannot be linked across systems; high fraud risk if orphan rate > 5%."
            ),
        }

    def _single_id_col_stats(
        self, eid: str, col: str, total_entities: int
    ) -> dict[str, Any]:
        """
        Forward check: GROUP BY eid → count distinct col
        Reverse check: GROUP BY col → count distinct eid

        SQL (forward):
            SELECT customer_id, COUNT(DISTINCT <col>) AS id_count
            FROM tbl
            GROUP BY customer_id
            HAVING COUNT(DISTINCT <col>) > 1;

        SQL (reverse):
            SELECT <col>, COUNT(DISTINCT customer_id) AS cust_count
            FROM tbl
            WHERE <col> IS NOT NULL
            GROUP BY <col>
            HAVING COUNT(DISTINCT customer_id) > 1;
        """
        # Null rate
        total_rows = self.df.count()
        null_count = self.df.filter(F.col(col).isNull()).count()

        # Forward: customers with multiple identifiers of this type
        fwd = (
            self.df.filter(F.col(col).isNotNull())
            .groupBy(eid)
            .agg(F.countDistinct(col).alias("id_count"))
        )
        multi_id_customers = fwd.filter(F.col("id_count") > 1).count()

        # Distribution of id_count
        fwd_dist_rows = (
            fwd.groupBy("id_count")
            .count()
            .orderBy("id_count")
            .limit(6)
            .collect()
        )
        fwd_distribution = {str(r["id_count"]): r["count"] for r in fwd_dist_rows}

        # Reverse: identifiers shared by multiple customers (graph fragmentation)
        rev = (
            self.df.filter(F.col(col).isNotNull())
            .groupBy(col)
            .agg(F.countDistinct(eid).alias("customer_count"))
        )
        ambiguous_ids     = rev.filter(F.col("customer_count") > 1).count()
        total_distinct_id = rev.count()

        return {
            "null_pct"                          : _pct(null_count, total_rows),
            "total_distinct_values"             : total_distinct_id,
            # Forward
            "customers_with_multiple_values"    : multi_id_customers,
            "multi_value_pct_of_customers"      : _pct(multi_id_customers, total_entities),
            "id_count_distribution"             : fwd_distribution,
            # Reverse (graph integrity)
            "ambiguous_values_multi_customer"   : ambiguous_ids,
            "ambiguous_pct_of_distinct_values"  : _pct(ambiguous_ids, total_distinct_id),
            "graph_integrity_note"              : (
                "ambiguous > 0 means the same identifier links to multiple customers. "
                "For phone/device this can be fraud signal or data error."
            ),
        }


# ─────────────────────────────────────────────────────────────
# Main tool  (façade over all three sub-checks)
# ─────────────────────────────────────────────────────────────

class PKUniquenessCheckTool:
    """
    Facade that runs all three sub-checks and returns a single stats dict
    for the LLM decision gate.

    Parameters
    ----------
    df     : Spark DataFrame
    config : PKCheckConfig
    """

    name = "pk_and_uniqueness"

    def __init__(self, df: DataFrame, config: PKCheckConfig):
        self.df     = df
        self.config = config

    def collect_stats(self) -> dict[str, Any]:
        total = self.df.count()

        dup_stats  = _DuplicateSubCheck(self.df, self.config).collect()
        ck_stats   = _CompositeKeySubCheck(self.df, self.config).collect()
        er_stats   = _EntityResolutionSubCheck(self.df, self.config).collect()

        return {
            "domain_context"           : (
                "Banking / fraud ML dataset. Identity integrity is critical: "
                "composite key violations → double-count labels; shared device/phone → fraud ring leakage."
            ),
            "total_rows"               : total,
            "check_2_1_duplicates"     : dup_stats,
            "check_2_2_composite_key"  : ck_stats,
            "check_2_3_entity_resolution": er_stats,
        }


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _pct(part: int, total: int, decimals: int = 2) -> float:
    if total == 0:
        return 0.0
    return round(part / total * 100, decimals)


def _first_numeric_non_key(df: DataFrame, exclude_cols: list[str]) -> str | None:
    """Return first numeric column not in exclude_cols, for feature variance probe."""
    from pyspark.sql.types import NumericType
    for name, dtype in df.dtypes:
        if name not in exclude_cols:
            try:
                if isinstance(df.schema[name].dataType, NumericType):
                    return name
            except Exception:
                continue
    return None


# ─────────────────────────────────────────────────────────────
# Quick standalone demo
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import random
    from datetime import date, timedelta
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local").appName("pk_demo").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    random.seed(0)
    base = date(2023, 1, 1)
    rows = []
    for i in range(500):
        cid = f"C{random.randint(1, 300):04d}"    # 300 unique customers, 500 rows → dups
        sd  = base + timedelta(days=random.randint(0, 90))
        rows.append({
            "customer_id"  : cid,
            "snapshot_date": sd,
            "cif_id"       : f"CIF{random.randint(1, 350):04d}",
            "phone_number" : f"09{random.randint(10000000, 99999999)}",
            "device_id"    : f"DEV{random.randint(1, 200):04d}",
            "national_id"  : f"NID{random.randint(1, 280):04d}" if random.random() > 0.1 else None,
            "income"       : random.gauss(50000, 20000),
            "default_flag" : random.choice([0, 0, 0, 1]),
        })

    df = spark.createDataFrame(rows)

    config = PKCheckConfig(
        entity_id_col      = "customer_id",
        snapshot_date_col  = "snapshot_date",
        composite_key_cols = ["customer_id", "snapshot_date"],
        label_col          = "default_flag",
        identity_cols      = {
            "cif"        : "cif_id",
            "phone"      : "phone_number",
            "device"     : "device_id",
            "national_id": "national_id",
        },
    )

    tool  = PKUniquenessCheckTool(df, config)
    stats = tool.collect_stats()
    print(json.dumps(stats, indent=2, default=str))
