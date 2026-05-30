"""
Process Home Credit Default Risk dataset.

Join hierarchy (per diagram):
  application_{train|test}  (SK_ID_CURR = PK)
    ├── bureau               (SK_ID_CURR)
    │    └── bureau_balance  (SK_ID_BUREAU → aggregate → SK_ID_CURR)
    ├── previous_application (SK_ID_CURR)
    │    ├── POS_CASH_balance        (SK_ID_PREV + SK_ID_CURR)
    │    ├── installments_payments   (SK_ID_PREV + SK_ID_CURR)
    │    └── credit_card_balance     (SK_ID_PREV + SK_ID_CURR)

Strategy:
- Numeric columns  → mean / sum / max / min / std / count
- Object columns   → nunique
- ID columns (SK_ID_PREV, SK_ID_BUREAU) are dropped before aggregation to
  avoid meaningless numeric stats on surrogate keys.

Outputs (saved in same folder as source data):
  application_train_processed.csv
  application_test_processed.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "home-credit-default-risk"
OUTPUT_DIR = DATA_DIR          # change here to redirect output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── aggregation helper ─────────────────────────────────────────────────────────

def _build_agg_dict(df: pd.DataFrame, group_col: str) -> dict:
    """Return agg spec: numeric → stats, object → nunique (skip group col)."""
    agg: dict = {}
    for col in df.columns:
        if col == group_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            agg[col] = ["mean", "sum", "max", "min", "std", "count"]
        else:
            agg[col] = ["nunique"]
    return agg


def aggregate(df: pd.DataFrame, group_col: str, prefix: str) -> pd.DataFrame:
    """Aggregate df by group_col; prefix all output columns with prefix_."""
    agg_dict = _build_agg_dict(df, group_col)
    if not agg_dict:
        log.warning("  aggregate('%s'): no columns to aggregate — returning key only", prefix)
        return df[[group_col]].drop_duplicates()

    result = df.groupby(group_col, sort=False).agg(agg_dict)
    result.columns = [f"{prefix}_{col}_{fn}" for col, fn in result.columns]
    result = result.reset_index()
    log.info("  %-30s  rows=%d  cols=%d", prefix, len(result), result.shape[1] - 1)
    return result


def _drop_surrogate(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    """Drop surrogate key columns if present (SK_ID_PREV, SK_ID_BUREAU, …)."""
    return df.drop(columns=[c for c in cols if c in df.columns])


# ── per-table feature builders ─────────────────────────────────────────────────

def build_bureau_features() -> pd.DataFrame:
    """
    bureau (→ SK_ID_CURR) + bureau_balance (→ SK_ID_BUREAU).

    bureau_balance is first aggregated to SK_ID_BUREAU level,
    then merged into bureau, then the whole thing is aggregated
    to SK_ID_CURR level.
    """
    log.info("Processing bureau …")
    bureau = pd.read_csv(DATA_DIR / "bureau.csv")

    log.info("  Loading bureau_balance …")
    bb = pd.read_csv(DATA_DIR / "bureau_balance.csv")
    bb_agg = aggregate(bb, group_col="SK_ID_BUREAU", prefix="bb")

    # Attach bureau_balance aggregates to bureau rows (one row per bureau loan)
    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    # Drop surrogate key before aggregating to client level
    bureau = _drop_surrogate(bureau, "SK_ID_BUREAU")

    return aggregate(bureau, group_col="SK_ID_CURR", prefix="bur")


def build_previous_application_features() -> pd.DataFrame:
    """previous_application (→ SK_ID_CURR)."""
    log.info("Processing previous_application …")
    prev = pd.read_csv(DATA_DIR / "previous_application.csv")
    prev = _drop_surrogate(prev, "SK_ID_PREV")
    return aggregate(prev, group_col="SK_ID_CURR", prefix="prev")


def build_pos_cash_features() -> pd.DataFrame:
    """POS_CASH_balance (→ SK_ID_CURR, via SK_ID_PREV + SK_ID_CURR both present)."""
    log.info("Processing POS_CASH_balance …")
    pos = pd.read_csv(DATA_DIR / "POS_CASH_balance.csv")
    pos = _drop_surrogate(pos, "SK_ID_PREV")
    return aggregate(pos, group_col="SK_ID_CURR", prefix="pos")


def build_installments_features() -> pd.DataFrame:
    """installments_payments (→ SK_ID_CURR)."""
    log.info("Processing installments_payments …")
    ins = pd.read_csv(DATA_DIR / "installments_payments.csv")
    ins = _drop_surrogate(ins, "SK_ID_PREV")
    return aggregate(ins, group_col="SK_ID_CURR", prefix="ins")


def build_credit_card_features() -> pd.DataFrame:
    """credit_card_balance (→ SK_ID_CURR)."""
    log.info("Processing credit_card_balance …")
    cc = pd.read_csv(DATA_DIR / "credit_card_balance.csv")
    cc = _drop_surrogate(cc, "SK_ID_PREV")
    return aggregate(cc, group_col="SK_ID_CURR", prefix="cc")


# ── main join ──────────────────────────────────────────────────────────────────

_FEATURE_BUILDERS = [
    build_bureau_features,
    build_previous_application_features,
    build_pos_cash_features,
    build_installments_features,
    build_credit_card_features,
]


def process(split: str) -> pd.DataFrame:
    """
    Load application_{split}.csv and left-join all feature blocks on SK_ID_CURR.
    split: 'train' or 'test'
    """
    assert split in ("train", "test"), f"split must be 'train' or 'test', got '{split}'"

    log.info("=" * 60)
    log.info("Loading application_%s.csv …", split)
    app = pd.read_csv(DATA_DIR / f"application_{split}.csv")
    log.info("  application_%s  shape=%s", split, app.shape)

    for builder in _FEATURE_BUILDERS:
        feats = builder()
        cols_before = app.shape[1]
        app = app.merge(feats, on="SK_ID_CURR", how="left")
        log.info("  joined %-30s  +%d cols  total=%d",
                 builder.__name__.replace("build_", "").replace("_features", ""),
                 app.shape[1] - cols_before,
                 app.shape[1])

    log.info("Final shape for %s: %s", split, app.shape)
    return app


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        df = process(split)
        out_path = OUTPUT_DIR / f"application_{split}_processed.csv"
        df.to_csv(out_path, index=False)
        log.info("Saved → %s", out_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
