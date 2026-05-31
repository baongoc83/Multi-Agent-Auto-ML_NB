import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from logger import AgentLogger
from Agents.DataCleaner.agent_data_cleaner import DataCleanerAgent
from config import Config
from datetime import datetime
from typing import Optional, Tuple, Dict, List
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
import os
import warnings
warnings.filterwarnings("ignore")


def create_sample_data(
    n_rows: int = 2_000,
    n_months: int = 12,
    output_path: str = "outputs/sample_data.csv",
    seed: int = 42,
) -> str:
    """Create a small synthetic dataset for agent testing.

    Includes: numeric continuous, integer, encoded categorical, binary flags,
    string categoricals, and deliberately bad columns (high-null, constant,
    high-drift) so Agent 1 has something to clean.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    log.info("Creating sample data: %d rows ...", n_rows)

    # ── Snapshot date (temporal column) ──────────────────────────────────────
    start_date = pd.Timestamp("2023-01-01")
    month_offsets = np.arange(n_months)
    month_weights = np.linspace(0.5, 1.5, n_months)
    month_weights /= month_weights.sum()
    assigned_months = rng.choice(month_offsets, size=n_rows, p=month_weights)
    snap_dt = pd.to_datetime([
        start_date + pd.offsets.MonthBegin(int(m)) + pd.Timedelta(days=int(rng.integers(0, 28)))
        for m in assigned_months
    ])

    # ── Target label (~6% event rate) ────────────────────────────────────────
    base_prob = 0.06
    time_trend = (assigned_months / n_months) * 0.02
    label_prob = np.clip(base_prob + time_trend, 0.01, 0.30)
    label = (rng.random(n_rows) < label_prob).astype(int)

    cols: Dict[str, np.ndarray] = {}

    # Numeric continuous (20 cols, ~10 with signal)
    for i in range(20):
        vals = rng.normal(loc=rng.uniform(0, 100), scale=rng.uniform(5, 30), size=n_rows)
        if i < 10:
            vals += label * rng.uniform(0.3, 1.5) * np.std(vals)
        nan_mask = rng.random(n_rows) < rng.uniform(0.05, 0.20)
        vals = vals.astype(float)
        vals[nan_mask] = np.nan
        cols[f"num_cont_{i:02d}"] = vals

    # Numeric integer (10 cols)
    for i in range(10):
        vals = rng.integers(0, 101, size=n_rows).astype(float)
        if i < 5:
            vals += label * rng.uniform(5, 20)
        nan_mask = rng.random(n_rows) < rng.uniform(0.03, 0.10)
        vals[nan_mask] = np.nan
        cols[f"num_int_{i:02d}"] = vals

    # Encoded categorical (8 cols, nunique <= 8)
    for i in range(8):
        n_cats = rng.integers(3, 9)
        cat_probs = rng.dirichlet(np.ones(n_cats))
        vals = rng.choice(np.arange(n_cats), size=n_rows, p=cat_probs).astype(float)
        nan_mask = rng.random(n_rows) < 0.08
        vals[nan_mask] = np.nan
        cols[f"enc_cat_{i:02d}"] = vals

    # Binary flags (5 cols)
    for i in range(5):
        p_one = rng.uniform(0.05, 0.50)
        vals = (rng.random(n_rows) < p_one).astype(float)
        nan_mask = rng.random(n_rows) < 0.05
        vals[nan_mask] = np.nan
        cols[f"bin_flag_{i:02d}"] = vals

    # String categorical (4 cols, object dtype) — some with mixed case / whitespace
    cat_alphabets = [["LOW", "MED", "HIGH"], ["A", "B", "C", "D", "E"],
                     ["NEW", "ACTIVE", "DORMANT", "CLOSED"], ["TYPE1", "TYPE2", "TYPE3"]]
    for i in range(4):
        alphabet = cat_alphabets[i % len(cat_alphabets)]
        cat_probs = rng.dirichlet(np.ones(len(alphabet)))
        vals_idx = rng.choice(len(alphabet), size=n_rows, p=cat_probs)
        vals = np.array(alphabet, dtype=object)[vals_idx]
        if i == 0:  # inject mixed-case issue
            mask = rng.random(n_rows) < 0.3
            vals[mask] = np.char.lower(vals[mask].astype(str))
        nan_mask = rng.random(n_rows) < 0.10
        vals = vals.astype(object)
        vals[nan_mask] = None
        cols[f"str_cat_{i:02d}"] = vals

    # Bad columns Agent 1 should detect / clean
    cols["useless_all_nan"] = np.full(n_rows, np.nan)                    # 100% null → drop
    cols["useless_const"] = np.full(n_rows, 42.0)                        # constant → drop
    for i in range(2):                                                    # high-drift
        shift = (assigned_months / n_months) * rng.uniform(4, 8)
        vals = rng.normal(0, 1, n_rows) + shift
        vals[rng.random(n_rows) < 0.05] = np.nan
        cols[f"drift_col_{i:02d}"] = vals

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(cols)
    df.insert(0, "snap_dt", snap_dt)
    df.insert(0, "customer_id", np.arange(1, n_rows + 1))
    df["label"] = label

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    log.info("Sample data created: %s  shape=%s  event_rate=%.2f%%",
             output_path, df.shape, df["label"].mean() * 100)
    return output_path


def test_agent1():
    print("Testing Agent 1: Data Cleaner")

    try:
        Config.validate()
        print("Configuration validated\n")
    except ValueError as e:
        print(f"Configuration error: {e}")
        return

    input_csv = create_sample_data(n_rows=2_000, output_path="outputs/sample_data.csv")

    logger = AgentLogger("outputs/test_agent1.log")
    agent = DataCleanerAgent(
        logger,
        entity_id_col="customer_id",
        composite_key_cols=["customer_id", "snap_dt"],
        target_column="label",
    )

    print("Agent Data Cleaner is now processing the data...")

    try:
        clean_data_path, report = agent.process(input_csv)

        print("Agent Data Cleaner results")
        print(f"\nCleaned data saved to: {clean_data_path}")
        print(f"Report saved to: outputs/data_cleaner_report.json")
        print(f"\nShape Change: {report['original_shape']} -> {report['final_shape']}")

        print("\nActions Taken:")
        for i, action in enumerate(report['actions_taken'], 1):
            print(f"  {i}. {action}")

        print(f"\nSummary: {report['summary']}")
        print(f"\nRemaining Columns: {', '.join(report['columns_remaining'])}")

        cleaned_df = pd.read_csv(clean_data_path)
        print("\nCleaned Data Sample (first 5 rows):")
        print(cleaned_df.head())
        print(f"\n  Missing values total: {cleaned_df.isnull().sum().sum()}")

        logger.save()
        print(f"\nExecution log saved to: {logger.log_file}")

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_agent1()
