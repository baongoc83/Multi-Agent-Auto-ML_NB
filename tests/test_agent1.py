import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from logger import AgentLogger
from agent_data_cleaner import DataCleanerAgent
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

# def create_sample_data(
#     n_rows: int = 500_000,
#     n_months: int = 18,        # trải đều data qua 18 tháng
#     output_path: str = "outputs/sample_data.csv",
#     seed: int = 42,
# ) -> pd.DataFrame:
#     """
#     Tạo dataset giả lập thực tế cho pipeline ML:
#       - 500k rows × 503 cột (500 features + id + snap_dt + label)
#       - Trải theo thời gian 18 tháng để split_data() hoạt động đúng
#       - Có đủ loại: numeric continuous, integer, encoded categorical,
#         binary flag, categorical string, và các cột vô dụng
#       - Có missing values, drift (PSI cao ở OOT), và feature ổn định/không ổn định

#     Returns:
#         pd.DataFrame: full dataset sẵn sàng truyền vào pipeline
#     """
#     rng = np.random.default_rng(seed)
#     os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

#     log.info("Đang tạo sample data: %d rows × 500 features ...", n_rows)

#     # ── Tạo cột ngày quan sát (snap_dt) ──────────────────────────────────
#     # Phân phối không đều: tháng gần đây có nhiều data hơn (thực tế)
#     start_date = pd.Timestamp("2023-01-01")
#     month_offsets = np.arange(n_months)
#     # Xác suất tăng dần theo thời gian (tháng mới nhất có nhiều obs hơn)
#     month_weights = np.linspace(0.5, 1.5, n_months)
#     month_weights /= month_weights.sum()
#     assigned_months = rng.choice(month_offsets, size=n_rows, p=month_weights)
#     # Thêm random day trong tháng
#     snap_dt = pd.to_datetime([
#         start_date + pd.offsets.MonthBegin(int(m)) + pd.Timedelta(days=int(rng.integers(0, 28)))
#         for m in assigned_months
#     ])

#     # ── Target (label) với event rate ~5–8%, có trend tăng theo thời gian ─
#     base_prob   = 0.06
#     time_trend  = (assigned_months / n_months) * 0.02   # drift nhẹ theo thời gian
#     label_prob  = np.clip(base_prob + time_trend, 0.01, 0.30)
#     label       = (rng.random(n_rows) < label_prob).astype(int)

#     cols: Dict[str, np.ndarray] = {}

#     # ── Nhóm 1: numeric_continuous (200 cột) ─────────────────────────────
#     # Có 3 loại phân phối: normal, lognormal, uniform
#     # ~40 cột có signal thực với target, còn lại noise
#     for i in range(200):
#         nan_rate = rng.uniform(0.05, 0.20)
#         dist     = i % 3

#         if dist == 0:       # normal
#             vals = rng.normal(loc=rng.uniform(0, 100), scale=rng.uniform(5, 30), size=n_rows)
#         elif dist == 1:     # lognormal (thu nhập, dư nợ, ...)
#             vals = rng.lognormal(mean=rng.uniform(3, 8), sigma=rng.uniform(0.3, 1.2), size=n_rows)
#         else:               # uniform
#             lo = rng.uniform(0, 50)
#             vals = rng.uniform(lo, lo + rng.uniform(10, 200), size=n_rows)

#         # Thêm signal cho ~40 cột đầu
#         if i < 40:
#             signal_strength = rng.uniform(0.3, 1.5)
#             vals += label * signal_strength * np.std(vals)

#         # Inject NaN
#         nan_mask      = rng.random(n_rows) < nan_rate
#         vals          = vals.astype(float)
#         vals[nan_mask] = np.nan

#         cols[f"num_cont_{i:03d}"] = vals

#     # ── Nhóm 2: numeric_integer (100 cột) ────────────────────────────────
#     for i in range(100):
#         nan_rate = rng.uniform(0.03, 0.15)
#         vals     = rng.integers(0, 1001, size=n_rows).astype(float)
#         if i < 20:
#             vals += label * rng.uniform(10, 80)
#         nan_mask      = rng.random(n_rows) < nan_rate
#         vals[nan_mask] = np.nan
#         cols[f"num_int_{i:03d}"] = vals

#     # ── Nhóm 3: encoded_categorical (80 cột, nunique ≤ 10) ───────────────
#     # Giả lập Label Encoded: 0–9, NaN ~10%
#     for i in range(80):
#         nan_rate  = rng.uniform(0.05, 0.15)
#         n_cats    = rng.integers(3, 11)           # 3 đến 10 categories
#         cat_probs = rng.dirichlet(np.ones(n_cats))
#         vals      = rng.choice(np.arange(n_cats), size=n_rows, p=cat_probs).astype(float)
#         if i < 15:
#             # Một số category liên quan target
#             high_risk_cat = int(rng.integers(0, n_cats))
#             vals += (vals == high_risk_cat) * label * 0.5
#         nan_mask      = rng.random(n_rows) < nan_rate
#         vals[nan_mask] = np.nan
#         cols[f"enc_cat_{i:03d}"] = vals

#     # ── Nhóm 4: binary_flag (50 cột) ─────────────────────────────────────
#     for i in range(50):
#         nan_rate  = rng.uniform(0.02, 0.08)
#         p_one     = rng.uniform(0.05, 0.50)
#         vals      = (rng.random(n_rows) < p_one).astype(float)
#         if i < 10:
#             # Một số flag liên quan target
#             vals += label * (rng.random(n_rows) < 0.3).astype(float)
#             vals  = np.clip(vals, 0, 1)
#         nan_mask      = rng.random(n_rows) < nan_rate
#         vals[nan_mask] = np.nan
#         cols[f"bin_flag_{i:03d}"] = vals

#     # ── Nhóm 5: categorical_string (40 cột, object dtype) ────────────────
#     cat_alphabets = [
#         list("ABCDE"),
#         list("ABCDEFG"),
#         ["LOW", "MED", "HIGH"],
#         ["TYPE1", "TYPE2", "TYPE3", "TYPE4"],
#         ["NEW", "ACTIVE", "DORMANT", "CLOSED"],
#     ]
#     for i in range(40):
#         nan_rate   = rng.uniform(0.05, 0.20)
#         alphabet   = cat_alphabets[i % len(cat_alphabets)]
#         cat_probs  = rng.dirichlet(np.ones(len(alphabet)))
#         vals_idx   = rng.choice(len(alphabet), size=n_rows, p=cat_probs)
#         vals       = np.array(alphabet, dtype=object)[vals_idx]
#         nan_mask   = rng.random(n_rows) < nan_rate
#         vals       = vals.astype(object)
#         vals[nan_mask] = None
#         cols[f"str_cat_{i:03d}"] = vals

#     # ── Nhóm 6: useless columns (30 cột) ─────────────────────────────────
#     # 6a: 10 cột 100% NaN
#     for i in range(10):
#         cols[f"useless_all_nan_{i:02d}"] = np.full(n_rows, np.nan)

#     # 6b: 10 cột constant (variance = 0)
#     for i in range(10):
#         const_val = rng.integers(0, 5)
#         cols[f"useless_const_{i:02d}"] = np.full(n_rows, float(const_val))

#     # 6c: 10 cột high-PSI: phân phối thay đổi mạnh theo thời gian
#     #     (để PSI filter loại được)
#     for i in range(10):
#         # Tháng đầu ~ N(0,1), tháng cuối ~ N(5,1) → PSI rất cao
#         shift       = (assigned_months / n_months) * rng.uniform(4, 8)
#         vals        = rng.normal(0, 1, n_rows) + shift
#         nan_mask    = rng.random(n_rows) < 0.05
#         vals[nan_mask] = np.nan
#         cols[f"useless_drift_{i:02d}"] = vals

#     # ── Ghép DataFrame ────────────────────────────────────────────────────
#     df = pd.DataFrame(cols)
#     df.insert(0, "snap_dt",     snap_dt)
#     df.insert(0, "customer_id", np.arange(1, n_rows + 1))
#     df["label"] = label

#     # ── Shuffle để không có thứ tự theo label ────────────────────────────
#     df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
#     Path("outputs").mkdir(exist_ok=True)
#     # ── Save & report ─────────────────────────────────────────────────────
#     df.to_csv(output_path, index=False)

#     total_features = len([c for c in df.columns if c not in ("customer_id", "snap_dt", "label")])
#     missing_summary = df.isnull().sum()

#     log.info("✔ Sample data created: %s", output_path)
#     log.info("  Shape             : %s", df.shape)
#     log.info("  Date range        : %s → %s",
#              df["snap_dt"].min().date(), df["snap_dt"].max().date())
#     log.info("  Months            : %d", df["snap_dt"].dt.to_period("M").nunique())
#     log.info("  Event rate (label): %.2f%%", df["label"].mean() * 100)
#     log.info("  Total features    : %d", total_features)
#     log.info("  Cols with NaN     : %d / %d",
#              (missing_summary > 0).sum(), len(df.columns))
#     log.info("  Feature groups    :")
#     log.info("    num_cont   : 200 cột | avg NaN ~12%%")
#     log.info("    num_int    : 100 cột | avg NaN ~9%%")
#     log.info("    enc_cat    :  80 cột | avg NaN ~10%% | nunique ≤ 10")
#     log.info("    bin_flag   :  50 cột | avg NaN ~5%%")
#     log.info("    str_cat    :  40 cột | avg NaN ~12%%  | object dtype")
#     log.info("    useless    :  30 cột | all_nan/const/high-drift")

#     return output_path

def test_agent1():
    print("Testing Agent 1: Data Cleaner")

    try:
        Config.validate()
        print("OpenAI configuration validated\n")
    except ValueError as e:
        print(f"Configuration error: {e}")
        return
    
    # input_csv = create_sample_data()
    input_csv = "outputs/sample_data.csv"
    logger = AgentLogger("outputs/test_agent1.log")
    agent = DataCleanerAgent(logger)
    
    print("Agent 1 is now processing the data...")
    
    try:
        clean_data_path, report = agent.process(input_csv)
        
        print("Agent 1 results")
        
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
        
        print("\nCleaned Data Info:")
        print(f"  - Missing values: {cleaned_df.isnull().sum().sum()}")
        
        logger.save()
        print(f"\nExecution log saved to: {logger.log_file}")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent1()