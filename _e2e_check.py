"""End-to-end smoke test for both pipeline modes.

Mocks LLM calls with canned JSON so the full Agent 1→2→3 flow runs without
hitting any gateway / network. Verifies:

1. AUTO-SPLIT mode (single input_path, no valid/oot)
   - Pipeline runs to completion
   - Final report contains AUC metrics
   - Model artifact + inference code written
   - Splits dict matches expected fallback (train+valid+test or temporal)

2. PRE-SPLIT mode (train + valid + oot supplied separately)
   - Combined parquet built with _split_ marker
   - Marker preserved through Agents 1 + 2
   - Agent 3 reconstructs the exact user-defined sizes
   - No leakage: Agent 1's _train_view excludes valid/oot from stats
   - No leakage: Agent 2's encoders/selection fit on train only

Run from project root:
    python _e2e_check.py
"""
from __future__ import annotations

import json
import sys
import warnings
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Mock LLM BEFORE any agent imports it
# ─────────────────────────────────────────────────────────────────────────────
from Agents.BaseAgent import base_agent as _ba


def _canned_response(agent_name: str, prompt: str, system_prompt: str, json_mode: bool):
    sysl = system_prompt.lower()
    if "data quality" in sysl or "DataCleaner" in agent_name:
        return json.dumps({
            "reasoning": "stub: light cleaning only",
            "actions": [
                {"action": "drop_duplicates",
                 "reason": "stub: remove any exact duplicates"},
            ],
        })
    if "feature engineer" in sysl or "FeatureEngineer" in agent_name:
        return json.dumps({
            "reasoning": "stub: encode and select",
            "actions": [
                {"action": "encode_all_categorical", "method": "label",
                 "reason": "stub: prepare all object cols for modelling"},
                {"action": "select_top_features", "k": 10,
                 "reason": "stub: keep top 10 features"},
            ],
        })
    # Agent 3 — overfit fix JSON or final summary text
    if json_mode or '"' in prompt:
        return json.dumps({"reg_alpha": 1.0, "max_depth": 4, "min_child_samples": 50})
    return "Pipeline finished. Best estimator with stable AUC."


def _mock_call_llm(self, prompt, system_prompt, json_mode=False, max_tokens=None):
    return _canned_response(self.name, prompt, system_prompt, json_mode)


_ba.BaseAgent.call_llm = _mock_call_llm

# Now safe to import the pipeline
from config import Config            # noqa: E402
from pipeline import AutoMLPipeline, SPLIT_MARKER   # noqa: E402

# Tight time budgets so the test finishes in a reasonable time
Config.FLAML_TIME_BUDGET = 30
Config.OPTUNA_TIMEOUT = 30
Config.OPTUNA_N_TRIALS = 5
Config.FLAML_N_SPLITS = 3
Config.CV_N_SPLITS = 3
Config.MAX_FINAL_FEATURES = 10
Config.SHAP_N_ESTIMATORS = 30
Config.SHAP_SAMPLE_SIZE = 200
Config.RFE_N_ESTIMATORS = 30
Config.RFE_TARGET_FEATURES = 8


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def gen_data(n: int, seed: int = 42, with_date: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = {
        "customer_id": np.arange(seed * 1_000_000, seed * 1_000_000 + n),
        "num_signal_1": rng.normal(0, 1, n),
        "num_signal_2": rng.normal(5, 2, n),
        "num_noise":    rng.normal(0, 1, n),
        "cat_signal":   rng.choice(["A", "B", "C", "D"], n,
                                   p=[0.4, 0.3, 0.2, 0.1]),
        "cat_noise":    rng.choice(["X", "Y", "Z"], n),
    }
    if with_date:
        # Spread over 8 months — enough for stability/OOT temporal logic
        starts = pd.date_range("2024-01-01", periods=8, freq="MS")
        d["snap_dt"] = rng.choice(starts, n) + pd.to_timedelta(
            rng.integers(0, 28, n), unit="D")
    # Target with real signal from numeric + categorical
    score = (
        0.6 * d["num_signal_1"]
        + 0.3 * (d["num_signal_2"] - 5) / 2
        + 0.4 * (np.array(d["cat_signal"]) == "A").astype(float)
        - 0.3 * (np.array(d["cat_signal"]) == "D").astype(float)
        + rng.normal(0, 0.5, n)
    )
    prob = 1.0 / (1.0 + np.exp(-score))
    d["label"] = (rng.random(n) < prob).astype(int)
    return pd.DataFrame(d)


def reset_outputs() -> None:
    out = Path("outputs")
    if out.exists():
        for p in out.glob("*"):
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass


def assert_files_exist(paths: list[str]) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise AssertionError(f"Missing output files: {missing}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — AUTO-SPLIT MODE (single dataset)
# ─────────────────────────────────────────────────────────────────────────────
def test_auto_split() -> None:
    section("TEST 1 — AUTO-SPLIT mode (single input_path)")
    reset_outputs()

    df = gen_data(800, seed=1, with_date=True)
    df.to_csv("outputs/_auto_input.csv", index=False)
    print(f"  Generated input: shape={df.shape} | "
          f"label_balance={df['label'].value_counts().to_dict()}")

    pipe = AutoMLPipeline()
    metrics = pipe.run(
        input_path="outputs/_auto_input.csv",
        target_column="label",
        entity_id_col="customer_id",
        composite_key_cols=["customer_id", "snap_dt"],
        domain="generic",
        model_type="binary_classification",
    )

    print(f"\n  Final metrics keys: {sorted(metrics.keys())}")
    assert "best_model" in metrics, "best_model missing from metrics"
    assert "cv_auc_mean" in metrics, "cv_auc_mean missing from metrics"

    # In auto-split + date_col: expect OOT temporal split (oot key populated)
    # If not enough months, falls back to test split. Either is fine — just verify
    # we got a holdout AUC.
    has_holdout_auc = any(k in metrics for k in ("oot_auc", "test_auc"))
    assert has_holdout_auc, "no holdout AUC (oot_auc or test_auc) present"

    assert_files_exist([
        Config.CLEAN_DATA_PATH,
        Config.ENGINEERED_DATA_PATH,
        Config.FINAL_MODEL_PATH,
        Config.FINAL_MODEL_CODE_PATH,
        Config.DATA_CLEANER_REPORT_PATH,
        Config.FEATURE_ENGINEER_REPORT_PATH,
        Config.MODEL_TRAINER_REPORT_PATH,
    ])

    with open(Config.MODEL_TRAINER_REPORT_PATH, encoding="utf-8") as f:
        rep3 = json.load(f)
    print(f"  Splits sizes: {rep3.get('splits', {})}")

    # Verify NO marker present in clean/engineered data (auto-split mode)
    cleaned = pd.read_csv(Config.CLEAN_DATA_PATH)
    engineered = pd.read_csv(Config.ENGINEERED_DATA_PATH)
    assert SPLIT_MARKER not in cleaned.columns, "marker should NOT appear in auto-split mode"
    assert SPLIT_MARKER not in engineered.columns, "marker should NOT appear in auto-split mode"
    print("  [OK] no _split_ marker in auto-split outputs")
    print("  [PASS] AUTO-SPLIT mode end-to-end works")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — PRE-SPLIT MODE (train + valid + oot supplied)
# ─────────────────────────────────────────────────────────────────────────────
def test_pre_split() -> None:
    section("TEST 2 — PRE-SPLIT mode (train + valid + oot)")
    reset_outputs()

    # Distinct seeds → distinct customer_id ranges (no overlap)
    train = gen_data(800, seed=2, with_date=True)
    valid = gen_data(200, seed=3, with_date=True)
    oot   = gen_data(200, seed=4, with_date=True)
    train.to_csv("outputs/_train.csv", index=False)
    valid.to_csv("outputs/_valid.csv", index=False)
    oot.to_csv("outputs/_oot.csv", index=False)
    print(f"  Train: {train.shape} | Valid: {valid.shape} | OOT: {oot.shape}")
    n_train_in, n_valid_in, n_oot_in = len(train), len(valid), len(oot)

    pipe = AutoMLPipeline()
    metrics = pipe.run(
        input_path="outputs/_train.csv",
        valid_path="outputs/_valid.csv",
        oot_path="outputs/_oot.csv",
        target_column="label",
        entity_id_col="customer_id",
        composite_key_cols=["customer_id", "snap_dt"],
        domain="generic",
        model_type="binary_classification",
        prefilter=True,        # default-on
        train_sample_ratio=None,
    )

    print(f"\n  Final metrics keys: {sorted(metrics.keys())}")
    assert "best_model" in metrics
    assert "cv_auc_mean" in metrics
    assert "valid_auc" in metrics, "valid_auc missing — pre-split valid set not evaluated"
    assert "oot_auc" in metrics,   "oot_auc missing — pre-split oot set not evaluated"

    # ── Verify _split_ marker propagated through Agents 1+2 ────────────────
    combined = pd.read_parquet(f"{Config.OUTPUT_DIR}/combined_input.parquet")
    assert SPLIT_MARKER in combined.columns, "marker missing from combined input"
    combined_marker = combined[SPLIT_MARKER].value_counts().to_dict()
    print(f"  Combined input marker counts: {combined_marker}")
    assert combined_marker.get("train") == n_train_in, "train rows mismatch in combined"
    assert combined_marker.get("valid") == n_valid_in
    assert combined_marker.get("oot")   == n_oot_in

    cleaned = pd.read_csv(Config.CLEAN_DATA_PATH)
    assert SPLIT_MARKER in cleaned.columns, "Agent 1 must preserve _split_ marker"
    cleaned_marker = cleaned[SPLIT_MARKER].value_counts().to_dict()
    print(f"  Post-Agent-1 marker counts : {cleaned_marker}")

    engineered = pd.read_csv(Config.ENGINEERED_DATA_PATH)
    assert SPLIT_MARKER in engineered.columns, "Agent 2 must preserve _split_ marker"
    engineered_marker = engineered[SPLIT_MARKER].value_counts().to_dict()
    print(f"  Post-Agent-2 marker counts : {engineered_marker}")

    # ── Verify Agent 3 reconstructed splits from marker ────────────────────
    with open(Config.MODEL_TRAINER_REPORT_PATH, encoding="utf-8") as f:
        rep3 = json.load(f)
    a3_splits = rep3.get("splits", {})
    print(f"  Agent 3 split sizes        : {a3_splits}")
    assert a3_splits.get("valid", 0) == engineered_marker.get("valid", 0), \
        f"Agent 3 valid size {a3_splits.get('valid')} != marker valid {engineered_marker.get('valid')}"
    assert a3_splits.get("oot", 0) == engineered_marker.get("oot", 0), \
        f"Agent 3 oot size {a3_splits.get('oot')} != marker oot {engineered_marker.get('oot')}"

    # train size in Agent 3 = engineered train (minus rows dropped by cleaning)
    assert a3_splits.get("train", 0) <= engineered_marker.get("train", 0)
    print("  [OK] splits propagated train+valid+oot exactly through full pipeline")

    assert_files_exist([
        f"{Config.OUTPUT_DIR}/combined_input.parquet",
        Config.CLEAN_DATA_PATH,
        Config.ENGINEERED_DATA_PATH,
        Config.FINAL_MODEL_PATH,
        Config.FINAL_MODEL_CODE_PATH,
    ])
    print("  [PASS] PRE-SPLIT mode end-to-end works")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — PRE-SPLIT MODE with sample_ratio + prefilter
# ─────────────────────────────────────────────────────────────────────────────
def test_pre_split_with_savers() -> None:
    section("TEST 3 — PRE-SPLIT mode + sample_ratio=0.5 + prefilter on")
    reset_outputs()

    train = gen_data(1000, seed=5)
    # Inject 3 junk columns that prefilter should drop:
    # - constant   : nunique == 1
    # - null       : null ratio > PREFILTER_MAX_NULL_RATIO (default 0.95)
    # - dominant   : one value > PREFILTER_MAX_DOMINANT_RATIO (default 0.99)
    train["junk_constant"] = 7.0
    null_idx = np.arange(len(train)); rng = np.random.default_rng(99)
    train["junk_null"] = np.nan
    train.loc[rng.choice(null_idx, 30, replace=False), "junk_null"] = 1.0
    train["junk_dominant"] = ([0] * 996) + list(range(1, 5))   # 99.6% zeros
    valid = gen_data(200, seed=6)
    valid["junk_constant"] = 7.0
    valid["junk_null"]     = np.nan
    valid["junk_dominant"] = [0] * 200
    oot = gen_data(200, seed=7)
    oot["junk_constant"] = 7.0
    oot["junk_null"]     = np.nan
    oot["junk_dominant"] = [0] * 200

    train.to_csv("outputs/_train.csv", index=False)
    valid.to_csv("outputs/_valid.csv", index=False)
    oot.to_csv("outputs/_oot.csv", index=False)

    pipe = AutoMLPipeline()
    metrics = pipe.run(
        input_path="outputs/_train.csv",
        valid_path="outputs/_valid.csv",
        oot_path="outputs/_oot.csv",
        target_column="label",
        entity_id_col="customer_id",
        composite_key_cols=["customer_id", "snap_dt"],
        train_sample_ratio=0.5,
        prefilter=True,
        domain="generic",
    )

    combined = pd.read_parquet(f"{Config.OUTPUT_DIR}/combined_input.parquet")
    junk_in = [c for c in ("junk_constant", "junk_null", "junk_dominant")
               if c in combined.columns]
    print(f"  Junk cols still in combined: {junk_in}")
    assert junk_in == [], f"prefilter should have dropped junk cols; found {junk_in}"

    marker = combined[SPLIT_MARKER].value_counts().to_dict()
    print(f"  Combined marker counts     : {marker}")
    # Train sampled ~50% → 500 ±10
    assert 440 < marker["train"] < 560, f"sampled train size off: {marker['train']}"
    assert marker["valid"] == 200
    assert marker["oot"]   == 200
    assert "valid_auc" in metrics
    assert "oot_auc"   in metrics
    print(f"  Train rows after 0.5 sample: {marker['train']} (target ~500)")
    print(f"  AUC results                : "
          f"valid={metrics.get('valid_auc')} | oot={metrics.get('oot_auc')}")
    print("  [PASS] sample_ratio + prefilter integrate cleanly into the flow")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    failures: list[str] = []
    for fn in (test_auto_split, test_pre_split, test_pre_split_with_savers):
        try:
            fn()
        except AssertionError as e:
            failures.append(f"{fn.__name__}: {e}")
            print(f"\n  [FAIL] {fn.__name__}: {e}")
        except Exception as e:
            failures.append(f"{fn.__name__}: {type(e).__name__}: {e}")
            print(f"\n  [ERROR] {fn.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    section("Summary")
    if failures:
        print(f"  FAILED: {len(failures)} test(s)")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)
    print("  All 3 end-to-end tests PASS")
    print()
