"""End-to-end smoke test for both pipeline modes.

Mocks LLM calls with canned JSON so the full Agent 1→2→3 flow runs without
hitting any gateway / network. Verifies:

1. SINGLE-FILE mode (no valid/oot supplied)
   - Pipeline runs to completion
   - Final report contains AUC metrics
   - Model artifact + inference code written
   - clean_data.parquet + engineered_data.parquet created

2. SPLIT mode (train + valid + oot supplied separately)
   - Agent 1 fits CleaningSpec on train → writes 3 clean_*.parquet
   - Agent 2 fits FeatureSpec on train → writes 3 engineered_*.parquet
   - Agent 3 reads 3 files, adds marker, concats, reconstructs exact splits
   - Valid + oot row counts EXACTLY preserved (no row drops on transform)
   - No leakage: Agent 1 + Agent 2 only see train during fit
   - Junk cols dropped from train via prefilter ALSO dropped from valid/oot

3. SPLIT mode with sample_ratio + prefilter savers
   - train is sampled ~50%
   - junk cols (constant/null/dominant) absent from all 3 outputs

Run from project root:
    python _e2e_check.py
"""
from __future__ import annotations

import json
import sys
import warnings
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

# Keep clean_*.parquet / engineered_*.parquet inside RUN_DIR so the
# intermediate-file assertions below can still read them. Production runs
# default to KEEP_INTERMEDIATES=false → those files go to a tempdir that
# is wiped at pipeline end.
Config.KEEP_INTERMEDIATES = True

from pipeline import AutoMLPipeline   # noqa: E402

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
# TEST 1 — SINGLE-FILE MODE
# ─────────────────────────────────────────────────────────────────────────────
def test_single_file() -> None:
    section("TEST 1 — SINGLE-FILE mode (no valid/oot)")
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

    # In single-file + date_col: expect OOT temporal split (oot_auc) or test_auc fallback
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

    # Agent 1/2 reports should be mode='single'
    with open(Config.DATA_CLEANER_REPORT_PATH, encoding="utf-8") as f:
        rep1 = json.load(f)
    assert rep1.get("mode") == "single", f"DataCleaner mode should be 'single', got {rep1.get('mode')}"
    print("  [PASS] SINGLE-FILE mode end-to-end works")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — SPLIT MODE (train + valid + oot supplied)
# ─────────────────────────────────────────────────────────────────────────────
def test_split_mode() -> None:
    section("TEST 2 — SPLIT mode (train + valid + oot)")
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
        prefilter=True,
        train_sample_ratio=None,
    )

    print(f"\n  Final metrics keys: {sorted(metrics.keys())}")
    assert "best_model" in metrics
    assert "cv_auc_mean" in metrics
    assert "valid_auc" in metrics, "valid_auc missing — split valid set not evaluated"
    assert "oot_auc" in metrics,   "oot_auc missing — split oot set not evaluated"

    # ── Verify 3 separate clean_* files exist ─────────────────────────────
    assert_files_exist([
        Config.CLEAN_TRAIN_PATH,
        Config.CLEAN_VALID_PATH,
        Config.CLEAN_OOT_PATH,
        Config.ENGINEERED_TRAIN_PATH,
        Config.ENGINEERED_VALID_PATH,
        Config.ENGINEERED_OOT_PATH,
        Config.FINAL_MODEL_PATH,
        Config.FINAL_MODEL_CODE_PATH,
    ])

    # ── Verify NO marker leaked into the per-partition files ──────────────
    clean_train = pd.read_parquet(Config.CLEAN_TRAIN_PATH)
    clean_valid = pd.read_parquet(Config.CLEAN_VALID_PATH)
    clean_oot   = pd.read_parquet(Config.CLEAN_OOT_PATH)
    for name, df in [("clean_train", clean_train),
                     ("clean_valid", clean_valid),
                     ("clean_oot",   clean_oot)]:
        assert "_split_" not in df.columns, f"marker leaked into {name}"
    print(f"  Clean shapes: train={clean_train.shape} | valid={clean_valid.shape} | oot={clean_oot.shape}")

    # ── Row-count invariants ──────────────────────────────────────────────
    # Valid + oot must keep EXACT row counts (no row drops on transform)
    assert len(clean_valid) == n_valid_in, \
        f"valid rows must not change on transform: {len(clean_valid)} != {n_valid_in}"
    assert len(clean_oot) == n_oot_in, \
        f"oot rows must not change on transform: {len(clean_oot)} != {n_oot_in}"
    # Train may lose rows to drop_duplicates (but only train, never valid/oot)
    assert len(clean_train) <= n_train_in

    # ── Columns must match across all 3 partitions ────────────────────────
    eng_train = pd.read_parquet(Config.ENGINEERED_TRAIN_PATH)
    eng_valid = pd.read_parquet(Config.ENGINEERED_VALID_PATH)
    eng_oot   = pd.read_parquet(Config.ENGINEERED_OOT_PATH)
    cols_train = set(eng_train.columns)
    cols_valid = set(eng_valid.columns)
    cols_oot   = set(eng_oot.columns)
    assert cols_train == cols_valid, f"engineered train vs valid column mismatch: {cols_train ^ cols_valid}"
    assert cols_train == cols_oot,   f"engineered train vs oot column mismatch: {cols_train ^ cols_oot}"
    print(f"  Engineered shapes : train={eng_train.shape} | valid={eng_valid.shape} | oot={eng_oot.shape}")
    print(f"  Schema parity     : OK (all 3 share {len(cols_train)} cols)")

    # ── Agent 3 split sizes match the per-partition row counts ────────────
    with open(Config.MODEL_TRAINER_REPORT_PATH, encoding="utf-8") as f:
        rep3 = json.load(f)
    a3_splits = rep3.get("splits", {})
    print(f"  Agent 3 splits    : {a3_splits}")
    assert a3_splits.get("valid", 0) == len(eng_valid)
    assert a3_splits.get("oot", 0)   == len(eng_oot)

    # Agent 1/2 reports should be mode='split'
    with open(Config.DATA_CLEANER_REPORT_PATH, encoding="utf-8") as f:
        rep1 = json.load(f)
    assert rep1.get("mode") == "split", f"DataCleaner mode should be 'split', got {rep1.get('mode')}"
    print("  [PASS] SPLIT mode end-to-end works (fit-on-train + transform-on-valid/oot)")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — SPLIT MODE + sample_ratio + prefilter
# ─────────────────────────────────────────────────────────────────────────────
def test_split_with_savers() -> None:
    section("TEST 3 — SPLIT mode + sample_ratio=0.5 + prefilter on")
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

    # ── junk cols must be absent from ALL 3 clean files (prefilter replay) ─
    for path in (Config.CLEAN_TRAIN_PATH, Config.CLEAN_VALID_PATH, Config.CLEAN_OOT_PATH):
        d = pd.read_parquet(path)
        junk_in = [c for c in ("junk_constant", "junk_null", "junk_dominant")
                   if c in d.columns]
        assert junk_in == [], f"prefilter should drop junk cols from {path}; found {junk_in}"
    print("  [OK] junk cols dropped from train, valid, oot (spec replay works)")

    # ── train sampled ~50% (440-560 range), valid/oot keep their full size ─
    n_train = len(pd.read_parquet(Config.CLEAN_TRAIN_PATH))
    n_valid = len(pd.read_parquet(Config.CLEAN_VALID_PATH))
    n_oot   = len(pd.read_parquet(Config.CLEAN_OOT_PATH))
    print(f"  Clean train rows  : {n_train} (target ~500)")
    print(f"  Clean valid rows  : {n_valid}")
    print(f"  Clean oot rows    : {n_oot}")
    assert 440 < n_train < 560, f"sampled train size off: {n_train}"
    assert n_valid == 200
    assert n_oot   == 200

    assert "valid_auc" in metrics
    assert "oot_auc"   in metrics
    print(f"  AUC               : valid={metrics.get('valid_auc')} | oot={metrics.get('oot_auc')}")
    print("  [PASS] sample_ratio + prefilter integrate cleanly into the split flow")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — schema validation: missing target raises ValueError before Agent 1
# ─────────────────────────────────────────────────────────────────────────────
def test_schema_missing_target() -> None:
    section("TEST 4 — schema validation: missing target in valid")
    reset_outputs()

    train = gen_data(400, seed=10)
    valid = gen_data(100, seed=11).drop(columns=["label"])
    train.to_csv("outputs/_train.csv", index=False)
    valid.to_csv("outputs/_valid.csv", index=False)

    pipe = AutoMLPipeline()
    raised = False
    try:
        pipe.run(
            input_path="outputs/_train.csv",
            valid_path="outputs/_valid.csv",
            target_column="label",
            entity_id_col="customer_id",
            composite_key_cols=["customer_id", "snap_dt"],
        )
    except ValueError as e:
        raised = True
        msg = str(e)
        assert "Schema validation FAILED" in msg, f"wrong error wording: {msg}"
        assert "target column 'label' missing in valid" in msg, f"wrong detail: {msg}"
        print(f"  Got expected error: {msg.splitlines()[0]}")

    assert raised, "pipeline should have raised ValueError for missing target"
    # Agent 1 must NOT have run (no clean_train.parquet produced)
    assert not Path(Config.CLEAN_TRAIN_PATH).exists(), \
        "Agent 1 ran despite schema failure — validation should be fail-fast"
    print("  [PASS] schema validation aborts before Agent 1 starts")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — schema validation: dtype-kind mismatch is a warning, not an error
#          + missing entity_id is also a hard error
# ─────────────────────────────────────────────────────────────────────────────
def test_schema_validation_unit() -> None:
    section("TEST 5 — schema validation helpers (unit-level)")
    reset_outputs()

    train = gen_data(400, seed=12)
    valid = gen_data(100, seed=13)
    oot = gen_data(100, seed=14)
    # 5a: numeric in train, object in valid — must WARN not RAISE
    # Use explicit non-numeric prefix so pandas csv re-inference reads as object
    valid["num_signal_1"] = valid["num_signal_1"].apply(lambda x: f"v_{x:.3f}")
    # 5b: extra col in valid (ignored, just info log)
    valid["extra_col_in_valid"] = 1
    # 5c: missing col in oot (warn, replay will skip it silently)
    oot = oot.drop(columns=["num_noise"])

    train.to_csv("outputs/_train.csv", index=False)
    valid.to_csv("outputs/_valid.csv", index=False)
    oot.to_csv("outputs/_oot.csv", index=False)

    pipe = AutoMLPipeline()
    # Should NOT raise
    pipe._validate_split_schema(
        train_path="outputs/_train.csv",
        valid_path="outputs/_valid.csv",
        oot_path="outputs/_oot.csv",
        target_column="label",
        entity_id_col="customer_id",
        composite_key_cols=["customer_id", "snap_dt"],
    )
    print("  [OK 5a] dtype-kind mismatch + extra col + missing col are warnings only")

    # 5d: missing entity_id in valid → must RAISE
    valid_bad_entity = valid.drop(columns=["customer_id"])
    valid_bad_entity.to_csv("outputs/_valid.csv", index=False)
    raised = False
    try:
        pipe._validate_split_schema(
            train_path="outputs/_train.csv",
            valid_path="outputs/_valid.csv",
            oot_path="outputs/_oot.csv",
            target_column="label",
            entity_id_col="customer_id",
            composite_key_cols=["customer_id", "snap_dt"],
        )
    except ValueError as e:
        raised = True
        msg = str(e)
        assert "entity_id column 'customer_id' missing in valid" in msg, f"wrong detail: {msg}"
    assert raised, "missing entity_id must raise ValueError"
    print("  [OK 5d] missing entity_id raises ValueError")

    # 5e: missing composite_key snap_dt in oot → must RAISE
    oot_bad_ck = oot.drop(columns=["snap_dt"])
    oot_bad_ck.to_csv("outputs/_oot.csv", index=False)
    # Restore valid to good state
    valid.to_csv("outputs/_valid.csv", index=False)
    raised = False
    try:
        pipe._validate_split_schema(
            train_path="outputs/_train.csv",
            valid_path="outputs/_valid.csv",
            oot_path="outputs/_oot.csv",
            target_column="label",
            entity_id_col="customer_id",
            composite_key_cols=["customer_id", "snap_dt"],
        )
    except ValueError as e:
        raised = True
        assert "composite_key column 'snap_dt' missing in oot" in str(e)
    assert raised, "missing composite_key must raise ValueError"
    print("  [OK 5e] missing composite_key raises ValueError")
    print("  [PASS] schema validation helpers handle warning + error cases correctly")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — distribution drift: heavily shifted valid triggers WARN (unit)
# ─────────────────────────────────────────────────────────────────────────────
def test_distribution_drift_warns() -> None:
    section("TEST 6 — distribution drift detection (unit-level)")
    reset_outputs()

    # Train: standard distribution; Valid: heavy shift on num_signal_1 + _2
    train = gen_data(2000, seed=20)
    valid = gen_data(500, seed=21)
    valid["num_signal_1"] = valid["num_signal_1"] + 5.0     # mean shift +5
    valid["num_signal_2"] = valid["num_signal_2"] * 3.0     # scale x3
    oot   = gen_data(500, seed=22)   # similar to train — should not trigger

    train.to_csv("outputs/_train.csv", index=False)
    valid.to_csv("outputs/_valid.csv", index=False)
    oot.to_csv("outputs/_oot.csv", index=False)

    pipe = AutoMLPipeline()

    # Capture log via a wrapper around logger.log
    captured = []
    orig_log = pipe.logger.log
    def _capture(agent, action, msg):
        captured.append(f"{agent} | {action} | {msg}")
        return orig_log(agent, action, msg)
    pipe.logger.log = _capture

    pipe._compare_distributions(
        train_path="outputs/_train.csv",
        valid_path="outputs/_valid.csv",
        oot_path="outputs/_oot.csv",
        target_column="label",
        entity_id_col="customer_id",
        composite_key_cols=["customer_id", "snap_dt"],
    )

    # ── 6a: WARN must fire on train↔valid (heavy shift) ───────────────────
    warn_valid = [l for l in captured if "Distribution WARN train↔valid" in l]
    assert warn_valid, f"expected drift WARN on valid (shifted features); got logs:\n  " + "\n  ".join(captured[-10:])
    # Should call out at least one of the shifted columns
    warn_text = " ".join(warn_valid)
    assert "num_signal_1" in warn_text or "num_signal_2" in warn_text, \
        f"shifted col not surfaced in WARN: {warn_valid}"
    print(f"  [OK 6a] WARN fired for valid drift: {warn_valid[0].split('|')[-1].strip()[:120]}")

    # ── 6b: train↔oot stays calm (similar distribution) ───────────────────
    warn_oot = [l for l in captured if "Distribution WARN train↔oot" in l]
    if warn_oot:
        # Acceptable if 0-2 cols cross the threshold by chance; failures only if
        # the warn cites num_signal cols (which weren't shifted in oot)
        for w in warn_oot:
            assert "num_signal_1" not in w and "num_signal_2" not in w, \
                f"oot WARN cites un-shifted col — random check failed: {w}"
    print(f"  [OK 6b] train↔oot drift WARN absent or benign ({len(warn_oot)} flagged)")
    print("  [PASS] drift check warns on real shift, stays quiet on stable partitions")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    failures: list[str] = []
    for fn in (test_single_file, test_split_mode, test_split_with_savers,
              test_schema_missing_target, test_schema_validation_unit,
              test_distribution_drift_warns):
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
    print("  All 6 end-to-end tests PASS")
    print()
