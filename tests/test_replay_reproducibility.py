#!/usr/bin/env python
"""Regression test for the replay bundle's two reproducibility traps.

Both of these were real failures caught while building the bundle, and both are
silent — the replay runs fine, produces plausible numbers, and is simply a
different model. Nothing downstream complains. So they get a test.

TRAP 1 — row order inside a partition.
    `auto_split` returns train shuffled by `train_test_split`, and `valid` as
    valid_temporal concatenated with valid_random. Neither is in input-file
    order. StratifiedKFold assigns folds by position, so a replay that rebuilds
    the right ROWS in the wrong ORDER trains on different folds and lands on a
    different best_iteration. Guarded by the `_split_pos_` column.

TRAP 2 — train-only row ops.
    CleaningSpec.apply() deliberately skips drop_duplicates / dedup / sampling
    because replaying them on valid/oot would delete evaluation rows. But that
    means replaying the spec on TRAIN yields more rows than the original run
    trained on. Guarded by CleaningSpec.row_ops + apply_row_ops().

Run:  py -3.12 tests/test_replay_reproducibility.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd

import splitting
from config import Config

Config.init_run()

from logger import AgentLogger
from Agents.DataCleaner.agent_data_cleaner import CleaningSpec
from Agents.FeatureEngineer.agent_feature_engineer import FeatureEngineerAgent, FeatureSpec
from Agents.TrainModel.agent_train_model import TrainModelAgent

MARKER, POS = "_split_", "_split_pos_"
KEY = ["cust_id", "snap_date"]


def _make_raw(n=9000, seed=11):
    rng = np.random.default_rng(seed)
    months = pd.period_range("2023-01", periods=12, freq="M")
    df = pd.DataFrame({
        "cust_id": np.arange(n),
        "snap_date": pd.PeriodIndex(rng.choice(months, n)).to_timestamp(how="end").normalize(),
        "income": rng.lognormal(10, 0.6, n).round(2),
        "debt": rng.lognormal(9, 0.8, n).round(2),
        "age": rng.integers(21, 70, n),
        "junk": 1.0,
    })
    lin = 0.9 * np.log1p(df.debt) / np.log1p(df.income) + 0.02 * (70 - df.age) / 10
    df["target"] = (rng.random(n) < 1 / (1 + np.exp(-(lin - lin.mean()) * 2))).astype(int)
    # Exact duplicate rows so TRAP 2's drop_duplicates actually has work to do.
    return pd.concat([df, df.iloc[:400]], ignore_index=True)


def _build_run(raw, src_path):
    """Do what a real pipeline run does, and emit the replay bundle."""
    # Use the pipeline's own split + freeze code, not a copy of it — a test
    # that reimplements the thing it is testing proves nothing.
    import pipeline as P
    pl = P.AutoMLPipeline.__new__(P.AutoMLPipeline)
    pl.logger = AgentLogger()
    pl._split_key_cols, pl._split_key_positional = [], False

    staged = raw.copy()
    key_cols, is_positional = pl._resolve_split_key(list(staged.columns), "cust_id", KEY)
    assert key_cols == KEY and not is_positional
    staged[P._KEY_OCC_COL] = staged.groupby(key_cols, sort=False,
                                            dropna=False).cumcount()
    assert (staged[P._KEY_OCC_COL] > 0).any(), (
        "test setup is weak: no duplicate keys, so the occurrence tiebreak is untested")

    res = splitting.auto_split(staged, target_column="target",
                               date_col="snap_date", temporal_freq="auto",
                               week_closing_day="")
    parts = {"train": res.train, "valid": res.valid, "oot": res.oot, "test": res.test}
    pl._write_split_assignment(parts, key_cols, is_positional)

    parts = {t: p.drop(columns=[c for c in (P._KEY_OCC_COL,) if c in p.columns])
             for t, p in parts.items()}

    cspec = CleaningSpec()
    cspec.drops = ["junk"]
    cspec.row_ops = [{"op": "drop_duplicates"}]
    joblib.dump(cspec, Config.PIPELINE_PROCESS_DC_SPEC_PATH)

    clean = {t: cspec.apply(p.copy()) for t, p in parts.items() if len(p)}
    n_before = len(clean["train"])
    clean["train"] = cspec.apply_row_ops(clean["train"], target_column="target")
    assert len(clean["train"]) < n_before, (
        "test setup is weak: drop_duplicates removed nothing, so TRAP 2 is untested")

    ag2 = FeatureEngineerAgent.__new__(FeatureEngineerAgent)
    ag2.logger = AgentLogger()
    ag2.name = "FeatureEngineer"
    ag2.target_column = "target"
    ag2._protected_cols = set(KEY)
    ag2._last_iv_scores, ag2._last_woe_data = {}, {}
    ag2._last_woe_maps, ag2._last_applied_woe = {}, {}
    ag2._last_interaction_fill = None

    tr = clean["train"].copy()
    fspec = FeatureSpec(target_column="target")
    expr = "df['debt'] / (df['income'] + 1)"
    tr = ag2._tool_create_interaction(tr, "dti", expr)
    fspec.interactions.append(("dti", expr, ag2._last_interaction_fill))
    ag2._tool_compute_iv(tr, "target", bins=10, method="quantile")
    tr = ag2._tool_apply_woe_transform(tr, min_iv=Config.WOE_MIN_IV)
    fspec.woe_maps.update(ag2._last_applied_woe)
    fspec.selected_features = list(tr.columns)
    joblib.dump(fspec, Config.PIPELINE_PROCESS_FE_SPEC_PATH)

    eng = {"train": tr}
    for t in ("valid", "oot", "test"):
        if t in clean:
            eng[t] = fspec.apply(clean[t].copy())

    feat = [c for c in eng["train"].columns if c not in ("target", *KEY)]
    empty = pd.DataFrame(columns=eng["train"].columns)
    splits = {"train": eng["train"], "valid": eng.get("valid", empty.copy()),
              "oot": eng.get("oot", empty.copy()), "test": eng.get("test", empty.copy()),
              "valid_temporal": empty.copy(), "valid_random": empty.copy()}

    agent = TrainModelAgent(AgentLogger())
    est = "lgbm"
    params = {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 15}
    metrics = agent.replay_fit(splits={k: v.copy() for k, v in splits.items()},
                               target_column="target", estimator_name=est,
                               best_params=params, feature_cols=feat,
                               date_col="snap_date")
    agent._save_model(feat)

    manifest = {
        "schema_version": 1, "created_at": "regression-test",
        "run_dir": Config.RUN_DIR, "source_input": str(src_path),
        "pipeline_mode": "single-file (auto-split)", "target_column": "target",
        "entity_id_col": "cust_id", "composite_key_cols": KEY,
        "split": {"assignment_file": Path(Config.SPLIT_ASSIGNMENT_PATH).name,
                  "key_cols": KEY, "positional_fallback": False,
                  "temporal": res.temporal_meta,
                  "sizes": {k: len(v) for k, v in eng.items()}},
        "cleaning": {"spec_file": Path(Config.PIPELINE_PROCESS_DC_SPEC_PATH).name,
                     "row_ops": cspec.row_ops},
        "feature_engineering": {"spec_file": Path(Config.PIPELINE_PROCESS_FE_SPEC_PATH).name,
                                "n_final_features": len(feat)},
        "training": {"estimator": est, "best_params": params, "final_features": feat,
                     "n_final_features": len(feat),
                     "model_file": Path(Config.FINAL_MODEL_PATH).name,
                     "calibration": None, "random_state": Config.RANDOM_STATE,
                     "multi_seed_n": Config.MULTI_SEED_N,
                     "cv_n_splits": Config.CV_N_SPLITS},
        "expected_metrics": metrics,
    }
    with open(Config.REPLAY_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    repo = Path(__file__).resolve().parents[1]
    Path(Config.REPLAY_DRIVER_PATH).write_text(
        (repo / "replay_driver.py").read_text(encoding="utf-8"), encoding="utf-8")
    return manifest, metrics


def _load_driver():
    import importlib.util
    spec = importlib.util.spec_from_file_location("drv", Config.REPLAY_DRIVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _numeric(metrics):
    return {k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def main():
    src = Path(tempfile.mkdtemp(prefix="replay_test_")) / "raw.parquet"
    raw = _make_raw()
    raw.to_parquet(src, index=False)
    print(f"raw {raw.shape} (incl. 400 duplicate rows) -> {src}")

    manifest, original = _build_run(raw, src)
    print(f"original run: {Config.RUN_DIR}")
    print(f"  train rows={manifest['split']['sizes']['train']} "
          f"best_iteration={original.get('best_iteration')}")

    drv = _load_driver()

    # ── 1. full replay reproduces every metric ───────────────────────────
    replayed = {}
    labelled = drv._assign_splits(raw.copy(), manifest)
    splits = {}
    for tag in ("train", "valid", "oot", "test"):
        part = labelled[labelled[MARKER] == tag].sort_values(POS, kind="mergesort")
        part = part.drop(columns=[c for c in (MARKER, POS) if c in part.columns])
        splits[tag] = drv._replay_specs(part.reset_index(drop=True), manifest,
                                        tag == "train").reset_index(drop=True)
    empty = pd.DataFrame(columns=splits["train"].columns)
    splits["valid_temporal"] = empty.copy()
    splits["valid_random"] = empty.copy()

    tr = manifest["training"]
    replayed = TrainModelAgent(AgentLogger()).replay_fit(
        splits=splits, target_column="target", estimator_name=tr["estimator"],
        best_params=tr["best_params"], feature_cols=tr["final_features"],
        date_col="snap_date", calibration=None)

    a, b = _numeric(original), _numeric(replayed)
    diffs = {k: (a[k], b[k]) for k in a if k in b and abs(a[k] - b[k]) > 1e-9}
    assert not diffs, f"replay did not reproduce: {diffs}"
    print(f"PASS 1  replay reproduced all {len(a)} numeric metrics exactly "
          f"(oot_auc={b.get('oot_auc')}, best_iteration={b.get('best_iteration')})")

    # ── 2. TRAP 1: dropping the order column must change the result ──────
    # If this ever stops changing anything, _split_pos_ has become dead weight
    # and the guard is no longer guarding.
    shuffled = {k: v.copy() for k, v in splits.items()}
    shuffled["train"] = (splits["train"]
                         .sort_values(list(splits["train"].columns)[0], kind="mergesort")
                         .reset_index(drop=True))
    out_of_order = TrainModelAgent(AgentLogger()).replay_fit(
        splits=shuffled, target_column="target", estimator_name=tr["estimator"],
        best_params=tr["best_params"], feature_cols=tr["final_features"],
        date_col="snap_date", calibration=None)
    c = _numeric(out_of_order)
    changed = {k for k in a if k in c and abs(a[k] - c[k]) > 1e-9}
    assert changed, ("row order no longer affects the fit — either the model became "
                     "order-invariant or the test lost its teeth; re-check whether "
                     "_split_pos_ is still needed")
    print(f"PASS 2  TRAP 1 live: re-ordering train changed {len(changed)} metric(s) "
          f"({sorted(changed)[:3]}...) — _split_pos_ is load-bearing")

    # ── 3. TRAP 2: apply() alone must leave train too big ────────────────
    cspec = joblib.load(Config.PIPELINE_PROCESS_DC_SPEC_PATH)
    assert cspec.row_ops, "row_ops was not persisted into cleaning_spec.pkl"
    train_keys = labelled[labelled[MARKER] == "train"]
    raw_train = raw.merge(train_keys[KEY], on=KEY, how="inner")
    no_rowops = cspec.apply(raw_train.copy())
    with_rowops = cspec.apply_row_ops(cspec.apply(raw_train.copy()),
                                      target_column="target")
    assert len(with_rowops) < len(no_rowops), (
        "apply_row_ops dropped nothing — TRAP 2 is untested here")
    assert len(with_rowops) == manifest["split"]["sizes"]["train"], (
        f"row-op replay gave {len(with_rowops)} train rows, original run had "
        f"{manifest['split']['sizes']['train']}")
    print(f"PASS 3  TRAP 2 live: apply() alone gives {len(no_rowops)} train rows, "
          f"apply_row_ops() gives {len(with_rowops)} = original")

    # ── 4. apply() must never touch holdout rows ─────────────────────────
    oot_keys = labelled[labelled[MARKER] == "oot"]
    raw_oot = raw.merge(oot_keys[KEY], on=KEY, how="inner")
    assert len(cspec.apply(raw_oot.copy())) == len(raw_oot), (
        "CleaningSpec.apply() dropped holdout rows — that silently flatters metrics")
    print(f"PASS 4  apply() left all {len(raw_oot)} oot rows intact")

    # ── 5. TRAP 3: categorical encoding must not depend on file format ──
    # The pipeline writes parquet between agents, so Agent 2 fits on a
    # post-parquet frame while a replay feeds it the source CSV. `astype(str)`
    # renders a null as "nan" from CSV but "None" from parquet, which silently
    # encoded the same missing value two different ways. Found on real Home
    # Credit data: 5 columns diverged and shifted every downstream metric.
    from Agents.FeatureEngineer.agent_feature_engineer import _as_encoder_tokens

    cat = pd.DataFrame({
        "cat_a": ["A", "B", None, "C", np.nan, "A", "B", None],
        "cat_b": ["x", None, "y", "y", "x", None, "z", "x"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    })
    scratch = Path(tempfile.mkdtemp(prefix="fmt_"))
    cat.to_csv(scratch / "d.csv", index=False)
    cat.to_parquet(scratch / "d.parquet", index=False)
    from_csv = pd.read_csv(scratch / "d.csv")
    from_pq = pd.read_parquet(scratch / "d.parquet")

    raw_csv = set(from_csv["cat_a"].astype(str))
    raw_pq = set(from_pq["cat_a"].astype(str))
    assert raw_csv != raw_pq, (
        "test setup is weak: this pandas/pyarrow spells nulls identically via "
        "both formats, so the trap cannot fire here")
    assert (set(_as_encoder_tokens(from_csv["cat_a"]))
            == set(_as_encoder_tokens(from_pq["cat_a"]))), (
        "_as_encoder_tokens failed to normalise nulls across formats")

    ag = FeatureEngineerAgent.__new__(FeatureEngineerAgent)
    ag.logger = AgentLogger()
    ag.name = "FeatureEngineer"
    ag.target_column = "target"
    ag._protected_cols = set()
    ag._last_label_encoder = None

    fitted = from_pq.copy()
    fspec2 = FeatureSpec(target_column="target")
    for col in ("cat_a", "cat_b"):
        fitted = ag._tool_encode_categorical(fitted, col, "label")
        fspec2.label_encoders[col] = ag._last_label_encoder
    replayed = fspec2.apply(from_csv.copy())

    for col in ("cat_a", "cat_b"):
        assert fitted[col].tolist() == replayed[col].tolist(), (
            f"{col}: fit-on-parquet {fitted[col].tolist()} != "
            f"replay-from-csv {replayed[col].tolist()} — encoding is still "
            "format-dependent")
    print("PASS 5  TRAP 3 live: raw astype(str) differs across formats "
          f"({sorted(raw_csv - raw_pq)} vs {sorted(raw_pq - raw_csv)}), "
          "but encoded values match")

    print("\nALL REPLAY REPRODUCIBILITY CHECKS PASSED")


if __name__ == "__main__":
    main()
