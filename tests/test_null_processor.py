#!/usr/bin/env python
"""Tests for preprocessing.NullProcessor / suggest_rules.

Run:  py -3.12 tests/test_null_processor.py
"""
import json
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from preprocessing import (NullConfigError, NullContractError, NullProcessor,
                           suggest_rules)

CONFIG = {
    "feature_groups": {"numerical": ["income", "txn_count_3m", "ratio"],
                       "categorical": ["segment"], "binary": ["has_card"]},
    "group_defaults": {"numerical": {"missing_strategy": "median"}},
    "features": {
        "income": {"missing_type": "unknown", "add_missing_indicator": True},
        "txn_count_3m": {"missing_type": "behavioral", "missing_strategy": "zero"},
        "ratio": {"add_missing_indicator": True},
        "segment": {"missing_strategy": "explicit_category"},
        "has_card": {"missing_strategy": "missing_indicator_only",
                     "add_missing_indicator": True},
    },
}


def _train():
    return pd.DataFrame({
        "income": [10.0, 20.0, None, 30.0, 40.0],
        "txn_count_3m": [3, None, 5, 0, 2],
        "ratio": [0.1, np.inf, 0.3, None, 0.5],
        "segment": ["a", None, "b", "a", "b"],
        "has_card": [1, 0, None, 1, 0],
    })


def test_fit_on_train_only_and_indicator_before_impute():
    p = NullProcessor(CONFIG).fit(_train())
    valid = pd.DataFrame({"income": [None, 1000.0], "txn_count_3m": [None, 1],
                          "ratio": [0.2, 0.2], "segment": [None, "zz"],
                          "has_card": [None, 1]})
    out = p.transform(valid)
    assert p.params_["income"]["imputation_value"] == 25.0       # TRAIN median
    assert out["income"].tolist() == [25.0, 1000.0]              # not valid's own median
    assert out["income_missing"].tolist() == [1, 0]              # computed before fill
    assert out["txn_count_3m"].tolist() == [0.0, 1.0]            # explicit zero
    assert out["segment"].tolist() == ["__MISSING__", "zz"]
    assert out["has_card"].isna().tolist() == [True, False]      # indicator-only keeps NaN
    assert out["has_card_missing"].tolist() == [1, 0]
    assert "txn_count_3m_missing" not in out.columns


def test_null_is_not_zero_unless_configured():
    p = NullProcessor(CONFIG).fit(_train())
    out = p.transform(_train())
    assert (out["income"] != 0).all()                            # median, never 0


def test_inf_counts_as_missing():
    p = NullProcessor(CONFIG).fit(_train())
    assert p.params_["ratio"]["stats"]["missing_rate"] == 0.4    # None + inf
    out = p.transform(_train())
    assert np.isfinite(out["ratio"]).all()
    assert out["ratio_missing"].tolist() == [0, 1, 0, 1, 0]


def test_train_statistics_complete():
    p = NullProcessor(CONFIG).fit(_train())
    stats = p.train_statistics().set_index("feature")
    for col in ("dtype", "missing_rate", "zero_rate", "min", "max", "mean", "median",
                "p25", "p75", "p95", "unique_count", "imputation_value"):
        assert col in stats.columns, col
    assert stats.loc["income", "median"] == 25.0
    assert stats.loc["txn_count_3m", "zero_rate"] == 0.2


def test_fail_closed():
    p = NullProcessor(CONFIG).fit(_train())
    try:
        p.transform(_train().drop(columns=["income"]))
        raise AssertionError("missing column must raise")
    except NullContractError:
        pass
    dirty = _train()
    dirty["income"] = dirty["income"].astype(object)
    dirty.loc[0, "income"] = "N/A"
    try:
        p.transform(dirty)
        raise AssertionError("unparseable numeric must raise")
    except NullContractError:
        pass
    try:
        NullProcessor(CONFIG).fit(_train().assign(extra=1))
        raise AssertionError("unconfigured column must raise")
    except NullConfigError:
        pass
    NullProcessor(CONFIG, on_unconfigured="ignore").fit(_train().assign(extra=1))


def test_config_validation():
    bad = [
        {"features": {"a": {"group": "categorical", "missing_strategy": "median"}}},
        {"features": {"a": {"group": "numerical", "missing_strategy": "constant"}}},
        {"features": {"a": {"group": "numerical", "missing_strategy": "missing_indicator_only"}}},
        {"features": {"a": {"group": "numerical", "missing_strategy": "median",
                            "missing_type": "weird"}}},
        {"features": {"a": {"group": "numerical", "missing_strategy": "nope"}}},
    ]
    for cfg in bad:
        try:
            NullProcessor(cfg).fit(pd.DataFrame({"a": [1.0, None]}))
            raise AssertionError(f"accepted invalid config {cfg}")
        except NullConfigError:
            pass


def test_all_null_train_column_rejected():
    cfg = {"features": {"a": {"group": "numerical", "missing_strategy": "median"}}}
    try:
        NullProcessor(cfg).fit(pd.DataFrame({"a": [None, None]}, dtype="float64"))
        raise AssertionError("100% null train column with median must raise")
    except NullConfigError:
        pass


def test_drift_warning():
    p = NullProcessor(CONFIG).fit(_train())
    shifted = _train()
    shifted["income"] = np.nan
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p.transform(shifted)
    assert any("income" in str(x.message) and "drift" in str(x.message) for x in w)
    assert p.report_["alerts"]


def test_serialize_roundtrip_and_tamper():
    p = NullProcessor(CONFIG).fit(_train())
    d = Path(tempfile.mkdtemp())
    digest = p.save(d / "null.json")
    q = NullProcessor.load(d / "null.json", expected_sha256=digest)
    a, b = p.transform(_train()), q.transform(_train())
    pd.testing.assert_frame_equal(a, b)
    raw = json.loads((d / "null.json").read_text(encoding="utf-8"))
    raw["features"]["income"]["imputation_value"] = 0.0
    (d / "tampered.json").write_text(json.dumps(raw), encoding="utf-8")
    try:
        NullProcessor.load(d / "tampered.json")
        raise AssertionError("tampered artifact must be rejected")
    except NullContractError:
        pass


def test_sklearn_pipeline_and_names():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    p = Pipeline([("nulls", NullProcessor(CONFIG)),
                  ("noop", FunctionTransformer(validate=False))]).fit(_train())
    out = p.transform(_train())
    assert out.shape[0] == 5
    names = p.named_steps["nulls"].get_feature_names_out().tolist()
    assert names[-3:] == ["income_missing", "ratio_missing", "has_card_missing"]


def test_advisor_validates_llm_and_falls_back():
    df = _train()

    def fake_llm(prompt, system):
        assert '"a"' not in prompt and '"b"' not in prompt          # no category values, stats only
        return json.dumps({"features": {
            "income": {"missing_type": "structural", "missing_strategy": "median",
                       "add_missing_indicator": True, "reason": "unknown income"},
            "txn_count_3m": {"missing_type": "behavioral", "missing_strategy": "zero",
                             "add_missing_indicator": False, "reason": "no txn"},
            "ratio": {"missing_type": "data_quality", "missing_strategy": "explicit_category",
                      "add_missing_indicator": False},          # invalid for numerical
        }})

    cfg = suggest_rules(df, llm_call=fake_llm)["features"]
    assert cfg["income"]["source"] == "llm" and "reviewed" not in cfg["income"]
    assert cfg["ratio"]["source"] == "heuristic"                     # invalid -> fallback
    assert cfg["segment"]["missing_strategy"] == "explicit_category"
    NullProcessor({"features": cfg}).fit(df)                         # drafts are loadable

    def broken_llm(prompt, system):
        raise RuntimeError("down")
    cfg2 = suggest_rules(df, llm_call=broken_llm)["features"]
    assert all(v["source"] == "heuristic" for v in cfg2.values())


def test_guardrails_replace_human_review():
    df = pd.DataFrame({"bal": [-5.0, 3.0, None, 8.0], "cnt": [0, 2, None, 5],
                       "allnull": [None] * 4}, dtype="float64")

    def llm(prompt, system):
        zero = {"missing_type": "behavioral", "missing_strategy": "zero",
                "add_missing_indicator": False}
        return json.dumps({"features": {
            "bal": zero,                                   # negatives, no zeros -> zero rejected
            "cnt": zero,                                   # count with zeros -> zero kept
            "allnull": {"missing_type": "unknown", "missing_strategy": "median",
                        "add_missing_indicator": True}}})

    cfg = suggest_rules(df, llm_call=llm)["features"]
    assert cfg["bal"]["missing_strategy"] == "median" and cfg["bal"]["add_missing_indicator"]
    assert cfg["cnt"]["missing_strategy"] == "zero"
    assert cfg["allnull"]["missing_strategy"] == "none"
    with warnings.catch_warnings(record=True) as w:        # no "unreviewed" nagging
        warnings.simplefilter("always")
        NullProcessor({"features": cfg}).fit(df)
    assert not any("review" in str(x.message) for x in w)


def test_pipeline_stage_and_replay_hook():
    """Stage 1b on parquet partitions: keys/target untouched, holdouts use TRAIN median."""
    import os
    os.environ["OUTPUT_DIR"] = tempfile.mkdtemp(prefix="null_stage_")
    from config import Config
    Config.init_run()
    Config.NULL_PROCESSOR_USE_LLM = False
    import pipeline as P
    from logger import AgentLogger
    pl = P.AutoMLPipeline.__new__(P.AutoMLPipeline)
    pl.logger, pl._split_key_cols = AgentLogger(), ["cid"]
    d = Path(tempfile.mkdtemp())
    tr = pd.DataFrame({"cid": [1, 2, 3, 4], "x": [1.0, None, 3.0, 5.0], "c": ["a", None, "b", "a"],
                       "target": [0, 1, 0, 1]})
    va = pd.DataFrame({"cid": [5, 6], "x": [None, 100.0], "c": [None, "b"], "target": [0, 1]})
    paths = {"train": str(d / "tr.parquet"), "valid": str(d / "va.parquet")}
    tr.to_parquet(paths["train"], index=False)
    va.to_parquet(paths["valid"], index=False)
    info = pl._apply_null_processing(paths, "target", protected={"cid"})
    out_va = pd.read_parquet(paths["valid"])
    assert out_va["x"].tolist() == [3.0, 100.0]                  # TRAIN median, not valid's
    assert out_va["cid"].tolist() == [5, 6] and out_va["target"].tolist() == [0, 1]
    assert info["sha256"] and Path(Config.NULL_PROCESSOR_PATH).exists()
    # replay hook reproduces the same transform from the saved artifact
    import importlib.util
    spec = importlib.util.spec_from_file_location("drv", Path(__file__).resolve().parents[1] / "replay_driver.py")
    drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
    drv._HERE = Path(Config.NULL_PROCESSOR_PATH).parent
    again = drv._apply_nulls(va.copy(), {"null_processing": info})
    pd.testing.assert_frame_equal(again.reset_index(drop=True), out_va, check_dtype=False)


def _money_frame(n=500, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"amt": rng.lognormal(10, 0.6, n) * scale,
                         "cnt": rng.poisson(3, n).astype(float)})


SCALE_CFG = {"features": {"amt": {"group": "numerical", "missing_strategy": "median"},
                          "cnt": {"group": "numerical", "missing_strategy": "median"}}}


def test_scale_guard_catches_unit_change_only():
    p = NullProcessor(SCALE_CFG).fit(_money_frame(seed=0))
    p.transform(_money_frame(seed=1))                              # ordinary resample: fine
    p.transform(_money_frame(seed=2, scale=1.4))                   # 40% inflation: fine
    for factor in (0.001, 1000.0, 0.05):                           # unit changes
        try:
            p.transform(_money_frame(seed=3, scale=factor))
            raise AssertionError(f"x{factor} must be refused")
        except NullContractError as e:
            assert "amt" in str(e) and "cnt" not in str(e)
    with warnings.catch_warnings(record=True) as w:                # 5x: warn, not raise
        warnings.simplefilter("always")
        p.transform(_money_frame(seed=4, scale=5.0))
    assert any("scale" in str(x.message) for x in w)


def test_scale_guard_skips_small_batches_and_can_be_disabled():
    p = NullProcessor(SCALE_CFG).fit(_money_frame())
    p.transform(_money_frame(n=20, scale=1000.0))                  # single/online scoring
    q = NullProcessor(SCALE_CFG, scale_fail_ratio=0).fit(_money_frame())
    q.transform(_money_frame(scale=0.001))                         # disabled


def test_scale_guard_survives_serialization():
    p = NullProcessor(SCALE_CFG).fit(_money_frame())
    d = Path(tempfile.mkdtemp())
    p.save(d / "n.json")
    q = NullProcessor.load(d / "n.json")
    try:
        q.transform(_money_frame(scale=0.001))
        raise AssertionError("loaded artifact must still guard")
    except NullContractError:
        pass


def test_indicator_hygiene_dedup_constant_and_cap():
    rng = np.random.default_rng(0)
    n = 2000
    miss = rng.random(n) < 0.2
    y = (rng.random(n) < np.where(miss, 0.5, 0.05)).astype(int)    # missingness is informative
    df = pd.DataFrame({"a": np.where(miss, np.nan, 1.0), "a_twin": np.where(miss, np.nan, 2.0),
                       "never": 1.0, "noise": np.where(rng.random(n) < 0.3, np.nan, 1.0)})
    cfg = {"features": {c: {"group": "numerical", "missing_strategy": "median",
                            "add_missing_indicator": True} for c in df.columns}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = NullProcessor(cfg).fit(df, y)
    ind = set(p.indicator_columns_)
    assert "never_missing" not in ind                              # constant
    assert len(ind & {"a_missing", "a_twin_missing"}) == 1         # duplicate pattern collapsed
    assert "noise_missing" in ind
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = NullProcessor(cfg, max_indicators=1).fit(df, y)
    assert q.indicator_columns_ == ["a_missing"] or q.indicator_columns_ == ["a_twin_missing"]   # informative one wins
    out = q.transform(df)
    assert [c for c in out.columns if c.endswith("_missing")] == q.indicator_columns_


def test_advisor_salvages_truncated_json():
    df = pd.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": [1.0, None, 2.0, 5.0]})
    cut = ('{"features": {"a": {"missing_type": "unknown", "missing_strategy": "median", '
           '"add_missing_indicator": true, "reason": "x"}, "b": {"missing_type": "unkn')
    cfg = suggest_rules(df, llm_call=lambda p, s: cut)["features"]
    assert cfg["a"]["source"] == "llm" and cfg["b"]["source"] == "heuristic"


def test_guard_keeps_rules_valid_for_binary_and_all_rules_validate():
    df = pd.DataFrame({"flag": [1.0, 2.0, None, 1.0, 2.0, 1.0],
                       "amt": [-5.0, 3.0, None, 8.0, 2.0, 9.0]})
    zero = {"missing_type": "behavioral", "missing_strategy": "zero", "add_missing_indicator": False}
    cfg = suggest_rules(df, llm_call=lambda p, s: json.dumps({"features": {"flag": zero, "amt": zero}}))
    NullProcessor(cfg).fit(df)                                   # must not raise
    assert cfg["features"]["flag"]["missing_strategy"] == "missing_indicator_only"
    assert cfg["features"]["amt"]["missing_strategy"] == "median"


def test_scale_guard_ignores_sparse_columns_but_catches_their_unit_change():
    rng = np.random.default_rng(0)
    def sparse(n, scale=1.0, nz_share=0.4):
        v = rng.lognormal(4, 0.5, n) * scale
        return pd.DataFrame({"s": np.where(rng.random(n) < nz_share, v, 0.0)})
    cfg = {"features": {"s": {"group": "numerical", "missing_strategy": "median"}}}
    p = NullProcessor(cfg).fit(sparse(5000))
    p.transform(sparse(5000, nz_share=0.4))                          # same scale: fine
    p.transform(sparse(5000, nz_share=0.005))                        # p95 collapses to 0: not a unit change
    try:
        p.transform(sparse(5000, scale=1000))
        raise AssertionError("x1000 on a sparse column must be refused")
    except NullContractError:
        pass


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(tests)} NULL PROCESSOR CHECKS PASSED")


if __name__ == "__main__":
    main()
