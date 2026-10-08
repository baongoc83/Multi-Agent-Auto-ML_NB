#!/usr/bin/env python
"""FeatureSpec.apply must not fail silently at scoring time.

Run:  py -3.12 tests/test_feature_contract.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from Agents.FeatureEngineer.agent_feature_engineer import (FeatureContractError,
                                                          FeatureSpec)


def _spec_with_interaction():
    return FeatureSpec(interactions=[("ratio", "df['a'] / (df['b'] + 1)", 0.5)])


def test_interaction_fallback_is_loud():
    spec = _spec_with_interaction()
    bad = pd.DataFrame({"a": [1.0, 2.0]})                      # 'b' absent upstream
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = spec.apply(bad.copy())                           # no logger, non-strict
    assert out["ratio"].tolist() == [0.5, 0.5]
    assert any("ratio" in str(x.message) for x in w), "fallback must warn even without logger"
    assert spec.last_diagnostics["interaction_fallbacks"][0]["feature"] == "ratio"
    try:
        spec.apply(bad.copy(), strict=True)
        raise AssertionError("strict must raise")
    except FeatureContractError:
        pass
    ok = spec.apply(pd.DataFrame({"a": [1.0], "b": [1.0]}), strict=True)
    assert ok["ratio"].iloc[0] == 0.5


def test_onehot_is_batch_independent():
    spec = FeatureSpec(onehot_columns={"c": ["c_B", "c_C"]})   # train levels A(baseline),B,C
    full = spec.apply(pd.DataFrame({"c": ["A", "B", "C"]}))
    part = spec.apply(pd.DataFrame({"c": ["B", "C", "B"]}))
    one = spec.apply(pd.DataFrame({"c": ["B"]}))
    assert full[["c_B", "c_C"]].astype(int).values.tolist() == [[0, 0], [1, 0], [0, 1]]
    assert part[["c_B", "c_C"]].astype(int).values.tolist() == [[1, 0], [0, 1], [1, 0]]
    assert one[["c_B", "c_C"]].astype(int).values.tolist() == [[1, 0]]


def test_replay_order_follows_fit_order():
    le = LabelEncoder().fit(["A", "B", "C", "__NA__"])
    raw = pd.DataFrame({"seg": ["A", "B", "C"], "amt": [10.0] * 3})
    kw = dict(interactions=[("x", "df['seg'] * df['amt']", 0.0)], label_encoders={"seg": le})
    fitted = FeatureSpec(step_order=["encode", "interaction", "woe"], **kw)
    assert fitted.apply(raw.copy(), strict=True)["x"].tolist() == [0.0, 10.0, 20.0]
    legacy = FeatureSpec(**kw)                                  # default old order
    legacy.apply(raw.copy())                                    # falls back (warns)
    assert legacy.last_diagnostics["interaction_fallbacks"]
    del fitted.step_order                                       # specs pickled before the field
    fitted.apply(raw.copy())                                    # must not crash


def test_unseen_rate_guard():
    le = LabelEncoder().fit(["A", "B", "__NA__"])
    spec = FeatureSpec(label_encoders={"c": le})
    df = pd.DataFrame({"c": ["a", "b", "a", "b"]})              # recoded upstream: all unseen
    assert spec.apply(df.copy())["c"].nunique() == 1            # old behaviour: silently constant
    assert spec.last_diagnostics["unseen_rate"]["c"] == 1.0
    try:
        spec.apply(df.copy(), strict=True, max_unseen_rate=0.2)
        raise AssertionError("must refuse")
    except FeatureContractError:
        pass


def test_missing_configured_column_strict():
    spec = FeatureSpec(woe_maps={"w": {"edges": [-np.inf, 0, np.inf], "woe": [0.1, -0.1, 0.0]}},
                       selected_features=["w", "z"])
    try:
        spec.apply(pd.DataFrame({"z": [1]}), strict=True)
        raise AssertionError("absent WoE column must raise")
    except FeatureContractError as e:
        assert "w" in str(e)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(tests)} FEATURE CONTRACT CHECKS PASSED")


if __name__ == "__main__":
    main()
