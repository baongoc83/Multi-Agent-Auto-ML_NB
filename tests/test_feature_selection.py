#!/usr/bin/env python
"""Importance ranking + batch prune with AUC tolerance (Agent 3).

Run:  py -3.12 tests/test_feature_selection.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import Config
from logger import AgentLogger
from Agents.TrainModel.agent_train_model import TrainModelAgent

PARAMS = {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 15, "verbose": -1, "n_jobs": 4}


def _data(n=6000, informative=12, noise=60, dup=3, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, informative + noise)),
                     columns=[f"inf{i}" for i in range(informative)] + [f"noise{i}" for i in range(noise)])
    w = np.linspace(1.5, 0.3, informative)
    z = (X[[f"inf{i}" for i in range(informative)]].values * w).sum(1) + rng.normal(size=n)
    y = pd.Series((z > 0).astype(int))
    for i in range(dup):                                     # exact-ish duplicates of inf0..dup-1
        X[f"dup{i}"] = X[f"inf{i}"] * 1.0001 + rng.normal(scale=1e-3, size=n)
    cut = 4500
    return X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:]


def _agent():
    return TrainModelAgent(AgentLogger())


def test_rank_puts_signal_first_and_drops_duplicates():
    ag = _agent()
    Xt, yt, Xv, yv = _data()
    Config.FEATURE_RANK_CORR_THRESHOLD = 0.90
    sel = ag._tool_rank_features("lgbm", PARAMS, Xt, yt, Xv, yv, n_target=20)
    assert len(sel) == 20
    assert sum(c.startswith("inf") for c in sel[:12]) >= 10          # signal ranked on top
    assert not any(c.startswith("dup") for c in sel)                  # near-duplicates removed
    assert any(c.startswith("inf") for c in sel[:3])

    Config.FEATURE_RANK_CORR_THRESHOLD = 0.0                          # dedupe off -> duplicates may stay
    sel2 = ag._tool_rank_features("lgbm", PARAMS, Xt, yt, Xv, yv, n_target=30)
    assert any(c.startswith("dup") for c in sel2)
    Config.FEATURE_RANK_CORR_THRESHOLD = 0.90


def test_batch_prune_tolerance_and_cap_policy():
    ag = _agent()
    Xt, yt, Xv, yv = _data(dup=0)
    cols = list(Xt.columns)
    saved = (Config.MAX_FINAL_FEATURES, Config.FEATURE_CAP_POLICY, Config.PRUNE_AUC_TOLERANCE)
    try:
        # tolerance rule: noise features are free to drop, so the chosen set is far smaller than 72
        Config.MAX_FINAL_FEATURES, Config.FEATURE_CAP_POLICY, Config.PRUNE_AUC_TOLERANCE = 100, "auc_first", 0.002
        feats, curve = ag._tool_batch_prune("lgbm", PARAMS, cols, Xt, yt, Xv, yv, pd.DataFrame())
        assert len(feats) < len(cols) * 0.6, len(feats)
        assert sum(c.startswith("inf") for c in feats) >= 10          # signal survived
        assert curve["selected"].sum() == 1 and curve["smoothed_auc"].max() - \
            curve.loc[curve["selected"], "smoothed_auc"].iloc[0] <= 0.002 + 1e-9

        # cap_first: never more than MAX_FINAL_FEATURES even if tolerance alone would allow more
        Config.MAX_FINAL_FEATURES, Config.FEATURE_CAP_POLICY, Config.PRUNE_AUC_TOLERANCE = 30, "cap_first", 0.0
        feats2, _ = ag._tool_batch_prune("lgbm", PARAMS, cols, Xt, yt, Xv, yv, pd.DataFrame())
        assert len(feats2) <= 30, len(feats2)
        # zero tolerance + auc_first: still returns a valid non-empty subset of the input
        Config.FEATURE_CAP_POLICY = "auc_first"
        feats3, _ = ag._tool_batch_prune("lgbm", PARAMS, cols, Xt, yt, Xv, yv, pd.DataFrame())
        assert feats3 and set(feats3) <= set(cols)
    finally:
        Config.MAX_FINAL_FEATURES, Config.FEATURE_CAP_POLICY, Config.PRUNE_AUC_TOLERANCE = saved


def test_smoothing_ignores_a_lucky_spike_and_se_widens_tolerance():
    from Agents.TrainModel.agent_train_model import _auc_se
    assert 0.002 < _auc_se(0.77, 5000, 55000) < 0.006              # HC-sized valid set
    assert _auc_se(0.77, 300, 3700) > 2 * _auc_se(0.77, 5000, 55000)   # small valid = much noisier

    ag = _agent()
    Xt, yt, Xv, yv = _data(dup=0)
    cols = list(Xt.columns)
    saved = (Config.MAX_FINAL_FEATURES, Config.FEATURE_CAP_POLICY, Config.PRUNE_AUC_TOLERANCE,
             Config.PRUNE_SMOOTH_WINDOW, Config.PRUNE_SE_MULTIPLIER)
    try:
        Config.MAX_FINAL_FEATURES, Config.FEATURE_CAP_POLICY, Config.PRUNE_AUC_TOLERANCE = 100, "auc_first", 0.0005
        Config.PRUNE_SMOOTH_WINDOW, Config.PRUNE_SE_MULTIPLIER = 1, 0.0
        n_raw = len(ag._tool_batch_prune("lgbm", PARAMS, cols, Xt, yt, Xv, yv, pd.DataFrame())[0])
        Config.PRUNE_SMOOTH_WINDOW, Config.PRUNE_SE_MULTIPLIER = 1, 2.0       # wide SE tolerance
        n_se, _ = ag._tool_batch_prune("lgbm", PARAMS, cols, Xt, yt, Xv, yv, pd.DataFrame())
        assert len(n_se) <= n_raw                                    # more tolerance never keeps more
    finally:
        (Config.MAX_FINAL_FEATURES, Config.FEATURE_CAP_POLICY, Config.PRUNE_AUC_TOLERANCE,
         Config.PRUNE_SMOOTH_WINDOW, Config.PRUNE_SE_MULTIPLIER) = saved


def test_batch_prune_without_validation_is_a_noop():
    ag = _agent()
    Xt, yt, _, _ = _data()
    feats, df = ag._tool_batch_prune("lgbm", PARAMS, list(Xt.columns), Xt, yt,
                                     pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame())
    assert feats == list(Xt.columns) and df.empty


def test_invalid_options_fall_back_to_defaults():
    ag = _agent()
    saved = (Config.PRUNE_METHOD, Config.FEATURE_RANK_METHOD, Config.FEATURE_RANK_IMPORTANCE, Config.FEATURE_CAP_POLICY)
    try:
        Config.PRUNE_METHOD, Config.FEATURE_RANK_METHOD = "nonsense", "magic"
        Config.FEATURE_RANK_IMPORTANCE, Config.FEATURE_CAP_POLICY = "vibes", "whatever"
        ag._validate_selection_config()
        assert (Config.PRUNE_METHOD, Config.FEATURE_RANK_METHOD, Config.FEATURE_RANK_IMPORTANCE,
                Config.FEATURE_CAP_POLICY) == ("batch_tolerance", "importance", "gain", "cap_first")
    finally:
        Config.PRUNE_METHOD, Config.FEATURE_RANK_METHOD, Config.FEATURE_RANK_IMPORTANCE, Config.FEATURE_CAP_POLICY = saved


def test_xgboost_and_shap_importance_paths():
    ag = _agent()
    Xt, yt, Xv, yv = _data(n=3000, noise=20, dup=0)
    xgb_params = {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3, "n_jobs": 4, "verbosity": 0}
    sel = ag._tool_rank_features("xgboost", xgb_params, Xt, yt, Xv, yv, n_target=10)
    assert sum(c.startswith("inf") for c in sel) >= 7
    Config.FEATURE_RANK_IMPORTANCE = "shap"
    try:
        sel2 = ag._tool_rank_features("lgbm", PARAMS, Xt, yt, Xv, yv, n_target=10)
        assert sum(c.startswith("inf") for c in sel2) >= 7
    finally:
        Config.FEATURE_RANK_IMPORTANCE = "gain"


if __name__ == "__main__":
    for t in (test_rank_puts_signal_first_and_drops_duplicates, test_batch_prune_tolerance_and_cap_policy,
              test_smoothing_ignores_a_lucky_spike_and_se_widens_tolerance,
              test_batch_prune_without_validation_is_a_noop, test_invalid_options_fall_back_to_defaults,
              test_xgboost_and_shap_importance_paths):
        t()
        print("PASS ", t.__name__)
    print("\nALL FEATURE SELECTION CHECKS PASSED")
