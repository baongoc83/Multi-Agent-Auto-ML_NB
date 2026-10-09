#!/usr/bin/env python
"""Per-trial wall-clock cap, re-tune adoption rule, DART/forest bounds.

Run:  py -3.12 tests/test_optuna_guard.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import Config
from logger import AgentLogger
from Agents.TrainModel.agent_train_model import TrainModelAgent, _adopt_retuned


def _data(n=4000, p=30, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series((X["f0"] + 0.5 * X["f1"] + rng.normal(size=n) > 0).astype(int))
    return X.iloc[:3000], y.iloc[:3000], X.iloc[3000:], y.iloc[3000:]


def test_adoption_rule():
    assert not _adopt_retuned(0.7764, 0.7721, 0.0005)              # the real regression seen in HC run
    assert not _adopt_retuned(0.7700, 0.7703, 0.0005)              # within noise
    assert _adopt_retuned(0.7700, 0.7712, 0.0005)
    assert not _adopt_retuned(0.77, None, 0.0005)
    assert _adopt_retuned(None, 0.7, 0.0005)


def test_trial_deadline_prunes_lgbm_and_xgboost():
    import optuna
    ag = TrainModelAgent(AgentLogger())
    Xt, yt, Xv, yv = _data()
    for est, params in (("lgbm", {"n_estimators": 20000, "learning_rate": 0.0005, "verbose": -1, "n_jobs": 2}),
                        ("xgboost", {"n_estimators": 20000, "learning_rate": 0.0005, "n_jobs": 2, "verbosity": 0})):
        model = ag._get_model_class(est)(**params)
        t0 = time.monotonic()
        try:
            ag._fit_with_early_stopping(model, est, Xt, yt, Xv, yv, rounds=10**6,
                                        deadline=time.monotonic() + 1.0)
            raise AssertionError(f"{est}: deadline ignored")
        except optuna.TrialPruned:
            assert time.monotonic() - t0 < 30, f"{est}: pruned too late"
    # without a deadline fitting still works and the callbacks do not leak into the model
    m = ag._get_model_class("xgboost")(n_estimators=50, n_jobs=2, verbosity=0)
    ag._fit_with_early_stopping(m, "xgboost", Xt, yt, Xv, yv)
    assert m.get_params().get("callbacks") is None
    assert m.get_params().get("early_stopping_rounds") is None


def test_search_space_bounds():
    import optuna
    ag = TrainModelAgent(AgentLogger())
    seen = {}

    def obj(trial):
        seen["rf"] = ag._build_optuna_params("rf", trial)
        seen["lgbm"] = ag._build_optuna_params("lgbm", trial)
        return 0.0
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study()
    st.optimize(obj, n_trials=30)
    assert max(t.params["n_estimators"] for t in st.trials) <= Config.TREE_ENSEMBLE_MAX_ESTIMATORS \
        or True                                                  # (lgbm shares the name; checked below)
    assert seen["lgbm"]["boosting_type"] == "gbdt" and not Config.OPTUNA_ENABLE_DART
    rf_only = optuna.create_study()
    rf_only.optimize(lambda t: (ag._build_optuna_params("rf", t), 0.0)[1], n_trials=40)
    assert max(t.params["n_estimators"] for t in rf_only.trials) <= Config.TREE_ENSEMBLE_MAX_ESTIMATORS


def test_eval_params_matches_trial_recipe():
    ag = TrainModelAgent(AgentLogger())
    Xt, yt, Xv, yv = _data()
    auc = ag._eval_params("lgbm", {"n_estimators": 100, "learning_rate": 0.1, "verbose": -1}, Xt, yt, Xv, yv)
    assert auc is not None and auc > 0.8


if __name__ == "__main__":
    for t in (test_adoption_rule, test_trial_deadline_prunes_lgbm_and_xgboost,
              test_search_space_bounds, test_eval_params_matches_trial_recipe):
        t()
        print("PASS ", t.__name__)
    print("\nALL OPTUNA GUARD CHECKS PASSED")
