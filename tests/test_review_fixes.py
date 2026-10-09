#!/usr/bin/env python
"""Fixes from the 2026-10-08 run review: Stage-1b awareness in Agent 2, note actions,
final-CV tree headroom, in-sample metrics kept away from the LLM summary.

Run:  py -3.12 tests/test_review_fixes.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import Config
from logger import AgentLogger
from Agents.BaseAgent.base_agent import BaseAgent
from Agents.FeatureEngineer.agent_feature_engineer import FeatureEngineerAgent, FeatureSpec
from Agents.TrainModel.agent_train_model import TrainModelAgent


def _fe_agent(df):
    ag = FeatureEngineerAgent.__new__(FeatureEngineerAgent)
    ag.logger, ag.name = AgentLogger(), "FeatureEngineer"
    ag.df, ag.target_column = df, "y"
    ag._protected_cols = set()
    ag.create_interactions_enabled = True
    ag.null_context = {"imputed": {"ext2": "median"}, "kept_nan": ["ext1"], "indicators": ["ext2_missing"]}
    return ag


def test_agent2_skips_missing_flags_on_imputed_columns():
    df = pd.DataFrame({"ext1": [0.1, np.nan, 0.3, 0.5], "ext2": [0.2, 0.4, 0.6, 0.1],
                       "ext2_missing": [0, 1, 0, 0], "y": [0, 1, 0, 1]})
    ag = _fe_agent(df)
    spec = FeatureSpec(target_column="y")
    resp = json.dumps({"actions": [
        {"action": "create_interaction", "new_col": "miss2", "expression": "df['ext2'].isna().astype(int)"},
        {"action": "create_interaction", "new_col": "miss1", "expression": "df[\"ext1\"].isna().astype(int)"},
        {"action": "flag_for_review", "column": "ext1", "reason": "check source"},
    ]})
    ag._execute_llm_decisions(resp, spec=spec)
    names = [n for n, _, _ in spec.interactions]
    assert names == ["miss1"], names                               # imputed column skipped, NaN column kept
    log = "\n".join(f"{e['action']} {e['details']}" for e in ag.logger.logs)
    assert "already imputed" in log and "ext2_missing" in log
    assert "LLM Note" in log and "Unknown or missing action_type 'flag_for_review'" not in log


def test_prompt_mentions_stage_1b():
    df = pd.DataFrame({"ext1": [0.1, np.nan], "ext2": [0.2, 0.4], "y": [0, 1]})
    ag = _fe_agent(df)
    ag._col_descriptions = {}
    prompt = ag._build_engineering_prompt({"shape": [2, 3]}, {"summary": "s"})
    assert "NULL HANDLING ALREADY APPLIED" in prompt and "ext2_missing" in prompt


def _clf_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i}" for i in range(8)])
    y = ((X["f0"] + 0.7 * X["f1"] + rng.normal(size=n)) > 0).astype(int)
    df = X.assign(target=y)
    return df.iloc[:2000].reset_index(drop=True), df.iloc[2000:].reset_index(drop=True)


def _replay(headroom):
    tr, te = _clf_data()
    empty = pd.DataFrame(columns=tr.columns)
    splits = {"train": tr, "valid": te.iloc[:500], "test": te.iloc[500:], "oot": empty.copy(),
              "valid_temporal": empty.copy(), "valid_random": empty.copy()}
    saved = (Config.FINAL_ES_TREE_HEADROOM, Config.MULTI_SEED_N)
    Config.FINAL_ES_TREE_HEADROOM, Config.MULTI_SEED_N = headroom, 1
    try:
        ag = TrainModelAgent(AgentLogger())
        m = ag.replay_fit(splits=splits, target_column="target", estimator_name="lgbm",
                          best_params={"n_estimators": 20, "learning_rate": 0.03, "num_leaves": 7, "verbose": -1},
                          feature_cols=[f"f{i}" for i in range(8)])
        return m, "\n".join(f"{e['action']} {e['details']}" for e in ag.logger.logs)
    finally:
        Config.FINAL_ES_TREE_HEADROOM, Config.MULTI_SEED_N = saved


def test_cv_tree_headroom_and_cap_warning():
    m1, log1 = _replay(1.0)
    assert m1["best_iteration"] <= 20 and "stopped at the tree ceiling" in log1   # legacy: capped, now warned
    m2, _ = _replay(2.0)
    assert m2["best_iteration"] > 20                                              # ES had room to choose
    assert m2["valid_metrics_in_sample"] is True


def test_llm_summary_never_sees_in_sample_metrics():
    ag = TrainModelAgent(AgentLogger())
    seen = {}
    ag.call_llm = lambda prompt, system: seen.setdefault("p", prompt) or "ok"
    metrics = {"best_params": {}, "valid_auc": 0.89, "valid_brier": 0.05, "test_auc": 0.78, "cv_auc_mean": 0.779}
    ag._generate_llm_summary("lgbm", {}, metrics, 10, 8, 8, 8, 6, pd.DataFrame(), pd.DataFrame())
    assert "valid_auc" not in seen["p"] and "test_auc" in seen["p"]


def test_clip_refused_on_zero_inflated_column():
    # 2026-10-09 run: graph_1hop_* columns had q1=q3=0, the LLM clipped them and every
    # non-zero value (the 8.5% signal) became 0 on train, valid and oot.
    from Agents.DataCleaner.agent_data_cleaner import DataCleanerAgent, CleaningSpec
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": np.where(rng.random(1000) < 0.085, rng.lognormal(10, 1, 1000), 0.0),
                       "amt": rng.lognormal(10, 0.5, 1000)})
    ag = DataCleanerAgent.__new__(DataCleanerAgent)
    ag.logger, ag.name, ag.df = AgentLogger(), "DataCleaner", df.copy()
    ag._composite_key_cols, ag._entity_id_col, ag._target_column = [], None, None
    rep = json.loads(ag._tool_detect_outliers(df, "g"))
    assert "outlier_count" not in rep and "do NOT clip" in rep["note"]
    spec = CleaningSpec()
    ag._execute_llm_decisions(json.dumps({"actions": [
        {"action": "clip_outliers", "column": "g", "factor": 3.0},
        {"action": "clip_outliers", "column": "amt", "factor": 3.0}]}), spec=spec)
    assert "g" not in spec.clip_bounds and "amt" in spec.clip_bounds
    assert (ag.df["g"] > 0).sum() == (df["g"] > 0).sum()          # signal intact
    log = "\n".join(f"{e['action']} {e['details']}" for e in ag.logger.logs)
    assert "SKIP clip_outliers" in log and "IQR=0" in log


def test_data_quality_warnings_reach_reports():
    import tempfile
    from pipeline import AutoMLPipeline
    pl = AutoMLPipeline.__new__(AutoMLPipeline)
    pl.logger, pl._dq_issues = AgentLogger(), []
    pl._record_dq("Stage 0b distribution", "psi_drift", "warn", "valid",
                  "2 column(s) with PSI > 0.25 vs train", {"g1": 9.9, "g2": 1.2})
    pl._record_null_issues("oot", [
        {"column": "amt", "check": "scale_unit_change", "severity": "critical", "message": "amt: magnitude shifted x1000"},
        {"column": "g", "check": "scale_distribution_shift", "severity": "warn", "message": "g: distribution shifted x47"},
        {"column": "h", "check": "scale_distribution_shift", "severity": "warn", "message": "h: distribution shifted x12"}])
    pl.logger.log("DataCleaner", "SKIP clip_outliers", "'g' has IQR=0 — refused")
    md = pl._render_data_quality_section()
    assert "1 critical · 2 warning(s) · 1 other notice(s)" in md
    assert md.index("critical |") < md.index("psi_drift")                 # critical rows first
    assert "2 column(s) shifted median or p95" in md and "SKIP clip_outliers" in md
    saved = Config.DATA_QUALITY_REPORT_PATH
    Config.DATA_QUALITY_REPORT_PATH = str(Path(tempfile.mkdtemp()) / "dq.json")
    try:
        pl._save_dq_report()
        body = json.loads(Path(Config.DATA_QUALITY_REPORT_PATH).read_text(encoding="utf-8"))
    finally:
        Config.DATA_QUALITY_REPORT_PATH = saved
    assert body["summary"] == {"critical": 1, "warn": 2}
    assert body["issues"][0]["columns"] == {"g1": 9.9, "g2": 1.2} and body["issues"][0]["hint"]


if __name__ == "__main__":
    assert "flag_for_review" in BaseAgent.NOTE_ACTIONS
    for t in (test_agent2_skips_missing_flags_on_imputed_columns, test_prompt_mentions_stage_1b,
              test_cv_tree_headroom_and_cap_warning, test_llm_summary_never_sees_in_sample_metrics,
              test_clip_refused_on_zero_inflated_column, test_data_quality_warnings_reach_reports):
        t()
        print("PASS ", t.__name__)
    print("\nALL REVIEW-FIX CHECKS PASSED")
