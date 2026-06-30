import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from logger import AgentLogger
from Agents.TrainModel.agent_train_model import TrainModelAgent
from config import Config


def test_agent3():
    print("Testing Agent 3: Model Trainer")

    try:
        Config.validate()
        print("LLM configuration validated\n")
    except ValueError as e:
        print(f"Configuration error: {e}")
        return

    engineered_data_path = "outputs/engineered_data.csv"

    if not Path(engineered_data_path).exists():
        print(f"Error: {engineered_data_path} not found")
        print("  Please run test_agent2.py first to generate engineered data")
        return

    df = pd.read_csv(engineered_data_path)
    print(f"Loaded engineered data from Agent 2")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}\n")

    fe_report_path = Path("outputs/feature_engineer_report.json")
    if fe_report_path.exists():
        with open(fe_report_path, encoding="utf-8") as f:
            previous_report = json.load(f)
        print(f"Loaded FeatureEngineer report from: {fe_report_path}")
        print(f"  entity_id_col     : {previous_report.get('entity_id_col')}")
        print(f"  composite_key_cols: {previous_report.get('composite_key_cols', [])}")
        print(f"  target_column     : {previous_report.get('target_column', 'not set')}\n")
    else:
        previous_report = {
            "agent": "FeatureEngineer",
            "summary": "Created 2 interaction features and encoded categorical variables",
        }
        print("WARNING: feature_engineer_report.json not found — using mock report (no key columns)\n")

    # Optional: load pre-split OOT (supports .csv, .parquet, .feather, S3 paths, …)
    # When provided, agent skips temporal OOT extraction and splits main df 80/20.
    oot_path = Path("outputs/oot_data.csv")
    oot_df = None
    if oot_path.exists():
        oot_df = pd.read_csv(oot_path)
        print(f"Loaded pre-split OOT from: {oot_path}  shape={oot_df.shape}")
        print("  → agent will use 80/20 split on main df (OOT provided externally)\n")
    else:
        print("No pre-split OOT found — agent will auto-detect date col and extract OOT temporally,")
        print("or fall back to 60/20/20 (train/valid/test) if no date column exists.\n")

    target_col = previous_report.get("target_column", "TARGET")

    logger = AgentLogger("outputs/test_agent3.log")
    agent = TrainModelAgent(logger)

    print("Agent 3 is now training models…")
    print("  Pipeline: FLAML → Optuna → RFE → PSI → Stability → SHAP+PSI → Final model\n")

    try:
        final_metrics, report = agent.process(
            df,
            previous_report,
            target_col,
            oot_df=oot_df,
        )

        print("\n" + "=" * 60)
        print("AGENT 3 RESULTS")
        print("=" * 60)

        print(f"\nBest Estimator : {report.get('best_estimator', 'unknown')}")

        # ── Feature pipeline ──────────────────────────────────────────────────
        fp = report.get("feature_pipeline", {})
        n_init      = fp.get("n_init", "N/A")
        n_rfe       = fp.get("n_after_rfe", "N/A")
        n_psi       = fp.get("n_after_psi", "N/A")
        n_stab      = fp.get("n_after_stability", "N/A")
        n_shap_psi  = fp.get("n_after_shap_psi", "N/A")
        n_final     = fp.get("n_final", "N/A")

        print("\nFeature Pipeline:")
        print(f"  {'Init features':<22}: {n_init}")
        print(f"  {'After RFE':<22}: {n_rfe}")
        print(f"  {'After PSI (raw)':<22}: {n_psi}")
        print(f"  {'After Stability':<22}: {n_stab}")
        print(f"  {'After SHAP+PSI prune':<22}: {n_shap_psi}")
        print(f"  {'Final features':<22}: {n_final}")
        print(f"\n  Final feature list ({n_final}):")
        for feat in fp.get("final_features", []):
            print(f"    - {feat}")

        # ── SHAP+PSI prune log ────────────────────────────────────────────────
        prune_log_path = Path("outputs/shap_psi_prune_log.csv")
        if prune_log_path.exists():
            prune_log = pd.read_csv(prune_log_path)
            print(f"\nSHAP+PSI Prune Log  ({len(prune_log)} steps):")
            print(f"  {'Step':<5} {'Removed feature':<35} {'Remaining':<10} {'AUC':<8} {'Improved'}")
            for _, row in prune_log.head(10).iterrows():
                mark = "✓" if row.get("improved") else "✗"
                print(f"  {int(row['step']):<5} {str(row['removed_feature']):<35} "
                      f"{int(row['n_remaining']):<10} {row['candidate_auc']:<8.4f} {mark}")
            if len(prune_log) > 10:
                print(f"  … {len(prune_log) - 10} more steps (see {prune_log_path})")

        # ── Data splits ───────────────────────────────────────────────────────
        splits = report.get("splits", {})
        if splits:
            print("\nData Splits (rows):")
            for split_name, n_rows in splits.items():
                print(f"  {split_name:<20}: {n_rows}")

        # ── Model metrics ─────────────────────────────────────────────────────
        print("\nModel Metrics:")
        auc_keys = [
            ("cv_auc_mean",        "CV AUC (mean)   "),
            ("cv_auc_std",         "CV AUC (std)    "),
            ("valid_temporal_auc", "Valid temporal  "),
            ("valid_random_auc",   "Valid random    "),
            ("valid_auc",          "Valid           "),
            ("oot_auc",            "OOT             "),
            ("test_auc",           "Test (holdout)  "),
        ]
        for key, label in auc_keys:
            val = final_metrics.get(key)
            if isinstance(val, float):
                bar_len = int(max(0.0, min(1.0, val)) * 40)
                bar = "█" * bar_len + "░" * (40 - bar_len)
                print(f"  {label}: {val:.4f}  {bar}")

        # ── Hyperparameters ───────────────────────────────────────────────────
        best_params = report.get("best_params", {})
        if best_params:
            print("\nTop Hyperparameters:")
            for k, v in list(best_params.items())[:6]:
                print(f"  {k}: {v}")

        # ── LLM summary ───────────────────────────────────────────────────────
        print(f"\nLLM Summary:\n  {report.get('summary', 'N/A')}")

        # ── Generated files ───────────────────────────────────────────────────
        print("\nGenerated Files:")
        files = [
            ("outputs/engineered_data.csv",          "Engineered features        — Agent 2 (input)"),
            ("outputs/model_trainer_report.json",     "Model trainer report       — Agent 3"),
            ("outputs/psi_report.csv",                "PSI drift report (raw)     — Agent 3"),
            ("outputs/stability_report.csv",          "Feature stability report   — Agent 3"),
            ("outputs/shap_psi_prune_log.csv",        "SHAP+PSI pruning log       — Agent 3"),
            *Config.chart_files(),                    # 9 model diagnostic charts
            ("outputs/final_model.pkl",               "Trained model artifact     — Agent 3"),
            ("outputs/final_model_code.py",           "Standalone inference code  — Agent 3"),
            ("outputs/final_report.md",               "Full pipeline report"),
            ("outputs/test_agent3.log",               "Agent 3 test execution log"),
        ]
        for filepath, description in files:
            mark = "✓" if Path(filepath).exists() else "✗"
            print(f"  {mark} {filepath:<45} {description}")

        logger.save()
        print(f"\nExecution log saved to: {logger.log_file}")

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_agent3()
