import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from logger import AgentLogger
from agent_train_model import TrainModelAgent
from config import Config

def test_agent3():
    print("Testing Agent 3: Model Trainer")
    
    try:
        Config.validate()
        print("OpenAI configuration validated\n")
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
    print(f"  Features: {list(df.columns)}\n")
    
    fe_report_path = Path("outputs/feature_engineer_report.json")
    if fe_report_path.exists():
        with open(fe_report_path, encoding="utf-8") as f:
            previous_report = json.load(f)
        print(f"Loaded FeatureEngineer report from: {fe_report_path}")
        print(f"  entity_id_col     : {previous_report.get('entity_id_col')}")
        print(f"  composite_key_cols: {previous_report.get('composite_key_cols', [])}\n")
    else:
        previous_report = {
            "agent": "FeatureEngineer",
            "summary": "Created 2 interaction features and encoded categorical variables",
        }
        print("WARNING: feature_engineer_report.json not found — using mock report (no key columns)\n")
    
    logger = AgentLogger("outputs/test_agent3.log")
    # agent = ModelTrainerAgent(logger)
    agent = TrainModelAgent(logger)
    
    print("Agent 3 is now training models with feedback loop...")
    
    try:
        final_metrics, report = agent.process(df, previous_report, "target")

        print("\nAGENT 3 RESULTS")
        print(f"Report saved to: outputs/model_trainer_report.json")

        # Best model
        print(f"\nBest Estimator : {report.get('best_estimator', 'unknown')}")

        # Feature pipeline reduction
        fp = report.get("feature_pipeline", {})
        print("\nFeature Pipeline:")
        print(f"  Init features    : {fp.get('n_init', 'N/A')}")
        print(f"  After RFE        : {fp.get('n_after_rfe', 'N/A')}")
        print(f"  After PSI        : {fp.get('n_after_psi', 'N/A')}")
        print(f"  After Stability  : {fp.get('n_after_stability', 'N/A')}")
        print(f"  Final features   : {fp.get('n_final', 'N/A')}")
        print(f"  Final feature list: {fp.get('final_features', [])}")

        # Data split sizes
        splits = report.get("splits", {})
        if splits:
            print("\nData Splits (rows):")
            for split_name, n_rows in splits.items():
                print(f"  {split_name:<18}: {n_rows}")

        # Metrics — final_metrics is the same object as report["metrics"]
        print("\nModel Metrics:")
        for key in ("cv_auc_mean", "cv_auc_std", "valid_temporal_auc",
                    "valid_random_auc", "valid_auc", "oot_auc"):
            val = final_metrics.get(key)
            if isinstance(val, float):
                print(f"  {key:<22}: {val:.4f}")

        # Best params (top 6)
        best_params = report.get("best_params", {})
        if best_params:
            print("\nTop Hyperparameters:")
            for k, v in list(best_params.items())[:6]:
                print(f"  {k}: {v}")

        print(f"\nSummary: {report.get('summary', 'N/A')}")

        logger.save()
        print(f"\nExecution log saved to: {logger.log_file}")

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent3()