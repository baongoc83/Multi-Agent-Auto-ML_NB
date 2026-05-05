import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
from pathlib import Path
from logger import AgentLogger
from agent_model_trainer import ModelTrainerAgent
from config import Config
import json

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
    
    previous_report = {
        "agent": "FeatureEngineer",
        "summary": "Created 2 interaction features and encoded categorical variables"
    }
    
    logger = AgentLogger("outputs/test_agent3.log")
    agent = ModelTrainerAgent(logger)
    
    print("Agent 3 is now training models with feedback loop...")
    
    try:
        final_metrics, report = agent.process(df, previous_report, "target")
        
        print("AGENT 3 RESULTS")
        
        print(f"\nTraining complete!")
        print(f"Report saved to: outputs/model_trainer_report.json")
        print(f"Final code saved to: outputs/final_model_code.py")
        
        print(f"\nTotal Iterations: {report['total_iterations']}")
        
        print("\nTraining History:")
        for entry in report['training_history']:
            print(f"  Iteration {entry['iteration']} (best: {entry['metrics'].get('best_model', 'unknown')}):")
            for metric, value in entry['metrics'].items():
                if isinstance(value, float):
                    print(f"    - {metric}: {value:.4f}")
            comparison = entry['metrics'].get('model_comparison', {})
            if comparison:
                primary = 'roc_auc_score' if any('roc_auc_score' in v for v in comparison.values()) else 'r2'
                scores = ', '.join(f"{m}: {v.get(primary, 'N/A'):.4f}" if isinstance(v.get(primary), float) else f"{m}: N/A" for m, v in comparison.items())
                print(f"    - comparison ({primary}): {scores}")

        print(f"\nBest Model: {final_metrics.get('best_model', 'unknown')}")
        print("\nFinal Metrics:")
        for metric, value in final_metrics.items():
            if isinstance(value, float):
                print(f"  - {metric}: {value:.4f}")
        comparison = final_metrics.get('model_comparison', {})
        if comparison:
            primary = 'roc_auc_score' if any('roc_auc_score' in v for v in comparison.values()) else 'r2'
            print(f"  - model_comparison ({primary}):")
            for model_name, model_metrics in comparison.items():
                val = model_metrics.get(primary)
                print(f"      {model_name}: {val:.4f}" if isinstance(val, float) else f"      {model_name}: N/A")

        print(f"\nSummary: {report['summary']}")

        final_code_path = Path("outputs/final_model_code.py")
        if final_code_path.exists():
            final_code = final_code_path.read_text()
            print("\nFinal Model Code (first 500 chars):")
            print(final_code[:500] + ("..." if len(final_code) > 500 else ""))
        else:
            print(f"\nFinal model code not found at {final_code_path}")
        
        logger.save()
        print(f"\nExecution log saved to: {logger.log_file}")
        
        print("FEEDBACK LOOP VERIFICATION")
        if report['total_iterations'] > 1:
            print(f"Agent made {report['total_iterations']} attempts")
            print("Feedback loop worked: Agent iterated to improve performance")
        else:
            print("Agent achieved satisfactory performance in first attempt")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent3()