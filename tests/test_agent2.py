import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
from logger import AgentLogger
from agent_feature_engineer import FeatureEngineerAgent
from config import Config

def test_agent2():
    print("Testing Agent 2: Feature Engineer")    
    try:
        Config.validate()
        print("OpenAI configuration validated\n")
    except ValueError as e:
        print(f"Configuration error: {e}")
        return
    
    clean_data_path = "outputs/clean_data.csv"
    
    if not Path(clean_data_path).exists():
        print(f"Error: {clean_data_path} not found")
        print("  Please run test_agent1.py first to generate clean data")
        return
    
    df = pd.read_csv(clean_data_path)
    print(f"Loaded clean data from Agent 1")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}\n")
    
    previous_report = {
        "agent": "DataCleaner",
        "summary": "Dropped 2 columns (id, useless_col) and imputed missing values in 3 columns"
    }
    
    logger = AgentLogger("outputs/test_agent2.log")
    agent = FeatureEngineerAgent(logger)
    
    print("Agent 2 is now engineering features...")
    
    try:
        engineered_data_path, report = agent.process(df, previous_report, "target")
        
        print("AGENT 2 RESULTS")
        
        print(f"\nEngineered data saved to: {engineered_data_path}")
        print(f"Report saved to: outputs/feature_engineer_report.json")
        
        print(f"\nShape Change: {report['original_shape']} -> {report['final_shape']}")
        
        print("\nActions Taken:")
        for i, action in enumerate(report['actions_taken'], 1):
            print(f"  {i}. {action}")
        
        print(f"\nSummary: {report['summary']}")
        
        print(f"\nFinal Features: {', '.join(report['final_features'])}")
        
        engineered_df = pd.read_csv(engineered_data_path)
        print("\nEngineered Data Sample (first 5 rows):")
        print(engineered_df.head())
        
        logger.save()
        print(f"\nExecution log saved to: {logger.log_file}")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent2()