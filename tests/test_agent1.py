import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from logger import AgentLogger
from agent_data_cleaner import DataCleanerAgent
from config import Config

def create_sample_data():
    """create a sample messy dataset for testing."""
    np.random.seed(42)
    
    data = {
        'id': range(1, 101),  # useless ID column
        'age': [np.random.randint(18, 80) if np.random.random() > 0.2 else np.nan for _ in range(100)],
        'income': [np.random.randint(20000, 150000) if np.random.random() > 0.15 else np.nan for _ in range(100)],
        'category': [np.random.choice(['A', 'B', 'C', None], p=[0.4, 0.3, 0.15, 0.15]) for _ in range(100)],
        'useless_col': [np.nan] * 100,  # 100% missing
        'target': np.random.choice([0, 1], size=100)  # target variable
    }
    
    df = pd.DataFrame(data)
    
    Path("outputs").mkdir(exist_ok=True)
    
    # Save sample data
    df.to_csv("outputs/sample_data.csv", index=False)
    print("Created sample_data.csv with intentional issues:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Missing values in 'age': {df['age'].isnull().sum()}")
    print(f"  - Missing values in 'income': {df['income'].isnull().sum()}")
    print(f"  - Missing values in 'category': {df['category'].isnull().sum()}")
    print(f"  - Missing values in 'useless_col': {df['useless_col'].isnull().sum()}")
    print(f"  - ID column: present (should be dropped)")
    
    return "outputs/sample_data.csv"

def test_agent1():
    print("Testing Agent 1: Data Cleaner")

    try:
        Config.validate()
        print("OpenAI configuration validated\n")
    except ValueError as e:
        print(f"Configuration error: {e}")
        return
    
    input_csv = create_sample_data()
    
    logger = AgentLogger("outputs/test_agent1.log")
    agent = DataCleanerAgent(logger)
    
    print("Agent 1 is now processing the data...")
    
    try:
        clean_data_path, report = agent.process(input_csv)
        
        print("Agent 1 results")
        
        print(f"\nCleaned data saved to: {clean_data_path}")
        print(f"Report saved to: outputs/data_cleaner_report.json")
        
        print(f"\nShape Change: {report['original_shape']} -> {report['final_shape']}")
        
        print("\nActions Taken:")
        for i, action in enumerate(report['actions_taken'], 1):
            print(f"  {i}. {action}")
        
        print(f"\nSummary: {report['summary']}")
        
        print(f"\nRemaining Columns: {', '.join(report['columns_remaining'])}")
        
        cleaned_df = pd.read_csv(clean_data_path)
        print("\nCleaned Data Sample (first 5 rows):")
        print(cleaned_df.head())
        
        print("\nCleaned Data Info:")
        print(f"  - Missing values: {cleaned_df.isnull().sum().sum()}")
        
        logger.save()
        print(f"\nExecution log saved to: {logger.log_file}")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent1()