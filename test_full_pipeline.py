import pandas as pd
import numpy as np
from pathlib import Path
from pipeline import AutoMLPipeline
from config import Config

def create_realistic_dataset():
    np.random.seed(42)
    n_samples = 200
    
    age = np.random.randint(18, 70, n_samples)
    experience = np.clip(age - 22 + np.random.randint(-3, 10, n_samples), 0, 50)
    education = np.random.choice([1, 2, 3, 4], n_samples, p=[0.15, 0.55, 0.2, 0.1])  # 1=HS, 2=Bachelor, 3=Master, 4=PhD
    
    # income based on education, experience, and age with some noise
    base_income = 30000 + (education * 15000) + (experience * 1000) + (age * 500)
    income = base_income + np.random.normal(0, 15000, n_samples)
    income = np.clip(income, 25000, 300000)
    
    # create target: 1 if high performer (income > median and experience > 10)
    # target = ((income > np.median(income)) & (experience > 10)).astype(int)
    latent = np.random.normal(0, 0.2, len(income))

    score = (
            0.52 * (income > np.median(income)).astype(float) +
            0.32 * (experience > 10).astype(float) +
            0.16 * latent
    )
    prob = 1 / (1 + np.exp(-3 * (score - 0.5)))
    target = (np.random.rand(len(prob)) < prob).astype(int)
    # add some categorical features
    department = np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], n_samples)
    location = np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples)
    
    # add some missing values
    age_mask = np.random.random(n_samples) < 0.15
    age = age.astype(float)
    age[age_mask] = np.nan
    
    income_mask = np.random.random(n_samples) < 0.10
    income[income_mask] = np.nan
    
    # create useless columns
    id_col = range(1, n_samples + 1)
    random_col = np.random.random(n_samples)
    mostly_missing = [np.nan] * int(n_samples * 0.9) + list(np.random.random(int(n_samples * 0.1)))
    np.random.shuffle(mostly_missing)
    
    # build dataframe
    df = pd.DataFrame({
        'employee_id': id_col,
        'age': age,
        'years_experience': experience,
        'education_level': education,
        'annual_income': income,
        'department': department,
        'location': location,
        'random_noise': random_col,
        'useless_feature': mostly_missing,
        'high_performer': target
    })
    
    return df

def main():
    print("FULL PIPELINE TEST - Multi-Agent AutoML System")
    
    try:
        Config.validate()
        print("Test LLM AI Models configuration validated\n")
    except ValueError as e:
        print(f"Configuration error: {e}")
        return
    
    print("Creating realistic test dataset...")
    df = create_realistic_dataset()
    
    Path("outputs").mkdir(exist_ok=True)
    test_data_path = "outputs/test_employee_data.csv"
    df.to_csv(test_data_path, index=False)
    
    print(f"Test dataset created: {test_data_path}")
    print(f"  - Shape: {df.shape}")
    print(f"  - Features: {list(df.columns)}")
    print(f"  - Target: high_performer (binary classification)")
    print(f"  - Missing values: {df.isnull().sum().sum()}")
    print(f"  - Target distribution: {df['high_performer'].value_counts().to_dict()}")
    print()
    
    print("STARTING THE MULTI-AGENT PIPELINE")
    print("\nThe three agents will now work sequentially:")
    print("  1. Data Cleaner: Audit and clean the data")
    print("  2. Feature Engineer: Create and select features")
    print("  3. Model Trainer: Train and compare multiple ML models")
    print()
    
    pipeline = AutoMLPipeline()
    
    try:
        final_metrics = pipeline.run(test_data_path, "high_performer")
        
        print("PIPELINE COMPLETED")
        
        print(f"\n Final Model Performance (best: {final_metrics.get('best_model', 'unknown')}):")
        for metric, value in final_metrics.items():
            if isinstance(value, float):
                bar_length = int(max(0.0, min(1.0, value)) * 50)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"  {metric:12s}: {value:.4f} {bar}")
        comparison = final_metrics.get('model_comparison', {})
        if comparison:
            primary = 'roc_auc_score' if any('roc_auc_score' in v for v in comparison.values()) else 'r2'
            print(f"\n  Model Comparison ({primary}):")
            for model_name, model_metrics in comparison.items():
                val = model_metrics.get(primary)
                if isinstance(val, float):
                    bar_length = int(max(0.0, min(1.0, val)) * 50)
                    bar = "█" * bar_length + "░" * (50 - bar_length)
                    print(f"    {model_name:14s}: {val:.4f} {bar}")
        
        print("\n Generated Files:")
        files = [
            ("outputs/clean_data.csv", "Cleaned dataset from Agent 1"),
            ("outputs/engineered_data.csv", "Engineered features from Agent 2"),
            ("outputs/final_model_code.py", "Best model code from Agent 3"),
            ("outputs/data_cleaner_report.json", "Agent 1 detailed report"),
            ("outputs/feature_engineer_report.json", "Agent 2 detailed report"),
            ("outputs/model_trainer_report.json", "Agent 3 detailed report"),
            ("outputs/final_report.md", "Complete execution report"),
            ("outputs/agent_execution.log", "Full execution log"),
        ]
        
        for filepath, description in files:
            exists = "✓" if Path(filepath).exists() else "✗"
            print(f"  {exists} {filepath:45s} - {description}")
        
        print("  1. Review the final_report.md for complete details")
        print("  2. Check agent_execution.log for agent reasoning")
        print("  3. Inspect final_model_code.py to see the trained model")
        print("  4. Use your own data: python main.py your_data.csv target_column")

        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()