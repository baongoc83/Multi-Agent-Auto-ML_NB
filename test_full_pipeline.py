import pandas as pd
import numpy as np
from pathlib import Path
from pipeline import AutoMLPipeline
from config import Config

# def create_realistic_dataset():
#     np.random.seed(42)
#     n_samples = 200
    
#     age = np.random.randint(18, 70, n_samples)
#     experience = np.clip(age - 22 + np.random.randint(-3, 10, n_samples), 0, 50)
#     education = np.random.choice([1, 2, 3, 4], n_samples, p=[0.15, 0.55, 0.2, 0.1])  # 1=HS, 2=Bachelor, 3=Master, 4=PhD
    
#     # income based on education, experience, and age with some noise
#     base_income = 30000 + (education * 15000) + (experience * 1000) + (age * 500)
#     income = base_income + np.random.normal(0, 15000, n_samples)
#     income = np.clip(income, 25000, 300000)
    
#     # create target: 1 if high performer (income > median and experience > 10)
#     # target = ((income > np.median(income)) & (experience > 10)).astype(int)
#     latent = np.random.normal(0, 0.2, len(income))

#     score = (
#             0.52 * (income > np.median(income)).astype(float) +
#             0.32 * (experience > 10).astype(float) +
#             0.16 * latent
#     )
#     prob = 1 / (1 + np.exp(-3 * (score - 0.5)))
#     target = (np.random.rand(len(prob)) < prob).astype(int)
#     # add some categorical features
#     department = np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], n_samples)
#     location = np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples)
    
#     # add some missing values
#     age_mask = np.random.random(n_samples) < 0.15
#     age = age.astype(float)
#     age[age_mask] = np.nan
    
#     income_mask = np.random.random(n_samples) < 0.10
#     income[income_mask] = np.nan
    
#     # create useless columns
#     id_col = range(1, n_samples + 1)
#     random_col = np.random.random(n_samples)
#     mostly_missing = [np.nan] * int(n_samples * 0.9) + list(np.random.random(int(n_samples * 0.1)))
#     np.random.shuffle(mostly_missing)
    
#     # build dataframe
#     df = pd.DataFrame({
#         'employee_id': id_col,
#         'age': age,
#         'years_experience': experience,
#         'education_level': education,
#         'annual_income': income,
#         'department': department,
#         'location': location,
#         'random_noise': random_col,
#         'useless_feature': mostly_missing,
#         'high_performer': target
#     })
    
#     return df

def main():
    print("FULL PIPELINE TEST - Multi-Agent AutoML System")
    
    try:
        Config.validate()
        print("Test LLM AI Models configuration validated\n")
    except ValueError as e:
        print(f"Configuration error: {e}")
        return
    
    # print("Creating realistic test dataset...")
    # df = create_realistic_dataset()
    # df.to_csv(test_data_path, index=False)
    
    Path("outputs").mkdir(exist_ok=True)
    # test_data_path = "D:/Ngoc/AI Project/Test_1st_version/multi-agent-auto-ml-v2/outputs/sample_data.csv"
    test_data_path = r"D:\Ngoc\AI Project\Test_1st_version\multi-agent-auto-ml-v2\data\home-credit-default-risk\application_train_processed.csv"
    # test_data_path = "outputs/test_employee_data.csv"
    
    print(f"Test dataset sample: {test_data_path}")
    # print(f"  - Shape: {df.shape}")
    # print(f"  - Features: {list(df.columns)}")
    # print(f"  - Target: high_performer (binary classification)")
    # print(f"  - Missing values: {df.isnull().sum().sum()}")
    # print(f"  - Target distribution: {df['high_performer'].value_counts().to_dict()}")
    print()
    
    print("STARTING THE MULTI-AGENT PIPELINE")
    print("\nThe three agents will now work sequentially:")
    print("  1. Data Cleaner: Audit data quality and clean the data")
    print("  2. Feature Engineer: Create and select meaningful features")
    print("  3. Model Trainer: Train and compare multiple ML models select the best parameters and model")
    print()
    
    pipeline = AutoMLPipeline()
    
    try:
        # ── Mode 1: AUTO-SPLIT (mặc định) ──────────────────────────────────────
        # input_path là toàn bộ dataset; pipeline tự split:
        #   - Có date_col → train + valid_temporal + valid_random + oot (OOT temporal)
        #   - Không date_col → train (60%) + valid (20%) + test (20%)
        # final_metrics = pipeline.run(
        #     input_path=test_data_path,
        #     target_column="label",
        #     col_descriptions_path="D:/Ngoc/AI Project/Test_1st_version/multi-agent-auto-ml-v2/data/col_descriptions.json",
        #     # col_descriptions_kwargs=dict(
        #     #     col_name_field="Row",
        #     #     col_desc_field="Description",
        #     #     col_group_field="Table",
        #     # ),
        #     entity_id_col="customer_id",
        #     composite_key_cols=["customer_id","snap_dt"],
        #     domain="credit_risk",
        #     model_type="binary_classification",
        # )
        final_metrics = pipeline.run(
            input_path=test_data_path,
            target_column="TARGET",
            col_descriptions_path="data/HomeCredit_columns_description.csv",
            col_descriptions_kwargs=dict(
                col_name_field="Row",
                col_desc_field="Description",
                col_group_field="Table",
            ),
            entity_id_col="SK_ID_CURR",
            composite_key_cols=["SK_ID_CURR"],
            domain="credit_risk",
            model_type="binary_classification",
        )

        # ── Mode 2: PRE-SPLIT — train + valid + oot tách sẵn ───────────────────
        # Khi bạn đã có train/valid/oot riêng biệt và muốn giữ chính xác splits đó.
        # Pipeline concat 3 file với cột marker `_split_` → Agents 1+2 xử lý đồng
        # nhất → Agent 3 đọc marker dựng lại splits.
        #
        # final_metrics = pipeline.run(
        #     input_path="data/train.csv",         # train
        #     valid_path="data/valid.csv",         # valid riêng
        #     oot_path="data/oot.csv",             # oot riêng
        #     target_column="TARGET",
        #     entity_id_col="customer_id",
        #     composite_key_cols=["customer_id", "snap_dt"],
        #     domain="credit_risk",
        #     model_type="binary_classification",
        # )

        # ── Mode 3: PRE-SPLIT — chỉ train + valid (không oot) ──────────────────
        # final_metrics = pipeline.run(
        #     input_path="data/train.csv",
        #     valid_path="data/valid.csv",
        #     target_column="TARGET",
        #     domain="credit_risk",
        # )

        # ── Mode 4: PRE-SPLIT — chỉ train + oot (không valid) ──────────────────
        # Pipeline auto-split 20% train → valid; oot vẫn đi qua Agents 1+2.
        # final_metrics = pipeline.run(
        #     input_path="data/train.csv",
        #     oot_path="data/oot.csv",
        #     target_column="TARGET",
        #     domain="credit_risk",
        # )

        # ── Mode 5: PRE-SPLIT + memory savers (dataset rất lớn) ────────────────
        # Khi train 1M+ rows × 1000+ features có nguy cơ OOM:
        #   - prefilter=True (default): drop cột null>95% / constant / dominant>99%
        #   - train_sample_ratio=0.3: stratified sample 30% train (valid/oot intact)
        # final_metrics = pipeline.run(
        #     input_path="data/train.parquet",
        #     valid_path="data/valid.parquet",
        #     oot_path="data/oot.parquet",
        #     target_column="TARGET",
        #     entity_id_col="customer_id",
        #     composite_key_cols=["customer_id", "snap_dt"],
        #     train_sample_ratio=0.3,     # sample 30% train
        #     prefilter=True,              # auto-drop junk columns
        #     domain="credit_risk",
        # )
        
        print("PIPELINE COMPLETED")

        # ── Best model ────────────────────────────────────────────────────────
        print(f"\nBest Model : {final_metrics.get('best_model', 'unknown')}")

        # ── AUC metrics with progress bar ─────────────────────────────────────
        auc_keys = [
            ("cv_auc_mean",        "CV AUC (mean)   "),
            ("cv_auc_std",         "CV AUC (std)    "),
            ("valid_temporal_auc", "Valid temporal  "),
            ("valid_random_auc",   "Valid random    "),
            ("valid_auc",          "Valid           "),
            ("oot_auc",            "OOT             "),
            ("test_auc",           "Test (holdout)  "),
        ]
        print("\nModel Metrics:")
        for key, label in auc_keys:
            val = final_metrics.get(key)
            if isinstance(val, float):
                bar_len = int(max(0.0, min(1.0, val)) * 40)
                bar = "█" * bar_len + "░" * (40 - bar_len)
                print(f"  {label}: {val:.4f}  {bar}")

        # ── Top hyperparams ───────────────────────────────────────────────────
        best_params = final_metrics.get("best_params", {})
        if best_params:
            print("\nTop Hyperparameters:")
            for k, v in list(best_params.items())[:6]:
                print(f"  {k}: {v}")

        # ── Generated files ───────────────────────────────────────────────────
        print(f"\nRun directory: {Config.RUN_DIR}")
        print("Generated Files:")
        files = [
            (Config.DATA_CLEANER_REPORT_PATH,        "Data cleaner report         — Agent 1"),
            (Config.FEATURE_ENGINEER_REPORT_PATH,    "Feature engineer report     — Agent 2"),
            (Config.MODEL_TRAINER_REPORT_PATH,       "Model trainer report        — Agent 3"),
            (Config.PSI_REPORT_PATH,                 "PSI drift report            — Agent 3"),
            (Config.STABILITY_REPORT_PATH,           "Feature stability report    — Agent 3"),
            (Config.SHAP_PSI_PRUNE_LOG_PATH,         "SHAP+PSI pruning log        — Agent 3"),
            (Config.SHAP_PLOT_PATH,                  "SHAP final bar plot         — Agent 3"),
            (Config.SHAP_BEESWARM_PATH,              "SHAP final beeswarm plot    — Agent 3"),
            (Config.SHAP_FEATURE_REPORT_PATH,        "SHAP top features CSV       — Agent 3"),
            (Config.SHAP_FINAL_MODEL_REPORT_PATH,    "SHAP combined markdown rpt  — Agent 3"),
            (Config.FINAL_MODEL_PATH,                "Trained model artifact      — Agent 3"),
            (Config.FINAL_MODEL_CODE_PATH,           "Standalone inference code   — Agent 3"),
            (Config.FINAL_REPORT_PATH,               "Full pipeline report"),
            (Config.EXECUTION_LOG_PATH,              "Agent execution log"),
            (Config.PIPELINE_PROCESS_DC_PATH,        "Replay script               — Agent 1"),
            (Config.PIPELINE_PROCESS_FE_PATH,        "Replay script               — Agent 2"),
            (Config.PIPELINE_PROCESS_FE_SPEC_PATH,   "Fitted FeatureSpec sidecar  — Agent 2"),
            (Config.PIPELINE_PROCESS_TM_PATH,        "Replay script               — Agent 3"),
        ]
        for filepath, description in files:
            mark = "✓" if Path(filepath).exists() else "✗"
            print(f"  {mark} {filepath:<70} {description}")

        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()