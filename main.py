import sys
from pipeline import AutoMLPipeline
from config import Config


def main():
    print("Multi-Agent AutoML Team")
    print("Three AI Agents Working Together")

    try:
        Config.validate()
        print("Configuration validated\n")
    except ValueError as e:
        print(f"Configuration error: {e}")
        return

    if len(sys.argv) > 2:
        input_csv = sys.argv[1]
        target_column = sys.argv[2]
        col_descriptions_path = sys.argv[3] if len(sys.argv) > 3 else None
    else:
        input_csv = f"{Config.OUTPUT_DIR}/sample_data.csv"
        target_column = "target"
        col_descriptions_path = None
        print(f"Using sample data: {input_csv}")
        print(f"Target column: {target_column}\n")

    pipeline = AutoMLPipeline()

    try:
        print("Starting the Multi-Agent Pipeline...\n")
        final_metrics = pipeline.run(input_csv, target_column, col_descriptions_path=col_descriptions_path)

        print("\nPIPELINE COMPLETE!")
        print(f"\nFinal Model Metrics (best: {final_metrics.get('best_model', 'unknown')}):")
        for metric, value in final_metrics.items():
            if isinstance(value, float):
                print(f"  - {metric}: {value:.4f}")
        comparison = final_metrics.get("model_comparison", {})
        if comparison:
            primary = "roc_auc_score" if any("roc_auc_score" in v for v in comparison.values()) else "r2"
            print(f"  Model comparison ({primary}):")
            for model_name, model_metrics in comparison.items():
                val = model_metrics.get(primary)
                print(f"    {model_name}: {val:.4f}" if isinstance(val, float) else f"    {model_name}: N/A")

        print("\nGenerated Files:")
        print(f"  {Config.CLEAN_DATA_PATH} - Cleaned dataset")
        print(f"  {Config.ENGINEERED_DATA_PATH} - With new features")
        print(f"  {Config.FINAL_MODEL_CODE_PATH} - Best model code")
        print(f"  {Config.FINAL_REPORT_PATH} - Complete execution report")
        print(f"  {Config.EXECUTION_LOG_PATH} - Detailed logs")

    except Exception as e:
        print(f"\nPipeline error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nUsage:")
    print("  python main.py                                                    # use sample data")
    print("  python main.py data.csv target_column                            # use your own data")
    print("  python main.py data.csv target_column col_descriptions.json      # with column metadata")
    print()
    main()
