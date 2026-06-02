import sys
import argparse
from pathlib import Path
from pipeline import AutoMLPipeline
from config import Config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Multi-Agent AutoML Pipeline: DataCleaner -> FeatureEngineer -> TrainModel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Minimal - auto-detect everything (CSV, Parquet, Excel, etc.)
  python main.py data/train.csv TARGET
  python main.py data/train.parquet TARGET
  python main.py data/train.xlsx TARGET

  # Credit risk dataset (DPD30/90 default flag, MOB6/MOB12 window)
  python main.py data/train.csv TARGET \\
      --domain credit_risk --model-type binary_classification \\
      --entity-id SK_ID_CURR --keys SK_ID_CURR,MONTH_DT

  # Propensity model (buy / churn / interest)
  python main.py data/train.csv TARGET \\
      --domain propensity --model-type binary_classification

  # Fraud detection
  python main.py data/train.csv TARGET \\
      --domain fraud --model-type binary_classification

  # With column descriptions (HomeCredit format)
  python main.py data/train.csv TARGET \\
      --domain credit_risk \\
      --col-desc data/HomeCredit_columns_description.csv \\
      --col-name-field Row --col-desc-field Description --col-group-field Table

  # With pre-split OOT dataset (legacy — OOT bypasses Agents 1+2)
  python main.py data/train.parquet TARGET --oot data/oot.parquet

  # Pre-split mode — train + valid + oot all preserved exactly
  python main.py data/train.parquet TARGET \\
      --valid data/valid.parquet --oot data/oot.parquet \\
      --domain credit_risk --entity-id customer_id --keys customer_id,snap_dt

  # Full options
  python main.py data/train.csv TARGET \\
      --domain credit_risk --model-type binary_classification \\
      --entity-id customer_id --keys customer_id,snapshot_date \\
      --col-desc data/columns.csv \\
      --col-name-field Row --col-desc-field Description --col-group-field Table \\
      --oot data/oot.csv
        """,
    )
    parser.add_argument(
        "input_path", nargs="?", default=None,
        help=(
            f"Path to input dataset (default: {Config.OUTPUT_DIR}/sample_data.csv). "
            "Supported formats: .csv, .tsv, .parquet, .orc, .feather, "
            ".xlsx/.xls/.xlsm, .json, and remote paths (s3://, gs://, az://)."
        ),
    )
    parser.add_argument(
        "target_column", nargs="?", default=None,
        help="Name of the binary target / label column (default: target)",
    )
    parser.add_argument(
        "--entity-id", metavar="COL", dest="entity_id_col", default=None,
        help="Primary entity identifier column, e.g. customer_id, SK_ID_CURR",
    )
    parser.add_argument(
        "--keys", metavar="COL1,COL2", dest="composite_key_cols", default=None,
        help="Comma-separated composite key columns. Agent 1 uses these for PK checks; "
             "Agent 3 extracts the date partner for OOT temporal split.",
    )
    parser.add_argument(
        "--valid", metavar="PATH", dest="valid_path", default=None,
        help="Pre-split VALID dataset. When provided, switches to PRE-SPLIT MODE: "
             "input_path (train) + this file + --oot are concatenated with a "
             "`_split_` marker so Agents 1+2 apply identical cleaning + feature "
             "engineering to every partition; Agent 3 then reconstructs the exact "
             "user-defined train/valid/oot via the marker.",
    )
    parser.add_argument(
        "--oot", metavar="PATH", dest="oot_path", default=None,
        help="Pre-split OOT dataset (.csv / .parquet / .feather / .xlsx). "
             "Without --valid: legacy behaviour — pool input_path 80/20 + raw OOT (bypasses Agents 1+2). "
             "With --valid: OOT joins the concat + marker flow.",
    )
    parser.add_argument(
        "--col-desc", metavar="PATH", dest="col_descriptions_path", default=None,
        help="Column descriptions file (.csv / .json / .parquet / .xlsx). "
             "Passed to Agent 2 to guide interaction feature creation.",
    )
    parser.add_argument(
        "--col-name-field", metavar="FIELD", dest="col_name_field", default="column_name",
        help="Field name for column names in col-desc file (default: column_name)",
    )
    parser.add_argument(
        "--col-desc-field", metavar="FIELD", dest="col_desc_field", default="description",
        help="Field name for descriptions in col-desc file (default: description)",
    )
    parser.add_argument(
        "--col-group-field", metavar="FIELD", dest="col_group_field", default=None,
        help="Field name for the source table / group in col-desc file (default: None)",
    )
    parser.add_argument(
        "--domain", metavar="DOMAIN", dest="domain", default="generic",
        choices=["credit_risk", "propensity", "fraud", "generic"],
        help="Feature engineering domain context for Agent 2 (default: generic). "
             "Options: credit_risk | propensity | fraud | generic",
    )
    parser.add_argument(
        "--model-type", metavar="TYPE", dest="model_type", default="binary_classification",
        choices=["binary_classification", "regression", "multiclass"],
        help="ML problem type passed to Agent 2 prompt (default: binary_classification). "
             "Options: binary_classification | regression | multiclass",
    )
    return parser.parse_args()


def _print_metrics(final_metrics: dict) -> None:
    print(f"\nBest Model : {final_metrics.get('best_model', 'unknown')}")

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

    best_params = final_metrics.get("best_params", {})
    if best_params:
        print("\nTop Hyperparameters:")
        for k, v in list(best_params.items())[:6]:
            print(f"  {k}: {v}")


def _print_files() -> None:
    print("\nGenerated Files:")
    files = [
        (Config.CLEAN_DATA_PATH,                               "Cleaned dataset           — Agent 1"),
        (Config.ENGINEERED_DATA_PATH,                          "Engineered features       — Agent 2"),
        (Config.DATA_CLEANER_REPORT_PATH,                      "Data cleaner report       — Agent 1"),
        (Config.FEATURE_ENGINEER_REPORT_PATH,                  "Feature engineer report   — Agent 2"),
        (Config.MODEL_TRAINER_REPORT_PATH,                     "Model trainer report      — Agent 3"),
        (f"{Config.OUTPUT_DIR}/psi_report.csv",                "PSI drift report          — Agent 3"),
        (f"{Config.OUTPUT_DIR}/stability_report.csv",          "Feature stability report  — Agent 3"),
        (f"{Config.OUTPUT_DIR}/shap_psi_prune_log.csv",        "SHAP+PSI pruning log      — Agent 3"),
        (Config.FINAL_MODEL_PATH,                              "Trained model artifact    — Agent 3"),
        (Config.FINAL_MODEL_CODE_PATH,                         "Standalone inference code — Agent 3"),
        (Config.FINAL_REPORT_PATH,                             "Full pipeline report"),
        (Config.EXECUTION_LOG_PATH,                            "Agent execution log"),
    ]
    for filepath, description in files:
        mark = "✓" if Path(filepath).exists() else "✗"
        print(f"  {mark} {filepath:<47} {description}")


def main() -> None:
    print("=" * 65)
    print("  Multi-Agent AutoML Pipeline")
    print("  DataCleaner -> FeatureEngineer -> TrainModel")
    print("=" * 65)

    try:
        Config.validate()
        print("Configuration validated\n")
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    args = _parse_args()

    input_path = args.input_path or f"{Config.OUTPUT_DIR}/sample_data.csv"
    target_column = args.target_column or "target"
    composite_key_cols = (
        [c.strip() for c in args.composite_key_cols.split(",")]
        if args.composite_key_cols else None
    )
    col_desc_kwargs = (
        {
            "col_name_field": args.col_name_field,
            "col_desc_field": args.col_desc_field,
            "col_group_field": args.col_group_field,
        }
        if args.col_descriptions_path else None
    )

    print(f"Input        : {input_path}")
    print(f"Target       : {target_column}")
    if args.entity_id_col:
        print(f"Entity ID    : {args.entity_id_col}")
    if composite_key_cols:
        print(f"Composite key: {composite_key_cols}")
    if args.valid_path:
        print(f"Valid path   : {args.valid_path}  (pre-split mode)")
    if args.oot_path:
        print(f"OOT path     : {args.oot_path}")
    if args.col_descriptions_path:
        print(f"Col desc     : {args.col_descriptions_path}")
        if col_desc_kwargs:
            print(f"  name_field={args.col_name_field} | "
                  f"desc_field={args.col_desc_field} | "
                  f"group_field={args.col_group_field}")
    print(f"Domain       : {args.domain}")
    print(f"Model type   : {args.model_type}")
    print()

    pipeline = AutoMLPipeline()

    try:
        print("Starting the Multi-Agent Pipeline...\n")
        print("  Agent 1 — Data Cleaner   : audit quality, clean data")
        print("  Agent 2 — Feature Engineer: interaction features, encoding, selection")
        print("  Agent 3 — Train Model    : FLAML -> Optuna -> RFE -> PSI -> Stability -> SHAP+PSI -> Final")
        print()

        final_metrics = pipeline.run(
            input_path=input_path,
            target_column=target_column,
            col_descriptions_path=args.col_descriptions_path,
            col_descriptions_kwargs=col_desc_kwargs,
            entity_id_col=args.entity_id_col,
            composite_key_cols=composite_key_cols,
            valid_path=args.valid_path,
            oot_path=args.oot_path,
            domain=args.domain,
            model_type=args.model_type,
        )

        print("\n" + "=" * 65)
        print("  PIPELINE COMPLETE")
        print("=" * 65)

        _print_metrics(final_metrics)
        _print_files()

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
