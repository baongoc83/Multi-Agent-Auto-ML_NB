from logger import AgentLogger
from handoff import Handoff
from pathlib import Path
from config import Config
from Agents.BaseAgent.base_agent import BaseAgent
from Agents.DataCleaner.agent_data_cleaner import DataCleanerAgent
from Agents.FeatureEngineer.agent_feature_engineer import FeatureEngineerAgent
from Agents.TrainModel.agent_train_model import TrainModelAgent


class AutoMLPipeline:
    """Orchestrates the three-agent pipeline."""

    def __init__(self):
        self.logger = AgentLogger()
        self.handoff = Handoff(self.logger)
        Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)

    def run(
        self,
        input_path: str,
        target_column: str,
        col_descriptions_path: str = None,
        col_descriptions_kwargs: dict = None,
        entity_id_col: str = None,
        composite_key_cols: list = None,
        oot_path: str = None,
        domain: str = "generic",
        model_type: str = "binary_classification",
    ):
        """Execute the full three-agent pipeline.

        Args:
            input_path: Path to the input dataset. Any format supported by
                        BaseAgent.load_dataframe: .csv, .tsv, .parquet, .orc,
                        .feather, .xlsx, .xls, .xlsm, .json, and remote paths
                        (s3://, gs://, az://).
            oot_path: Optional path to a pre-split OOT dataset. Supports the same
                      formats as input_path. When provided, Agent 3 skips temporal
                      extraction and only splits the pool 80/20.
            domain: Feature engineering domain context. Options: credit_risk,
                    propensity, fraud, generic (default).
            model_type: ML problem type passed to Agent 2 prompt context.
                        Options: binary_classification (default), regression, multiclass.
        """
        self.logger.log("PIPELINE", "Starting",
            f"Input: {input_path} | Target: {target_column} | Domain: {domain} | Model: {model_type}")

        self.logger.log("PIPELINE", "Stage 1", "Initializing Data Cleaner Agent")
        agent1 = DataCleanerAgent(
            self.logger,
            entity_id_col=entity_id_col,
            composite_key_cols=composite_key_cols,
            target_column=target_column,
        )
        clean_data_path, report1 = agent1.process(input_path)
        self.handoff.set_data(clean_data_path, report1, "DataCleaner")

        self.logger.log("PIPELINE", "Stage 2", "Initializing Feature Engineer Agent")
        agent2 = FeatureEngineerAgent(
            self.logger,
            col_descriptions_path=col_descriptions_path,
            domain=domain,
            model_type=model_type,
            **(col_descriptions_kwargs or {}),
        )
        engineered_data_path, report2 = agent2.process(
            self.handoff.get_data(),
            self.handoff.get_report(),
            target_column,
        )
        self.handoff.set_data(engineered_data_path, report2, "FeatureEngineer")

        oot_df = BaseAgent.load_dataframe(oot_path) if oot_path else None
        if oot_df is not None:
            self.logger.log("PIPELINE", "Stage 3", f"OOT provided externally: {oot_path} ({len(oot_df)} rows)")

        self.logger.log("PIPELINE", "Stage 3", "Initializing Train Model Agent")
        agent3 = TrainModelAgent(self.logger)
        final_metrics, report3 = agent3.process(
            self.handoff.get_data(),
            self.handoff.get_report(),
            target_column,
            oot_df=oot_df,
        )

        self._generate_final_report(report1, report2, report3, final_metrics)

        self.logger.log("PIPELINE", "Complete", "All agents finished successfully")
        self.logger.save()

        return final_metrics

    def _generate_final_report(self, report1, report2, report3, metrics):
        markdown = self.logger.get_markdown_report()
        markdown += "\n# Final Summary\n\n"
        markdown += "## Agent 1: Data Cleaner\n"
        markdown += f"- Actions: {report1.get('summary', 'N/A')}\n\n"
        markdown += "## Agent 2: Feature Engineer\n"
        markdown += f"- Strategy: {report2.get('summary', 'N/A')}\n\n"
        markdown += "## Agent 3: Model Trainer\n"
        markdown += f"- Final Metrics: {metrics}\n\n"

        with open(Config.FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"Final Report saved to: {Config.FINAL_REPORT_PATH}")
