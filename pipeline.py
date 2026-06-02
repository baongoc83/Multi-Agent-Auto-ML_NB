from logger import AgentLogger
from handoff import Handoff
from pathlib import Path
import pandas as pd
from config import Config
from Agents.BaseAgent.base_agent import BaseAgent
from Agents.DataCleaner.agent_data_cleaner import DataCleanerAgent
from Agents.FeatureEngineer.agent_feature_engineer import FeatureEngineerAgent
from Agents.TrainModel.agent_train_model import TrainModelAgent


# Marker column used to preserve user-specified splits through Agents 1+2.
# Agents protect this column from drop/encode/interaction; Agent 3 reads it
# to reconstruct train/valid/oot instead of auto-splitting.
SPLIT_MARKER = "_split_"


class AutoMLPipeline:
    """Orchestrates the three-agent pipeline."""

    def __init__(self):
        self.logger = AgentLogger()
        self.handoff = Handoff(self.logger)
        Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)

    def _build_combined_input(
        self,
        input_path: str,
        valid_path: str = None,
        oot_path: str = None,
    ) -> str:
        """Concat train + (optional) valid + (optional) oot with a `_split_`
        marker column so Agents 1+2 apply identical cleaning + feature
        engineering to every partition, while Agent 3 still reconstructs the
        exact user-defined splits.

        - train: always present (= input_path)
        - valid: included only if valid_path provided; otherwise Agent 3
                 auto-splits 20% of train into valid
        - oot:   included only if oot_path provided

        Returns the combined file path; written to OUTPUT_DIR/combined_input.csv.
        """
        train_df = BaseAgent.load_dataframe(input_path)
        train_df[SPLIT_MARKER] = "train"
        parts = [train_df]
        sizes = {"train": len(train_df)}

        if valid_path:
            valid_df = BaseAgent.load_dataframe(valid_path)
            valid_df[SPLIT_MARKER] = "valid"
            parts.append(valid_df)
            sizes["valid"] = len(valid_df)

        if oot_path:
            oot_in_df = BaseAgent.load_dataframe(oot_path)
            oot_in_df[SPLIT_MARKER] = "oot"
            parts.append(oot_in_df)
            sizes["oot"] = len(oot_in_df)

        combined = pd.concat(parts, ignore_index=True, sort=False)
        combined_path = f"{Config.OUTPUT_DIR}/combined_input.csv"
        combined.to_csv(combined_path, index=False)
        self.logger.log("PIPELINE", "Pre-split concat",
            f"Combined sizes={sizes} | total={len(combined)} | saved={combined_path}")
        return combined_path

    def run(
        self,
        input_path: str,
        target_column: str,
        col_descriptions_path: str = None,
        col_descriptions_kwargs: dict = None,
        entity_id_col: str = None,
        composite_key_cols: list = None,
        valid_path: str = None,
        oot_path: str = None,
        domain: str = "generic",
        model_type: str = "binary_classification",
    ):
        """Execute the full three-agent pipeline.

        Args:
            input_path: Path to the input dataset (always interpreted as
                        TRAIN when valid_path or oot_path is also provided).
                        Any format supported by BaseAgent.load_dataframe.
            valid_path: Optional path to a pre-split VALID dataset.
            oot_path:   Optional path to a pre-split OOT dataset.

            Pre-split mode is enabled whenever valid_path OR oot_path is set.
            In that mode, the three (or two) files are concatenated with a
            `_split_` marker column so Agents 1+2 apply identical cleaning +
            feature engineering to every partition. Agent 3 then reconstructs
            the splits from the marker:
              - train+valid+oot supplied → use all three exactly as marked.
              - train+valid only        → no OOT; use train/valid as marked.
              - train+oot only          → train auto-splits 20% into valid;
                                          OOT used as marked.
              - train only              → 60/20/20 auto-split (no marker).

            domain:     Feature engineering domain context. Options:
                        credit_risk, propensity, fraud, generic (default).
            model_type: ML problem type passed to Agent 2 prompt.
                        Options: binary_classification (default), regression, multiclass.
        """
        pre_split = (valid_path is not None) or (oot_path is not None)
        mode_label = "auto-split"
        if pre_split:
            parts_label = ["train"]
            if valid_path: parts_label.append("valid")
            if oot_path: parts_label.append("oot")
            mode_label = f"pre-split ({'+'.join(parts_label)})"
        self.logger.log("PIPELINE", "Starting",
            f"Input: {input_path} | Target: {target_column} | Domain: {domain} | "
            f"Model: {model_type} | Mode: {mode_label}")

        # Pre-split mode: concat train + (valid) + (oot) with marker → use as input.
        # This makes Agents 1+2 process the SAME schema for every partition, so
        # column drops / engineered features from Agent 2 apply consistently.
        if pre_split:
            input_path = self._build_combined_input(input_path, valid_path, oot_path)

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

        # OOT routing: when pre_split is on, OOT is already in the combined input
        # via the marker (went through Agents 1+2), so we don't pass it separately
        # to Agent 3 here. The auto-split branch (pre_split=False) means no OOT at all.
        oot_df = None

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
