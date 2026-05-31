from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from logger import AgentLogger
from Agents.BaseAgent.base_agent import BaseAgent


class Handoff:
    """Manages data and information transfer between agents."""

    def __init__(self, logger: AgentLogger):
        self.logger = logger
        self.data_path: Optional[Path] = None
        self.report: Dict[str, Any] = {}
        self._cached_df: Optional[pd.DataFrame] = None
        self._from_agent: Optional[str] = None

    def set_data(self, data_path: str, report: Dict[str, Any], from_agent: str):
        self.data_path = Path(data_path)
        self.report = report
        self._from_agent = from_agent
        self._cached_df = None   # invalidate cache — new producer overrides previous data
        self.logger.log(
            "HANDOFF",
            f"Data Transfer from {from_agent}",
            f"Data: {data_path}, Report keys: {list(report.keys())}",
        )

    def get_data(self, use_cache: bool = True) -> pd.DataFrame:
        """Return the current DataFrame, loading from disk on first access.

        The result is cached so repeated calls within the same handoff do not
        re-read the file. Pass use_cache=False to force a fresh read.
        """
        if self.data_path is None:
            raise ValueError(
                "No data has been set for handoff — call set_data() first."
            )
        if use_cache and self._cached_df is not None:
            return self._cached_df
        df = BaseAgent.load_dataframe(str(self.data_path))
        self._cached_df = df
        return df

    def get_report(self) -> Dict[str, Any]:
        return self.report

    def clear(self) -> None:
        """Reset all state — useful when reusing the same instance across runs."""
        self.data_path = None
        self.report = {}
        self._cached_df = None
        self._from_agent = None
        self.logger.log("HANDOFF", "Cleared", "All state reset")
