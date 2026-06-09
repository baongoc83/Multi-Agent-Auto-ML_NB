import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from config import Config


class AgentLogger:

    def __init__(self, log_file: str = None):
        # Force UTF-8 on Windows terminals that default to cp1252
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        self.log_file = Path(log_file or Config.EXECUTION_LOG_PATH)
        self.log_file.parent.mkdir(exist_ok=True)
        self.logs: List[Dict] = []
        # Cumulative LLM token usage across all agents in this pipeline run.
        # Populated by BaseAgent._log_token_usage; surfaced in the final report
        # so users can attribute cost between agents and across runs.
        self.token_usage: Dict[str, Dict[str, int]] = {}   # {model: {input, output, total, calls}}

    def record_tokens(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Accumulate token usage for cost attribution. Idempotent across calls."""
        bucket = self.token_usage.setdefault(
            model, {"input": 0, "output": 0, "total": 0, "calls": 0}
        )
        bucket["input"]  += int(input_tokens or 0)
        bucket["output"] += int(output_tokens or 0)
        bucket["total"]  += int((input_tokens or 0) + (output_tokens or 0))
        bucket["calls"]  += 1

    def token_summary(self) -> Dict[str, int]:
        """Aggregate across all models. Returns {input, output, total, calls}."""
        agg = {"input": 0, "output": 0, "total": 0, "calls": 0}
        for bucket in self.token_usage.values():
            for k in agg:
                agg[k] += bucket.get(k, 0)
        return agg

    def log(self, agent_name: str, action: str, details: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "timestamp": timestamp,
            "agent": agent_name,
            "action": action,
            "details": details,
        }
        self.logs.append(entry)
        print(f"\n[{timestamp}] {agent_name} - {action}")
        print(f"  {details}")

    def save(self):
        with open(self.log_file, "w", encoding="utf-8") as f:
            for entry in self.logs:
                f.write(f"[{entry['timestamp']}] {entry['agent']} - {entry['action']}\n")
                f.write(f"  {entry['details']}\n\n")

    def get_markdown_report(self) -> str:
        report = "# Multi-Agent AutoML Execution Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        for entry in self.logs:
            report += f"## [{entry['timestamp']}] {entry['agent']}\n"
            report += f"**Action:** {entry['action']}\n\n"
            report += f"{entry['details']}\n\n"
            report += "---\n\n"
        return report
