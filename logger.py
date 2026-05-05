from datetime import datetime
from pathlib import Path
from typing import List, Dict
from config import Config


class AgentLogger:

    def __init__(self, log_file: str = None):
        self.log_file = Path(log_file or Config.EXECUTION_LOG_PATH)
        self.log_file.parent.mkdir(exist_ok=True)
        self.logs: List[Dict] = []

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
        with open(self.log_file, "w") as f:
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
