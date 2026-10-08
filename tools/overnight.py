#!/usr/bin/env python
"""Run the full pipeline on several datasets back to back and summarise each run.

    py -3.12 tools/overnight.py

Everything is written to outputs/overnight_<timestamp>/ :
    <name>.stdout.log   full console output of each run (tracebacks included)
    <name>.summary.md   tools/run_summary.py output (also copied into the run dir)
    OVERNIGHT_SUMMARY.md  rolling index, updated after every run
Runs are sequential (they each use all cores) and a failure in one does not stop the next.
"""
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = Path(r"D:\Ngoc\AI Project\Test_1st_version\multi-agent-auto-ml-v2")

RUNS = [
    {"name": "snapdate_sample_data",
     "args": [str(V2 / "outputs" / "sample_data.csv"), "label",
              "--entity-id", "customer_id", "--keys", "customer_id,snap_dt",
              "--domain", "credit_risk", "--product-type", "consumer_unsecured",
              "--col-desc", str(ROOT / "data" / "col_descriptions.json")]},
    {"name": "home_credit_application_train",
     "args": [str(V2 / "data" / "home-credit-default-risk" / "application_train_processed.csv"), "TARGET",
              "--entity-id", "SK_ID_CURR", "--keys", "SK_ID_CURR",
              "--domain", "credit_risk", "--product-type", "consumer_unsecured",
              "--col-desc", str(ROOT / "data" / "HomeCredit_columns_description.csv"),
              "--col-name-field", "Row", "--col-desc-field", "Description", "--col-group-field", "Table"]},
]


def _keep_awake():
    """Block system sleep while this process lives (Windows); reverts on exit. No settings are changed."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # CONTINUOUS | SYSTEM_REQUIRED
    except Exception:
        pass


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="run names to execute (default: all)")
    ap.add_argument("--env", nargs="*", default=[], help="extra KEY=VALUE for the pipeline processes")
    args = ap.parse_args()
    runs = [r for r in RUNS if not args.only or r["name"] in args.only]
    _keep_awake()
    out = ROOT / "outputs" / f"overnight_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    index = out / "OVERNIGHT_SUMMARY.md"
    rows = []
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
           **dict(kv.split("=", 1) for kv in args.env)}
    for r in runs:
        t0 = time.time()
        log = out / f"{r['name']}.stdout.log"
        print(f"[{datetime.now():%H:%M:%S}] START {r['name']}", flush=True)
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.run([sys.executable, str(ROOT / "main.py"), *r["args"]],
                               cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        text = log.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"persisted → (\S+)", text)
        run_dir = (ROOT / m.group(1)) if m else None
        status = "OK" if p.returncode == 0 else f"FAILED (exit {p.returncode})"
        dur = f"{(time.time() - t0) / 3600:.2f} h"
        summ = out / f"{r['name']}.summary.md"
        if run_dir and run_dir.exists():
            smoke = ["--smoke-input", r["args"][0]] if p.returncode == 0 else []
            s = subprocess.run([sys.executable, str(ROOT / "tools" / "run_summary.py"), str(run_dir),
                                "--title", r["name"], *smoke],
                               cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
            summ.write_text(s.stdout if s.returncode == 0 else f"summary failed:\n{s.stderr}", encoding="utf-8")
        else:
            summ.write_text("run directory not found; see stdout log\n\n" + text[-3000:], encoding="utf-8")
        rows.append(f"| {r['name']} | {status} | {dur} | `{run_dir}` | [{summ.name}]({summ.name}) |")
        index.write_text("# Overnight runs\n\n| run | status | wall | run dir | summary |\n|---|---|---|---|---|\n"
                         + "\n".join(rows) + "\n", encoding="utf-8")
        print(f"[{datetime.now():%H:%M:%S}] DONE  {r['name']} -> {status} in {dur}", flush=True)
    (out / "FINISHED").write_text(datetime.now().isoformat())


if __name__ == "__main__":
    main()
