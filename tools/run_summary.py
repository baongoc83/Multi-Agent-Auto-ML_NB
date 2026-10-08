#!/usr/bin/env python
"""Write run_summary.md for a finished run dir (numbers only, no LLM prose).

    py -3.12 tools/run_summary.py outputs/2026-10-07/run_01 [--title "..."] [--smoke-input file.csv]

Honest-metrics rule: valid_* are IN-SAMPLE (valid is merged into the refit), so the
headline is OOT when it exists, else the random test holdout, plus CV (OOF).
--smoke-input scores the first N rows through the bundle's replay driver (strict mode,
hash + provenance checks) as an end-to-end deployability check.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, float) else ("-" if v is None else str(v))


def _elapsed(log_text):
    ts = re.findall(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]", log_text, flags=re.M)
    if len(ts) < 2:
        return "n/a"
    from datetime import datetime
    f = "%Y-%m-%d %H:%M:%S"
    s = (datetime.strptime(ts[-1], f) - datetime.strptime(ts[0], f)).total_seconds()
    return f"{int(s // 3600)}h {int(s % 3600 // 60)}m"


def smoke(run: Path, manifest: dict, src: str, n: int):
    import numpy as np
    import pandas as pd
    out = ["", "## Bundle smoke test (replay driver, strict mode)", ""]
    try:
        head = pd.read_csv(src, nrows=n) if src.endswith(".csv") else pd.read_parquet(src).head(n)
        tmp = Path(tempfile.mkdtemp())
        inp, outp = tmp / "in.parquet", tmp / "scores.parquet"
        head.to_parquet(inp, index=False)
        t = time.time()
        r = subprocess.run([sys.executable, str(run / "replay_pipeline.py"), str(inp),
                            "--mode", "score", "--out", str(outp),
                            "--repo", str(Path(__file__).resolve().parents[1])],
                           capture_output=True, text=True)
        dt = time.time() - t
        if r.returncode != 0:
            out.append(f"- **FAILED** exit={r.returncode}: `{(r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ['?'])[-1][:200]}`")
            return out
        sc = pd.read_parquet(outp)
        out.append(f"- scored {len(sc)} rows in {dt:.1f}s | integrity/provenance lines: "
                   + "; ".join(l.strip() for l in r.stdout.splitlines() if "integrity" in l or "provenance" in l or "WARNING" in l))
        s = sc["score"]
        out.append(f"- PD min/mean/max = {s.min():.5f} / {s.mean():.5f} / {s.max():.5f} | "
                   f"distinct values {s.nunique()} | rows at PD floor {(s <= s.min() + 1e-12).sum()}")
        tgt = manifest.get("target_column")
        keys = [k for k in manifest["split"]["key_cols"] if k in sc.columns and k in head.columns]
        if tgt in head.columns and keys:
            from sklearn.metrics import roc_auc_score
            m = head[keys + [tgt]].merge(sc, on=keys, how="inner")
            if m[tgt].nunique() == 2:
                out.append(f"- AUC on these rows = {roc_auc_score(m[tgt], m['score']):.4f} "
                           "(mixture of train/valid/holdout rows: a wiring check, NOT a performance claim)")
    except Exception as e:
        out.append(f"- smoke test could not run: {type(e).__name__}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--title", default=None)
    ap.add_argument("--smoke-input", default=None)
    ap.add_argument("--smoke-rows", type=int, default=20000)
    a = ap.parse_args()
    run = Path(a.run_dir)
    rep = _load(run / "model_trainer_report.json") or {}
    man = _load(run / "replay_manifest.json") or {}
    fe = _load(run / "feature_engineer_report.json") or {}
    dc = _load(run / "data_cleaner_report.json") or {}
    log = (run / "agent_execution.log").read_text(encoding="utf-8", errors="replace") \
        if (run / "agent_execution.log").exists() else ""
    m = rep.get("metrics", {})
    L = [f"# {a.title or run.name} — run summary", "", f"`{run}`", ""]

    if not rep:
        L += ["**The run did not produce model_trainer_report.json — it failed or was interrupted.**", "",
              "Last log entries:", "```"] + log.strip().splitlines()[-12:] + ["```"]
        (run / "run_summary.md").write_text("\n".join(L), encoding="utf-8")
        print("\n".join(L[:6]))
        return

    # ---- headline ------------------------------------------------------------
    L += ["## Headline (honest holdout numbers)", ""]
    rows = [("OOT (time-based)", m.get("oot_auc"), m.get("oot_brier")),
            ("Test (random holdout)", m.get("test_auc"), m.get("test_brier")),
            ("CV OOF on train+valid", m.get("cv_auc_mean"), None)]
    L += ["| Split | AUC | Gini | Brier |", "|---|---|---|---|"]
    for name, auc, br in rows:
        if auc is not None:
            L.append(f"| {name} | {_fmt(auc)} | {_fmt(2 * auc - 1)} | {_fmt(br)} |")
    if m.get("cv_auc_std") is not None:
        L.append(f"\nCV std {m['cv_auc_std']}. In-sample (do NOT quote): "
                 f"valid_auc={_fmt(m.get('valid_auc'))}, valid_temporal_auc={_fmt(m.get('valid_temporal_auc'))}, "
                 f"valid_random_auc={_fmt(m.get('valid_random_auc'))}.")
    if m.get("oot_auc") is not None and m.get("cv_auc_mean"):
        L.append(f"\nOOT vs CV gap: {m['cv_auc_mean'] - m['oot_auc']:+.4f} AUC "
                 f"({(m['cv_auc_mean'] - m['oot_auc']) / m['cv_auc_mean']:.1%} relative).")
    if m.get("oot_auc") is None:
        L.append("\n> **No OOT in this run** (no date key) — temporal stability is unmeasured; PSI/Stability were skipped.")

    # ---- model ----------------------------------------------------------------
    fp = rep.get("feature_pipeline", {})
    L += ["", "## Model", "",
          f"- estimator **{rep.get('best_estimator')}**, best_iteration {m.get('best_iteration')}, seeds {m.get('n_seeds')}, "
          f"final fit rows {m.get('final_train_rows')} ({m.get('final_trained_on')})",
          f"- calibration {m.get('calibration_method')} ({m.get('calibration_split')}), "
          f"PD floor/cap {man.get('provenance', {}).get('config', {}).get('PD_FLOOR')}/"
          f"{man.get('provenance', {}).get('config', {}).get('PD_CAP')}",
          f"- feature funnel: {fp.get('n_init')} → RFE {fp.get('n_after_rfe')} → PSI {fp.get('n_after_psi')} → "
          f"Stability {fp.get('n_after_stability')} → SHAP+PSI {fp.get('n_after_shap_psi')}",
          f"- splits: {rep.get('splits')}",
          f"- temporal: {rep.get('temporal', {}).get('cadence')} "
          f"{ {k: v for k, v in (rep.get('temporal') or {}).items() if k in ('first_period','last_period','oot_periods','valid_temporal_periods','train_periods')} }"]
    top = [r.get("feature") for r in rep.get("shap_top_features", [])[:8]]
    if top:
        L.append(f"- top SHAP features: {', '.join(top)}")

    # ---- data handling ---------------------------------------------------------
    nul = man.get("null_processing") or {}
    ivs = fe.get("iv_stability") or {}
    L += ["", "## Data handling", "",
          f"- cleaning: shape {dc.get('original_shape')} → {dc.get('final_shape')}, "
          f"{len((dc.get('cleaning_spec') or {}).get('drops', []))} columns dropped, row_ops {(dc.get('cleaning_spec') or {}).get('row_ops')}",
          f"- feature engineering: {len((fe.get('feature_spec') or {}).get('interactions', []))} interactions, "
          f"{len((fe.get('feature_spec') or {}).get('woe_maps', {}))} WoE maps, final cols {fe.get('final_shape')}",
          f"- null processor: {nul.get('n_rules')} rules, {nul.get('n_indicators')} indicators, llm={nul.get('llm')}"
          if nul else "- null processor: not used",
          ]
    for part in ("valid", "oot", "test"):
        d = ivs.get(part)
        if d:
            L.append(f"- IV stability train↔{part}: {d.get('n_unstable')}/{d.get('n_evaluated')} features flagged unstable (diagnostic)")

    # ---- events ----------------------------------------------------------------
    pat = re.compile(r"^\[[^\]]+\] (.*(?:OVERFIT|Overfit|Time Budget|Fallback|SKIP|BLOCK|Null processor|"
                     r"drift|Distribution WARN|Split \(OOT\)|Split \(simple\)|Temporal cadence|PSI|Stability|"
                     r"Provenance|Replay bundle ERROR|Re-tune|Optuna|ERROR).*)$\n\s+(.*)$", re.M)
    ev = {}
    for head, detail in pat.findall(log):
        key = head.split(" - ")[-1] if " - " in head else head
        ev.setdefault(key, []).append(detail.strip()[:180])
    L += ["", "## Key events from the log", ""]
    for k, v in list(ev.items())[:25]:
        L.append(f"- **{k}** ×{len(v)} — {v[0]}")
    llm = re.findall(r"fallback=(\w+)", log)
    L += ["", f"- LLM calls answered via fallback: {len(llm)} ({', '.join(sorted(set(llm))) or 'none'})",
          f"- wall time: {_elapsed(log)}"]
    tok = re.findall(r"\| \*\*Total\*\* \| \*\*(\d+)\*\* \| \*\*([\d,]+)\*\* \| \*\*([\d,]+)\*\*", "")  # kept simple

    # ---- provenance ------------------------------------------------------------
    pv = man.get("provenance") or {}
    L += ["", "## Provenance", "",
          f"- git {str((pv.get('code') or {}).get('git_sha'))[:10]} "
          f"({'dirty, ' + str((pv.get('code') or {}).get('git_changed_paths')) + ' paths' if (pv.get('code') or {}).get('git_dirty') else 'clean'}), "
          f"code_fingerprint {str((pv.get('code') or {}).get('code_fingerprint'))[:12]}",
          f"- input {(pv.get('input') or {}).get('path')} — {(pv.get('input') or {}).get('size_bytes')} bytes, "
          f"sha256 {str((pv.get('input') or {}).get('sha256'))[:12]}",
          f"- env overrides: {pv.get('env_overrides') or 'none'}",
          f"- artifacts hashed: {list((man.get('artifacts') or {}).keys())}"]

    if a.smoke_input:
        L += smoke(run, man, a.smoke_input, a.smoke_rows)

    text = "\n".join(L) + "\n"
    (run / "run_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
