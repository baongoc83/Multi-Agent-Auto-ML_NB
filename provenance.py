"""Run provenance: everything needed to say *exactly* what produced a model.

Written into replay_manifest.json["provenance"] and checked by replay_driver:
  * code     : git SHA/branch/dirty + `code_fingerprint` (SHA-256 over every .py in
               the repo, so uncommitted and untracked files are covered too)
  * config   : every UPPERCASE scalar of Config at run time (secrets excluded) plus
               the environment variables that override it
  * input    : path, size, mtime and SHA-256 of the source file
  * runtime  : python, platform, CPU count and library versions

Stdlib only, so the replay driver can import it from any environment.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

REPO = Path(__file__).resolve().parent
_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD")
_SKIP_DIRS = {"outputs", ".git", "__pycache__", ".venv", "venv", "node_modules", ".vscode", ".claude"}
_LIBS = ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "xgboost", "lightgbm",
         "catboost", "flaml", "optuna", "shap", "joblib", "openai", "anthropic")
_ENV_PREFIXES = ("FLAML_", "OPTUNA_", "RFE_", "PSI_", "STABILITY_", "SHAP_", "MULTI_SEED_",
                 "CALIBRATION_", "CLASS_WEIGHT_", "OOT_", "VALID_", "TEMPORAL_", "RANDOM_STATE",
                 "CV_N_SPLITS", "PD_", "NULL_", "FE_", "MAX_FINAL_", "TOP_K_", "WOE_", "IV_",
                 "OVERFIT_", "EARLY_STOPPING", "FEATURE_", "PRUNE_", "RETUNE_", "TREE_ENSEMBLE_", "KEEP_INTERMEDIATES", "LLM_BACKEND", "TRAIN_TEST_")


def sha256_file(path: Any, max_bytes: Optional[int] = None) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    if max_bytes is not None and p.stat().st_size > max_bytes:
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def code_fingerprint(repo: Path = REPO) -> str:
    """SHA-256 over (relative path, content hash) of every .py file, sorted."""
    h = hashlib.sha256()
    for p in sorted(repo.rglob("*.py")):
        rel = p.relative_to(repo)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        h.update(str(rel).replace("\\", "/").encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _git(*args: str) -> Optional[str]:
    try:
        r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _versions() -> Dict[str, str]:
    from importlib import metadata
    out = {}
    for lib in _LIBS:
        try:
            out[lib] = metadata.version(lib)
        except Exception:
            out[lib] = "not installed"
    return out


def _config_snapshot(config_cls: Any) -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    for k in dir(config_cls):
        if not k.isupper() or any(h in k for h in _SECRET_HINTS):
            continue
        v = getattr(config_cls, k)
        if isinstance(v, (str, int, float, bool, type(None))):
            snap[k] = v
    return snap


def collect(input_path: Any = None, config_cls: Any = None) -> Dict[str, Any]:
    status = _git("status", "--porcelain")
    dirty_files = [l for l in (status or "").splitlines() if l.strip()]
    inp: Dict[str, Any] = {"path": str(input_path) if input_path else None}
    if input_path and Path(str(input_path)).is_file():
        st = Path(str(input_path)).stat()
        inp.update(size_bytes=st.st_size,
                   mtime=datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                   sha256=sha256_file(input_path, max_bytes=30 * 1024 ** 3))
    return {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "code": {
            "git_sha": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(dirty_files),
            "git_changed_paths": len(dirty_files),
            "code_fingerprint": code_fingerprint(),
        },
        "config": _config_snapshot(config_cls) if config_cls is not None else {},
        "env_overrides": {k: v for k, v in sorted(os.environ.items())
                          if k.startswith(_ENV_PREFIXES) and not any(h in k for h in _SECRET_HINTS)},
        "input": inp,
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(),
                    "machine": platform.machine(), "cpu_count": os.cpu_count(),
                    "libraries": _versions()},
    }
