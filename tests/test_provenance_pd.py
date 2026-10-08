#!/usr/bin/env python
"""Provenance block + PD floor/cap.   Run: py -3.12 tests/test_provenance_pd.py"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import provenance
from config import Config


def test_provenance_content():
    f = Path(tempfile.mkdtemp()) / "in.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    p = provenance.collect(f, Config)
    assert p["code"]["code_fingerprint"] and len(p["code"]["code_fingerprint"]) == 64
    assert p["input"]["sha256"] == provenance.sha256_file(f) and p["input"]["size_bytes"] > 0
    assert p["config"]["RANDOM_STATE"] == Config.RANDOM_STATE
    assert not any(any(h in k for h in ("KEY", "TOKEN", "SECRET")) for k in p["config"])
    assert "xgboost" in p["runtime"]["libraries"]
    json.dumps(p)                                              # manifest-serialisable


def test_fingerprint_changes_with_code():
    a = provenance.code_fingerprint()
    probe = ROOT / "_fp_probe_tmp.py"
    probe.write_text("x = 1\n")
    try:
        assert provenance.code_fingerprint() != a              # untracked file is covered
    finally:
        probe.unlink()
    assert provenance.code_fingerprint() == a


def test_pd_floor_and_cap_applied():
    from Agents.TrainModel.agent_train_model import TrainModelAgent
    ag = TrainModelAgent.__new__(TrainModelAgent)
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    raw = np.linspace(0, 1, 200)
    y = (raw > 0.5).astype(int)                                # lowest bands: 0 defaults, top: all
    iso.fit(raw, y)
    ag.calibrator = iso
    out = ag._apply_calibrator(raw)
    assert iso.predict(raw).min() == 0.0                       # the problem
    assert out.min() >= Config.PD_FLOOR - 1e-12 and out.max() <= Config.PD_CAP + 1e-12
    assert out.min() > 0


if __name__ == "__main__":
    for t in (test_provenance_content, test_fingerprint_changes_with_code, test_pd_floor_and_cap_applied):
        t()
        print("PASS ", t.__name__)
    print("\nALL PROVENANCE / PD CHECKS PASSED")
