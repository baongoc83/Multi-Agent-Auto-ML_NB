#!/usr/bin/env python
"""Security hardening: expression allowlist, artifact hashes, picklable calibrator.

Run:  py -3.12 tests/test_security.py
"""
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from preprocessing.safe_expr import UnsafeExpressionError, safe_eval, validate_expression
from Agents.FeatureEngineer.agent_feature_engineer import FeatureSpec

GOOD = [
    "df['a'] / (df['b'] + 1)",
    "-df['DAYS_BIRTH'] / 365.25",
    "df[['x','y','z']].mean(axis=1)",
    "df['x'].isna().astype(int)",
    "np.log1p(df['a'].clip(lower=0))",
    "df['f'] * df['a']",
    "np.where(df['a'] > 0, df['a'], 0)",
    "df['a'] ** 2",
    "(df['a'] - df['a'].median()) / (df['a'].std() + 1e-9)",
]
BAD = [
    "df.__class__.__init__.__globals__['__builtins__']['__import__']('os').getcwd()",
    "__import__('os').system('echo x')",
    "().__class__.__bases__[0].__subclasses__()",
    "open('/etc/passwd').read()",
    "eval('1+1')",
    "getattr(df, 'values')",
    "(lambda: 1)()",
    "[c for c in df]",
    "f'{df}'",
    "df.a + 1",                       # attribute column access: must use df['a']
    "df['a'].__class__",
    "np.load('x.npy')",
    "np.ctypeslib",
    "df['a'] ** 99999",
    "'a' * 1000000000",
    "df['a'].to_csv('x.csv')",
    "df['a'].eval('b')",
    "1; import os",
]


def test_allowlist_accepts_real_features_rejects_escapes():
    for e in GOOD:
        validate_expression(e)
    for e in BAD:
        try:
            validate_expression(e)
        except UnsafeExpressionError:
            continue
        raise AssertionError(f"accepted unsafe expression: {e}")


def test_safe_eval_values_match_plain_python():
    df = pd.DataFrame({"a": [1.0, 4.0], "b": [1.0, 3.0]})
    assert safe_eval("df['a'] / (df['b'] + 1)", df).tolist() == [0.5, 1.0]
    assert safe_eval("np.log1p(df['a'].clip(lower=0))", df).round(4).tolist() == \
        np.log1p([1.0, 4.0]).round(4).tolist()


def test_every_expression_of_run_02_still_validates():
    p = ROOT / "outputs" / "2026-10-06" / "run_02" / "feature_spec.pkl"
    if not p.exists():
        print("   (run_02 bundle not present, skipped)")
        return
    spec = joblib.load(p)
    assert spec.interactions
    for _, expr, _ in spec.interactions:
        validate_expression(expr)


def test_feature_spec_refuses_tampered_expression():
    spec = FeatureSpec(interactions=[("x", "__import__('os').getcwd()", 0.0)])
    try:
        spec.apply(pd.DataFrame({"a": [1]}))
        raise AssertionError("tampered spec must raise, not fall back to median")
    except UnsafeExpressionError:
        pass


def test_sigmoid_calibrator_is_picklable():
    from Agents.TrainModel.agent_train_model import TrainModelAgent
    rs = np.random.RandomState(0)
    raw = rs.rand(500)
    y = (rs.rand(500) < raw).astype(int)
    cal = TrainModelAgent._fit_calibrator(raw, y, "sigmoid")
    buf = io.BytesIO()
    joblib.dump({"calibrator": cal}, buf)
    buf.seek(0)
    back = joblib.load(buf)["calibrator"]
    np.testing.assert_allclose(back.predict(raw), cal.predict(raw))


def _driver():
    spec = importlib.util.spec_from_file_location("drv", ROOT / "replay_driver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_replay_refuses_modified_artifact():
    d = Path(tempfile.mkdtemp())
    f = d / "feature_spec.pkl"
    f.write_bytes(b"original-bytes")
    manifest = {"artifacts": {"feature_spec.pkl": hashlib.sha256(b"original-bytes").hexdigest()}}
    drv = _driver()
    drv._HERE = d
    drv._verify_artifacts(manifest)                              # intact: ok
    f.write_bytes(b"swapped-bytes")
    try:
        drv._verify_artifacts(manifest)
        raise AssertionError("modified artifact must be refused")
    except SystemExit as e:
        assert "hash mismatch" in str(e)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(tests)} SECURITY CHECKS PASSED")


if __name__ == "__main__":
    main()
