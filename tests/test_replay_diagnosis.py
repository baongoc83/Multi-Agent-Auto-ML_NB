#!/usr/bin/env python
"""A hard fail must name the cause, not just report a mismatch.

"final_train_rows differ" tells the user something broke but nothing about
what to do. Each stage of the replay fails in a recognisable way, so the
driver checks them in the order they run and reports the first one that is
off. This test breaks each stage deliberately and asserts the right cause is
named — and, just as importantly, that the wrong ones are not.

Run:  py -3.12 tests/test_replay_diagnosis.py
"""

import importlib.util
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
drv_spec = importlib.util.spec_from_file_location("drv", REPO / "replay_driver.py")
drv = importlib.util.module_from_spec(drv_spec)
drv_spec.loader.exec_module(drv)

KEY = ["cust_id"]


def _setup():
    """A tiny recorded run: 10 rows, 6 train / 4 oot."""
    scratch = Path(tempfile.mkdtemp(prefix="diag_"))
    drv._HERE = scratch

    raw = pd.DataFrame({"cust_id": range(10),
                        "x": np.arange(10.0),
                        "target": [0, 1] * 5})
    assign = pd.DataFrame({
        "cust_id": list(range(10)),
        "_key_occ_": [0] * 10,
        "_split_": ["train"] * 6 + ["oot"] * 4,
        "_split_pos_": list(range(6)) + list(range(4)),
    })
    assign.to_parquet(scratch / "split_assignment.parquet", index=False)

    manifest = {
        "created_at": "t", "run_dir": str(scratch), "target_column": "target",
        "split": {"assignment_file": "split_assignment.parquet",
                  "key_cols": KEY, "positional_fallback": False,
                  "sizes": {"train": 6, "oot": 4}},
        "cleaning": {"spec_file": "c.pkl",
                     "row_ops": [{"op": "drop_duplicates"}]},
        "feature_engineering": {"spec_file": "f.pkl"},
        "training": {"model_file": "m.pkl", "estimator": "lgbm",
                     "best_params": {}, "final_features": ["x"]},
    }
    (scratch / "replay_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return raw, manifest


def _run_diagnose(raw, manifest, splits, unmatched=0, labelled=None):
    if labelled is None:
        labelled = raw.copy()
        a = pd.read_parquet(drv._HERE / "split_assignment.parquet")
        labelled = labelled.merge(a, on=KEY, how="left")
    labelled.attrs["unmatched"] = unmatched
    buf = io.StringIO()
    with redirect_stdout(buf):
        drv._diagnose(raw, labelled, splits, manifest)
    return buf.getvalue()


def main() -> int:
    raw, manifest = _setup()
    ok = True

    def expect(label, out, must_have, must_not_have=()):
        nonlocal ok
        missing = [m for m in must_have if m.lower() not in out.lower()]
        wrong = [m for m in must_not_have if m.lower() in out.lower()]
        if missing or wrong:
            ok = False
            print(f"FAIL  {label}")
            if missing:
                print(f"        did not mention: {missing}")
            if wrong:
                print(f"        wrongly blamed : {wrong}")
            print("        ---- output ----")
            for line in out.strip().splitlines():
                print(f"        {line}")
        else:
            print(f"PASS  {label}")
            for line in out.strip().splitlines()[1:3]:
                print(f"         {line.strip()}")

    # 1. everything fine -> must fall through to "it is the fit itself"
    good = {"train": pd.DataFrame({"x": range(6)}), "oot": pd.DataFrame({"x": range(4)})}
    out = _run_diagnose(raw, manifest, good)
    expect("healthy replay -> points at library versions", out,
           ["library versions"], ["not unique", "row_ops", "absent"])

    # 2. rows lost during cleaning (join was right, row ops ate rows)
    short = {"train": pd.DataFrame({"x": range(4)}), "oot": pd.DataFrame({"x": range(4)})}
    out = _run_diagnose(raw, manifest, short)
    expect("rows lost in cleaning -> blames row_ops", out,
           ["row_ops", "train: 4 rows", "cleaning"], ["not unique"])

    # 3. duplicate key -> join matched the wrong number of rows
    dup_assign = pd.read_parquet(drv._HERE / "split_assignment.parquet")
    dup_lab = raw.copy().merge(dup_assign, on=KEY, how="left")
    dup_lab = pd.concat([dup_lab, dup_lab.iloc[:3]], ignore_index=True)  # fan-out
    out = _run_diagnose(raw, manifest, short, labelled=dup_lab)
    expect("join fan-out -> blames the key", out,
           ["not unique", "cust_id"])

    # 4. input rows missing from the recorded split
    out = _run_diagnose(raw, manifest, good, unmatched=3)
    expect("unmatched input rows -> blames the input file", out,
           ["3 input row", "not in the recorded split"])

    # 5. feature spec did not rebuild the recorded columns
    nofeat = {"train": pd.DataFrame({"other": range(6)}),
              "oot": pd.DataFrame({"other": range(4)})}
    out = _run_diagnose(raw, manifest, nofeat)
    expect("missing recorded feature -> blames the feature spec", out,
           ["absent after the feature spec", "'x'"])

    print("\n" + ("ALL DIAGNOSIS CHECKS PASSED" if ok else "DIAGNOSIS CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
