#!/usr/bin/env python
"""The same table must behave identically whichever reader produced it.

CSV, parquet, Excel and database drivers each hand pandas a different set of
dtypes for the same data, and those differences do not stay cosmetic — they
change the fitted model:

  * An integer column with nulls is float64 from CSV/parquet but Int64 from a
    DB driver. `astype(str)` gives "1.0" vs "1", so the label encoder builds
    different classes and get_dummies emits differently-named columns.
  * Nulls stringify as "nan", "None" or "<NA>" depending on dtype and whether
    the frame has been through parquet.
  * A date is datetime64 from parquet but text from CSV — epoch nanoseconds as
    a feature versus a string category.
  * SQL NUMERIC arrives as decimal.Decimal in an object column.

`BaseAgent.normalize_loaded_frame` collapses all of that at load time. This
test pins the property down: every reader, every column kind, same answer.

Run:  py -3.12 tests/test_reader_normalization.py
"""

import sys
import tempfile
import warnings
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from Agents.BaseAgent.base_agent import BaseAgent
from Agents.FeatureEngineer.agent_feature_engineer import (
    FeatureEngineerAgent, FeatureSpec, _as_encoder_tokens)
from logger import AgentLogger

NORM = BaseAgent.normalize_loaded_frame


def _build_readers():
    src = pd.DataFrame({
        "obj_null":    ["A", "B", None, "C", np.nan],
        "int_null":    [1, 2, None, 4, 5],
        "int_clean":   [10, 20, 30, 40, 50],
        "bool_col":    [True, False, True, False, True],
        "float_col":   [1.5, 2.25, 3.125, 4.0, 5.5],
        "date_col":    pd.to_datetime(["2023-01-31"] * 5),
        "stamp_col":   pd.to_datetime(["2023-01-31 14:30:00"] * 5),
        "cat_numeric": [100, 200, 100, 300, 200],
    })
    tmp = Path(tempfile.mkdtemp(prefix="readers_"))
    src.to_csv(tmp / "d.csv", index=False)
    src.to_parquet(tmp / "d.parquet", index=False)

    readers = {
        "csv": NORM(pd.read_csv(tmp / "d.csv")),
        "parquet": NORM(pd.read_parquet(tmp / "d.parquet")),
    }
    try:
        src.to_excel(tmp / "d.xlsx", index=False)
        readers["excel"] = NORM(pd.read_excel(tmp / "d.xlsx"))
    except Exception as e:                                   # openpyxl absent
        print(f"  (excel skipped: {type(e).__name__})")

    # What a DB driver / pyarrow backend hands over: nullable extension dtypes
    # and Decimal objects.
    sqlish = src.copy()
    sqlish["int_null"] = pd.array([1, 2, None, 4, 5], dtype="Int64")
    sqlish["obj_null"] = pd.array(["A", "B", None, "C", None], dtype="string")
    sqlish["float_col"] = [Decimal("1.5"), Decimal("2.25"), Decimal("3.125"),
                           Decimal("4.0"), Decimal("5.5")]
    sqlish["cat_numeric"] = sqlish["cat_numeric"].astype("category")
    readers["sql/nullable"] = NORM(sqlish)
    return src, readers


def _check(label, readers, cols, fn):
    """Every reader must give the same answer for every column."""
    bad = []
    for c in cols:
        seen = {}
        for name, df in readers.items():
            try:
                v = str(fn(df[c]))
            except Exception as e:
                v = f"ERR {type(e).__name__}: {e}"
            seen.setdefault(v, []).append(name)
        if len(seen) > 1:
            bad.append((c, seen))
    if bad:
        print(f"FAIL  {label}")
        for c, seen in bad:
            print(f"        {c}:")
            for v, who in seen.items():
                print(f"          {','.join(who):<24} {v[:70]}")
        return False
    print(f"PASS  {label}")
    return True


def main() -> int:
    src, readers = _build_readers()
    cols = list(src.columns)
    print(f"readers under test: {', '.join(readers)}\n")

    ok = True
    ok &= _check("dtype identical across readers", readers, cols,
                 lambda s: s.dtype)
    ok &= _check("encoder tokens identical (label encoding)", readers, cols,
                 lambda s: sorted(set(_as_encoder_tokens(s))))
    ok &= _check("get_dummies column names identical (one-hot)", readers, cols,
                 lambda s: sorted(pd.get_dummies(s, prefix="p", drop_first=True).columns))
    ok &= _check("to_numeric values identical (WoE / binning path)", readers, cols,
                 lambda s: np.round(pd.to_numeric(s, errors="coerce")
                                    .to_numpy(dtype=np.float64), 9).tolist())
    ok &= _check("null mask identical", readers, cols,
                 lambda s: s.isna().tolist())

    # End-to-end: fit the encoder on one reader's frame, replay on another's.
    # This is the shape of the real failure — Agent 2 fits on the parquet the
    # pipeline wrote, a replay feeds it the source CSV.
    names = list(readers)
    for fit_on in names:
        for replay_on in names:
            if fit_on == replay_on:
                continue
            ag = FeatureEngineerAgent.__new__(FeatureEngineerAgent)
            ag.logger = AgentLogger()
            ag.name = "FeatureEngineer"
            ag.target_column = None
            ag._protected_cols = set()
            ag._last_label_encoder = None

            fitted = readers[fit_on].copy()
            spec = FeatureSpec()
            for col in ("obj_null", "int_null", "cat_numeric"):
                fitted = ag._tool_encode_categorical(fitted, col, "label")
                spec.label_encoders[col] = ag._last_label_encoder
            replayed = spec.apply(readers[replay_on].copy())
            for col in ("obj_null", "int_null", "cat_numeric"):
                if fitted[col].tolist() != replayed[col].tolist():
                    print(f"FAIL  fit on {fit_on} -> replay on {replay_on}: {col} "
                          f"{fitted[col].tolist()} != {replayed[col].tolist()}")
                    ok = False
    if ok:
        print(f"PASS  label encoder survives every fit/replay reader pair "
              f"({len(names)}x{len(names) - 1} combinations)")

    print("\n" + ("ALL READER NORMALIZATION CHECKS PASSED" if ok
                  else "READER NORMALIZATION CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
