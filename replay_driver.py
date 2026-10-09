#!/usr/bin/env python
"""End-to-end replay of one pipeline run.

Copied verbatim into each run directory as `replay_pipeline.py`, next to the
manifest and specs it reads. Re-runs everything that run decided, here or on
another machine:

    raw input -> split -> cleaning spec -> feature spec -> retrain | score

Nothing here re-decides anything. The LLM calls, FLAML, Optuna, RFE, the PSI
and stability filters and the SHAP prune were all searches over the training
data; their answers are already baked into the files beside this script, so the
replay reads those answers instead of searching again. That is what makes the
result reproducible rather than merely similar.

    python replay_pipeline.py <raw_input> --mode retrain
    python replay_pipeline.py <raw_input> --mode score --out scores.parquet

Modes
-----
retrain : rebuild the exact train/valid/oot partitions, replay both specs,
          refit the final model from the recorded estimator + hyperparameters
          + feature list, and evaluate. Prints a comparison against the metrics
          the original run recorded.
score   : replay both specs and score with final_model.pkl. No split needed.

Requirements
------------
This script imports the pipeline repo, because cleaning_spec.pkl and
feature_spec.pkl are pickles of its classes, and because retraining calls the
same final-fit code the original run used rather than a copy of it. Point it at
the repo with --repo or the AUTOML_REPO environment variable; it also finds the
repo automatically when the run directory still sits inside it.

Pin your library versions to the ones recorded in final_model.pkl. A different
scikit-learn or LightGBM can change a fitted model's behaviour; the script
warns when it sees a mismatch.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_MARKER = "_split_"
_POS = "_split_pos_"
_OCC = "_key_occ_"
# Unseen-category share above which scoring is refused (silent -> __NA__ otherwise).
MAX_UNSEEN_RATE = 0.20


def _load_manifest() -> dict:
    path = _HERE / "replay_manifest.json"
    if not path.exists():
        sys.exit(f"ERROR: {path.name} not found next to this script.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _add_repo_to_path(repo):
    """Make the pipeline package importable."""
    candidates = []
    if repo:
        candidates.append(Path(repo))
    if os.environ.get("AUTOML_REPO"):
        candidates.append(Path(os.environ["AUTOML_REPO"]))
    # outputs/<date>/run_NN/ -> the repo root is a few levels up
    candidates.extend(_HERE.parents[:4])

    for cand in candidates:
        if (cand / "Agents").is_dir() and (cand / "config.py").is_file():
            sys.path.insert(0, str(cand))
            print(f"  repo        : {cand}")
            return
    sys.exit(
        "ERROR: could not locate the pipeline repo.\n"
        "       Pass --repo /path/to/multi-agent-auto-ml-v1.1 or set AUTOML_REPO.\n"
        "       It is needed to unpickle the specs and to reuse the original\n"
        "       final-fit code."
    )


def _check_files(manifest: dict, mode: str) -> None:
    """Fail early, naming every missing file at once.

    This bundle gets copied between machines by hand, so a forgotten file is
    the most likely failure. One clear list beats a FileNotFoundError traceback
    from somewhere deep in the run.
    """
    needed = [
        (manifest["cleaning"]["spec_file"], "Agent 1 cleaning spec"),
        *([(manifest["null_processing"]["file"], "null processor")]
          if manifest.get("null_processing") else []),
        (manifest["feature_engineering"]["spec_file"], "Agent 2 feature spec"),
        (manifest["training"]["model_file"], "Agent 3 trained model"),
    ]
    if mode == "retrain":
        assign = manifest["split"].get("assignment_file")
        if not assign:
            sys.exit("ERROR: this run recorded no split assignment, so retrain mode is "
                     "unavailable. Use --mode score, or re-run the pipeline with an "
                     "--entity-id / --composite-key so the split can be frozen.")
        needed.append((assign, "train/valid/oot split assignment"))

    missing = [(f, what) for f, what in needed if not (_HERE / f).exists()]
    if missing:
        lines = "\n".join(f"         {f:<28} ({what})" for f, what in missing)
        sys.exit(f"ERROR: {len(missing)} file(s) missing from {_HERE}:\n{lines}\n"
                 "       Copy the whole run directory, or at least these files, "
                 "alongside replay_pipeline.py.")


def _report_provenance(manifest: dict) -> None:
    """Show what produced the bundle and warn when this checkout/input differs."""
    prov = manifest.get("provenance")
    if not prov:
        print("  NOTE: manifest has no provenance block (older bundle).")
        return
    code = prov.get("code", {})
    print(f"  provenance  : git {str(code.get('git_sha'))[:10]} "
          f"({'dirty' if code.get('git_dirty') else 'clean'}) | python "
          f"{prov.get('runtime', {}).get('python')}")
    try:
        from provenance import code_fingerprint
        if code.get("code_fingerprint") and code_fingerprint() != code["code_fingerprint"]:
            print("  WARNING: the repo code here differs from the code that produced this bundle "
                  "(code_fingerprint mismatch). Spec pickles depend on that code.")
    except Exception as e:                         # provenance is advisory
        print(f"  NOTE: could not compare code fingerprint ({type(e).__name__})")


def _verify_artifacts(manifest: dict) -> None:
    """Refuse to unpickle a file whose SHA-256 differs from the manifest.

    joblib.load executes code, so a swapped or edited spec/model is arbitrary
    code execution. Bundles written before hashes existed are loaded with a notice.
    """
    import hashlib
    expected = manifest.get("artifacts")
    if not expected:
        print("  NOTE: manifest has no artifact hashes (older bundle) — integrity NOT verified.")
        return
    bad = []
    for name, want in expected.items():
        path = _HERE / name
        if not path.exists():
            continue
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            bad.append(name)
    if bad:
        sys.exit(f"ERROR: artifact hash mismatch for {bad}. The file differs from what the "
                 "original run wrote; refusing to load it.")
    print(f"  integrity   : {len(expected)} artifact hash(es) verified")


def _check_versions(manifest: dict, allow_mismatch: bool = False) -> None:
    """Refuse to run on different library versions unless explicitly allowed.

    Covers python/sklearn/pandas/numpy AND the booster that was pickled
    (xgboost / lightgbm / catboost) — the library whose pickle is most
    version-sensitive.
    """
    import importlib
    import joblib
    model_file = _HERE / manifest["training"]["model_file"]
    if not model_file.exists():
        return
    trained = (joblib.load(model_file) or {}).get("versions", {})
    if not trained:
        return
    import sklearn
    current = {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
               "pandas": pd.__version__, "numpy": np.__version__}
    for lib in ("xgboost", "lightgbm", "catboost"):
        if lib in trained:
            try:
                current[lib] = importlib.import_module(lib).__version__
            except ImportError:
                current[lib] = "NOT INSTALLED"
    diffs = [f"{k}: run={trained[k]} here={current[k]}"
             for k in current if k in trained and trained[k] != current[k]]
    if diffs:
        print("  Library versions differ from the original run:")
        for d in diffs:
            print(f"             {d}")
        if not allow_mismatch:
            sys.exit("ERROR: refusing to continue on mismatched versions (results could silently "
                     "differ). Pin the versions above, or pass --allow-version-mismatch.")
        print("  WARNING: continuing because --allow-version-mismatch was given.")


def _assign_splits(df: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    """Attach the original partition label to every row.

    Joins on the recorded key. Rows the original run never saw get no label and
    are dropped with a notice — silently training on them would be exactly the
    kind of drift this bundle exists to prevent.
    """
    split_cfg = manifest["split"]
    if not split_cfg.get("assignment_file"):
        sys.exit("ERROR: this run has no split assignment recorded, so retrain mode "
                 "is unavailable. Use --mode score, or re-run the pipeline with an "
                 "--entity-id / --composite-key so the split can be frozen.")

    assignment = pd.read_parquet(_HERE / split_cfg["assignment_file"])
    key_cols = list(split_cfg["key_cols"])

    if split_cfg.get("positional_fallback"):
        # The original run had no identity key, so rows were frozen by position.
        pos_col = key_cols[0]
        if len(df) != len(assignment):
            sys.exit(f"ERROR: the split was frozen by row position, but this input has "
                     f"{len(df)} rows and the original had {len(assignment)}. "
                     "Positional replay needs the identical file.")
        print("  NOTE: split frozen by ROW POSITION (no key was available). This is "
              "only correct if the input rows are in their original order.")
        df = df.copy()
        df[pos_col] = np.arange(len(df), dtype=np.int64)

    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: input is missing split key column(s): {missing}")

    join_cols = list(key_cols)
    if _OCC in assignment.columns:
        # Rebuild the same occurrence rank the original run numbered on the raw
        # file. All zeros when the key is unique, so this stays a plain key
        # join; it only matters for rows that share a key, which a run with
        # dedup leaves behind by construction.
        df = df.copy()
        df[_OCC] = df.groupby(key_cols, sort=False, dropna=False).cumcount()
        join_cols.append(_OCC)

    before = len(df)
    out = df.merge(assignment, on=join_cols, how="left")
    if len(out) > before:
        sys.exit(f"ERROR: joining on {join_cols} turned {before} rows into {len(out)} — "
                 "those rows cannot be matched to a partition. Re-run the pipeline "
                 "with a composite key that is unique per row.")
    if _OCC in out.columns:
        out = out.drop(columns=[_OCC])
    if _POS not in out.columns:
        print("  NOTE: this bundle predates row-order recording. Partition "
              "membership will match, but row order inside each partition may "
              "not, which can shift cross-validation folds slightly.")
    unmatched = int(out[_MARKER].isna().sum())
    out.attrs["unmatched"] = unmatched
    if unmatched:
        print(f"  NOTE: {unmatched}/{before} input rows are not in the original split "
              "and were dropped.")
        out = out[out[_MARKER].notna()]
    return out


def _apply_nulls(df: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    """Stage 1b of the original run. Bundles that predate it have no entry."""
    info = manifest.get("null_processing")
    if not info:
        return df
    from preprocessing import NullProcessor
    proc = NullProcessor.load(_HERE / info["file"], expected_sha256=info.get("sha256"))
    return proc.transform(df)


def _replay_specs(df: pd.DataFrame, manifest: dict, is_train: bool) -> pd.DataFrame:
    """CleaningSpec then FeatureSpec, exactly as the original run applied them."""
    import joblib
    clean_spec = joblib.load(_HERE / manifest["cleaning"]["spec_file"])
    feat_spec = joblib.load(_HERE / manifest["feature_engineering"]["spec_file"])

    df = clean_spec.apply(df)
    if is_train:
        # Row ops ran on train only in the original run. Replaying them on a
        # holdout would delete evaluation rows and flatter the metrics.
        df = clean_spec.apply_row_ops(df, target_column=manifest["target_column"])
    df = _apply_nulls(df, manifest)
    return feat_spec.apply(df, strict=True, max_unseen_rate=MAX_UNSEEN_RATE)


def run_score(df: pd.DataFrame, manifest: dict, out_path) -> None:
    import joblib
    engineered = _replay_specs(df, manifest, is_train=False)

    artifact = joblib.load(_HERE / manifest["training"]["model_file"])
    models = artifact.get("ensemble_models") or [artifact["model"]]
    calibrator = artifact.get("calibrator")
    features = artifact["feature_cols"]
    encoders = artifact["cat_encoders"]

    avail = [c for c in features if c in engineered.columns]
    if len(avail) != len(features):
        absent = [c for c in features if c not in engineered.columns]
        sys.exit(f"ERROR: {len(absent)} model feature(s) absent from the engineered frame, "
                 f"e.g. {absent[:8]}. Refusing to score with a partial feature set.")
    X = engineered[avail].copy()
    for col in X.columns:
        if col in encoders:
            le = encoders[col]
            vals = X[col].fillna("__NA__").astype(str)
            X[col] = le.transform(vals.where(vals.isin(set(le.classes_)), "__NA__"))
        else:
            X[col] = X[col].fillna(-999)

    raw = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    score = (np.clip(calibrator.predict(raw), artifact.get("pd_floor", 1e-15),
                     artifact.get("pd_cap", 1 - 1e-15))
             if calibrator is not None else raw)

    print(f"\n  rows scored : {len(score)}")
    print(f"  score range : {score.min():.6f} - {score.max():.6f}")
    print(f"  mean score  : {score.mean():.6f}")

    if out_path:
        result = pd.DataFrame({"score": score})
        for k in manifest["split"]["key_cols"]:
            if k in engineered.columns:
                result[k] = engineered[k].values
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if str(out_path).endswith(".csv"):
            result.to_csv(out_path, index=False)
        else:
            result.to_parquet(out_path, index=False)
        print(f"  written     : {out_path}")


# Row counts and seed counts must match exactly — a difference there means the
# replayed DATA is wrong, which no tolerance should forgive. Score metrics live
# in [0,1] and get a numeric tolerance. Everything else (notably best_iteration)
# is reported but does not decide the verdict.
_STRUCTURAL_METRICS = {"final_train_rows", "n_seeds"}
_SCORE_HINTS = ("auc", "brier", "ks", "gini", "logloss", "precision",
                "recall", "f1", "accuracy")


def _metric_kind(key: str) -> str:
    if key in _STRUCTURAL_METRICS:
        return "structural"
    if any(h in key.lower() for h in _SCORE_HINTS):
        return "score"
    return "info"


def _diagnose(df: pd.DataFrame, labelled: pd.DataFrame, splits: dict,
              manifest: dict) -> None:
    """Walk the chain and name the earliest stage that diverged.

    A bare "final_train_rows differ" tells the user something is wrong but not
    what to do about it. The partition counts, the join, the row ops and the
    feature spec each fail in a recognisable way, so check them in the order
    they run and report the first one that is off, with the fix.
    """
    print()
    print("  == why it did not reproduce ==")
    recorded = manifest["split"].get("sizes") or {}
    key_cols = manifest["split"]["key_cols"]
    found = False

    # 1. did every input row find a partition?
    unmatched = int(labelled.attrs.get("unmatched", 0))
    if unmatched:
        found = True
        print(f"  * {unmatched} input row(s) are not in the recorded split.")
        print(f"    The input file does not contain exactly the rows the original run")
        print(f"    read. Point the replay at that same file, or re-run the pipeline.")

    # 2. per-partition counts, before vs after the specs
    for tag, want in recorded.items():
        if tag in ("valid_temporal", "valid_random") or tag not in splits:
            continue
        got = len(splits[tag])
        if got == want:
            continue
        found = True
        raw_n = int((labelled["_split_"] == tag).sum())
        print(f"  * {tag}: {got} rows after replaying the specs, "
              f"original run had {want}.")
        if raw_n == want and tag == "train":
            print(f"    The join found the right {raw_n} rows, so the loss happened in")
            print(f"    cleaning. Check cleaning.row_ops in replay_manifest.json — a")
            print(f"    dedup or sample there behaves differently on this input.")
        elif raw_n != want:
            print(f"    The join matched {raw_n} rows for this partition, so the split")
            print(f"    assignment and the input file disagree before cleaning even runs.")
            print(f"    Most often the key {key_cols} is not unique in this file.")
        else:
            print(f"    Rows were added or dropped while applying the feature spec.")

    # 3. are the recorded features actually present?
    train = splits.get("train")
    if train is not None and len(train):
        want_feats = manifest["training"].get("final_features") or []
        missing = [c for c in want_feats if c not in train.columns]
        if missing:
            found = True
            print(f"  * {len(missing)} recorded feature(s) absent after the feature "
                  f"spec, e.g. {missing[:5]}.")
            print(f"    The spec did not rebuild the same columns — usually a source")
            print(f"    column was renamed or dropped upstream of this replay.")

    if not found:
        print("  * The partitions and features all match, so the difference is in the")
        print("    fit itself. Compare the library versions printed above against the")
        print("    `versions` recorded in final_model.pkl.")


def run_retrain(df: pd.DataFrame, manifest: dict) -> None:
    from logger import AgentLogger
    from Agents.TrainModel.agent_train_model import TrainModelAgent

    labelled = _assign_splits(df, manifest)

    splits = {}

    for tag in ("train", "valid", "oot", "test"):
        part = labelled[labelled[_MARKER] == tag]
        if _POS in part.columns:
            # Restore the original run's row order within the partition. Fold
            # assignment is positional, so same rows in a different order is a
            # different model.
            part = part.sort_values(_POS, kind="mergesort")
        part = part.drop(columns=[c for c in (_MARKER, _POS) if c in part.columns])
        part = _replay_specs(part.reset_index(drop=True), manifest, tag == "train")
        splits[tag] = part.reset_index(drop=True)
        if len(part):
            print(f"  {tag:<6}      : {len(part)} rows x {part.shape[1]} cols")

    empty = pd.DataFrame(columns=splits["train"].columns)
    splits.setdefault("valid_temporal", empty.copy())
    splits.setdefault("valid_random", empty.copy())

    for tag, n in (manifest["split"].get("sizes") or {}).items():
        if tag in splits and len(splits[tag]) != n:
            print(f"  WARNING: {tag} has {len(splits[tag])} rows, "
                  f"the original run had {n}.")

    tr = manifest["training"]
    metrics = TrainModelAgent(AgentLogger()).replay_fit(
        splits=splits,
        target_column=manifest["target_column"],
        estimator_name=tr["estimator"],
        best_params=tr["best_params"],
        feature_cols=tr["final_features"],
        date_col=(manifest["split"].get("temporal") or {}).get("date_col"),
        calibration=tr.get("calibration"),
    )

    print()
    print("  -- metrics: replay vs original run --")
    expected = manifest.get("expected_metrics") or {}
    worst_score = 0.0
    structural_bad, info_drift = [], []

    for key in sorted(set(metrics) | set(expected)):
        got, want = metrics.get(key), expected.get(key)
        numeric = (isinstance(got, (int, float)) and isinstance(want, (int, float))
                   and not isinstance(got, bool) and not isinstance(want, bool))
        if not numeric:
            if want is not None or got is not None:
                same = "ok" if got == want else "DIFFERS"
                print(f"    {key:<28} replay={got} original={want} {same}")
                if same == "DIFFERS":
                    structural_bad.append(key)
            continue

        delta = abs(got - want)
        kind = _metric_kind(key)
        if kind == "structural":
            flag = "ok" if delta == 0 else "DIFFERS"
            if delta:
                structural_bad.append(key)
        elif kind == "score":
            worst_score = max(worst_score, delta)
            flag = "ok" if delta < 1e-9 else ("close" if delta < 1e-3 else "DIFFERS")
        else:
            # Counts like best_iteration: a few trees either way is normal when
            # the booster is fitted multi-threaded, so it is reported but does
            # not decide the verdict. Weighing its absolute delta against an AUC
            # tolerance would be comparing a tree count to a probability.
            flag = "ok" if delta == 0 else "info"
            if delta:
                info_drift.append(f"{key} {want:g}->{got:g}")
        print(f"    {key:<28} replay={got:<12.6g} original={want:<12.6g} {flag}")

    print()
    if structural_bad:
        print("  DID NOT reproduce: " + ", ".join(structural_bad) + " differ.")
        _diagnose(df, labelled, splits, manifest)
    elif worst_score < 1e-9:
        print("  Reproduced exactly.")
    elif worst_score < 1e-3:
        print(f"  Reproduced within {worst_score:.1e} on every score metric.")
        if info_drift:
            print("  Non-decisive drift: " + "; ".join(info_drift) + ".")
            print("  A few trees either way is expected from multi-threaded boosting "
                  "(float reduction order), not a sign the replay failed.")
    else:
        print(f"  DID NOT reproduce: worst score metric off by {worst_score:.4f}.")
        _diagnose(df, labelled, splits, manifest)

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay a finished pipeline run end to end.")
    ap.add_argument("input", help="raw input file (the one the original run read)")
    ap.add_argument("--mode", choices=["retrain", "score"], default="retrain")
    ap.add_argument("--repo", default=None, help="path to the pipeline repo")
    ap.add_argument("--out", default=None, help="score mode: where to write scores")
    ap.add_argument("--allow-version-mismatch", action="store_true",
                    help="continue even if library versions differ from the original run")
    args = ap.parse_args()

    manifest = _load_manifest()
    print(f"Replaying run from {manifest['created_at']}  (mode={args.mode})")
    print(f"  run dir     : {manifest['run_dir']}")
    _add_repo_to_path(args.repo)

    from Agents.BaseAgent.base_agent import BaseAgent
    _check_files(manifest, args.mode)
    _verify_artifacts(manifest)
    _report_provenance(manifest)
    _check_versions(manifest, args.allow_version_mismatch)

    df = BaseAgent.load_dataframe(args.input)
    print(f"  input       : {args.input}  {df.shape}")

    if args.mode == "score":
        run_score(df, manifest, args.out)
    else:
        run_retrain(df, manifest)


if __name__ == "__main__":
    main()
