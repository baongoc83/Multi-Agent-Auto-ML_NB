"""Config-driven missing-value processor (Pandas + scikit-learn).

Contract
--------
    TRAIN                      -> fit()        learn + freeze every parameter
    VALID / OOT / PRODUCTION   -> transform()  apply the frozen parameters only

`transform` never learns anything. There is deliberately no code path that
re-estimates a median/mean on non-train data.

Per-feature rule (see `FeatureRule`):
    missing_strategy : none | zero | mean | median | constant |
                       explicit_category | missing_indicator_only
    add_missing_indicator : bool   -> adds `<col>_missing` (0/1), computed
                            BEFORE any imputation
    missing_type     : structural | behavioral | data_quality | unknown
                       (documentation + lint only; it never changes values)

NULL is never turned into 0 unless the rule says `zero`.

Scale guard (unit changes): for numerical columns the batch's magnitude
(median and p95 of the non-zero |x|) is compared with the TRAIN statistics
frozen at fit. It is `critical` (raises with guard_action="raise", otherwise
reported in `report_["issues"]`) only when BOTH move the same way by >= `scale_fail_ratio`
(a VND -> thousand-VND switch rescales every quantile and otherwise scores
silently); any single ratio beyond `scale_warn_ratio` — e.g. a heavy-tailed
column whose median moves while p95 holds — only warns as a distribution shift.
Only batches with >= `min_rows_for_distribution` non-null values are judged,
so single-row online scoring is never rejected on distribution grounds.

Fail-closed at transform time (scoring silently degrading was a verified
production risk): missing required columns, non-numeric junk in a numeric
column above `max_coerce_rate`, and unconfigured columns (default) all raise.
Missing-rate drift vs train is reported in `report_` and emitted as warnings.

The fitted artifact is plain JSON (no pickle) with a SHA-256 over its
content, verified on load.

No dependency on the rest of the repo: a scoring environment needs only
pandas, numpy and scikit-learn (PyYAML only if you load YAML configs).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer

ARTIFACT_VERSION = 1

STRATEGIES = ("none", "zero", "mean", "median", "constant",
              "explicit_category", "missing_indicator_only")
MISSING_TYPES = ("structural", "behavioral", "data_quality", "unknown")
GROUPS = ("numerical", "categorical", "binary")

# Which strategies make sense for which group.
_ALLOWED: Dict[str, Tuple[str, ...]] = {
    "numerical": ("none", "zero", "mean", "median", "constant", "missing_indicator_only"),
    "binary": ("none", "zero", "constant", "missing_indicator_only"),
    "categorical": ("none", "constant", "explicit_category", "missing_indicator_only"),
}
DEFAULT_CATEGORY = "__MISSING__"


class NullConfigError(ValueError):
    """The rule configuration is invalid."""


class NullContractError(ValueError):
    """Data handed to transform() violates what was frozen at fit()."""


# ───────────────────────────── rules / config ──────────────────────────────

@dataclass
class FeatureRule:
    group: str = "numerical"
    missing_strategy: str = "median"
    add_missing_indicator: bool = False
    missing_type: str = "unknown"
    fill_value: Any = None            # for `constant`; optional override for explicit_category
    source: str = "manual"            # manual | llm | heuristic (audit trail only)
    reason: str = ""                  # free-text justification (advisor / reviewer), metadata only

    def validate(self, name: str) -> "FeatureRule":
        if self.group not in GROUPS:
            raise NullConfigError(f"{name}: group '{self.group}' not in {GROUPS}")
        if self.missing_strategy not in STRATEGIES:
            raise NullConfigError(f"{name}: missing_strategy '{self.missing_strategy}' not in {STRATEGIES}")
        if self.missing_type not in MISSING_TYPES:
            raise NullConfigError(f"{name}: missing_type '{self.missing_type}' not in {MISSING_TYPES}")
        if self.missing_strategy not in _ALLOWED[self.group]:
            raise NullConfigError(
                f"{name}: strategy '{self.missing_strategy}' is not valid for group "
                f"'{self.group}' (allowed: {_ALLOWED[self.group]})")
        if self.missing_strategy == "constant" and self.fill_value is None:
            raise NullConfigError(f"{name}: strategy 'constant' needs fill_value")
        if self.missing_strategy == "missing_indicator_only" and not self.add_missing_indicator:
            raise NullConfigError(
                f"{name}: 'missing_indicator_only' requires add_missing_indicator: true")
        return self


_RULE_KEYS = {f for f in FeatureRule.__dataclass_fields__}


def _rule_from_dict(name: str, d: Dict[str, Any], base: Optional[FeatureRule] = None) -> FeatureRule:
    unknown = set(d) - _RULE_KEYS - {"type"}
    if unknown:
        raise NullConfigError(f"{name}: unknown rule keys {sorted(unknown)}")
    data = asdict(base) if base else {}
    data.update({k: v for k, v in d.items() if k in _RULE_KEYS})
    # `type: numerical` in the example config is an alias of `group`
    if "type" in d and "group" not in d:
        data["group"] = d["type"]
    return FeatureRule(**data).validate(name)


def load_config(config: Any) -> Dict[str, Any]:
    """dict | path to .yaml/.yml/.json -> dict."""
    if isinstance(config, dict):
        return config
    path = Path(config)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml  # optional dependency
        return yaml.safe_load(text) or {}
    return json.loads(text)


def resolve_rules(config: Dict[str, Any]) -> Dict[str, FeatureRule]:
    """Expand `feature_groups` + `group_defaults` + per-feature overrides."""
    group_defaults: Dict[str, FeatureRule] = {}
    for g, d in (config.get("group_defaults") or {}).items():
        if g not in GROUPS:
            raise NullConfigError(f"group_defaults: unknown group '{g}'")
        group_defaults[g] = _rule_from_dict(f"group_defaults.{g}", {"group": g, **d})

    rules: Dict[str, FeatureRule] = {}
    group_of: Dict[str, str] = {}
    for g, cols in (config.get("feature_groups") or {}).items():
        if g not in GROUPS:
            raise NullConfigError(f"feature_groups: unknown group '{g}'")
        for c in cols or []:
            if c in group_of and group_of[c] != g:
                raise NullConfigError(f"{c}: listed in groups '{group_of[c]}' and '{g}'")
            group_of[c] = g

    features = config.get("features") or {}
    for name in list(group_of) + [f for f in features if f not in group_of]:
        override = dict(features.get(name) or {})
        g = override.get("group") or override.get("type") or group_of.get(name)
        if g is None:
            raise NullConfigError(f"{name}: no group (set `group`/`type` or list it in feature_groups)")
        if name in group_of and g != group_of[name]:
            raise NullConfigError(f"{name}: group '{g}' conflicts with feature_groups '{group_of[name]}'")
        base = group_defaults.get(g)
        if base is None and not (override.keys() & {"missing_strategy"}):
            raise NullConfigError(
                f"{name}: no missing_strategy and no group_defaults for group '{g}'")
        override["group"] = g
        rules[name] = _rule_from_dict(name, override, base)
    return rules


# ───────────────────────────── the transformer ─────────────────────────────

class NullProcessor(BaseEstimator, TransformerMixin):
    """sklearn-compatible, config-driven NULL handling.

    Parameters
    ----------
    config : dict | str | Path
        See module docstring / `resolve_rules`.
    on_unconfigured : "error" | "ignore"
        Columns of X with no rule. "error" (default) forces an explicit decision.
    indicator_suffix : str
    treat_inf_as_missing : bool
        +-inf in numeric columns (e.g. ratios) count as NULL.
    max_coerce_rate : float
        transform(): max share of non-null values in a numeric column that may
        be unparseable (-> NaN). Above it raises NullContractError.
    drift_warn_threshold : float
        transform(): warn when |missing_rate - train_missing_rate| exceeds this.
    strict : bool
        transform(): raise on missing required columns (default True).
    scale_warn_ratio, scale_fail_ratio : float
        Magnitude ratio (batch / train, either direction) that warns / raises.
        scale_fail_ratio <= 0 disables the guard.
    guard_action : "raise" | "warn"
        What a data-quality guard does when tripped (scale shift >= scale_fail_ratio,
        unparseable share > max_coerce_rate). "raise" stops with NullContractError;
        "warn" records a `critical` issue in `report_["issues"]`, emits a warning and
        carries on (thresholds are domain-specific, the user decides). Structural
        contract errors (absent required column, tampered artifact) always raise.
    min_rows_for_distribution : int
        Minimum non-null rows before a batch is judged on distribution.
    max_indicators : int
        Cap on `<col>_missing` columns requested through add_missing_indicator.
        fit() first drops indicators that are constant or identical to an earlier
        one, then keeps the `max_indicators` most informative (|P(y|missing) -
        P(y|present)| z-score when y is given, else highest missing rate).
        `missing_indicator_only` rules are exempt (their indicator is the feature).
    """

    def __init__(self, config: Any, on_unconfigured: str = "error",
                 indicator_suffix: str = "_missing", treat_inf_as_missing: bool = True,
                 max_coerce_rate: float = 0.0, drift_warn_threshold: float = 0.10,
                 strict: bool = True, scale_warn_ratio: float = 3.0,
                 scale_fail_ratio: float = 10.0, min_rows_for_distribution: int = 100,
                 max_indicators: int = 300, guard_action: str = "raise"):
        if guard_action not in ("raise", "warn"):
            raise ValueError(f"guard_action must be 'raise' or 'warn', got {guard_action!r}")
        self.config = config
        self.on_unconfigured = on_unconfigured
        self.indicator_suffix = indicator_suffix
        self.treat_inf_as_missing = treat_inf_as_missing
        self.max_coerce_rate = max_coerce_rate
        self.drift_warn_threshold = drift_warn_threshold
        self.strict = strict
        self.scale_warn_ratio = scale_warn_ratio
        self.scale_fail_ratio = scale_fail_ratio          # <= 0 disables the guard
        self.min_rows_for_distribution = min_rows_for_distribution
        self.max_indicators = max_indicators              # <= 0: unlimited
        self.guard_action = guard_action

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _numeric(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce") if not pd.api.types.is_numeric_dtype(s) else s

    def _as_missing_aware(self, s: pd.Series, group: str) -> Tuple[pd.Series, int, int]:
        """-> (series with inf->NaN and coerced to numeric if numeric-like,
        n_coerced, n_inf)."""
        n_coerced = n_inf = 0
        if group in ("numerical", "binary"):
            was_null = s.isna()
            num = self._numeric(s)
            n_coerced = int((num.isna() & ~was_null).sum())
            if self.treat_inf_as_missing:
                inf_mask = np.isinf(num.astype("float64"))
                n_inf = int(inf_mask.sum())
                if n_inf:
                    num = num.mask(inf_mask)
            return num, n_coerced, n_inf
        return s, 0, 0

    @staticmethod
    def _stats(s: pd.Series, group: str) -> Dict[str, Any]:
        n = max(len(s), 1)
        out: Dict[str, Any] = {
            "dtype": str(s.dtype),
            "missing_rate": round(float(s.isna().mean()), 6),
            "unique_count": int(s.nunique(dropna=True)),
        }
        if group in ("numerical", "binary"):
            nn = s.dropna().astype("float64")
            out["zero_rate"] = round(float((nn == 0).sum() / n), 6)
            if len(nn):
                q = nn.quantile([0.25, 0.5, 0.75, 0.95])
                a = nn.abs()
                nz = a[a > 0]                      # sparse / zero-heavy columns: judge non-zero values
                out.update(min=float(nn.min()), max=float(nn.max()), mean=float(nn.mean()),
                           median=float(q[0.5]), p25=float(q[0.25]), p75=float(q[0.75]),
                           p95=float(q[0.95]), abs_median=float(a.median()),
                           abs_p95=float(a.quantile(0.95)), nz_count=int(len(nz)),
                           abs_nz_median=float(nz.median()) if len(nz) else None,
                           abs_nz_p95=float(nz.quantile(0.95)) if len(nz) else None)
            else:
                out.update(min=None, max=None, mean=None, median=None, p25=None, p75=None,
                           p95=None, abs_median=None, abs_p95=None, nz_count=0,
                           abs_nz_median=None, abs_nz_p95=None)
        else:
            out["zero_rate"] = None
        return out

    def _fill_value(self, name: str, s: pd.Series, rule: FeatureRule) -> Any:
        st = rule.missing_strategy
        if st in ("none", "missing_indicator_only"):
            return None
        if st == "zero":
            return 0.0
        if st == "explicit_category":
            return DEFAULT_CATEGORY if rule.fill_value is None else rule.fill_value
        if st == "constant":
            return rule.fill_value
        # mean / median -> let scikit-learn compute the statistic
        if s.notna().sum() == 0:
            raise NullConfigError(
                f"{name}: strategy '{st}' impossible, column is 100% null in TRAIN. "
                "Use constant/zero/none or drop the column upstream.")
        imp = SimpleImputer(strategy=st, keep_empty_features=False)
        imp.fit(s.astype("float64").to_frame())
        return float(imp.statistics_[0])

    # ---- fit ---------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Any = None) -> "NullProcessor":
        if self.on_unconfigured not in ("error", "ignore"):
            raise NullConfigError("on_unconfigured must be 'error' or 'ignore'")
        cfg = load_config(self.config)
        rules = resolve_rules(cfg)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("NullProcessor needs a pandas DataFrame")

        unconfigured = [c for c in X.columns if c not in rules]
        if unconfigured and self.on_unconfigured == "error":
            raise NullConfigError(
                f"{len(unconfigured)} column(s) have no rule, e.g. {unconfigured[:8]}. "
                "Add them to the config, or set on_unconfigured='ignore'.")
        absent = [c for c in rules if c not in X.columns]
        if absent:
            raise NullConfigError(f"rules refer to columns absent from TRAIN: {absent[:8]}")

        params: Dict[str, Dict[str, Any]] = {}
        lint: List[str] = []
        yv = None
        if y is not None:
            yv = pd.to_numeric(pd.Series(np.asarray(y)), errors="coerce").to_numpy(dtype="float64")
        cand: Dict[str, Dict[str, Any]] = {}       # indicator candidates: digest + score
        for name, rule in rules.items():
            s, _, _ = self._as_missing_aware(X[name], rule.group)
            if rule.add_missing_indicator and rule.missing_strategy != "missing_indicator_only":
                mk = s.isna().to_numpy()
                n1 = int(mk.sum())
                z = float(s.isna().mean())
                if yv is not None and 0 < n1 < len(mk):
                    ok = ~np.isnan(yv)
                    a, b = yv[ok & mk], yv[ok & ~mk]
                    if len(a) and len(b):
                        p = np.nanmean(yv[ok])
                        se = np.sqrt(max(p * (1 - p), 1e-12) * (1 / len(a) + 1 / len(b)))
                        z = abs(a.mean() - b.mean()) / se
                cand[name] = {"n_missing": n1, "score": z,
                              "digest": hashlib.md5(np.packbits(mk).tobytes()).hexdigest()}
            st = self._stats(s, rule.group)
            fill = self._fill_value(name, s, rule)
            if rule.group == "categorical" and rule.missing_strategy == "constant":
                fill = rule.fill_value
            params[name] = {"rule": asdict(rule), "stats": st, "imputation_value": fill}
            if rule.missing_type == "structural" and not rule.add_missing_indicator:
                lint.append(f"{name}: structural missing without a missing indicator discards signal")
            if rule.missing_strategy == "zero" and rule.missing_type in ("structural", "unknown"):
                lint.append(f"{name}: 'zero' on a {rule.missing_type} NULL: confirm NULL really means 0")

        # ---- indicator hygiene: constant / duplicate / over the cap -> dropped ----
        kept: Dict[str, str] = {}
        n_kept = 0
        for name in sorted(cand, key=lambda n: -cand[n]["score"]):
            c = cand[name]
            why = None
            if c["n_missing"] == 0:
                why = "never missing in TRAIN (constant)"
            elif c["digest"] in kept:
                why = f"same missing pattern as '{kept[c['digest']]}'"
            elif self.max_indicators and self.max_indicators > 0 and n_kept >= self.max_indicators:
                why = f"over max_indicators={self.max_indicators}"
            if why:
                rules[name].add_missing_indicator = False
                params[name]["rule"]["add_missing_indicator"] = False
                lint.append(f"{name}: indicator dropped ({why})")
            else:
                kept[c["digest"]] = name
                n_kept += 1

        self.rules_: Dict[str, FeatureRule] = rules
        self.params_: Dict[str, Dict[str, Any]] = params
        self.columns_in_: List[str] = [c for c in X.columns]
        self.configured_columns_: List[str] = list(rules)
        self.passthrough_columns_: List[str] = unconfigured
        self.indicator_columns_: List[str] = [
            f"{n}{self.indicator_suffix}" for n, r in rules.items() if r.add_missing_indicator]
        self.lint_: List[str] = lint
        self.report_: Dict[str, Any] = {}
        # Per-indicator drops are routine on wide data: one summary warning, details in lint_.
        dropped = [m for m in lint if "indicator dropped" in m]
        for msg in lint:
            if "indicator dropped" not in msg:
                warnings.warn(f"NullProcessor lint: {msg}", stacklevel=2)
        if dropped:
            warnings.warn(f"NullProcessor: {len(dropped)} indicator(s) dropped "
                          "(constant / duplicate / over cap); see lint_", stacklevel=2)
        return self

    def _scale_check(self, name: str, s: pd.Series, st: Dict[str, Any]) -> Dict[str, Any]:
        """Magnitude comparison of a batch against frozen TRAIN statistics."""
        # Magnitude of the NON-ZERO values only: a column that is mostly 0/NULL must not
        # trip the guard because its p95 collapses to 0 in one partition (no scale change).
        if self.scale_fail_ratio <= 0 or st.get("abs_nz_p95") is None:
            return {}
        a = s.dropna().astype("float64").abs()
        a = a[a > 0]
        if len(a) < self.min_rows_for_distribution or (st.get("nz_count") or 0) < self.min_rows_for_distribution:
            return {}
        res: Dict[str, Any] = {}
        for key, now in (("abs_nz_p95", float(a.quantile(0.95))), ("abs_nz_median", float(a.median()))):
            ref = st.get(key)
            if ref is None or ref <= 0:          # zero-dominated in train: ratio undefined
                continue
            ratio = (now + 1e-12) / ref
            res[key] = {"ratio": round(ratio, 6), "train": ref, "now": now}
        if not res:
            return {}
        ratios = [v["ratio"] for v in res.values()]
        fold = lambda r: max(r, 1.0 / max(r, 1e-12))
        worst = max(fold(r) for r in ratios)
        # A unit change rescales EVERY quantile by the same factor, so median and p95
        # must move together in the same direction. Only that common shift can fail;
        # one quantile moving alone (heavy-tailed / sparse columns) is a shape drift.
        if all(r > 1 for r in ratios) or all(r < 1 for r in ratios):
            unit = min(fold(r) for r in ratios)
        else:
            unit = 1.0
        res["worst_factor"] = round(worst, 3)
        res["unit_factor"] = round(unit, 3)
        res["level"] = ("fail" if unit >= self.scale_fail_ratio
                        else "warn" if worst >= self.scale_warn_ratio else "ok")
        return res

    # ---- transform ---------------------------------------------------------
    def _check_fitted(self) -> None:
        if not hasattr(self, "params_"):
            raise NotFittedError("NullProcessor is not fitted; call fit(X_train) first")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        missing_cols = [c for c in self.configured_columns_ if c not in X.columns]
        if missing_cols and self.strict:
            raise NullContractError(
                f"{len(missing_cols)} required column(s) absent from input, e.g. {missing_cols[:8]}")

        out = X.copy()
        # issues: structured, one entry per (column, check) — what reports render.
        # alerts: the same messages as plain strings (kept for older callers).
        report: Dict[str, Any] = {"columns": {}, "alerts": [], "issues": [],
                                  "missing_columns": missing_cols, "scale": {},
                                  "guard_action": self.guard_action}
        indicators: Dict[str, pd.Series] = {}
        scale_fail: List[str] = []

        def issue(column: str, check: str, severity: str, msg: str) -> None:
            report["issues"].append({"column": column, "check": check,
                                     "severity": severity, "message": msg})
            report["alerts"].append(msg)

        for name in self.configured_columns_:
            if name not in out.columns:           # strict=False path
                continue
            p = self.params_[name]
            rule = self.rules_[name]
            s, n_coerced, n_inf = self._as_missing_aware(out[name], rule.group)
            nonnull = int((~out[name].isna()).sum())
            coerce_rate = (n_coerced / nonnull) if nonnull else 0.0
            if coerce_rate > self.max_coerce_rate:
                msg = (f"{name}: {n_coerced} unparseable value(s) ({coerce_rate:.1%}) in a "
                       f"{rule.group} column; max_coerce_rate={self.max_coerce_rate:.1%}")
                if self.guard_action == "raise":
                    raise NullContractError(msg)
                issue(name, "unparseable", "critical", msg + " — treated as NULL")
                warnings.warn(f"NullProcessor: {msg}", stacklevel=2)

            is_missing = s.isna()                  # BEFORE imputation
            rate = float(is_missing.mean()) if len(s) else 0.0
            train_rate = p["stats"]["missing_rate"]
            info = {"missing_rate": round(rate, 6), "train_missing_rate": train_rate,
                    "n_coerced": n_coerced, "n_inf": n_inf}
            if len(s) and abs(rate - train_rate) > self.drift_warn_threshold:
                msg = (f"{name}: missing rate {rate:.1%} vs train {train_rate:.1%} "
                       f"(|delta| > {self.drift_warn_threshold:.0%})")
                issue(name, "missing_rate_drift", "warn", msg)
                warnings.warn(f"NullProcessor drift: {msg}", stacklevel=2)
            report["columns"][name] = info

            if rule.group == "numerical":
                sc = self._scale_check(name, s, p["stats"])
                if sc:
                    report["scale"][name] = sc
                    if sc["level"] != "ok":
                        kind = "magnitude shifted" if sc["level"] == "fail" else "distribution shifted"
                        detail = ", ".join(
                            f"{lbl} {sc[k]['train']:.4g} -> {sc[k]['now']:.4g}"
                            for k, lbl in (("abs_nz_median", "median"), ("abs_nz_p95", "p95"))
                            if k in sc)
                        msg = (f"{name}: {kind} x{sc['unit_factor'] if sc['level'] == 'fail' else sc['worst_factor']:g} "
                               f"vs train (non-zero |x|: {detail})")
                        if sc["level"] == "fail":
                            issue(name, "scale_unit_change", "critical", msg)
                            scale_fail.append(msg)
                        else:
                            issue(name, "scale_distribution_shift", "warn", msg)
                            warnings.warn(f"NullProcessor scale: {msg}", stacklevel=2)

            if rule.add_missing_indicator:
                indicators[f"{name}{self.indicator_suffix}"] = is_missing.astype("int8")

            fill = p["imputation_value"]
            if rule.missing_strategy in ("none", "missing_indicator_only"):
                out[name] = s
            else:
                out[name] = s.fillna(fill)

        if scale_fail:
            head = (f"{len(scale_fail)} column(s) changed scale beyond x{self.scale_fail_ratio:g} "
                    f"(unit change?): " + "; ".join(scale_fail[:6]))
            if self.guard_action == "raise":
                self.report_ = report
                raise NullContractError(head)
            warnings.warn(f"NullProcessor: {head}", stacklevel=2)

        for col in self.indicator_columns_:        # stable schema even if strict=False
            base = col[: -len(self.indicator_suffix)]
            # Only reachable with strict=False and the source column absent.
            out[col] = indicators.get(col, pd.Series(0, index=out.index, dtype="int8"))
        self.report_ = report
        return out

    # ---- sklearn plumbing --------------------------------------------------
    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        self._check_fitted()
        return np.array(self.columns_in_ + self.indicator_columns_, dtype=object)

    # ---- serialization -----------------------------------------------------
    def train_statistics(self) -> pd.DataFrame:
        """One row per feature: dtype, missing_rate, zero_rate, min, max, mean,
        median, p25, p75, p95, unique_count, imputation_value."""
        self._check_fitted()
        rows = []
        for name, p in self.params_.items():
            st = p["stats"]
            rows.append({"feature": name, **{k: st.get(k) for k in (
                "dtype", "missing_rate", "zero_rate", "min", "max", "mean", "median",
                "p25", "p75", "p95", "unique_count")},
                "missing_strategy": p["rule"]["missing_strategy"],
                "imputation_value": p["imputation_value"]})
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        self._check_fitted()
        body = {
            "artifact_version": ARTIFACT_VERSION,
            "settings": {"on_unconfigured": self.on_unconfigured,
                         "indicator_suffix": self.indicator_suffix,
                         "treat_inf_as_missing": self.treat_inf_as_missing,
                         "max_coerce_rate": self.max_coerce_rate,
                         "drift_warn_threshold": self.drift_warn_threshold,
                         "strict": self.strict,
                         "scale_warn_ratio": self.scale_warn_ratio,
                         "scale_fail_ratio": self.scale_fail_ratio,
                         "min_rows_for_distribution": self.min_rows_for_distribution,
                         "max_indicators": self.max_indicators,
                         "guard_action": self.guard_action},
            "columns_in": self.columns_in_,
            "configured_columns": self.configured_columns_,
            "passthrough_columns": self.passthrough_columns_,
            "indicator_columns": self.indicator_columns_,
            "features": self.params_,
        }
        return body

    @staticmethod
    def _digest(body: Dict[str, Any]) -> str:
        canon = json.dumps(body, sort_keys=True, ensure_ascii=False, default=_json_default)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def save(self, path: Any) -> str:
        """Write the artifact as JSON. Returns its SHA-256."""
        body = self.to_dict()
        digest = self._digest(body)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({**body, "content_sha256": digest}, indent=2, ensure_ascii=False,
                       default=_json_default), encoding="utf-8")
        return digest

    @classmethod
    def load(cls, path: Any, expected_sha256: Optional[str] = None) -> "NullProcessor":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        stored = raw.pop("content_sha256", None)
        actual = cls._digest(raw)
        if stored != actual:
            raise NullContractError(f"{path}: content hash mismatch (artifact edited or corrupt)")
        if expected_sha256 and expected_sha256 != actual:
            raise NullContractError(f"{path}: sha256 {actual[:12]}… != expected {expected_sha256[:12]}…")
        if raw.get("artifact_version") != ARTIFACT_VERSION:
            raise NullContractError(f"unsupported artifact_version {raw.get('artifact_version')}")
        obj = cls(config={}, **raw["settings"])
        obj.params_ = raw["features"]
        obj.rules_ = {n: FeatureRule(**p["rule"]) for n, p in obj.params_.items()}
        obj.columns_in_ = raw["columns_in"]
        obj.configured_columns_ = raw["configured_columns"]
        obj.passthrough_columns_ = raw["passthrough_columns"]
        obj.indicator_columns_ = raw["indicator_columns"]
        obj.lint_, obj.report_ = [], {}
        return obj


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")
