import re
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Tuple, List, Optional
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif, f_regression
from Agents.BaseAgent.base_agent import BaseAgent, ToolRegistry
from logger import AgentLogger
from config import Config
from preprocessing.safe_expr import safe_eval, UnsafeExpressionError


# Whitelisted builtins for interaction-expression eval. An empty __builtins__
# strips EVERYTHING, so common type casts (e.g. `.astype(int)`, missing-flags
# like `EXT_SOURCE_1.isna().astype(int)`) raise "name 'int' is not defined".
# We expose only value-level constructors / numeric helpers — no __import__,
# open, eval, exec, getattr, etc. — so the sandbox stays as tight as before
# (attribute-chain gadgets were already reachable via df/np regardless).
_SAFE_EVAL_BUILTINS: Dict[str, Any] = {
    "int": int, "float": float, "bool": bool, "str": str, "object": object,
    "abs": abs, "min": min, "max": max, "round": round, "len": len,
}


def _as_encoder_tokens(s: pd.Series) -> pd.Series:
    """Stringify a column for label encoding, mapping every null to __NA__.

    `s.astype(str)` alone renders a null as whatever the current dtype happens
    to spell it: "nan" for a column read from CSV, "None" once the same column
    has been through a parquet round-trip. The pipeline writes parquet between
    agents, so fitting on the post-parquet frame and replaying straight from
    the source CSV silently produced two different encodings of the same
    missing value — and therefore two different models.

    Normalising first makes the encoding depend on the data, not on how the
    data happened to be serialised. It also makes the __NA__ sentinel that
    `_tool_encode_categorical` adds to every fit actually reachable; before
    this it was dead, because nulls had already become an ordinary category.
    """
    return s.astype(str).where(s.notna(), "__NA__")


class FeatureContractError(ValueError):
    """Scoring data cannot be transformed faithfully with the frozen FeatureSpec."""


@dataclass
class FeatureSpec:
    """Captured transforms from TRAIN, replayed on VALID/OOT.

    - interactions: list of (new_col, expression, fill_value) where fill_value
      is the train-median used to fill NaN/inf so valid/oot get a deterministic
      imputation, not their own median (leakage-free).
    - label_encoders: fitted LabelEncoder per column. The fit includes an
      `__NA__` sentinel so unseen categories in valid/oot map to it cleanly.
    - onehot_columns: per-col list of dummy column names produced on train,
      used to reindex valid/oot dummies (extra cols dropped, missing cols
      back-filled with 0).
    - woe_maps: per-column WoE replacement table captured at fit. Each entry
      has {edges, woe}: edges has n+1 floats fed to np.searchsorted[1:-1],
      woe has n+1 floats where woe[k] is the replacement for bin k and
      woe[-1] is the dedicated NaN-bin WoE. Applied IN PLACE — the original
      raw values are replaced by the log-odds WoE values. This is the
      scorecard-standard transformation: linearises feature ↔ target and
      smooths out non-linearity for downstream linear baselines.
    - selected_features: final column subset chosen by select_top_features
      (target appended automatically at apply time).
    """
    interactions:      List[Tuple[str, str, float]] = field(default_factory=list)
    label_encoders:    Dict[str, Any]               = field(default_factory=dict)
    onehot_columns:    Dict[str, List[str]]         = field(default_factory=dict)
    woe_maps:          Dict[str, Dict[str, Any]]    = field(default_factory=dict)
    selected_features: Optional[List[str]]          = None
    target_column:     Optional[str]                = None
    # Order in which the fit executed the step KINDS (first occurrence):
    # "encode" | "interaction" | "woe". apply() replays in this order, because an
    # interaction written on an already-encoded column must see the encoded
    # values at replay too. Default = legacy order of specs pickled before this field.
    step_order:        List[str]                    = field(
        default_factory=lambda: ["interaction", "encode", "woe"])

    # ---- individual replay steps (each reports into `diag`) ----------------
    def _step_interactions(self, df, diag, logger, name):
        for new_col, expression, fill_value in self.interactions:
            try:
                df[new_col] = safe_eval(expression, df)
                df[new_col] = df[new_col].replace([np.inf, -np.inf], np.nan)
                df[new_col] = df[new_col].fillna(fill_value)
            except UnsafeExpressionError:
                raise                      # a tampered / invalid spec must never degrade to a median
            except Exception as e:
                diag["interaction_fallbacks"].append(
                    {"feature": new_col, "error": f"{type(e).__name__}: {e}"})
                if logger is not None:
                    logger.log(name, "Transform WARN",
                        f"interaction '{new_col}' failed on transform set: {e} — filling with train median")
                df[new_col] = fill_value
        return df

    def _step_encode(self, df, diag):
        # Label encoders: unseen categories map to the __NA__ sentinel (vectorised).
        for col, le in self.label_encoders.items():
            if col not in df.columns:
                diag["missing_columns"].append(col)
                continue
            known = set(le.classes_)
            vals = _as_encoder_tokens(df[col])
            unseen = ~vals.isin(known)
            if len(vals):
                diag["unseen_rate"][col] = round(float(unseen.mean()), 6)
            vals = vals.where(~unseen, "__NA__")
            df[col] = le.transform(vals)

        # One-hot: batch-independent. Compare against TRAIN's dummy names only;
        # get_dummies(drop_first=True) on the scoring batch drops the batch's own
        # first level, which is wrong for any batch lacking train's baseline.
        for col, dummy_cols in self.onehot_columns.items():
            if col not in df.columns:
                diag["missing_columns"].append(col)
                continue
            tokens = df[col].astype(str)
            prefix = f"{col}_"
            dummies = pd.DataFrame({d: (tokens == d[len(prefix):]) for d in dummy_cols},
                                   index=df.index)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        return df

    def _step_woe(self, df, diag, on_bins):
        for col, m in self.woe_maps.items():
            if col not in df.columns:
                diag["missing_columns"].append(col)
                continue
            edges = np.asarray(m["edges"], dtype=np.float64)
            woe   = np.asarray(m["woe"],   dtype=np.float64)
            if len(woe) == 0 or len(edges) < 2:
                continue
            nan_bin_idx = len(woe) - 1
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            nan_mask = np.isnan(x)
            # Inner edges only; NaN routes to the reserved last bin
            bin_idx = np.searchsorted(edges[1:-1], x, side="right").astype(np.int64)
            bin_idx = np.where(nan_mask, nan_bin_idx, bin_idx)
            bin_idx = np.clip(bin_idx, 0, nan_bin_idx)
            if on_bins is not None:
                on_bins(col, bin_idx)
            df[col] = woe[bin_idx]
        return df

    def apply(
        self,
        df: pd.DataFrame,
        logger=None,
        name: str = "",
        on_bins: Optional[Callable[[str, np.ndarray], None]] = None,
        strict: bool = False,
        max_unseen_rate: Optional[float] = None,
    ) -> pd.DataFrame:
        """Replay every captured transform on `df`, in the order the fit ran them.

        Nothing is silent: fallbacks, absent columns and unseen-category rates are
        collected in `self.last_diagnostics` and always raised as warnings (even
        without a logger).

        strict=True (scoring / replay) raises FeatureContractError when an
        interaction fell back to its train median, a configured column is absent,
        or an encoder's unseen rate exceeds `max_unseen_rate`.

        on_bins: optional callback on_bins(col, bin_idx) invoked during WoE with
        the train-derived bin assignment (used by the IV-stability check); it must
        consume the array immediately.
        """
        diag: Dict[str, Any] = {"interaction_fallbacks": [], "missing_columns": [],
                                "unseen_rate": {}}
        steps = {"interaction": lambda d: self._step_interactions(d, diag, logger, name),
                 "encode":      lambda d: self._step_encode(d, diag),
                 "woe":         lambda d: self._step_woe(d, diag, on_bins)}
        order = [k for k in getattr(self, "step_order", ["interaction", "encode", "woe"]) if k in steps]  # old pickles lack the field
        order += [k for k in steps if k not in order]
        for kind in order:
            df = steps[kind](df)

        # Column selection: keep only the train-chosen features (+ target)
        if self.selected_features is not None:
            keep = [c for c in self.selected_features if c in df.columns]
            diag["missing_columns"] += [c for c in self.selected_features
                                        if c not in df.columns and c != self.target_column]
            if self.target_column and self.target_column in df.columns and self.target_column not in keep:
                keep.append(self.target_column)
            df = df[keep]

        diag["missing_columns"] = sorted(set(diag["missing_columns"]))
        self.last_diagnostics = diag
        problems: List[str] = []
        if diag["interaction_fallbacks"]:
            problems.append("interaction fell back to train median: " +
                            ", ".join(f["feature"] for f in diag["interaction_fallbacks"][:8]))
        if diag["missing_columns"]:
            problems.append(f"{len(diag['missing_columns'])} configured column(s) absent: "
                            f"{diag['missing_columns'][:8]}")
        if max_unseen_rate is not None:
            hot = {c: r for c, r in diag["unseen_rate"].items() if r > max_unseen_rate}
            if hot:
                problems.append(f"unseen-category rate above {max_unseen_rate:.0%}: {hot}")
        if problems:
            msg = "FeatureSpec.apply: " + " | ".join(problems)
            if strict:
                raise FeatureContractError(msg)
            warnings.warn(msg, stacklevel=2)
        return df

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interactions": [[n, e, f] for (n, e, f) in self.interactions],
            "label_encoders": sorted(self.label_encoders.keys()),
            "onehot_columns": {c: list(v) for c, v in self.onehot_columns.items()},
            "woe_maps": {
                c: {"n_bins": len(m["woe"]) - 1,
                    "iv":     round(float(m.get("iv", 0.0)), 4),
                    "edges_min": float(min(e for e in m["edges"] if np.isfinite(e))) if any(np.isfinite(e) for e in m["edges"]) else None,
                    "edges_max": float(max(e for e in m["edges"] if np.isfinite(e))) if any(np.isfinite(e) for e in m["edges"]) else None}
                for c, m in self.woe_maps.items()
            },
            "selected_features": self.selected_features,
            "target_column": self.target_column,
        }


class FeatureEngineerAgent(BaseAgent):

    _DOMAIN_GUIDANCE: Dict[str, str] = {
        "credit_risk": """\
DOMAIN: Credit Risk ({model_type})
Target: Binary default flag — 1 = defaulted (DPD ≥ threshold) within observation window (e.g. MOB6/MOB12)
Key interaction patterns:
- Debt burden        : credit_balance / (income + 1), installment_amount / (income + 1)
- Credit utilization : credit_drawn / (credit_limit + 1)
- Repayment quality  : overdue_amount / (total_due + 1), overdue_days / (loan_term + 1)
- Bureau delinquency : bureau_dpd_count / (bureau_total_loans + 1)
- Behavioral trend   : compare recent (MOB1-3) vs historical aggregates (MOB1-12)
Avoid leakage: do not use post-origination data or outcomes after the observation cutoff.""",

        "propensity": """\
DOMAIN: Propensity Model ({model_type})
Target: Binary flag — 1 = positive outcome (purchased / retained / responded / interested)
Key interaction patterns:
- RFM               : days_since_last_action, event_count_30d / (event_count_90d + 1)
- Engagement rate   : interactions / (total_contacts + 1), clicks / (impressions + 1)
- Product affinity  : category_count / (total_purchase_count + 1)
- Value             : avg_order_value, cumulative_spend / (tenure_days + 1)
Avoid: features derived from the future or from post-event behavior not available at scoring time.""",

        "fraud": """\
DOMAIN: Fraud Detection ({model_type})
Target: Binary flag — 1 = fraudulent (extreme class imbalance expected: 0.1–5%)
Key interaction patterns:
- Velocity          : txn_count_1h / (txn_count_24h + 1), amount_1h / (amount_30d + 1)
- Amount anomaly    : current_amount / (entity_avg_amount + 1)
- Network signals   : shared_device_count, shared_phone_count across entities
- Time patterns     : hour_of_day, days_since_account_open, weekend_flag * amount
- Risk ratio        : high_value_txn_count / (total_txn_count + 1)
Note: extreme imbalance — prefer interactions that amplify differences between fraud and non-fraud behavior.""",

        "generic": """\
DOMAIN: General ML ({model_type})
Apply standard feature engineering: ratio features between semantically related columns, \
frequency encoding for high-cardinality categoricals, interaction terms between correlated predictors.""",
    }

    # Product-line guidance — orthogonal to DOMAIN. The DOMAIN dict tells the
    # LLM *what the target means* (default / propensity / fraud); this dict
    # tells it *which behavioural family to emphasise* given the lending
    # product. Used to allocate 40-50% of the feature budget per the prompt.
    _PRODUCT_GUIDANCE: Dict[str, str] = {
        "consumer_unsecured": """\
► CONSUMER_UNSECURED  (vay tiêu dùng tín chấp)
  Primary drivers — income stability, bureau distress, application channel.
  Must-have feature families:
    - DTI variants: debt / income, annuity / income, total_dpd / income
    - Bureau enquiry recency: days_since_last_enquiry, count_enquiries_3m
    - Employment stability: days_employed / days_birth (career-stage ratio)
    - External bureau scores: ext_source_1 * ext_source_2, mean across
      ext_source_1..3, missing-flag per ext_source
    - Income vs peer (median by occupation_type / region)""",

        "credit_card": """\
► CREDIT_CARD  (thẻ tín dụng)
  Primary drivers — utilisation, payment behaviour, cycle stress.
  Must-have feature families:
    - Current utilisation: amt_balance / (amt_credit_limit + 1)
    - Peak utilisation: max across last N monthly utilisation columns
    - Min-payment ratio: amt_payment_current / (amt_inst_min_regularity + 1)
      (paying only the minimum repeatedly = stress signal)
    - Cash-advance share: amt_drawings_atm / (amt_drawings_total + 1)
    - Utilisation volatility: std of monthly utilisation
    - Delta utilisation (current - 3m mean) — sudden spend spikes
    - Months-on-book × utilisation interaction""",

        "mortgage": """\
► MORTGAGE / AUTO  (secured loans)
  Primary drivers — LTV, LTI, collateral stability, term burden.
  Must-have feature families:
    - LTV: amt_credit / (amt_goods_price + 1)
    - LTI: amt_credit / (amt_income_total + 1)
    - Down-payment ratio: 1 - LTV
    - Term burden: amt_annuity * cnt_payment / (amt_income_total + 1)
    - For auto: own_car_age × amt_credit (depreciation-adjusted exposure)""",

        "auto": """\
► AUTO  (secured by vehicle)
  Primary drivers — LTV on depreciating collateral, term burden, owner age.
  Must-have feature families:
    - LTV: amt_credit / (amt_goods_price + 1)
    - Depreciation-adjusted exposure: own_car_age * amt_credit
    - Term burden: amt_annuity * cnt_payment / (amt_income_total + 1)
    - Owner age × car age interaction""",

        "overdraft": """\
► OVERDRAFT / BNPL  (revolving short-term)
  Primary drivers — transactional velocity, balance volatility, repayment cadence.
  Must-have feature families:
    - Days_negative / (days_active + 1) ratio
    - Average overdraft depth: mean negative balance
    - Repayment cadence: median days between drawdown and repayment
    - Tenure-adjusted exposure: amt_credit / (months_on_book + 1)""",

        "bnpl": """\
► BNPL  (Buy Now Pay Later)
  Primary drivers — transactional velocity, basket size, repayment cadence.
  Must-have feature families:
    - Average basket size + std
    - Days between transactions (velocity)
    - Repayment cadence vs schedule
    - Tenure-adjusted exposure""",

        "sme": """\
► SME  (small / medium enterprise lending)
  Primary drivers — turnover stability, sector risk, owner credit history.
  Must-have feature families:
    - Revenue CAGR / volatility (std / mean of monthly turnover)
    - Account turnover / declared revenue (sanity-check ratio)
    - Cross-link owner application features (DTI, bureau) to entity
    - Sector benchmark deviation""",

        "generic": """\
► GENERIC  (no product-specific guidance)
  Apply universal Family A–H interactions guided by COLUMN DESCRIPTIONS.
  Same-group ratios + cross-group ratios should fill 40-50% of the budget.""",
    }

    def __init__(
        self,
        logger: AgentLogger,
        col_descriptions_path: Optional[str] = None,
        col_name_field: str = "column_name",
        col_desc_field: str = "description",
        col_group_field: Optional[str] = None,
        domain: str = "generic",
        model_type: str = "binary_classification",
        product_type: str = "generic",
    ):
        super().__init__(name="FeatureEngineer", role="Feature Architect", logger=logger)
        self.df: pd.DataFrame = None
        self.target_column: str = None
        self._protected_cols: set = set()
        self._date_col: Optional[str] = None
        self.domain = domain.lower()
        self.model_type = model_type.lower()
        self.product_type = product_type.lower()
        # Master switch for the create_interaction step (Config default; can be
        # overridden per-run via process(..., create_interactions=...)).
        self.create_interactions_enabled: bool = Config.FE_CREATE_INTERACTIONS_ENABLED
        self._col_descriptions: Dict[str, str] = self._load_col_descriptions(
            col_descriptions_path, col_name_field, col_desc_field, col_group_field
        )
        # Capture buffers populated by tool methods and harvested by _execute_llm_decisions
        # into the FeatureSpec. Initialised to None so we never read stale state.
        self._last_label_encoder: Optional[Any] = None
        self._last_onehot_cols: Optional[List[str]] = None
        self._last_selected_features: Optional[List[str]] = None
        self._last_interaction_fill: float = 0.0
        self._batch_label_encoders: Dict[str, Any] = {}
        self._batch_onehot_cols: Dict[str, List[str]] = {}
        # IV cache — populated by compute_iv tool, read by select_top_features
        # when criterion='iv'. Empty dict means "IV not computed yet" so the
        # selector falls back to f_classif / f_regression.
        self._last_iv_scores: Dict[str, float] = {}
        self._last_woe_maps: Dict[str, Dict[int, float]] = {}
        # Full WoE payload {col: {edges, woe, iv}} populated by compute_iv,
        # consumed by apply_woe_transform to do the in-place WoE replacement.
        self._last_woe_data: Dict[str, Dict[str, Any]] = {}
        # WoE transforms actually applied (subset of _last_woe_data, after
        # min_iv filtering). Copied into FeatureSpec by _execute_llm_decisions
        # so valid/oot replay uses the same bin edges + WoE values.
        self._last_applied_woe: Dict[str, Dict[str, Any]] = {}
        self.tool_registry = ToolRegistry()
        self._register_tools()

    def _apply_create_interactions_override(self, create_interactions: Optional[bool]) -> None:
        """Resolve the interaction-step switch: None keeps the Config default,
        an explicit bool overrides it for this run and logs the decision."""
        if create_interactions is not None:
            self.create_interactions_enabled = bool(create_interactions)
            self.logger.log(self.name, "Interaction step override",
                f"create_interactions={self.create_interactions_enabled} (per-run, overrides Config)")
        if not self.create_interactions_enabled:
            self.logger.log(self.name, "Interaction step",
                "DISABLED — create_interaction actions will be skipped this run")

    def _get_domain_guidance(self) -> str:
        if self.domain not in self._DOMAIN_GUIDANCE:
            self.logger.log(self.name, "WARN",
                f"Unknown domain '{self.domain}' — using generic guidance. "
                f"Valid options: {list(self._DOMAIN_GUIDANCE.keys())}")
        template = self._DOMAIN_GUIDANCE.get(self.domain, self._DOMAIN_GUIDANCE["generic"])
        return template.format(model_type=self.model_type)

    def _get_product_guidance(self) -> str:
        if self.product_type not in self._PRODUCT_GUIDANCE:
            self.logger.log(self.name, "WARN",
                f"Unknown product_type '{self.product_type}' — using generic guidance. "
                f"Valid options: {list(self._PRODUCT_GUIDANCE.keys())}")
        return self._PRODUCT_GUIDANCE.get(self.product_type, self._PRODUCT_GUIDANCE["generic"])

    def _load_col_descriptions(
        self,
        path: Optional[str],
        col_name_field: str = "column_name",
        col_desc_field: str = "description",
        col_group_field: Optional[str] = None,
    ) -> Dict[str, str]:
        """Load column descriptions from JSON, CSV, Parquet, or Excel.

        For tabular files (CSV/Parquet/Excel), specify which columns hold the
        column name, description, and optional group via the *_field params.

        Example — HomeCredit CSV (Table=group, Row=name, Description=desc):
            _load_col_descriptions(path, col_name_field="Row",
                                   col_desc_field="Description",
                                   col_group_field="Table")
        """
        if not path:
            return {}
        try:
            suffix = Path(path).suffix.lower()
            if suffix == ".json":
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    # Detect 3-level: {group: {col_name: desc, ...}, ...}
                    if data and isinstance(next(iter(data.values())), dict):
                        return self._parse_nested_json_descriptions(data, col_group_field)
                    # Flat 2-level: {col_name: desc, ...}
                    return {str(k): str(v) for k, v in data.items()}
                if isinstance(data, list):
                    # List of dicts — treat as tabular rows
                    return self._parse_tabular_descriptions(
                        pd.DataFrame(data), col_name_field, col_desc_field, col_group_field
                    )
                self.logger.log(self.name, "WARN",
                    f"JSON col_descriptions: expected dict or list, got {type(data).__name__}")
                return {}
            elif suffix == ".csv":
                for _enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
                    try:
                        df_desc = pd.read_csv(path, encoding=_enc)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise UnicodeDecodeError("csv", b"", 0, 1,
                        f"Could not decode '{path}' with utf-8 / cp1252 / latin-1")
            elif suffix in (".xls", ".xlsx", ".xlsm"):
                # _read_excel_smart sniffs file magic so a mislabelled .xls /
                # .xlsx pair is still readable. Surfaces a clearer error when
                # the file is encrypted/corrupt or when openpyxl/xlrd is
                # missing from the environment.
                df_desc = self._read_excel_smart(path)
            elif suffix == ".parquet":
                df_desc = pd.read_parquet(path)
            else:
                self.logger.log(self.name, "WARN",
                    f"Unsupported col_descriptions file type: '{suffix}'. "
                    "Supported: .json, .csv, .parquet, .xls, .xlsx")
                return {}
            return self._parse_tabular_descriptions(df_desc, col_name_field, col_desc_field, col_group_field)
        except Exception as e:
            self.logger.log(self.name, "WARN", f"Could not load col_descriptions from '{path}': {e}")
        return {}

    def _parse_tabular_descriptions(
        self,
        df_desc: pd.DataFrame,
        col_name_field: str,
        col_desc_field: str,
        col_group_field: Optional[str],
    ) -> Dict[str, str]:
        """Parse a tabular description file into {column_name: description} dict.

        Only col_name_field and col_desc_field are read; all other columns
        (e.g. Special, Notes) are silently ignored.
        """
        if col_name_field not in df_desc.columns:
            self.logger.log(self.name, "WARN",
                f"col_name_field '{col_name_field}' not found in description file. "
                f"Available columns: {list(df_desc.columns)}")
            return {}
        if col_desc_field not in df_desc.columns:
            self.logger.log(self.name, "WARN",
                f"col_desc_field '{col_desc_field}' not found in description file. "
                f"Available columns: {list(df_desc.columns)}")
            return {}
        has_group = bool(col_group_field and col_group_field in df_desc.columns)
        result: Dict[str, str] = {}
        for _, row in df_desc.iterrows():
            col_name = str(row[col_name_field]).strip()
            if not col_name or col_name == "nan":
                continue
            desc = str(row[col_desc_field]).strip() if pd.notna(row[col_desc_field]) else ""
            if has_group and pd.notna(row.get(col_group_field)):
                group = str(row[col_group_field]).strip()
                desc = f"[{group}] {desc}" if desc else f"[{group}]"
            result[col_name] = desc
        self.logger.log(self.name, "Col Descriptions Loaded",
            f"Loaded {len(result)} descriptions"
            + (f" | name_field='{col_name_field}' desc_field='{col_desc_field}'"
               f" group_field='{col_group_field}'" if has_group
               else f" | name_field='{col_name_field}' desc_field='{col_desc_field}'"))
        return result

    def _parse_nested_json_descriptions(
        self,
        data: dict,
        col_group_field: Optional[str],
    ) -> Dict[str, str]:
        """Parse 3-level JSON: {group: {col_name: desc, ...}, ...} → {col_name: desc}.

        If col_group_field is not None, prefixes each description with [group].
        If a column name appears in multiple groups, the last group wins.
        """
        include_group = col_group_field is not None
        result: Dict[str, str] = {}
        for group, cols in data.items():
            if not isinstance(cols, dict):
                continue
            for col_name, desc in cols.items():
                col_name = str(col_name).strip()
                desc = str(desc).strip() if desc is not None else ""
                result[col_name] = f"[{group}] {desc}" if include_group and desc else (
                    f"[{group}]" if include_group else desc
                )
        self.logger.log(self.name, "Col Descriptions Loaded",
            f"Loaded {len(result)} descriptions from nested JSON "
            f"({len(data)} groups)"
            + (" with group prefix" if include_group else ""))
        return result

    def _register_tools(self):
        self.tool_registry.register(
            "create_interaction",
            "Creates a new column using mathematical expressions between columns based on meaning of columns",
            {
                "df": "The dataframe",
                "new_col": "Name for the new column",
                "expression": "Python expression using df['col'] syntax, e.g. \"df['a'] / df['b']\"",
            },
        )
        self.tool_registry.register(
            "encode_categorical",
            "Encodes a single categorical column into numeric format",
            {
                "df": "The dataframe",
                "col": "Column name to encode",
                "method": "Either 'label' (ordinal) or 'onehot' (binary columns)",
            },
        )
        self.tool_registry.register(
            "encode_all_categorical",
            "Encodes ALL remaining object/categorical columns at once (preferred for wide datasets)",
            {
                "df": "The dataframe",
                "method": "Either 'label' (default) or 'onehot'",
            },
        )
        self.tool_registry.register(
            "correlation_analysis",
            "Analyzes correlation between features and target",
            {"df": "The dataframe", "target": "Target column name"},
        )
        self.tool_registry.register(
            "compute_iv",
            "Computes Information Value (IV) per feature against a binary target — "
            "credit-risk gold standard. Captures non-linear signal via WoE binning. "
            "Run BEFORE apply_woe_transform and select_top_features when using "
            "criterion='iv'. Also caches per-feature WoE tables so the optional "
            "apply_woe_transform step can replay them.",
            {
                "df": "The dataframe",
                "target": "Target column name (must be binary)",
                "bins": "Number of quantile bins for numeric features (default 10)",
                "method": "Binning method: 'quantile' (default) or 'uniform'",
            },
        )
        self.tool_registry.register(
            "apply_woe_transform",
            f"Replaces each value with its bin's WoE (log-odds). Banking scorecard "
            f"standard — linearises feature/target relationship and smooths "
            f"non-linearity; typically +1-3% AUC on credit data. Requires "
            f"compute_iv to have run first. Only transforms features with "
            f"IV >= min_iv (default {Config.WOE_MIN_IV}); low-IV features keep "
            f"raw values so GBM can still split on them.",
            {
                "df": "The dataframe",
                "min_iv": "Skip WoE transform for features with IV below this "
                          f"(default {Config.WOE_MIN_IV} = Siddiqi useless/weak boundary)",
            },
        )
        self.tool_registry.register(
            "select_top_features",
            f"Keeps only the k most predictive features. With criterion='iv', uses "
            f"IV scores from compute_iv and greedily prunes multicollinear pairs "
            f"(|corr| > {Config.MULTICOLLINEARITY_THRESHOLD} drops the lower-IV partner, "
            f"keeps the higher-IV one).",
            {
                "df": "The dataframe",
                "target": "Target column name",
                "k": "Number of top features to keep",
                "criterion": "'iv' (uses compute_iv cache) or 'default' (f_classif / f_regression)",
            },
        )

    def _tool_create_interaction(self, df: pd.DataFrame, new_col: str, expression: str) -> pd.DataFrame:
        # Expressions are LLM-written: validated against an AST allowlist before
        # evaluation (preprocessing.safe_expr). A builtins whitelist alone is not a sandbox.
        try:
            df[new_col] = safe_eval(expression, df)
            df[new_col] = df[new_col].replace([np.inf, -np.inf], np.nan)
            train_median = df[new_col].median()
            # Median can still be NaN if every value is NaN/inf — use 0.0 as a safe sentinel
            if pd.isna(train_median):
                train_median = 0.0
            df[new_col] = df[new_col].fillna(train_median)
            # Drop if constant or all-NaN — causes divide-by-zero in correlation/SelectKBest
            if df[new_col].isna().all() or df[new_col].std() == 0:
                df = df.drop(columns=[new_col])
                raise ValueError(f"Generated feature '{new_col}' is constant or all-NaN after fill — dropped")
            # Expose train median so _execute_llm_decisions can capture it for replay on valid/oot
            self._last_interaction_fill = float(train_median)
            return df
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Error creating interaction '{new_col}': {e}")

    def _tool_encode_categorical(self, df: pd.DataFrame, col: str, method: str) -> pd.DataFrame:
        """FIT-on-train encoding. Fitted encoders / column lists are exposed via
        `_last_label_encoder` and `_last_onehot_cols` so `_execute_llm_decisions`
        can capture them into a FeatureSpec for valid/oot replay.
        """
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")

        if method == "label":
            le = LabelEncoder()
            # Train values + __NA__ sentinel — guarantees transform never raises
            # on unseen valid/oot categories when we replay this encoder.
            train_vals = _as_encoder_tokens(df[col])
            le.fit(sorted(set(train_vals.tolist()) | {"__NA__"}))
            df[col] = le.transform(train_vals)
            self._last_label_encoder = le

        elif method == "onehot":
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            self._last_onehot_cols = list(dummies.columns)
        else:
            raise ValueError(f"Unknown encoding method: {method}")

        return df

    def _tool_encode_all_categorical(self, df: pd.DataFrame, method: str = "label") -> pd.DataFrame:
        """Encode every non-numeric column (except protected/target) using `method`.

        For each column, the fitted encoder is stored in `_batch_label_encoders` /
        `_batch_onehot_cols` so `_execute_llm_decisions` can transfer them
        into the FeatureSpec.
        """
        skip = self._protected_cols | {self.target_column}
        cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c not in skip]
        self._batch_label_encoders: Dict[str, Any] = {}
        self._batch_onehot_cols: Dict[str, List[str]] = {}
        encoded = 0
        for col in cat_cols:
            effective_method = method
            if method == "onehot":
                col_nu = df[col].nunique()
                if col_nu > 5:
                    effective_method = "label"
                    self.logger.log(self.name, "encode_all_categorical",
                        f"'{col}' nunique={col_nu} > 5 — falling back to label encoding")
            self._last_label_encoder = None
            self._last_onehot_cols = None
            df = self._tool_encode_categorical(df, col, effective_method)
            if effective_method == "label" and self._last_label_encoder is not None:
                self._batch_label_encoders[col] = self._last_label_encoder
            elif effective_method == "onehot" and self._last_onehot_cols is not None:
                self._batch_onehot_cols[col] = self._last_onehot_cols
            encoded += 1
        self.logger.log(self.name, "encode_all_categorical",
            f"Encoded {encoded} categorical columns (requested method='{method}')")
        return df

    def _tool_correlation_analysis(self, df: pd.DataFrame, target: str) -> str:
        """Correlation of each feature with the target. `df` is the TRAIN frame
        in split mode (caller passes self.df, which IS train), so the score is
        leakage-free.
        """
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        y = pd.to_numeric(df[target], errors="coerce")
        if y.isna().all():
            raise ValueError(
                f"Target '{target}' cannot be converted to numeric — correlation analysis skipped"
            )

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)

        correlations = {}
        unreliable = []
        for col in numeric_cols:
            null_pct = df[col].isnull().mean() * 100
            if df[col].std() == 0:
                correlations[col] = 0.0
                continue
            corr = df[col].corr(y)
            correlations[col] = round(corr, 4) if not pd.isna(corr) else 0.0
            if null_pct > 30:
                unreliable.append(f"{col} ({null_pct:.1f}% null)")

        sorted_corrs = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))

        result = {
            "correlations": sorted_corrs,
            "high_correlation": [k for k, v in sorted_corrs.items() if abs(v) > Config.HIGH_CORRELATION_THRESHOLD],
            "low_correlation": [k for k, v in sorted_corrs.items() if abs(v) < Config.LOW_CORRELATION_THRESHOLD],
            "unreliable_due_to_nulls": unreliable,
        }
        return json.dumps(result, indent=2)

    @staticmethod
    def _compute_iv_single(
        x: np.ndarray,
        y: np.ndarray,
        bins: int,
        method: str,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Vectorised IV + per-bin WoE for one numeric column.

        Returns (iv, woe_array, edges_array) where:
          - woe_array: length = n_value_bins + 1; last index is the NaN bin.
            woe_array[bin_idx] gives the WoE replacement value.
          - edges_array: length = n_value_bins + 1; passed to np.searchsorted
            on its INNER slice [1:-1] to map values → bin indices [0..n_value_bins-1].
            Stored so valid/oot replay can bin identically without re-quantizing.

        Pure numpy hot path — avoids the per-column pandas overhead that
        dominates IV compute at 1.5M × 5k scale (qcut + crosstab is ~10x slower
        than searchsorted + bincount).

        Low-nunique branch (≤ `bins` unique values): treats each unique value as
        its own bin. This catches label-encoded categoricals (CODE_GENDER,
        FLAG_*, etc.) — quantile binning collapses them to 1 bin and loses signal.

        NaN values always get the dedicated last bin so missing-pattern signal
        (FAMILY F: ext_source_1 missingness etc.) contributes to IV the same way
        a real bin would. Laplace smoothing (eps=0.5) protects against log(0).
        """
        nan_mask = np.isnan(x)
        x_valid = x[~nan_mask]

        if len(x_valid) < 2:
            return 0.0, np.array([]), np.array([])

        unique_vals = np.unique(x_valid)
        n_unique = len(unique_vals)
        # < 2 unique → no signal regardless of method
        if n_unique < 2:
            return 0.0, np.array([]), np.array([])

        # ── Edges selection ────────────────────────────────────────────────
        # Low-cardinality branch: each unique value becomes its own bin.
        # Edges = [-inf, midpoint_1, ..., midpoint_{n-1}, +inf] so searchsorted
        # routes each value to a stable bin index. Critical for encoded
        # categoricals after label encoding.
        if n_unique <= bins:
            sorted_unique = np.sort(unique_vals)
            midpoints = (sorted_unique[:-1] + sorted_unique[1:]) / 2.0
            edges = np.concatenate([[-np.inf], midpoints, [np.inf]])
        elif method == "quantile":
            edges = np.unique(np.quantile(x_valid, np.linspace(0, 1, bins + 1)))
        else:  # uniform
            edges = np.linspace(x_valid.min(), x_valid.max(), bins + 1)

        if len(edges) < 2:
            return 0.0, np.array([]), np.array([])

        # ── Bin assignment ────────────────────────────────────────────────
        # Inner edges only — searchsorted maps non-NaN values to [0, n_value_bins-1].
        # NaN values get the dedicated last bin index = n_value_bins.
        n_value_bins = len(edges) - 1
        nan_bin_idx = n_value_bins
        total_bins = n_value_bins + 1  # value bins + NaN bin (always reserved)
        bin_idx = np.searchsorted(edges[1:-1], x, side="right").astype(np.int64)
        bin_idx = np.where(nan_mask, nan_bin_idx, bin_idx)

        # ── Cross-tab via bincount ────────────────────────────────────────
        pos, neg = FeatureEngineerAgent._bin_class_counts(bin_idx, y, total_bins)
        iv, woe = FeatureEngineerAgent._iv_from_bin_counts(pos, neg)
        if not len(woe):
            return 0.0, np.array([]), np.array([])
        return iv, woe, edges

    @staticmethod
    def _bin_class_counts(
        bin_idx: np.ndarray, y: np.ndarray, total_bins: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-bin (positive, negative) counts. Cheap enough to run per partition."""
        pos = np.bincount(bin_idx, weights=y.astype(np.float64), minlength=total_bins)
        total = np.bincount(bin_idx, minlength=total_bins).astype(np.float64)
        return pos, total - pos

    @staticmethod
    def _iv_from_bin_counts(
        pos: np.ndarray, neg: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Laplace-smoothed IV + per-bin WoE from bin counts.

        Split out from `_compute_iv_single` so the IV-stability check can score a
        holdout partition with the exact same formula the train IV was computed
        with — two copies of this arithmetic would make iv_train and iv_oot
        quietly incomparable, which is the one thing that check must not do.

        Returns (0.0, empty) when a partition is single-class, matching how
        `_compute_iv_single` treats a column with no usable signal.
        """
        total_bins = len(pos)
        total_pos = pos.sum()
        total_neg = neg.sum()
        if total_bins == 0 or total_pos == 0 or total_neg == 0:
            return 0.0, np.array([])

        # eps=0.5 per bin protects log(0) when a bin is pure-class.
        eps = 0.5
        p_pos = (pos + eps) / (total_pos + eps * total_bins)
        p_neg = (neg + eps) / (total_neg + eps * total_bins)
        woe = np.log(p_neg / p_pos)
        iv = float(np.sum((p_neg - p_pos) * woe))
        return iv, woe

    def _tool_compute_iv(
        self,
        df: pd.DataFrame,
        target: str,
        bins: int = None,
        method: str = "quantile",
    ) -> str:
        """Information Value per feature (binary target only).

        Populates `self._last_iv_scores` so `select_top_features` with
        criterion='iv' can read them without recomputing. Returns a JSON
        summary with Siddiqi IV bands, top-20 features, and leakage suspects
        (IV >= IV_LEAKAGE_THRESHOLD).

        Designed for wide data (1.5M × 15k): column-by-column iteration,
        pre-filters constant / heavily-null columns, gc.collect() every 1000
        cols. Total cost ~10-15 min at that scale. Categorical columns are
        skipped (encode_all_categorical must run first per the prompt).
        """
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        if bins is None:
            bins = Config.IV_BINS_DEFAULT

        y = pd.to_numeric(df[target], errors="coerce")
        valid_mask = y.notna()
        y_valid = y[valid_mask].astype(np.int64).values
        n_unique = pd.Series(y_valid).nunique()
        if n_unique != 2:
            raise ValueError(
                f"IV requires a BINARY target — got {n_unique} unique values. "
                "Use correlation_analysis / select_top_features (default criterion) for regression."
            )

        # Iterate dtypes by metadata only — slicing df[feature_cols] would
        # trigger block consolidation and OOM on wide TRAIN data (the same
        # trap select_top_features avoids).
        dtypes = df.dtypes
        feature_cols = [
            c for c in df.columns
            if c != target
            and c not in self._protected_cols
            and pd.api.types.is_numeric_dtype(dtypes[c])
        ]
        if not feature_cols:
            self._last_iv_scores = {}
            return json.dumps({"warning": "no numeric features to score"}, indent=2)

        score_idx = df.index[valid_mask]

        iv_scores: Dict[str, float] = {}
        # Full WoE structures keyed by column. Stored as plain lists/floats so
        # they are JSON / pickle friendly when persisted into FeatureSpec.
        #   edges:  inner+outer edges (n+1 floats) — fed to np.searchsorted[1:-1]
        #   woe:    n+1 floats (last = NaN bin WoE)
        #   iv:     diagnostic for debugging / leakage audit
        woe_data: Dict[str, Dict[str, Any]] = {}
        skipped_null = skipped_const = computed = 0

        import gc as _gc
        for i, col in enumerate(feature_cols):
            s = df[col]
            # Cheap pre-filters first — avoid quantile on dead columns
            if s.isnull().mean() > Config.IV_MAX_NULL_RATIO:
                iv_scores[col] = 0.0
                skipped_null += 1
                continue
            x = s.loc[score_idx].to_numpy(dtype=np.float64, copy=False)
            if np.isnan(x).all() or np.nanstd(x) == 0:
                iv_scores[col] = 0.0
                skipped_const += 1
                continue
            try:
                iv, woe_arr, edges_arr = self._compute_iv_single(
                    x, y_valid, bins=bins, method=method
                )
                iv_scores[col] = iv
                if iv > 0 and len(woe_arr) > 0:
                    woe_data[col] = {
                        "edges": edges_arr.tolist(),
                        "woe":   woe_arr.tolist(),
                        "iv":    iv,
                    }
                computed += 1
            except Exception:
                iv_scores[col] = 0.0
            # Periodic GC + progress log — keeps peak RSS bounded on 15k-col runs
            if (i + 1) % 1000 == 0:
                _gc.collect()
                self.logger.log(self.name, "compute_iv progress",
                    f"{i + 1}/{len(feature_cols)} cols scored")

        self._last_iv_scores = iv_scores
        # Stored as full edges+woe payload (used by apply_woe_transform). Backwards-
        # compat shim: _last_woe_maps mirrors just the per-bin woe values so any
        # old caller that only looked up WoE per bin still works.
        self._last_woe_data = woe_data
        self._last_woe_maps = {c: {i: w for i, w in enumerate(d["woe"])}
                               for c, d in woe_data.items()}

        sorted_iv = sorted(iv_scores.items(), key=lambda kv: kv[1], reverse=True)
        leakage_threshold = Config.IV_LEAKAGE_THRESHOLD
        bands = {
            "useless_(<0.02)":      sum(1 for _, v in sorted_iv if v < 0.02),
            "weak_(0.02-0.10)":     sum(1 for _, v in sorted_iv if 0.02 <= v < 0.10),
            "medium_(0.10-0.30)":   sum(1 for _, v in sorted_iv if 0.10 <= v < 0.30),
            "strong_(0.30-0.50)":   sum(1 for _, v in sorted_iv if 0.30 <= v < leakage_threshold),
            f"leakage_suspect_(>={leakage_threshold})":
                                    sum(1 for _, v in sorted_iv if v >= leakage_threshold),
        }
        result = {
            "n_features_scored": computed,
            "n_skipped_high_null": skipped_null,
            "n_skipped_constant": skipped_const,
            "bins": bins,
            "method": method,
            "iv_bands_siddiqi": bands,
            "top_20_iv": dict(sorted_iv[:20]),
            "leakage_suspects": [c for c, v in sorted_iv if v >= leakage_threshold],
        }
        self.logger.log(self.name, "compute_iv done",
            f"scored={computed} skip_null={skipped_null} skip_const={skipped_const} "
            f"bands={bands}")
        return json.dumps(result, indent=2)

    def _tool_apply_woe_transform(
        self,
        df: pd.DataFrame,
        min_iv: float = None,
    ) -> pd.DataFrame:
        """Replace each value in qualifying columns with its bin's WoE.

        Requires compute_iv to have run first (uses cached edges + WoE per
        column from `self._last_woe_data`). Columns with IV < min_iv are
        skipped (Siddiqi 'useless' band — WoE there is noise more than signal).

        After this step:
          - Feature columns are in log-odds scale (typically [-3, +3]).
          - Feature ↔ target relationship is linearised — boosts LR baseline,
            smooths GBM split thresholds at tail bins, +1-3% AUC on credit data.
          - NaN is replaced by the dedicated NaN-bin WoE (missing-pattern signal
            is preserved as a real numeric value, no separate flag needed).

        Updates `self._last_applied_woe` so `_execute_llm_decisions` can copy
        the per-column edges+WoE into FeatureSpec for valid/oot replay.
        Idempotent if called twice — second call sees no IV cache mismatch.
        """
        if min_iv is None:
            min_iv = Config.WOE_MIN_IV
        woe_data = getattr(self, "_last_woe_data", None) or {}
        if not woe_data:
            raise ValueError(
                "apply_woe_transform requires compute_iv to have run first — "
                "no cached WoE data found. Run compute_iv before apply_woe_transform."
            )

        applied: Dict[str, Dict[str, Any]] = {}
        skipped_low_iv = skipped_missing = transformed = 0
        for col, m in woe_data.items():
            if col not in df.columns:
                skipped_missing += 1
                continue
            iv = float(m.get("iv", 0.0))
            if iv < min_iv:
                skipped_low_iv += 1
                continue
            edges = np.asarray(m["edges"], dtype=np.float64)
            woe   = np.asarray(m["woe"],   dtype=np.float64)
            if len(woe) == 0 or len(edges) < 2:
                skipped_low_iv += 1
                continue
            nan_bin_idx = len(woe) - 1
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            nan_mask = np.isnan(x)
            bin_idx = np.searchsorted(edges[1:-1], x, side="right").astype(np.int64)
            bin_idx = np.where(nan_mask, nan_bin_idx, bin_idx)
            bin_idx = np.clip(bin_idx, 0, nan_bin_idx)
            df[col] = woe[bin_idx]
            applied[col] = m  # store the same edges+woe payload for replay
            transformed += 1

        self._last_applied_woe = applied
        self.logger.log(self.name, "apply_woe_transform",
            f"transformed={transformed} | skipped_low_iv(<{min_iv})={skipped_low_iv} | "
            f"skipped_missing_col={skipped_missing}")
        return df

    def _tool_select_top_features(
        self,
        df: pd.DataFrame,
        target: str,
        k: int,
        criterion: str = "default",
    ) -> pd.DataFrame:
        """Score features on the TRAIN frame (caller passes self.df=train) and
        keep the top-k. The retained column list is exposed via
        `_last_selected_features` so it can be captured into FeatureSpec and
        applied to valid/oot during transform.

        criterion:
          'iv'      — use Information Value cached by compute_iv. Also runs
                      a greedy multicollinearity prune (|corr| > MULTICOLLINEARITY_THRESHOLD
                      drops the lower-IV partner). Falls back to 'default' if
                      no IV scores are cached (compute_iv was not called).
          'default' — f_classif (binary) / f_regression. No multicollinearity prune.
        """
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        feature_cols = [c for c in df.columns if c != target]
        # Iterate dtypes by metadata only — slicing df[feature_cols] forces pandas to
        # consolidate matching blocks into a single (n_cols, n_rows) float64 matrix,
        # which OOMs on wide TRAIN data (e.g. 1335 cols × 493k rows ≈ 5 GB).
        dtypes = df.dtypes
        numeric_features  = [c for c in feature_cols if pd.api.types.is_numeric_dtype(dtypes[c])]
        non_numeric_cols  = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(dtypes[c])]

        if not numeric_features:
            return df

        y = pd.to_numeric(df[target], errors="coerce")
        valid_mask = y.notna()
        y_valid = y[valid_mask].values
        if len(y_valid) < 2 or pd.Series(y_valid).nunique() < 2:
            # Not enough training signal — return df unchanged
            return df

        score_idx = df.index[valid_mask]
        use_iv = (criterion == "iv") and bool(self._last_iv_scores)
        if criterion == "iv" and not self._last_iv_scores:
            self.logger.log(self.name, "select_top_features WARN",
                "criterion='iv' requested but no IV scores cached — "
                "did compute_iv run? Falling back to f_classif / f_regression.")

        if use_iv:
            # Only score features that have a cached IV (skipped cols got 0.0 from compute_iv)
            scores: Dict[str, float] = {
                c: self._last_iv_scores.get(c, 0.0) for c in numeric_features
            }
        else:
            is_classification = pd.Series(y_valid).nunique() < Config.CLASSIFICATION_UNIQUE_THRESHOLD
            score_func = f_classif if is_classification else f_regression

            scores: Dict[str, float] = {}
            for col in numeric_features:
                x = df.loc[score_idx, col]
                med = x.median()
                if pd.isna(med):
                    scores[col] = 0.0
                    continue
                x = x.fillna(med).values
                if x.std() == 0:
                    scores[col] = 0.0
                    continue
                try:
                    f_stat, _ = score_func(x.reshape(-1, 1), y_valid)
                    scores[col] = float(f_stat[0]) if not np.isnan(f_stat[0]) else 0.0
                except Exception:
                    scores[col] = 0.0

        eligible = [c for c, s in scores.items() if s > 0]
        k = min(k, len(eligible) if eligible else len(scores))
        sorted_features = sorted(scores, key=scores.__getitem__, reverse=True)

        if use_iv:
            # Greedy multicollinearity prune — walk down the IV-sorted list,
            # accept a candidate only if it's not strongly correlated with any
            # already-kept feature (|corr| > MULTICOLLINEARITY_THRESHOLD).
            # Cost is bounded at O(k * kept) pair-corrs; we extract one column
            # at a time so we never materialise a k×n_rows matrix.
            mc_threshold = Config.MULTICOLLINEARITY_THRESHOLD
            kept_series: Dict[str, np.ndarray] = {}  # cache numpy arrays for kept cols
            selected_numeric: List[str] = []
            dropped_collinear: List[Tuple[str, str, float]] = []
            for col in sorted_features:
                if len(selected_numeric) >= k:
                    break
                if scores.get(col, 0.0) <= 0:
                    continue
                x_cand = df.loc[score_idx, col].to_numpy(dtype=np.float64, copy=False)
                # Replace NaN with column mean for the corr coefficient — same as np.corrcoef
                # treats it but vectorised so 200²=40k pair-corrs stay manageable.
                if np.isnan(x_cand).any():
                    m = np.nanmean(x_cand)
                    x_cand = np.where(np.isnan(x_cand), m, x_cand)
                drop = False
                drop_partner: Optional[str] = None
                drop_corr: float = 0.0
                for kept_col, x_kept in kept_series.items():
                    # np.corrcoef returns 2x2; we want the off-diagonal
                    if x_cand.std() == 0 or x_kept.std() == 0:
                        continue
                    c = float(np.corrcoef(x_cand, x_kept)[0, 1])
                    if not np.isnan(c) and abs(c) > mc_threshold:
                        drop = True
                        drop_partner = kept_col
                        drop_corr = c
                        break
                if drop:
                    dropped_collinear.append((col, drop_partner, drop_corr))
                    continue
                kept_series[col] = x_cand
                selected_numeric.append(col)
            if dropped_collinear:
                self.logger.log(self.name, "select_top_features prune",
                    f"Dropped {len(dropped_collinear)} multicollinear features "
                    f"(|corr| > {mc_threshold}) — sample: "
                    f"{dropped_collinear[:5]}")
        else:
            selected_numeric = sorted_features[:k]
        final_cols = selected_numeric + non_numeric_cols + [target]
        # Exposed for capture into FeatureSpec
        self._last_selected_features = list(final_cols)
        # Drop unwanted cols in-place — slicing df[final_cols] would trigger block
        # consolidation on the wide TRAIN frame and spike memory by gigabytes just
        # to reorder columns. Pandas keeps original column order after drop, which
        # is fine downstream since access is by name, not position.
        keep = set(final_cols)
        to_drop = [c for c in df.columns if c not in keep]
        if to_drop:
            df.drop(columns=to_drop, inplace=True)
        return df

    # ── IV stability (train ↔ holdout) ────────────────────────────────────

    def _make_iv_stability_collector(
        self, y: np.ndarray, spec: "FeatureSpec"
    ) -> Tuple[Callable[[str, np.ndarray], None], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """Build the `on_bins` callback that accumulates per-bin class counts.

        The callback folds each column's bin assignment straight into a
        (n_bins+1,) count pair and drops the array, so peak memory is a few
        hundred floats per column rather than one int64 array per column.

        Scope is `selected_features ∩ woe_maps`: a feature only has frozen train
        bins if it was WoE-transformed, and only matters if it survived
        selection.
        """
        scope = set(spec.woe_maps)
        if spec.selected_features is not None:
            scope &= set(spec.selected_features)
        counts: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        def on_bins(col: str, bin_idx: np.ndarray) -> None:
            if col not in scope:
                return
            m = spec.woe_maps.get(col) or {}
            n_bins = len(m.get("woe", ()))
            if n_bins == 0 or len(bin_idx) != len(y):
                return
            counts[col] = self._bin_class_counts(bin_idx, y, n_bins)

        return on_bins, counts

    def _evaluate_iv_stability(
        self,
        counts: Dict[str, Tuple[np.ndarray, np.ndarray]],
        spec: "FeatureSpec",
        tag: str,
    ) -> Dict[str, Any]:
        """Compare holdout IV against train IV under FROZEN train binning.

        Answers the question PSI cannot: has the feature's *relationship to the
        target* held up out of time? A feature can have a perfectly stable
        X-distribution (PSI ~ 0) while its bin-level log-odds weaken or invert,
        and nothing else in this pipeline would notice.

        Three signals per feature:
          iv_drop_pct    — relative IV loss vs train.
          woe_corr       — Pearson between the train WoE vector and the holdout
                           WoE vector over populated bins. Negative = the
                           relationship reversed.
          woe_sign_flips — bins whose WoE changed sign. Bins thinner than
                           FE_IV_STABILITY_MIN_BIN_COUNT are excluded; tail bins
                           are too noisy to read anything into.

        Diagnostic only — nothing is dropped. Agent 3's PSI / SHAP+PSI prune
        stay the only places features are removed.
        """
        min_cnt   = Config.FE_IV_STABILITY_MIN_BIN_COUNT
        max_drop  = Config.FE_IV_STABILITY_MAX_DROP
        min_corr  = Config.FE_IV_STABILITY_MIN_WOE_CORR
        iv_key    = f"iv_{tag}"

        features: Dict[str, Dict[str, Any]] = {}
        n_skipped_weak = 0
        for col, (pos, neg) in counts.items():
            iv_train = float(self._last_iv_scores.get(col, 0.0))
            if iv_train < Config.WOE_MIN_IV:
                # Nothing to be stable about. A feature that was already in the
                # Siddiqi "useless" band on train would show wild iv_drop_pct
                # swings off a near-zero baseline and flag as UNSTABLE every
                # run — noise, not a finding.
                n_skipped_weak += 1
                continue
            iv_part, woe_part = self._iv_from_bin_counts(pos, neg)
            woe_train = np.asarray(spec.woe_maps[col]["woe"], dtype=np.float64)
            if len(woe_part) != len(woe_train):
                continue

            # Only bins with enough holdout rows are trustworthy enough to compare
            populated = (pos + neg) >= min_cnt
            n_pop = int(populated.sum())
            if n_pop >= 2:
                a, b = woe_train[populated], woe_part[populated]
                if a.std() > 0 and b.std() > 0:
                    woe_corr = float(np.corrcoef(a, b)[0, 1])
                else:
                    woe_corr = float("nan")
                flips = int(np.sum(np.sign(a) != np.sign(b)))
            else:
                woe_corr = float("nan")
                flips = 0

            drop_pct = (iv_train - iv_part) / iv_train
            reasons = []
            if drop_pct > max_drop:
                reasons.append("iv_drop")
            if iv_part < Config.WOE_MIN_IV:
                reasons.append("iv_below_floor")
            if not np.isnan(woe_corr) and woe_corr < min_corr:
                reasons.append("woe_reversal")

            features[col] = {
                "iv_train":       round(iv_train, 4),
                iv_key:           round(float(iv_part), 4),
                "iv_drop_pct":    round(float(drop_pct), 4),
                "woe_corr":       None if np.isnan(woe_corr) else round(woe_corr, 4),
                "woe_sign_flips": flips,
                "n_bins_scored":  n_pop,
                "flag":           "UNSTABLE" if reasons else "STABLE",
                "reasons":        reasons,
            }

        unstable = {c: m for c, m in features.items() if m["flag"] == "UNSTABLE"}
        result = {
            "n_evaluated":      len(features),
            "n_unstable":       len(unstable),
            "n_skipped_weak":   n_skipped_weak,
            "features":         features,
        }

        if unstable:
            worst = sorted(unstable.items(),
                           key=lambda kv: kv[1]["iv_drop_pct"], reverse=True)
            top = worst[:Config.FE_IV_STABILITY_TOP_N]
            detail = ", ".join(
                f"{c}(iv {m['iv_train']:.3f}→{m[iv_key]:.3f}, "
                f"corr={m['woe_corr']}, {'+'.join(m['reasons'])})"
                for c, m in top
            )
            more = f" (+{len(worst) - len(top)} more)" if len(worst) > len(top) else ""
            self.logger.log(self.name, f"IV stability WARN train↔{tag}",
                f"{len(unstable)}/{len(features)} features unstable. "
                f"Worst {len(top)}: {detail}{more}. "
                "Diagnostic only — Agent 3's PSI / SHAP+PSI prune decide what gets dropped.")
        else:
            self.logger.log(self.name, f"IV stability train↔{tag}",
                f"{len(features)} features scored, none flagged "
                f"(max_drop={max_drop:.0%}, min_woe_corr={min_corr:.2f})")
        return result

    def _setup_protected_cols(self, previous_report: Dict[str, Any]) -> None:
        """Compute self._protected_cols + self._date_col from prior agent's report.
        Idempotent — called by both fit_transform and process (single-file mode).
        """
        _entity_id = previous_report.get("entity_id_col")
        _composite = previous_report.get("composite_key_cols", [])
        self._protected_cols = set(_composite)
        if _entity_id:
            self._protected_cols.add(_entity_id)
        self._protected_cols.discard(self.target_column)

        # Infer date column from composite key (the non-entity partner, if any).
        self._date_col = next(
            (c for c in _composite if c != _entity_id and c in self.df.columns),
            None,
        )
        if self._date_col:
            self._protected_cols.add(self._date_col)

        if self._protected_cols:
            self.logger.log(self.name, "Protected Cols",
                f"Skipping feature engineering for: {sorted(self._protected_cols)}")
        if self._date_col:
            self.logger.log(self.name, "Date Col",
                f"'{self._date_col}' identified as temporal column — excluded from interactions")

    # ── FIT / TRANSFORM API ───────────────────────────────────────────────

    def fit_transform(
        self,
        train_path: str,
        previous_report: Dict[str, Any],
        target_column: str,
        create_interactions: Optional[bool] = None,
    ) -> Tuple[pd.DataFrame, FeatureSpec, List[str], Tuple[int, int]]:
        """Load TRAIN, run LLM feature-engineering, return (engineered_train, spec, actions, original_shape).

        The spec captures every column transform so valid/oot can be transformed
        without re-running the LLM. Encoders are fitted on train only.

        create_interactions: per-run override for the interaction step. None →
        use Config.FE_CREATE_INTERACTIONS_ENABLED; False → skip all
        create_interaction actions this run.
        """
        self._apply_create_interactions_override(create_interactions)
        self.logger.log(self.name, "fit_transform start", f"Loading train from {train_path}")
        self.df = self.load_dataframe(train_path)
        self.target_column = self._resolve_target_column(self.df, target_column)
        original_shape = self.df.shape

        self._setup_protected_cols(previous_report)
        self.logger.log(self.name, "Previous Agent Summary", previous_report.get("summary", "No summary"))

        analysis = self._analyze_features()
        llm_response = self.call_llm(
            self._build_engineering_prompt(analysis, previous_report),
            self._get_system_prompt(),
            json_mode=True,
            max_tokens=Config.LLM_MAX_TOKENS_LARGE,
        )

        spec = FeatureSpec(target_column=self.target_column)
        actions_taken = self._execute_llm_decisions(llm_response, spec=spec)
        return self.df, spec, actions_taken, original_shape

    def transform(self, df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
        """Apply a captured FeatureSpec to valid/oot. No LLM, no refit."""
        return spec.apply(df, logger=self.logger, name=self.name)

    def process_splits(
        self,
        train_path: str,
        previous_report: Dict[str, Any],
        target_column: str,
        valid_path: Optional[str] = None,
        oot_path: Optional[str] = None,
        test_path: Optional[str] = None,
        create_interactions: Optional[bool] = None,
    ) -> Tuple[Dict[str, Optional[str]], Dict[str, Any]]:
        """Pre-split mode entry point. Fit on train, transform the holdouts one at a time.

        create_interactions: per-run override for the interaction step. None →
        use Config.FE_CREATE_INTERACTIONS_ENABLED; False → skip all
        create_interaction actions this run.

        `test` is the random holdout single-file mode produces when there is no
        usable date column; it replays the FeatureSpec exactly like valid/oot.

        Each holdout is also scored for IV stability against train while it is
        in memory — see `_evaluate_iv_stability`.

        Returns ({"train": path, "valid": path|None, "oot": path|None,
        "test": path|None}, report).
        """
        import gc

        train_df, spec, actions_taken, original_shape = self.fit_transform(
            train_path, previous_report, target_column,
            create_interactions=create_interactions)

        Path(Config.ENGINEERED_TRAIN_PATH).parent.mkdir(exist_ok=True)
        train_df.to_parquet(Config.ENGINEERED_TRAIN_PATH, compression="snappy", index=False)
        self.logger.log(self.name, "Train saved",
            f"shape={train_df.shape} | path={Config.ENGINEERED_TRAIN_PATH}")
        train_final_shape = train_df.shape
        train_columns = list(train_df.columns)
        del train_df, self.df
        self.df = None
        gc.collect()

        out_paths: Dict[str, Optional[str]] = {
            "train": Config.ENGINEERED_TRAIN_PATH,
            "valid": None, "oot": None, "test": None,
        }
        # IV stability is scored during the transform pass — the bin assignment
        # it needs is already computed there, so this costs one callback per
        # WoE'd column and no second read of the partition.
        stability_enabled = (
            Config.FE_IV_STABILITY_ENABLED
            and bool(self._last_iv_scores)
            and bool(spec.woe_maps)
        )
        if Config.FE_IV_STABILITY_ENABLED and not stability_enabled:
            self.logger.log(self.name, "IV stability skipped",
                "no cached IV scores or no WoE maps — compute_iv / apply_woe_transform "
                "did not run, so there is no frozen train binning to compare against")
        iv_stability: Dict[str, Any] = {}

        for tag, in_path, out_path in [
            ("valid", valid_path, Config.ENGINEERED_VALID_PATH),
            ("oot",   oot_path,   Config.ENGINEERED_OOT_PATH),
            ("test",  test_path,  Config.ENGINEERED_TEST_PATH),
        ]:
            if in_path is None:
                continue
            self.logger.log(self.name, f"Transform {tag} start", f"path={in_path}")
            df = self.load_dataframe(in_path)
            orig = df.shape

            on_bins = None
            counts: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            if stability_enabled and self.target_column in df.columns:
                y_part = pd.to_numeric(df[self.target_column], errors="coerce")
                # Rows with no label carry no IV signal; bin 0 them out by
                # treating them as negatives would skew the counts, so only
                # score when the partition is fully labelled.
                if y_part.notna().all() and y_part.nunique() == 2:
                    on_bins, counts = self._make_iv_stability_collector(
                        y_part.to_numpy(dtype=np.float64), spec)
                else:
                    self.logger.log(self.name, f"IV stability skipped ({tag})",
                        f"target is not a fully-labelled binary column on this partition "
                        f"(nulls={int(y_part.isna().sum())}, nunique={y_part.nunique()})")

            df = spec.apply(df, logger=self.logger, name=self.name, on_bins=on_bins)
            df.to_parquet(out_path, compression="snappy", index=False)
            self.logger.log(self.name, f"Transform {tag} done",
                f"shape {orig} -> {df.shape} | saved={out_path}")
            out_paths[tag] = out_path
            del df
            gc.collect()

            if counts:
                iv_stability[tag] = self._evaluate_iv_stability(counts, spec, tag)
                del counts
                gc.collect()

        report = {
            "agent": self.name,
            "mode": "split",
            "original_shape": list(original_shape),
            "final_shape": list(train_final_shape),
            "actions_taken": actions_taken,
            "summary": self._generate_summary(actions_taken),
            "final_features": train_columns,
            "entity_id_col": previous_report.get("entity_id_col"),
            "composite_key_cols": previous_report.get("composite_key_cols", []),
            "target_column": self.target_column,
            "out_paths": {k: v for k, v in out_paths.items() if v is not None},
            "feature_spec": spec.to_dict(),
        }
        if iv_stability:
            report["iv_stability"] = {
                "thresholds": {
                    "max_iv_drop_pct": Config.FE_IV_STABILITY_MAX_DROP,
                    "min_woe_corr":    Config.FE_IV_STABILITY_MIN_WOE_CORR,
                    "min_iv":          Config.WOE_MIN_IV,
                    "min_bin_count":   Config.FE_IV_STABILITY_MIN_BIN_COUNT,
                },
                "note": "Holdout IV recomputed under train-frozen bin edges. "
                        "Diagnostic only — no feature is dropped here.",
                **iv_stability,
            }
        self.save_report(report, Config.FEATURE_ENGINEER_REPORT_PATH)
        self._emit_pipeline_process_script(spec)
        self.logger.log(self.name, "Process Complete",
            f"train={train_final_shape} | valid={'y' if out_paths['valid'] else '-'} "
            f"| oot={'y' if out_paths['oot'] else '-'} "
            f"| test={'y' if out_paths['test'] else '-'}")
        return out_paths, report

    def process(
        self,
        df: pd.DataFrame,
        previous_report: Dict[str, Any],
        target_column: str,
        create_interactions: Optional[bool] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Fit and transform ONE frame. Standalone entry point — not used by the pipeline.

        AutoMLPipeline always calls `process_splits` now: single-file runs are
        split upfront so this agent only ever fits on TRAIN. This method is kept
        for callers who hand over a single frame themselves (tests/test_agent2.py).

        Note what that means: everything here — IV, WoE bin edges, feature
        selection, interaction medians — is fitted on every row of `df`. If the
        caller intends to carve a holdout out of the result afterwards, those
        statistics will already have seen it. Use `process_splits` when that
        matters.

        create_interactions: per-run override for the interaction step. None →
        use Config.FE_CREATE_INTERACTIONS_ENABLED; False → skip all
        create_interaction actions this run.
        """
        self._apply_create_interactions_override(create_interactions)
        self.logger.log(self.name, "Process Start", f"Received clean data with shape {df.shape}")

        self.df = df.copy()
        self.target_column = self._resolve_target_column(self.df, target_column)
        original_shape = self.df.shape

        self._setup_protected_cols(previous_report)
        self.logger.log(self.name, "Previous Agent Summary", previous_report.get("summary", "No summary"))

        analysis = self._analyze_features()
        llm_response = self.call_llm(
            self._build_engineering_prompt(analysis, previous_report),
            self._get_system_prompt(),
            json_mode=True,
            max_tokens=Config.LLM_MAX_TOKENS_LARGE,
        )
        spec = FeatureSpec(target_column=self.target_column)
        actions_taken = self._execute_llm_decisions(llm_response, spec=spec)

        self.df.to_parquet(Config.ENGINEERED_DATA_PATH, compression="snappy", index=False)
        self.logger.log(self.name, "Data Saved", f"Engineered data saved to {Config.ENGINEERED_DATA_PATH}")

        report = {
            "agent": self.name,
            "mode": "single",
            "original_shape": list(original_shape),
            "final_shape": list(self.df.shape),
            "actions_taken": actions_taken,
            "summary": self._generate_summary(actions_taken),
            "final_features": list(self.df.columns),
            "entity_id_col": previous_report.get("entity_id_col"),
            "composite_key_cols": previous_report.get("composite_key_cols", []),
            "target_column": self.target_column,
            "feature_spec": spec.to_dict(),
        }
        self.save_report(report, Config.FEATURE_ENGINEER_REPORT_PATH)
        self._emit_pipeline_process_script(spec)
        self.logger.log(self.name, "Process Complete", f"Shape: {original_shape} -> {self.df.shape}")
        return Config.ENGINEERED_DATA_PATH, report

    def _emit_pipeline_process_script(self, spec: FeatureSpec) -> None:
        """Pickle the fitted FeatureSpec + emit a standalone replay script.

        Encoders can't be embedded as literals (they're fitted sklearn objects),
        so they go into a sidecar `.pkl` next to the script. The script itself
        delegates to FeatureSpec.apply in the repo (needs the repo on sys.path).

        Usage at replay time:
            python pipeline_process_feature_engineer.py <input_file> <output_parquet>
        """
        import joblib
        from datetime import datetime

        # Persist the spec as a sidecar — the script loads it by relative path
        joblib.dump(spec, Config.PIPELINE_PROCESS_FE_SPEC_PATH)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        spec_filename = Path(Config.PIPELINE_PROCESS_FE_SPEC_PATH).name

        code = f'''"""
Auto-generated by FeatureEngineerAgent  |  {ts}
Interactions    : {len(spec.interactions)}
Label encoders  : {len(spec.label_encoders)}
One-hot columns : {len(spec.onehot_columns)}
WoE transforms  : {len(spec.woe_maps)}
Selected features : {0 if spec.selected_features is None else len(spec.selected_features)}

Replay the feature engineering transforms captured during the original
pipeline run. Loads the fitted FeatureSpec from {spec_filename!r} (sidecar)
and applies it deterministically — no LLM call, no re-fit.

Usage:
    python pipeline_process_feature_engineer.py <input_file> <output_file.parquet>
"""

import os
import sys
from pathlib import Path
import pandas as pd


def _add_repo_to_path() -> None:
    """The spec is a pickle of this repo's classes, and the transform logic lives
    in FeatureSpec.apply (single implementation, no copy to drift)."""
    here = Path(__file__).resolve().parent
    cands = ([Path(os.environ["AUTOML_REPO"])] if os.environ.get("AUTOML_REPO") else []) \
        + list(here.parents[:4])
    for c in cands:
        if (c / "Agents").is_dir() and (c / "config.py").is_file():
            sys.path.insert(0, str(c))
            return
    sys.exit("ERROR: pipeline repo not found. Set AUTOML_REPO=/path/to/multi-agent-auto-ml-v1.1")


_add_repo_to_path()
import joblib
from Agents.BaseAgent.base_agent import BaseAgent

_spec = joblib.load(Path(__file__).parent / "{spec_filename}")


def apply(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Replay the captured transforms. strict=True raises FeatureContractError
    instead of silently substituting train medians / dropping columns."""
    return _spec.apply(df, strict=strict)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pipeline_process_feature_engineer.py <input_file> <output_file.parquet>",
              file=sys.stderr)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    df = BaseAgent.load_dataframe(src)      # same reader normalisation as training
    before = df.shape
    df = apply(df)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False, compression="snappy")
    print(f"Engineered: {{before}} -> {{df.shape}}  |  saved -> {{dst}}")
'''
        Path(Config.PIPELINE_PROCESS_FE_PATH).write_text(code, encoding="utf-8")
        self.logger.log(self.name, "Process Script Saved",
            f"{Config.PIPELINE_PROCESS_FE_PATH} (+ sidecar {spec_filename})")

    def _analyze_features(self) -> Dict[str, Any]:
        """Build stats for the LLM prompt from self.df. In split mode self.df IS train."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

        exclude = self._protected_cols | {self.target_column}
        numeric_cols = [c for c in numeric_cols if c not in exclude]
        categorical_cols = [c for c in categorical_cols if c not in exclude]

        numeric_meta: Dict[str, Any] = {}
        numeric_limit = min(
            max(10, int(len(numeric_cols) * Config.FEATURE_META_NUMERIC_RATIO)),
            Config.FEATURE_META_MAX_NUMERIC_COLS,
        )
        for col in numeric_cols[:numeric_limit]:
            s = self.df[col]
            non_null = s.dropna()
            entry: Dict[str, Any] = {
                "null_pct": round(float(s.isnull().mean() * 100), 2),
                "nunique": int(s.nunique()),
                "mean":  round(float(non_null.mean()), 4)  if len(non_null) > 0 else None,
                "std":   round(float(non_null.std()),  4)  if len(non_null) > 1 else None,
                "min":   round(float(non_null.min()),  4)  if len(non_null) > 0 else None,
                "max":   round(float(non_null.max()),  4)  if len(non_null) > 0 else None,
                "skew":  round(float(non_null.skew()), 4)  if len(non_null) > 2 else None,
            }
            if col in self._col_descriptions:
                entry["description"] = self._col_descriptions[col]
            numeric_meta[col] = entry

        categorical_meta: Dict[str, Any] = {}
        for col in categorical_cols[:Config.FEATURE_META_MAX_CATEGORICAL_COLS]:
            s = self.df[col]
            top_vals = s.value_counts().head(5).to_dict()
            entry = {
                "null_pct": round(float(s.isnull().mean() * 100), 2),
                "nunique": int(s.nunique()),
                "top_values": {str(k): int(v) for k, v in top_vals.items()},
            }
            if col in self._col_descriptions:
                entry["description"] = self._col_descriptions[col]
            categorical_meta[col] = entry

        return {
            "shape": self.df.shape,
            "total_features": len(self.df.columns) - 1,
            "numeric_features_total": len(numeric_cols),
            "numeric_features_shown": len(numeric_meta),
            "categorical_features_total": len(categorical_cols),
            "categorical_features_shown": len(categorical_meta),
            "protected_cols": sorted(self._protected_cols),
            "numeric_features": numeric_meta,
            "categorical_features": categorical_meta,
        }

    def _build_col_desc_section(self) -> Optional[str]:
        """Return a group-organised column description block for the LLM prompt.

        Descriptions stored as "[group] text" are clustered by group so the LLM
        can easily identify which columns come from the same source table and
        therefore form natural interaction pairs.
        """
        if not self._col_descriptions:
            return None

        current_cols = set(self.df.columns) - self._protected_cols - {self.target_column}
        relevant = {c: d for c, d in self._col_descriptions.items() if c in current_cols}
        if not relevant:
            return None

        groups: Dict[str, List[str]] = {}
        ungrouped: List[str] = []
        for col, desc in sorted(relevant.items()):
            m = re.match(r"^\[([^\]]+)\]\s*(.*)", desc)
            if m:
                group_name, col_desc = m.group(1), m.group(2).strip()
                entry = f"  - {col}: {col_desc}" if col_desc else f"  - {col}"
                groups.setdefault(group_name, []).append(entry)
            else:
                ungrouped.append(f"  - {col}: {desc}" if desc else f"  - {col}")

        lines: List[str] = []
        for group_name, entries in sorted(groups.items()):
            lines.append(f"[{group_name}] — {len(entries)} columns:")
            lines.extend(entries[: Config.MAX_DESC_PER_GROUP])
            if len(entries) > Config.MAX_DESC_PER_GROUP:
                lines.append(f"  ... and {len(entries) - Config.MAX_DESC_PER_GROUP} more")
        if ungrouped:
            lines.append(f"(no group) — {len(ungrouped)} columns:")
            lines.extend(ungrouped[: Config.MAX_DESC_PER_GROUP])
            if len(ungrouped) > Config.MAX_DESC_PER_GROUP:
                lines.append(f"  ... and {len(ungrouped) - Config.MAX_DESC_PER_GROUP} more")

        return "\n".join(lines)

    def _suggest_top_k(self) -> int:
        exclude = self._protected_cols | ({self.target_column} if self.target_column else set())
        n = max(1, sum(1 for c in self.df.columns if c not in exclude))
        return max(10, min(int(n * Config.TOP_K_RATIO), Config.TOP_K_FEATURES_CAP))

    def _get_system_prompt(self) -> str:
        exclude = self._protected_cols | ({self.target_column} if self.target_column else set())
        n_engineerable = max(1, sum(1 for c in self.df.columns if c not in exclude))
        protected_display = str(sorted(self._protected_cols)) if self._protected_cols else "none"
        date_col_display = f"'{self._date_col}'" if self._date_col else "none"
        suggested_k = self._suggest_top_k()
        top_k_ratio_pct = int(Config.TOP_K_RATIO * 100)
        return self._load_prompt(
            self._PROMPTS_ROOT / "FeatureEngineer" / "prompts" / "system.txt",
            TOOL_DESCRIPTIONS=self.tool_registry.get_tool_descriptions(),
            MIN_CORRELATION_THRESHOLD=Config.MIN_CORRELATION_THRESHOLD,
            N_ENGINEERABLE_FEATURES=n_engineerable,
            TOP_K_RATIO_PCT=top_k_ratio_pct,
            TOP_K_FEATURES_CAP=Config.TOP_K_FEATURES_CAP,
            SUGGESTED_K=suggested_k,
            PROTECTED_COLS=protected_display,
            DATE_COL=date_col_display,
            DOMAIN_GUIDANCE=self._get_domain_guidance(),
            PRODUCT_TYPE=self.product_type,
            PRODUCT_GUIDANCE=self._get_product_guidance(),
            TARGET_NEW_FEATURE_COUNT=Config.TARGET_NEW_FEATURE_COUNT,
            IV_LEAKAGE_THRESHOLD=Config.IV_LEAKAGE_THRESHOLD,
            MULTICOLLINEARITY_THRESHOLD=Config.MULTICOLLINEARITY_THRESHOLD,
            IV_MAX_NULL_RATIO_PCT=int(Config.IV_MAX_NULL_RATIO * 100),
        )

    def _build_engineering_prompt(self, analysis: Dict, previous_report: Dict) -> str:
        col_desc_block = self._build_col_desc_section()
        col_desc_section = (
            f"\nCOLUMN DESCRIPTIONS BY GROUP (format: [source_table] column_name: meaning):\n"
            f"{col_desc_block}\n"
            "Use the group names above to identify same-group vs cross-group interaction candidates.\n"
            if col_desc_block else ""
        )
        nc = getattr(self, "null_context", None)
        if nc:
            imputed = sorted(nc.get("imputed", {}))
            inds = list(nc.get("indicators", []))
            kept = sorted(nc.get("kept_nan", []))
            col_desc_section += (
                "\nNULL HANDLING ALREADY APPLIED (Stage 1b, fitted on train, replayed at scoring):\n"
                f"- {len(imputed)} columns had their NULLs imputed (median / zero / category). "
                "`isna()` on them is always False: do NOT create missing-flag features for them.\n"
                f"- Missing indicators already exist ({len(inds)}), e.g. {inds[:60]} — reuse these "
                "instead of re-deriving missingness.\n"
                f"- Columns that still carry NaN by design ({len(kept)}): {kept[:40]}.\n"
                "- Ratios / aggregates built from imputed columns use the imputed values; prefer "
                "combining them with the matching `<col>_missing` indicator when missingness matters.\n"
            )
        return self._load_prompt(
            self._PROMPTS_ROOT / "FeatureEngineer" / "prompts" / "user.txt",
            PREVIOUS_SUMMARY=previous_report.get("summary", "Data cleaning completed"),
            ANALYSIS_JSON=json.dumps(analysis, indent=2),
            COL_DESC_SECTION=col_desc_section,
            TARGET_COLUMN=self.target_column,
        )

    def _execute_llm_decisions(self, llm_response: str, spec: Optional["FeatureSpec"] = None) -> List[str]:
        """Apply LLM feature-engineering decisions AND populate `spec` for valid/oot replay.

        Captures into spec:
          - create_interaction → spec.interactions (new_col, expression, train_median fill)
          - encode_categorical (label) → spec.label_encoders[col]
          - encode_categorical (onehot) → spec.onehot_columns[col]
          - encode_all_categorical → spec.label_encoders / spec.onehot_columns (batch)
          - select_top_features → spec.selected_features
          - correlation_analysis: logged only, no transform recorded
        """
        if spec is None:
            spec = FeatureSpec(target_column=self.target_column)
        self.logger.log(self.name, "LLM Decision", "Parsing feature engineering decisions")
        actions_taken: List[str] = []
        fit_order: List[str] = []

        def _note_step(kind: str) -> None:
            # Record the order the fit really ran the step kinds in; apply() replays it.
            if kind not in fit_order:
                fit_order.append(kind)
            spec.step_order = fit_order + [k for k in ("interaction", "encode", "woe")
                                           if k not in fit_order]

        try:
            decisions = json.loads(self._extract_json(llm_response))
            self.logger.log(self.name, "LLM Reasoning", decisions.get("reasoning", "No reasoning provided"))

            for action_spec in decisions.get("actions", []):
                action_type = action_spec.get("action")
                reason = action_spec.get("reason", "No reason provided")

                self.logger.log(self.name, f"Action: {action_type}", reason)

                try:
                    if action_type == "create_interaction":
                        _note_step("interaction")
                        new_col = action_spec.get("new_col")
                        expression = action_spec.get("expression", "")
                        # Master switch: skip the whole interaction step when disabled
                        if not self.create_interactions_enabled:
                            self.logger.log(self.name, f"SKIP {action_type}",
                                f"interaction step disabled (FE_CREATE_INTERACTIONS_ENABLED=false) — skipped '{new_col}'")
                            continue
                        # Block if expression references any protected column
                        refs_protected = any(f"'{c}'" in expression or f'"{c}"' in expression
                                            for c in self._protected_cols)
                        if refs_protected:
                            self.logger.log(self.name, f"SKIP {action_type}",
                                f"Expression references a protected column — skipped: {expression}")
                            continue
                        # Missing-flag on a column Stage 1b already imputed is constant:
                        # skip it and point at the indicator that carries the signal instead.
                        imputed = (getattr(self, "null_context", None) or {}).get("imputed", {})
                        if imputed:
                            hits = [c for c in re.findall(
                                r"df\[\s*['\"]([^'\"]+)['\"]\s*\]\s*\.\s*(?:isna|isnull|notna|notnull)\s*\(",
                                expression) if c in imputed]
                            if hits:
                                self.logger.log(self.name, f"SKIP {action_type}",
                                    f"'{new_col}': NULLs of {hits} were already imputed by the null "
                                    f"processor (use {[h + '_missing' for h in hits]} if present)")
                                continue
                        self._last_interaction_fill = 0.0
                        self.df = self.execute_tool("create_interaction", df=self.df,
                                                    new_col=new_col, expression=expression)
                        spec.interactions.append((new_col, expression, float(self._last_interaction_fill)))
                        actions_taken.append(f"Created feature '{new_col}': {reason}")

                    elif action_type == "encode_all_categorical":
                        _note_step("encode")
                        method = action_spec.get("method", "label")
                        self._batch_label_encoders = {}
                        self._batch_onehot_cols = {}
                        self.df = self.execute_tool("encode_all_categorical", df=self.df, method=method)
                        spec.label_encoders.update(self._batch_label_encoders)
                        spec.onehot_columns.update(self._batch_onehot_cols)
                        actions_taken.append(f"Encoded all categorical columns with {method}: {reason}")

                    elif action_type == "encode_categorical":
                        _note_step("encode")
                        column = action_spec.get("column")
                        if column in self._protected_cols:
                            self.logger.log(self.name, f"SKIP {action_type}",
                                f"'{column}' is a protected key column — skipped")
                            continue
                        method = action_spec.get("method", "label")
                        self._last_label_encoder = None
                        self._last_onehot_cols = None
                        self.df = self.execute_tool("encode_categorical", df=self.df,
                                                    col=column, method=method)
                        if method == "label" and self._last_label_encoder is not None:
                            spec.label_encoders[column] = self._last_label_encoder
                        elif method == "onehot" and self._last_onehot_cols is not None:
                            spec.onehot_columns[column] = self._last_onehot_cols
                        actions_taken.append(f"Encoded '{column}' with {method}: {reason}")

                    elif action_type == "correlation_analysis":
                        result = self.execute_tool("correlation_analysis", df=self.df, target=self.target_column)
                        actions_taken.append(f"Analyzed correlations: {reason}")
                        self.logger.log(self.name, "Correlation Results", result[:500])

                    elif action_type == "compute_iv":
                        bins = int(action_spec.get("bins", Config.IV_BINS_DEFAULT))
                        method = action_spec.get("method", "quantile")
                        # Reset IV cache so a re-run starts clean; on failure the
                        # selector falls back to f_classif rather than reading stale scores.
                        self._last_iv_scores = {}
                        self._last_woe_maps = {}
                        self._last_woe_data = {}
                        result = self.execute_tool("compute_iv", df=self.df,
                                                   target=self.target_column,
                                                   bins=bins, method=method)
                        actions_taken.append(f"Computed IV (bins={bins}, method={method}): {reason}")
                        self.logger.log(self.name, "IV Results", result[:1000])

                    elif action_type == "apply_woe_transform":
                        _note_step("woe")
                        min_iv = float(action_spec.get("min_iv", Config.WOE_MIN_IV))
                        self._last_applied_woe = {}
                        self.df = self.execute_tool("apply_woe_transform",
                                                    df=self.df, min_iv=min_iv)
                        # Persist the applied WoE payload into the spec so valid/oot
                        # replay binning + WoE-lookup is identical to TRAIN.
                        spec.woe_maps.update(self._last_applied_woe)
                        actions_taken.append(
                            f"Applied WoE transform to {len(self._last_applied_woe)} "
                            f"features (min_iv={min_iv}): {reason}"
                        )

                    elif action_type == "select_top_features":
                        k = int(action_spec.get("k", self._suggest_top_k()))
                        criterion = action_spec.get("criterion", "default")
                        # Snapshot protected cols — SelectKBest drops them since they're non-numeric / not scored
                        protected_snapshot = {
                            col: self.df[col].copy()
                            for col in self._protected_cols
                            if col in self.df.columns
                        }
                        self._last_selected_features = None
                        self.df = self.execute_tool("select_top_features", df=self.df,
                                                    target=self.target_column, k=k,
                                                    criterion=criterion)
                        # Re-add any protected cols that were removed by selection
                        restored: List[str] = []
                        for col, series in protected_snapshot.items():
                            if col not in self.df.columns:
                                self.df[col] = series.values
                                restored.append(col)
                                self.logger.log(self.name, "Protected Col Restored",
                                    f"Re-added '{col}' (composite key / entity ID) after feature selection")
                        # Capture the FINAL column list (including restored protected cols)
                        # so transform applies the same selection to valid/oot
                        spec.selected_features = list(self.df.columns)
                        actions_taken.append(f"Selected top {k} features (criterion={criterion}): {reason}")

                    elif action_type in BaseAgent.NOTE_ACTIONS:
                        self.logger.log(self.name, "LLM Note",
                            f"{action_type} (column={action_spec.get('column')}): {reason}")
                    else:
                        self.logger.log(self.name, "WARN",
                            f"Unknown or missing action_type '{action_type}' — skipped")

                except Exception as action_err:
                    self.logger.log(self.name, f"SKIP {action_type}",
                        f"Action failed, continuing with next action: {action_err}")

        except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
            self.logger.log(self.name, "ERROR", f"Failed to parse LLM response: {e}")
            self.logger.log(self.name, "Raw Response", llm_response[:500])
            actions_taken.append("ERROR: Could not parse LLM decisions, performed basic encoding")
            self._fallback_engineering(spec)

        return actions_taken

    def _fallback_engineering(self, spec: Optional["FeatureSpec"] = None):
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        skip = self._protected_cols | {self.target_column}
        for col in categorical_cols:
            if col in skip:
                continue
            self._last_label_encoder = None
            self.df = self._tool_encode_categorical(self.df, col, "label")
            if spec is not None and self._last_label_encoder is not None:
                spec.label_encoders[col] = self._last_label_encoder
            self.logger.log(self.name, "Fallback", f"Label encoded {col}")

    def _generate_summary(self, actions: List[str]) -> str:
        if not actions:
            return "No feature engineering was necessary."
        preview = actions[:Config.SUMMARY_ACTION_PREVIEW]
        summary = f"Performed {len(actions)} feature engineering actions: " + "; ".join(preview)
        if len(actions) > Config.SUMMARY_ACTION_PREVIEW:
            summary += f"; and {len(actions) - Config.SUMMARY_ACTION_PREVIEW} more actions."
        return summary
