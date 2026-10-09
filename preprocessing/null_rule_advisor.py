"""Suggest NULL-handling rules (missing_type / strategy / indicator) per feature.

Business semantics of a NULL are not declared anywhere, so a human or an LLM
has to propose them. Two hard rules:

  * No human gate (AutoML). Instead every LLM answer passes deterministic
    guardrails (`_guard`) and falls back to a heuristic when invalid; each
    rule records `source` + `reason` as an audit trail in the artifact.
  * Nothing row-level leaves the process. The LLM prompt contains column
    names, dtype kind and aggregate statistics only, never values.

`llm_call(prompt, system_prompt) -> str` is injected (e.g. BaseAgent.call_llm),
so this module has no LLM dependency. Any invalid / missing LLM answer falls
back to a conservative heuristic for that column.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd

from .null_processor import MISSING_TYPES, STRATEGIES, _ALLOWED, FeatureRule

_SYSTEM = (
    "You are a credit-risk data scientist. For each feature decide what a NULL "
    "means and how to treat it. Return ONLY a JSON object, no prose."
)

_PROMPT = """Features (aggregate statistics only):
{payload}

{desc}For each feature return:
  "missing_type": one of {types}
  "missing_strategy": one of {strategies} (must be valid for its group)
  "add_missing_indicator": true|false
  "reason": <= 8 words

Guidance: NULL is NOT zero unless absence of activity means zero (counts/amounts of
transactions over a window). Use median + indicator when NULL means "unknown/not
available". Prefer an indicator whenever missing_rate > 0.02 and the cause is structural.

Return: {{"features": {{"<name>": {{...}}, ...}}}}"""


def _group_of(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "binary"
    if pd.api.types.is_numeric_dtype(s):
        return "binary" if s.dropna().nunique() <= 2 else "numerical"
    return "categorical"


def _profile(df: pd.DataFrame, cols: Iterable[str]) -> List[Dict[str, Any]]:
    out = []
    for c in cols:
        s = df[c]
        g = _group_of(s)
        row: Dict[str, Any] = {"name": c, "group": g,
                               "missing_rate": round(float(s.isna().mean()), 4),
                               "nunique": int(s.nunique(dropna=True))}
        if g != "categorical":
            nn = pd.to_numeric(s, errors="coerce").dropna()
            row["zero_rate"] = round(float((nn == 0).mean()), 4) if len(nn) else None
            row["min"] = float(nn.min()) if len(nn) else None
            row["max"] = float(nn.max()) if len(nn) else None
        out.append(row)
    return out


def heuristic_rule(group: str, missing_rate: float) -> Dict[str, Any]:
    """Conservative default when nothing better is known."""
    if missing_rate >= 1.0:                 # nothing to learn a fill value from
        strat, ind = "none", False
    elif group == "categorical":
        strat, ind = "explicit_category", False
    elif group == "binary":
        strat, ind = ("missing_indicator_only", True) if missing_rate > 0 else ("none", False)
    else:
        strat, ind = "median", missing_rate > 0.02
    return {"group": group, "missing_type": "unknown", "missing_strategy": strat,
            "add_missing_indicator": ind, "source": "heuristic"}


def _guard(rule: Dict[str, Any], prof: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic safety net applied to every LLM suggestion."""
    mr = prof["missing_rate"]
    note = ""
    if mr >= 1.0 and rule["missing_strategy"] in ("mean", "median"):
        rule["missing_strategy"], rule["add_missing_indicator"], note = "none", False, "all-null in train"
    # `zero` only when 0 is a plausible value: non-negative column that already has zeros
    if rule["missing_strategy"] == "zero" and not (
            (prof.get("min") is not None and prof["min"] >= 0) and (prof.get("zero_rate") or 0) > 0):
        # fallback must stay valid for the column's group (median is not allowed on binary)
        rule["missing_strategy"] = ("missing_indicator_only" if rule["group"] == "binary" else "median")
        rule["add_missing_indicator"] = True
        note = "zero rejected: column not a non-negative count/amount with zeros"
    # structural / sizeable NULLs keep their signal unless NULL is explicitly 'no activity'
    if (rule["missing_type"] == "structural" or mr > 0.02) and not rule["add_missing_indicator"]             and rule["missing_strategy"] not in ("none", "zero"):
        rule["add_missing_indicator"], note = True, (note + "; indicator forced").strip("; ")
    if note:
        rule["reason"] = (rule.get("reason", "") + f" [guard: {note}]").strip()
    return rule


def _validated(group: str, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        mt = raw["missing_type"]
        st = raw["missing_strategy"]
        ind = bool(raw["add_missing_indicator"])
    except (KeyError, TypeError):
        return None
    if mt not in MISSING_TYPES or st not in STRATEGIES or st not in _ALLOWED[group]:
        return None
    if st == "constant":                # needs a value the LLM cannot know
        return None
    if st == "missing_indicator_only" and not ind:
        ind = True
    return {"group": group, "missing_type": mt, "missing_strategy": st,
            "add_missing_indicator": ind, "source": "llm",
            "reason": str(raw.get("reason", ""))[:160]}


def suggest_rules(
    train: pd.DataFrame,
    llm_call: Optional[Callable[[str, str], str]] = None,
    exclude: Iterable[str] = (),
    col_descriptions: Optional[Dict[str, str]] = None,
    batch_size: int = 40,
    max_workers: int = 4,
) -> Dict[str, Any]:
    """-> config dict ready for `NullProcessor(config=...)` (after human review)."""
    cols = [c for c in train.columns if c not in set(exclude)]
    profile = {p["name"]: p for p in _profile(train, cols)}
    features: Dict[str, Dict[str, Any]] = {}

    # Columns with no NULL in train carry no semantics to decide: skip the LLM (saves tokens).
    todo = [c for c in cols if profile[c]["missing_rate"] > 0]

    def _ask(batch):
        desc = ""
        if col_descriptions:
            lines = [f"- {c}: {col_descriptions[c]}" for c in batch if c in col_descriptions]
            desc = ("Column descriptions:\n" + "\n".join(lines) + "\n\n") if lines else ""
        prompt = _PROMPT.format(payload=json.dumps([profile[c] for c in batch]),
                                desc=desc, types=list(MISSING_TYPES),
                                strategies=list(STRATEGIES))
        answer = {}
        try:
            text = llm_call(prompt, _SYSTEM)
        except Exception:               # advisory step: never block on LLM failure
            text = ""
        try:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            answer = json.loads(m.group(0)).get("features", {}) if m else {}
        except Exception:
            # Truncated / malformed JSON: salvage every complete per-feature object.
            for name, body in re.findall(r'"([^"]+)"\s*:\s*(\{[^{}]*\})', text):
                try:
                    answer[name] = json.loads(body)
                except Exception:
                    pass
        out = {}
        for c in batch:
            rule = _validated(profile[c]["group"], answer.get(c) or {})
            if rule:
                out[c] = _guard(rule, profile[c])
        return out

    if llm_call is not None and todo:
        batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
        # Batches are independent: run them concurrently (wide datasets = dozens of calls).
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            for part in ex.map(_ask, batches):
                features.update(part)

    for c in cols:
        if c not in features:
            features[c] = heuristic_rule(profile[c]["group"], profile[c]["missing_rate"])

    # Last line of defence: every rule must pass NullProcessor's own validation, otherwise it
    # is replaced by the heuristic. A bad suggestion must never abort a multi-hour run.
    allowed = {k for k in FeatureRule.__dataclass_fields__}
    for c, r in list(features.items()):
        try:
            FeatureRule(**{k: v for k, v in r.items() if k in allowed}).validate(c)
        except Exception:
            features[c] = heuristic_rule(profile[c]["group"], profile[c]["missing_rate"])
    return {"features": features}
