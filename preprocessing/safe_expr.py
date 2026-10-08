"""Allowlist-validated evaluation of LLM-written feature expressions.

Why: `eval(expr, {"__builtins__": whitelist})` is not a sandbox. Attribute
chains such as `df.__class__.__init__.__globals__[...]` reach `__import__`
regardless of the builtins whitelist (reproduced on this repo). Expressions come
from an LLM and are re-evaluated at every scoring call, so they are validated
structurally instead:

  * parsed with `ast` in eval mode; every node type must be on an allowlist
  * names: `df`, `np`, and a few value-level builtins only
  * attributes: never dunder/underscore; methods from a fixed pandas list;
    `np.<fn>` from a fixed numpy list; no attribute access on anything else
    (so `df.col` is rejected, use `df['col']`)
  * no lambda / comprehension / f-string / walrus / star-args / await
  * bounded size; `**` only with a small constant exponent; no str arithmetic

A rejected expression raises `UnsafeExpressionError` (a ValueError). Specs that
carry one must fail hard rather than fall back to a median.
"""

from __future__ import annotations

import ast
from typing import Any, Dict

import numpy as np

MAX_LEN = 600
MAX_NODES = 250

_BUILTINS: Dict[str, Any] = {
    "int": int, "float": float, "bool": bool, "str": str,
    "abs": abs, "min": min, "max": max, "round": round, "len": len,
}
_NAMES = {"df", "np"} | set(_BUILTINS)

_PD_METHODS = {
    "astype", "isna", "isnull", "notna", "notnull", "fillna", "clip", "abs", "mean",
    "min", "max", "std", "var", "sum", "median", "round", "where", "mask", "replace",
    "between", "isin", "rank", "count", "nunique", "prod", "pow", "div", "mul", "add",
    "sub", "floordiv", "mod", "eq", "ne", "lt", "le", "gt", "ge", "log", "sqrt",
    "exp", "floor", "ceil", "sign", "diff", "cumsum", "any", "all", "to_numpy",
    "values", "loc",
}
_NP_FUNCS = {
    "log", "log1p", "log2", "log10", "exp", "expm1", "sqrt", "abs", "absolute",
    "where", "minimum", "maximum", "clip", "sign", "isnan", "isfinite", "floor",
    "ceil", "round", "power", "square", "mean", "median", "nan", "inf", "pi",
    "float64", "int64", "nan_to_num", "tanh", "sin", "cos", "select", "divide",
}
_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
           ast.BitAnd, ast.BitOr, ast.BitXor)
_UNOPS = (ast.USub, ast.UAdd, ast.Not, ast.Invert)
_CMPOPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
           ast.Is, ast.IsNot)
_LEAF_OK = (ast.Load, ast.And, ast.Or) + _BINOPS + _UNOPS + _CMPOPS


class UnsafeExpressionError(ValueError):
    """Expression uses a construct outside the allowlist."""


def _is_str_const(n: ast.AST) -> bool:
    return isinstance(n, ast.Constant) and isinstance(n.value, str)


def validate_expression(expr: str) -> ast.Expression:
    if not isinstance(expr, str) or not expr.strip():
        raise UnsafeExpressionError("empty expression")
    if len(expr) > MAX_LEN:
        raise UnsafeExpressionError(f"expression longer than {MAX_LEN} chars")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise UnsafeExpressionError(f"syntax error: {e.msg}") from e

    count = 0
    for node in ast.walk(tree):
        count += 1
        if count > MAX_NODES:
            raise UnsafeExpressionError(f"expression has more than {MAX_NODES} nodes")
        t = type(node)
        if isinstance(node, _LEAF_OK) or t in (ast.Expression, ast.Tuple, ast.List,
                                               ast.Slice, ast.IfExp, ast.keyword):
            continue
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool, str, type(None))):
                if isinstance(node.value, str) and len(node.value) > 200:
                    raise UnsafeExpressionError("string constant too long")
                if isinstance(node.value, (int, float)) and abs(node.value) > 1e15:
                    raise UnsafeExpressionError("numeric constant out of range")
                continue
            raise UnsafeExpressionError(f"constant of type {type(node.value).__name__}")
        if isinstance(node, ast.Name):
            if node.id not in _NAMES:
                raise UnsafeExpressionError(f"name '{node.id}' not allowed")
            continue
        if isinstance(node, ast.BinOp):
            if _is_str_const(node.left) or _is_str_const(node.right):
                raise UnsafeExpressionError("string arithmetic not allowed")
            if isinstance(node.op, ast.Pow):
                r = node.right
                if not (isinstance(r, ast.Constant) and isinstance(r.value, (int, float))
                        and abs(r.value) <= 10):
                    raise UnsafeExpressionError("'**' needs a constant exponent with |n| <= 10")
            continue
        if isinstance(node, (ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Subscript)):
            continue
        if isinstance(node, ast.Attribute):
            name = node.attr
            if name.startswith("_"):
                raise UnsafeExpressionError(f"attribute '{name}' not allowed")
            base = node.value
            if isinstance(base, ast.Name) and base.id == "np":
                if name not in _NP_FUNCS:
                    raise UnsafeExpressionError(f"np.{name} not allowed")
            elif isinstance(base, ast.Name) and base.id == "df":
                raise UnsafeExpressionError("use df['column'], not attribute access on df")
            elif name not in _PD_METHODS:
                raise UnsafeExpressionError(f"attribute '{name}' not allowed")
            continue
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                if f.id not in _BUILTINS:
                    raise UnsafeExpressionError(f"call to '{f.id}' not allowed")
            elif not isinstance(f, ast.Attribute):
                raise UnsafeExpressionError("only direct function / method calls allowed")
            for a in node.args:
                if isinstance(a, ast.Starred):
                    raise UnsafeExpressionError("star-args not allowed")
            for k in node.keywords:
                if k.arg is None:
                    raise UnsafeExpressionError("**kwargs not allowed")
            continue
        raise UnsafeExpressionError(f"construct '{t.__name__}' not allowed")
    return tree


def safe_eval(expr: str, df: Any) -> Any:
    """Validate, then evaluate with only df / np / value-level builtins visible."""
    tree = validate_expression(expr)
    code = compile(tree, "<feature-expression>", "eval")
    env = {"df": df, "np": np, **_BUILTINS}
    return eval(code, {"__builtins__": {}}, env)  # noqa: S307  (validated above)
