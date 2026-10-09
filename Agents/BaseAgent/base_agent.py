import re
import time
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from logger import AgentLogger
from config import Config
import json
from pathlib import Path


class BaseAgent:

    # Actions some models emit that are notes, not transforms (e.g. Haiku's "flag_for_review").
    # Logged as an audit note instead of an "unknown action" warning.
    NOTE_ACTIONS = frozenset({"flag_for_review", "note", "comment", "review", "observation"})

    # Root folder for all agent prompt templates (Agents/BaseAgent/ → Agents/)
    _PROMPTS_ROOT: Path = Path(__file__).parent.parent

    # Lazy-initialized pyarrow S3FileSystem (built once on first S3 access)
    _S3_FS: Optional[Any] = None

    # Outer compression suffix → pandas compression arg.
    # .b2 is a non-standard alias for .bz2 kept for convenience.
    # zstd reading requires `pip install zstandard`.
    _COMPRESSION_MAP: Dict[str, str] = {
        ".gz":  "gzip",
        ".bz2": "bz2",
        ".b2":  "bz2",
        ".xz":  "xz",
        ".zst": "zstd",
        ".zip": "zip",
    }

    # S3 URI schemes recognised by load_dataframe. s3a:// and s3n:// are
    # Hadoop/Spark conventions; functionally identical to s3:// for object
    # storage access — pyarrow only needs the bucket/key portion.
    _S3_SCHEMES: tuple = ("s3://", "s3a://", "s3n://")

    def __init__(self, name: str, role: str, logger: AgentLogger):
        self.name = name
        self.role = role
        self.logger = logger
        self.client = Config.get_client()

    @staticmethod
    def _load_prompt(path: str | Path, **kwargs) -> str:
        """Load a prompt template file and substitute <<VAR>> placeholders."""
        text = Path(path).read_text(encoding="utf-8")
        for key, value in kwargs.items():
            text = text.replace(f"<<{key}>>", str(value))
        return text

    def _choose_model(self, prompt: str) -> str:
        if len(prompt) > Config.MODEL_ROUTING_THRESHOLD:
            return Config.CLOUD_MODEL
        return Config.LOCAL_MODEL

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        return any(kw in str(exc).lower() for kw in ("connection", "refused", "unreachable", "timeout", "connect"))

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(kw in msg for kw in ("rate limit", "429", "500", "502", "503", "overloaded", "server error"))

    def _build_call_kwargs(self, model: str, system_prompt: str, prompt: str,
                           json_mode: bool, max_tokens: Optional[int] = None) -> Dict:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": Config.LLM_TEMPERATURE,
            "max_tokens": max_tokens if max_tokens is not None else Config.LLM_MAX_TOKENS,
            "top_p": Config.LLM_TOP_P,
            "frequency_penalty": Config.LLM_FREQUENCY_PENALTY,
            "presence_penalty": Config.LLM_PRESENCE_PENALTY,
            "timeout": Config.TIMEOUT,
        }
        if Config.LLM_SEED is not None:
            kwargs["seed"] = Config.LLM_SEED
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _log_token_usage(self, model: str, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        # OpenAI-style: prompt_tokens / completion_tokens
        if hasattr(usage, "prompt_tokens"):
            self.logger.log(self.name, "LLM Tokens",
                f"model={model} | prompt={usage.prompt_tokens} | "
                f"completion={usage.completion_tokens} | total={usage.total_tokens}")
            self.logger.record_tokens(model, usage.prompt_tokens, usage.completion_tokens)
        # Anthropic-style: input_tokens / output_tokens
        elif hasattr(usage, "input_tokens"):
            total = usage.input_tokens + usage.output_tokens
            self.logger.log(self.name, "LLM Tokens",
                f"model={model} | input={usage.input_tokens} | "
                f"output={usage.output_tokens} | total={total}")
            self.logger.record_tokens(model, usage.input_tokens, usage.output_tokens)

    def _call_via_proxy(self, model: str, system_prompt: str, prompt: str,
                        json_mode: bool = False, max_tokens: Optional[int] = None) -> str:
        kwargs = self._build_call_kwargs(model, system_prompt, prompt, json_mode, max_tokens)
        last_exc = None
        for attempt in range(1, Config.LLM_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                self._log_token_usage(model, response)
                return response.choices[0].message.content
            except Exception as e:
                if self._is_connection_error(e):
                    raise
                last_exc = e
                if attempt < Config.LLM_MAX_RETRIES and self._is_retryable_error(e):
                    wait = Config.LLM_RETRY_DELAY * (2 ** (attempt - 1))
                    self.logger.log(self.name, "LLM Retry",
                        f"Attempt {attempt}/{Config.LLM_MAX_RETRIES} — {e} — retrying in {wait:.1f}s")
                    time.sleep(wait)
                else:
                    raise
        raise last_exc

    def _call_direct_api(self, system_prompt: str, prompt: str,
                         json_mode: bool = False, max_tokens: Optional[int] = None) -> str:
        direct_client = Config.get_direct_client()
        if direct_client is None:
            raise RuntimeError(
                f"LiteLLM proxy at {Config.LITELLM_URL} is unreachable and OPENAI_API_KEY is not set.\n"
                f"To fix this, do ONE of the following:\n"
                f"  1. Start the LiteLLM proxy:  litellm --config config.yaml\n"
                f"  2. Add OPENAI_API_KEY=sk-... to your .env file for direct OpenAI fallback"
            )
        model = Config.OPENAI_DIRECT_MODEL
        kwargs = self._build_call_kwargs(model, system_prompt, prompt, json_mode, max_tokens)
        last_exc = None
        for attempt in range(1, Config.LLM_MAX_RETRIES + 1):
            try:
                response = direct_client.chat.completions.create(**kwargs)
                self._log_token_usage(model, response)
                result = response.choices[0].message.content
                self.logger.log(self.name, "LLM Response",
                    f"Direct API ({model}) | {len(result)} chars | fallback=direct")
                return result
            except Exception as e:
                last_exc = e
                if attempt < Config.LLM_MAX_RETRIES and self._is_retryable_error(e):
                    wait = Config.LLM_RETRY_DELAY * (2 ** (attempt - 1))
                    self.logger.log(self.name, "LLM Retry",
                        f"Direct API attempt {attempt}/{Config.LLM_MAX_RETRIES} — {e} — retrying in {wait:.1f}s")
                    time.sleep(wait)
                else:
                    raise
        raise last_exc

    def _call_claude_api(self, system_prompt: str, prompt: str,
                         json_mode: bool = False, max_tokens: Optional[int] = None) -> str:
        claude_client = Config.get_claude_client()
        if claude_client is None:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set or 'anthropic' package not installed. "
                "Run: pip install anthropic"
            )
        model = Config.CLAUDE_DIRECT_MODEL
        system = system_prompt
        if json_mode:
            system += "\n\nIMPORTANT: Respond with valid JSON only. No text outside the JSON object."

        effective_max_tokens = max_tokens if max_tokens is not None else Config.LLM_MAX_TOKENS
        last_exc = None
        skip_temperature = False
        for attempt in range(1, Config.LLM_MAX_RETRIES + 1):
            try:
                create_kwargs: Dict = {
                    "model": model,
                    "max_tokens": effective_max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if not skip_temperature:
                    create_kwargs["temperature"] = Config.LLM_TEMPERATURE
                response = claude_client.messages.create(**create_kwargs)
                text_blocks = [b for b in response.content if hasattr(b, "text")]
                if not text_blocks:
                    raise ValueError(f"No text block in response: {[type(b).__name__ for b in response.content]}")
                result = text_blocks[0].text
                self._log_token_usage(model, response)
                self.logger.log(self.name, "LLM Response",
                    f"Claude ({model}) | {len(result)} chars | fallback=claude")
                return result
            except Exception as e:
                # Claude 4 Opus+ has deprecated the temperature parameter
                if not skip_temperature and "temperature" in str(e) and "deprecated" in str(e):
                    skip_temperature = True
                    continue
                last_exc = e
                if attempt < Config.LLM_MAX_RETRIES and self._is_retryable_error(e):
                    wait = Config.LLM_RETRY_DELAY * (2 ** (attempt - 1))
                    self.logger.log(self.name, "LLM Retry",
                        f"Claude attempt {attempt}/{Config.LLM_MAX_RETRIES} — {e} — retrying in {wait:.1f}s")
                    time.sleep(wait)
                else:
                    raise
        raise last_exc

    def _call_via_gateway(self, prompt: str, system_prompt: str,
                          json_mode: bool = False,
                          max_tokens: Optional[int] = None) -> str:
        """Single-source call through the LiteLLM gateway (anthropic SDK).

        No application-level fallback — the gateway itself is expected to
        handle internal failover / multi-provider routing.
        """
        model = Config.choose_model(prompt)
        effective_max = max_tokens if max_tokens is not None else Config.LLM_MAX_TOKENS
        system = system_prompt
        if json_mode:
            system += "\n\nIMPORTANT: Respond with valid JSON only. No text outside the JSON object."

        self.logger.log(self.name, "LLM Call",
            f"model={model} | prompt_len={len(prompt)} chars | "
            f"temp={Config.LLM_TEMPERATURE} | max_tokens={effective_max} | "
            f"json_mode={json_mode} | backend=gateway")

        last_exc = None
        skip_temperature = False
        for attempt in range(1, Config.LLM_MAX_RETRIES + 1):
            try:
                create_kwargs: Dict[str, Any] = {
                    "model": model,
                    "max_tokens": effective_max,
                    "system": system,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if not skip_temperature:
                    create_kwargs["temperature"] = Config.LLM_TEMPERATURE
                response = self.client.messages.create(**create_kwargs)
                self._log_token_usage(model, response)
                text_blocks = [b for b in response.content if hasattr(b, "text")]
                if not text_blocks:
                    raise ValueError(f"No text block in response: {[type(b).__name__ for b in response.content]}")
                result = text_blocks[0].text
                self.logger.log(self.name, "LLM Response",
                    f"model={model} | {len(result)} chars | backend=gateway")
                return result
            except Exception as e:
                # Some Claude variants reject temperature — drop it and retry once
                if not skip_temperature and "temperature" in str(e) and "deprecated" in str(e):
                    skip_temperature = True
                    continue
                last_exc = e
                if attempt < Config.LLM_MAX_RETRIES and self._is_retryable_error(e):
                    wait = Config.LLM_RETRY_DELAY * (2 ** (attempt - 1))
                    self.logger.log(self.name, "LLM Retry",
                        f"Gateway attempt {attempt}/{Config.LLM_MAX_RETRIES} — "
                        f"{e} — retrying in {wait:.1f}s")
                    time.sleep(wait)
                else:
                    self.logger.log(self.name, "ERROR", f"Gateway call failed: {e}")
                    raise
        raise last_exc

    def call_llm(self, prompt: str, system_prompt: str, json_mode: bool = False,
                 max_tokens: Optional[int] = None) -> str:
        # ── Gateway backend: single source, no application-level fallback ─────
        if getattr(Config, "BACKEND", "legacy") == "gateway":
            return self._call_via_gateway(prompt, system_prompt, json_mode, max_tokens)

        # ── Legacy backend: 3-tier fallback ───────────────────────────────────
        model = self._choose_model(prompt)
        fallback_model = Config.CLOUD_MODEL if model == Config.LOCAL_MODEL else Config.LOCAL_MODEL
        effective_max = max_tokens if max_tokens is not None else Config.LLM_MAX_TOKENS
        self.logger.log(self.name, "LLM Call",
            f"model={model} | prompt_len={len(prompt)} chars | "
            f"temp={Config.LLM_TEMPERATURE} | top_p={Config.LLM_TOP_P} | "
            f"max_tokens={effective_max} | json_mode={json_mode}")

        # ── No proxy hosted: straight to the direct providers ─────────────────
        if getattr(Config, "LLM_SKIP_PROXY", False):
            if Config.OPENAI_API_KEY:
                try:
                    return self._call_direct_api(system_prompt, prompt, json_mode, max_tokens)
                except Exception as eo:
                    self.logger.log(self.name, "ERROR", f"OpenAI direct failed: {eo}")
            return self._call_claude_api(system_prompt, prompt, json_mode, max_tokens)

        # ── Primary attempt ───────────────────────────────────────────────────
        try:
            result = self._call_via_proxy(model, system_prompt, prompt, json_mode, max_tokens)
            self.logger.log(self.name, "LLM Response",
                f"model={model} | {len(result)} chars")
            return result
        except Exception as e:
            if self._is_connection_error(e):
                self.logger.log(self.name, "ERROR",
                    f"Cannot reach LiteLLM proxy at {Config.LITELLM_URL}: {e}")
                if Config.OPENAI_API_KEY:
                    try:
                        self.logger.log(self.name, "LLM Fallback",
                            f"Proxy down — trying OpenAI direct ({Config.OPENAI_DIRECT_MODEL})...")
                        return self._call_direct_api(system_prompt, prompt, json_mode, max_tokens)
                    except Exception as eo:
                        self.logger.log(self.name, "ERROR", f"OpenAI direct failed: {eo}")
                self.logger.log(self.name, "LLM Fallback",
                    f"Trying Claude ({Config.CLAUDE_DIRECT_MODEL}) as last resort...")
                return self._call_claude_api(system_prompt, prompt, json_mode, max_tokens)
            self.logger.log(self.name, "ERROR", f"Error calling {model}: {e}")

        # ── Fallback 1: swap to other proxy model ────────────────────────────
        try:
            self.logger.log(self.name, "LLM Fallback", f"Switching proxy model to {fallback_model}")
            result = self._call_via_proxy(fallback_model, system_prompt, prompt, json_mode, max_tokens)
            self.logger.log(self.name, "LLM Response",
                f"model={fallback_model} | {len(result)} chars | fallback=proxy")
            return result
        except Exception as e2:
            self.logger.log(self.name, "ERROR", f"Both proxy models failed: {e2}")

        # ── Fallback 2: OpenAI direct API ────────────────────────────────────
        if Config.OPENAI_API_KEY:
            try:
                self.logger.log(self.name, "LLM Fallback",
                    f"Trying OpenAI direct ({Config.OPENAI_DIRECT_MODEL})...")
                return self._call_direct_api(system_prompt, prompt, json_mode, max_tokens)
            except Exception as e3:
                self.logger.log(self.name, "ERROR", f"OpenAI direct API failed: {e3}")

        # ── Fallback 3: Claude (Anthropic) — last resort ─────────────────────
        self.logger.log(self.name, "LLM Fallback",
            f"Trying Claude ({Config.CLAUDE_DIRECT_MODEL}) as last resort...")
        return self._call_claude_api(system_prompt, prompt, json_mode, max_tokens)

    # ── Shared column helpers ─────────────────────────────────────────────────

    _TARGET_ALIASES = [
        "target", "label", "y", "class", "output",
        "default_flag", "fraud", "is_fraud", "churn", "bad_flag",
    ]
    # Boundary-aware patterns: matched against the column name in lower-case.
    # Substring lists like ("dt", "day", "month") used to pick up
    # `amt_dti`, `days_birth`, `months_balance` etc. — all of which are features,
    # not snapshot timestamps. We now require either an exact name match, an
    # `_date` / `_dt` / `_ts` suffix, or a snapshot-style token.
    _DATE_NAME_PATTERNS = (
        re.compile(r"^(date|dt|datetime|timestamp|ts|snap_dt|snapshot|"
                   r"observation_date|report_date|as_of|as_of_date|"
                   r"date_decision|process_date|month_id|year_month|"
                   r"period|month|cohort)$"),
        re.compile(r"_(date|dt|ts|timestamp|snap_dt|snap|month_id|year_month)$"),
    )

    @classmethod
    def _looks_like_date_name(cls, col: str) -> bool:
        col_l = col.lower()
        return any(p.search(col_l) for p in cls._DATE_NAME_PATTERNS)

    def _resolve_target_column(self, df, target_column: str) -> str:
        """Resolve target column: exact → case-insensitive → common aliases → raise."""
        if target_column in df.columns:
            return target_column
        lower_map = {c.lower(): c for c in df.columns}
        if target_column.lower() in lower_map:
            resolved = lower_map[target_column.lower()]
            self.logger.log(self.name, "Target Resolved",
                f"'{target_column}' → '{resolved}' (case-insensitive match)")
            return resolved
        for alias in self._TARGET_ALIASES:
            if alias in lower_map:
                resolved = lower_map[alias]
                self.logger.log(self.name, "Target Resolved",
                    f"'{target_column}' not found → using '{resolved}' (common alias)")
                return resolved
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    def _auto_detect_date_col(self, df) -> Optional[str]:
        """Pick the first genuinely temporal column.

        Hardened against false positives that caused the cascading "Must have
        at least 1 validation dataset for early stopping" failure when a
        feature column (e.g. days_birth, years_employed, amt_dti) was
        misclassified as the snapshot date column:

          1. Name must match a boundary-aware date pattern (not just "contains
             the substring 'dt'").
          2. Numeric dtypes are REJECTED — pd.to_datetime on int/float
             interprets values as nanoseconds-since-epoch and silently parses
             every integer to a near-epoch timestamp. Real date columns are
             either datetime64 already, or strings that need parsing.
          3. Parsed values must span ≥ 30 days. All-near-epoch clusters
             (the failure mode for int-as-ns parsing) collapse to one period,
             which would then trip the temporal-split fallback.

        Caller should pass `date_col=` explicitly if the snapshot column is
        encoded as a raw integer (e.g. days_from_anchor).
        """
        for col in df.columns:
            if not self._looks_like_date_name(col):
                continue
            s = df[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                return col
            # Reject numeric columns — int/float as date is almost always a feature
            if pd.api.types.is_numeric_dtype(s):
                continue
            try:
                parsed = pd.to_datetime(s, errors="coerce")
            except Exception:
                continue
            if parsed.notna().mean() <= 0.9:
                continue
            # Sanity: real snapshot columns span days/months, not a single instant
            try:
                span_days = (parsed.max() - parsed.min()).days
            except Exception:
                span_days = 0
            if span_days < 30:
                continue
            return col
        return None

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip markdown fences; fall back to regex extraction of first {...} block."""
        text = text.strip()
        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        if "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        # Fallback: LLM returned prose before/after the JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return text

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        self.logger.log(self.name, f"Tool: {tool_name}", f"Parameters: {kwargs}")
        try:
            tool_method = getattr(self, f"_tool_{tool_name}")
            result = tool_method(**kwargs)
            preview = str(result)[:Config.LOG_RESULT_PREVIEW_CHARS] if result is not None else "None"
            self.logger.log(self.name, f"Tool Result: {tool_name}", f"Success: {preview}")
            return result
        except Exception as e:
            self.logger.log(self.name, "ERROR", f"Error executing tool {tool_name}: {e}")
            raise

    @classmethod
    def _split_compression(cls, path: str):
        """Return (inner_suffix, compression) tuple.

        'train.csv.gz'   → ('.csv', 'gzip')
        'train.json.zst' → ('.json', 'zstd')
        'train.parquet'  → ('.parquet', None)
        'train.gz'       → ('', 'gzip')
        """
        from pathlib import PurePosixPath
        p = PurePosixPath(path)
        outer = p.suffix.lower()
        compression = cls._COMPRESSION_MAP.get(outer)
        if compression is None:
            return outer, None
        inner = PurePosixPath(p.stem).suffix.lower()
        return inner, compression

    # Excel magic-byte signatures — used by _read_excel_smart to route the
    # file to the correct engine when the extension doesn't match the content.
    _EXCEL_MAGIC_ZIP:  bytes = b"PK\x03\x04"                          # .xlsx / .xlsm (Open XML zip)
    _EXCEL_MAGIC_OLE2: bytes = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"   # .xls (OLE2 compound)

    @classmethod
    def _read_excel_smart(cls, path: str, **kwargs) -> pd.DataFrame:
        """pd.read_excel with the engine chosen from file magic bytes.

        Pandas picks its Excel engine based purely on extension, which fails
        when the file's real format doesn't match — the common case is a
        true .xls (OLE2 binary) saved with a .xlsx extension, which yields
        the cryptic ``Can't find workbook in OLE2 compound document`` error
        because xlrd is asked to parse a non-existent workbook stream.

        We sniff the first 8 bytes:
          PK\\x03\\x04           → ZIP / Open XML  → openpyxl
          D0 CF 11 E0 A1 B1 1A E1 → OLE2 compound  → xlrd

        If detection is ambiguous we fall back to pandas' default. When the
        chosen engine isn't installed we retry the other one and finally
        raise a self-describing error that points at the missing dependency.
        """
        try:
            with open(path, "rb") as f:
                head = f.read(8)
        except Exception:
            # Path doesn't exist locally (S3, etc.) — defer to pandas
            return pd.read_excel(path, **kwargs)

        if head.startswith(cls._EXCEL_MAGIC_ZIP):
            engine_order = ("openpyxl", "xlrd")
            detected = "ZIP (Open XML / .xlsx)"
        elif head.startswith(cls._EXCEL_MAGIC_OLE2):
            engine_order = ("xlrd", "openpyxl")
            detected = "OLE2 (.xls binary)"
        else:
            # Magic doesn't match either Excel container. Don't bother trying
            # an engine — pandas would raise an opaque "Excel file format
            # cannot be determined" error. Tell the user what's actually wrong.
            raise RuntimeError(
                f"File '{path}' has an Excel extension but its content is not a valid "
                f"Excel file (first 4 bytes: {head[:4].hex()!r}). "
                "Expected ZIP signature '504b0304' (.xlsx/.xlsm) or OLE2 signature "
                "'d0cf11e0' (.xls). The file may be corrupted, encrypted, or a "
                "different format (CSV / HTML / JSON) saved with an .xlsx extension."
            )

        last_exc: Optional[Exception] = None
        for engine in engine_order:
            try:
                return pd.read_excel(path, engine=engine, **kwargs)
            except ImportError as e:
                # engine module not installed → try the next one
                last_exc = e
                continue
            except Exception as e:
                # The format is right but the file failed for another reason
                # (corrupt, encrypted, wrong workbook stream). No engine swap
                # will fix that, so re-raise with helpful context.
                raise RuntimeError(
                    f"Could not read Excel file '{path}' with engine='{engine}'. "
                    f"Detected format: {detected}. Underlying error: {e}. "
                    "If the file is password-protected, decrypt it first. "
                    "If you renamed an .xls file to .xlsx (or vice-versa), restore the original extension."
                ) from e
        raise RuntimeError(
            f"Could not read Excel file '{path}' — detected format: {detected} "
            f"but neither '{engine_order[0]}' nor '{engine_order[1]}' is installed. "
            f"Run: pip install openpyxl xlrd"
        ) from last_exc

    @classmethod
    def load_dataframe(cls, path: str) -> pd.DataFrame:
        """Load a DataFrame and normalise it to a reader-independent form.

        See `normalize_loaded_frame` for why the normalisation exists. Every
        agent and the pipeline load through here, so a frame can never reach
        feature engineering still carrying a reader's private dtype choices.
        """
        return cls.normalize_loaded_frame(cls._load_dataframe_raw(path))

    @staticmethod
    def normalize_loaded_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Erase the differences between file readers, in place where possible.

        The same table read as CSV, parquet, Excel or from a database arrives
        with different dtypes, and those differences change the MODEL — not
        just the representation:

          * An integer column containing nulls is float64 from CSV/parquet but
            Int64 from a DB driver or the pyarrow backend. `astype(str)` then
            yields "1.0" vs "1", so the label encoder builds different classes
            and `get_dummies` emits differently-named columns.
          * Nulls stringify as "nan" (numpy object), "None" (after a parquet
            round-trip) or "<NA>" (nullable string dtype) — three spellings of
            one thing.
          * A date is datetime64 from parquet/Excel but plain text from CSV.
            Downstream that is the difference between a feature holding epoch
            nanoseconds and one holding a string category.
          * A SQL NUMERIC arrives as decimal.Decimal objects in an object
            column, which is neither numeric nor categorical to pandas.

        Normalising at the boundary means the rest of the pipeline sees one
        canonical frame and no component has to defend itself. The choices here
        follow what `pd.read_csv` produces, because CSV is the format with the
        least metadata — anything richer can be reduced to it, not the reverse.
        """
        from decimal import Decimal

        for col in df.columns:
            s = df[col]
            dt = s.dtype

            # pandas nullable extension dtypes -> numpy equivalents + np.nan
            if isinstance(dt, pd.api.extensions.ExtensionDtype) and not isinstance(
                    dt, pd.CategoricalDtype):
                kind = getattr(dt, "kind", None)
                if kind in ("i", "u", "f"):
                    df[col] = pd.to_numeric(s, errors="coerce").astype("float64")
                    continue
                if kind == "b":
                    df[col] = s.astype("object").where(s.notna(), np.nan)
                    continue
                # string[python] / string[pyarrow] and anything else -> object
                df[col] = s.astype("object").where(s.notna(), np.nan)
                continue

            # Categorical -> its plain values; the codes are a storage detail
            # and would otherwise encode differently than the same data as text.
            # A categorical OF NUMBERS has to land back on a numeric dtype, or
            # it stays object and gets label-encoded downstream while the same
            # column from CSV is treated as numeric.
            if isinstance(dt, pd.CategoricalDtype):
                plain = s.astype("object").where(s.notna(), np.nan)
                if pd.api.types.is_numeric_dtype(dt.categories):
                    plain = pd.to_numeric(plain, errors="coerce")
                    if plain.notna().all() and (plain % 1 == 0).all():
                        plain = plain.astype("int64")
                df[col] = plain
                continue

            # Datetime -> ISO text, which is what CSV would have given us.
            # Everything that needs real timestamps (the splitter, the temporal
            # checks, the stability step) calls pd.to_datetime explicitly.
            if pd.api.types.is_datetime64_any_dtype(dt):
                # Date-only when every timestamp is midnight, matching what
                # `to_csv` writes for the same column. Otherwise the CSV and
                # parquet copies of one table would still disagree here.
                midnight_only = bool(
                    ((s.dt.hour == 0) & (s.dt.minute == 0)
                     & (s.dt.second == 0) & (s.dt.microsecond == 0)
                     & (s.dt.nanosecond == 0)).all()
                )
                fmt = "%Y-%m-%d" if midnight_only else "%Y-%m-%d %H:%M:%S"
                df[col] = s.dt.strftime(fmt).where(s.notna(), np.nan)
                continue

            # Decimal objects from SQL drivers -> float64
            if dt == object:
                first = next((v for v in s.head(100) if v is not None and v is not pd.NaT
                              and not (isinstance(v, float) and np.isnan(v))), None)
                if isinstance(first, Decimal):
                    df[col] = pd.to_numeric(s, errors="coerce").astype("float64")

        return df

    @classmethod
    def _load_dataframe_raw(cls, path: str) -> pd.DataFrame:
        """Load a DataFrame from a local or remote path.

        Supported formats   : .csv, .tsv, .parquet, .orc, .feather/.ftr,
                              .xls, .xlsx, .xlsm, .json
        Compression suffixes: .gz, .bz2 (alias .b2), .xz, .zst, .zip
                              (e.g. train.csv.gz, data.json.zst) — auto-decoded.
                              zstd requires `pip install zstandard`.
        S3 / S3-compatible  : s3://bucket/key, s3a://..., s3n://... — uses
                              pyarrow with credentials from Config
                              (S3_ACCESS_KEY / S3_SECRET_KEY / S3_ENDPOINT_URL).
                              Supports partitioned datasets (point at a folder
                              for parquet/feather/orc).
        Other remote        : gs://..., az://...  (pandas + gcsfs/adlfs)
        CSV encoding        : auto-tries utf-8 → utf-8-sig → cp1252 → latin-1
        """
        suffix, compression = cls._split_compression(path)

        if path.startswith(cls._S3_SCHEMES):
            return cls._load_from_s3(path, suffix, compression)

        # Uncompressed columnar / binary formats
        if compression is None:
            if suffix == ".parquet":
                return pd.read_parquet(path)
            if suffix == ".orc":
                return pd.read_orc(path)
            if suffix in (".feather", ".ftr"):
                return pd.read_feather(path)
            if suffix in (".xls", ".xlsx", ".xlsm"):
                return cls._read_excel_smart(path)

        read_kwargs: Dict[str, Any] = {}
        if compression is not None:
            read_kwargs["compression"] = compression

        if suffix == ".json":
            return pd.read_json(path, **read_kwargs)

        # CSV / TSV (default for unknown extensions and bare .gz/.bz2/... files too)
        sep = "\t" if suffix == ".tsv" else ","
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return pd.read_csv(path, encoding=enc, sep=sep, **read_kwargs)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, encoding="latin-1", sep=sep, **read_kwargs)

    @classmethod
    def _get_s3_filesystem(cls):
        """Build pyarrow.fs.S3FileSystem from Config — cached after first call.

        Empty credentials fall back to pyarrow's default chain
        (env vars → ~/.aws/credentials → IAM role).
        """
        if cls._S3_FS is not None:
            return cls._S3_FS
        try:
            import pyarrow.fs as fs
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for s3:// paths. Run: pip install pyarrow"
            ) from e
        kwargs: Dict[str, Any] = {
            "connect_timeout": Config.S3_CONNECT_TIMEOUT,
            "request_timeout": Config.S3_REQUEST_TIMEOUT,
        }
        if Config.S3_ACCESS_KEY:
            kwargs["access_key"] = Config.S3_ACCESS_KEY
        if Config.S3_SECRET_KEY:
            kwargs["secret_key"] = Config.S3_SECRET_KEY
        if Config.S3_ENDPOINT_URL:
            kwargs["endpoint_override"] = Config.S3_ENDPOINT_URL
        cls._S3_FS = fs.S3FileSystem(**kwargs)
        return cls._S3_FS

    @classmethod
    def _load_from_s3(cls, path: str, suffix: str,
                       compression: Optional[str] = None) -> pd.DataFrame:
        """Load DataFrame from s3:// / s3a:// / s3n:// path via pyarrow.

        Columnar formats (parquet/feather/orc) use pyarrow.dataset → supports
        partitioned folders (path may have no file extension, e.g. a folder
        ending with `/`). Row-oriented formats (json/excel/csv) stream the
        single object through pandas, decompressing on the fly when needed.
        """
        import pyarrow.dataset as ds
        s3 = cls._get_s3_filesystem()

        # Strip whichever scheme the caller used (s3://, s3a://, s3n://).
        # pyarrow expects bare 'bucket/key' without the scheme prefix.
        s3_path = path
        for scheme in cls._S3_SCHEMES:
            if path.startswith(scheme):
                s3_path = path[len(scheme):]
                break
        # Drop trailing slash so pyarrow.dataset treats it as a folder consistently
        s3_path = s3_path.rstrip("/")

        # Uncompressed columnar / partitioned-capable — single file OR folder both work
        if compression is None:
            if suffix in ("", ".parquet"):
                return ds.dataset(s3_path, filesystem=s3, format="parquet").to_table().to_pandas()
            if suffix in (".feather", ".ftr"):
                return ds.dataset(s3_path, filesystem=s3, format="feather").to_table().to_pandas()
            if suffix == ".orc":
                return ds.dataset(s3_path, filesystem=s3, format="orc").to_table().to_pandas()

        # Row-oriented / compressed — stream through pandas
        read_kwargs: Dict[str, Any] = {}
        if compression is not None:
            read_kwargs["compression"] = compression

        with s3.open_input_stream(s3_path) as f:
            if suffix == ".json":
                return pd.read_json(f, **read_kwargs)
            if suffix in (".xls", ".xlsx", ".xlsm"):
                return pd.read_excel(f)   # excel rarely wrapped in outer compression
            sep = "\t" if suffix == ".tsv" else ","
            return pd.read_csv(f, sep=sep, **read_kwargs)

    def save_report(self, report: Dict[str, Any], filename: str):
        report_path = Path(filename)
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, indent=2, fp=f)
        self.logger.log(self.name, "Report Saved", f"Saved to {filename}")


class ToolRegistry:

    def __init__(self):
        self.tools: Dict[str, Any] = {}

    def register(self, name: str, description: str, parameters: Dict[str, str]):
        self.tools[name] = {"description": description, "parameters": parameters}

    def get_tool_descriptions(self) -> str:
        descriptions = []
        for name, info in self.tools.items():
            desc = f"\n{name}:\n  {info['description']}\n  Parameters:"
            for param, param_desc in info["parameters"].items():
                desc += f"\n    - {param}: {param_desc}"
            descriptions.append(desc)
        return "\n".join(descriptions)
