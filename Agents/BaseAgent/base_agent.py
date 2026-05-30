import re
import time
from typing import Any, Dict, Optional
import pandas as pd
from logger import AgentLogger
from config import Config
import json
from pathlib import Path


class BaseAgent:

    # Root folder for all agent prompt templates (Agents/BaseAgent/ → Agents/)
    _PROMPTS_ROOT: Path = Path(__file__).parent.parent

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
        # Anthropic-style: input_tokens / output_tokens
        elif hasattr(usage, "input_tokens"):
            total = usage.input_tokens + usage.output_tokens
            self.logger.log(self.name, "LLM Tokens",
                f"model={model} | input={usage.input_tokens} | "
                f"output={usage.output_tokens} | total={total}")

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
                result = response.content[0].text
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

    def call_llm(self, prompt: str, system_prompt: str, json_mode: bool = False,
                 max_tokens: Optional[int] = None) -> str:
        model = self._choose_model(prompt)
        fallback_model = Config.CLOUD_MODEL if model == Config.LOCAL_MODEL else Config.LOCAL_MODEL
        effective_max = max_tokens if max_tokens is not None else Config.LLM_MAX_TOKENS
        self.logger.log(self.name, "LLM Call",
            f"model={model} | prompt_len={len(prompt)} chars | "
            f"temp={Config.LLM_TEMPERATURE} | top_p={Config.LLM_TOP_P} | "
            f"max_tokens={effective_max} | json_mode={json_mode}")

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
    _DATE_KEYWORDS = ("date", "dt", "time", "month", "snap", "period", "year", "week", "day")

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
        """Return the first column whose name contains a date keyword and is parseable as datetime."""
        for col in df.columns:
            if any(kw in col.lower() for kw in self._DATE_KEYWORDS):
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    return col
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().mean() > 0.9:
                        return col
                except Exception:
                    continue
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

    @staticmethod
    def load_dataframe(path: str) -> pd.DataFrame:
        """Load a DataFrame from a local or S3 path.

        Supported formats  : .csv, .tsv, .parquet, .orc, .feather/.ftr,
                             .xls, .xlsx, .xlsm, .json
        Remote storage     : s3://bucket/key  (requires: pip install s3fs)
                             gs://bucket/key  (requires: pip install gcsfs)
                             az://...         (requires: pip install adlfs)
        CSV encoding       : auto-tries utf-8 → utf-8-sig → cp1252 → latin-1
        """
        from pathlib import PurePosixPath
        suffix = PurePosixPath(path).suffix.lower()

        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".orc":
            return pd.read_orc(path)
        if suffix in (".feather", ".ftr"):
            return pd.read_feather(path)
        if suffix in (".xls", ".xlsx", ".xlsm"):
            return pd.read_excel(path)
        if suffix == ".json":
            return pd.read_json(path)
        # CSV / TSV (default for unknown extensions too)
        sep = "\t" if suffix == ".tsv" else ","
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return pd.read_csv(path, encoding=enc, sep=sep)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, encoding="latin-1", sep=sep)

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
