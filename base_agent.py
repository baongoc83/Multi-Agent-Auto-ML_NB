from typing import Any, Dict
from logger import AgentLogger
from config import Config
import json
from pathlib import Path


class BaseAgent:

    def __init__(self, name: str, role: str, logger: AgentLogger):
        self.name = name
        self.role = role
        self.logger = logger
        self.client = Config.get_client()

    def _choose_model(self, prompt: str) -> str:
        if len(prompt) > Config.MODEL_ROUTING_THRESHOLD:
            return Config.CLOUD_MODEL
        return Config.LOCAL_MODEL

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        return any(kw in str(exc).lower() for kw in ("connection", "refused", "unreachable", "timeout", "connect"))

    def _call_via_proxy(self, model: str, system_prompt: str, prompt: str) -> str:
        response = self.client.chat.completions.create( 
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS,
            timeout=Config.TIMEOUT,
        )
        return response.choices[0].message.content

    def _call_direct_api(self, system_prompt: str, prompt: str) -> str:
        direct_client = Config.get_direct_client()
        if direct_client is None:
            raise RuntimeError(
                f"LiteLLM proxy at {Config.LITELLM_URL} is unreachable and OPENAI_API_KEY is not set.\n"
                f"To fix this, do ONE of the following:\n"
                f"  1. Start the LiteLLM proxy:  litellm --config config.yaml\n"
                f"  2. Add OPENAI_API_KEY=sk-... to your .env file for direct OpenAI fallback"
            )
        model = Config.OPENAI_DIRECT_MODEL
        response = direct_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS,
            timeout=Config.TIMEOUT,
        )
        result = response.choices[0].message.content
        self.logger.log(self.name, "LLM Response",
                        f"Direct API ({model}) | {len(result)} chars | fallback=direct")
        return result

    def call_llm(self, prompt: str, system_prompt: str) -> str:
        self.logger.log(self.name, "LLM Call", "Sending prompt to LLM...")
        model = self._choose_model(prompt)
        fallback_model = Config.CLOUD_MODEL if model == Config.LOCAL_MODEL else Config.LOCAL_MODEL

        # ── Primary attempt ───────────────────────────────────────────────────
        try:
            result = self._call_via_proxy(model, system_prompt, prompt)
            self.logger.log(self.name, "LLM Response", f"Received response ({len(result)} chars)")
            return result
        except Exception as e:
            if self._is_connection_error(e):
                self.logger.log(self.name, "ERROR",
                                f"Cannot reach LiteLLM proxy at {Config.LITELLM_URL}: {e}")
                self.logger.log(self.name, "LLM Fallback",
                                "Proxy unreachable — trying direct cloud API...")
                return self._call_direct_api(system_prompt, prompt)
            self.logger.log(self.name, "ERROR", f"Error calling {model}: {e}")

        # ── Secondary attempt: swap model via proxy ───────────────────────────
        try:
            self.logger.log(self.name, "LLM Fallback", f"Switching to {fallback_model}")
            result = self._call_via_proxy(fallback_model, system_prompt, prompt)
            self.logger.log(self.name, "LLM Response",
                            f"Model={fallback_model} | {len(result)} chars | fallback=True")
            return result
        except Exception as e2:
            self.logger.log(self.name, "ERROR", f"Both proxy models failed: {e2}")

        # ── Last resort: direct OpenAI API (bypasses proxy entirely) ─────────
        # Reaches here for any proxy failure: connection error, 401, 5xx, etc.
        self.logger.log(self.name, "LLM Fallback",
                        "Bypassing proxy — calling OpenAI API directly...")
        return self._call_direct_api(system_prompt, prompt)

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

    def save_report(self, report: Dict[str, Any], filename: str):
        report_path = Path(filename)
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w") as f:
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
