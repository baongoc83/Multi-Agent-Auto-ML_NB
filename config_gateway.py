"""Gateway-style LLM config — all calls routed through a single LiteLLM
gateway using the Anthropic API convention.

Enable by setting LLM_BACKEND=gateway in .env. When the dispatcher at the
bottom of config.py detects this, it re-exports GatewayConfig as `Config`
so all existing `from config import Config` imports transparently get the
gateway-aware class.

Co-exists with the original Config in config.py — non-LLM settings
(data cleaning thresholds, training params, paths, S3, ...) are
inherited unchanged. Only LLM-related class attributes and the
get_client / validate / choose_model methods are overridden.
"""
import os
from typing import Tuple

from config import Config as _LegacyConfig


class GatewayConfig(_LegacyConfig):
    """Anthropic-compatible gateway config.

    Routes every LLM call through ANTHROPIC_BASE_URL using the anthropic
    SDK with bearer auth (ANTHROPIC_AUTH_TOKEN). The gateway itself
    handles internal failover / model routing — the application does
    NOT fall back to direct provider APIs.
    """

    BACKEND: str = "gateway"

    # ── Gateway endpoint ─────────────────────────────────────────────────
    ANTHROPIC_AUTH_TOKEN: str = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
    ANTHROPIC_BASE_URL: str = os.getenv("ANTHROPIC_BASE_URL", "http://localhost:4000")
    # Per-request timeout in milliseconds (LiteLLM convention).
    # 3_000_000 ms = 50 min — covers very long agent reasoning steps.
    API_TIMEOUT_MS: int = int(os.getenv("API_TIMEOUT_MS", 3_000_000))

    # ── Model tier aliases (gateway dispatches to actual provider) ───────
    ANTHROPIC_DEFAULT_HAIKU_MODEL: str = os.getenv(
        "ANTHROPIC_DEFAULT_HAIKU_MODEL", "claude-haiku-4-5"
    )
    ANTHROPIC_DEFAULT_SONNET_MODEL: str = os.getenv(
        "ANTHROPIC_DEFAULT_SONNET_MODEL", "claude-sonnet-4-6"
    )
    ANTHROPIC_DEFAULT_OPUS_MODEL: str = os.getenv(
        "ANTHROPIC_DEFAULT_OPUS_MODEL", "claude-opus-4-7"
    )

    # ── Routing strategy ─────────────────────────────────────────────────
    # effort_level decides which tier handles which prompt size:
    #   low    → always Haiku
    #   medium → Haiku for short prompts, Sonnet for long  (default)
    #   high   → Sonnet for short prompts, Opus for long
    EFFORT_LEVEL: str = os.getenv("EFFORT_LEVEL", "medium").lower()

    # ── Overrides ────────────────────────────────────────────────────────

    @classmethod
    def validate(cls):
        """Ensure gateway endpoint + auth are configured."""
        if not cls.ANTHROPIC_AUTH_TOKEN:
            raise ValueError(
                "ANTHROPIC_AUTH_TOKEN is required for gateway backend. "
                "Set it in .env to authenticate with the LiteLLM gateway."
            )
        if not cls.ANTHROPIC_BASE_URL.startswith(("http://", "https://")):
            raise ValueError(
                f"Invalid ANTHROPIC_BASE_URL '{cls.ANTHROPIC_BASE_URL}'. "
                "Expected a URL starting with http:// or https://"
            )
        if cls.EFFORT_LEVEL not in ("low", "medium", "high"):
            raise ValueError(
                f"Invalid EFFORT_LEVEL '{cls.EFFORT_LEVEL}'. "
                "Expected one of: low, medium, high"
            )

    @classmethod
    def get_client(cls):
        """Return an anthropic.Anthropic client pointed at the gateway."""
        import anthropic
        cls.validate()
        return anthropic.Anthropic(
            auth_token=cls.ANTHROPIC_AUTH_TOKEN,
            base_url=cls.ANTHROPIC_BASE_URL,
            timeout=cls.API_TIMEOUT_MS / 1000.0,
        )

    @classmethod
    def choose_model(cls, prompt: str) -> str:
        """Pick model tier based on EFFORT_LEVEL + prompt size."""
        is_large = len(prompt) > cls.MODEL_ROUTING_THRESHOLD
        effort = cls.EFFORT_LEVEL
        if effort == "low":
            return cls.ANTHROPIC_DEFAULT_HAIKU_MODEL
        if effort == "high":
            return (cls.ANTHROPIC_DEFAULT_OPUS_MODEL
                    if is_large else cls.ANTHROPIC_DEFAULT_SONNET_MODEL)
        # medium (default)
        return (cls.ANTHROPIC_DEFAULT_SONNET_MODEL
                if is_large else cls.ANTHROPIC_DEFAULT_HAIKU_MODEL)

    @classmethod
    def is_reachable(cls) -> bool:
        """Best-effort health check on the gateway base URL."""
        import urllib.request
        try:
            urllib.request.urlopen(f"{cls.ANTHROPIC_BASE_URL}/health", timeout=3)
            return True
        except Exception:
            return False

    # ── Legacy methods disabled — gateway is single source ──────────────

    @classmethod
    def get_direct_client(cls):
        """Disabled in gateway mode — gateway is the single LLM source."""
        return None

    @classmethod
    def get_claude_client(cls):
        """Disabled in gateway mode — gateway is the single LLM source."""
        return None
