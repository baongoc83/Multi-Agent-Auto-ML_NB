"""Gateway backend demo — minh hoạ flow gọi LLM qua LiteLLM gateway.

Có 2 chế độ:

1. DRY-RUN mode (mặc định) — không gọi LLM thật.
   Chỉ load config + show routing decisions cho prompt ngắn/dài/medium.
   Chạy được mà không cần gateway đang chạy.

      python examples/gateway_demo.py

2. REAL mode — gọi LLM thật qua gateway.
   Set GATEWAY_DEMO_REAL=1 và đảm bảo .env có
   ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL trỏ tới gateway sống.

      $env:GATEWAY_DEMO_REAL = "1"
      python examples/gateway_demo.py

Khi chạy mode 2, script gọi 2 prompt (ngắn + dài) qua chính hàm
BaseAgent.call_llm, log model nào được chọn, response length, token
usage — y hệt cách 3 agent thật dùng gateway.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows terminals defaulting to cp1252
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Buộc backend = gateway trước khi import Config
# (sẽ chỉ active nếu đã có ANTHROPIC_* env vars)
os.environ.setdefault("LLM_BACKEND", "gateway")

# Bảo đảm import được từ project root khi chạy script từ examples/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import Config
from logger import AgentLogger
from Agents.BaseAgent.base_agent import BaseAgent


def print_header(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def show_config() -> None:
    print_header("1. Config đang dùng")
    print(f"  BACKEND                       : {Config.BACKEND}")
    print(f"  ANTHROPIC_BASE_URL            : {Config.ANTHROPIC_BASE_URL}")
    auth = Config.ANTHROPIC_AUTH_TOKEN
    masked = (auth[:6] + "..." + auth[-4:]) if len(auth) > 12 else ("(empty)" if not auth else "(set)")
    print(f"  ANTHROPIC_AUTH_TOKEN          : {masked}")
    print(f"  API_TIMEOUT_MS                : {Config.API_TIMEOUT_MS} (~{Config.API_TIMEOUT_MS/1000:.0f}s)")
    print(f"  EFFORT_LEVEL                  : {Config.EFFORT_LEVEL}")
    print(f"  MODEL_ROUTING_THRESHOLD       : {Config.MODEL_ROUTING_THRESHOLD} chars")
    print(f"  Haiku  alias                  : {Config.ANTHROPIC_DEFAULT_HAIKU_MODEL}")
    print(f"  Sonnet alias                  : {Config.ANTHROPIC_DEFAULT_SONNET_MODEL}")
    print(f"  Opus   alias                  : {Config.ANTHROPIC_DEFAULT_OPUS_MODEL}")


def show_routing_matrix() -> None:
    print_header("2. Routing decisions — ma trận EFFORT x SIZE")

    samples = [
        ("short",  "Cleaning task: 2 outliers in col_x" + "."),                       # ~40 chars
        ("medium", "metadata=" + ("x" * 3000)),                                       # ~3010 chars
        ("long",   "metadata=" + ("x" * 12000)),                                      # ~12010 chars
    ]

    print(f"\n  Effort = {Config.EFFORT_LEVEL!r}  (threshold = {Config.MODEL_ROUTING_THRESHOLD} chars)")
    print(f"  {'Label':<8}  {'Length':>8}    {'Selected model'}")
    print(f"  {'-'*8}  {'-'*8}    {'-'*40}")
    for label, prompt in samples:
        model = Config.choose_model(prompt)
        print(f"  {label:<8}  {len(prompt):>8}    {model}")


def show_dry_run_call_flow() -> None:
    print_header("3. Dry-run — mô phỏng bước gọi (không connect gateway)")
    prompt = "Hello, classify this column as numeric or categorical: AGE_YEARS"
    system_prompt = "You are a data quality auditor. Reply with one word."
    model = Config.choose_model(prompt)
    print(f"\n  Prompt        : {prompt!r}")
    print(f"  Prompt length : {len(prompt)} chars")
    print(f"  System prompt : {system_prompt!r}")
    print(f"  Chosen model  : {model}")
    print(f"  Endpoint      : POST {Config.ANTHROPIC_BASE_URL}/v1/messages")
    print(f"\n  Payload that would be sent (anthropic SDK call):")
    print(f"    client.messages.create(")
    print(f"        model={model!r},")
    print(f"        max_tokens={Config.LLM_MAX_TOKENS},")
    print(f"        temperature={Config.LLM_TEMPERATURE},")
    print(f"        system={system_prompt!r},")
    print(f"        messages=[{{\"role\": \"user\", \"content\": {prompt!r}}}],")
    print(f"    )")


def run_real_call() -> None:
    print_header("4. REAL call — gọi LLM thật qua gateway")
    if not Config.ANTHROPIC_AUTH_TOKEN:
        print("\n  SKIPPED: ANTHROPIC_AUTH_TOKEN chưa set. Điền vào .env rồi chạy lại.")
        return

    logger = AgentLogger("outputs/gateway_demo.log")
    Path("outputs").mkdir(exist_ok=True)

    # Tạo agent minimal để dùng BaseAgent.call_llm
    class DemoAgent(BaseAgent):
        pass

    agent = DemoAgent(name="GatewayDemo", role="demo", logger=logger)

    test_cases = [
        ("SHORT prompt (HAIKU tier expected)",
         "Hello, this is a connectivity test. Reply with the word PONG."),
        ("LONG prompt (SONNET/OPUS tier expected)",
         "You are analysing a banking dataset. " + ("Column meta " * 600) +
         "Output: 'OK' if you received this message."),
    ]

    for label, prompt in test_cases:
        print(f"\n  --- {label} ---")
        print(f"  Prompt length : {len(prompt)} chars")
        print(f"  Chosen model  : {Config.choose_model(prompt)}")
        try:
            response = agent.call_llm(
                prompt=prompt,
                system_prompt="Be concise. Reply in 1-2 sentences.",
                max_tokens=200,
            )
            print(f"  Response      : {response.strip()[:200]}")
            print(f"  Response len  : {len(response)} chars")
        except Exception as exc:
            print(f"  ERROR         : {type(exc).__name__}: {exc}")


def main() -> None:
    show_config()
    show_routing_matrix()

    if os.getenv("GATEWAY_DEMO_REAL", "0") == "1":
        run_real_call()
    else:
        show_dry_run_call_flow()
        print(f"\n  [tip] Để gọi LLM thật: set GATEWAY_DEMO_REAL=1 rồi chạy lại.")

    print(f"\n{'='*70}\n  Demo finished.\n{'='*70}\n")


if __name__ == "__main__":
    main()
