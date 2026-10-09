#!/usr/bin/env python
"""On-prem LLM readiness: reasoning-model outputs, request shaping, no public-API egress,
and repo portability (no machine paths, case-exact imports for Linux).

Run:  py -3.12 tests/test_llm_onprem.py
"""
import ast
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from logger import AgentLogger
from Agents.BaseAgent.base_agent import BaseAgent

ANSWERS = {
    "qwen3 think with braces": '<think>The user wants {"actions": ...}. Think {x}.</think>\n{"reasoning": "ok", "actions": [{"action": "drop_column", "column": "a"}]}',
    "deepseek-r1 no opening tag": 'Okay, drop {col}.\n</think>\n\n{"reasoning": "ok", "actions": []}',
    "glm box tokens": '<|begin_of_box|>{"reasoning": "ok", "actions": []}<|end_of_box|>',
    "fenced": 'Sure:\n```json\n{"reasoning": "ok", "actions": []}\n```',
    "prose + trailing brace": 'Plan: {"reasoning": "ok", "actions": []} Hope this helps {:)}',
    "nested object": '{"features": {"a": {"missing_type": "unknown"}}}',
}


def test_json_extraction_reasoning_models():
    for name, text in ANSWERS.items():
        obj = json.loads(BaseAgent._extract_json(text))
        assert isinstance(obj, dict) and obj, name


class _Resp:
    def __init__(self, content=None, reasoning=None):
        msg = SimpleNamespace(content=content, reasoning_content=reasoning)
        self.choices = [SimpleNamespace(message=msg)]
        self.usage = None


class _Client:
    def __init__(self, behaviour):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.behaviour = behaviour

    def _create(self, **kw):
        self.calls.append(kw)
        return self.behaviour(kw)


def _agent(client):
    ag = BaseAgent.__new__(BaseAgent)
    ag.name, ag.role, ag.logger, ag.client = "T", "t", AgentLogger(), client
    return ag


def _with(**kv):
    old = {k: getattr(Config, k) for k in kv}
    for k, v in kv.items():
        setattr(Config, k, v)
    return old


def test_request_shaping_for_onprem_servers():
    old = _with(LLM_OMIT_PARAMS=("frequency_penalty", "presence_penalty", "seed"),
                LLM_EXTRA_BODY={"chat_template_kwargs": {"enable_thinking": False}},
                LLM_JSON_MODE_NATIVE=False, LLM_SEED=7)
    try:
        c = _Client(lambda kw: _Resp('{"ok": true}'))
        out = _agent(c).call_llm("p", "sys", json_mode=True)
        kw = c.calls[0]
        assert out == '{"ok": true}'
        assert "frequency_penalty" not in kw and "presence_penalty" not in kw and "seed" not in kw
        assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
        assert "response_format" not in kw and "valid JSON only" in kw["messages"][0]["content"]
    finally:
        _with(**old)


def test_reasoning_stripped_and_reasoning_content_used():
    c = _Client(lambda kw: _Resp("<think>long {thoughts}</think>\n{\"a\": 1}"))
    assert _agent(c).call_llm("p", "s") == '{"a": 1}'
    c2 = _Client(lambda kw: _Resp(content="", reasoning='{"a": 2}'))
    assert _agent(c2).call_llm("p", "s") == '{"a": 2}'


def test_no_public_api_when_external_fallback_disabled():
    old = _with(LLM_ALLOW_EXTERNAL_FALLBACK=False, LLM_RETRY_DELAY=0.0, LLM_MAX_RETRIES=3,
                OPENAI_API_KEY="sk-test-not-used", ANTHROPIC_API_KEY="x", LLM_SKIP_PROXY=False)
    try:
        def down(kw):
            raise Exception("Connection error.")
        c = _Client(down)
        ag = _agent(c)
        ag._call_direct_api = lambda *a, **k: (_ for _ in ()).throw(AssertionError("OpenAI direct called"))
        ag._call_claude_api = lambda *a, **k: (_ for _ in ()).throw(AssertionError("Claude direct called"))
        try:
            ag.call_llm("p", "s")
            raise AssertionError("must fail")
        except RuntimeError as e:
            assert "LLM_ALLOW_EXTERNAL_FALLBACK=false" in str(e)
        assert len(c.calls) == 2 * Config.LLM_MAX_RETRIES          # retried on both proxy models
        # a transient connection error is now retried instead of escaping to a public API
        flaky = {"n": 0}

        def once_down(kw):
            flaky["n"] += 1
            if flaky["n"] == 1:
                raise Exception("Connection error.")
            return _Resp('{"ok": 1}')
        assert _agent(_Client(once_down)).call_llm("p", "s") == '{"ok": 1}'
        # skip-proxy + no external providers is a configuration error, not a silent call
        Config.LLM_SKIP_PROXY = True
        try:
            _agent(_Client(down)).call_llm("p", "s")
            raise AssertionError("must refuse")
        except RuntimeError as e:
            assert "LLM_SKIP_PROXY" in str(e)
    finally:
        _with(**old)


def test_no_machine_specific_paths_in_code():
    bad = re.compile(r"[A-Za-z]:\\\\(Users|Ngoc)|[A-Za-z]:/(Users|Ngoc)|DATANEST|multi-agent-auto-ml-v2")
    skip = {"outputs", ".git", "__pycache__", ".venv", "venv", ".claude", ".vscode"}
    hits = []
    for p in ROOT.rglob("*"):
        if p.suffix not in (".py", ".ps1", ".yaml", ".yml", ".txt", ".example", ".toml") or \
                any(s in p.parts for s in skip) or p.name in ("config_backup.py", "test_llm_onprem.py"):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if bad.search(line):
                hits.append(f"{p.relative_to(ROOT)}:{i}")
    assert not hits, hits


def test_output_dir_anchored_to_repo():
    assert Path(Config.OUTPUT_DIR).is_absolute()
    assert Path(Config.OUTPUT_DIR).resolve().parent == ROOT or os.environ.get("OUTPUT_DIR")


def test_imports_and_prompt_paths_are_case_exact():
    """Windows ignores case, Linux does not: every local import / prompt path must match on disk."""
    def exact(path: Path) -> bool:
        cur = ROOT
        for part in path.relative_to(ROOT).parts:
            if part not in os.listdir(cur):
                return False
            cur = cur / part
        return True

    problems = []
    for p in list(ROOT.glob("*.py")) + list((ROOT / "Agents").rglob("*.py")) + list((ROOT / "preprocessing").glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 and n.module.split(".")[0] in ("Agents", "preprocessing"):
                parts = n.module.split(".")
                target = ROOT.joinpath(*parts[:-1], parts[-1] + ".py")
                if not target.exists():
                    target = ROOT.joinpath(*parts, "__init__.py")
                if not exact(target):
                    problems.append(f"{p.name}: {n.module}")
    for prompt in (ROOT / "Agents").rglob("prompts/*.txt"):
        assert exact(prompt)
    for rel in ("DataCleaner/prompts/system.txt", "DataCleaner/prompts/user.txt",
                "FeatureEngineer/prompts/system.txt", "FeatureEngineer/prompts/user.txt",
                "TrainModel/prompts/llm_summary_user.txt", "TrainModel/prompts/llm_summary_system.txt"):
        if not exact(ROOT / "Agents" / rel):
            problems.append(rel)
    assert not problems, problems


if __name__ == "__main__":
    for t in (test_json_extraction_reasoning_models, test_request_shaping_for_onprem_servers,
              test_reasoning_stripped_and_reasoning_content_used,
              test_no_public_api_when_external_fallback_disabled,
              test_no_machine_specific_paths_in_code, test_output_dir_anchored_to_repo,
              test_imports_and_prompt_paths_are_case_exact):
        t()
        print("PASS ", t.__name__)
    print("\nALL ON-PREM CHECKS PASSED")
