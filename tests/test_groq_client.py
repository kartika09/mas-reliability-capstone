"""
Tests for llm_client.GroqClient. The real `groq` package isn't called here
(no network, no API key needed) -- instead we monkeypatch groq.Groq itself
with a fake that matches the real SDK's shape (confirmed against the actual
installed `groq` package: client.chat.completions.create(..., messages=...,
response_format=...) -> an object with .choices[0].message.content).

Run with: pytest tests/test_groq_client.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest

import llm_client
from pipeline import run_pipeline


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class FakeGroqSDK:
    """Stands in for groq.Groq. Records every call so tests can inspect
    exactly what was sent, and returns a scripted response per call."""

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.calls = []
        self.chat = self  # groq.Groq exposes .chat.completions.create
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        responder = FakeGroqSDK._next_response
        return _FakeCompletion(responder(kwargs))

    # overridden per-test
    _next_response = staticmethod(lambda kwargs: "{}")


@pytest.fixture
def fake_groq(monkeypatch):
    """Patches the `groq` module's Groq class before GroqClient imports it."""
    import types
    fake_module = types.ModuleType("groq")
    fake_module.Groq = FakeGroqSDK
    monkeypatch.setitem(sys.modules, "groq", fake_module)
    yield FakeGroqSDK


def _make_client(fake_groq, response_fn):
    fake_groq._next_response = staticmethod(response_fn)
    client = llm_client.GroqClient(api_key="fake-key", model="llama-3.3-70b-versatile")
    return client


# ---------------------------------------------------------------------------
# Schema-instruction routing
# ---------------------------------------------------------------------------

def test_planner_prompt_gets_planner_schema_instruction(fake_groq):
    client = _make_client(fake_groq, lambda kwargs: json.dumps({
        "topic": "t", "constraints": [], "subtask_evidence": "e", "subtask_critical": "c",
    }))
    client.complete("You are the Planner agent in a research-analysis MAS.", "Research question: X")
    sent_system = client.client.calls[0]["messages"][0]["content"]
    assert "subtask_evidence" in sent_system  # planner schema instruction was appended


def test_synthesis_prompt_gets_synthesis_schema_instruction(fake_groq):
    client = _make_client(fake_groq, lambda kwargs: json.dumps({"summary": "s", "supporting_claims": []}))
    client.complete("You are the Planner agent performing final synthesis.", "...")
    sent_system = client.client.calls[0]["messages"][0]["content"]
    assert "supporting_claims" in sent_system


@pytest.mark.parametrize("prompt", [
    "You are the Evidence agent. Work independently -- you have not seen the Critical agent's output.",
    "You are the Critical agent. Work independently -- you have not seen the Evidence agent's output.",
])
def test_evidence_and_critical_prompts_get_claims_schema_instruction(fake_groq, prompt):
    client = _make_client(fake_groq, lambda kwargs: json.dumps({"claims": []}))
    client.complete(prompt, "Subtask: X")
    sent_system = client.client.calls[0]["messages"][0]["content"]
    assert '"claims"' in sent_system
    assert "passage_id" in sent_system  # citation-grounding instruction included


def test_unrecognized_prompt_is_sent_unmodified(fake_groq):
    client = _make_client(fake_groq, lambda kwargs: "{}")
    client.complete("You are some other agent.", "...")
    sent_system = client.client.calls[0]["messages"][0]["content"]
    assert sent_system == "You are some other agent."


# ---------------------------------------------------------------------------
# API call shape
# ---------------------------------------------------------------------------

def test_requests_json_object_response_format(fake_groq):
    client = _make_client(fake_groq, lambda kwargs: "{}")
    client.complete("You are the Planner agent in a research-analysis MAS.", "...")
    call = client.client.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["model"] == "llama-3.3-70b-versatile"
    assert call["messages"][1] == {"role": "user", "content": "..."}


def test_complete_returns_the_message_content_directly(fake_groq):
    client = _make_client(fake_groq, lambda kwargs: '{"claims": []}')
    result = client.complete("You are the Evidence agent.", "Subtask: X")
    assert result == '{"claims": []}'


def test_custom_temperature_and_seed_are_passed_through(fake_groq):
    fake_groq._next_response = staticmethod(lambda kwargs: "{}")
    client = llm_client.GroqClient(api_key="k", temperature=0.7, seed=42)
    client.complete("You are the Planner agent in a research-analysis MAS.", "...")
    call = client.client.calls[0]
    assert call["temperature"] == 0.7
    assert call["seed"] == 42


def test_reasoning_effort_defaults_to_low_and_is_sent(fake_groq):
    # gpt-oss-120b is a reasoning model -- reasoning tokens count against the
    # Groq daily quota, so this default matters for cost, not just behavior.
    client = _make_client(fake_groq, lambda kwargs: "{}")
    client.complete("You are the Planner agent in a research-analysis MAS.", "...")
    assert client.client.calls[0]["reasoning_effort"] == "low"


def test_reasoning_effort_can_be_overridden_via_constructor(fake_groq):
    fake_groq._next_response = staticmethod(lambda kwargs: "{}")
    client = llm_client.GroqClient(api_key="k", reasoning_effort="high")
    client.complete("You are the Planner agent in a research-analysis MAS.", "...")
    assert client.client.calls[0]["reasoning_effort"] == "high"


# ---------------------------------------------------------------------------
# Full pipeline run with GroqClient standing in for the LLM
# ---------------------------------------------------------------------------

def _scripted_pipeline_responses(kwargs):
    system = kwargs["messages"][0]["content"].lower()
    if "synthesis" in system:
        return json.dumps({"summary": "Final synthesized answer.", "supporting_claims": []})
    if "planner" in system:
        return json.dumps({
            "topic": "t", "constraints": [],
            "subtask_evidence": "find support", "subtask_critical": "find objections",
        })
    if "evidence agent" in system:
        return json.dumps({"claims": [{"claim": "Evidence claim.", "supporting_reference": "D1"}]})
    if "critical agent" in system:
        return json.dumps({"claims": [{"claim": "Critical claim.", "supporting_reference": "D2"}]})
    return "{}"


def test_full_pipeline_run_with_groq_client(fake_groq):
    client = _make_client(fake_groq, _scripted_pipeline_responses)
    report, result = run_pipeline("Some research question", condition="C_proposed", llm_client=client)
    assert report["summary"] == "Final synthesized answer."
    assert result.model == "llama-3.3-70b-versatile"
    # 4 calls: planner.decompose, evidence, critical, planner.synthesize
    assert len(client.client.calls) == 4