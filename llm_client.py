"""
A tiny abstraction so agents don't call an API directly. This means:
  - anyone can run the whole pipeline with MockLLMClient, no API key needed
  - swapping in the real Grok client later is a one-line change in pipeline.py

RULE FOR THE TEAM: agents should only ever call self.llm.complete(...).
Never hardcode an API call inside an agent -- it breaks testing for everyone else.
"""

from __future__ import annotations
import json
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


class MockLLMClient:
    """Returns canned, deterministic JSON so the pipeline can run end-to-end
    with zero API cost while modules are being built and integrated.
    Replace with GrokClient (see real_llm_client.py) once you're ready to
    test with real model outputs."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if "synthesis" in system_prompt.lower():
            return json.dumps({
                "summary": "Draft synthesis of Evidence and Critical findings.",
                "supporting_claims": [],
            })
        if "Planner" in system_prompt:
            return json.dumps({
                "topic": "Compare GPT-4 and Gemini",
                "constraints": ["budget: 50000", "min_papers: 5"],
                "subtask_evidence": "Find papers supporting a comparison of GPT-4 and Gemini.",
                "subtask_critical": "Find papers that challenge or complicate that comparison.",
            })
        # NOTE: match on how the prompt OPENS, not just whether a role name
        # appears anywhere in it. Each agent's prompt mentions the OTHER
        # agent's name too (e.g. Evidence's prompt says "...have not seen
        # the Critical agent's output"), so a plain substring check on either
        # name matches both prompts and makes the two agents always agree.
        if system_prompt.startswith("You are the Critical agent"):
            return json.dumps({
                "claims": [
                    {"claim": "Paper X actually reports no significant difference between the two models.", "supporting_reference": "Paper X"}
                ]
            })
        if system_prompt.startswith("You are the Evidence agent"):
            return json.dumps({
                "claims": [
                    {"claim": "Paper X shows GPT-4 outperforms Gemini on reasoning benchmarks.", "supporting_reference": "Paper X"}
                ]
            })
        return json.dumps({})


class RealGrokClient:
    """Fill this in when you're ready to test against the real model.
    Kept separate from MockLLMClient so nobody accidentally burns API
    budget while debugging pipeline wiring."""

    def __init__(self, api_key: str, model: str = "grok-beta"):
        self.api_key = api_key
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError(
            "Plug in the actual Grok API call here. Keep the return type a "
            "JSON string matching whatever schema the caller expects."
        )
