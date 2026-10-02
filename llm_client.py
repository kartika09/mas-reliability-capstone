"""
A tiny abstraction so agents don't call an API directly. This means:
  - anyone can run the whole pipeline with MockLLMClient, no API key needed
  - swapping in the real Grok client later is a one-line change in pipeline.py

RULE FOR THE TEAM: agents should only ever call self.llm.complete(...).
Never hardcode an API call inside an agent -- it breaks testing for everyone else.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional, Protocol


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


class GroqClient:
    """Real LLM client backed by Groq (https://groq.com), via its
    OpenAI-compatible chat completions API. Install with: pip install groq

    IMPORTANT: agents.py's prompts (e.g. "You are the Evidence agent...")
    were written for MockLLMClient, which pattern-matches on them and
    returns a hand-built dict -- they don't tell a real model what JSON
    shape to return. This client fixes that WITHOUT editing the shared
    agents.py: it detects which schema is expected from the same prompt
    text MockLLMClient already keys off of, and appends an explicit
    "return ONLY this JSON shape" instruction before calling the model.
    If agents.py's prompts change, update _SCHEMA_INSTRUCTIONS below to
    match -- the detection keys here must stay in sync with
    MockLLMClient's checks above.
    """

    # One instruction block per schema in schemas/models.py. Keep these in
    # sync with that file -- if a field is added there, add it here too, or
    # pipeline.py's `Model(**json.loads(raw))` call will raise a validation
    # error on real responses.
    _SCHEMA_INSTRUCTIONS = {
        "synthesis": (
            'Respond with ONLY a JSON object, no other text, matching exactly: '
            '{"summary": "<string>", "supporting_claims": ['
            '{"claim": "<string>", "supporting_reference": "<string or null>"}, ...]}'
        ),
        "planner": (
            'Respond with ONLY a JSON object, no other text, matching exactly: '
            '{"topic": "<string>", "constraints": ["<string>", ...], '
            '"subtask_evidence": "<string>", "subtask_critical": "<string>"}'
        ),
        "agent_claims": (
            'Respond with ONLY a JSON object, no other text, matching exactly: '
            '{"claims": [{"claim": "<string>", "supporting_reference": "<string or null>"}, ...]}. '
            "If context passages were given above, supporting_reference MUST be "
            "one of their passage_id values -- never invent one."
        ),
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ):
        from groq import Groq  # local import: only required when this client is actually used

        # Load .env (if present) so GROQ_API_KEY works without anyone setting
        # a shell variable by hand. Anchored to THIS file's own directory
        # (the repo root), not the current working directory -- running
        # `python pipeline.py` from a subfolder, or pytest from elsewhere,
        # must not silently miss the key. override=False means a shell
        # variable a teammate already set always wins over .env.
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
        except ImportError:
            pass  # python-dotenv not installed -- real env vars still work

        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and fill in "
                "your key, or set it directly: PowerShell: "
                "$env:GROQ_API_KEY = \"your-key-here\"."
            )
        self.client = Groq(api_key=api_key)
        self.model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        # openai/gpt-oss-120b is a reasoning model -- it spends extra tokens on
        # an internal chain of thought before answering, and that reasoning
        # counts against your Groq token quota even though it's not the JSON
        # output pipeline.py actually uses. Our tasks (plan/evidence/critical/
        # synthesis) are simple extraction, not deep multi-step reasoning, so
        # "low" cuts token usage substantially with no expected quality loss.
        # Override with GROQ_REASONING_EFFORT=medium (or "high") in .env if
        # you find low-effort answers are too shallow for your experiments.
        self.reasoning_effort = reasoning_effort or os.environ.get("GROQ_REASONING_EFFORT", "low")

    def _schema_instruction(self, system_prompt: str) -> str:
        lowered = system_prompt.lower()
        if "synthesis" in lowered:
            return self._SCHEMA_INSTRUCTIONS["synthesis"]
        if "planner" in lowered:
            return self._SCHEMA_INSTRUCTIONS["planner"]
        if "evidence agent" in lowered or "critical agent" in lowered:
            return self._SCHEMA_INSTRUCTIONS["agent_claims"]
        # Unknown caller -- return as-is rather than guess a schema (better to
        # fail loudly in pipeline.py's json.loads/Pydantic step than silently
        # send the wrong shape).
        return ""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        instruction = self._schema_instruction(system_prompt)
        full_system = f"{system_prompt}\n\n{instruction}" if instruction else system_prompt

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            reasoning_effort=self.reasoning_effort,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": full_system},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content