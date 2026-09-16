"""
The three agents. Each one:
  1. builds a prompt
  2. calls self.llm.complete(...)
  3. parses the JSON response into the agreed schema

Keep it this simple. Anything smarter (retries, better prompts) is welcome,
but the input/output shape must not change without a team discussion --
everything downstream (validator, peer checker, guard) depends on it.
"""

from __future__ import annotations
import json
from schemas.models import PlannerOutput, EvidenceOutput, CriticalOutput, FinalReport
from llm_client import LLMClient


class PlannerAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def decompose(self, research_question: str) -> PlannerOutput:
        system = "You are the Planner agent in a research-analysis MAS."
        user = f"Research question: {research_question}\nBreak this into subtasks."
        raw = self.llm.complete(system, user)
        return PlannerOutput(**json.loads(raw))

    def synthesize(self, evidence: EvidenceOutput, critical: CriticalOutput) -> FinalReport:
        system = "You are the Planner agent performing final synthesis."
        user = f"Evidence claims: {evidence.model_dump()}\nCritical claims: {critical.model_dump()}"
        raw = self.llm.complete(system, user)
        return FinalReport(**json.loads(raw))


class EvidenceAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def produce(self, subtask: str) -> EvidenceOutput:
        system = "You are the Evidence agent. Work independently -- you have not seen the Critical agent's output."
        user = f"Subtask: {subtask}"
        raw = self.llm.complete(system, user)
        return EvidenceOutput(**json.loads(raw))


class CriticalAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def produce(self, subtask: str) -> CriticalOutput:
        system = "You are the Critical agent. Work independently -- you have not seen the Evidence agent's output."
        user = f"Subtask: {subtask}"
        raw = self.llm.complete(system, user)
        return CriticalOutput(**json.loads(raw))
