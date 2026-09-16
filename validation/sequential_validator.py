"""
OWNER: Person A

Checks a single agent's output against what it was supposed to produce:
structural, context-preservation, intent-preservation, and semantic-
consistency checks. NO GENERATIVE LLM CALL IN THIS FILE for the actual
pass/fail decision -- use Pydantic (structural), embeddings (context/
semantic similarity), and NLI (intent/contradiction). See the project's
implementation guide, Section 2, for the exact design.

CONTRACT: validate(...) must always return a ValidationResult, no matter
what. Never raise an uncaught exception here -- a crash in this module
should not take down the whole pipeline.
"""

from __future__ import annotations
from schemas.models import ValidationResult, PlannerOutput, EvidenceOutput, CriticalOutput


def validate(
    planner_output: PlannerOutput,
    agent_output: EvidenceOutput | CriticalOutput,
    agent_name: str,
) -> ValidationResult:
    """
    STUB -- replace this with real structural/context/intent/semantic checks.

    Real implementation should:
      1. Structural: confirm agent_output matches its Pydantic schema
         (this mostly happens for free when the agent parses its own
         response -- but re-validate here in case upstream data was
         tampered with by fault injection).
      2. Context: for each constraint in planner_output.constraints,
         check whether it's still respected/referenced downstream
         (exact match for structured values, embedding similarity for
         free text).
      3. Intent: NLI(planner_output.topic, agent claims) -- flag on
         "contradiction".
      4. Semantic: embedding similarity between claims and planner_output.topic.
    """
    return ValidationResult(valid=True, reason="stub - not yet implemented", failed_checks=[])
