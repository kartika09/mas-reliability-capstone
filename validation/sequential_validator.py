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
from typing import Optional
from schemas.models import ValidationResult, PlannerOutput, EvidenceOutput, CriticalOutput


def validate(
    planner_output: PlannerOutput,
    agent_output: EvidenceOutput | CriticalOutput,
    agent_name: str,
    context: Optional[list[dict]] = None,
) -> ValidationResult:
    """
    STUB -- replace this with real structural/context/intent/semantic checks.

    `context`: the same retrieved passages (from retrieval.retriever.retrieve)
    that were given to the agent, if retrieval was on -- each dict has at
    least "passage_id" and "text"/"sentences". None if retrieval is off.
    This is what makes a CITATION check possible: is agent_output's
    supporting_reference one of these passage_ids, and does that passage
    actually support the claim? Without it, wrong_source and
    fabricated_citation faults are undetectable from this function alone.

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
      5. Citation (needs `context`): supporting_reference in
         {p["passage_id"] for p in context}, and ideally an entailment
         check between the claim and that passage's text.
    """
    return ValidationResult(valid=True, reason="stub - not yet implemented", failed_checks=[])