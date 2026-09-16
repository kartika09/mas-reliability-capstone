"""
OWNER: Person D

Unlike the other three modules, this one is implemented for real, not
stubbed -- it's meant to stay a simple, deterministic, interpretable rule,
not a learned model. Keep it that way; the whole point of this project is
that the DECISION isn't made by a generative LLM.

Feel free to tune the threshold or the rule itself, but keep it a plain
function of the two signals -- no LLM call here.
"""

from __future__ import annotations
from schemas.models import ValidationResult, PeerConsistencyResult, GuardDecision

DISAGREEMENT_THRESHOLD = 0.7


def decide(validation: ValidationResult, peer: PeerConsistencyResult) -> GuardDecision:
    triggered_by = []
    if not validation.valid:
        triggered_by.append("sequential")
    if peer.disagreement_score > DISAGREEMENT_THRESHOLD:
        triggered_by.append("peer")

    if triggered_by:
        return GuardDecision(
            decision="RECOVER",
            triggered_by=triggered_by,
            explanation=f"Flagged by: {', '.join(triggered_by)}.",
        )
    return GuardDecision(decision="ALLOW", triggered_by=[], explanation="No issues detected.")
