"""
Shared data contracts for the whole pipeline.

RULE FOR THE TEAM: nobody edits this file alone. If you need a new field,
raise it in the group chat first -- every module downstream depends on
these shapes staying stable.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Optional


class AgentClaim(BaseModel):
    """One atomic claim an agent is making. Keep claims short and specific --
    this is what the peer-consistency checker (Person B) compares between
    Evidence and Critical, so vague multi-part claims are hard to match."""
    claim: str
    supporting_reference: Optional[str] = None


class PlannerOutput(BaseModel):
    topic: str
    constraints: list[str] = Field(default_factory=list)
    subtask_evidence: str
    subtask_critical: str


class EvidenceOutput(BaseModel):
    claims: list[AgentClaim]


class CriticalOutput(BaseModel):
    claims: list[AgentClaim]


class FinalReport(BaseModel):
    summary: str
    supporting_claims: list[AgentClaim]


# ---- Validation / peer-consistency / guard contracts ----
# These are the shapes every stub currently returns, and the shapes every
# real implementation MUST keep returning so the pipeline doesn't break.

class ValidationResult(BaseModel):
    """Output of the Sequential Handoff Validator (Person A)."""
    valid: bool
    reason: str = ""
    failed_checks: list[str] = Field(default_factory=list)  # e.g. ["structural", "context"]


class PeerConsistencyResult(BaseModel):
    """Output of the Peer-Consistency Checker (Person B)."""
    disagreement_score: float = 0.0  # 0.0 = full agreement, 1.0 = full contradiction
    flagged_claim_pairs: list[tuple[str, str]] = Field(default_factory=list)


class GuardDecision(BaseModel):
    """Output of the Handoff Guard (Person D)."""
    decision: Literal["ALLOW", "RECOVER", "ABSTAIN"]
    triggered_by: list[str] = Field(default_factory=list)  # e.g. ["sequential"], ["peer"], ["sequential", "peer"]
    explanation: str = ""


class FaultRecord(BaseModel):
    """Ground truth for an injected fault (Person C). Every injected run
    must produce one of these so detection can be scored against a known
    answer."""
    fault_type: str          # e.g. "drop_field", "contradict_claim"
    target_field: Optional[str] = None
    severity: Literal["low", "medium", "high"] = "medium"
    injected_at: str = ""    # which handoff, e.g. "planner->evidence"


class RunResult(BaseModel):
    """One row of the evaluation harness output (shared / whoever owns eval)."""
    run_id: str
    condition: Literal["A_baseline", "B_sequential_only", "C_proposed"]
    fault: Optional[FaultRecord] = None
    detected: bool = False
    detected_by: Literal["none", "sequential", "peer", "both"] = "none"
    guard_decision: Optional[str] = None
    recovered: bool = False
    verified: bool = False
    false_recovery: bool = False
    abstained: bool = False
    final_correct: Optional[bool] = None
    latency_seconds: float = 0.0
