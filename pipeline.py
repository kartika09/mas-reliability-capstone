"""
Wires the whole system together. This file should barely change once the
team agrees on it -- everyone else's work plugs INTO this, this doesn't
plug into theirs.

Supports three conditions (matches the experiment design in the project
document):
  A_baseline          -- no validation, no peer check, no intervention
  B_sequential_only   -- sequential validator active, peer check off
  C_proposed          -- both signals active (the actual contribution)

Run this file directly for a quick smoke test:  python pipeline.py
"""

from __future__ import annotations
import uuid
import time
from typing import Literal, Optional

from agents.agents import PlannerAgent, EvidenceAgent, CriticalAgent
from llm_client import MockLLMClient
from validation.sequential_validator import validate
from peer_consistency.checker import check as peer_check
from guard.decision import decide, DISAGREEMENT_THRESHOLD
from guard.recovery import recover
from evaluation.logger import log_message, log_run_result
from schemas.models import RunResult, FaultRecord, ValidationResult, PeerConsistencyResult, GuardDecision

Condition = Literal["A_baseline", "B_sequential_only", "C_proposed"]


def run_pipeline(
    research_question: str,
    condition: Condition = "C_proposed",
    injected_fault: Optional[FaultRecord] = None,
    fault_injector_fn=None,  # callable(evidence_output) -> (mutated_output, FaultRecord), or None
    llm_client=None,
) -> tuple[dict, RunResult]:
    run_id = str(uuid.uuid4())[:8]
    start = time.time()
    llm = llm_client or MockLLMClient()

    planner = PlannerAgent(llm)
    evidence_agent = EvidenceAgent(llm)
    critical_agent = CriticalAgent(llm)

    # 1. Planner decomposes the task
    planner_output = planner.decompose(research_question)
    log_message(run_id, "Planner", "Evidence+Critical", planner_output.model_dump())

    # 2. Evidence and Critical produce INDEPENDENT outputs (neither sees the other yet)
    evidence_output = evidence_agent.produce(planner_output.subtask_evidence)
    critical_output = critical_agent.produce(planner_output.subtask_critical)

    # 2b. Optionally inject a fault into Evidence's output, post-hoc, before validation
    fault_record = injected_fault
    if fault_injector_fn is not None:
        evidence_output, fault_record = fault_injector_fn(evidence_output)

    log_message(run_id, "Evidence", "HandoffValidator", evidence_output.model_dump())
    log_message(run_id, "Critical", "HandoffValidator", critical_output.model_dump())

    # 3. Sequential validation (conditions B and C)
    if condition in ("B_sequential_only", "C_proposed"):
        validation_result = validate(planner_output, evidence_output, "Evidence")
    else:
        validation_result = ValidationResult(valid=True, reason="validation disabled (baseline)")

    # 4. Peer-consistency check (condition C only)
    if condition == "C_proposed":
        peer_result = peer_check(evidence_output, critical_output)
    else:
        peer_result = PeerConsistencyResult(disagreement_score=0.0)

    # 5. Guard decision
    if condition == "A_baseline":
        guard_result = GuardDecision(decision="ALLOW", explanation="baseline: no guard active")
    else:
        guard_result = decide(validation_result, peer_result)

    # 6. Recovery: if the guard says RECOVER, actually regenerate Evidence's
    # output and verify it with the OTHER signal than the one that flagged it
    # (never let a check verify its own detection -- see guard/recovery.py).
    recovered = verified = abstained = False
    final_evidence_output = evidence_output
    if guard_result.decision == "RECOVER":
        def regenerate_fn(_current_output):
            return evidence_agent.produce(planner_output.subtask_evidence)

        def verify_fn(new_output):
            if "sequential" in guard_result.triggered_by:
                # sequential validator was the flag -> verify with peer check
                return peer_check(new_output, critical_output).disagreement_score <= DISAGREEMENT_THRESHOLD
            # peer check was the flag -> verify with sequential validator
            return validate(planner_output, new_output, "Evidence").valid

        final_evidence_output, recovered, verified, abstained = recover(
            evidence_output, regenerate_fn, verify_fn, guard_result.triggered_by
        )

    # 7. Planner synthesis -- uses the (possibly recovered) Evidence output,
    # and is skipped on ABSTAIN so a report the pipeline couldn't verify
    # never goes out looking identical to a normal one.
    if abstained:
        final_report_dict = {"summary": None, "supporting_claims": [], "abstained": True}
        log_message(run_id, "Planner", "FinalReport", final_report_dict)
    else:
        final_report = planner.synthesize(final_evidence_output, critical_output)
        final_report_dict = final_report.model_dump()
        log_message(run_id, "Planner", "FinalReport", final_report_dict)

    detected = guard_result.decision != "ALLOW"
    detected_by = "none"
    if detected:
        if "sequential" in guard_result.triggered_by and "peer" in guard_result.triggered_by:
            detected_by = "both"
        elif "sequential" in guard_result.triggered_by:
            detected_by = "sequential"
        elif "peer" in guard_result.triggered_by:
            detected_by = "peer"

    result = RunResult(
        run_id=run_id,
        condition=condition,
        fault=fault_record,
        detected=detected,
        detected_by=detected_by,
        guard_decision=guard_result.decision,
        recovered=recovered,
        verified=verified,
        abstained=abstained,
        disagreement_score=peer_result.disagreement_score,
        validation_reason=validation_result.reason,
        model=getattr(llm, "model", llm.__class__.__name__),
        latency_seconds=time.time() - start,
    )
    log_run_result(result.model_dump())
    return final_report_dict, result


if __name__ == "__main__":
    print("Running end-to-end smoke test with MockLLMClient (no API key needed)...\n")
    report, result = run_pipeline(
        "Write a comparison between GPT-4 and Gemini using at least 5 research papers, under a 50000 budget.",
        condition="C_proposed",
    )
    print("Final report:", report)
    print("\nRun result:", result.model_dump())
    print("\nCheck logs/messages.jsonl and logs/runs.jsonl for the full trace.")