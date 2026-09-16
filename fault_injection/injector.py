"""
OWNER: Person C

Adapts AutoTransform / AutoInject from the base paper's repo
(github.com/CUHK-ARISE/MAS-Resilience) to this project's schemas. Read
their implementation before writing your own -- don't rebuild from scratch.

CONTRACT: every inject_* function takes a schema object and returns
(mutated_object, FaultRecord). The FaultRecord is your ground truth --
without it, nothing downstream can be scored.

One real example (drop_field) is included below so the interface is
concrete. The rest are stubs for you to fill in.
"""

from __future__ import annotations
import copy
from schemas.models import FaultRecord, EvidenceOutput, CriticalOutput


def drop_field_example(evidence_output: EvidenceOutput, field_index: int = 0) -> tuple[EvidenceOutput, FaultRecord]:
    """Working example: removes a claim's supporting_reference to simulate
    context loss. Use this as the template for the other fault types."""
    mutated = copy.deepcopy(evidence_output)
    if mutated.claims:
        mutated.claims[field_index].supporting_reference = None
    fault = FaultRecord(fault_type="drop_field", target_field="supporting_reference", severity="medium")
    return mutated, fault


def contradict_claim(evidence_output: EvidenceOutput) -> tuple[EvidenceOutput, FaultRecord]:
    """STUB -- rewrite a claim to directly contradict its original meaning.
    This is the fault type most relevant to testing the peer-consistency
    checker, since a contradiction here is exactly what Person B's module
    is meant to catch relative to the Critical agent's independent output."""
    mutated = copy.deepcopy(evidence_output)
    # TODO: implement the actual contradiction rewrite
    fault = FaultRecord(fault_type="contradict_claim", severity="high")
    return mutated, fault


def truncate_context(evidence_output: EvidenceOutput) -> tuple[EvidenceOutput, FaultRecord]:
    """STUB -- drop all but the first claim."""
    mutated = copy.deepcopy(evidence_output)
    mutated.claims = mutated.claims[:1]
    fault = FaultRecord(fault_type="truncate_context", severity="high")
    return mutated, fault
