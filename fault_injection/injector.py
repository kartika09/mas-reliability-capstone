"""
OWNER: Member 1 (fault taxonomy + controlled injection + evaluation)

Adapts AutoInject from the base paper (Huang et al., ICML 2025,
github.com/CUHK-ARISE/MAS-Resilience) to this project's schemas. AutoInject
in the base paper works by directly intercepting a message and injecting a
typed, controllable error into it -- unlike AutoTransform (which corrupts an
agent's persona and can't control error rate/type precisely, per the paper's
own ablation). Every function here follows that same idea: intercept
Evidence's output post-hoc and inject one typed, controllable fault.

The base paper's AutoInject taxonomy is task-specific (their code-generation
taxonomy is Logical / Indexing / Mathematical / Formatting / Initialization /
Infinite-loop / Runtime-invocation errors -- see their Appendix B.1). Our
task is claim verification against retrieved abstracts, not code, so this
file defines an analogous taxonomy for THIS task instead of reusing theirs
literally.

CONTRACT: every inject_* function takes an EvidenceOutput and returns
(mutated_output, FaultRecord). The FaultRecord is the ground truth --
without it, nothing downstream (validator, peer checker, guard) can be
scored. Every function also accepts an optional `severity` (see
schemas.models.FaultRecord: "low" | "medium" | "high") so the taxonomy
supports Fault Severity as its own experimental axis (Section 10 of the
novelty doc), not just Fault Type.

Every fault here is injected at the same handoff (Evidence -> Validator),
matching what pipeline.py's fault_injector_fn hook currently supports.
Injecting at other handoffs (e.g. Planner -> Evidence) would need a change
to pipeline.py and is a natural extension, not done here.
"""

from __future__ import annotations
import copy
import random
from typing import Optional
from schemas.models import FaultRecord, EvidenceOutput


# ---------------------------------------------------------------------------
# Fault taxonomy
# ---------------------------------------------------------------------------
# 1. drop_field          -- context loss: a claim silently loses its citation
# 2. contradict_claim    -- stance flip: claim is rewritten to mean the opposite
# 3. wrong_source        -- citation error: claim points at a REAL but WRONG
#                            abstract (plausible-looking, still detectable by
#                            checking claim-vs-citation entailment)
# 4. fabricated_citation -- citation error: claim points at an abstract ID
#                            that does not exist in the corpus at all
# 5. fabricated_detail   -- hallucination: an invented, specific detail is
#                            appended to the claim, unsupported by any source
# 6. truncate_context    -- downstream claims are silently dropped
# ---------------------------------------------------------------------------


def drop_field_example(
    evidence_output: EvidenceOutput, field_index: int = 0
) -> tuple[EvidenceOutput, FaultRecord]:
    """Removes a claim's supporting_reference, simulating context loss."""
    mutated = copy.deepcopy(evidence_output)
    if mutated.claims:
        mutated.claims[field_index].supporting_reference = None
    fault = FaultRecord(
        fault_type="drop_field",
        target_field="supporting_reference",
        severity="medium",
        injected_at="evidence->validator",
    )
    return mutated, fault


# Simple rule-based negation. Not linguistically perfect (that's fine --
# the point is a controlled, known ground-truth flip, not natural prose).
_NEGATION_PATTERNS = [
    (" does not ", " does "),
    (" do not ", " do "),
    (" is not ", " is "),
    (" are not ", " are "),
    (" cannot ", " can "),
    (" is ", " is not "),
    (" are ", " are not "),
    (" does ", " does not "),
    (" do ", " do not "),
    (" can ", " cannot "),
]


def _negate(text: str) -> str:
    lowered = f" {text} "
    for pattern, replacement in _NEGATION_PATTERNS:
        if pattern in lowered:
            return lowered.replace(pattern, replacement, 1).strip()
    return f"It is not true that {text[0].lower()}{text[1:]}" if text else text


def contradict_claim(
    evidence_output: EvidenceOutput, field_index: int = 0, severity: str = "high"
) -> tuple[EvidenceOutput, FaultRecord]:
    """Rewrites a claim to directly contradict its original meaning.
    This is the fault type most relevant to testing the peer-consistency
    checker: a contradiction here is exactly what disagreement with the
    Critical agent's independent output should catch."""
    mutated = copy.deepcopy(evidence_output)
    if mutated.claims:
        original = mutated.claims[field_index].claim
        if severity == "low":
            # subtle: hedge the claim rather than flatly reverse it
            mutated.claims[field_index].claim = f"{original}, though this is disputed"
        else:
            mutated.claims[field_index].claim = _negate(original)
    fault = FaultRecord(
        fault_type="contradict_claim",
        target_field="claim",
        severity=severity,
        injected_at="evidence->validator",
    )
    return mutated, fault


def wrong_source_citation(
    evidence_output: EvidenceOutput,
    corpus_passage_ids: list[str],
    field_index: int = 0,
    severity: str = "medium",
) -> tuple[EvidenceOutput, FaultRecord]:
    """Swaps a claim's citation for a REAL but WRONG abstract id, drawn from
    corpus_passage_ids. Needs the corpus's passage IDs so the wrong citation
    is a real one -- pass data["corpus"] passage_ids from dataset.load_dataset().
    Detectable only by an entailment-style check (does the cited abstract
    actually say this?), not by a missing-field check."""
    mutated = copy.deepcopy(evidence_output)
    original_ref = None
    if mutated.claims:
        original_ref = mutated.claims[field_index].supporting_reference
        candidates = [pid for pid in corpus_passage_ids if pid != original_ref]
        if not candidates:
            raise ValueError("corpus_passage_ids needs at least 2 distinct ids")
        wrong_id = random.choice(candidates)
        mutated.claims[field_index].supporting_reference = wrong_id
    fault = FaultRecord(
        fault_type="wrong_source",
        target_field="supporting_reference",
        severity=severity,
        injected_at="evidence->validator",
    )
    return mutated, fault


def fabricated_citation(
    evidence_output: EvidenceOutput, field_index: int = 0, severity: str = "high"
) -> tuple[EvidenceOutput, FaultRecord]:
    """Points a claim's citation at an abstract ID that does not exist in
    the corpus at all -- a fabricated reference. Detectable by checking the
    id against the known corpus (no lookup needed to catch this one)."""
    mutated = copy.deepcopy(evidence_output)
    if mutated.claims:
        mutated.claims[field_index].supporting_reference = "D000000000"  # never a real SciFact doc_id
    fault = FaultRecord(
        fault_type="fabricated_citation",
        target_field="supporting_reference",
        severity=severity,
        injected_at="evidence->validator",
    )
    return mutated, fault


_FABRICATED_DETAILS = [
    "affecting approximately 73% of cases",
    "based on a sample of over 10,000 participants",
    "with a statistically significant p-value below 0.001",
    "according to a 2019 follow-up study",
]


def fabricated_detail(
    evidence_output: EvidenceOutput, field_index: int = 0, severity: str = "medium"
) -> tuple[EvidenceOutput, FaultRecord]:
    """Appends an invented, specific-sounding detail to a claim that no
    source actually supports -- a hallucination fault. Harder to catch than
    fabricated_citation, since the citation field itself looks unchanged."""
    mutated = copy.deepcopy(evidence_output)
    if mutated.claims:
        detail = random.choice(_FABRICATED_DETAILS)
        mutated.claims[field_index].claim = f"{mutated.claims[field_index].claim} ({detail})"
    fault = FaultRecord(
        fault_type="fabricated_detail",
        target_field="claim",
        severity=severity,
        injected_at="evidence->validator",
    )
    return mutated, fault


def truncate_context(evidence_output: EvidenceOutput, severity: str = "high") -> tuple[EvidenceOutput, FaultRecord]:
    """Drops all but the first claim, simulating lost downstream context."""
    mutated = copy.deepcopy(evidence_output)
    mutated.claims = mutated.claims[:1]
    fault = FaultRecord(
        fault_type="truncate_context",
        severity=severity,
        injected_at="evidence->validator",
    )
    return mutated, fault


# ---------------------------------------------------------------------------
# Registry: lets the evaluation harness iterate over every fault type
# without hardcoding the list in two places.
# ---------------------------------------------------------------------------
FAULT_REGISTRY = {
    "drop_field": drop_field_example,
    "contradict_claim": contradict_claim,
    "wrong_source": wrong_source_citation,   # needs corpus_passage_ids -- see make_wrong_source_injector
    "fabricated_citation": fabricated_citation,
    "fabricated_detail": fabricated_detail,
    "truncate_context": truncate_context,
}


def make_wrong_source_injector(corpus_passage_ids: list[str], severity: str = "medium"):
    """wrong_source_citation needs the corpus's id list, which the other
    fault functions don't -- this wraps it into the plain
    (evidence_output) -> (mutated, FaultRecord) shape pipeline.py expects,
    so it can be used the same way as every other fault_injector_fn."""
    def _inject(evidence_output: EvidenceOutput) -> tuple[EvidenceOutput, FaultRecord]:
        return wrong_source_citation(evidence_output, corpus_passage_ids, severity=severity)
    return _inject