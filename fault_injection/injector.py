"""
OWNER: Member 1 (fault taxonomy + controlled injection + evaluation)

Adapts AutoInject from the base paper (Huang et al., ICML 2025,
github.com/CUHK-ARISE/MAS-Resilience) to this project's schemas. AutoInject
works by directly intercepting a message and injecting a typed, controllable
error into it -- unlike AutoTransform (which corrupts an agent's persona and
can't control error rate/type precisely). Every function here follows that
idea: intercept Evidence's output post-hoc and inject one typed fault.

The base paper's taxonomy is task-specific (code generation), so this file
defines an analogous taxonomy for THIS task: claim verification against
retrieved abstracts.

CONTRACT: every fault function takes an EvidenceOutput and returns
(mutated_output, FaultRecord). The FaultRecord is the ground truth -- without
it nothing downstream can be scored. The original object is never mutated.

SEVERITY (an experimental axis, Section 10 of the novelty doc) now changes
behavior for every fault type. Unless noted, severity controls BREADTH -- how
many of the Evidence agent's claims are corrupted:
    low    -> exactly one claim (the one at field_index, default the first)
    medium -> the first half of the claims (rounded up)
    high   -> every claim
Exceptions, documented on each function: contradict_claim("low") hedges a
claim instead of flipping it, and truncate_context drops claims rather than
corrupting them.

REPRODUCIBILITY: functions that make random choices accept an optional
`rng` (random.Random) so the harness can replay exactly the same fault.

Every fault is injected at the Evidence -> Validator handoff, matching the
fault_injector_fn hook in pipeline.py. Other locations (Critical,
Planner -> Evidence) would need a pipeline.py change.
"""

from __future__ import annotations
import copy
import math
import random
from typing import Callable, Optional
from schemas.models import FaultRecord, EvidenceOutput


SEVERITIES = ("low", "medium", "high")


# ---------------------------------------------------------------------------
# Fault taxonomy
# ---------------------------------------------------------------------------
# 1. drop_field          -- omission: claims silently lose their citation
# 2. contradict_claim    -- contradiction: claim rewritten to mean the opposite
# 3. wrong_source        -- citation error: cites a REAL but WRONG abstract
# 4. fabricated_citation -- citation error: cites an abstract id that does not
#                            exist in the corpus
# 5. fabricated_detail   -- hallucination: an invented, specific detail is
#                            appended to the claim
# 6. truncate_context    -- omission: downstream claims are silently dropped
# 7. semantic_drift      -- drift: claim is replaced by an off-topic statement
#                            while its citation is left intact
# ---------------------------------------------------------------------------


def _check_severity(severity: str) -> None:
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity!r}")


def _target_indices(n_claims: int, severity: str, start: int = 0) -> list[int]:
    """Which claim positions a fault touches, by severity (see module doc)."""
    _check_severity(severity)
    if n_claims <= 0:
        return []
    if severity == "low":
        return [min(max(start, 0), n_claims - 1)]
    if severity == "medium":
        return list(range(math.ceil(n_claims / 2)))
    return list(range(n_claims))


def _record(fault_type: str, target_field: Optional[str], severity: str) -> FaultRecord:
    return FaultRecord(
        fault_type=fault_type,
        target_field=target_field,
        severity=severity,
        injected_at="evidence->validator",
    )


def drop_field_example(
    evidence_output: EvidenceOutput, field_index: int = 0, severity: str = "medium"
) -> tuple[EvidenceOutput, FaultRecord]:
    """Removes supporting_reference from the targeted claims (context loss)."""
    mutated = copy.deepcopy(evidence_output)
    for i in _target_indices(len(mutated.claims), severity, field_index):
        mutated.claims[i].supporting_reference = None
    return mutated, _record("drop_field", "supporting_reference", severity)


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
    """Rewrites claims to contradict their original meaning.
      low    -> hedges ONE claim ("...though this is disputed") -- a subtle
                weakening, not a flip
      medium -> flips the first half of the claims
      high   -> flips every claim
    This is the fault type most relevant to the peer-consistency checker."""
    mutated = copy.deepcopy(evidence_output)
    for i in _target_indices(len(mutated.claims), severity, field_index):
        original = mutated.claims[i].claim
        if severity == "low":
            mutated.claims[i].claim = f"{original}, though this is disputed"
        else:
            mutated.claims[i].claim = _negate(original)
    return mutated, _record("contradict_claim", "claim", severity)


def wrong_source_citation(
    evidence_output: EvidenceOutput,
    corpus_passage_ids: list[str],
    field_index: int = 0,
    severity: str = "medium",
    rng: Optional[random.Random] = None,
) -> tuple[EvidenceOutput, FaultRecord]:
    """Swaps citations for a REAL but WRONG abstract id drawn from
    corpus_passage_ids. Detectable only by an entailment-style check (does the
    cited abstract actually say this?), not by a missing-field check."""
    rng = rng or random
    mutated = copy.deepcopy(evidence_output)
    for i in _target_indices(len(mutated.claims), severity, field_index):
        original_ref = mutated.claims[i].supporting_reference
        candidates = [pid for pid in corpus_passage_ids if pid != original_ref]
        if not candidates:
            raise ValueError("corpus_passage_ids needs at least 2 distinct ids")
        mutated.claims[i].supporting_reference = rng.choice(candidates)
    return mutated, _record("wrong_source", "supporting_reference", severity)


def fabricated_citation(
    evidence_output: EvidenceOutput, field_index: int = 0, severity: str = "high"
) -> tuple[EvidenceOutput, FaultRecord]:
    """Points citations at an abstract id that does not exist in the corpus.
    Detectable by checking the id against the known corpus."""
    mutated = copy.deepcopy(evidence_output)
    for i in _target_indices(len(mutated.claims), severity, field_index):
        mutated.claims[i].supporting_reference = "D000000000"  # never a real SciFact doc_id
    return mutated, _record("fabricated_citation", "supporting_reference", severity)


_FABRICATED_DETAILS = [
    "affecting approximately 73% of cases",
    "based on a sample of over 10,000 participants",
    "with a statistically significant p-value below 0.001",
    "according to a 2019 follow-up study",
]


def fabricated_detail(
    evidence_output: EvidenceOutput,
    field_index: int = 0,
    severity: str = "medium",
    rng: Optional[random.Random] = None,
) -> tuple[EvidenceOutput, FaultRecord]:
    """Appends an invented, specific-sounding detail no source supports (a
    hallucination fault). Harder to catch than fabricated_citation, since the
    citation field itself looks unchanged."""
    rng = rng or random
    mutated = copy.deepcopy(evidence_output)
    for i in _target_indices(len(mutated.claims), severity, field_index):
        detail = rng.choice(_FABRICATED_DETAILS)
        mutated.claims[i].claim = f"{mutated.claims[i].claim} ({detail})"
    return mutated, _record("fabricated_detail", "claim", severity)


def truncate_context(
    evidence_output: EvidenceOutput, severity: str = "high"
) -> tuple[EvidenceOutput, FaultRecord]:
    """Silently drops claims, simulating lost downstream context.
      low    -> drops only the last claim
      medium -> keeps the first half of the claims (rounded up)
      high   -> keeps only the first claim
    NOTE: with a single claim there is nothing to drop, so this is a no-op;
    the evaluation harness detects and excludes such no-op injections."""
    _check_severity(severity)
    mutated = copy.deepcopy(evidence_output)
    n = len(mutated.claims)
    if n > 0:
        keep = {"low": n - 1, "medium": math.ceil(n / 2), "high": 1}[severity]
        keep = max(1, min(keep, n))
        mutated.claims = mutated.claims[:keep]
    return mutated, _record("truncate_context", None, severity)


_OFF_TOPIC_CLAIMS = [
    "The global market for renewable energy storage has grown steadily over the last decade.",
    "Urban public transport ridership depends heavily on fare pricing and service frequency.",
    "Coastal erosion rates vary with sediment supply and storm frequency.",
    "Consumer preferences have shifted toward ad-supported streaming tiers.",
]


def semantic_drift(
    evidence_output: EvidenceOutput,
    field_index: int = 0,
    severity: str = "medium",
    rng: Optional[random.Random] = None,
) -> tuple[EvidenceOutput, FaultRecord]:
    """Replaces claims with an off-topic statement while leaving the citation
    intact -- the handoff drifts away from the task's topic. Simulates the
    'semantic drift during handoff' scenario in the project proposal."""
    rng = rng or random
    mutated = copy.deepcopy(evidence_output)
    for i in _target_indices(len(mutated.claims), severity, field_index):
        mutated.claims[i].claim = rng.choice(_OFF_TOPIC_CLAIMS)
    return mutated, _record("semantic_drift", "claim", severity)


# ---------------------------------------------------------------------------
# Registry + metadata
# ---------------------------------------------------------------------------
FAULT_REGISTRY = {
    "drop_field": drop_field_example,
    "contradict_claim": contradict_claim,
    "wrong_source": wrong_source_citation,   # needs corpus_passage_ids -- see make_injector
    "fabricated_citation": fabricated_citation,
    "fabricated_detail": fabricated_detail,
    "truncate_context": truncate_context,
    "semantic_drift": semantic_drift,
}

# HYPOTHESES about which signal should catch each fault -- to be TESTED by the
# experiments, not assumed. "category" is our own descriptive label; mapping
# faults onto MAST / MAS-FIRE categories is left for the paper.
FAULT_META = {
    "drop_field":          {"category": "omission",      "expected_detector": "sequential"},
    "contradict_claim":    {"category": "contradiction", "expected_detector": "peer"},
    "wrong_source":        {"category": "citation",      "expected_detector": "sequential"},
    "fabricated_citation": {"category": "citation",      "expected_detector": "sequential"},
    "fabricated_detail":   {"category": "hallucination", "expected_detector": "unclear"},
    "truncate_context":    {"category": "omission",      "expected_detector": "sequential"},
    "semantic_drift":      {"category": "drift",         "expected_detector": "both"},
}

_RNG_FAULTS = {"fabricated_detail", "semantic_drift"}


def make_injector(
    fault_name: str,
    severity: str = "medium",
    corpus_passage_ids: Optional[list[str]] = None,
    rng: Optional[random.Random] = None,
) -> Callable[[EvidenceOutput], tuple[EvidenceOutput, FaultRecord]]:
    """Returns a plain (evidence_output) -> (mutated, FaultRecord) function --
    the shape pipeline.py's fault_injector_fn expects -- for ANY fault type,
    so callers never special-case the ones that need extra arguments."""
    if fault_name not in FAULT_REGISTRY:
        raise KeyError(f"unknown fault type {fault_name!r}; known: {sorted(FAULT_REGISTRY)}")
    _check_severity(severity)
    fn = FAULT_REGISTRY[fault_name]

    def _inject(evidence_output: EvidenceOutput) -> tuple[EvidenceOutput, FaultRecord]:
        if fault_name == "wrong_source":
            if not corpus_passage_ids:
                raise ValueError("wrong_source needs corpus_passage_ids")
            return fn(evidence_output, corpus_passage_ids, severity=severity, rng=rng)
        if fault_name in _RNG_FAULTS:
            return fn(evidence_output, severity=severity, rng=rng)
        return fn(evidence_output, severity=severity)

    return _inject


def make_wrong_source_injector(
    corpus_passage_ids: list[str],
    severity: str = "medium",
    rng: Optional[random.Random] = None,
):
    """Kept for backward compatibility; equivalent to
    make_injector("wrong_source", ...)."""
    return make_injector("wrong_source", severity, corpus_passage_ids, rng)