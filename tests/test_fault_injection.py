"""
Regression tests for fault_injection/injector.py (Member 1's deliverable).

Covers, per fault type: correct mutation + correct FaultRecord fields,
deepcopy isolation (original object never mutated), and edge cases
(empty claims list, wrong_source_citation's ValueError). Also covers
negation quality against real-shaped claims, and full pipeline
integration across all three conditions.

Run with: pytest tests/test_fault_injection.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import pytest

from schemas.models import EvidenceOutput, AgentClaim, FaultRecord
from fault_injection.injector import (
    drop_field_example,
    contradict_claim,
    wrong_source_citation,
    fabricated_citation,
    fabricated_detail,
    truncate_context,
    FAULT_REGISTRY,
    make_wrong_source_injector,
)
from pipeline import run_pipeline


def _make_evidence(claims):
    return EvidenceOutput(claims=[AgentClaim(claim=c, supporting_reference=ref) for c, ref in claims])


# ---------------------------------------------------------------------------
# 1. drop_field
# ---------------------------------------------------------------------------

def test_drop_field_removes_supporting_reference():
    ev = _make_evidence([("Paper X shows Y.", "Paper X")])
    mutated, fault = drop_field_example(ev)
    assert mutated.claims[0].supporting_reference is None
    assert fault.fault_type == "drop_field"
    assert fault.target_field == "supporting_reference"
    assert fault.severity == "medium"


def test_drop_field_does_not_mutate_original():
    ev = _make_evidence([("Paper X shows Y.", "Paper X")])
    original_ref = ev.claims[0].supporting_reference
    drop_field_example(ev)
    assert ev.claims[0].supporting_reference == original_ref


def test_drop_field_empty_claims_no_crash():
    ev = EvidenceOutput(claims=[])
    mutated, fault = drop_field_example(ev)
    assert mutated.claims == []
    assert fault.fault_type == "drop_field"


# ---------------------------------------------------------------------------
# 2. contradict_claim
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("claim_text", [
    "BRCA1 mutations increase the risk of breast cancer.",
    "Elevated homocysteine is associated with folate deficiency.",
    "Microcin J25 inhibits bacterial RNA polymerase.",
    "L-carnitine supplementation raises circulating TMAO levels.",
    "FOXO3 variants are protective against Crohn's disease.",
])
def test_contradict_claim_negates_real_shaped_claims(claim_text):
    ev = _make_evidence([(claim_text, "D1")])
    mutated, fault = contradict_claim(ev, severity="high")
    new_claim = mutated.claims[0].claim
    assert new_claim != claim_text
    # sensible negation: either a recognized pattern flipped, or the
    # "It is not true that..." fallback prefix was used
    assert (
        "not" in new_claim.lower()
        or "It is not true that" in new_claim
    )
    assert fault.fault_type == "contradict_claim"
    assert fault.severity == "high"


def test_contradict_claim_low_severity_hedges_instead_of_flipping():
    ev = _make_evidence([("Paper X shows GPT-4 outperforms Gemini.", "Paper X")])
    mutated, fault = contradict_claim(ev, severity="low")
    assert mutated.claims[0].claim.startswith("Paper X shows GPT-4 outperforms Gemini.")
    assert "disputed" in mutated.claims[0].claim
    assert fault.severity == "low"


def test_contradict_claim_does_not_mutate_original():
    ev = _make_evidence([("Paper X shows Y.", "Paper X")])
    original_claim = ev.claims[0].claim
    contradict_claim(ev)
    assert ev.claims[0].claim == original_claim


def test_contradict_claim_empty_claims_no_crash():
    ev = EvidenceOutput(claims=[])
    mutated, fault = contradict_claim(ev)
    assert mutated.claims == []
    assert fault.fault_type == "contradict_claim"


# ---------------------------------------------------------------------------
# 3. wrong_source_citation
# ---------------------------------------------------------------------------

def test_wrong_source_citation_swaps_to_a_real_but_different_id():
    ev = _make_evidence([("Paper X shows Y.", "D1")])
    corpus_ids = ["D1", "D2", "D3"]
    mutated, fault = wrong_source_citation(ev, corpus_ids)
    assert mutated.claims[0].supporting_reference in corpus_ids
    assert mutated.claims[0].supporting_reference != "D1"
    assert fault.fault_type == "wrong_source"
    assert fault.severity == "medium"


def test_wrong_source_citation_raises_with_fewer_than_two_ids():
    ev = _make_evidence([("Paper X shows Y.", "D1")])
    with pytest.raises(ValueError):
        wrong_source_citation(ev, ["D1"])  # only the original id -- no valid candidate


def test_wrong_source_citation_does_not_mutate_original():
    ev = _make_evidence([("Paper X shows Y.", "D1")])
    original_ref = ev.claims[0].supporting_reference
    wrong_source_citation(ev, ["D1", "D2"])
    assert ev.claims[0].supporting_reference == original_ref


def test_make_wrong_source_injector_matches_plain_fault_injector_fn_shape():
    ev = _make_evidence([("Paper X shows Y.", "D1")])
    inject = make_wrong_source_injector(["D1", "D2", "D3"])
    mutated, fault = inject(ev)  # plain (evidence_output) -> (mutated, FaultRecord)
    assert mutated.claims[0].supporting_reference in ("D2", "D3")
    assert fault.fault_type == "wrong_source"


# ---------------------------------------------------------------------------
# 4. fabricated_citation
# ---------------------------------------------------------------------------

def test_fabricated_citation_points_at_nonexistent_id():
    ev = _make_evidence([("Paper X shows Y.", "D1")])
    mutated, fault = fabricated_citation(ev)
    assert mutated.claims[0].supporting_reference == "D000000000"
    assert fault.fault_type == "fabricated_citation"
    assert fault.severity == "high"


def test_fabricated_citation_does_not_mutate_original():
    ev = _make_evidence([("Paper X shows Y.", "D1")])
    fabricated_citation(ev)
    assert ev.claims[0].supporting_reference == "D1"


def test_fabricated_citation_empty_claims_no_crash():
    ev = EvidenceOutput(claims=[])
    mutated, fault = fabricated_citation(ev)
    assert mutated.claims == []


# ---------------------------------------------------------------------------
# 5. fabricated_detail
# ---------------------------------------------------------------------------

def test_fabricated_detail_appends_unsupported_detail():
    ev = _make_evidence([("Paper X shows Y.", "D1")])
    mutated, fault = fabricated_detail(ev)
    assert mutated.claims[0].claim.startswith("Paper X shows Y.")
    assert mutated.claims[0].claim != "Paper X shows Y."
    assert mutated.claims[0].supporting_reference == "D1"  # citation field untouched
    assert fault.fault_type == "fabricated_detail"
    assert fault.severity == "medium"


def test_fabricated_detail_does_not_mutate_original():
    ev = _make_evidence([("Paper X shows Y.", "D1")])
    original_claim = ev.claims[0].claim
    fabricated_detail(ev)
    assert ev.claims[0].claim == original_claim


# ---------------------------------------------------------------------------
# 6. truncate_context
# ---------------------------------------------------------------------------

def test_truncate_context_keeps_only_first_claim():
    ev = _make_evidence([("Claim A", "D1"), ("Claim B", "D2"), ("Claim C", "D3")])
    mutated, fault = truncate_context(ev)
    assert len(mutated.claims) == 1
    assert mutated.claims[0].claim == "Claim A"
    assert fault.fault_type == "truncate_context"
    assert fault.severity == "high"


def test_truncate_context_does_not_mutate_original():
    ev = _make_evidence([("Claim A", "D1"), ("Claim B", "D2")])
    truncate_context(ev)
    assert len(ev.claims) == 2


def test_truncate_context_empty_claims_no_crash():
    ev = EvidenceOutput(claims=[])
    mutated, fault = truncate_context(ev)
    assert mutated.claims == []


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_fault_registry_has_all_six_fault_types():
    expected = {
        "drop_field", "contradict_claim", "wrong_source",
        "fabricated_citation", "fabricated_detail", "truncate_context",
    }
    assert set(FAULT_REGISTRY.keys()) == expected


# ---------------------------------------------------------------------------
# Full pipeline integration: every fault type, every condition, no crashes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("condition", ["A_baseline", "B_sequential_only", "C_proposed"])
@pytest.mark.parametrize("fault_name", [
    "drop_field", "contradict_claim", "fabricated_citation",
    "fabricated_detail", "truncate_context",
])
def test_every_plain_fault_runs_through_full_pipeline(condition, fault_name):
    fault_fn = FAULT_REGISTRY[fault_name]
    report, result = run_pipeline(
        "Test question", condition=condition, fault_injector_fn=fault_fn
    )
    assert result.condition == condition
    assert result.fault is not None
    assert result.fault.fault_type == fault_name


@pytest.mark.parametrize("condition", ["A_baseline", "B_sequential_only", "C_proposed"])
def test_wrong_source_runs_through_full_pipeline_via_factory(condition):
    inject = make_wrong_source_injector(["D1", "D2", "D3"])
    report, result = run_pipeline(
        "Test question", condition=condition, fault_injector_fn=inject
    )
    assert result.condition == condition
    assert result.fault is not None
    assert result.fault.fault_type == "wrong_source"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])