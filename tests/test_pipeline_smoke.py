"""
Smoke test: run this after every merge (see the integration cadence rule
in the team plan). It doesn't check correctness of anyone's real logic --
it only checks that the pipeline runs end-to-end without crashing and
that every module returns the shape everything else expects.

Run with: pytest tests/test_pipeline_smoke.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import run_pipeline
from fault_injection.injector import drop_field_example
from agents.agents import EvidenceAgent, CriticalAgent
from llm_client import MockLLMClient


def test_pipeline_runs_baseline():
    report, result = run_pipeline("Test question", condition="A_baseline")
    assert result.condition == "A_baseline"
    assert result.guard_decision == "ALLOW"


def test_pipeline_runs_sequential_only():
    report, result = run_pipeline("Test question", condition="B_sequential_only")
    assert result.condition == "B_sequential_only"


def test_pipeline_runs_proposed():
    report, result = run_pipeline("Test question", condition="C_proposed")
    assert result.condition == "C_proposed"


def test_pipeline_with_fault_injection():
    report, result = run_pipeline(
        "Test question", condition="C_proposed", fault_injector_fn=drop_field_example
    )
    assert result.fault is not None
    assert result.fault.fault_type == "drop_field"


def test_all_three_conditions_produce_a_report():
    for condition in ["A_baseline", "B_sequential_only", "C_proposed"]:
        report, result = run_pipeline("Test question", condition=condition)
        assert "summary" in report

def test_mock_gives_critical_and_evidence_different_outputs():
    llm = MockLLMClient()
    evidence_claim = EvidenceAgent(llm).produce("x").claims[0].claim
    critical_claim = CriticalAgent(llm).produce("x").claims[0].claim
    assert evidence_claim != critical_claim

if __name__ == "__main__":
    test_pipeline_runs_baseline()
    test_pipeline_runs_sequential_only()
    test_pipeline_runs_proposed()
    test_pipeline_with_fault_injection()
    test_all_three_conditions_produce_a_report()
    print("All smoke tests passed.")
