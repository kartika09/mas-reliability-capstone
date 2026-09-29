"""
Regression tests for the pipeline.py change that (1) passes the retrieved
`context` through to validate() and recovery's verify_fn, and (2) logs the
validator, peer-checker and guard results as messages (previously only
agent outputs and the final report were logged).

Run with: pytest tests/test_pipeline_logging.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest

from pipeline import run_pipeline
from schemas.models import ValidationResult, PeerConsistencyResult


def _read_messages(run_id, logs_dir="logs"):
    path = os.path.join(logs_dir, "messages.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if json.loads(line)["run_id"] == run_id]


@pytest.fixture(autouse=True)
def _isolated_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield


def test_context_reaches_validate(monkeypatch):
    seen = {}

    def fake_validate(planner_output, agent_output, agent_name, context=None):
        seen["context"] = context
        return ValidationResult(valid=True, reason="ok")

    monkeypatch.setattr("pipeline.validate", fake_validate)
    fake_index = object()  # retrieve() is what actually reads it; stub that too
    monkeypatch.setattr("pipeline.retrieve", lambda index, claim, k=5: [{"passage_id": "D1", "text": "t"}])
    run_pipeline("q", condition="C_proposed", retrieval_index=fake_index)
    assert seen["context"] == [{"passage_id": "D1", "text": "t"}]


def test_context_is_none_when_retrieval_off(monkeypatch):
    seen = {}

    def fake_validate(planner_output, agent_output, agent_name, context=None):
        seen["context"] = context
        return ValidationResult(valid=True, reason="ok")

    monkeypatch.setattr("pipeline.validate", fake_validate)
    run_pipeline("q", condition="C_proposed")
    assert seen["context"] is None


def test_guard_and_validator_results_are_logged():
    report, result = run_pipeline("q", condition="C_proposed")
    messages = _read_messages(result.run_id)
    senders = {m["sender"] for m in messages}
    assert "HandoffValidator" in senders
    assert "PeerConsistencyChecker" in senders
    assert "Guard" in senders
    guard_msg = next(m for m in messages if m["sender"] == "Guard")
    assert guard_msg["content"]["decision"] == result.guard_decision


def test_peer_checker_not_logged_outside_condition_c():
    report, result = run_pipeline("q", condition="B_sequential_only")
    messages = _read_messages(result.run_id)
    senders = {m["sender"] for m in messages}
    assert "HandoffValidator" in senders
    assert "PeerConsistencyChecker" not in senders
    assert "Guard" in senders


def test_baseline_still_logs_guard_allow():
    report, result = run_pipeline("q", condition="A_baseline")
    messages = _read_messages(result.run_id)
    guard_msg = next(m for m in messages if m["sender"] == "Guard")
    assert guard_msg["content"]["decision"] == "ALLOW"


def test_recovery_module_logged_when_recover_triggered(monkeypatch):
    monkeypatch.setattr(
        "pipeline.validate",
        lambda planner_output, agent_output, agent_name, context=None: ValidationResult(
            valid=False, reason="forced"
        ),
    )
    report, result = run_pipeline("q", condition="C_proposed")
    messages = _read_messages(result.run_id)
    senders = {m["sender"] for m in messages}
    assert "RecoveryModule" in senders
    assert result.recovered is True


def test_verify_fn_passes_context_to_validate_during_recovery(monkeypatch):
    seen_contexts = []

    def fake_validate(planner_output, agent_output, agent_name, context=None):
        seen_contexts.append(context)
        # first call (step 3) fails to trigger RECOVER via peer path instead;
        # force peer to flag it so verify_fn's sequential branch is NOT used,
        # then force peer disagreement high so guard triggers via "peer",
        # meaning verify_fn calls validate() (the other signal) -- that's the
        # call whose context we want to confirm.
        return ValidationResult(valid=True, reason="ok")

    monkeypatch.setattr("pipeline.validate", fake_validate)
    monkeypatch.setattr(
        "pipeline.peer_check",
        lambda evidence_output, critical_output: PeerConsistencyResult(disagreement_score=0.9),
    )
    monkeypatch.setattr("pipeline.retrieve", lambda index, claim, k=5: [{"passage_id": "D1", "text": "t"}])
    run_pipeline("q", condition="C_proposed", retrieval_index=object())
    # validate() is called once for verify_fn's "peer flagged -> verify with sequential" path
    assert seen_contexts and seen_contexts[-1] == [{"passage_id": "D1", "text": "t"}]