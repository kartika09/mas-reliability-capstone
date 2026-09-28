"""
Tests for evaluation/harness.py (Member 1's evaluation framework).

Run with: pytest tests/test_harness.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from schemas.models import ValidationResult
from evaluation.harness import (
    CONDITIONS,
    CountingLLMClient,
    sample_questions,
    wilson_ci,
    mcnemar_exact_p,
    compute_metrics,
    format_report,
    run_grid,
    load_rows,
)
from llm_client import MockLLMClient


def _questions(per_label=3, splits=("eval",)):
    qs = []
    for label in ("SUPPORT", "CONTRADICT", "NOINFO"):
        for i in range(per_label):
            qs.append({
                "qid": f"{label[0]}{i}",
                "claim": f"Claim {label} {i}",
                "gold_label": label,
                "split": splits[i % len(splits)],
            })
    return qs


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_sample_is_balanced_across_labels_and_deterministic():
    qs = _questions(per_label=4)
    a = sample_questions(qs, "eval", 6, seed=1)
    b = sample_questions(qs, "eval", 6, seed=1)
    assert [q["qid"] for q in a] == [q["qid"] for q in b]
    labels = [q["gold_label"] for q in a]
    assert {l: labels.count(l) for l in set(labels)} == {"SUPPORT": 2, "CONTRADICT": 2, "NOINFO": 2}


def test_sample_respects_split_and_pool_size():
    qs = _questions(per_label=4, splits=("tune", "eval"))
    tune = sample_questions(qs, "tune", 100)
    assert tune and all(q["split"] == "tune" for q in tune)
    assert len(tune) == 6  # only 2 tune questions per label exist
    assert len(sample_questions(qs, "all", 100)) == len(qs)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_wilson_ci_known_values():
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson_ci(10, 10)
    assert lo == pytest.approx(0.7225, abs=1e-3) and hi == 1.0
    lo, hi = wilson_ci(5, 10)
    assert (lo, hi) == (pytest.approx(0.2366, abs=1e-3), pytest.approx(0.7634, abs=1e-3))
    assert wilson_ci(0, 0) == (None, None)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact_p(5, 0) == pytest.approx(0.0625)
    assert mcnemar_exact_p(3, 3) == 1.0
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(0, 8) == pytest.approx(2 / 256)


# ---------------------------------------------------------------------------
# compute_metrics on hand-built rows
# ---------------------------------------------------------------------------

def _row(condition="C_proposed", fault="drop_field", severity="low", *, detected=False,
         detected_by="none", effective=True, recovered=False, verified=False,
         abstained=False, correct=None, qid="Q", seed=0, calls=4, latency=0.1):
    return {
        "run_id": "r", "qid": qid, "gold_label": "SUPPORT", "split": "eval", "seed": seed,
        "condition": condition, "fault_type": fault, "severity": severity,
        "fault_effective": effective if fault != "none" else False,
        "is_faulty": fault != "none" and effective,
        "detected": detected, "detected_by": detected_by,
        "guard_decision": "RECOVER" if detected else "ALLOW",
        "recovered": recovered, "verified": verified, "abstained": abstained,
        "final_correct": correct,
        "false_recovery": bool(recovered and verified and correct is False),
        "disagreement_score": 0.0, "validation_reason": "", "model": "m",
        "latency_seconds": latency, "llm_calls": calls, "llm_chars": 100,
    }


def test_compute_metrics_counts():
    rows = [
        # 4 faulty runs: 2 caught by sequential, 1 by peer, 1 missed
        _row(qid="1", detected=True, detected_by="sequential", recovered=True, verified=True),
        _row(qid="2", detected=True, detected_by="sequential", recovered=True, verified=True),
        _row(qid="3", detected=True, detected_by="peer", abstained=True, recovered=True, verified=False),
        _row(qid="4"),
        # 4 clean runs, one false positive
        _row(qid="5", fault="none", severity="-"),
        _row(qid="6", fault="none", severity="-"),
        _row(qid="7", fault="none", severity="-"),
        _row(qid="8", fault="none", severity="-", detected=True, detected_by="peer"),
        # a no-op fault: counts as clean, never as a missed detection
        _row(qid="9", fault="truncate_context", effective=False),
    ]
    m = compute_metrics(rows)["C_proposed"]
    assert (m["n_faulty"], m["n_clean"], m["n_noop_faults_excluded"]) == (4, 5, 1)
    assert m["detection_rate"]["k"] == 3 and m["detection_rate"]["n"] == 4
    assert m["detection_by_signal"] == {"sequential": 2, "peer": 1, "both": 0}
    assert m["false_positive_rate"]["k"] == 1 and m["false_positive_rate"]["n"] == 5
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(3 / 4)
    assert m["recovery_success_rate"]["k"] == 2 and m["recovery_success_rate"]["n"] == 3
    assert m["propagation_rate"]["k"] == 1
    assert m["abstention_rate_faulty"]["k"] == 1
    assert m["final_task_failure_rate"] is None      # no judge -> not guessed
    assert m["false_recovery_rate"] is None


def test_false_recovery_and_task_failure_use_judged_rows_only():
    rows = [
        _row(qid="1", detected=True, detected_by="peer", recovered=True, verified=True, correct=False),
        _row(qid="2", detected=True, detected_by="peer", recovered=True, verified=True, correct=True),
        _row(qid="3", detected=True, detected_by="peer", recovered=True, verified=True, correct=None),
    ]
    m = compute_metrics(rows)["C_proposed"]
    assert m["false_recovery_rate"]["k"] == 1 and m["false_recovery_rate"]["n"] == 2
    assert m["final_task_failure_rate"]["k"] == 1 and m["final_task_failure_rate"]["n"] == 2


def test_paired_b_vs_c_counts():
    rows = []
    # (qid, B detected, C detected)
    for qid, b, c in [("1", True, True), ("2", False, True), ("3", False, True),
                      ("4", True, False), ("5", False, False)]:
        rows.append(_row("B_sequential_only", qid=qid, detected=b, detected_by="sequential" if b else "none"))
        rows.append(_row("C_proposed", qid=qid, detected=c, detected_by="both" if c else "none"))
    p = compute_metrics(rows)["paired_B_vs_C"]
    assert (p["n_pairs"], p["both_detected"], p["only_B_detected"], p["only_C_detected"], p["neither_detected"]) == (5, 1, 1, 2, 1)
    assert p["mcnemar_exact_p"] == 1.0


def test_format_report_runs_on_metrics():
    rows = [_row(qid="1", detected=True, detected_by="peer"), _row(qid="2", fault="none", severity="-")]
    text = format_report(compute_metrics(rows))
    assert "C_proposed" in text and "detection rate" in text


# ---------------------------------------------------------------------------
# Counting client
# ---------------------------------------------------------------------------

def test_counting_client_counts_calls_and_keeps_model_name():
    c = CountingLLMClient(MockLLMClient())
    assert c.model == "MockLLMClient"
    c.complete("You are the Planner agent", "q")
    assert c.calls == 1 and c.chars > 0


# ---------------------------------------------------------------------------
# run_grid end to end (mock LLM)
# ---------------------------------------------------------------------------

def _run(tmp_path, monkeypatch, **kwargs):
    monkeypatch.chdir(tmp_path)  # pipeline writes logs/ relative to the cwd
    out = str(tmp_path / "results" / "run.jsonl")
    params = dict(
        questions=_questions(per_label=1)[:2],
        out_path=out,
        fault_names=["drop_field", "truncate_context"],
        severities=["low", "high"],
        seeds=[0],
        corpus_passage_ids=["D1", "D2", "D3"],
        progress_every=0,
    )
    params.update(kwargs)
    new = run_grid(**params)
    return new, out


def test_run_grid_row_count_and_fields(tmp_path, monkeypatch):
    new, out = _run(tmp_path, monkeypatch)
    rows = load_rows(out)
    # 2 questions x 1 seed x 3 conditions x (1 clean + 2 faults x 2 severities)
    assert new == len(rows) == 2 * 1 * 3 * 5
    assert {r["condition"] for r in rows} == set(CONDITIONS)
    clean = [r for r in rows if r["fault_type"] == "none"]
    assert clean and all(not r["is_faulty"] for r in clean)
    assert all(r["llm_calls"] == 4 for r in clean)  # planner, evidence, critical, synthesis


def test_run_grid_flags_noop_faults(tmp_path, monkeypatch):
    _, out = _run(tmp_path, monkeypatch)
    rows = load_rows(out)
    # mock Evidence returns ONE claim: dropping its citation changes it, truncating cannot
    assert all(r["fault_effective"] for r in rows if r["fault_type"] == "drop_field")
    assert not any(r["fault_effective"] for r in rows if r["fault_type"] == "truncate_context")
    assert not any(r["is_faulty"] for r in rows if r["fault_type"] == "truncate_context")


def test_run_grid_resumes_without_duplicating(tmp_path, monkeypatch):
    first, out = _run(tmp_path, monkeypatch)
    second, _ = _run(tmp_path, monkeypatch)
    assert first > 0 and second == 0
    assert len(load_rows(out)) == first


def test_stub_detectors_give_zero_detection(tmp_path, monkeypatch):
    _, out = _run(tmp_path, monkeypatch)
    m = compute_metrics(load_rows(out))
    assert m["C_proposed"]["detection_rate"]["k"] == 0
    assert m["C_proposed"]["false_positive_rate"]["k"] == 0


def test_detection_flows_from_pipeline_into_rows(tmp_path, monkeypatch):
    # Pretend the sequential validator flags everything.
    monkeypatch.setattr(
        "pipeline.validate",
        lambda planner_output, agent_output, agent_name: ValidationResult(
            valid=False, reason="forced", failed_checks=["structural"]
        ),
    )
    _, out = _run(tmp_path, monkeypatch, fault_names=["drop_field"], severities=["low"])
    rows = load_rows(out)
    by_cond = {c: [r for r in rows if r["condition"] == c] for c in CONDITIONS}
    assert not any(r["detected"] for r in by_cond["A_baseline"])          # baseline never checks
    for cond in ("B_sequential_only", "C_proposed"):
        assert all(r["detected"] and r["detected_by"] == "sequential" for r in by_cond[cond])
        assert all(r["recovered"] and r["verified"] for r in by_cond[cond])
    m = compute_metrics(rows)
    assert m["B_sequential_only"]["detection_rate"]["rate"] == 1.0
    assert m["B_sequential_only"]["false_positive_rate"]["rate"] == 1.0   # it flagged clean runs too
    assert m["A_baseline"]["detection_rate"]["rate"] == 0.0


def test_judge_fn_populates_final_correct_and_false_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pipeline.validate",
        lambda planner_output, agent_output, agent_name: ValidationResult(valid=False, reason="forced"),
    )
    _, out = _run(
        tmp_path, monkeypatch, fault_names=["drop_field"], severities=["low"],
        judge_fn=lambda report, question: False,
    )
    rows = load_rows(out)
    assert all(r["final_correct"] is False for r in rows)
    recovered = [r for r in rows if r["recovered"] and r["verified"]]
    assert recovered and all(r["false_recovery"] for r in recovered)
    m = compute_metrics(rows)
    assert m["C_proposed"]["final_task_failure_rate"]["rate"] == 1.0