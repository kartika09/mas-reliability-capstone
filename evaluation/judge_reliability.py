"""
evaluation/judge_reliability.py

Meta-evaluation of the judge itself: is it accurate, stable, and unbiased?

Run these functions against a *fixed* reference set of (report, question)
pairs your team has already collected -- not against fresh live pipeline
runs each time, since several of these checks (repeated_run_agreement,
position_bias_test) specifically need to hold the input constant and vary
only one thing (repetition, claim order) to isolate what they're measuring.

Uses a judge model from a DIFFERENT family than the pipeline's own agents
wherever possible, particularly for cross_judge_agreement and
self_preference_bias_test -- judging your own family's outputs with a
same-family judge is exactly the confound self_preference_bias_test exists
to catch.

Requires: scikit-learn (for cohen_kappa_score)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from sklearn.metrics import cohen_kappa_score

JudgeFn = Callable[[Dict[str, Any], Dict[str, Any]], Optional[bool]]


def accuracy(
    judge_fn: JudgeFn,
    reports: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Accuracy of the judge's True/False output against gold_label,
    excluding abstained/ungraded (None) cases from the denominator.
    """
    total = 0
    correct = 0
    ungraded = 0

    for report, question in zip(reports, questions):
        result = judge_fn(report, question)
        if result is None:
            ungraded += 1
            continue
        total += 1
        if result:
            correct += 1

    return {
        "accuracy": correct / total if total else float("nan"),
        "graded": total,
        "ungraded": ungraded,
        "n": len(reports),
    }


def repeated_run_agreement(
    judge_fn: JudgeFn,
    reports: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    n_runs: int = 2,
) -> Dict[str, Any]:
    """
    Runs the SAME judge on the SAME (report, question) pairs multiple times
    and measures Cohen's kappa between the first two runs (self-consistency).

    Most informative when the judge is non-deterministic (temperature > 0).
    If the judge is deterministic (temperature 0, greedy decoding), kappa
    should come back as 1.0 -- that itself is worth confirming, not just
    assuming.
    """
    if n_runs < 2:
        raise ValueError("n_runs must be >= 2")

    all_runs: List[List[Optional[bool]]] = []
    for _ in range(n_runs):
        run_results = [judge_fn(r, q) for r, q in zip(reports, questions)]
        all_runs.append(run_results)

    run_a, run_b = all_runs[0], all_runs[1]
    paired = [(a, b) for a, b in zip(run_a, run_b) if a is not None and b is not None]

    if len(paired) < 2:
        return {"kappa": float("nan"), "n_paired": len(paired), "note": "not enough graded pairs"}

    labels_a = [a for a, _ in paired]
    labels_b = [b for _, b in paired]
    kappa = cohen_kappa_score(labels_a, labels_b)

    return {"kappa": kappa, "n_paired": len(paired), "n_total": len(reports)}


def cross_judge_agreement(
    judge_fn_a: JudgeFn,
    judge_fn_b: JudgeFn,
    reports: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Agreement between two DIFFERENT judge models (e.g. a GPT-4-based judge
    vs a Claude-based judge), built via two separate make_judge(llm_client)
    calls with different clients, run on the same (report, question) pairs.
    """
    results_a = [judge_fn_a(r, q) for r, q in zip(reports, questions)]
    results_b = [judge_fn_b(r, q) for r, q in zip(reports, questions)]

    paired = [(a, b) for a, b in zip(results_a, results_b) if a is not None and b is not None]
    if len(paired) < 2:
        return {"kappa": float("nan"), "n_paired": len(paired), "note": "not enough graded pairs"}

    labels_a = [a for a, _ in paired]
    labels_b = [b for _, b in paired]
    kappa = cohen_kappa_score(labels_a, labels_b)
    agreement_pct = sum(1 for a, b in paired if a == b) / len(paired)

    return {"kappa": kappa, "agreement_pct": agreement_pct, "n_paired": len(paired)}


def position_bias_test(
    judge_fn: JudgeFn,
    reports: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Checks whether reordering `supporting_claims` inside the report changes
    the judge's verdict. A genuinely content-based verdict should be
    invariant to claim order; a verdict that flips on reorder alone is
    position bias (see MT-Bench, JudgeLM).

    Only reports with 2+ claims are usable for this test; others are
    skipped (not counted as "no bias").
    """
    flips = 0
    compared = 0

    for report, question in zip(reports, questions):
        claims = report.get("supporting_claims") or []
        if len(claims) < 2 or report.get("abstained"):
            continue

        original = dict(report)
        reversed_report = dict(report)
        reversed_report["supporting_claims"] = list(reversed(claims))

        verdict_original = judge_fn(original, question)
        verdict_reversed = judge_fn(reversed_report, question)

        if verdict_original is None or verdict_reversed is None:
            continue

        compared += 1
        if verdict_original != verdict_reversed:
            flips += 1

    return {
        "flip_rate": flips / compared if compared else float("nan"),
        "flips": flips,
        "compared": compared,
    }


def self_preference_bias_test(
    judge_fn: JudgeFn,
    reports_same_family: List[Dict[str, Any]],
    reports_other_family: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Checks whether the judge scores reports MORE favorably when they were
    generated by an agent from the judge's own model family, versus a
    different family, given matched questions.

    reports_same_family / reports_other_family: two lists of reports
    answering the SAME questions (same order, same length as `questions`),
    ideally matched for comparable objective quality so the only real
    difference is which model family produced them.

    A large positive gap suggests self-preference bias (see JudgeLM,
    MT-Bench): the judge is not evaluating content neutrally.
    """
    acc_same = accuracy(judge_fn, reports_same_family, questions)
    acc_other = accuracy(judge_fn, reports_other_family, questions)

    return {
        "judge_accuracy_on_same_family_reports": acc_same["accuracy"],
        "judge_accuracy_on_other_family_reports": acc_other["accuracy"],
        "gap": acc_same["accuracy"] - acc_other["accuracy"],
        "note": (
            "A large positive gap suggests the judge favors reports from its "
            "own model family (self-preference bias), independent of actual "
            "report quality. A gap near zero suggests the judge is roughly "
            "neutral with respect to which model produced the report."
        ),
    }
