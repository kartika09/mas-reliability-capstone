"""
Unit tests for evaluation/judge.py using a fake LLM client defined here,
so nobody has to edit the shared llm_client.py to run these.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.judge import faithfulness_score, make_judge


class FakeLLMClient:
    """
    Stand-in for the real llm_client, matching the team's LLMClient Protocol
    (llm_client.py): complete(system_prompt, user_prompt) -> str.
    Returns a scripted response regardless of prompt content -- or cycles
    through a list of scripted responses, one per call, if given a list.
    """

    def __init__(self, response_or_responses):
        if isinstance(response_or_responses, list):
            self._responses = list(response_or_responses)
        else:
            self._responses = [response_or_responses]
        self._call_count = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return response


def _verdict_response(verdict: str, confidence: float = 0.9) -> str:
    return json.dumps(
        {
            "reasoning": "test reasoning",
            "verdict": verdict,
            "confidence": confidence,
            "rationale": "test rationale",
        }
    )


def _make_report(summary="Test summary.", claims=None, abstained=False):
    if abstained:
        return {"summary": None, "supporting_claims": None, "abstained": True}
    return {"summary": summary, "supporting_claims": claims if claims is not None else ["claim 1", "claim 2"]}


def _make_question(gold_label="SUPPORT"):
    # NOTE: the real dataset's question dicts use "claim", not "question" or
    # "text" -- this must match schemas produced by dataset.load_dataset(),
    # or these tests pass while the judge silently sees a blank question
    # against real data.
    return {
        "claim": "Does the evidence support the claim?",
        "gold_label": gold_label,
        "gold_passages": ["P1", "P2"],
        "gold_sentences": {"P1": [0]},
    }


def _make_passages_lookup():
    # Shape matches dataset.load_dataset(...)['passages']: passage_id ->
    # {"passage_id", "title", "sentences", "text"}.
    return {
        "P1": {"passage_id": "P1", "title": "t1", "sentences": ["P1 sentence 0.", "P1 sentence 1."], "text": "P1 full text."},
        "P2": {"passage_id": "P2", "title": "t2", "sentences": ["P2 sentence 0."], "text": "P2 full text."},
    }


def test_judge_returns_true_when_verdict_matches_gold():
    client = FakeLLMClient(_verdict_response("SUPPORT"))
    judge_fn = make_judge(client)

    result = judge_fn(_make_report(), _make_question(gold_label="SUPPORT"))

    assert result is True


def test_judge_returns_false_when_verdict_mismatches_gold():
    client = FakeLLMClient(_verdict_response("CONTRADICT"))
    judge_fn = make_judge(client)

    result = judge_fn(_make_report(), _make_question(gold_label="SUPPORT"))

    assert result is False


def test_judge_returns_none_for_abstained_report():
    client = FakeLLMClient(_verdict_response("SUPPORT"))
    judge_fn = make_judge(client)

    result = judge_fn(_make_report(abstained=True), _make_question())

    assert result is None


def test_judge_returns_none_on_unparseable_response():
    client = FakeLLMClient("not valid json at all")
    judge_fn = make_judge(client)

    result = judge_fn(_make_report(), _make_question())

    assert result is None


def test_judge_returns_none_on_invalid_verdict_label():
    client = FakeLLMClient(_verdict_response("MAYBE"))
    judge_fn = make_judge(client)

    result = judge_fn(_make_report(), _make_question())

    assert result is None


def test_judge_handles_noinfo_correctly():
    client = FakeLLMClient(_verdict_response("NOINFO"))
    judge_fn = make_judge(client)

    result = judge_fn(_make_report(), _make_question(gold_label="NOINFO"))

    assert result is True


def test_judge_is_case_insensitive_on_gold_label():
    client = FakeLLMClient(_verdict_response("SUPPORT"))
    judge_fn = make_judge(client)

    result = judge_fn(_make_report(), _make_question(gold_label="support"))

    assert result is True


def test_judge_sends_the_real_claim_text_to_the_llm():
    # Regression test: earlier code read question["question"]/["text"], but
    # the real dataset (dataset.load_dataset()) only has question["claim"].
    # That mismatch let the judge silently send a BLANK question to the LLM
    # against real data, while these tests still passed against fake data
    # shaped differently. This test builds the actual prompt and checks the
    # claim text is really in it.
    from evaluation.judge import _build_user_prompt, _load_rubric

    question = _make_question()
    prompt = _build_user_prompt(_make_report(), question, _load_rubric())
    assert question["claim"] in prompt


def test_faithfulness_score_returns_fraction_supported():
    client = FakeLLMClient(json.dumps({"claim_verdicts": [True, False]}))

    report = _make_report(claims=["claim 1", "claim 2"])
    question = _make_question()

    score = faithfulness_score(report, question, client, passages_lookup=_make_passages_lookup())

    assert score == pytest.approx(0.5)


def test_faithfulness_score_returns_none_when_abstained():
    client = FakeLLMClient(json.dumps({"claim_verdicts": [True]}))

    result = faithfulness_score(
        _make_report(abstained=True), _make_question(), client,
        passages_lookup=_make_passages_lookup(),
    )

    assert result is None


def test_faithfulness_score_returns_none_when_no_claims():
    client = FakeLLMClient(json.dumps({"claim_verdicts": []}))

    report = _make_report(claims=[])
    result = faithfulness_score(report, _make_question(), client, passages_lookup=_make_passages_lookup())
    assert result is None


def test_faithfulness_score_returns_none_on_mismatched_length():
    # judge returned 1 verdict but report has 2 claims -- malformed response
    client = FakeLLMClient(json.dumps({"claim_verdicts": [True]}))

    report = _make_report(claims=["claim 1", "claim 2"])
    result = faithfulness_score(report, _make_question(), client, passages_lookup=_make_passages_lookup())
    assert result is None


def test_faithfulness_score_returns_none_without_passages_lookup():
    # gold_passages are bare IDs like "D9394119", not text.
    # Without a lookup to resolve them, there's nothing real to show the
    # judge, so this must return None rather than sending the judge raw IDs.
    client = FakeLLMClient(json.dumps({"claim_verdicts": [True]}))
    report = _make_report(claims=["claim 1"])
    result = faithfulness_score(report, _make_question(), client)  # no passages_lookup
    assert result is None


def test_faithfulness_score_uses_exact_gold_sentences_when_available():
    # question["gold_sentences"] = {"P1": [0]} -> only sentence index 0 of
    # P1 should be shown, not P1's full text, and P2 has no gold_sentences
    # entry so its full text is used instead.
    seen_prompts = []

    class RecordingClient:
        def complete(self, system_prompt, user_prompt):
            seen_prompts.append(user_prompt)
            return json.dumps({"claim_verdicts": [True]})

    report = _make_report(claims=["claim 1"])
    question = _make_question()
    faithfulness_score(report, question, RecordingClient(), passages_lookup=_make_passages_lookup())

    prompt = seen_prompts[0]
    assert "P1 sentence 0." in prompt
    assert "P1 sentence 1." not in prompt  # only the gold sentence, not the rest of P1
    assert "P2 full text." in prompt       # P2 has no gold_sentences entry -> falls back to full text