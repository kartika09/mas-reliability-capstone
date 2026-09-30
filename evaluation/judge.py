"""
evaluation/judge.py

LLM-as-a-Judge for claim verification against gold labels.

Confirmed against the team's real llm_client.py (LLMClient Protocol):
    llm_client.complete(system_prompt: str, user_prompt: str) -> str

`final_report["supporting_claims"]` is a list of AgentClaim shapes
(schemas/models.py): {"claim": str, "supporting_reference": Optional[str]}.
It may arrive as pydantic model instances OR plain dicts depending on
whether the caller did a .model_dump() before invoking judge_fn -- this
module handles both (see `_claim_text`).

Public API:
    make_judge(llm_client) -> judge_fn
    judge_fn(final_report: dict, question: dict) -> Optional[bool]
    faithfulness_score(final_report, question, llm_client, passages_lookup) -> Optional[float]
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

VALID_VERDICTS = {"SUPPORT", "CONTRADICT", "NOINFO"}
RUBRIC_PATH = Path(__file__).parent / "judge_rubric.md"

_JUDGE_SYSTEM_PROMPT = (
    "You are an impartial fact-verification judge for a research-question "
    "answering pipeline."
)


def _load_rubric() -> str:
    return RUBRIC_PATH.read_text(encoding="utf-8")


def _call_llm(llm_client, system_prompt: str, user_prompt: str) -> str:
    """Single point of contact with the LLM client, matching the team's
    LLMClient Protocol: complete(system_prompt, user_prompt) -> str."""
    return llm_client.complete(system_prompt, user_prompt)


def _claim_text(claim: Any) -> str:
    """AgentClaim may arrive as a pydantic model instance or a plain dict
    (e.g. after .model_dump()) -- handle both without assuming which."""
    if isinstance(claim, dict):
        return claim.get("claim", "")
    return getattr(claim, "claim", str(claim))


def _build_user_prompt(final_report: Dict[str, Any], question: Dict[str, Any], rubric: str) -> str:
    summary = final_report.get("summary", "") or ""
    supporting_claims = final_report.get("supporting_claims") or []

    claims_block = (
        "\n".join(f"- {_claim_text(c)}" for c in supporting_claims)
        if supporting_claims
        else "(none provided)"
    )

    return rubric.format(
        question_text=question.get("claim", ""),
        summary=summary,
        supporting_claims=claims_block,
    )


def _extract_json(text: str) -> Dict[str, Any]:
    """Pull the first {...} JSON object out of a model response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text!r}")
    return json.loads(match.group(0))


def _parse_verdict(raw_response: str) -> Dict[str, Any]:
    parsed = _extract_json(raw_response)
    verdict = str(parsed.get("verdict", "")).strip().upper()
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"Judge returned invalid verdict: {verdict!r} (raw: {raw_response!r})")
    return {
        "verdict": verdict,
        "rationale": parsed.get("rationale", ""),
        "reasoning": parsed.get("reasoning", ""),
        "confidence": parsed.get("confidence"),
    }


def make_judge(llm_client) -> Callable[[Dict[str, Any], Dict[str, Any]], Optional[bool]]:
    """
    Returns a judge_fn(final_report, question) -> Optional[bool] closure
    bound to the given llm_client, matching the harness's expected
    signature for `judge_fn` passed into `run_grid`.

    Return semantics:
        True  -> judge's extracted verdict matches question["gold_label"]
        False -> judge's extracted verdict does not match gold_label
        None  -> report was abstained, OR the judge's response could not
                 be parsed into a valid verdict (ungraded, not "wrong")
    """
    rubric = _load_rubric()

    def judge_fn(final_report: Dict[str, Any], question: Dict[str, Any]) -> Optional[bool]:
        # Abstained pipeline runs are not graded -- return None per team agreement.
        # NOTE (flagged for team confirmation): this treats "abstained" as the
        # only report-level case returning None. A judge response that fails
        # to parse (see except branch below) is a *separate* reason for None
        # and should probably be logged/counted separately in practice so it
        # doesn't get silently conflated with legitimate abstentions when you
        # compute accuracy in judge_reliability.py.
        if final_report.get("abstained"):
            return None

        user_prompt = _build_user_prompt(final_report, question, rubric)
        raw_response = _call_llm(llm_client, _JUDGE_SYSTEM_PROMPT, user_prompt)

        try:
            result = _parse_verdict(raw_response)
        except ValueError:
            # Judge failed to produce a parseable verdict -- ungraded, not
            # silently marked wrong. Worth logging raw_response in a real run.
            return None

        gold_label = str(question.get("gold_label", "")).strip().upper()
        return result["verdict"] == gold_label

    return judge_fn


def faithfulness_score(
    final_report: Dict[str, Any],
    question: Dict[str, Any],
    llm_client,
    passages_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[float]:
    """
    Optional secondary check: are the claims in final_report actually
    supported by the cited gold passages, independent of whether the
    final verdict matches gold_label?

    This is separate from the SUPPORT/CONTRADICT/NOINFO verdict -- a report
    can get the right verdict for the wrong (unsupported) reasons, and this
    catches that.

    `passages_lookup`: dataset.load_dataset(...)["passages"], a dict of
    passage_id -> {"passage_id", "title", "sentences", "text"}.
    question["gold_passages"] is a list of passage IDs (e.g. "D9394119"),
    NOT text -- without this lookup there is nothing to show the judge, so
    the check returns None rather than sending the judge bare IDs.

    When question["gold_sentences"] gives the exact evidence sentence
    indices for a passage_id (e.g. {"D9394119": [16, 17]}), those specific
    sentences are used instead of the full abstract -- that's the precise
    evidence SciFact annotators identified, so it's a tighter check than
    the whole abstract.

    Returns a score in [0, 1] (fraction of claims judged supported by the
    gold passages), or None if there's nothing to check (abstained report,
    no claims listed, no gold passages for this question, or no
    passages_lookup provided to resolve them to text).
    """
    if final_report.get("abstained"):
        return None

    claims = final_report.get("supporting_claims") or []
    passage_ids = question.get("gold_passages") or []
    if not claims or not passage_ids or not passages_lookup:
        return None

    gold_sentences = question.get("gold_sentences") or {}
    passage_texts = []
    for pid in passage_ids:
        passage = passages_lookup.get(pid)
        if not passage:
            continue
        sentence_idxs = gold_sentences.get(pid)
        sentences = passage.get("sentences") or []
        if sentence_idxs and sentences:
            text = " ".join(
                sentences[i] for i in sentence_idxs if 0 <= i < len(sentences)
            )
        else:
            text = passage.get("text") or " ".join(sentences)
        if text:
            passage_texts.append(f"[{pid}] {text}")

    if not passage_texts:
        return None

    passages_block = "\n\n".join(passage_texts)
    claims_block = "\n".join(f"{i}. {_claim_text(c)}" for i, c in enumerate(claims))

    system_prompt = (
        "You are checking whether each claim below is directly supported by "
        "the provided source passages. A claim is SUPPORTED only if the "
        "passages state it explicitly or it follows by direct entailment -- "
        "not by plausibility or general knowledge."
    )
    user_prompt = (
        f"SOURCE PASSAGES:\n{passages_block}\n\n"
        f"CLAIMS:\n{claims_block}\n\n"
        "Return ONLY a JSON object: "
        '{"claim_verdicts": [true, false, ...]} '
        "with one boolean per claim, in the same order as listed above, "
        "true = supported by the passages, false = not supported."
    )

    raw_response = _call_llm(llm_client, system_prompt, user_prompt)
    try:
        parsed = _extract_json(raw_response)
        verdicts = parsed["claim_verdicts"]
        if len(verdicts) != len(claims):
            return None
    except (ValueError, KeyError, TypeError):
        return None

    return sum(1 for v in verdicts if v) / len(verdicts)