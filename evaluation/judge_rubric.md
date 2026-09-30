# Claim Verification Judge Rubric

This rubric is sent as the USER prompt to `llm_client.complete(system_prompt,
user_prompt)`. The system prompt (see `_JUDGE_SYSTEM_PROMPT` in judge.py)
sets the judge's role; this rubric carries the task definition, the input,
and the required output format.

## Definitions

- **SUPPORT**: The summary makes a claim that directly and correctly answers
  or addresses the question in the affirmative / confirmed direction.
- **CONTRADICT**: The summary makes a claim that directly opposes or negates
  what the question asks, or asserts the reverse of what would be expected
  given the question's premise.
- **NOINFO**: The summary does not take a clear position, is too vague,
  hedges without committing either way, or the information needed to decide
  SUPPORT/CONTRADICT is simply not present in the summary.

## Instructions

1. Read the question and the summary carefully.
2. Think step by step (briefly) about what the summary is actually
   claiming, and whether that claim, if taken at face value, would
   SUPPORT, CONTRADICT, or give NOINFO relative to the question.
3. Judge only what the summary *asserts* -- do not use outside knowledge
   to fact-check whether the claim is true in the world. That is a
   separate faithfulness check (see `faithfulness_score`), not this one.
4. Assign a confidence score from 0.0 (pure guess) to 1.0 (certain).
5. Output ONLY the JSON object below. No text before or after it, and no
   markdown code fences.

## Input

Question:
{question_text}

Summary:
{summary}

Supporting claims listed by the pipeline:
{supporting_claims}

## Required output format

Return exactly one JSON object, nothing else:

{{
  "reasoning": "<2-4 sentences of step-by-step reasoning>",
  "verdict": "SUPPORT" | "CONTRADICT" | "NOINFO",
  "confidence": <float between 0.0 and 1.0>,
  "rationale": "<one-sentence justification for the verdict>"
}}
