"""
OWNER: Member 1 (experimental evaluation)

Runs the experiment grid
    questions x seeds x conditions (A/B/C) x (clean + fault types x severities)
through pipeline.run_pipeline, writes one JSON row per run, and computes the
metrics from Section 11 of the novelty doc (plus precision/recall/F1 and
false-positive rate from the project proposal).

Design points worth knowing:
  * Results are appended to a JSONL file as they finish, and finished runs are
    skipped on re-run, so an interrupted experiment resumes where it stopped.
  * The fault is seeded by (seed, question, fault, severity) -- NOT by
    condition -- so A, B and C see exactly the same corrupted output. That
    makes B-vs-C a paired comparison (McNemar test below).
  * A fault that leaves the output unchanged (e.g. truncate_context on a
    one-claim output) is a no-op. Such runs are flagged fault_effective=False
    and treated as clean, so they never count as a "missed detection".
  * final_correct / false_recovery need a verdict extractor (does the final
    report agree with the question's gold_label?). Pass judge_fn to run_grid
    when one exists; until then those metrics are None, not guessed.
  * Detection latency and propagation depth are coarse: the pipeline has one
    validated handoff, so a fault either is caught there or reaches synthesis.
    Multi-hop depth needs faults at more locations.

Usage (from the repo root):
    python -m evaluation.harness --split eval --n 10 --seeds 1
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
from collections import defaultdict
from typing import Callable, Optional

from fault_injection.injector import FAULT_REGISTRY, SEVERITIES, make_injector
from pipeline import run_pipeline

CONDITIONS = ("A_baseline", "B_sequential_only", "C_proposed")
CLEAN = "none"


# ---------------------------------------------------------------------------
# Overhead accounting
# ---------------------------------------------------------------------------

class CountingLLMClient:
    """Wraps any LLM client and counts calls and characters, so runtime and
    communication overhead can be compared across conditions."""

    def __init__(self, inner):
        self.inner = inner
        self.model = getattr(inner, "model", inner.__class__.__name__)
        self.calls = 0
        self.chars = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        out = self.inner.complete(system_prompt, user_prompt)
        self.calls += 1
        self.chars += len(system_prompt) + len(user_prompt) + len(out)
        return out


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_questions(questions: list[dict], split: str = "eval", n: int = 20, seed: int = 42) -> list[dict]:
    """Deterministic sample of up to n questions from `split` ("tune", "eval"
    or "all"), balanced across gold labels round-robin."""
    pool = [q for q in questions if split == "all" or q.get("split") == split]
    by_label: dict[str, list[dict]] = defaultdict(list)
    for q in sorted(pool, key=lambda q: q["qid"]):
        by_label[q["gold_label"]].append(q)
    rng = random.Random(seed)
    for label in by_label:
        rng.shuffle(by_label[label])
    labels = sorted(by_label)
    picked: list[dict] = []
    while len(picked) < n and any(by_label[l] for l in labels):
        for label in labels:
            if by_label[label] and len(picked) < n:
                picked.append(by_label[label].pop())
    return picked


# ---------------------------------------------------------------------------
# Running the grid
# ---------------------------------------------------------------------------

def _row_key(qid: str, condition: str, fault: str, severity: str, seed: int) -> str:
    return f"{qid}|{condition}|{fault}|{severity}|{seed}"


def load_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_grid(
    questions: list[dict],
    out_path: str,
    conditions=CONDITIONS,
    fault_names=None,
    severities=SEVERITIES,
    seeds=(0,),
    llm_factory: Optional[Callable[[int], object]] = None,
    corpus_passage_ids: Optional[list[str]] = None,
    retrieval_index=None,
    top_k: int = 5,
    judge_fn: Optional[Callable[[dict, dict], Optional[bool]]] = None,
    progress_every: int = 50,
) -> int:
    """Runs every not-yet-done cell of the grid, appending rows to out_path.
    Returns the number of NEW rows written.

    llm_factory(seed) -> an LLM client (default: MockLLMClient).
    judge_fn(final_report_dict, question) -> True/False/None: is the final
    answer correct? Optional; enables final_correct and false_recovery."""
    if fault_names is None:
        fault_names = list(FAULT_REGISTRY)
    if llm_factory is None:
        from llm_client import MockLLMClient
        llm_factory = lambda seed: MockLLMClient()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = {
        _row_key(r["qid"], r["condition"], r["fault_type"], r["severity"], r["seed"])
        for r in load_rows(out_path)
    }
    cells = [(CLEAN, "-")] + [(f, s) for f in fault_names for s in severities]
    new_rows = 0

    with open(out_path, "a", encoding="utf-8") as out:
        for q in questions:
            for seed in seeds:
                for condition in conditions:
                    for fault, severity in cells:
                        if _row_key(q["qid"], condition, fault, severity, seed) in done:
                            continue

                        llm = CountingLLMClient(llm_factory(seed))
                        state = {"effective": False}
                        injector = None
                        if fault != CLEAN:
                            base = make_injector(
                                fault, severity, corpus_passage_ids,
                                random.Random(f"{seed}|{q['qid']}|{fault}|{severity}"),
                            )

                            def injector(evidence_output, base=base, state=state):
                                mutated, record = base(evidence_output)
                                state["effective"] = mutated.model_dump() != evidence_output.model_dump()
                                return mutated, record

                        report, result = run_pipeline(
                            q["claim"],
                            condition=condition,
                            fault_injector_fn=injector,
                            llm_client=llm,
                            retrieval_index=retrieval_index,
                            top_k=top_k,
                        )

                        effective = state["effective"] if fault != CLEAN else False
                        correct = judge_fn(report, q) if judge_fn else None
                        row = {
                            "run_id": result.run_id,
                            "qid": q["qid"],
                            "gold_label": q.get("gold_label"),
                            "split": q.get("split"),
                            "seed": seed,
                            "condition": condition,
                            "fault_type": fault,
                            "severity": severity,
                            "fault_effective": effective,
                            "is_faulty": fault != CLEAN and effective,
                            "detected": result.detected,
                            "detected_by": result.detected_by,
                            "guard_decision": result.guard_decision,
                            "recovered": result.recovered,
                            "verified": result.verified,
                            "abstained": result.abstained,
                            "final_correct": correct,
                            "false_recovery": bool(result.recovered and result.verified and correct is False),
                            "disagreement_score": result.disagreement_score,
                            "validation_reason": result.validation_reason,
                            "model": result.model,
                            "latency_seconds": result.latency_seconds,
                            "llm_calls": llm.calls,
                            "llm_chars": llm.chars,
                        }
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        new_rows += 1
                        if progress_every and new_rows % progress_every == 0:
                            print(f"  ... {new_rows} runs done")
    return new_rows


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion k/n. (None, None) if n == 0."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts
    (b = only B detected, c = only C detected)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": [lo, hi]}


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _condition_metrics(rows: list[dict]) -> dict:
    faulty = [r for r in rows if r["is_faulty"]]
    clean = [r for r in rows if not r["is_faulty"]]  # true clean runs + no-op faults
    noop = [r for r in rows if r["fault_type"] != CLEAN and not r["fault_effective"]]

    tp = sum(1 for r in faulty if r["detected"])
    fp = sum(1 for r in clean if r["detected"])
    fn = len(faulty) - tp
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    detected_faulty = [r for r in faulty if r["detected"]]
    by_signal = {s: sum(1 for r in detected_faulty if r["detected_by"] == s) for s in ("sequential", "peer", "both")}

    recovered_ok = [r for r in detected_faulty if r["recovered"] and r["verified"]]
    judged_recovered = [r for r in recovered_ok if r["final_correct"] is not None]
    judged_all = [r for r in rows if r["final_correct"] is not None]

    def breakdown(field):
        groups = defaultdict(list)
        for r in faulty:
            groups[r[field]].append(r)
        return {k: _rate(sum(1 for r in v if r["detected"]), len(v)) for k, v in sorted(groups.items())}

    return {
        "n_runs": len(rows),
        "n_faulty": len(faulty),
        "n_clean": len(clean),
        "n_noop_faults_excluded": len(noop),
        "detection_rate": _rate(tp, len(faulty)),
        "detection_by_signal": by_signal,
        "detection_by_fault_type": breakdown("fault_type"),
        "detection_by_severity": breakdown("severity"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": _rate(fp, len(clean)),
        "recovery_success_rate": _rate(len(recovered_ok), len(detected_faulty)),
        "false_recovery_rate": _rate(sum(1 for r in judged_recovered if r["false_recovery"]), len(judged_recovered)) if judged_recovered else None,
        "abstention_rate_faulty": _rate(sum(1 for r in faulty if r["abstained"]), len(faulty)),
        "false_abstention_rate_clean": _rate(sum(1 for r in clean if r["abstained"]), len(clean)),
        "propagation_rate": _rate(len(faulty) - tp, len(faulty)),
        "final_task_failure_rate": _rate(sum(1 for r in judged_all if r["final_correct"] is False), len(judged_all)) if judged_all else None,
        "mean_latency_s_clean": _mean([r["latency_seconds"] for r in clean]),
        "mean_latency_s_faulty": _mean([r["latency_seconds"] for r in faulty]),
        "mean_llm_calls_clean": _mean([r["llm_calls"] for r in clean]),
        "mean_llm_calls_faulty": _mean([r["llm_calls"] for r in faulty]),
        "mean_llm_chars_clean": _mean([r["llm_chars"] for r in clean]),
    }


def compute_metrics(rows: list[dict]) -> dict:
    """Metrics per condition, plus the paired B-vs-C comparison."""
    by_condition = defaultdict(list)
    for r in rows:
        by_condition[r["condition"]].append(r)
    metrics = {cond: _condition_metrics(by_condition[cond]) for cond in CONDITIONS if cond in by_condition}

    def paired(rows_for_cond):
        return {
            (r["qid"], r["seed"], r["fault_type"], r["severity"]): r
            for r in rows_for_cond if r["is_faulty"]
        }

    if "B_sequential_only" in by_condition and "C_proposed" in by_condition:
        pb, pc = paired(by_condition["B_sequential_only"]), paired(by_condition["C_proposed"])
        keys = pb.keys() & pc.keys()
        only_b = sum(1 for k in keys if pb[k]["detected"] and not pc[k]["detected"])
        only_c = sum(1 for k in keys if pc[k]["detected"] and not pb[k]["detected"])
        both = sum(1 for k in keys if pb[k]["detected"] and pc[k]["detected"])
        neither = len(keys) - only_b - only_c - both
        metrics["paired_B_vs_C"] = {
            "n_pairs": len(keys),
            "both_detected": both,
            "only_B_detected": only_b,
            "only_C_detected": only_c,
            "neither_detected": neither,
            "mcnemar_exact_p": mcnemar_exact_p(only_b, only_c),
        }
    return metrics


def _pct(rate_dict) -> str:
    if not rate_dict or rate_dict["rate"] is None:
        return "n/a"
    lo, hi = rate_dict["ci95"]
    return f"{rate_dict['rate']*100:5.1f}% [{lo*100:.1f}-{hi*100:.1f}] ({rate_dict['k']}/{rate_dict['n']})"


def format_report(metrics: dict) -> str:
    lines = []
    for cond in CONDITIONS:
        m = metrics.get(cond)
        if not m:
            continue
        lines += [
            f"=== {cond} ===",
            f"runs: {m['n_runs']}  faulty: {m['n_faulty']}  clean: {m['n_clean']}  no-op faults excluded: {m['n_noop_faults_excluded']}",
            f"detection rate        : {_pct(m['detection_rate'])}",
            f"  by signal           : {m['detection_by_signal']}",
            f"false positive rate   : {_pct(m['false_positive_rate'])}",
            f"recovery success      : {_pct(m['recovery_success_rate'])}",
            f"abstention (faulty)   : {_pct(m['abstention_rate_faulty'])}",
            f"propagation rate      : {_pct(m['propagation_rate'])}",
            f"final task failure    : {_pct(m['final_task_failure_rate'])}",
            f"false recovery        : {_pct(m['false_recovery_rate'])}",
            f"mean LLM calls clean/faulty: {m['mean_llm_calls_clean']} / {m['mean_llm_calls_faulty']}",
            "  detection by fault type:",
        ]
        for name, rate in m["detection_by_fault_type"].items():
            lines.append(f"    {name:20s} {_pct(rate)}")
        lines.append("")
    if "paired_B_vs_C" in metrics:
        p = metrics["paired_B_vs_C"]
        lines += [
            "=== B vs C (paired, faulty runs) ===",
            f"pairs: {p['n_pairs']}  both: {p['both_detected']}  only B: {p['only_B_detected']}  "
            f"only C: {p['only_C_detected']}  neither: {p['neither_detected']}",
            f"McNemar exact p-value: {p['mcnemar_exact_p']:.4f}",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _make_llm_factory(name: str):
    if name == "mock":
        from llm_client import MockLLMClient
        return lambda seed: MockLLMClient()
    if name == "groq":
        from llm_client import GroqClient
        GroqClient()  # eager check: fail fast before a long run, not on run #1
        return lambda seed: GroqClient(seed=seed)
    raise NotImplementedError(
        f"LLM backend {name!r} is not implemented. Known backends: 'mock', 'groq'."
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the fault-injection experiment grid.")
    ap.add_argument("--data", default="mas_scifact_dataset.json")
    ap.add_argument("--split", default="eval", choices=["tune", "eval", "all"])
    ap.add_argument("--n", type=int, default=10, help="number of questions to sample")
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (0..seeds-1)")
    ap.add_argument("--faults", nargs="*", default=None, help="fault types (default: all)")
    ap.add_argument("--severities", nargs="*", default=list(SEVERITIES))
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--out", default="results/run.jsonl")
    ap.add_argument("--llm", default="mock")
    ap.add_argument("--retrieval", action="store_true", help="use the embedding retriever (downloads a model)")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args(argv)

    from dataset import load_dataset
    data = load_dataset(args.data)
    questions = sample_questions(data["questions"], args.split, args.n)
    corpus_ids = [p["passage_id"] for p in data["corpus"]]

    index = None
    if args.retrieval:
        from retrieval.retriever import build_index
        index = build_index(data["corpus"])

    faults = args.faults or list(FAULT_REGISTRY)
    per_question = len(args.conditions) * (1 + len(faults) * len(args.severities)) * args.seeds
    print(f"{len(questions)} questions x {per_question} runs each = {len(questions) * per_question} runs "
          f"(finished runs in {args.out} are skipped)")

    new = run_grid(
        questions, args.out,
        conditions=args.conditions, fault_names=faults, severities=args.severities,
        seeds=range(args.seeds), llm_factory=_make_llm_factory(args.llm),
        corpus_passage_ids=corpus_ids, retrieval_index=index, top_k=args.top_k,
    )
    print(f"{new} new runs written to {args.out}\n")

    metrics = compute_metrics(load_rows(args.out))
    print(format_report(metrics))
    metrics_path = os.path.splitext(args.out)[0] + "_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nmetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()