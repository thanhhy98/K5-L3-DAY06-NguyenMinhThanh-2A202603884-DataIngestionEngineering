"""Evaluate retrieved evidence and preserve per-query traces."""
from __future__ import annotations

from typing import Any


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check source/page and enough gold evidence; keep hit@1 and hit@3 trace."""
    gold_source = question.get("gold_source_id") or question.get("source_id")
    gold_page = question.get("gold_page")
    gold_evidence = question.get("evidence_quote") or question.get("gold_evidence") or ""
    gold_page_list = question.get("gold_pages") or []
    if gold_page is not None and isinstance(gold_page_list, list):
        gold_page_list = [gold_page, *gold_page_list]

    if gold_page is not None and isinstance(gold_page_list, list) and gold_page not in gold_page_list:
        gold_page_list.append(gold_page)

    hit_at_1 = False
    evidence_hit_3 = False
    top_hit = None

    for index, hit in enumerate(hits[:3]):
        text = (hit.get("text") or "")
        source = hit.get("source") or (hit.get("metadata") or {}).get("source")
        page = hit.get("page") or (hit.get("metadata") or {}).get("page")

        page_match = bool(page is not None and (page == gold_page or page in gold_page_list))
        source_match = bool(gold_source and source and source == gold_source)
        quote_match = bool(gold_evidence and gold_evidence.lower() in text.lower())

        if index == 0 and (page_match or source_match or quote_match):
            hit_at_1 = True
        if page_match or source_match or quote_match:
            evidence_hit_3 = True

        if top_hit is None:
            top_hit = {"chunk_id": hit.get("chunk_id"), "source": source, "page": page, "score": hit.get("score")}

    return {
        "question_id": question.get("id"),
        "split": question.get("split"),
        "strategy": question.get("strategy"),
        "hit_at_1": int(hit_at_1),
        "evidence_hit_3": int(evidence_hit_3),
        "top_hit": top_hit,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate hit@1 and evidence hit@3 by strategy and dev/test split."""
    grouped: dict[str, dict[str, dict[str, float]]] = {}
    for trace in traces:
        strategy = str(trace.get("strategy") or "unknown")
        split = str(trace.get("split") or "unknown")
        bucket = grouped.setdefault(strategy, {}).setdefault(split, {"hit_at_1": [], "evidence_hit_3": []})
        bucket["hit_at_1"].append(float(trace.get("hit_at_1", 0)))
        bucket["evidence_hit_3"].append(float(trace.get("evidence_hit_3", 0)))

    summary: dict[str, Any] = {}
    for strategy, splits in grouped.items():
        summary[strategy] = {}
        for split, rows in splits.items():
            total = max(len(rows["hit_at_1"]), 1)
            summary[strategy][split] = {
                "hit_at_1": round(sum(rows["hit_at_1"]) / total, 3),
                "evidence_hit_3": round(sum(rows["evidence_hit_3"]) / total, 3),
                "questions": total,
            }
    return summary
