"""HybridQA answer prompts and retained experimental conditions.

This module deliberately does not retrieve passages or call a model.  It turns
one cohort item and the outputs of frozen retrieval stages into auditable
condition records and answer messages.  Keeping those responsibilities here
makes the differences among controls explicit and easy to test.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from ccr.retrieval.hybridqa_bridge import link_label, render_table

ANSWER_PROMPT_VERSION = "hybridqa-answer-v1"
CONDITIONS = (
    "question_only",
    "predicted_fact",
    "query_rewrite",
    "no_kb",
    "oracle_passage",
)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """Answer the question using the table and the supplied evidence.
Reason internally, but return only the required JSON object. Give the shortest
answer that completely answers the question. If the evidence is empty, use
only the table and your existing knowledge. Do not mention the evidence or
explain your answer."""

@dataclass(frozen=True)
class ConditionPlan:
    """Retrieval and prompt behavior for one answer condition."""

    name: str
    retrieval_query: str | None
    retrieval_source: str
    visible_bridge_fact: str | None = None
    forced_links: tuple[str, ...] = ()


def build_condition_plans(
    item: Mapping[str, Any],
    *,
    predicted_fact: str,
    query_rewrite: str,
) -> dict[str, ConditionPlan]:
    """Construct the five answer plans represented in the public release."""
    question = str(item["question"]).strip()
    gold_link = str(item["link"]).strip()
    predicted_fact = predicted_fact.strip()
    query_rewrite = query_rewrite.strip()
    if not all((question, gold_link, predicted_fact, query_rewrite)):
        raise ValueError("condition inputs must be nonempty")

    plans = {
        "question_only": ConditionPlan(
            "question_only", question, "question"
        ),
        "predicted_fact": ConditionPlan(
            "predicted_fact", predicted_fact, "predicted_fact"
        ),
        "query_rewrite": ConditionPlan(
            "query_rewrite", query_rewrite, "query_rewrite"
        ),
        "no_kb": ConditionPlan("no_kb", None, "none"),
        "oracle_passage": ConditionPlan(
            "oracle_passage", None, "oracle", forced_links=(gold_link,)
        ),
    }
    if tuple(plans) != CONDITIONS:
        raise AssertionError("condition ordering drifted")
    return plans


def plans_as_dicts(plans: Mapping[str, ConditionPlan]) -> dict[str, dict[str, Any]]:
    """Return JSON-serializable plans for manifests and task ledgers."""
    if tuple(plans) != CONDITIONS:
        raise ValueError("expected all conditions in canonical order")
    return {name: asdict(plan) for name, plan in plans.items()}


def materialize_evidence(
    plan: ConditionPlan,
    *,
    rankings: Mapping[str, Sequence[str]],
    passage_lookup: Callable[[str], str | None],
    top_k: int,
) -> list[dict[str, str]]:
    """Resolve a condition plan to ordered evidence without silent omissions."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if plan.retrieval_source == "none":
        return []
    links: Sequence[str]
    if plan.forced_links:
        links = plan.forced_links
    else:
        if plan.name not in rankings:
            raise KeyError(f"missing ranking for condition {plan.name}")
        links = rankings[plan.name][:top_k]
    evidence = []
    for link in links:
        passage = passage_lookup(link)
        if passage is None:
            raise KeyError(f"missing passage for ranked link {link}")
        evidence.append({"link": link, "title": link_label(link), "text": passage})
    return evidence


def render_evidence(
    evidence: Sequence[Mapping[str, str]], *, passage_char_limit: int | None = None
) -> str:
    """Render ranked passages with stable numbering and optional truncation."""
    if passage_char_limit is not None and passage_char_limit < 1:
        raise ValueError("passage_char_limit must be positive or None")
    if not evidence:
        return "(no external evidence)"
    blocks = []
    for index, passage in enumerate(evidence, start=1):
        text = str(passage["text"])
        if passage_char_limit is not None:
            text = text[:passage_char_limit]
        blocks.append(
            f"Passage {index} — {passage['title']} [{passage['link']}]\n{text}"
        )
    return "\n\n".join(blocks)


def build_answer_messages(
    *,
    question: str,
    table: Mapping[str, Any],
    evidence: Sequence[Mapping[str, str]],
    visible_bridge_fact: str | None = None,
    passage_char_limit: int | None = None,
) -> list[dict[str, str]]:
    """Build the model-visible prompt; gold answers are never accepted here."""
    if not question.strip():
        raise ValueError("question must be nonempty")
    hint = ""
    if visible_bridge_fact is not None:
        if not visible_bridge_fact.strip():
            raise ValueError("visible bridge fact must be nonempty when supplied")
        hint = f"\n\nSelected table bridge:\n{visible_bridge_fact.strip()}"
    user = f"""Question:
{question.strip()}

Table:
{render_table(dict(table))}{hint}

Evidence:
{render_evidence(evidence, passage_char_limit=passage_char_limit)}

Return a JSON object with one string field named answer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_answer_response(text: str) -> str:
    value = json.loads(text)
    if set(value) != {"answer"} or not isinstance(value["answer"], str):
        raise ValueError("answer response must contain exactly one string field")
    answer = value["answer"].strip()
    if not answer:
        raise ValueError("answer must be nonempty")
    return answer
