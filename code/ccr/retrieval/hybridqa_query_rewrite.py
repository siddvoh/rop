"""Provider-neutral direct query-rewrite baseline for HybridQA."""

from __future__ import annotations

import json
from typing import Any

from .hybridqa_bridge import render_table

QUERY_REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"search_query": {"type": "string"}},
    "required": ["search_query"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """Create one concise search query for retrieving the external
Wikipedia passage needed to answer a question using the supplied table. You
may use any table content or link shown. Do not answer the question. Return
only the required JSON object."""


def build_query_rewrite_messages(
    question: str, table: dict[str, Any]
) -> list[dict[str, str]]:
    user = f"""Question:
{question}

Table:
{render_table(table)}

Write the single search query most likely to retrieve the missing external
evidence. The query may name an entity, relation, or both. Do not provide the
final answer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_query_rewrite(text: str) -> str:
    value = json.loads(text)
    if set(value) != {"search_query"}:
        raise ValueError("query-rewrite response has unexpected fields")
    query = value["search_query"]
    if not isinstance(query, str) or not query.strip():
        raise ValueError("search_query must be a nonempty string")
    return query.strip()
