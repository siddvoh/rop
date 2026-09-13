"""Shared HybridQA table rendering and bridge-selection prompt."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import unquote

BRIDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_link": {"type": "string"},
        "retrieval_fact": {"type": "string"},
    },
    "required": ["selected_link", "retrieval_fact"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You select a bridge entity from a table. The bridge entity is the one
table-linked Wikipedia page whose passage is needed to answer the question.
Do not answer the question. Use the table and the question to choose exactly
one link that is explicitly shown. Return only the required JSON object."""


def link_label(link: str) -> str:
    value = unquote(link.removeprefix("/wiki/")).replace("_", " ")
    return " ".join(value.split())


def render_table(table: dict[str, Any]) -> str:
    headers = [str(cell[0]) for cell in table.get("header", [])]
    lines = [f"Title: {table.get('title', '')}", "Columns: " + " | ".join(headers)]
    for row_index, row in enumerate(table.get("data", []), start=1):
        rendered_cells = []
        for column_index, cell in enumerate(row):
            header = headers[column_index] if column_index < len(headers) else f"column_{column_index + 1}"
            text = str(cell[0])
            links = [str(link) for link in cell[1]]
            link_text = " | ".join(f"{link_label(link)} => {link}" for link in links)
            suffix = f" [links: {link_text}]" if links else ""
            rendered_cells.append(f"{header}: {text}{suffix}")
        lines.append(f"Row {row_index}: " + " || ".join(rendered_cells))
    return "\n".join(lines)


def build_bridge_messages(question: str, table: dict[str, Any]) -> list[dict[str, str]]:
    user = f"""Question:
{question}

Table:
{render_table(table)}

Choose the single shown link that best bridges the table to the external
information needed for the answer. Set selected_link to the exact /wiki/...
value shown. Set retrieval_fact to a concise human-readable name for that same
entity, suitable as a search query. Do not include the final answer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_bridge_response(text: str) -> dict[str, str]:
    value = json.loads(text)
    if set(value) != {"selected_link", "retrieval_fact"}:
        raise ValueError("bridge response has unexpected fields")
    if not all(isinstance(value[key], str) and value[key].strip() for key in value):
        raise ValueError("bridge response fields must be nonempty strings")
    return {key: value[key].strip() for key in value}
