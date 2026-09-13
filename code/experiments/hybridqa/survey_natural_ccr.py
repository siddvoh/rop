"""Read-only feasibility survey for automatically derived natural CCR cases.

The script uses only the original HybridQA traced questions and the linked
WikiTables artifacts. It does not call a model and does not create labels.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def norm(text: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def table_cells(table: dict) -> list[str]:
    return [str(cell[0]) for row in table.get("data", []) for cell in row]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--passages", type=Path, required=True)
    ap.add_argument("--output-candidates", type=Path)
    args = ap.parse_args()

    questions = json.loads(args.questions.read_text())
    counts: Counter[str] = Counter()
    examples: list[dict] = []
    all_candidates: list[dict] = []
    selection_words = re.compile(
        r"\b(most|least|highest|lowest|largest|smallest|biggest|oldest|youngest|"
        r"first|second|third|last|latest|earliest|more than|less than|finished|"
        r"winner|won|gold|silver|bronze|position|ranked)\b",
        re.IGNORECASE,
    )

    for q in questions:
        counts["questions"] += 1
        table_id = q["table_id"]
        table_path = args.tables / f"{table_id}.json"
        passage_path = args.passages / f"{table_id}.json"
        if not table_path.exists() or not passage_path.exists():
            counts["missing_artifact"] += 1
            continue
        table = json.loads(table_path.read_text())
        passages = json.loads(passage_path.read_text())
        answer = norm(q.get("answer-text", ""))
        question = norm(q.get("question", ""))
        cells = table_cells(table)

        answer_in_table = bool(answer and any(answer == norm(cell) for cell in cells))
        if answer_in_table:
            counts["answer_exactly_in_table"] += 1
        else:
            counts["answer_not_exactly_in_table"] += 1

        matching_links: dict[str, dict] = {}
        for node in q.get("answer-node", []):
            if len(node) < 4 or node[3] != "passage" or not node[2]:
                continue
            cell_text, position, link = node[0], node[1], node[2]
            passage = passages.get(link, "")
            if answer and answer in norm(passage):
                matching_links[link] = {
                    "F": str(cell_text),
                    "position": position,
                    "link": link,
                }

        if matching_links:
            counts["answer_in_linked_passage"] += 1
        if len(matching_links) == 1:
            counts["unique_answer_link"] += 1
            match = next(iter(matching_links.values()))
            f_norm = norm(match["F"])
            if f_norm and f_norm not in question:
                counts["unique_hidden_F"] += 1
                if selection_words.search(q.get("question", "")):
                    counts["unique_hidden_F_with_selection_language"] += 1
                    if not answer_in_table:
                        counts["strict_c1_c2_c3_candidate"] += 1
                        all_candidates.append(
                            {
                                "question_id": q["question_id"],
                                "question": q["question"],
                                "answer": q["answer-text"],
                                "table_id": table_id,
                                **match,
                            }
                        )
                    if not answer_in_table and len(examples) < 25:
                        examples.append(
                            {
                                "question_id": q["question_id"],
                                "question": q["question"],
                                "answer": q["answer-text"],
                                "table_id": table_id,
                                **match,
                            }
                        )
            else:
                counts["unique_F_already_in_question"] += 1
        elif len(matching_links) > 1:
            counts["ambiguous_answer_links"] += 1
        else:
            counts["no_traced_answer_passage"] += 1

    result = {
        "source": str(args.questions),
        "counts": dict(sorted(counts.items())),
        "candidate_examples": examples,
    }
    if args.output_candidates:
        args.output_candidates.parent.mkdir(parents=True, exist_ok=True)
        args.output_candidates.write_text(
            json.dumps(all_candidates, indent=2, ensure_ascii=False) + "\n"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
