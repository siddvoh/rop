"""Validation and summaries for the released Qwen retrieval records."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Callable
import zipfile


AUTHORITY_MEMBER = "data/generated/ccr_reviewer_authority.zip"
LINKS_MEMBER = (
    "data/results/released/"
    "hybridqa_natural_v3_retriever_sensitivity_v1/links.jsonl"
)
QWEN_MEMBER = "qwen/records.jsonl"
BROAD_MEMBER = "broad_development/records.jsonl"
PROTOCOL_MEMBER = "PROTOCOL.json"
EXPECTED_MEMBERS = {
    "MANIFEST.json",
    "PROTOCOL.json",
    "README.txt",
    BROAD_MEMBER,
    QWEN_MEMBER,
}
RETRIEVERS = (
    "bm25",
    "bge_m3",
    "colbert_v2",
    "hybrid_rrf",
    "hybrid_rrf_bge_reranker_v2_m3",
)
CUTOFFS = (1, 5, 20)
EXPECTED_BROAD_HITS = {
    "bm25": (297, 1068),
    "bge_m3": (268, 1337),
    "colbert_v2": (330, 1386),
    "hybrid_rrf": (339, 1246),
    "hybrid_rrf_bge_reranker_v2_m3": (502, 1387),
}


class EvidenceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def jsonl(payload: bytes, member: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(payload.splitlines(), start=1):
        require(bool(raw.strip()), f"{member}:{line_number}: blank row")
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise EvidenceError(f"{member}:{line_number}: invalid JSON") from error
        require(isinstance(row, dict), f"{member}:{line_number}: row is not an object")
        rows.append(row)
    return rows


def rate(numerator: int, denominator: int) -> dict[str, int | float]:
    return {
        "denominator": denominator,
        "numerator": numerator,
        "rate": numerator / denominator,
    }


def _safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def load_archive_payload(payload: bytes) -> dict[str, Any]:
    try:
        from io import BytesIO

        bundle = zipfile.ZipFile(BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise EvidenceError("Qwen evidence ZIP is invalid") from error
    with bundle:
        infos = bundle.infolist()
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)), "Qwen evidence ZIP has duplicate names")
        require(set(names) == EXPECTED_MEMBERS, "Qwen evidence ZIP member set changed")
        for info in infos:
            require(_safe_name(info.filename), f"unsafe ZIP path: {info.filename}")
            require(not stat.S_ISLNK(info.external_attr >> 16), "ZIP contains a symlink")
            require(not (info.flag_bits & 1), "ZIP contains an encrypted member")

        try:
            manifest = json.loads(bundle.read("MANIFEST.json"))
            protocol = json.loads(bundle.read(PROTOCOL_MEMBER))
        except json.JSONDecodeError as error:
            raise EvidenceError("Qwen evidence metadata is invalid") from error
        require(manifest.get("schema_version") == 1, "manifest schema changed")
        require(manifest.get("status") == "complete", "manifest status changed")
        declared: dict[str, dict[str, Any]] = {}
        for row in manifest.get("members", []):
            require(isinstance(row, dict), "manifest entry is not an object")
            require(set(row) == {"bytes", "path", "sha256"}, "manifest fields changed")
            name = row["path"]
            require(name in EXPECTED_MEMBERS - {"MANIFEST.json"}, "manifest path changed")
            require(name not in declared, "manifest contains a duplicate path")
            member_payload = bundle.read(name)
            require(len(member_payload) == row["bytes"], f"member size changed: {name}")
            require(sha256_bytes(member_payload) == row["sha256"], f"member hash changed: {name}")
            declared[name] = row
        require(set(declared) == EXPECTED_MEMBERS - {"MANIFEST.json"}, "manifest is incomplete")

        qwen_payload = bundle.read(QWEN_MEMBER)
        broad_payload = bundle.read(BROAD_MEMBER)
        qwen = jsonl(qwen_payload, QWEN_MEMBER)
        broad = jsonl(broad_payload, BROAD_MEMBER)

    validate_qwen(qwen)
    validate_broad(broad)
    require(protocol.get("status") == "paper_scoped_evidence", "protocol status changed")
    require(protocol.get("qwen", {}).get("item_count") == 335, "protocol Qwen count changed")
    require(
        protocol.get("broad_development", {}).get("item_count") == 1_397,
        "protocol broad-development count changed",
    )
    return {
        "archive": {
            "bytes": len(payload),
            "member_count": len(EXPECTED_MEMBERS),
            "sha256": sha256_bytes(payload),
        },
        "broad": broad,
        "broad_payload": broad_payload,
        "manifest": manifest,
        "protocol": protocol,
        "qwen": qwen,
        "qwen_payload": qwen_payload,
    }


def load_archive(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Qwen evidence ZIP is missing: {path}")
    return load_archive_payload(path.read_bytes())


def load_from_members(reader: Callable[[str], bytes]) -> dict[str, Any]:
    try:
        payload = reader(AUTHORITY_MEMBER)
    except (KeyError, OSError) as error:
        raise EvidenceError(f"missing artifact member: {AUTHORITY_MEMBER}") from error
    return load_archive_payload(payload)


def validate_ranking(record: Any, context: str) -> None:
    require(isinstance(record, dict), f"{context}: ranking record is invalid")
    require(record.get("status") == "succeeded", f"{context}: ranking status changed")
    links = record.get("top20")
    require(isinstance(links, list) and len(links) <= 20, f"{context}: top20 changed")
    require(len(links) == len(set(links)), f"{context}: duplicate returned ID")
    require(all(isinstance(link, str) and link.startswith("/wiki/") for link in links), f"{context}: returned ID is invalid")
    for field in ("input_sha256", "ranking_sha256", "source_row_sha256"):
        if field in record:
            value = record[field]
            require(isinstance(value, str) and len(value) == 64, f"{context}: {field} changed")


def validate_qwen(rows: list[dict[str, Any]]) -> None:
    require(len(rows) == 335, "Qwen row count changed")
    require(len({row.get("question_id") for row in rows}) == 335, "Qwen IDs are not unique")
    for row in rows:
        question_id = row.get("question_id")
        require(row.get("schema_version") == 2, f"{question_id}: schema changed")
        require(row.get("selected_matches_trace") == (row.get("selected_id") == row.get("trace_id")), f"{question_id}: identity label changed")
        selection = row.get("selection", {})
        require(selection.get("status") == "succeeded", f"{question_id}: selection status changed")
        for field in (
            "input_sha256",
            "parsed_sha256",
            "raw_sha256",
            "request_sha256",
            "source_row_sha256",
        ):
            value = selection.get(field)
            require(
                isinstance(value, str) and len(value) == 64,
                f"{question_id}: selection {field} changed",
            )
        validate_ranking(row.get("final_hybrid"), f"{question_id}:final_hybrid")
        validate_ranking(row.get("body_bm25"), f"{question_id}:body_bm25")
        exact = row.get("exact_decoded_field", {})
        expected_status = "resolved" if row.get("selected_id_resolvable") else "unresolvable"
        require(exact.get("status") == expected_status, f"{question_id}: resolver status changed")
        require(exact.get("top1") == ([row["selected_id"]] if row.get("selected_id_resolvable") else []), f"{question_id}: exact-field result changed")

    require(sum(bool(row["selected_id_resolvable"]) for row in rows) == 334, "Qwen resolvable count changed")
    require(sum(not bool(row["selected_matches_trace"]) for row in rows) == 106, "Qwen object-distinct count changed")
    require(sum(not bool(row["selection"]["selected_id_in_candidate_table"]) for row in rows) == 1, "Qwen candidate-validity count changed")
    unresolved = [row["question_id"] for row in rows if not row["selected_id_resolvable"]]
    require(unresolved == ["e987e2347dc36977"], "Qwen unresolved ID changed")
    require(Counter(row["body_bm25"]["query_source"] for row in rows) == {"model_fact_same_match_query": 331, "trace_label_same_selected_id": 4}, "decoded-ID query coverage changed")
    require(sum(bool(row["body_bm25"]["raw_equals_generated_fact"]) for row in rows) == 326, "decoded-ID raw equality count changed")


def validate_broad(rows: list[dict[str, Any]]) -> None:
    require(len(rows) == 1_397, "broad-development row count changed")
    require(len({row.get("question_id") for row in rows}) == 1_397, "broad-development IDs are not unique")
    require(Counter(row.get("bucket") for row in rows) == {"unique_hidden_no_selection": 1_182, "unique_hidden_answer_in_table": 215}, "broad-development buckets changed")
    for row in rows:
        question_id = row.get("question_id")
        require(row.get("schema_version") == 2, f"{question_id}: broad schema changed")
        require(set(row.get("rankings", {})) == set(RETRIEVERS), f"{question_id}: retriever grid changed")
        for query in row.get("queries", {}).values():
            require(query.get("status") == "ready", f"{question_id}: query status changed")
            for field in ("input_sha256", "source_row_sha256"):
                value = query.get(field)
                require(
                    isinstance(value, str) and len(value) == 64,
                    f"{question_id}: query {field} changed",
                )
        for retriever in RETRIEVERS:
            cell = row["rankings"][retriever]
            require(set(cell) == {"question", "decoded_trace_id"}, f"{question_id}:{retriever}: query grid changed")
            validate_ranking(cell["question"], f"{question_id}:{retriever}:question")
            validate_ranking(cell["decoded_trace_id"], f"{question_id}:{retriever}:decoded_trace_id")
    hits = broad_summary(rows)["retrievers"]
    observed = {
        retriever: (
            values["question"]["hits_at_5"],
            values["decoded_trace_id"]["hits_at_5"],
        )
        for retriever, values in hits.items()
    }
    require(observed == EXPECTED_BROAD_HITS, "broad-development table counts changed")


def broad_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    retrievers: dict[str, Any] = {}
    for retriever in RETRIEVERS:
        arms = {}
        for arm in ("question", "decoded_trace_id"):
            hits = sum(row["trace_id"] in row["rankings"][retriever][arm]["top20"][:5] for row in rows)
            arms[arm] = {"hits_at_5": hits, "recall_at_5": hits / len(rows)}
        arms["difference"] = arms["decoded_trace_id"]["recall_at_5"] - arms["question"]["recall_at_5"]
        retrievers[retriever] = arms
    return {
        "bucket_counts": dict(sorted(Counter(row["bucket"] for row in rows).items())),
        "item_count": len(rows),
        "retrievers": retrievers,
    }


def _joint(rows: list[dict[str, Any]], field: str, cutoff: int) -> dict[str, int]:
    result = {"p0_h0": 0, "p0_h1": 0, "p1_h0": 0, "p1_h1": 0}
    for row in rows:
        links = row[field].get("top20", row[field].get("top1", []))[:cutoff]
        p = int(row["selected_id"] in links)
        h = int(row["trace_id"] in links)
        result[f"p{p}_h{h}"] += 1
    return result


def qwen_group(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    resolvable = [row for row in rows if row["selected_id_resolvable"]]
    by_cutoff: dict[str, Any] = {}
    for cutoff in CUTOFFS:
        p_hits = sum(
            row["selected_id"]
            in row[field].get("top20", row[field].get("top1", []))[:cutoff]
            for row in resolvable
        )
        h_all = sum(
            row["trace_id"]
            in row[field].get("top20", row[field].get("top1", []))[:cutoff]
            for row in rows
        )
        by_cutoff[str(cutoff)] = {
            "p_selected_id_preserved_on_resolvable_rows": rate(p_hits, len(resolvable)),
            "h_traced_id_return_on_all_rows": rate(h_all, len(rows)),
            "selected_vs_traced_joint_on_resolvable_rows": {
                "denominator": len(resolvable),
                **_joint(resolvable, field, cutoff),
            },
        }
    return {
        "all_rows": len(rows),
        "corpus_resolvable_selected_ids": len(resolvable),
        "by_cutoff": by_cutoff,
    }


def qwen_groups(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    partitions = {
        "all": rows,
        "c1_selected_equals_traced": [row for row in rows if row["selected_matches_trace"]],
        "c0_selected_differs_from_traced": [row for row in rows if not row["selected_matches_trace"]],
    }
    return {name: qwen_group(group, field) for name, group in partitions.items()}
