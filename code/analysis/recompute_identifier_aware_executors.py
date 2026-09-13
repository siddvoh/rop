#!/usr/bin/env python3
"""Replay deterministic identifier-aware lexical executors offline.

The frozen selector emits an exact ``/wiki/...`` identifier.  This audit asks
whether three simple, model-free executors preserve that selected object when
the identifier is converted to its decoded display string:

1. exact equality against the decoded-title inventory;
2. SQLite FTS5 BM25 over decoded titles only; and
3. SQLite FTS5 BM25 over decoded titles and frozen passage bodies with equal,
   untuned column weights.

The title-only executor is additionally evaluated as a corpus-wide round trip
over all 286,270 identifiers.  The title-plus-body executor is evaluated on
the frozen 1,800-row primary selector cohort.  All six invalid selectors and
both selected identifiers absent from the corpus remain explicit all-planned
misses in all-planned denominators.  No model, API, network, clock, random
number generator, or answer outcome is used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import string
import sys
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import unquote
import zipfile


SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifact/generated/ccr-reviewer-artifact.zip"
DEFAULT_OUTPUT = ROOT / "evidence/identifier_aware_executor_audit.json"

MODELS = ("claude", "openai", "gemini")
CUTOFFS = (1, 5, 20)
PLANNED_PER_MODEL = 600
QUERY_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

BRIDGE_MEMBER = (
    "data/results/released/hybridqa_natural_v2/"
    "bridge_confirmatory/details.jsonl"
)
LINKS_MEMBER = (
    "data/results/released/"
    "hybridqa_natural_v3_retriever_sensitivity_v1/links.jsonl"
)

EXPECTED_MEMBERS: dict[str, dict[str, Any]] = {
    BRIDGE_MEMBER: {
        "bytes": 4_991_925,
        "records": 1_800,
        "sha256": "abf0027a3f61aa338eae4a259b8751f7adecc5ea7f79a08f25ad11b179a79fc5",
    },
    LINKS_MEMBER: {
        "bytes": 15_250_846,
        "records": 286_270,
        "sha256": "6e9488f4a09fbd5be96ffff23f4e956195e0c231932137ba6894bfcdfe2fb303",
    },
}

EXPECTED_SOURCE_DB = {
    "bytes": 418_295_808,
    "sha256": "08a787a3d9cf4db4d44ff2fd22f2ef05d8e017bf9b6ed6214927f132ff4e4e4a",
    "document_count": 286_270,
    "source_file_count": 15_314,
    "manifest_bytes": 513,
    "manifest_sha256": "8d5b4d1aab29040a3b88cc8d63c7dc09165fb3546dab98ed7827440d3ccd6c48",
    "source_inventory_sha256": "f29607c884b1a4b85325b0cac4995f3dc8116f1ab7744f17db18fb0215c354dd",
}

EXPECTED_TITLE_MAPPING_SHA256 = (
    "5a093f364ac32aa244080eb9759299fd31efdc7a68ba07504f851f44c5bafeb5"
)

# Independent first-replay targets.  Each model/executor tuple is
# (resolvable, P@1, P@5, P@20).
EXPECTED_PRIMARY_PRESERVATION = {
    "claude": {
        "exact_decoded_equality": (599, 599, 599, 599),
        "title_only_bm25": (599, 548, 571, 583),
        "uniform_title_body_bm25": (599, 394, 479, 524),
    },
    "openai": {
        "exact_decoded_equality": (594, 594, 594, 594),
        "title_only_bm25": (594, 546, 569, 579),
        "uniform_title_body_bm25": (594, 393, 481, 521),
    },
    "gemini": {
        "exact_decoded_equality": (599, 599, 599, 599),
        "title_only_bm25": (599, 550, 576, 585),
        "uniform_title_body_bm25": (599, 398, 489, 529),
    },
}

# Corpus-wide title-only tuple: (queries, nonempty queries, P@1, P@5, P@20).
EXPECTED_CORPUS_TITLE_ONLY = (286_270, 286_129, 270_891, 278_603, 281_012)
EXPECTED_BODY_STREAM_SHA256 = (
    "963b5a269aadd8b201e772c388256437752a156c55ea4e393bae267d43d84646"
)
EXPECTED_PRIMARY_OBSERVATION_SHA256 = (
    "2ac43dd0528f703600a2026aab48a71729df788a3a9df88a6ba56401e3b89acb"
)
EXPECTED_PRIMARY_RANKING_SHA256 = {
    "exact_decoded_equality": (
        "cf0e41d71061a95787d07396a683cd9185f652492cf15ce67fb2331c17f44ec0"
    ),
    "title_only_bm25": (
        "4bc5ed1583af38a836d974229a0f0fff7a57c2943eb9a01816fd8e3b513786f3"
    ),
    "uniform_title_body_bm25": (
        "1ac63d1eee2b3bdd2da37c21beba59e7ab14895755efde72810a6bfad0fd1bd7"
    ),
}
EXPECTED_CORPUS_TITLE_RANKING_SHA256 = (
    "19691613c4f554f37282e4ff4d2cd6e17550104c3651f267f749d780f3c55e58"
)
EXPECTED_POOLED_CH5 = {
    "exact_decoded_equality": (1_463, 0, 0, 337),
    "title_only_bm25": (1_397, 66, 20, 317),
    "uniform_title_body_bm25": (1_179, 284, 40, 297),
}
EXPECTED_INVENTORY_QUERY_SUMMARY = {
    "distinct_ascii_or_query_count": 282_223,
    "empty_ascii_or_query_count": 141,
    "ascii_or_query_collision_group_count": 3_164,
    "identifiers_in_ascii_or_query_collision_groups": 7_211,
}


class AuditError(RuntimeError):
    """Raised when a frozen input or deterministic invariant changes."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
    )


def identifier_title(link: str) -> str:
    """Apply the frozen percent-decode/underscore/whitespace projection."""

    require(
        isinstance(link, str) and link.startswith("/wiki/") and len(link) > 6,
        f"invalid passage identifier: {link!r}",
    )
    decoded = unquote(link.removeprefix("/wiki/"))
    return " ".join(decoded.replace("_", " ").split())


def query_terms(text: str) -> str:
    """Return the frozen deduplicated ASCII-token OR query."""

    tokens = [
        token
        for token in QUERY_TOKEN_PATTERN.findall(text.lower())
        if len(token) > 1
    ]
    return " OR ".join(dict.fromkeys(tokens))


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    require(denominator > 0, "rate denominator must be positive")
    require(0 <= numerator <= denominator, "rate numerator outside denominator")
    return {
        "denominator": denominator,
        "numerator": numerator,
        "rate": numerator / denominator,
    }


def _jsonl_rows(payload: bytes, member: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuditError(f"{member}: invalid UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AuditError(f"{member}:{line_number}: invalid JSON") from exc
        require(
            isinstance(value, dict),
            f"{member}:{line_number}: row must be an object",
        )
        rows.append(value)
    return rows


def _validate_member(
    payload: bytes,
    member: str,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    require(len(payload) == expected["bytes"], f"{member}: byte count drift")
    digest = sha256_bytes(payload)
    require(digest == expected["sha256"], f"{member}: SHA-256 drift")
    records = sum(bool(line.strip()) for line in payload.splitlines())
    require(records == expected["records"], f"{member}: record count drift")
    return {"bytes": len(payload), "records": records, "sha256": digest}


def _load_inventory(
    payload: bytes,
) -> tuple[list[str], list[str], dict[str, int], dict[str, str], dict[str, Any]]:
    rows = _jsonl_rows(payload, LINKS_MEMBER)
    require(len(rows) == 286_270, "link inventory row count drift")
    links: list[str] = []
    titles: list[str] = []
    link_to_pid: dict[str, int] = {}
    title_to_link: dict[str, str] = {}
    title_mapping_digest = hashlib.sha256()
    query_signatures: Counter[str] = Counter()

    for pid, row in enumerate(rows):
        require(set(row) == {"link", "pid"}, f"links:{pid + 1}: schema drift")
        require(row.get("pid") == pid, f"links:{pid + 1}: PID sequence drift")
        link = row.get("link")
        require(
            isinstance(link, str) and link.startswith("/wiki/"),
            f"links:{pid + 1}: invalid identifier",
        )
        require(link not in link_to_pid, f"duplicate identifier: {link}")
        title = identifier_title(link)
        require(title, f"empty decoded title: {link}")
        require(
            title not in title_to_link,
            f"decoded-title collision: {title!r}",
        )
        links.append(link)
        titles.append(title)
        link_to_pid[link] = pid
        title_to_link[title] = link
        query_signatures[query_terms(title)] += 1
        title_mapping_digest.update(
            (canonical_json([pid, link, title]) + "\n").encode("utf-8")
        )

    mapping_sha256 = title_mapping_digest.hexdigest()
    require(
        mapping_sha256 == EXPECTED_TITLE_MAPPING_SHA256,
        "identifier-title mapping identity drift",
    )
    empty_query_count = query_signatures.get("", 0)
    signature_collision_groups = sum(
        1 for count in query_signatures.values() if count > 1
    )
    signature_collision_identifiers = sum(
        count for count in query_signatures.values() if count > 1
    )
    summary = {
        "identifier_count": len(links),
        "distinct_decoded_title_count": len(title_to_link),
        "decoded_title_collision_count": len(links) - len(title_to_link),
        "identifier_title_mapping_sha256": mapping_sha256,
        "distinct_ascii_or_query_count": len(query_signatures),
        "empty_ascii_or_query_count": empty_query_count,
        "ascii_or_query_collision_group_count": signature_collision_groups,
        "identifiers_in_ascii_or_query_collision_groups": (
            signature_collision_identifiers
        ),
    }
    require(
        {key: summary[key] for key in EXPECTED_INVENTORY_QUERY_SUMMARY}
        == EXPECTED_INVENTORY_QUERY_SUMMARY,
        "inventory lexical-query summary drift",
    )
    return links, titles, link_to_pid, title_to_link, summary


def _load_primary_rows(
    payload: bytes,
    link_to_pid: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _jsonl_rows(payload, BRIDGE_MEMBER)
    require(len(rows) == 1_800, "bridge row count drift")
    model_counts: Counter[str] = Counter()
    valid_counts: Counter[str] = Counter()
    resolvable_counts: Counter[str] = Counter()
    correct_counts: Counter[str] = Counter()
    identities: set[tuple[str, str]] = set()
    observation_digest = hashlib.sha256()

    for row_number, row in enumerate(rows, 1):
        model = row.get("model")
        question_id = row.get("question_id")
        require(model in MODELS, f"bridge:{row_number}: unexpected model")
        require(
            isinstance(question_id, str) and question_id,
            f"bridge:{row_number}: invalid question_id",
        )
        identity = (model, question_id)
        require(identity not in identities, f"bridge duplicate: {identity}")
        identities.add(identity)
        require(row.get("status") == "succeeded", f"bridge {identity}: status drift")
        require(type(row.get("valid_json")) is bool, f"bridge {identity}: validity drift")
        gold_link = row.get("gold_link")
        require(
            isinstance(gold_link, str) and gold_link in link_to_pid,
            f"bridge {identity}: unresolved gold link",
        )
        model_counts[model] += 1
        selected_link: str | None = None
        selection_correct = False
        if row["valid_json"]:
            selected_link = row.get("selected_link")
            require(
                isinstance(selected_link, str)
                and selected_link.startswith("/wiki/"),
                f"bridge {identity}: invalid selected link",
            )
            require(
                row.get("primary_predicted_query") == identifier_title(selected_link),
                f"bridge {identity}: decoded query drift",
            )
            require(
                type(row.get("selection_correct")) is bool,
                f"bridge {identity}: selection flag drift",
            )
            selection_correct = selected_link == gold_link
            require(
                row["selection_correct"] is selection_correct,
                f"bridge {identity}: selection flag disagrees with identity",
            )
            valid_counts[model] += 1
            if selected_link in link_to_pid:
                resolvable_counts[model] += 1
            if selection_correct:
                correct_counts[model] += 1
        else:
            require(model == "openai", f"bridge {identity}: unexpected invalid model")
            require(row.get("selected_link") is None, f"bridge {identity}: invalid selected link")
            require(row.get("selection_correct") is None, f"bridge {identity}: invalid selection flag")

        normalized = {
            "gold_link": gold_link,
            "model": model,
            "question_id": question_id,
            "selected_link": selected_link,
            "selection_correct": selection_correct,
            "valid_json": row["valid_json"],
        }
        observation_digest.update(
            (canonical_json(normalized) + "\n").encode("utf-8")
        )

    require(
        model_counts == Counter({model: 600 for model in MODELS}),
        f"bridge model counts drifted: {dict(model_counts)}",
    )
    observation_sha256 = observation_digest.hexdigest()
    require(
        observation_sha256 == EXPECTED_PRIMARY_OBSERVATION_SHA256,
        "primary observation identity drift",
    )
    return rows, {
        "all_planned_rows": len(rows),
        "planned_by_model": dict(sorted(model_counts.items())),
        "valid_selector_rows": sum(valid_counts.values()),
        "valid_by_model": dict(sorted(valid_counts.items())),
        "invalid_selector_rows": len(rows) - sum(valid_counts.values()),
        "resolvable_selected_ids": sum(resolvable_counts.values()),
        "resolvable_by_model": dict(sorted(resolvable_counts.items())),
        "unresolvable_valid_selected_ids": (
            sum(valid_counts.values()) - sum(resolvable_counts.values())
        ),
        "selection_correct_rows": sum(correct_counts.values()),
        "selection_correct_by_model": dict(sorted(correct_counts.items())),
        "primary_observation_identity_sha256": observation_sha256,
    }


def _validate_source_database(path: Path) -> tuple[sqlite3.Connection, dict[str, Any]]:
    require(path.is_file(), f"source BM25 database not found: {path}")
    size = path.stat().st_size
    require(size == EXPECTED_SOURCE_DB["bytes"], "source BM25 DB byte count drift")
    digest = sha256_file(path)
    require(digest == EXPECTED_SOURCE_DB["sha256"], "source BM25 DB SHA-256 drift")
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
    require(quick_check == "ok", f"source BM25 database quick_check: {quick_check}")
    document_count = connection.execute("SELECT COUNT(*) FROM links").fetchone()[0]
    fts_count = connection.execute("SELECT COUNT(*) FROM passage_fts").fetchone()[0]
    source_file_count = connection.execute(
        "SELECT COUNT(*) FROM source_files"
    ).fetchone()[0]
    require(
        document_count == EXPECTED_SOURCE_DB["document_count"] == fts_count,
        "source BM25 document count drift",
    )
    require(
        source_file_count == EXPECTED_SOURCE_DB["source_file_count"],
        "source BM25 source-file count drift",
    )
    row = connection.execute(
        "SELECT value FROM metadata WHERE key='manifest'"
    ).fetchone()
    require(row is not None, "source BM25 manifest is missing")
    manifest_text = row[0]
    manifest_bytes = manifest_text.encode("utf-8")
    require(
        len(manifest_bytes) == EXPECTED_SOURCE_DB["manifest_bytes"],
        "source BM25 manifest byte count drift",
    )
    manifest_sha256 = sha256_bytes(manifest_bytes)
    require(
        manifest_sha256 == EXPECTED_SOURCE_DB["manifest_sha256"],
        "source BM25 manifest SHA-256 drift",
    )
    manifest = json.loads(manifest_text)
    require(
        manifest.get("source_inventory_sha256")
        == EXPECTED_SOURCE_DB["source_inventory_sha256"],
        "source inventory identity drift",
    )
    return connection, {
        "bytes": size,
        "sha256": digest,
        "quick_check": quick_check,
        "document_count": document_count,
        "source_file_count": source_file_count,
        "manifest_bytes": len(manifest_bytes),
        "manifest_sha256": manifest_sha256,
        "source_inventory_sha256": manifest["source_inventory_sha256"],
    }


def _create_executor_database(
    source: sqlite3.Connection,
    scratch_path: Path,
    links: Sequence[str],
    titles: Sequence[str],
) -> tuple[sqlite3.Connection, str]:
    require(len(links) == len(titles) == 286_270, "executor input count drift")
    if scratch_path.exists():
        scratch_path.unlink()
    scratch_path.parent.mkdir(parents=True, exist_ok=True)
    target = sqlite3.connect(scratch_path)
    target.execute("PRAGMA journal_mode=OFF")
    target.execute("PRAGMA synchronous=OFF")
    target.execute("PRAGMA temp_store=MEMORY")
    target.execute("PRAGMA locking_mode=EXCLUSIVE")
    try:
        target.executescript(
            """
            CREATE VIRTUAL TABLE title_fts USING fts5(
                link UNINDEXED,
                title,
                tokenize='unicode61'
            );
            CREATE VIRTUAL TABLE title_body_fts USING fts5(
                link UNINDEXED,
                title,
                body,
                tokenize='unicode61'
            );
            """
        )
        source_rows = source.execute(
            """SELECT p.rowid AS rowid, p.link AS link, p.text AS body,
                      l.fts_rowid AS mapped_rowid
               FROM passage_fts AS p
               JOIN links AS l ON l.link=p.link
               ORDER BY p.rowid ASC"""
        )
        body_stream_digest = hashlib.sha256()
        batch_title: list[tuple[int, str, str]] = []
        batch_combined: list[tuple[int, str, str, str]] = []
        seen = 0
        for row in source_rows:
            pid = seen
            rowid = int(row["rowid"])
            link = row["link"]
            body = row["body"]
            require(rowid == pid + 1, f"source BM25 rowid drift at PID {pid}")
            require(int(row["mapped_rowid"]) == rowid, f"source link rowid drift at PID {pid}")
            require(link == links[pid], f"source/inventory link mismatch at PID {pid}")
            require(isinstance(body, str), f"source body is not text at PID {pid}")
            title = titles[pid]
            body_stream_digest.update(
                (canonical_json([pid, link, body]) + "\n").encode("utf-8")
            )
            batch_title.append((rowid, link, title))
            batch_combined.append((rowid, link, title, body))
            seen += 1
            if len(batch_title) == 2_000:
                target.executemany(
                    "INSERT INTO title_fts(rowid,link,title) VALUES(?,?,?)",
                    batch_title,
                )
                target.executemany(
                    "INSERT INTO title_body_fts(rowid,link,title,body) VALUES(?,?,?,?)",
                    batch_combined,
                )
                batch_title.clear()
                batch_combined.clear()
                if seen % 20_000 == 0:
                    print(
                        f"indexed identifier-aware executors: {seen}/286270",
                        file=sys.stderr,
                        flush=True,
                    )
        if batch_title:
            target.executemany(
                "INSERT INTO title_fts(rowid,link,title) VALUES(?,?,?)",
                batch_title,
            )
            target.executemany(
                "INSERT INTO title_body_fts(rowid,link,title,body) VALUES(?,?,?,?)",
                batch_combined,
            )
        require(seen == 286_270, "source BM25 row count drift during indexing")
        target.execute("INSERT INTO title_fts(title_fts) VALUES('optimize')")
        target.execute("INSERT INTO title_body_fts(title_body_fts) VALUES('optimize')")
        target.commit()
        require(
            target.execute("SELECT COUNT(*) FROM title_fts").fetchone()[0]
            == seen,
            "title-only executor row count drift",
        )
        require(
            target.execute("SELECT COUNT(*) FROM title_body_fts").fetchone()[0]
            == seen,
            "title-plus-body executor row count drift",
        )
        body_stream_sha256 = body_stream_digest.hexdigest()
        require(
            body_stream_sha256 == EXPECTED_BODY_STREAM_SHA256,
            "source passage body stream identity drift",
        )
        return target, body_stream_sha256
    except Exception:
        target.close()
        raise


def _fts_ranking(
    connection: sqlite3.Connection,
    table: str,
    decoded_title: str,
    k: int = 20,
) -> list[str]:
    require(table in {"title_fts", "title_body_fts"}, "unrecognized FTS table")
    match = query_terms(decoded_title)
    if not match:
        return []
    if table == "title_fts":
        sql = (
            "SELECT link FROM title_fts WHERE title_fts MATCH ? "
            "ORDER BY bm25(title_fts,1.0) ASC,rowid ASC LIMIT ?"
        )
    else:
        sql = (
            "SELECT link FROM title_body_fts WHERE title_body_fts MATCH ? "
            "ORDER BY bm25(title_body_fts,1.0,1.0) ASC,rowid ASC LIMIT ?"
        )
    return [row[0] for row in connection.execute(sql, (match, k)).fetchall()]


def _summarize_primary_executor(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    link_to_pid: Mapping[str, int],
    ranking_for_selected_link: Any,
) -> dict[str, Any]:
    preservation: dict[str, Counter[int]] = {
        model: Counter() for model in MODELS
    }
    resolvable: Counter[str] = Counter()
    planned: Counter[str] = Counter()
    valid: Counter[str] = Counter()
    empty_query: Counter[str] = Counter()
    ch: dict[str, dict[int, Counter[str]]] = {
        model: {cutoff: Counter() for cutoff in CUTOFFS} for model in MODELS
    }
    ranking_digest = hashlib.sha256()
    cache: dict[str, list[str]] = {}

    for row in rows:
        model = row["model"]
        question_id = row["question_id"]
        gold_link = row["gold_link"]
        planned[model] += 1
        selected_link = row.get("selected_link") if row["valid_json"] else None
        ranking: list[str] = []
        if selected_link is not None:
            valid[model] += 1
            if not query_terms(identifier_title(selected_link)) and name != "exact_decoded_equality":
                empty_query[model] += 1
            if selected_link not in cache:
                cache[selected_link] = ranking_for_selected_link(selected_link)
            ranking = cache[selected_link]
            require(len(ranking) <= 20, f"{name}: ranking exceeds 20")
            require(len(ranking) == len(set(ranking)), f"{name}: duplicate ranking link")
            if selected_link in link_to_pid:
                resolvable[model] += 1
                for cutoff in CUTOFFS:
                    if selected_link in ranking[:cutoff]:
                        preservation[model][cutoff] += 1
        c = bool(row["valid_json"] and row["selection_correct"])
        for cutoff in CUTOFFS:
            h = gold_link in ranking[:cutoff]
            ch[model][cutoff][f"c{int(c)}_h{int(h)}"] += 1
        ranking_digest.update(
            (
                canonical_json(
                    [model, question_id, selected_link, gold_link, ranking]
                )
                + "\n"
            ).encode("utf-8")
        )

    per_model: dict[str, Any] = {}
    for model in MODELS:
        require(planned[model] == PLANNED_PER_MODEL, f"{name}/{model}: planned drift")
        model_record = {
            "planned_rows": planned[model],
            "valid_selector_rows": valid[model],
            "resolvable_selected_ids": resolvable[model],
            "empty_lexical_queries": empty_query[model],
            "selected_id_preservation_on_resolvable": {
                str(cutoff): _rate(preservation[model][cutoff], resolvable[model])
                for cutoff in CUTOFFS
            },
            "selected_id_preservation_all_planned": {
                str(cutoff): _rate(preservation[model][cutoff], planned[model])
                for cutoff in CUTOFFS
            },
            "selection_delivery_joint_all_planned": {
                str(cutoff): {
                    "c1_h1": ch[model][cutoff]["c1_h1"],
                    "c1_h0": ch[model][cutoff]["c1_h0"],
                    "c0_h1": ch[model][cutoff]["c0_h1"],
                    "c0_h0": ch[model][cutoff]["c0_h0"],
                }
                for cutoff in CUTOFFS
            },
        }
        for cutoff in CUTOFFS:
            cells = model_record["selection_delivery_joint_all_planned"][str(cutoff)]
            require(sum(cells.values()) == planned[model], f"{name}/{model}: C/H sum drift")
        per_model[model] = model_record

    pooled_resolvable = sum(resolvable.values())
    pooled_planned = sum(planned.values())
    pooled = {
        "planned_rows": pooled_planned,
        "valid_selector_rows": sum(valid.values()),
        "resolvable_selected_ids": pooled_resolvable,
        "empty_lexical_queries": sum(empty_query.values()),
        "selected_id_preservation_on_resolvable": {
            str(cutoff): _rate(
                sum(preservation[model][cutoff] for model in MODELS),
                pooled_resolvable,
            )
            for cutoff in CUTOFFS
        },
        "selected_id_preservation_all_planned": {
            str(cutoff): _rate(
                sum(preservation[model][cutoff] for model in MODELS),
                pooled_planned,
            )
            for cutoff in CUTOFFS
        },
        "selection_delivery_joint_all_planned": {
            str(cutoff): {
                cell: sum(ch[model][cutoff][cell] for model in MODELS)
                for cell in ("c1_h1", "c1_h0", "c0_h1", "c0_h0")
            }
            for cutoff in CUTOFFS
        },
    }
    return {
        "per_model": per_model,
        "pooled_descriptive": pooled,
        "ranking_identity_sha256": ranking_digest.hexdigest(),
        "unique_valid_selected_id_queries_executed": len(cache),
    }


def _corpus_title_roundtrip(
    connection: sqlite3.Connection,
    links: Sequence[str],
    titles: Sequence[str],
) -> dict[str, Any]:
    preserved = Counter()
    nonempty = 0
    ranking_digest = hashlib.sha256()
    for pid, (link, title) in enumerate(zip(links, titles, strict=True)):
        ranking = _fts_ranking(connection, "title_fts", title)
        if query_terms(title):
            nonempty += 1
        for cutoff in CUTOFFS:
            if link in ranking[:cutoff]:
                preserved[cutoff] += 1
        ranking_digest.update(
            (canonical_json([pid, link, ranking]) + "\n").encode("utf-8")
        )
        if (pid + 1) % 20_000 == 0:
            print(
                f"title-only corpus round trip: {pid + 1}/286270",
                file=sys.stderr,
                flush=True,
            )
    result = {
        "queries": len(links),
        "nonempty_lexical_queries": nonempty,
        "empty_lexical_queries": len(links) - nonempty,
        "preservation": {
            str(cutoff): _rate(preserved[cutoff], len(links))
            for cutoff in CUTOFFS
        },
        "ranking_identity_sha256": ranking_digest.hexdigest(),
    }
    observed = (
        len(links),
        nonempty,
        preserved[1],
        preserved[5],
        preserved[20],
    )
    require(
        observed == EXPECTED_CORPUS_TITLE_ONLY,
        f"corpus title-only targets drifted: {observed}",
    )
    require(
        result["ranking_identity_sha256"]
        == EXPECTED_CORPUS_TITLE_RANKING_SHA256,
        "corpus title-only ranking identity drift",
    )
    return result


def _enforce_primary_targets(executors: Mapping[str, Any]) -> None:
    for model in MODELS:
        for executor, expected in EXPECTED_PRIMARY_PRESERVATION[model].items():
            record = executors[executor]["per_model"][model]
            observed = (
                record["resolvable_selected_ids"],
                record["selected_id_preservation_on_resolvable"]["1"]["numerator"],
                record["selected_id_preservation_on_resolvable"]["5"]["numerator"],
                record["selected_id_preservation_on_resolvable"]["20"]["numerator"],
            )
            require(
                observed == expected,
                f"{model}/{executor} preservation targets drifted: {observed}",
            )
    for executor, expected in EXPECTED_POOLED_CH5.items():
        cells = executors[executor]["pooled_descriptive"][
            "selection_delivery_joint_all_planned"
        ]["5"]
        observed = tuple(
            cells[key] for key in ("c1_h1", "c1_h0", "c0_h1", "c0_h0")
        )
        require(
            observed == expected,
            f"{executor} pooled C/H@5 targets drifted: {observed}",
        )
        require(
            executors[executor]["ranking_identity_sha256"]
            == EXPECTED_PRIMARY_RANKING_SHA256[executor],
            f"{executor} primary ranking identity drift",
        )


def build_report(
    artifact_path: Path,
    source_db_path: Path,
    scratch_path: Path,
    script_path: Path,
) -> dict[str, Any]:
    require(artifact_path.is_file(), f"reviewer artifact not found: {artifact_path}")
    with zipfile.ZipFile(artifact_path) as bundle:
        payloads = {member: bundle.read(member) for member in EXPECTED_MEMBERS}
    member_metadata = {
        member: _validate_member(payloads[member], member, expected)
        for member, expected in EXPECTED_MEMBERS.items()
    }
    links, titles, link_to_pid, title_to_link, inventory_summary = _load_inventory(
        payloads[LINKS_MEMBER]
    )
    primary_rows, cohort_summary = _load_primary_rows(
        payloads[BRIDGE_MEMBER], link_to_pid
    )
    source, source_metadata = _validate_source_database(source_db_path)
    executor_db: sqlite3.Connection | None = None
    try:
        executor_db, body_stream_sha256 = _create_executor_database(
            source, scratch_path, links, titles
        )

        def exact_ranking(selected_link: str) -> list[str]:
            title = identifier_title(selected_link)
            match = title_to_link.get(title)
            return [match] if match is not None else []

        executors = {
            "exact_decoded_equality": _summarize_primary_executor(
                "exact_decoded_equality",
                primary_rows,
                link_to_pid,
                exact_ranking,
            ),
            "title_only_bm25": _summarize_primary_executor(
                "title_only_bm25",
                primary_rows,
                link_to_pid,
                lambda link: _fts_ranking(
                    executor_db, "title_fts", identifier_title(link)
                ),
            ),
            "uniform_title_body_bm25": _summarize_primary_executor(
                "uniform_title_body_bm25",
                primary_rows,
                link_to_pid,
                lambda link: _fts_ranking(
                    executor_db, "title_body_fts", identifier_title(link)
                ),
            ),
        }
        _enforce_primary_targets(executors)
        corpus_title_only = _corpus_title_roundtrip(executor_db, links, titles)
    finally:
        if executor_db is not None:
            executor_db.close()
        source.close()

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_zero_call_identifier_aware_executor_audit",
        "scope": {
            "claim": (
                "Preservation of the selector-emitted identifier under three "
                "deterministic decoded-title executors"
            ),
            "not_claimed": [
                "a model's ability to infer the selected identifier",
                "new answer-generation outcomes",
                "causality of retrieval on answer correctness",
                "deployment or cross-dataset generalization",
            ],
            "primary_cohort_note": (
                "The same 600 questions occur for each hosted-model family; "
                "pooled rows are a descriptive inventory, not independent samples."
            ),
        },
        "inputs": {
            "artifact_members": member_metadata,
            "source_bm25_database": {
                **source_metadata,
                "body_stream_sha256": body_stream_sha256,
            },
        },
        "inventory": inventory_summary,
        "primary_cohort": cohort_summary,
        "executor_specifications": {
            "decoded_title_projection": {
                "steps": [
                    "remove the literal /wiki/ prefix",
                    "UTF-8 percent-decode",
                    "replace underscores with spaces",
                    "collapse Unicode whitespace",
                ],
                "fragment_or_anchor_stripping": False,
            },
            "exact_decoded_equality": {
                "index": "one-to-one decoded-title to exact identifier map",
                "query": "exact case-sensitive Unicode string equality",
                "ranking": "the unique equality match, or an empty result",
                "ties": "impossible in the verified frozen inventory",
            },
            "title_only_bm25": {
                "index": "ephemeral SQLite FTS5 over decoded title only",
                "document_tokenizer": "unicode61 with SQLite defaults",
                "query_token_pattern": "[a-z0-9]+ on lowercase decoded title",
                "minimum_query_token_length": 2,
                "query_token_deduplication": "first occurrence retained",
                "query_operator": "OR",
                "bm25_column_weights": [1.0],
                "tie_break": "ascending frozen PID (FTS rowid)",
                "cutoff": 20,
            },
            "uniform_title_body_bm25": {
                "index": (
                    "ephemeral SQLite FTS5 over decoded title and exact frozen "
                    "passage body"
                ),
                "document_tokenizer": "unicode61 with SQLite defaults",
                "query_token_pattern": "[a-z0-9]+ on lowercase decoded title",
                "minimum_query_token_length": 2,
                "query_token_deduplication": "first occurrence retained",
                "query_operator": "OR",
                "bm25_column_weights": {"title": 1.0, "body": 1.0},
                "weight_selection": "fixed uniform weights; no outcome tuning",
                "tie_break": "ascending frozen PID (FTS rowid)",
                "cutoff": 20,
            },
        },
        "primary_selected_id_preservation": executors,
        "corpus_wide_round_trip": {
            "exact_decoded_equality": {
                "queries": len(links),
                "preservation": {
                    str(cutoff): _rate(len(links), len(links))
                    for cutoff in CUTOFFS
                },
            },
            "title_only_bm25": corpus_title_only,
            "uniform_title_body_bm25": {
                "status": "not_run_corpus_wide",
                "reason": (
                    "The combined executor is evaluated on every frozen primary "
                    "selection; a 286,270-query corpus sweep is not needed for "
                    "the primary-cohort preservation claim."
                ),
            },
        },
        "api_calls": 0,
        "model_calls": 0,
        "network_used": False,
        "answer_calls": 0,
        "replay": {
            "script_sha256": sha256_file(script_path),
            "scratch_database_is_scientific_input": False,
            "scratch_database_retained": False,
            "outcome_targets_enforced": True,
            "corpus_title_only_target_enforced": True,
        },
    }
    require(_is_sha256(report["replay"]["script_sha256"]), "invalid script digest")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument(
        "--source-db",
        type=Path,
        required=True,
        help="Path to the hash-locked 418,295,808-byte frozen BM25 database",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="Directory for a disposable FTS5 executor index (default: system temp)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()
    if args.scratch_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="ccr_identifier_executor_")
        scratch_root = Path(temporary.name)
    else:
        temporary = None
        scratch_root = args.scratch_dir.resolve()
        scratch_root.mkdir(parents=True, exist_ok=True)
    scratch_path = scratch_root / "identifier_aware_executors.sqlite"
    try:
        report = build_report(
            args.artifact.resolve(),
            args.source_db.resolve(),
            scratch_path,
            script_path,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_json_bytes(report))
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        if temporary is not None:
            temporary.cleanup()
        else:
            for suffix in ("", "-journal", "-shm", "-wal"):
                candidate = Path(str(scratch_path) + suffix)
                if candidate.exists():
                    candidate.unlink()
            try:
                scratch_root.rmdir()
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
