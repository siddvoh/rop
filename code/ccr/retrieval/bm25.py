"""Resumable, deterministic SQLite FTS5 BM25 passage index."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

INDEX_VERSION = "hybridqa-sqlite-fts5-v1"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def query_terms(text: str) -> str:
    """Return a deduplicated OR query using the frozen ASCII token rule."""
    tokens = [token for token in TOKEN_PATTERN.findall(text.lower()) if len(token) > 1]
    return " OR ".join(dict.fromkeys(tokens))


def _connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS source_files (
            relative_path TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL,
            byte_count INTEGER NOT NULL,
            passage_count INTEGER NOT NULL,
            duplicate_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS links (
            link TEXT PRIMARY KEY,
            fts_rowid INTEGER NOT NULL UNIQUE,
            source_file TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS passage_fts USING fts5(
            link UNINDEXED,
            text,
            source_file UNINDEXED,
            tokenize='unicode61'
        );
        """
    )
    existing = connection.execute(
        "SELECT value FROM metadata WHERE key='index_version'"
    ).fetchone()
    if existing and existing["value"] != INDEX_VERSION:
        raise RuntimeError(
            f"index version mismatch: {existing['value']} != {INDEX_VERSION}"
        )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES('index_version',?)",
        (INDEX_VERSION,),
    )
    connection.commit()


def build_index(passages_dir: Path, db_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Build or resume an index, committing each source JSON atomically."""
    passages_dir = passages_dir.resolve()
    paths = sorted(passages_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no JSON passage files under {passages_dir}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect(db_path)
    _initialize(connection)

    source_inventory: list[dict[str, Any]] = []
    for path in paths:
        relative_path = path.relative_to(passages_dir).as_posix()
        raw = path.read_bytes()
        digest = sha256_bytes(raw)
        stored = connection.execute(
            "SELECT * FROM source_files WHERE relative_path=?", (relative_path,)
        ).fetchone()
        if stored:
            if stored["sha256"] != digest or stored["byte_count"] != len(raw):
                raise RuntimeError(f"source changed after indexing: {relative_path}")
            source_inventory.append(dict(stored))
            continue

        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise TypeError(f"expected an object mapping link to text: {relative_path}")
        inserted = 0
        duplicates = 0
        try:
            connection.execute("BEGIN IMMEDIATE")
            for link, text in payload.items():
                if not isinstance(link, str) or not isinstance(text, str):
                    raise TypeError(f"non-string passage in {relative_path}")
                if connection.execute(
                    "SELECT 1 FROM links WHERE link=?", (link,)
                ).fetchone():
                    duplicates += 1
                    continue
                cursor = connection.execute(
                    "INSERT INTO passage_fts(link,text,source_file) VALUES(?,?,?)",
                    (link, text, relative_path),
                )
                connection.execute(
                    "INSERT INTO links(link,fts_rowid,source_file) VALUES(?,?,?)",
                    (link, cursor.lastrowid, relative_path),
                )
                inserted += 1
            record = {
                "relative_path": relative_path,
                "sha256": digest,
                "byte_count": len(raw),
                "passage_count": inserted,
                "duplicate_count": duplicates,
            }
            connection.execute(
                """INSERT INTO source_files(
                    relative_path,sha256,byte_count,passage_count,duplicate_count
                ) VALUES(:relative_path,:sha256,:byte_count,:passage_count,:duplicate_count)""",
                record,
            )
            connection.commit()
            source_inventory.append(record)
        except Exception:
            connection.rollback()
            connection.close()
            raise

    indexed_paths = {
        row["relative_path"]
        for row in connection.execute("SELECT relative_path FROM source_files")
    }
    expected_paths = {path.relative_to(passages_dir).as_posix() for path in paths}
    unexpected = sorted(indexed_paths - expected_paths)
    if unexpected:
        connection.close()
        raise RuntimeError(f"indexed source files disappeared: {unexpected[:3]}")

    document_count = connection.execute("SELECT COUNT(*) FROM links").fetchone()[0]
    duplicate_count = sum(int(record["duplicate_count"]) for record in source_inventory)
    inventory_payload = [
        {
            "relative_path": record["relative_path"],
            "sha256": record["sha256"],
            "byte_count": record["byte_count"],
        }
        for record in source_inventory
    ]
    manifest = {
        "index_version": INDEX_VERSION,
        "status": "complete",
        "source_file_count": len(source_inventory),
        "document_count": document_count,
        "duplicate_link_count": duplicate_count,
        "source_inventory_sha256": sha256_bytes(canonical_bytes(inventory_payload)),
        "query": {
            "token_pattern": "[a-z0-9]+",
            "minimum_token_length": 2,
            "deduplicate_tokens": True,
            "operator": "OR",
        },
        "ranking": {
            "implementation": "SQLite FTS5 bm25",
            "document_tokenizer": "unicode61",
            "tie_break": "ascending FTS rowid",
        },
        "deduplication": "first link in sorted source-file order",
    }
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES('manifest',?)",
        (canonical_bytes(manifest).decode("utf-8"),),
    )
    connection.execute("INSERT INTO passage_fts(passage_fts) VALUES('optimize')")
    connection.commit()
    connection.close()
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    return manifest


def retrieve(connection: sqlite3.Connection, query: str, k: int = 20) -> list[dict[str, Any]]:
    """Return ranked links and raw FTS5 BM25 scores (lower is better)."""
    match = query_terms(query)
    if not match:
        return []
    rows = connection.execute(
        """SELECT rowid, link, bm25(passage_fts) AS score
        FROM passage_fts
        WHERE passage_fts MATCH ?
        ORDER BY score ASC, rowid ASC
        LIMIT ?""",
        (match, k),
    ).fetchall()
    return [
        {"rank": rank, "link": row["link"], "score": row["score"]}
        for rank, row in enumerate(rows, start=1)
    ]


def open_index(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection
