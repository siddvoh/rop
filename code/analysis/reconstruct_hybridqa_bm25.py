#!/usr/bin/env python3
"""Reconstruct the frozen HybridQA SQLite FTS5 database from public sources.

The scientific input is the ``request_tok`` directory at one pinned
WikiTables-WithLinks commit.  The script accepts an already available source
directory, a local GitHub source archive, or an explicit ``--download`` mode.
The archive reader does not extract case-variant filenames, so reconstruction
also works on case-insensitive filesystems.  No model or API call is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from typing import Any, Iterable, Iterator
import urllib.request


SCHEMA_VERSION = 2
INDEX_VERSION = "hybridqa-sqlite-fts5-v1"
COMMIT = "dc066e1a6d5281511d8b73a6107d5ad2824cc2b2"
ARCHIVE_URL = (
    "https://github.com/wenhuchen/WikiTables-WithLinks/archive/"
    f"{COMMIT}.tar.gz"
)
RAW_SOURCE_FILE_COUNT = 15_316
EXCLUDED_FILES = (
    "2014_Malaysia_Premier_League_0.json",
    "2014_Malaysia_Premier_League_1.json",
)
CURATED_SOURCE_FILE_COUNT = 15_314
SOURCE_INVENTORY_SHA256 = (
    "f29607c884b1a4b85325b0cac4995f3dc8116f1ab7744f17db18fb0215c354dd"
)
DOCUMENT_COUNT = 286_270
DUPLICATE_LINK_COUNT = 226_400
DATABASE_BYTES = 418_295_808
DATABASE_SHA256 = (
    "08a787a3d9cf4db4d44ff2fd22f2ef05d8e017bf9b6ed6214927f132ff4e4e4a"
)
MANIFEST_BYTES = 599
MANIFEST_SHA256 = (
    "86a8f7032988d3c7a0b7776db249b17f51dd8bedd717e76e8325e339c4277283"
)
INTERNAL_MANIFEST_BYTES = 513
INTERNAL_MANIFEST_SHA256 = (
    "8d5b4d1aab29040a3b88cc8d63c7dc09165fb3546dab98ed7827440d3ccd6c48"
)

# Static record of the environment in which the frozen byte identities below
# were observed.  This is deliberately not a portability claim: another
# runtime is accepted only when it produces every frozen byte count and hash.
KNOWN_GOOD_REBUILD_RUNTIME = {
    "python": {"implementation": "CPython", "version": "3.14.5"},
    "sqlite": {
        "version": "3.53.1",
        "database_page_size_bytes": 4_096,
        "compile_options": {
            "count": 53,
            "canonicalization": (
                "lexicographically sorted option strings encoded as UTF-8 "
                "canonical JSON"
            ),
            "canonical_sha256": (
                "99d80fee03818112b412ae76c1b334602ab6b6de6899610155a90a043ed5bbbc"
            ),
        },
    },
    "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
}
REBUILD_RUNTIME_ACCEPTANCE = {
    "frozen_output_exact_bytes_observed_in_known_good_runtime": True,
    "cross_environment_portability_claimed": False,
    "other_runtime_pass_condition": (
        "all frozen output byte counts and SHA-256 identities match exactly"
    ),
}

DATABASE_NAME = "hybridqa_bm25.sqlite"
MANIFEST_NAME = "hybridqa_bm25.manifest.json"
REPORT_NAME = "hybridqa_bm25_reconstruction.json"


class ReconstructionError(RuntimeError):
    """Raised when a public source or deterministic output drifts."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReconstructionError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _curated_names(
    names: Iterable[str],
    *,
    raw_count: int = RAW_SOURCE_FILE_COUNT,
    curated_count: int = CURATED_SOURCE_FILE_COUNT,
    excluded: tuple[str, ...] = EXCLUDED_FILES,
) -> list[str]:
    values = sorted(names)
    require(len(values) == len(set(values)), "duplicate source basenames")
    exclusions = set(excluded)
    if len(values) == raw_count and exclusions <= set(values):
        values = [name for name in values if name not in exclusions]
    elif len(values) == curated_count and exclusions.isdisjoint(values):
        pass
    else:
        raise ReconstructionError(
            "source set is neither the pinned raw inventory nor its exact "
            "15,314-file curated view"
        )
    require(len(values) == curated_count, "curated source count drift")
    return values


def _directory_records(source: Path) -> Iterator[tuple[str, bytes]]:
    passages = source / "request_tok" if (source / "request_tok").is_dir() else source
    require(passages.is_dir(), f"request_tok source directory not found: {source}")
    paths = {path.name: path for path in passages.glob("*.json")}
    names = _curated_names(paths)
    for name in names:
        yield name, paths[name].read_bytes()


def _archive_records(archive_path: Path) -> Iterator[tuple[str, bytes]]:
    require(archive_path.is_file(), f"source archive not found: {archive_path}")
    try:
        with tarfile.open(archive_path, "r:*") as archive:
            members: dict[str, tarfile.TarInfo] = {}
            for member in archive.getmembers():
                parts = Path(member.name).parts
                if (
                    member.isfile()
                    and len(parts) >= 3
                    and parts[-2] == "request_tok"
                    and parts[-1].endswith(".json")
                ):
                    name = parts[-1]
                    require(name not in members, f"duplicate archive member: {name}")
                    members[name] = member
            names = _curated_names(members)
            for name in names:
                handle = archive.extractfile(members[name])
                require(handle is not None, f"cannot read archive member: {name}")
                yield name, handle.read()
    except (tarfile.TarError, OSError) as exc:
        raise ReconstructionError(f"invalid source archive: {archive_path}") from exc


def _initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=FULL;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE source_files (
            relative_path TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL,
            byte_count INTEGER NOT NULL,
            passage_count INTEGER NOT NULL,
            duplicate_count INTEGER NOT NULL
        );
        CREATE TABLE links (
            link TEXT PRIMARY KEY,
            fts_rowid INTEGER NOT NULL UNIQUE,
            source_file TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE passage_fts USING fts5(
            link UNINDEXED,
            text,
            source_file UNINDEXED,
            tokenize='unicode61'
        );
        """
    )
    connection.execute(
        "INSERT INTO metadata(key,value) VALUES('index_version',?)",
        (INDEX_VERSION,),
    )
    connection.commit()


def _remove_outputs(output_dir: Path) -> None:
    for name in (DATABASE_NAME, MANIFEST_NAME, REPORT_NAME):
        path = output_dir / name
        for candidate in (path, Path(str(path) + "-journal"), Path(str(path) + "-shm"), Path(str(path) + "-wal")):
            candidate.unlink(missing_ok=True)


def build_database(
    records: Iterable[tuple[str, bytes]],
    output_dir: Path,
    *,
    enforce_frozen: bool = True,
) -> dict[str, Any]:
    """Build the index from ordered source bytes and return observed identities."""

    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / DATABASE_NAME
    manifest_path = output_dir / MANIFEST_NAME
    require(not database_path.exists(), f"output already exists: {database_path}")
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    source_inventory: list[dict[str, Any]] = []
    try:
        _initialize(connection)
        previous_name: str | None = None
        for index, (relative_path, raw) in enumerate(records, start=1):
            require(relative_path.endswith(".json"), "non-JSON source record")
            require(
                previous_name is None or previous_name < relative_path,
                "source records are not in strict lexical order",
            )
            previous_name = relative_path
            payload = json.loads(raw)
            require(isinstance(payload, dict), f"source is not an object: {relative_path}")
            inserted = 0
            duplicates = 0
            connection.execute("BEGIN IMMEDIATE")
            for link, text in payload.items():
                require(
                    isinstance(link, str) and isinstance(text, str),
                    f"non-string passage in {relative_path}",
                )
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
                "sha256": sha256_bytes(raw),
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
            if index % 2_000 == 0:
                print(f"indexed source files: {index}/{CURATED_SOURCE_FILE_COUNT}", file=sys.stderr)

        document_count = int(
            connection.execute("SELECT COUNT(*) FROM links").fetchone()[0]
        )
        duplicate_count = sum(int(row["duplicate_count"]) for row in source_inventory)
        inventory_payload = [
            {
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
                "byte_count": row["byte_count"],
            }
            for row in source_inventory
        ]
        inventory_sha256 = sha256_bytes(canonical_bytes(inventory_payload))
        manifest = {
            "index_version": INDEX_VERSION,
            "status": "complete",
            "source_file_count": len(source_inventory),
            "document_count": document_count,
            "duplicate_link_count": duplicate_count,
            "source_inventory_sha256": inventory_sha256,
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
        internal_manifest = canonical_bytes(manifest)
        connection.execute(
            "INSERT INTO metadata(key,value) VALUES('manifest',?)",
            (internal_manifest.decode("utf-8"),),
        )
        connection.execute("INSERT INTO passage_fts(passage_fts) VALUES('optimize')")
        connection.commit()
    except Exception:
        connection.rollback()
        connection.close()
        _remove_outputs(output_dir)
        raise
    connection.close()

    manifest_payload = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    manifest_path.write_bytes(manifest_payload)
    observed = {
        "source_file_count": len(source_inventory),
        "source_inventory_sha256": inventory_sha256,
        "document_count": document_count,
        "duplicate_link_count": duplicate_count,
        "database_bytes": database_path.stat().st_size,
        "database_sha256": sha256_file(database_path),
        "manifest_bytes": len(manifest_payload),
        "manifest_sha256": sha256_bytes(manifest_payload),
        "internal_manifest_bytes": len(internal_manifest),
        "internal_manifest_sha256": sha256_bytes(internal_manifest),
    }
    if enforce_frozen:
        expected = {
            "source_file_count": CURATED_SOURCE_FILE_COUNT,
            "source_inventory_sha256": SOURCE_INVENTORY_SHA256,
            "document_count": DOCUMENT_COUNT,
            "duplicate_link_count": DUPLICATE_LINK_COUNT,
            "database_bytes": DATABASE_BYTES,
            "database_sha256": DATABASE_SHA256,
            "manifest_bytes": MANIFEST_BYTES,
            "manifest_sha256": MANIFEST_SHA256,
            "internal_manifest_bytes": INTERNAL_MANIFEST_BYTES,
            "internal_manifest_sha256": INTERNAL_MANIFEST_SHA256,
        }
        require(observed == expected, f"reconstruction drifted: {observed}")
    return observed


def reconstruction_report(observed: dict[str, Any], script_path: Path) -> dict[str, Any]:
    expected = {
        "source_file_count": CURATED_SOURCE_FILE_COUNT,
        "source_inventory_sha256": SOURCE_INVENTORY_SHA256,
        "document_count": DOCUMENT_COUNT,
        "duplicate_link_count": DUPLICATE_LINK_COUNT,
        "database_bytes": DATABASE_BYTES,
        "database_sha256": DATABASE_SHA256,
        "manifest_bytes": MANIFEST_BYTES,
        "manifest_sha256": MANIFEST_SHA256,
        "internal_manifest_bytes": INTERNAL_MANIFEST_BYTES,
        "internal_manifest_sha256": INTERNAL_MANIFEST_SHA256,
    }
    require(observed == expected, "observed reconstruction does not match the frozen contract")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_exact_public_source_reconstruction",
        "source_contract": {
            "repository": "wenhuchen/WikiTables-WithLinks",
            "commit": COMMIT,
            "archive_url": ARCHIVE_URL,
            "subdirectory": "request_tok",
            "raw_json_files": RAW_SOURCE_FILE_COUNT,
            "excluded_files": list(EXCLUDED_FILES),
            "curated_json_files": CURATED_SOURCE_FILE_COUNT,
            "curated_inventory_sha256": SOURCE_INVENTORY_SHA256,
            "case_variant_filenames_retained_distinctly": True,
        },
        "index_contract": {
            "index_version": INDEX_VERSION,
            "builder": "SQLite FTS5",
            "document_tokenizer": "unicode61 with SQLite defaults",
            "deduplication": "first link in sorted source-file order",
            "tie_break": "ascending FTS rowid",
        },
        "known_good_rebuild_runtime": KNOWN_GOOD_REBUILD_RUNTIME,
        "rebuild_runtime_acceptance": REBUILD_RUNTIME_ACCEPTANCE,
        "frozen_output": expected,
        "reconstructed_output": observed,
        "byte_identical_to_frozen_database": True,
        "byte_identical_to_frozen_manifest": True,
        "path_free": True,
        "model_calls": 0,
        "api_calls": 0,
        "script_sha256": sha256_file(script_path),
    }


def _download_archive(destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={"User-Agent": "CCR-artifact-reconstruction/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    partial.replace(destination)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-dir", type=Path)
    source.add_argument("--source-archive", type=Path)
    source.add_argument("--download", action="store_true")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--force",
        action="store_true",
        help="replace only the three named reconstruction outputs",
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    if args.force:
        _remove_outputs(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    try:
        if args.source_dir is not None:
            records = _directory_records(args.source_dir.resolve())
            observed = build_database(records, output_dir)
        elif args.source_archive is not None:
            records = _archive_records(args.source_archive.resolve())
            observed = build_database(records, output_dir)
        else:
            with tempfile.TemporaryDirectory(prefix="ccr-wikitables-source-") as name:
                archive_path = Path(name) / f"WikiTables-WithLinks-{COMMIT}.tar.gz"
                _download_archive(archive_path)
                observed = build_database(_archive_records(archive_path), output_dir)
        report = reconstruction_report(observed, script_path)
        (output_dir / REPORT_NAME).write_bytes(pretty_bytes(report))
    except (OSError, json.JSONDecodeError, ReconstructionError) as exc:
        _remove_outputs(output_dir)
        print(f"HybridQA BM25 reconstruction failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
