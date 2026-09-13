#!/usr/bin/env python3
"""Regenerate the checksum inventory and the closed manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay

EXCLUDED_DIRECTORIES = replay.EXCLUDED_DIRECTORIES
EXCLUDED_NAMES = replay.EXCLUDED_NAMES

MANIFEST = "data/manifests/ARTIFACT_MANIFEST.json"
CHECKSUMS = "data/manifests/RELEASE_SHA256SUMS.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_files(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.name not in EXCLUDED_NAMES
            and not EXCLUDED_DIRECTORIES.intersection(path.relative_to(root).parts)
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    checksum_path = root / CHECKSUMS
    manifest_path = root / MANIFEST
    checksum_members = [
        path
        for path in release_files(root)
        if path not in {checksum_path, manifest_path}
    ]
    checksum_text = "".join(
        f"{sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in checksum_members
    )
    checksum_path.write_text(checksum_text, encoding="ascii")

    manifest_members = [
        {
            "bytes": path.stat().st_size,
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
        }
        for path in release_files(root)
        if path != manifest_path
    ]
    manifest = {
        "hash_algorithm": "sha256",
        "manifest_excludes": [MANIFEST],
        "members": manifest_members,
        "schema_version": 1,
        "status": "complete",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "checksum_members": len(checksum_members),
                "manifest_members": len(manifest_members),
                "status": "refreshed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
