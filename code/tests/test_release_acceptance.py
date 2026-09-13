from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import unittest
import zipfile


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import replay


ALLOWED_ROOT = {
    ".gitattributes",
    ".gitignore",
    "LICENSE",
    "README.md",
    "code",
    "configs",
    "data",
    "pyproject.toml",
    "requirements.txt",
    "requirements-gpu.txt",
    "verify.py",
}
FORBIDDEN_PARTS = {
    "work" + "ing",
    "archive",
    "rebut" + "tal",
    "submission",
    "iter" + "ations",
}
ABSENT_DIRECTORIES = (".pytest_cache", "__pycache__", ".venv")
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files() -> list[Path]:
    return [path for path in replay.release_files(ROOT) if not path.is_symlink()]


def stream_contains_private_pattern(
    handle,
    *,
    chunk_bytes: int = 1024 * 1024,
) -> bool:
    return replay.stream_contains_private_material(
        handle,
        chunk_bytes=chunk_bytes,
    )


class ReleaseAcceptanceTests(unittest.TestCase):
    def test_closed_outer_inventory(self) -> None:
        present = {
            path.name
            for path in ROOT.iterdir()
            if path.name not in replay.EXCLUDED_DIRECTORIES
            and path.name not in replay.EXCLUDED_NAMES
        }
        self.assertEqual(present, ALLOWED_ROOT)
        manifest = json.loads(
            (ROOT / "data/manifests/ARTIFACT_MANIFEST.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(files()), len(manifest["members"]) + 1)
        self.assertEqual(
            [path.relative_to(ROOT).as_posix() for path in files() if path.suffix.lower() == ".md"],
            ["README.md"],
        )
        for name in ABSENT_DIRECTORIES:
            self.assertEqual(list(ROOT.rglob(name)), [], name)
        for path in ROOT.rglob("*"):
            relative = path.relative_to(ROOT)
            if replay.EXCLUDED_DIRECTORIES.intersection(relative.parts):
                continue
            if path.name in replay.EXCLUDED_NAMES:
                continue
            self.assertFalse(path.is_symlink(), relative.as_posix())
            self.assertFalse(any(part in FORBIDDEN_PARTS for part in relative.parts), relative.as_posix())

    def test_manifest_and_checksums_close_over_every_file(self) -> None:
        manifest_path = ROOT / "data/manifests/ARTIFACT_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            path.relative_to(ROOT).as_posix(): path
            for path in files()
            if path != manifest_path
        }
        rows = manifest["members"]
        declared = {row["path"]: row for row in rows}
        self.assertEqual(len(rows), len(declared))
        self.assertEqual(set(declared), set(expected))
        for name, path in expected.items():
            with self.subTest(path=name):
                self.assertEqual(declared[name]["bytes"], path.stat().st_size)
                self.assertEqual(declared[name]["sha256"], sha256(path))

        checksum_path = ROOT / "data/manifests/RELEASE_SHA256SUMS.txt"
        checksums = {}
        for line in checksum_path.read_text(encoding="ascii").splitlines():
            digest, separator, name = line.partition("  ")
            self.assertEqual(separator, "  ")
            self.assertEqual(len(digest), 64)
            self.assertNotIn(name, checksums)
            checksums[name] = digest
        checksum_expected = {
            name: path
            for name, path in expected.items()
            if path != checksum_path
        }
        self.assertEqual(set(checksums), set(checksum_expected))
        for name, path in checksum_expected.items():
            with self.subTest(checksum=name):
                self.assertEqual(checksums[name], sha256(path))

    def test_nested_archives_are_safe_and_private(self) -> None:
        nested_members = 0
        nested_archives = 0
        for path in files():
            if path.suffix.lower() != ".zip":
                continue
            nested_archives += 1
            with zipfile.ZipFile(path) as archive:
                self.assertIsNone(archive.testzip(), path.relative_to(ROOT).as_posix())
                names = [info.filename for info in archive.infolist()]
                self.assertEqual(len(names), len(set(names)))
                nested_members += len(names)
                for info in archive.infolist():
                    name = info.filename
                    relative = PurePosixPath(name)
                    mode = info.external_attr >> 16
                    self.assertTrue(name and not name.startswith("/"), name)
                    self.assertNotIn("..", relative.parts)
                    self.assertNotIn("\\", name)
                    self.assertFalse(stat.S_ISLNK(mode), name)
                    self.assertFalse(any(part in FORBIDDEN_PARTS for part in relative.parts), name)
                    self.assertNotEqual(relative.suffix.lower(), ".md", name)
                    with archive.open(info) as member:
                        self.assertFalse(
                            stream_contains_private_pattern(member), name
                        )
        self.assertEqual(nested_archives, 2)
        self.assertEqual(nested_members, 9)

    def test_nested_archive_streaming_privacy_crosses_chunks(self) -> None:
        payload = (
            b"safe-prefix-"
            + b"reviewer"
            + b"@"
            + b"example"
            + b".org"
            + b"-safe-suffix"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("nested.bin", payload)
        with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
            with archive.open("nested.bin") as member:
                self.assertTrue(
                    stream_contains_private_pattern(member, chunk_bytes=3)
                )
        clean = io.BytesIO(b"ordinary reviewer-safe nested member")
        self.assertFalse(stream_contains_private_pattern(clean, chunk_bytes=2))

    def test_streaming_privacy_matches_materialized(self) -> None:
        cases = {
            "forbidden_path": b"prefix-" + replay.FORBIDDEN_CONTENT[0] + b"-suffix",
            "machine_path": (
                b"prefix /" + b"home" + b"/reviewer/project/file.json suffix"
            ),
            "hidden_directory": (
                b"prefix " + b"." + b"private/attachments/cache suffix"
            ),
            "email": (
                b"prefix-"
                + b"reviewer"
                + b"@"
                + b"example"
                + b".org"
                + b"-suffix"
            ),
            "clean": b"ordinary reviewer-safe nested member",
        }
        expected = {
            name: replay.contains_private_material(payload)
            for name, payload in cases.items()
        }
        self.assertEqual(
            expected,
            {
                "forbidden_path": True,
                "machine_path": True,
                "hidden_directory": True,
                "email": True,
                "clean": False,
            },
        )
        for name, payload in cases.items():
            for chunk_bytes in range(1, 8):
                with self.subTest(name=name, chunk_bytes=chunk_bytes):
                    self.assertEqual(
                        replay.stream_contains_private_material(
                            io.BytesIO(payload),
                            chunk_bytes=chunk_bytes,
                        ),
                        expected[name],
                    )

    def test_manifest_only_replay(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "verify.py", "--manifest-only"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "verified")
        self.assertFalse(report["verification_used_model_or_api_calls"])
        self.assertFalse(report["network_used"])
        self.assertFalse(report["credentials_used"])
        self.assertEqual(report["privacy"]["nested_archive_members_checked"], 9)
        self.assertEqual(report["scope"]["status"], "rop_paper_scoped")
        self.assertTrue(
            report["scope"]["evidence_collection_used_model_or_api_calls"]
        )
        self.assertFalse(report["scope"]["verification_uses_model_or_api_calls"])
        self.assertEqual(
            report["scope"]["paper_title"],
            "Does the Selected Object Reach the Reader? Auditing Identity Handoffs in Grounded Language-Model Pipelines",
        )
        binding = json.loads(
            (ROOT / "data/manifests/PAPER_BINDING.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["paper"]["bound_pdf_sha256"], binding["paper_pdf"]["sha256"])
        self.assertEqual(
            report["paper"]["bound_source_sha256"], binding["paper_source"]["sha256"]
        )


if __name__ == "__main__":
    unittest.main()
