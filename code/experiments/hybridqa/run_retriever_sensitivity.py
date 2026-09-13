"""Run the frozen HybridQA neural-retriever sensitivity sweep on Colab.

Heavy dependencies are imported only inside the stages that require them.
Run ``--help`` or the CPU-only ``prepare/export/verify`` stages locally.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

from ccr.retrieval.bm25 import open_index, retrieve
from ccr.retrieval.hybridqa_sensitivity import (
    ResultStore,
    atomic_json,
    export_results,
    iter_jsonl,
    prepare_assets,
    reciprocal_rank_fusion,
    sha256_file,
    verify_export,
)

RANDOM_SEED = 12_345


def _set_seed(seed: int = RANDOM_SEED) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _read_collection(path: Path) -> tuple[list[int], list[str]]:
    pids, texts = [], []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                raw_pid, text = line.rstrip("\n").split("\t", 1)
                pid = int(raw_pid)
            except Exception as error:
                raise RuntimeError(f"invalid collection row {line_number}") from error
            if pid != len(pids):
                raise RuntimeError(f"collection PID is not contiguous at row {line_number}")
            pids.append(pid)
            texts.append(text)
    return pids, texts


def _load_links(path: Path) -> tuple[list[str], dict[str, int]]:
    rows = list(iter_jsonl(path))
    links = [str(row["link"]) for row in rows]
    if [int(row["pid"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError("links PIDs are not contiguous")
    if len(set(links)) != len(links):
        raise RuntimeError("links mapping is not unique")
    return links, {link: pid for pid, link in enumerate(links)}


def _gpu_profile() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "device": "cpu", "name": "cpu", "memory_gib": 0,
                "corpus_batch": 8, "query_batch": 16, "rerank_batch": 8,
            }
        props = torch.cuda.get_device_properties(0)
        memory = props.total_memory / (1024**3)
        large = memory >= 70
        profile = {
            "device": "cuda:0",
            "name": props.name,
            "memory_gib": round(memory, 2),
            "corpus_batch": 256 if large else 96,
            "query_batch": 512 if large else 192,
            "rerank_batch": 256 if large else 64,
        }
        for key, variable in (
            ("corpus_batch", "CCR_CORPUS_BATCH"),
            ("query_batch", "CCR_QUERY_BATCH"),
            ("rerank_batch", "CCR_RERANK_BATCH"),
        ):
            if os.environ.get(variable):
                profile[key] = max(1, int(os.environ[variable]))
        return profile
    except ImportError:
        return {
            "device": "cpu", "name": "cpu", "memory_gib": 0,
            "corpus_batch": 8, "query_batch": 16, "rerank_batch": 8,
        }


def _adaptive_cuda_batch(
    label: str, initial_batch: int, operation: Callable[[int], Any]
) -> tuple[Any, int]:
    """Retry a GPU operation with geometric batch backoff on genuine CUDA OOM."""

    batch_size = max(1, int(initial_batch))
    while True:
        try:
            return operation(batch_size), batch_size
        except RuntimeError as error:
            message = str(error).lower()
            if "out of memory" not in message or batch_size == 1:
                raise
            reduced = max(1, batch_size // 2)
            print(
                f"{label}: CUDA OOM at batch={batch_size}; retrying batch={reduced}",
                flush=True,
            )
            torch = sys.modules.get("torch")
            if torch is not None:
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
            batch_size = reduced


def _resolved_revision(model_name: str, requested_revision: str | None) -> str:
    from huggingface_hub import model_info

    info = model_info(model_name, revision=requested_revision or "main")
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a commit for {model_name}")
    return str(info.sha)


def _snapshot(model_name: str, revision: str, root: Path) -> str:
    from huggingface_hub import snapshot_download

    cache_root = Path(os.environ.get("CCR_MODEL_CACHE_ROOT", str(root / "models")))
    cache_root.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=model_name,
        revision=revision,
        cache_dir=str(cache_root),
    )


def _file_inventory(directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"index directory is missing: {directory}")
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def _progress(root: Path, stage: str, counts: dict[str, int]) -> None:
    atomic_json(root / "checkpoints" / f"progress_{stage}.json", counts)
    print(stage, json.dumps(counts, sort_keys=True), flush=True)


def build_bge_index(config: dict[str, Any], assets: Path, root: Path) -> dict[str, Any]:
    import faiss
    import numpy as np
    from FlagEmbedding import BGEM3FlagModel

    _set_seed()
    profile = _gpu_profile()
    model_cfg = config["models"]["bge_m3"]
    model_name = str(model_cfg["name"])
    index_dir = root / "indexes" / "bge_m3"
    manifest_path = index_dir / "manifest.json"
    model_lock_path = index_dir / "model_lock.json"
    prior = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    lock = json.loads(model_lock_path.read_text()) if model_lock_path.is_file() else {}
    if lock and lock.get("model_name") != model_name:
        raise RuntimeError("BGE model name changed after the first checkpoint")
    revision = str(
        prior.get("model_revision")
        or lock.get("model_revision")
        or _resolved_revision(model_name, model_cfg.get("revision"))
    )
    index_path = index_dir / "bge_m3_flatip.faiss"
    if prior.get("status") == "complete":
        if (
            prior.get("collection_sha256") != sha256_file(assets / "collection.tsv")
            or not index_path.is_file()
            or prior.get("index_sha256") != sha256_file(index_path)
        ):
            raise RuntimeError("completed BGE index failed its hash checks")
        return prior
    atomic_json(model_lock_path, {"model_name": model_name, "model_revision": revision})
    snapshot = _snapshot(model_name, revision, root)
    shard_dir = index_dir / "embedding_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    _, texts = _read_collection(assets / "collection.tsv")
    model = BGEM3FlagModel(
        snapshot,
        use_fp16=profile["device"] != "cpu",
        devices=[profile["device"]],
        trust_remote_code=False,
    )
    shard_size = int(config["execution"]["embedding_shard_size"])
    shard_records = []
    active_corpus_batch = int(profile["corpus_batch"])
    for start in range(0, len(texts), shard_size):
        stop = min(start + shard_size, len(texts))
        path = shard_dir / f"{start:09d}-{stop:09d}.npy"
        marker = path.with_suffix(".json")
        expected = {"start": start, "stop": stop, "model_revision": revision}
        if path.is_file() and marker.is_file():
            record = json.loads(marker.read_text())
            if any(record.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"incompatible BGE checkpoint: {marker}")
            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"corrupt BGE checkpoint: {path}")
            shard_records.append(record)
            _progress(
                root,
                "bge_index",
                {"documents_complete": stop, "documents_total": len(texts)},
            )
            continue
        encoded_result, active_corpus_batch = _adaptive_cuda_batch(
            "bge-index",
            active_corpus_batch,
            lambda batch: model.encode(
                texts[start:stop],
                batch_size=batch,
                max_length=int(model_cfg["document_max_length"]),
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            ),
        )
        encoded = encoded_result["dense_vecs"]
        values = np.asarray(encoded, dtype="float32")
        faiss.normalize_L2(values)
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        record = {
            **expected,
            "shape": list(values.shape),
            "sha256": sha256_file(path),
            "effective_batch_size": active_corpus_batch,
        }
        atomic_json(marker, record)
        shard_records.append(record)
        _progress(
            root,
            "bge_index",
            {"documents_complete": stop, "documents_total": len(texts)},
        )

    dimension = int(shard_records[0]["shape"][1])
    index = faiss.IndexFlatIP(dimension)
    for record in shard_records:
        path = shard_dir / f"{record['start']:09d}-{record['stop']:09d}.npy"
        index.add(np.load(path, mmap_mode="r"))
    temporary_index = index_path.with_suffix(".faiss.tmp")
    faiss.write_index(index, str(temporary_index))
    os.replace(temporary_index, index_path)
    manifest = {
        "status": "complete",
        "model_name": model_name,
        "model_revision": revision,
        "index_type": "exact_inner_product_on_l2_normalized_dense_vectors",
        "document_count": index.ntotal,
        "dimension": dimension,
        "document_max_length": int(model_cfg["document_max_length"]),
        "random_seed": RANDOM_SEED,
        "collection_sha256": sha256_file(assets / "collection.tsv"),
        "index_sha256": sha256_file(index_path),
        "shards": shard_records,
        "gpu_profile": profile,
    }
    atomic_json(index_dir / "manifest.json", manifest)
    return manifest


def build_colbert_index(config: dict[str, Any], assets: Path, root: Path) -> dict[str, Any]:
    import colbert.infra.launcher as colbert_launcher
    from colbert import Indexer
    from colbert.infra import ColBERTConfig, Run, RunConfig

    _set_seed()
    profile = _gpu_profile()
    expected_version = str(config["runtime"]["colbert_ai_version"])
    actual_version = importlib.metadata.version("colbert-ai")
    if actual_version != expected_version:
        raise RuntimeError(
            f"colbert-ai version mismatch: {actual_version} != {expected_version}"
        )
    index_source = inspect.getsource(Indexer.index)
    if "'resume'" not in index_source or "resume=overwrite=='resume'" not in index_source:
        raise RuntimeError("pinned ColBERT Indexer lacks the audited resume API")
    indexer_source_sha256 = hashlib.sha256(index_source.encode()).hexdigest()
    launcher_source = inspect.getsource(colbert_launcher)
    if "set_seed(12345)" not in launcher_source:
        raise RuntimeError("pinned ColBERT launcher lacks the audited worker seed")
    launcher_source_sha256 = hashlib.sha256(launcher_source.encode()).hexdigest()
    model_cfg = config["models"]["colbert_v2"]
    model_name = str(model_cfg["name"])
    output = root / "indexes" / "colbert_v2"
    manifest_path = output / "manifest.json"
    model_lock_path = output / "model_lock.json"
    prior = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    lock = json.loads(model_lock_path.read_text()) if model_lock_path.is_file() else {}
    if lock and lock.get("model_name") != model_name:
        raise RuntimeError("ColBERT model name changed after the first checkpoint")
    revision = str(
        prior.get("model_revision")
        or lock.get("model_revision")
        or _resolved_revision(model_name, model_cfg.get("revision"))
    )
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(model_lock_path, {"model_name": model_name, "model_revision": revision})
    snapshot = _snapshot(model_name, revision, root)
    collection_hash = sha256_file(assets / "collection.tsv")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "complete":
            if (
                manifest["collection_sha256"] != collection_hash
                or manifest["model_revision"] != revision
            ):
                raise RuntimeError("completed ColBERT index has incompatible frozen inputs")
            index_root = Path(manifest["index_root"])
            if _file_inventory(index_root) != manifest.get("file_inventory"):
                raise RuntimeError("completed ColBERT index failed its file hash checks")
            return manifest
    colbert_config = ColBERTConfig(
        nbits=int(model_cfg["nbits"]),
        doc_maxlen=int(model_cfg["document_max_length"]),
        query_maxlen=int(model_cfg["query_max_length"]),
        kmeans_niters=int(model_cfg["kmeans_iterations"]),
    )
    # Native ColBERT stores plan/chunk checkpoints under this persistent root.
    # resume=True reuses all completed chunks after a Colab disconnect.
    with Run().context(RunConfig(root=str(output), experiment="frozen", nranks=1)):
        Indexer(checkpoint=snapshot, config=colbert_config).index(
            name="hybridqa_colbert_v2",
            collection=str(assets / "collection.tsv"),
            overwrite="resume",
        )
    index_root = output / "frozen" / "indexes" / "hybridqa_colbert_v2"
    manifest = {
        "status": "complete",
        "model_name": model_name,
        "model_revision": revision,
        "collection_sha256": collection_hash,
        "nbits": int(model_cfg["nbits"]),
        "document_max_length": int(model_cfg["document_max_length"]),
        "query_max_length": int(model_cfg["query_max_length"]),
        "random_seed": RANDOM_SEED,
        "colbert_ai_version": actual_version,
        "indexer_source_sha256": indexer_source_sha256,
        "launcher_source_sha256": launcher_source_sha256,
        "resume_mode": "Indexer.index(overwrite='resume')",
        "index_root": str(index_root),
        "file_inventory": _file_inventory(index_root),
        "gpu_profile": profile,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def _run_bge_batched(
    config: dict[str, Any], assets: Path, root: Path, queries: list[dict[str, Any]]
) -> dict[str, int]:
    import faiss
    import numpy as np
    from FlagEmbedding import BGEM3FlagModel

    _set_seed()
    profile = _gpu_profile()
    manifest = json.loads((root / "indexes" / "bge_m3" / "manifest.json").read_text())
    snapshot = _snapshot(manifest["model_name"], manifest["model_revision"], root)
    links, _ = _load_links(assets / "links.jsonl")
    model = BGEM3FlagModel(
        snapshot, use_fp16=profile["device"] != "cpu", devices=[profile["device"]]
    )
    cpu_index = faiss.read_index(
        str(root / "indexes" / "bge_m3" / "bge_m3_flatip.faiss")
    )
    index = cpu_index
    if profile["device"] != "cpu":
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index)
    store = ResultStore(root / "checkpoints" / "rankings.sqlite")
    counts = {"completed": 0, "resumed": 0, "upstream_invalid": 0}
    pending = []
    for query in queries:
        if query.get("text") is None:
            counts["upstream_invalid"] += 1
        elif store.get("bge_m3", query) is not None:
            counts["resumed"] += 1
        else:
            pending.append(query)
    batch_size = int(profile["query_batch"])
    active_query_batch = batch_size
    top_k = int(config["retrieval"]["candidate_k"])
    try:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            encoded_result, active_query_batch = _adaptive_cuda_batch(
                "bge-query",
                active_query_batch,
                lambda inner_batch: model.encode(
                    [str(query["text"]) for query in batch],
                    batch_size=inner_batch,
                    max_length=int(config["models"]["bge_m3"]["query_max_length"]),
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                ),
            )
            vectors = encoded_result["dense_vecs"]
            vectors = np.asarray(vectors, dtype="float32")
            faiss.normalize_L2(vectors)
            scores, pids = index.search(vectors, top_k)
            for query, query_pids, query_scores in zip(batch, pids, scores):
                ranking = [
                    {
                        "rank": rank,
                        "pid": int(pid),
                        "link": links[int(pid)],
                        "score": float(score),
                    }
                    for rank, (pid, score) in enumerate(
                        zip(query_pids, query_scores), start=1
                    )
                    if pid >= 0
                ]
                store.put("bge_m3", query, ranking)
                counts["completed"] += 1
            _progress(
                root,
                "bge_run",
                {
                    **counts,
                    "callable_total": len(pending) + counts["resumed"],
                    "effective_batch_size": active_query_batch,
                },
            )
        return counts
    finally:
        store.close()


def _colbert_retriever(config: dict[str, Any], assets: Path, root: Path):
    from colbert import Searcher
    from colbert.infra import Run, RunConfig

    _set_seed()
    links, _ = _load_links(assets / "links.jsonl")
    index_root = root / "indexes" / "colbert_v2"
    top_k = int(config["retrieval"]["candidate_k"])
    context = Run().context(RunConfig(root=str(index_root), experiment="frozen", nranks=1))
    context.__enter__()
    searcher = Searcher(index="hybridqa_colbert_v2", collection=str(assets / "collection.tsv"))

    def search(text: str) -> list[dict[str, Any]]:
        pids, ranks, scores = searcher.search(text, k=top_k)
        return [
            {"rank": int(rank), "pid": int(pid), "link": links[int(pid)], "score": float(score)}
            for pid, rank, score in zip(pids, ranks, scores)
        ]

    return search


def run_base_retriever(
    name: str, config: dict[str, Any], assets: Path, root: Path
) -> dict[str, int]:
    queries = list(iter_jsonl(assets / "queries.jsonl"))
    if name == "bge_m3":
        return _run_bge_batched(config, assets, root, queries)
    store = ResultStore(root / "checkpoints" / "rankings.sqlite")
    search = _colbert_retriever(config, assets, root)
    try:
        counts = {"completed": 0, "resumed": 0, "upstream_invalid": 0}
        for query in queries:
            if query.get("text") is None:
                counts["upstream_invalid"] += 1
            elif store.get(name, query) is not None:
                counts["resumed"] += 1
            else:
                store.put(name, query, search(str(query["text"])))
                counts["completed"] += 1
            if sum(counts.values()) % 50 == 0:
                _progress(root, "colbert_run", counts)
        _progress(root, "colbert_run", counts)
        return counts
    finally:
        store.close()


def run_hybrid(config: dict[str, Any], assets: Path, root: Path, *, rerank: bool) -> dict[str, int]:
    _set_seed()
    queries = list(iter_jsonl(assets / "queries.jsonl"))
    links, link_to_pid = _load_links(assets / "links.jsonl")
    store = ResultStore(root / "checkpoints" / "rankings.sqlite")
    bm25 = open_index(Path(config["inputs"]["bm25_db"]))
    rrf_k = int(config["retrieval"]["rrf_k"])
    candidate_k = int(config["retrieval"]["candidate_k"])
    output_k = int(config["retrieval"]["output_k"])
    reranker = None
    passage_by_pid: dict[int, str] = {}
    profile = _gpu_profile()
    if rerank:
        from FlagEmbedding import FlagReranker

        reranker_cfg = config["models"]["bge_reranker_v2_m3"]
        reranker_marker = root / "checkpoints" / "reranker_model.json"
        locked_reranker = (
            json.loads(reranker_marker.read_text()) if reranker_marker.is_file() else {}
        )
        if locked_reranker and locked_reranker.get("name") != reranker_cfg["name"]:
            raise RuntimeError("reranker model changed after the first checkpoint")
        revision = str(
            locked_reranker.get("revision")
            or _resolved_revision(str(reranker_cfg["name"]), reranker_cfg.get("revision"))
        )
        snapshot = _snapshot(str(reranker_cfg["name"]), revision, root)
        reranker = FlagReranker(
            snapshot, use_fp16=profile["device"] != "cpu", devices=[profile["device"]]
        )
        db_uri = f"file:{Path(config['inputs']['bm25_db']).resolve()}?mode=ro"
        connection = sqlite3.connect(db_uri, uri=True)
        corpus_rows = connection.execute("SELECT link,text FROM passage_fts ORDER BY rowid")
        passage_by_pid = {pid: text for pid, (_, text) in enumerate(corpus_rows)}
        connection.close()
        atomic_json(
            reranker_marker,
            {
                "name": reranker_cfg["name"],
                "revision": revision,
                "profile": profile,
                "random_seed": RANDOM_SEED,
            },
        )
    name = "hybrid_rrf_bge_reranker_v2_m3" if rerank else "hybrid_rrf"

    def search(query: str, query_row: dict[str, Any]) -> list[dict[str, Any]]:
        dense = store.get("bge_m3", query_row)
        if dense is None:
            raise RuntimeError(f"BGE ranking missing for {query_row['query_id']}")
        sparse = []
        for row in retrieve(bm25, query, candidate_k):
            pid = link_to_pid.get(str(row["link"]))
            if pid is not None:
                sparse.append({**row, "pid": pid})
        fused = reciprocal_rank_fusion([sparse, dense], rrf_k=rrf_k, limit=candidate_k)
        if not rerank:
            return fused[:output_k]
        assert reranker is not None
        pairs = [[query, passage_by_pid[int(row["pid"])] ] for row in fused]
        scores, effective_rerank_batch = _adaptive_cuda_batch(
            "bge-reranker",
            int(profile["rerank_batch"]),
            lambda batch: reranker.compute_score(
                pairs,
                batch_size=batch,
                max_length=int(config["models"]["bge_reranker_v2_m3"]["max_length"]),
                normalize=True,
            ),
        )
        if effective_rerank_batch != profile["rerank_batch"]:
            profile["rerank_batch"] = effective_rerank_batch
        scored = [
            {**row, "score": float(score), "rrf_score": row["score"]}
            for row, score in zip(fused, scores)
        ]
        scored.sort(key=lambda row: (-row["score"], -row["rrf_score"], row["pid"]))
        return [{**row, "rank": rank} for rank, row in enumerate(scored[:output_k], start=1)]

    counts = {"completed": 0, "resumed": 0, "upstream_invalid": 0}
    try:
        for query in queries:
            if query.get("text") is None:
                counts["upstream_invalid"] += 1
            elif store.get(name, query) is not None:
                counts["resumed"] += 1
            else:
                store.put(name, query, search(str(query["text"]), query))
                counts["completed"] += 1
            if sum(counts.values()) % 50 == 0:
                _progress(root, name, counts)
        _progress(root, name, counts)
        return counts
    finally:
        bm25.close()
        store.close()


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    for key in ("runtime", "inputs", "models", "retrieval", "execution"):
        if key not in config:
            raise RuntimeError(f"config is missing {key}")
    if int(config["runtime"].get("random_seed", -1)) != RANDOM_SEED:
        raise RuntimeError("runtime random seed differs from the frozen implementation")
    return config


def _validate_frozen_inputs(config: dict[str, Any]) -> None:
    expected = config.get("expected_input_sha256", {})
    frozen_inputs = (
        ("cohort", "cohort"),
        ("bridge_details", "bridge_details"),
        ("bm25_db", "bm25_db"),
        ("bm25_manifest", "bm25_manifest"),
    )
    for key, input_key in frozen_inputs:
        path = Path(config["inputs"][input_key])
        if not path.is_file():
            raise FileNotFoundError(f"required frozen input is missing: {path}")
        actual = sha256_file(path)
        if expected.get(key) and actual != expected[key]:
            raise RuntimeError(
                f"frozen input hash mismatch for {key}: {actual} != {expected[key]}"
            )


def _validate_prepared_assets(assets: Path) -> None:
    manifest_path = assets / "prepare_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("prepare stage has not completed")
    manifest = json.loads(manifest_path.read_text())
    for key, name in (
        ("queries_sha256", "queries.jsonl"),
        ("collection_sha256", "collection.tsv"),
        ("links_sha256", "links.jsonl"),
    ):
        path = assets / name
        if not path.is_file() or sha256_file(path) != manifest.get(key):
            raise RuntimeError(f"prepared asset failed its hash check: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "prepare", "bge-index", "colbert-index", "bge-run", "colbert-run",
            "hybrid-run", "reranker-run", "export", "verify",
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, required=True, help="persistent Drive/local run directory"
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    assets = args.root / "assets"
    args.root.mkdir(parents=True, exist_ok=True)
    config_hash = sha256_file(args.config)
    config_lock_path = args.root / "config_lock.json"
    if config_lock_path.is_file():
        locked = json.loads(config_lock_path.read_text())
        if locked.get("config_sha256") != config_hash:
            raise RuntimeError("configuration changed after this run root was initialized")
    else:
        atomic_json(config_lock_path, {"config_sha256": config_hash})
    _validate_frozen_inputs(config)
    if args.stage == "prepare":
        result = prepare_assets(
            Path(config["inputs"]["cohort"]), Path(config["inputs"]["bridge_details"]),
            Path(config["inputs"]["bm25_db"]), Path(config["inputs"]["bm25_manifest"]), assets,
        )
        print(json.dumps(result, indent=2))
        return 0
    _validate_prepared_assets(assets)
    if args.stage == "bge-index":
        result = build_bge_index(config, assets, args.root)
    elif args.stage == "colbert-index":
        result = build_colbert_index(config, assets, args.root)
    elif args.stage == "bge-run":
        result = run_base_retriever("bge_m3", config, assets, args.root)
    elif args.stage == "colbert-run":
        result = run_base_retriever("colbert_v2", config, assets, args.root)
    elif args.stage == "hybrid-run":
        result = run_hybrid(config, assets, args.root, rerank=False)
    elif args.stage == "reranker-run":
        result = run_hybrid(config, assets, args.root, rerank=True)
    elif args.stage == "export":
        result = export_results(
            assets,
            args.root / "checkpoints" / "rankings.sqlite",
            args.root / "results",
        )
    else:
        result = verify_export(assets, args.root / "results")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
