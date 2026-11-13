# Does the Selected Object Reach the Reader?

Code, data, and an offline replay for the paper *Does the Selected Object Reach the Reader?
Auditing Identity Handoffs in Grounded Language-Model Pipelines*.

Siddharth Vohra, Runmin Jiang, Xiaomo Li, and Min Xu (Carnegie Mellon University).
Published in the Proceedings of the 1st Workshop on Grounding Language Models (GroundLM 2026)
at EMNLP 2026.

- Paper: https://aclanthology.org/2026.groundlm-1.70/
- Project page: https://siddvoh.com/identity-handoffs/
- arXiv preprint: https://arxiv.org/abs/2609.04579

## What the work is

A grounded language-model pipeline does three things in sequence: it selects an object, it
retrieves passages for that object, and it answers from the retrieved evidence. If the object
the selector picked never reaches the reader, the handoff between those stages has failed.
Benchmark recall does not catch this, because it scores the dataset-linked object, which is not
always the object the selector chose.

We audit 600 human-written HybridQA questions across three selector families. On the 1,463
resolvable records where the selected object matches the dataset-traced passage, exact key
lookup and exact title matching return the object every time. Given the same decoded selected
title, body-only BM25 omits it on 389 records (26.6 percent) at cutoff five, while hybrid
retrieval with reranking omits it on 14 records (1.0 percent). The two identities differ on 329
of 1,792 resolvable records, and with original-question rankings their top-five checks disagree
on 106 records (5.9 percent). Frozen reader comparisons associate the aligned object's presence
with 28.6 to 31.0 points higher exact match. On a deliberately selected 64-item cohort, deleting
that passage lowers exact match sharply, while deleting a comparison passage of similar length
does not reproduce the drop.

The released instrument is the Returned-Object Profile (ROP): an executable record of the target,
the returned-ID field, the cutoff, the membership rule, and the complete expected population.

## Layout

```
verify.py            the single entry point; recomputes the reported aggregates offline
code/
  ccr/               the implementation the experiments ran
  ccr_retrieval/     the import shim the frozen implementation hash was taken through
  contracts/         the ROP schema, reference evaluator, kernel, fixtures and examples
  experiments/       the programs that ran the experiments
  analysis/          the programs that recompute the reported aggregates
  tests/             contract, replay, privacy and release acceptance tests
  tools/             regenerates the checksum manifests
  replay.py          the replay driver verify.py calls
configs/             the frozen experiment configurations
data/
  released/          the question cohorts
  results/released/  the frozen result bundles and retriever rankings
  evidence/          one deterministic report per reported aggregate
  generated/         sanitized hosted-call and Qwen authority archives
  manifests/         the checksum lists, the paper binding, and the run provenance
```

## Install

Nothing to install for verification. It needs CPython 3.10 or newer on macOS or Linux, about
150 MiB of free space, and the Python standard library. It uses no credentials and makes no
network calls. `requirements.txt` and `requirements-gpu.txt` cover the experiment code and the
GPU retriever runs, neither of which verification touches.

## Reproduce

```bash
PYTHONDONTWRITEBYTECODE=1 python3 verify.py
cd code && PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_*.py'
```

A successful replay reports `status: verified`, `verification_used_model_or_api_calls: false`,
`credentials_used: false`, and `network_used: false`. It accounts for 10,192 issued primary
hosted-model calls, 10,194 transport attempts, and eight planned answer cells that were not
issued after invalid upstream outputs. It also verifies 9,600 retriever records, the 335-question
Qwen decoded-ID replication, and the 561-task Gemini intervention, of which 537 were valid
successes and 24 were terminal failures.

To check a paper PDF and TeX source against the recorded bindings:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 verify.py --manifest-only \
  --paper-pdf /path/to/main.pdf \
  --paper-source /path/to/main.tex
```

To exercise only the ROP schema and reference evaluator:

```bash
cd code && PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tests -p 'test_returned_object_preservation*.py'
```

## Where the recorded paths point

The evidence files under `data/evidence/` record the path each input had when the experiment ran.
Every data path resolves as recorded. Three code paths moved, and `code/replay.py` maps them in
`RECORDED_PATH_PREFIXES` before reading:

| Recorded in the evidence       | Where the file is now |
| ------------------------------ | --------------------- |
| `code/src/ccr/...`             | `code/ccr/...`        |
| `code/configs/experiments/...` | `configs/...`         |
| `scripts/...`                  | `code/analysis/...`   |

## Integrity

Every file is bound by SHA-256 in `data/manifests/ARTIFACT_MANIFEST.json`, with a transport
checksum list in `data/manifests/RELEASE_SHA256SUMS.txt`. Editing any file invalidates both and
fails verification. `code/tools/refresh_artifact_inventories.py --root .` regenerates them.

Each cohort file carries a `config_sha256` taken over the configuration as it stood when that run
was made, so hashing `configs/hybridqa_natural_v2.yaml` or `configs/hybridqa_natural_v3.yaml` today
gives a different digest. `code/analysis/recompute_mismatch_stack_cutoffs.py` rebuilds the v2 form
and checks it against the run manifest that used it.

## Data

The primary cohort is 600 human-written HybridQA questions selected by a fixed hash rule, at most
one question per table. The hosted-model records cover question-only, predicted-fact, and
query-rewrite conditions for three model families, with Gemini no-knowledge-base and
oracle-passage controls. The retrieval release covers five frozen stacks. The Qwen release
contains 335 selector rows and a 1,397-item development audit with 13,970 ranking cells. Invalid
responses, terminal failures, and unissued answer cells stay explicit in their declared
denominators rather than being dropped.

## Paper and archives

The paper source and compiled PDF are distributed separately and bound to this package by SHA-256
in `data/manifests/PAPER_BINDING.json`. The binding is to the authors' own build. The copy in the
ACL Anthology carries the proceedings page numbers and footer, so its hash differs; it is the
version to cite.

`code/analysis/reconstruct_hybridqa_bm25.py --download` rebuilds the keyword index from the pinned
WikiTables-WithLinks commit. It is the only command here that reaches the network, and the
verification path never calls it.

The hosted-call archives under `data/generated/` are sanitized. The three route and schema
preflight calls made during the Gemini intervention are stored separately and excluded from the
analysis, and that exclusion is recorded in the evidence.

## License

MIT, see `LICENSE`. The HybridQA questions and tables come from the public HybridQA release and
keep their own terms.

## Citation

```bibtex
@inproceedings{vohra-etal-2026-selected,
  title     = {Does the Selected Object Reach the Reader? Auditing Identity Handoffs in Grounded Language-Model Pipelines},
  author    = {Vohra, Siddharth and Jiang, Runmin and Li, Xiaomo and Xu, Min},
  booktitle = {Proceedings of the 1st Workshop on Grounding Language Models: Learning Faithfully and Efficiently ({G}round{LM} 2026)},
  month     = oct,
  year      = {2026},
  address   = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics},
  pages     = {966--980},
  doi       = {10.18653/2026.groundlm-1.70},
  url       = {https://aclanthology.org/2026.groundlm-1.70/}
}
```

## Contact

Siddharth Vohra, Carnegie Mellon University.
