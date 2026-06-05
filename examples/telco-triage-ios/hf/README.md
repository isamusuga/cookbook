---
license: other
language:
  - en
tags:
  - gguf
  - ios
  - on-device
  - rag
  - telco
  - liquid-ai
library_name: llama.cpp
---

# Telco Triage iOS Runtime Pack

This private runtime pack supports the **Telco Triage iOS** cookbook example:
an anonymized on-device telco support assistant for home-internet Q&A, cited
RAG answers, safe local action handshakes, multi-turn follow-ups, and cloud or
agent handoff when a request is outside the local support scope.

The current customer Q&A path is **zero-generation composer RAG**:

```text
User turn
  -> conversation state
  -> BM25HierarchyRetriever over rag-units-v1.json
  -> route policy
  -> DeterministicAnswerComposer
  -> cited chat answer + source chip + optional confirmed action
```

Grounded support answers are composed from canonical RAG units. The packaged
LFM2.5-350M base model is included for optional on-device model features and
explicit tool-support paths. It is **not** used to synthesize normal support
answers in the current customer Q&A path.

## Contents

| File | Role |
| --- | --- |
| `lfm25-350m-base-Q4_K_M.gguf` | LFM2.5-350M base model used by optional on-device model features and explicit tool-support paths. |
| `telco-tool-selector-v3.gguf` | Tool-support adapter for explicit local action/tool paths. |
| `rag-units-v1.json` | Canonical RAG corpus used by the app's retriever and composer. |
| `page-link-table-v1.json` | Canonical in-app links used for source chips and navigation. |
| `model_manifest.json` | Machine-readable pack contract. |
| `checksums.sha256` | SHA-256 checksums for all shipped artifacts. |

No ColBERT model, Stage B answer LoRA, chat-mode router, topic gate,
refusal-flags adapter, relational adapter, classifier-head binaries, or raw
source documents are part of this pack.

## Download

```bash
hf auth login
hf download "$HF_REPO_ID" --local-dir models/telco
```

Then from the cookbook example:

```bash
cd examples/telco-triage-ios
TELCO_MODELS_DIR=models/telco ./bootstrap-models.sh
xcodegen generate
open TelcoTriage.xcodeproj
```

## Runtime Boundary

Normal support Q&A:

- model forwards: 0
- generation calls: 0
- Q&A LoRA adapters: 0
- citation source: selected `RAGUnit.canonicalURL`

Explicit local action/tool paths can use the packaged LFM artifacts when the app
invokes them. Unsupported or account-specific requests are handled by local
policy as cloud/system handoff or live-agent escalation; the pack does not
include customer account data.

## Expected Repository Files

The Hugging Face repository should contain only:

- `README.md`
- `checksums.sha256`
- `lfm25-350m-base-Q4_K_M.gguf`
- `telco-tool-selector-v3.gguf`
- `rag-units-v1.json`
- `page-link-table-v1.json`
- `model_manifest.json`

## Access

This pack is intended for private POC delivery. Grant access through the
Hugging Face organization or use a gated/private repo. Uploads should use a
write token or a fine-grained token scoped to this repository.
