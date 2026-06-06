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

The current customer Q&A path is **single-forward understanding + grounded
composer RAG**:

```text
User turn
  -> conversation state
  -> LFM2.5-350M shared understanding pass (9 heads)
  -> BM25HierarchyRetriever over rag-units-v1.json
  -> route policy
  -> DeterministicAnswerComposer
  -> optional V4 dialogue-repair verbalizer for selected follow-up wording
  -> cited chat answer + source chip + optional confirmed action
```

Grounded support answers are composed from canonical RAG units. The LFM produces
compact understanding signals; Swift policy owns the route, source, citation,
tool confirmation, and handoff decisions. The V4 dialogue repair verbalizer is
bounded to customer-facing wording for repair/follow-up turns and must echo the
provided route/source/handoff fields.

## Contents

| File | Role |
| --- | --- |
| `lfm25-350m-base-Q4_K_M.gguf` | LFM2.5-350M base model. |
| `telco-shared-clf-v1.gguf` | Shared understanding adapter used by the single-forward 9-head classifier runtime. |
| `telco-*_classifier_{weights,bias,meta}` | Classifier heads for support intent, routing lane, required tool, cloud requirements, issue complexity, escalation risk, PII risk, transcript quality, and slot completeness. |
| `telco_shared_clf_schema.json` | Label schema for the shared understanding heads. |
| `telco-dialogue-repair-v4.gguf` | Bounded verbalizer for selected repair/follow-up turns. |
| `telco-tool-selector-v3.gguf` | Tool-support adapter for ambiguous local action/tool paths. |
| `rag-units-v1.json` | Canonical RAG corpus used by the app's retriever and composer. |
| `page-link-table-v1.json` | Canonical in-app links used for source chips and navigation. |
| `model_manifest.json` | Machine-readable pack contract. |
| `checksums.sha256` | SHA-256 checksums for all shipped artifacts. |

No ColBERT model, Stage B answer LoRA, legacy chat-mode router, legacy topic
gate, legacy refusal-flags adapter, relational adapter, or raw source documents
are required by the current runtime.

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

- understanding forwards: 1 LFM2.5-350M pass
- first-turn answer generation calls: 0
- repair verbalizer calls: only selected repair/follow-up acts
- citation source: selected `RAGUnit.canonicalURL`

Unsupported or account-specific requests are handled by local policy as
cloud/system handoff or live-agent escalation; the pack does not include
customer account data.

## Expected Repository Files

The Hugging Face repository should contain only:

- `README.md`
- `checksums.sha256`
- `lfm25-350m-base-Q4_K_M.gguf`
- `telco-shared-clf-v1.gguf`
- `telco-dialogue-repair-v4.gguf`
- `telco-tool-selector-v3.gguf`
- `telco-*_classifier_{weights,bias,meta}`
- `telco_shared_clf_schema.json`
- `rag-units-v1.json`
- `page-link-table-v1.json`
- `model_manifest.json`

## Access

This pack is intended for private POC delivery. Grant access through the
Hugging Face organization or use a gated/private repo. Uploads should use a
write token or a fine-grained token scoped to this repository.
