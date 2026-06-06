# Models And Runtime Boundaries

Telco Triage is an on-device iOS reference app for home-internet support. The
normal customer/demo path uses Liquid models for **closed-set understanding** and
uses deterministic Swift code for retrieval, policy, citations, and answer
composition.

The current distinction is important:

- **One model forward:** LFM2.5-350M + `telco-shared-clf-v1` + nine classifier
  heads.
- **Zero first-turn answer generation calls:** no model writes the grounded
  procedural answer; V4 may verbalize selected repair/follow-up wording.
- **Deterministic answer plane:** BM25 retrieval selects a canonical support unit;
  the composer renders only approved text fields and `vzhome://` links.

## Active Customer/Demo Path

```text
User turn
  -> ConversationState
  -> LFM2.5-350M + telco-shared-clf-v1
       + 9 classifier heads
  -> BM25HierarchyRetriever over rag-units-v1.json
  -> deterministic route policy
       heads inform; thresholds + evidence + ToolRegistry decide
  -> DeterministicAnswerComposer
  -> answer bubble + source chip + open button / confirmation copy
```

## Packaged Active Artifacts

| Artifact | Role in current runtime |
| --- | --- |
| `lfm25-350m-base-Q4_K_M.gguf` | Base LFM2.5-350M loaded for the shared understanding pass. |
| `telco-shared-clf-v1.gguf` | Shared LoRA/adaptor applied to the base for telco support understanding. |
| `telco-support-intent_classifier_*` | Support intent, including troubleshooting, outage, billing/account, and agent handoff. |
| `telco-issue-complexity_classifier_*` | Simple/guided/backend-required/human-required signal. |
| `telco-routing-lane_classifier_*` | Local answer, local tool, cloud assist, human escalation, or blocked lane. |
| `telco-cloud-requirements_classifier_*` | Multi-label live-system requirements such as account state, billing record, network status, auth. |
| `telco-required-tool_classifier_*` | Closed-set required tool signal: restart gateway, diagnostics, speed test, technician, no tool, cloud only. |
| `telco-customer-escalation-risk_classifier_*` | Low/frustrated/churn/complaint/urgent escalation signal. |
| `telco-pii-risk_classifier_*` | Safe/account/contact/payment-identity PII signal. |
| `telco-transcript-quality_classifier_*` | Clean/noisy/partial/ASR-uncertain transcript signal. |
| `telco-slot-completeness_classifier_*` | Missing-slot signal only. It does not extract slot values. |
| `telco_shared_clf_schema.json` | Schema contract for the nine-head understanding layer. |
| `rag-units-v1.json` | Canonical support corpus for BM25 retrieval and deterministic composition. |
| `page-link-table-v1.json` | Canonical app destinations for source chips and open buttons. |
| `telco-tool-selector-v3.gguf` | Action executor fallback for ambiguous tool and argument selection. Pure grounded Q&A does not invoke it; supported action flows may. |
| `telco-dialogue-repair-v4.gguf` | Bounded multi-turn repair verbalizer. It may rewrite customer-facing response text but must echo Swift-owned route, source ids, handoff, and confirmation policy. |

Each `*_classifier_*` head is a `{weights.bin, bias.bin, meta.json}` triplet.

## What The Nine Heads Do

The shared classifier is the semantic control plane:

| Head | Product question answered |
| --- | --- |
| support intent | What broad support situation is this turn about? |
| issue complexity | Can this be answered locally, or does it require backend/human help? |
| routing lane | Should policy prefer local answer, local tool, cloud assist, human escalation, or block? |
| cloud requirements | Which live systems would be needed if local data is insufficient? |
| required tool | Which supported local tool, if any, is needed? |
| escalation risk | Is this low risk, frustration, complaint, churn, or urgent? |
| PII risk | Does the user include sensitive account/contact/payment identity data? |
| transcript quality | Is the turn clean enough, or should the app ask for clarification? |
| slot completeness | Which argument categories are missing? Values remain TBD. |

The heads do **not** directly execute actions and do **not** write answers. The
deterministic route policy gates hard decisions with confidence thresholds and
requires evidence from the selected RAG unit plus `ToolRegistry` before an
action confirmation can appear.

## Slot Extraction Status

Slot value extraction is intentionally not claimed by the current pack.

- Current: `telco-slot-completeness` says which slot categories are missing.
- Current: simple existing extractors and local app state supply values where
  available.
- TBD: a token/span head or another explicit extractor for `{device, location,
  action, time, account reference}` values.

## Inactive / Evaluation Artifacts

Some artifacts may still be bundled or present in old branches for comparison,
degraded builds, or offline evaluation. They are **not invoked** by the normal
customer/demo Q&A path:

| Artifact | Current status |
| --- | --- |
| `chat-mode-router-v2.gguf` | Legacy generative router; bypassed by composer path. |
| `vz-topic-gate-clf-v1.lora.gguf` | Legacy Stage A topic gate; bypassed by composer path. |
| `vz-refusal-flags-clf-v1.lora.gguf` | Legacy Stage A refusal flags; bypassed by composer path. |
| `telco-relational-v1.gguf` | Legacy pairwise relational pass; bypassed by composer path. |
| Stage B answer generator | Not bundled for normal grounded answers; rejected by answer-layer eval. |
| ColBERT | Not adopted; lexical BM25 hierarchy plus aliases cleared the current corpus gates. |

## Runtime Cost Contract

| Component | Cost profile |
| --- | --- |
| Shared understanding | One LFM2.5-350M classifier forward pass plus small head projections. |
| Retrieval | Pure Swift BM25 over 49 canonical RAG units. |
| Route policy | Pure Swift, confidence-gated. |
| Composer | Pure Swift, deterministic. |
| Dialogue repair verbalizer | Optional V4 LoRA call for repair/follow-up wording only. Swift owns route, citations, handoff, and confirmation. |
| Tool execution | Deterministic fast path first; `telco-tool-selector-v3` may run for ambiguous tool/argument selection. Side effects still require explicit confirmation or registered no-confirm policy. |

Normal customer/demo Q&A:

- first-turn procedural answer generation calls: **0**
- multi-turn repair verbalizer: **`telco-dialogue-repair-v4` when a repair/follow-up act is detected**
- online classifier adapter: **`telco-shared-clf-v1`**
- online classifier heads: **9**
- action selector: **`telco-tool-selector-v3` only for ambiguous action/argument selection**
- citation source: selected `RAGUnit.canonicalURL`

## References

- [README](README.md)
- [Current architecture](../../../docs/VERIZON-POC-ARCHITECTURE.md)
- [Runtime diagram](../../../docs/diagrams/liquid-telco-composer-runtime-architecture.md)
- [ADR-026](../../../docs/architecture-decisions/ADR-026-telco-shared-understanding-composer-runtime.md)
