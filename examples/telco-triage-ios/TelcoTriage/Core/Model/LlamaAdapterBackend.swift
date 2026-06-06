import Foundation

/// Concrete `AdapterInferenceBackend` implementation wrapping a live
/// shared `LlamaBackend`.
///
/// One shared `LlamaBackend` loads the LFM2.5-350M base GGUF at
/// launch; this wrapper routes each call to either an adapter-applied
/// generation (for the intent classifier and tool selector) or a
/// base-only generation (for the chat provider, which uses the base
/// model directly for grounded QA / tool summaries / personalized
/// responses).
///
/// Convention: passing `adapterPath == ""` means "run the base model
/// without a LoRA adapter". The bridge calls `removeAdapter()` before
/// generation to ensure any previously-applied adapter is detached.
public struct LlamaAdapterBackend: AdapterInferenceBackend {
    public let backend: LlamaBackend

    public init(backend: LlamaBackend) {
        self.backend = backend
    }

    public func generate(
        messages: [AdapterChatMessage],
        adapterPath: String,
        maxTokens: Int,
        stopSequences: [String]
    ) async throws -> String {
        if adapterPath.isEmpty {
            await backend.removeAdapter()
        } else {
            try await backend.setAdapter(path: adapterPath, scale: 1.0)
        }
        // Translate POC-facing chat messages to the LFMEngine type and
        // route through the chat-template path. This is the ONLY correct
        // entrypoint for LoRA adapters trained via leap-finetune, which
        // applies the model's chat template at training time.
        let engineMessages = messages.map { m in
            LlamaChatMessage(role: m.role.rawValue, content: m.content)
        }
        let (text, _, _) = try await backend.generate(
            messages: engineMessages,
            maxTokens: maxTokens,
            temperature: 0,
            stopSequences: stopSequences,
            clearCache: true,
            outputMode: .text
        )
        return text
    }

    public func generate(
        prompt: String,
        adapterPath: String,
        maxTokens: Int,
        stopSequences: [String]
    ) async throws -> String {
        if adapterPath.isEmpty {
            await backend.removeAdapter()
        } else {
            try await backend.setAdapter(path: adapterPath, scale: 1.0)
        }
        let (text, _, _) = try await backend.generate(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: 0,
            stopSequences: stopSequences,
            clearCache: true,
            outputMode: .text
        )
        return text
    }
}

/// Resource paths for the GGUFs bundled under `Resources/Models/`.
/// Wrapped in a namespace so callers never typo the file name + extension.
///
/// Architecture (2026-04-20): ChatModeRouter is the first hop. It routes to
/// either KBExtractor (kb_question) or ToolSelector (tool_action). The old
/// IntentRouter is removed — ChatModeRouter replaced its gating role.
public enum TelcoModelBundle {
    // CRITICAL: Must be the BASE model, NOT DPO/instruct. LoRA adapters
    // are trained on LFM2.5-350M-Base — applying them to DPO weights causes
    // hallucinated outputs (67% → 84% accuracy fix, 2026-04-20 session).
    //
    // PRECISION: Q4_K_M, the original trained-against quant tier. The Phase δ
    // sprint (2026-05-27) briefly swapped to Q8_0 to recover the
    // turn_relationship LoRA's class-collapse on Q4 (F1 0.92 vs 0.19), but
    // the +140 MB bundle cost AND the destabilizing effect on other Q4-trained
    // adapters (chat-mode-router YAML format drift) didn't justify it for a
    // POC. Reverted to Q4_K_M same day along with dropping ColBERT entirely
    // for an aggressive memory diet (-359 MB Resources). The relational LoRA
    // ships but is now non-functional — it class-collapses on Q4. Multi-turn
    // anaphoric routing reverts to "every turn is independent". Acceptable
    // trade for a 477 MB Resources bundle. See ADR-024 Phase δ postmortem
    // + ADR-025 for the proper-fix paths (LoftQ retrain, LoRA-adapter ColBERT).
    public static let baseModelName = "lfm25-350m-base-Q4_K_M"
    // v3: retrained on LFM2.5-350M-Base (v2 was on DPO — see §0 of
    // docs/FINE-TUNE-KB-EXTRACTOR.md for why all adapters must use Base).
    public static let toolAdapterName = "telco-tool-selector-v3"
    // v2: retrained with augmented data targeting the original 14 sample Q&A
    // topics (docs/FINE-TUNE-CHATMODE-ROUTER.md §11 — 2026-04-22 session).
    // Fixes possessive-field lookups ("what is my ipv4", "what firmware am I
    // on") that v1 misrouted to personal_summary. Ship-gate eval dropped
    // broken-rate from 47% → 13% on the telco KB sample harness.
    public static let chatModeRouterAdapterName = "chat-mode-router-v2"
    public static let kbExtractorAdapterName = "kb-extractor-v1"

    // Classification LoRA adapters — paired with classifier head binaries.
    // These adapters specialize the backbone's hidden states for each
    // classification task. The classifier heads were trained WITH these
    // adapters applied; without them, accuracy drops ~30-70pp.
    public static let chatModeClfAdapterName = "chat-mode-clf-v1"
    public static let kbExtractClfAdapterName = "kb-extract-clf-v1"
    public static let toolSelectorClfAdapterName = "tool-selector-clf-v1"
    public static let sharedClfAdapterName = "telco-shared-clf-v1"

    // Verizon RAG step-format generator (ADR-021 §5.2).
    // Originally shipped as a merged Q4_K_M model — kept temporarily as
    // `verizonStageBMergedName` for fallback. The new LoRA-adapter flavor
    // (`verizonStageBLoraName`) swaps onto the shared base GGUF the rest of
    // the app uses — one foundation model, adapters change per lane.
    //
    // Probe-validated (merged Q4): 100% format compliance, 97.8% valid
    // deep-link on 89 RAG-eligible queries (ADR-021 §6.5 + bf16/Q4 deltas).
    // LoRA-adapter flavor re-tests pending — should match within rounding.
    public static let verizonStageBMergedName = "vz-stage-b-v1.Q4_K_M"
    public static let verizonStageBLoraName = "vz-stage-b-v1.lora.f16"

    // Dialogue-repair verbalizer v4.
    //
    // This LoRA is not a router, retriever, tool selector, or citation
    // authority. It is a response-only verbalizer trained to make
    // multi-turn repair turns sound natural while echoing the route,
    // evidence ids, handoff, and confirmation policy supplied by Swift.
    public static let dialogueRepairV4AdapterName = "telco-dialogue-repair-v4"

    // Verizon Stage A: probe-validated topic_gate + refusal_flags heads.
    // Each is a classifier head binary (weights/bias/meta) trained on top
    // of its own r=16 LoRA backbone. Schema lives in the meta JSON.
    //
    // Eval (v3 corpus, probe-labeled holdout):
    //   topic_gate     (3-class):        99.2% acc, 98.4% macro F1
    //   refusal_flags  (3-flag sigmoid): 96.9% acc, 97.6% macro F1
    //
    // The LoRA backbones are private to each head (not the shared
    // telco-shared-clf-v1) — see ADR-021 §6.5 follow-ups for the
    // shared-backbone v2 retrain.
    public static let verizonTopicGateAdapterName = "vz-topic-gate-clf-v1.lora"
    public static let verizonRefusalFlagsAdapterName = "vz-refusal-flags-clf-v1.lora"
    public static let verizonTopicGateHeadTask = "vz-topic-gate"
    public static let verizonRefusalFlagsHeadTask = "vz-refusal-flags"

    // ADR-022 v2 understanding layer.
    //
    // Single shared LoRA adapter ("telco-shared-clf-v2") + 5 classifier
    // heads — one forward pass per query, ~150 ms total. Replaces the
    // PR #30 path that costs 2 forward passes (topic_gate + refusal_flags
    // each on a private adapter) + 1 generative chat_mode router call.
    //
    // Each head's task name matches the wire filename prefix used by
    // `classifierHeadPaths(task:in:)`. Schemas live in
    // `data/finetune/clf/vz_{task}_label_schema.json`.
    //
    // Trained-when-ready inventory (only chatMode/topicGate/refusalFlags
    // were present in PR #30; emotionalState + slotCompleteness ship with
    // the Phase 1/2 retrain per ADR-022 §6).
    public static let understandingV2AdapterName = "telco-shared-clf-v2"
    public static let understandingV2ChatModeHeadTask = "vz-chat-mode-v2"
    public static let understandingV2TopicGateHeadTask = "vz-topic-gate-v2"
    public static let understandingV2RefusalFlagsHeadTask = "vz-refusal-flags-v2"
    public static let understandingV2EmotionalStateHeadTask = "vz-emotional-state-v2"
    public static let understandingV2SlotCompletenessHeadTask = "vz-slot-completeness-v2"

    // ADR-024 Phase β/γ/δ — pairwise relational heads. The adapter +
    // 3 heads are produced by `scripts/vz/relational/export_heads.py`
    // and ship into iOS Resources/ as a single bundle.
    //
    // **Bundle status**: NOT YET BUNDLED. The relational artifacts ship
    // after the H100 training run (Phase γ.0 — config_telco_relational_v1.py
    // on .99). Until then, all `relationalV1*Path` accessors return nil,
    // `ChatTemplateRelationalStrategy.bundled(...)` returns nil, and the
    // router falls back to `UnavailableRelationalStrategy` which returns
    // `.none` outcomes — the graceful degraded path.
    public static let relationalV1AdapterName = "telco-relational-v1"
    public static let relationalV1TurnRelHeadTask = "telco-relational-turn-rel"
    public static let relationalV1SlotAlignHeadTask = "telco-relational-slot-align"
    public static let relationalV1StanceChangeHeadTask = "telco-relational-stance-change"

    public static let ext = "gguf"

    public static func basePath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: baseModelName, ofType: ext)
    }

    public static func toolAdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: toolAdapterName, ofType: ext)
    }

    public static func chatModeRouterAdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: chatModeRouterAdapterName, ofType: ext)
    }

    public static func kbExtractorAdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: kbExtractorAdapterName, ofType: ext)
    }

    // Classification adapter paths — paired with classifier head binaries

    public static func chatModeClfAdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: chatModeClfAdapterName, ofType: ext)
    }

    public static func kbExtractClfAdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: kbExtractClfAdapterName, ofType: ext)
    }

    public static func toolSelectorClfAdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: toolSelectorClfAdapterName, ofType: ext)
    }

    public static func sharedClfAdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: sharedClfAdapterName, ofType: ext)
    }

    // Backwards-compatible alias for the merged Stage B GGUF.
    // VerizonRAGTestView still loads this via loadModel() to isolate the
    // generator from the chat backend; the merged GGUF stays bundled for
    // that probe view. Production chat now uses verizonStageBLoraPath via
    // adapter swapping on the shared base.
    public static let verizonStageBGeneratorName = verizonStageBMergedName

    /// Path to the Verizon Stage B MERGED GGUF (350M + step-format LoRA
    /// fused, Q4_K_M, ~219 MB). Self-contained — loaded via `loadModel()`
    /// directly. Kept for the engineering probe view (VerizonRAGTestView)
    /// where isolating Stage B from the chat backend lets us measure its
    /// latency without other state in the way. Returns nil when not
    /// bundled (engineering builds / CI clones without the artifact).
    public static func verizonStageBGeneratorPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: verizonStageBMergedName, ofType: ext)
    }

    /// Path to the Verizon Stage B LoRA-adapter GGUF (~23 MB, r=32).
    /// Applied to the shared base via `setAdapter(path:scale:)` on the
    /// `.ragStepByStep` lane. One foundation model, adapters change per
    /// lane — the single-backbone architecture the merged chat dispatcher
    /// runs on. Returns nil when not bundled.
    public static func verizonStageBLoraPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: verizonStageBLoraName, ofType: ext)
    }

    public static func dialogueRepairV4AdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: dialogueRepairV4AdapterName, ofType: ext)
    }

    // MARK: - Verizon Stage A (topic_gate + refusal_flags)

    /// Path to the Verizon topic-gate Stage A classification LoRA
    /// (~11 MB, r=16). Must be applied via `setAdapter` before the
    /// mean-pool forward pass that feeds the topic-gate classifier head
    /// — the head was trained with this adapter active.
    public static func verizonTopicGateAdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: verizonTopicGateAdapterName, ofType: ext)
    }

    /// Path to the Verizon refusal-flags Stage A classification LoRA
    /// (~11 MB, r=16). Same contract as `verizonTopicGateAdapterPath`.
    public static func verizonRefusalFlagsAdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: verizonRefusalFlagsAdapterName, ofType: ext)
    }

    /// Resolved paths for the Verizon topic-gate head triplet
    /// (`vz-topic-gate_classifier_{weights,bias,meta}`).
    public static func verizonTopicGateHeadPaths(in bundle: Bundle = .main) -> ClassifierHeadPaths? {
        classifierHeadPaths(task: verizonTopicGateHeadTask, in: bundle)
    }

    /// Resolved paths for the Verizon refusal-flags head triplet
    /// (`vz-refusal-flags_classifier_{weights,bias,meta}`).
    public static func verizonRefusalFlagsHeadPaths(in bundle: Bundle = .main) -> ClassifierHeadPaths? {
        classifierHeadPaths(task: verizonRefusalFlagsHeadTask, in: bundle)
    }

    /// True when the Verizon Stage A stack is fully bundled — both LoRA
    /// adapters and both classifier-head triplets present. Drives whether
    /// VerizonChatDispatcher can run the probe-validated router; falls
    /// back to ungrounded Stage B generation when false.
    public static func verizonStageABundled(in bundle: Bundle = .main) -> Bool {
        return verizonTopicGateAdapterPath(in: bundle) != nil
            && verizonRefusalFlagsAdapterPath(in: bundle) != nil
            && verizonTopicGateHeadPaths(in: bundle) != nil
            && verizonRefusalFlagsHeadPaths(in: bundle) != nil
    }

    /// True when the router artifacts required by the composer RAG path
    /// are bundled. Stage B is optional now: the normal answer layer is
    /// deterministic composer over `rag-units-v1.json`.
    public static func verizonRagStackBundled(in bundle: Bundle = .main) -> Bool {
        return verizonStageABundled(in: bundle)
    }

    // MARK: - ADR-022 v2 understanding layer

    /// Path to the shared LoRA adapter for the v2 understanding layer.
    /// Nil until the Phase 2 H100 retrain ships — that's the gate that
    /// flips the `QueryUnderstandingClassifier` strategy from
    /// `.composite` to `.shared`.
    public static func understandingV2AdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: understandingV2AdapterName, ofType: ext)
    }

    /// Bundled classifier head paths for v2, keyed by the head's task name.
    /// Defensive lookup — any missing head returns nil for that key and
    /// the classifier degrades that head to absent in `QueryUnderstanding`.
    public static func understandingV2HeadPaths(in bundle: Bundle = .main) -> [String: ClassifierHeadPaths] {
        var paths: [String: ClassifierHeadPaths] = [:]
        for task in [
            understandingV2ChatModeHeadTask,
            understandingV2TopicGateHeadTask,
            understandingV2RefusalFlagsHeadTask,
            understandingV2EmotionalStateHeadTask,
            understandingV2SlotCompletenessHeadTask,
        ] {
            if let p = classifierHeadPaths(task: task, in: bundle) {
                paths[task] = p
            }
        }
        return paths
    }

    /// True when at least one v2 head AND the shared LoRA are bundled —
    /// the minimum to run the `.shared` strategy. We don't require ALL
    /// five heads because the architecture supports partial rollout
    /// (chat_mode + topic_gate + refusal_flags can ship before
    /// emotional_state + slot_completeness without breaking routing).
    public static func understandingV2Bundled(in bundle: Bundle = .main) -> Bool {
        guard understandingV2AdapterPath(in: bundle) != nil else { return false }
        return !understandingV2HeadPaths(in: bundle).isEmpty
    }

    // MARK: - ADR-024 relational v1 (pairwise heads)

    /// Path to the relational v1 LoRA adapter (`telco-relational-v1.gguf`).
    /// Nil until Phase γ ships the trained artifact. Distinct from the
    /// understanding-v2 adapter — different LoRA, different training
    /// data, different head set.
    public static func relationalV1AdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: relationalV1AdapterName, ofType: ext)
    }

    /// Bundled relational head paths, keyed by head task name. Returns
    /// an empty dict when none of the 3 heads are bundled. Partial-
    /// bundle handling: `ChatTemplateRelationalStrategy.bundled` REQUIRES
    /// all 3 heads (stricter than understanding-v2 because the router
    /// fusion treats relational outcomes as a coherent triple — partial
    /// triples create asymmetric coverage in `decideMultiTurn`).
    public static func relationalV1HeadPaths(in bundle: Bundle = .main) -> [String: ClassifierHeadPaths] {
        var paths: [String: ClassifierHeadPaths] = [:]
        for task in [
            relationalV1TurnRelHeadTask,
            relationalV1SlotAlignHeadTask,
            relationalV1StanceChangeHeadTask,
        ] {
            if let p = classifierHeadPaths(task: task, in: bundle) {
                paths[task] = p
            }
        }
        return paths
    }

    /// True when the full relational stack (adapter + all 3 heads) is
    /// bundled. Drives `ChatTemplateRelationalStrategy.bundled(...)`
    /// returning non-nil. Until this is true, the router stays on the
    /// `UnavailableRelationalStrategy` graceful-nil path.
    public static func relationalV1Bundled(in bundle: Bundle = .main) -> Bool {
        guard relationalV1AdapterPath(in: bundle) != nil else { return false }
        let heads = relationalV1HeadPaths(in: bundle)
        return heads.count == 3
    }

    /// True when every GGUF is bundled. AppState fails fast when this
    /// returns false — see `AppState.buildLFMStack`.
    public static func isFullyBundled(in bundle: Bundle = .main) -> Bool {
        return basePath(in: bundle) != nil
            && toolAdapterPath(in: bundle) != nil
            && chatModeRouterAdapterPath(in: bundle) != nil
            && kbExtractorAdapterPath(in: bundle) != nil
    }

    /// True when classifier heads AND their paired classification adapters
    /// are all bundled. Both are required — heads without adapters produce
    /// ~30-70pp accuracy drops (Phase 7 eval, 2026-04-24).
    public static func classifierStackBundled(in bundle: Bundle = .main) -> Bool {
        return sharedClassifierStackBundled(in: bundle)
            || pairedClassifierStackBundled(in: bundle)
    }

    /// True when the shared-adapter multi-head classifier can run one
    /// forward pass for every telco sequence head.
    public static func sharedClassifierStackBundled(in bundle: Bundle = .main) -> Bool {
        return classifierHeadsBundled(in: bundle)
            && sharedClfAdapterPath(in: bundle) != nil
    }

    /// True when existing per-head adapters can run a mathematically
    /// correct transitional multi-head path.
    public static func pairedClassifierStackBundled(in bundle: Bundle = .main) -> Bool {
        return classifierHeadsBundled(in: bundle)
            && chatModeClfAdapterPath(in: bundle) != nil
            && kbExtractClfAdapterPath(in: bundle) != nil
            && toolSelectorClfAdapterPath(in: bundle) != nil
    }

    // MARK: - Classifier Head Binaries

    // Three classifier heads replace generative LoRA calls for
    // classification tasks. One backbone forward pass + a linear
    // head (<1ms) vs autoregressive decoding (~200ms).
    //
    // Files are named {task}_classifier_{weights,bias,meta}.{bin,json}
    // to avoid filename collisions in the flat app bundle.

    /// Resolved URLs for a single classifier head's three artifacts.
    public struct ClassifierHeadPaths {
        public let weightsURL: URL
        public let biasURL: URL
        public let metaURL: URL
    }

    /// Returns paths for a classifier head's three artifacts, or nil
    /// if any artifact is missing from the bundle.
    public static func classifierHeadPaths(
        task: String,
        in bundle: Bundle = .main
    ) -> ClassifierHeadPaths? {
        guard let w = bundle.url(forResource: "\(task)_classifier_weights", withExtension: "bin"),
              let b = bundle.url(forResource: "\(task)_classifier_bias", withExtension: "bin"),
              let m = bundle.url(forResource: "\(task)_classifier_meta", withExtension: "json")
        else { return nil }
        return ClassifierHeadPaths(weightsURL: w, biasURL: b, metaURL: m)
    }

    /// True when all three classifier head artifact sets are bundled.
    public static func classifierHeadsBundled(in bundle: Bundle = .main) -> Bool {
        return classifierHeadPaths(task: "chat-mode", in: bundle) != nil
            && classifierHeadPaths(task: "kb-extract", in: bundle) != nil
            && classifierHeadPaths(task: "tool-selector", in: bundle) != nil
    }

    // MARK: - ADR-015 telco multi-head classifier (Phase 2)

    /// Names of the 9 telco sequence heads from ADR-015. Each one is a
    /// `{name}_classifier_{weights,bias,meta}.{bin,json}` triplet.
    public static let adr015TelcoHeadNames: [String] = [
        "telco-support-intent",
        "telco-issue-complexity",
        "telco-routing-lane",
        "telco-cloud-requirements",
        "telco-required-tool",
        "telco-customer-escalation-risk",
        "telco-pii-risk",
        "telco-transcript-quality",
        "telco-slot-completeness",
    ]

    /// True when all 9 telco head artifact sets AND the shared classification
    /// LoRA are bundled. Drives the ADR-015 lane router on the iOS app.
    public static func adr015TelcoStackBundled(in bundle: Bundle = .main) -> Bool {
        guard sharedClfAdapterPath(in: bundle) != nil else { return false }
        for head in adr015TelcoHeadNames {
            if classifierHeadPaths(task: head, in: bundle) == nil {
                return false
            }
        }
        return true
    }
}
