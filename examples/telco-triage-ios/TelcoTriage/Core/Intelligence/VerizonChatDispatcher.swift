import Foundation
import os.log

/// Surface-area events the chat UI observes as the dispatcher runs.
///
/// Customer mode only acts on `.response` and `.failed`; engineering mode
/// renders every event as a row in the pipeline trace. This is the
/// same pattern the Banking POC `InferencePipeline` uses, just narrower
/// (one turn, no multi-step chain) — keeps the trace UI surface
/// consistent across both POCs for the inevitable future merge.
public enum VerizonDispatchEvent: Sendable, Equatable {
    case stageAStarted
    case stageAComplete(stageA: VerizonStageADecision)
    case laneSelected(VerizonLane)
    case retrievalStarted
    case retrievalComplete(result: ColBERTRetrievalResult)
    case stageBStarted
    case stageBComplete(StageBResponse)
    case faithfulnessChecked(score: FaithfulnessScore)
    case fallbackInvoked(reason: String)
    case response(VerizonDispatchResult)
    case failed(message: String)
}

/// Final response delivered to the chat view.
public struct VerizonDispatchResult: Sendable, Equatable {
    public enum Source: String, Sendable, Equatable {
        case composer  // Step 6 — deterministic composer on top of lexical retrieval
        case dialogueRepair  // V4 response-only verbalizer over composer evidence/policy
        case stageB    // Legacy Stage B path (disabled on v1 per Step 5 decision)
        case kbFallback
        case template
        case toolAction
        case personalSummary
    }

    /// User-facing response text. May contain Markdown links.
    public let text: String

    /// The lane the router picked. Surfaced for engineering trace +
    /// session telemetry.
    public let lane: VerizonLane

    /// Where the text came from (Stage B, deterministic KB fallback, a
    /// static template, etc.). Critical for the engineering-mode trace
    /// because Stage B vs fallback have very different latency profiles.
    public let source: Source

    /// vzhome:// deep link the user can tap, when the response carries
    /// one. Surfaced separately so the UI can render an actionable
    /// button below the message.
    public let deepLink: String?

    /// The RAG chunk Stage B was grounded against, when the response
    /// came from `.stageB` with a successful ColBERT retrieval.
    /// Drives the "Source: Network § Wi-Fi Management" citation chip
    /// per ADR §11.7. Nil when source is template, kbFallback, or
    /// when Stage B ran ungrounded (degraded mode).
    public let retrievedChunk: ColBERTChunk?

    /// The KB entry whose `.answer` was returned verbatim by the
    /// `kbFallback` source. Populated when ColBERT/Stage B failed and
    /// `handleUnknownFeature` matched a KeywordKBExtractor entry. The
    /// citation chip uses this when `retrievedChunk` is nil so the
    /// user sees "Read full article" regardless of which RAG path
    /// produced the text.
    public let kbEntry: KBEntry?

    /// Total wall-clock from query received to response ready. The
    /// engineering-mode latency badge reads this.
    public let totalMs: Double

    // -- Step 6 composer-path fields --

    /// Explicit timing breakdown for the deterministic composer path.
    /// These are nil for the legacy Stage B / KB fallback paths.
    public let retrievalMs: Double?
    public let routePolicyMs: Double?
    public let composerMs: Double?

    /// The composer route the dispatcher derived for this turn — when
    /// the composer was the answer source. Nil for legacy Stage B /
    /// kbFallback / template paths. Telemetry consumer.
    public let composerRoute: ComposerRoute?

    /// True iff the rendered text requires a real, executable tool
    /// confirmation (a registered `ToolIntent` exists for the unit's
    /// `linkID`). Surfaced so `ChatViewModel.runVerizonDispatch` can
    /// push a `PendingToolConfirmation` for the existing affirmative-
    /// recovery flow. NEVER true for view-only RAG answers (guardrail
    /// #3 in the Step 6 plan — no confirmation theatre).
    public let requiresConfirmation: Bool?

    /// Registered local tool behind this answer, when one exists.
    /// The UI/state layer uses this typed value to create pending
    /// confirmation state. It must never infer executable behavior
    /// from rendered text such as "Reply yes".
    public let executableToolIntent: ToolIntent?

    /// The canonical RAG unit the composer rendered from. Drives the
    /// citation chip on the composer path the same way `retrievedChunk`
    /// drove it on the legacy ColBERT path.
    public let citedRAGUnit: RAGUnit?

    public init(
        text: String,
        lane: VerizonLane,
        source: Source,
        deepLink: String? = nil,
        retrievedChunk: ColBERTChunk? = nil,
        kbEntry: KBEntry? = nil,
        totalMs: Double,
        retrievalMs: Double? = nil,
        routePolicyMs: Double? = nil,
        composerMs: Double? = nil,
        composerRoute: ComposerRoute? = nil,
        requiresConfirmation: Bool? = nil,
        executableToolIntent: ToolIntent? = nil,
        citedRAGUnit: RAGUnit? = nil
    ) {
        self.text = text
        self.lane = lane
        self.source = source
        self.deepLink = deepLink
        self.retrievedChunk = retrievedChunk
        self.kbEntry = kbEntry
        self.totalMs = totalMs
        self.retrievalMs = retrievalMs
        self.routePolicyMs = routePolicyMs
        self.composerMs = composerMs
        self.composerRoute = composerRoute
        self.requiresConfirmation = requiresConfirmation
        self.executableToolIntent = executableToolIntent
        self.citedRAGUnit = citedRAGUnit
    }
}

/// Spine of the merged chat flow.
///
/// Single entry point — `dispatch(query:)` returns an `AsyncStream` so
/// the UI can render progressive trace events as each stage completes.
/// Customer-mode views ignore intermediate events; engineering-mode
/// views render each as a trace row.
///
/// Lane handlers:
///   - `.liveAgentEscalation`     → static template + ETA
///   - `.navOnlyDeeplink`         → static template + Account deep link
///   - `.outOfScopeRefusal`       → Verizon canonical refusal
///   - `.greeting`                → friendly hello (no LLM)
///   - `.unknownFeature`          → KB fallback if it has a hit, else refusal
///   - `.clarification`           → ask user to pick (stub for now —
///                                  full multi-turn lands with Stage C)
///   - `.ragStepByStep`           → Stage B + LinkResolver, falls back to
///                                  KeywordKBExtractor on format break
///
/// Tool execution + personal summary are surfaced as `.toolAction` /
/// `.personalSummary` lane responses with placeholder bodies — the
/// dispatcher signals the lane and lets the existing ChatViewModel
/// scaffolding (ToolExecutor, LFMChatProvider + CustomerContext)
/// fulfill the user-visible response. That integration lands in the
/// next phase (ChatView rewire).
public actor VerizonChatDispatcher {
    private let stageA: VerizonStageAClassifying?
    private let stageB: StageBGenerating?
    private let kbFallback: KBExtractor
    private let kb: [KBEntry]
    private let retriever: ColBERTRetriever?
    private let modelHost: SwappingModelHost?
    // Step 6 composer-path dependencies. Optional for backwards
    // compatibility with existing test fixtures, but when all four are
    // wired (the production path), the dispatcher uses the composer
    // as the answer layer and Stage B / ColBERT / KeywordKBExtractor
    // become unreachable on .ragStepByStep.
    private let composer: AnswerComposing?
    private let corpus: RAGUnitCorpus?
    private let lexicalRetriever: BM25HierarchyRetriever?
    private let toolRegistry: ToolRegistry?
    private let toolAliasMap: ToolAliasMap?
    private let dialogueRepairVerbalizer: DialogueRepairVerbalizing?
    private let logger = Logger(
        subsystem: "ai.liquid.demos.telcotriage",
        category: "VerizonDispatcher"
    )

    public init(
        stageA: VerizonStageAClassifying?,
        stageB: StageBGenerating?,
        kbFallback: KBExtractor,
        kb: [KBEntry],
        retriever: ColBERTRetriever? = nil,
        modelHost: SwappingModelHost? = nil,
        composer: AnswerComposing? = nil,
        corpus: RAGUnitCorpus? = nil,
        lexicalRetriever: BM25HierarchyRetriever? = nil,
        toolRegistry: ToolRegistry? = nil,
        toolAliasMap: ToolAliasMap? = nil,
        dialogueRepairVerbalizer: DialogueRepairVerbalizing? = nil
    ) {
        self.stageA = stageA
        self.stageB = stageB
        self.kbFallback = kbFallback
        self.kb = kb
        self.retriever = retriever
        self.modelHost = modelHost
        self.composer = composer
        self.corpus = corpus
        self.lexicalRetriever = lexicalRetriever
        self.toolRegistry = toolRegistry
        self.toolAliasMap = toolAliasMap
        self.dialogueRepairVerbalizer = dialogueRepairVerbalizer
    }

    /// True when all Step 6 composer-path dependencies are present.
    /// The dispatcher takes the composer path iff this is true.
    private var composerPathEnabled: Bool {
        composer != nil && corpus != nil && lexicalRetriever != nil && toolRegistry != nil
    }

    /// First-principles online path for the Step 4/5 architecture:
    /// explicit conversation state + BM25Hierarchy retrieval +
    /// ToolRegistry policy + deterministic composer. This path never
    /// invokes Stage A, chat-mode-router, refusal flags, relational LoRA,
    /// Stage B, ColBERT, or KeywordKBExtractor.
    public nonisolated func dispatchComposer(
        query: String,
        retrievalContext: RetrievalContext = .empty,
        telcoUnderstanding: TelcoSharedUnderstanding? = nil,
        dialogueState: DialogueRepairConversationState = .empty
    ) -> AsyncStream<VerizonDispatchEvent> {
        AsyncStream { continuation in
            Task {
                await self.runComposerOnly(
                    query: query,
                    retrievalContext: retrievalContext,
                    telcoUnderstanding: telcoUnderstanding,
                    dialogueState: dialogueState,
                    continuation: continuation
                )
            }
        }
    }

    public nonisolated func dispatch(
        query: String,
        retrievalContext: RetrievalContext = .empty
    ) -> AsyncStream<VerizonDispatchEvent> {
        AsyncStream { continuation in
            Task {
                await self.run(
                    query: query,
                    prebuiltStageA: nil,
                    prebuiltLane: nil,
                    retrievalContext: retrievalContext,
                    continuation: continuation
                )
            }
        }
    }

    /// ADR-022 §4.3 Layer 1 entry point — caller supplies a pre-built
    /// Stage A decision (from the v2 understanding pass) and the
    /// already-resolved `VerizonLane` (from `VerizonUnderstandingRouter.decide`).
    /// The dispatcher skips its own Stage A classifier call (avoiding
    /// the duplicate ~150 ms forward pass) and goes straight to lane
    /// handling. The progressive trace stream still emits
    /// `.stageAStarted` / `.stageAComplete` / `.laneSelected` so the
    /// existing trace UI doesn't break.
    ///
    /// ADR-024 follow-up 2026-05-27 — `retrievalContext` carries
    /// optional cross-turn signal (prior assistant text) to augment
    /// ColBERT retrieval for anaphoric / negative-continuation
    /// follow-ups. Default `.empty` for back-compat.
    public nonisolated func dispatch(
        query: String,
        prebuiltStageA: VerizonStageADecision,
        prebuiltLane: VerizonLane,
        retrievalContext: RetrievalContext = .empty
    ) -> AsyncStream<VerizonDispatchEvent> {
        AsyncStream { continuation in
            Task {
                await self.run(
                    query: query,
                    prebuiltStageA: prebuiltStageA,
                    prebuiltLane: prebuiltLane,
                    retrievalContext: retrievalContext,
                    continuation: continuation
                )
            }
        }
    }

    private func run(
        query: String,
        prebuiltStageA: VerizonStageADecision?,
        prebuiltLane: VerizonLane?,
        retrievalContext: RetrievalContext = .empty,
        continuation: AsyncStream<VerizonDispatchEvent>.Continuation
    ) async {
        let t0 = CFAbsoluteTimeGetCurrent()

        continuation.yield(.stageAStarted)
        let stageADecision: VerizonStageADecision
        if let prebuiltStageA {
            // Caller already paid for Stage A — re-yield it through the
            // event stream so the trace UI renders the same chain.
            stageADecision = prebuiltStageA
        } else {
            guard let stageA else {
                continuation.yield(.failed(message: "Stage A classifier unavailable on legacy dispatch path"))
                continuation.finish()
                return
            }
            do {
                stageADecision = try await stageA.classify(query: query)
            } catch {
                logger.error("Stage A failed: \(error.localizedDescription, privacy: .public)")
                continuation.yield(.failed(message: "Stage A classifier failed: \(error.localizedDescription)"))
                continuation.finish()
                return
            }
        }
        continuation.yield(.stageAComplete(stageA: stageADecision))

        // Trust the caller's lane decision when supplied; the v2 router
        // includes signals (chat_mode, emotional_state) that this
        // dispatcher's local `VerizonRagRouter` doesn't know about, so
        // re-routing here would discard them.
        let lane = prebuiltLane ?? VerizonRagRouter.route(stageA: stageADecision)
        continuation.yield(.laneSelected(lane))

        let result: VerizonDispatchResult
        switch lane {
        case .liveAgentEscalation:
            result = handleLiveAgent(query: query, lane: lane, t0: t0)
        case .navOnlyDeeplink:
            result = handleNavOnly(query: query, lane: lane, t0: t0)
        case .oosRefusal:
            result = handleOutOfScope(query: query, lane: lane, t0: t0)
        case .greeting:
            result = handleGreeting(query: query, lane: lane, t0: t0)
        case .unknownFeature:
            result = await handleUnknownFeature(query: query, lane: lane, t0: t0)
        case .clarification:
            result = handleClarification(query: query, lane: lane, t0: t0)
        case .ragStepByStep:
            result = await handleRagStepByStep(
                query: query,
                stageADecision: stageADecision,
                lane: lane,
                retrievalContext: retrievalContext,
                continuation: continuation,
                t0: t0
            )
        }

        continuation.yield(.response(result))
        continuation.finish()
    }

    private func runComposerOnly(
        query: String,
        retrievalContext: RetrievalContext,
        telcoUnderstanding: TelcoSharedUnderstanding?,
        dialogueState: DialogueRepairConversationState,
        continuation: AsyncStream<VerizonDispatchEvent>.Continuation
    ) async {
        let t0 = CFAbsoluteTimeGetCurrent()

        if let policy = deriveComposerPolicyRoute(query: query, understanding: telcoUnderstanding) {
            let result = composeTemplateResponse(
                route: policy.route,
                query: query,
                lane: policy.lane,
                legacyText: policy.legacyText,
                legacyDeepLink: policy.legacyDeepLink,
                t0: t0
            )
            continuation.yield(.laneSelected(policy.lane))
            continuation.yield(.response(result))
            continuation.finish()
            return
        }

        let result = await handleRagStepByStepComposer(
            query: query,
            lane: .ragStepByStep,
            retrievalContext: retrievalContext,
            telcoUnderstanding: telcoUnderstanding,
            dialogueState: dialogueState,
            continuation: continuation,
            t0: t0
        )
        continuation.yield(.laneSelected(result.lane))
        continuation.yield(.response(result))
        continuation.finish()
    }

    private struct ComposerPolicyRoute: Sendable {
        let route: ComposerRoute
        let lane: VerizonLane
        let legacyText: String
        let legacyDeepLink: String?
    }

    /// Closed-set policy lanes that do not need retrieval evidence.
    /// This is deliberately small: live-agent handoff and greetings are
    /// intent-exact control states, not knowledge-base questions. All
    /// support Q&A, action/question separation, and no-answer handling
    /// stays downstream in retrieval evidence + `deriveComposerRoute`.
    private func deriveComposerPolicyRoute(
        query: String,
        understanding: TelcoSharedUnderstanding?
    ) -> ComposerPolicyRoute? {
        let normalized = query
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()

        if understanding?.isBlocked == true {
            return ComposerPolicyRoute(
                route: .outOfScope,
                lane: .oosRefusal,
                legacyText: "I'm here to help with home internet support. I can't process sensitive identity or payment details in this demo.",
                legacyDeepLink: nil
            )
        }

        if understanding?.needsClarification == true {
            return ComposerPolicyRoute(
                route: .clarify,
                lane: .clarification,
                legacyText: "Could you clarify what you're asking about?",
                legacyDeepLink: nil
            )
        }

        if understanding?.requiresHumanHandoff == true {
            return ComposerPolicyRoute(
                route: .liveAgent,
                lane: .liveAgentEscalation,
                legacyText: "Connecting you to a support agent. Estimated wait: a few minutes.",
                legacyDeepLink: VerizonLinkResolver.levelOneFallback(for: .liveAgent)
            )
        }

        if understanding?.requiresCloudAssist == true {
            return ComposerPolicyRoute(
                route: .accountNav,
                lane: .navOnlyDeeplink,
                legacyText: "This needs live account or network data. Open the app account area to continue.",
                legacyDeepLink: VerizonLinkResolver.levelOneFallback(for: .accountOOS)
            )
        }

        if ConversationStateRecorder.isLiveAgentRequest(query) {
            return ComposerPolicyRoute(
                route: .liveAgent,
                lane: .liveAgentEscalation,
                legacyText: "Connecting you to a Verizon support agent. Estimated wait: a few minutes.",
                legacyDeepLink: VerizonLinkResolver.levelOneFallback(for: .liveAgent)
            )
        }

        if Self.greetingQueries.contains(normalized) {
            return ComposerPolicyRoute(
                route: .greeting,
                lane: .greeting,
                legacyText: "Hi — I'm your home internet assistant. Ask me about your Wi-Fi, router, devices, parental controls, or equipment.",
                legacyDeepLink: nil
            )
        }

        return nil
    }

    private static let greetingQueries: Set<String> = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    ]

    // MARK: - Lane handlers
    //
    // Step 6.5 lane migration: when the composer is wired (production
    // path), every lane handler delegates to the composer so all
    // user-facing text comes from a single source of truth. The
    // composer's per-route templates are documented at
    // `scripts/vz/answer_composer.py::ROUTE_TEMPLATES` and mirrored in
    // `Core/Composer/AnswerComposer.swift`.
    //
    // The legacy inline templates remain as the dev / engineering
    // fallback when `composerPathEnabled == false` — same words as
    // before so legacy behaviour is unchanged for that path.

    private func handleLiveAgent(query: String, lane: VerizonLane, t0: CFAbsoluteTime) -> VerizonDispatchResult {
        composeTemplateResponse(
            route: .liveAgent,
            query: query,
            lane: lane,
            legacyText: "Connecting you to a Verizon support agent. Estimated wait: a few minutes.",
            legacyDeepLink: VerizonLinkResolver.levelOneFallback(for: .liveAgent),
            t0: t0
        )
    }

    private func handleNavOnly(query: String, lane: VerizonLane, t0: CFAbsoluteTime) -> VerizonDispatchResult {
        // Account / billing pivots — no LLM near financial data.
        // Composer renders the canonical My Verizon URL; legacy
        // fallback uses the in-app Account tab deep link.
        let legacyLink = VerizonLinkResolver.levelOneFallback(for: .accountOOS)
        let legacyText = "I can't help with that here. Go to [Account](\(legacyLink)) > Bills to manage your account."
        return composeTemplateResponse(
            route: .accountNav,
            query: query,
            lane: lane,
            legacyText: legacyText,
            legacyDeepLink: legacyLink,
            t0: t0
        )
    }

    private func handleOutOfScope(query: String, lane: VerizonLane, t0: CFAbsoluteTime) -> VerizonDispatchResult {
        composeTemplateResponse(
            route: .outOfScope,
            query: query,
            lane: lane,
            legacyText: "I'm here to help with Verizon Home Internet — router, network, devices, parental controls, equipment, or Digital Secure Home. Please try asking a different question.",
            legacyDeepLink: nil,
            t0: t0
        )
    }

    private func handleGreeting(query: String, lane: VerizonLane, t0: CFAbsoluteTime) -> VerizonDispatchResult {
        composeTemplateResponse(
            route: .greeting,
            query: query,
            lane: lane,
            legacyText: "Hi — I'm your Verizon Home Internet assistant. Ask me about your Wi-Fi, router, devices, parental controls, or equipment.",
            legacyDeepLink: nil,
            t0: t0
        )
    }

    private func handleClarification(query: String, lane: VerizonLane, t0: CFAbsoluteTime) -> VerizonDispatchResult {
        // Composer's `.clarify` template fills the "are you asking
        // about X?" slot from the user's query.
        composeTemplateResponse(
            route: .clarify,
            query: query,
            lane: lane,
            legacyText: "Could you tell me more about what you're trying to do? For example: change your Wi-Fi password, restart your router, or check connected devices.",
            legacyDeepLink: nil,
            t0: t0
        )
    }

    private func handleUnknownFeature(
        query: String,
        lane: VerizonLane,
        t0: CFAbsoluteTime
    ) async -> VerizonDispatchResult {
        // **Guardrail #2**: on the composer path, `KeywordKBExtractor`
        // is NOT consulted. We render the canonical no_rag_answer
        // template instead — composer's safe fallback URL only, no
        // alternative-answer corpus.
        if composerPathEnabled {
            return composeTemplateResponse(
                route: .noRagAnswer,
                query: query,
                lane: lane,
                legacyText: "It looks like I don't have information about that. Would you like me to connect you with a Verizon support agent?",
                legacyDeepLink: nil,
                t0: t0
            )
        }
        // ---- Legacy dev/engineering path: KB fallback ----
        let citation = await kbFallback.extract(query: query, kb: kb)
        if citation.isMatch, let entry = kb.first(where: { $0.id == citation.entryId }) {
            return VerizonDispatchResult(
                text: entry.answer,
                lane: lane,
                source: .kbFallback,
                deepLink: entry.deepLinks.first?.url,
                kbEntry: entry,
                totalMs: elapsed(t0)
            )
        }
        return VerizonDispatchResult(
            text: "It looks like I don't have information about that. Would you like me to connect you with a Verizon support agent?",
            lane: lane,
            source: .template,
            deepLink: nil,
            totalMs: elapsed(t0)
        )
    }

    /// Single source of truth for non-RAG template responses. When the
    /// composer is wired, delegates to it so route-text-link rendering
    /// stays consistent with the composer's contract (and with the
    /// Python `answer_composer.py` reference). When not wired, returns
    /// the legacy inline template as before.
    private func composeTemplateResponse(
        route: ComposerRoute,
        query: String,
        lane: VerizonLane,
        legacyText: String,
        legacyDeepLink: String?,
        t0: CFAbsoluteTime
    ) -> VerizonDispatchResult {
        if composerPathEnabled, let composer = composer {
            let composerStart = CFAbsoluteTimeGetCurrent()
            let answer = composer.compose(
                query: query,
                route: route,
                evidence: nil,
                requiresConfirmation: nil,
                history: [],
                expectedPolicyLinkID: nil
            )
            let composerMs = elapsed(composerStart)
            return VerizonDispatchResult(
                text: answer.text,
                lane: lane,
                source: .composer,
                deepLink: answer.renderedLinks.first,
                totalMs: elapsed(t0),
                retrievalMs: 0,
                routePolicyMs: 0,
                composerMs: composerMs,
                composerRoute: route,
                requiresConfirmation: false,
                executableToolIntent: nil,
                citedRAGUnit: nil
            )
        }
        return VerizonDispatchResult(
            text: legacyText,
            lane: lane,
            source: .template,
            deepLink: legacyDeepLink,
            totalMs: elapsed(t0)
        )
    }

    /// The .ragStepByStep handler.
    ///
    /// **Step 6 composer path** (production, taken when `composerPathEnabled`):
    ///   1. Lexical retrieve via `BM25HierarchyRetriever` (post-4c
    ///      alias-improved baseline, the retrieval source of truth
    ///      per the Step 5 decision record).
    ///   2. Derive composer route via the `ToolRegistry` gate
    ///      (guardrail #3 — confirmation iff a real executable
    ///      `ToolIntent` is registered for the unit's `linkID`).
    ///   3. `AnswerComposer.compose(...)` renders the user-facing text.
    ///      Cannot hallucinate a link by construction.
    ///   4. Emit. No faithfulness check needed — the composer can't
    ///      paraphrase off the evidence.
    ///
    /// **Legacy Stage B path** (taken only when composer dependencies
    /// are NOT wired — preserved for dev / engineering builds, not
    /// reachable in normal demo). The original 5-stage Ottoguard
    /// shape: ColBERT retrieve → Stage B generate → faithfulness gate →
    /// KeywordKBExtractor fallback on any failure.
    private func handleRagStepByStep(
        query: String,
        stageADecision: VerizonStageADecision,
        lane: VerizonLane,
        retrievalContext: RetrievalContext,
        continuation: AsyncStream<VerizonDispatchEvent>.Continuation,
        t0: CFAbsoluteTime
    ) async -> VerizonDispatchResult {
        if composerPathEnabled {
            return await handleRagStepByStepComposer(
                query: query,
                lane: lane,
                retrievalContext: retrievalContext,
                dialogueState: .empty,
                continuation: continuation,
                t0: t0
            )
        }
        // ---- Legacy Stage B path (dev / engineering only) ----
        guard let stageB = stageB else {
            continuation.yield(.fallbackInvoked(reason: "Stage B adapter not bundled"))
            return await handleUnknownFeature(query: query, lane: lane, t0: t0)
        }

        // ---- Stage 2: Retrieve ----
        // Skip if retriever or model host isn't bundled — degraded
        // mode generates Stage B without grounding, which is what
        // the §10 implementation did. Logged loudly so engineering
        // mode shows the missing signal.
        var retrieval: ColBERTRetrievalResult?
        if let retriever, let modelHost {
            continuation.yield(.retrievalStarted)
            do {
                let result: ColBERTRetrievalResult = try await modelHost.withColBERT { backend in
                    // ADR-024 follow-up 2026-05-27 — `retrievalContext`
                    // carries prior-assistant text when the router
                    // emitted `.augmentRetrievalWithPriorAssistant`.
                    // `.empty` for single-turn / first-turn paths.
                    try await retriever.retrieve(
                        query: query,
                        context: retrievalContext,
                        via: backend
                    )
                }
                continuation.yield(.retrievalComplete(result: result))
                retrieval = result
            } catch {
                continuation.yield(.fallbackInvoked(
                    reason: "ColBERT retrieval threw: \(error.localizedDescription)"
                ))
                logger.error(
                    "ColBERT retrieval failed: \(error.localizedDescription, privacy: .public)"
                )
                // Fall through with retrieval == nil → ungrounded Stage B
            }
        } else {
            continuation.yield(.fallbackInvoked(reason: "ColBERT not bundled"))
        }

        // ---- Re-route on retrieval signal ----
        // If retrieval came back ambiguous or below the floor, the
        // router may now pick .clarification or .unknownFeature
        // instead of staying on .ragStepByStep. Honor that.
        let refinedLane = VerizonRagRouter.route(
            stageA: stageADecision,
            retrieval: retrieval
        )
        if refinedLane != lane {
            // Router downgraded the lane after seeing retrieval. Yield
            // the refined lane so engineering trace shows the
            // re-routing, then dispatch to the appropriate handler.
            continuation.yield(.laneSelected(refinedLane))
            logger.info(
                "lane refined post-retrieval: \(lane.wireName, privacy: .public) → \(refinedLane.wireName, privacy: .public)"
            )
            switch refinedLane {
            case .unknownFeature:
                return await handleUnknownFeature(query: query, lane: refinedLane, t0: t0)
            case .clarification:
                return handleClarification(query: query, lane: refinedLane, t0: t0)
            case .ragStepByStep:
                break  // shouldn't happen given refinedLane != lane, but defensive
            default:
                // Any other lane swap is unexpected here (Stage A
                // already picked .ragStepByStep, retrieval can only
                // downgrade). Surface via fallback.
                continuation.yield(.fallbackInvoked(
                    reason: "unexpected lane swap: \(lane.wireName) → \(refinedLane.wireName)"
                ))
                return await handleUnknownFeature(query: query, lane: refinedLane, t0: t0)
            }
        }

        // ---- Stage 3: Generate (with grounding when retrieval succeeded) ----
        let topChunk = retrieval?.hits.first?.chunk
        continuation.yield(.stageBStarted)
        do {
            let response = try await stageB.generate(
                query: query,
                retrievedChunk: topChunk
            )
            continuation.yield(.stageBComplete(response))

            // ---- Stage 4: Validate (format compliance + faithfulness) ----
            // Format compliance gate — fast deterministic check
            // (extracted URL is known, intro present, no bangs).
            // The GBNF grammar (Phase 2) eliminates most shape
            // failures by construction; this catches escape hatches
            // like hallucinated-but-shaped URLs that pass GBNF but
            // fail isKnownDeepLink.
            guard response.isFormatCompliant else {
                continuation.yield(.fallbackInvoked(reason: "Stage B format check failed"))
                logger.warning(
                    "Stage B fallback (format): \(response.rawText.prefix(120), privacy: .public)"
                )
                return await handleUnknownFeature(query: query, lane: refinedLane, t0: t0)
            }

            // Faithfulness gate — did Stage B actually use the
            // retrieved chunk, or did it paraphrase from training?
            // Token-overlap v1 per ADR §11.4.2; reverse retrieval
            // is v2 (needs dual-backend topology to be cheap).
            // Skipped when topChunk is nil (degraded mode without
            // ColBERT) — nothing to be faithful to.
            let faithfulness = FaithfulnessScorer.score(
                response: response.rawText,
                chunk: topChunk
            )
            continuation.yield(.faithfulnessChecked(score: faithfulness))
            if !faithfulness.isFaithful {
                continuation.yield(.fallbackInvoked(
                    reason: "Stage B faithfulness \(String(format: "%.2f", faithfulness.bigramJaccard)) below floor \(String(format: "%.2f", faithfulness.floor))"
                ))
                logger.warning(
                    "Stage B fallback (faithfulness): jaccard=\(String(format: "%.3f", faithfulness.bigramJaccard), privacy: .public) below floor"
                )
                return await handleUnknownFeature(query: query, lane: refinedLane, t0: t0)
            }

            return VerizonDispatchResult(
                text: response.rawText,
                lane: refinedLane,
                source: .stageB,
                deepLink: response.extractedDeepLink,
                retrievedChunk: topChunk,
                totalMs: elapsed(t0)
            )
        } catch {
            continuation.yield(.fallbackInvoked(reason: "Stage B threw: \(error.localizedDescription)"))
            logger.error("Stage B threw: \(error.localizedDescription, privacy: .public)")
            return await handleUnknownFeature(query: query, lane: refinedLane, t0: t0)
        }
    }

    private nonisolated func elapsed(_ t0: CFAbsoluteTime) -> Double {
        (CFAbsoluteTimeGetCurrent() - t0) * 1000
    }

    // MARK: - Step 6 composer path

    /// Step 6 production answer path: lexical retrieve + ToolRegistry-
    /// gated route derivation + deterministic composer. No LLM call,
    /// no model swap, no GBNF, no faithfulness scorer. Sub-millisecond.
    ///
    /// Guarantees by construction:
    ///   * Every rendered `vzhome://` URL is the selected unit's
    ///     `canonicalURL` (or one of the safe external fallbacks).
    ///   * `requiresConfirmation = true` ONLY when `ToolRegistry`
    ///     has a real `ToolIntent` registered for the unit's `linkID`
    ///     (guardrail #3 — no confirmation theatre on view-only pages).
    ///   * Composer never auto-fires a tool — the dispatcher only
    ///     surfaces `requiresConfirmation` on the result envelope; the
    ///     ChatViewModel's existing affirmative-recovery flow handles
    ///     the actual tool execution after user confirmation.
    private func handleRagStepByStepComposer(
        query: String,
        lane: VerizonLane,
        retrievalContext: RetrievalContext,
        telcoUnderstanding: TelcoSharedUnderstanding? = nil,
        dialogueState: DialogueRepairConversationState = .empty,
        continuation: AsyncStream<VerizonDispatchEvent>.Continuation,
        t0: CFAbsoluteTime
    ) async -> VerizonDispatchResult {
        guard
            let composer = composer,
            let corpus = corpus,
            let lexicalRetriever = lexicalRetriever,
            let toolRegistry = toolRegistry
        else {
            // composerPathEnabled gated this call, so all four must be
            // present. Defensive fallback in case a future caller
            // bypasses the gate.
            continuation.yield(.fallbackInvoked(reason: "composer dependencies missing"))
            return VerizonDispatchResult(
                text: "It looks like I don't have specific information about that. You can check [Verizon Home Internet](https://www.verizon.com/home/internet) for more details.",
                lane: lane,
                source: .composer,
                deepLink: nil,
                totalMs: elapsed(t0),
                retrievalMs: 0,
                routePolicyMs: 0,
                composerMs: 0,
                composerRoute: .noRagAnswer,
                requiresConfirmation: false,
                executableToolIntent: nil,
                citedRAGUnit: nil
            )
        }

        // ---- Retrieve (BM25 hierarchy, post-4c alias-improved) ----
        continuation.yield(.retrievalStarted)
        // RetrievalContext carries at most one prior assistant turn
        // (see ADR-024 / RetrievalContext.swift). Wrap it as a single-
        // element history snippet so the retriever's history-hint
        // mechanism (extract_history_page_hints in the Python source)
        // can mine its deep links.
        let history: [ConversationTurnSnippet] = retrievalContext.priorAssistantText.map {
            [ConversationTurnSnippet(role: "ASSISTANT", body: $0)]
        } ?? []

        // State-conditioned retrieval: the current utterance and the
        // conversation's active cited page are scored together in the
        // retriever. This avoids route-level phrase rules for follow-ups
        // while still allowing high-confidence new-topic evidence to win.
        let retrievalStart = CFAbsoluteTimeGetCurrent()
        let hits = lexicalRetriever.rank(query: query, context: retrievalContext, k: 3)
        let retrievalMs = elapsed(retrievalStart)
        let topUnit = hits.first.flatMap { corpus.unit(forPageID: $0.pageID) }

        // ---- Route derivation (ToolRegistry + alias-map gated) ----
        let routePolicyStart = CFAbsoluteTimeGetCurrent()
        let routeDecision = deriveComposerRoute(
            query: query,
            evidence: topUnit,
            understanding: telcoUnderstanding,
            toolRegistry: toolRegistry,
            aliasMap: toolAliasMap
        )
        let routePolicyMs = elapsed(routePolicyStart)

        // ---- Compose ----
        let composerStart = CFAbsoluteTimeGetCurrent()
        let answer = composer.compose(
            query: query,
            route: routeDecision.route,
            evidence: topUnit,
            requiresConfirmation: routeDecision.requiresConfirmation,
            history: history,
            expectedPolicyLinkID: nil
        )
        let composerMs = elapsed(composerStart)

        var finalText = answer.text
        var finalSource: VerizonDispatchResult.Source = .composer
        var finalComposerMs = composerMs
        if let dialogueRepairVerbalizer,
           let act = DialogueRepairActDeriver.derive(
                query: query,
                route: routeDecision.route,
                evidence: topUnit,
                retrievalContext: retrievalContext,
                understanding: telcoUnderstanding
           ),
           Self.shouldUseDialogueRepairVerbalizer(
                act: act,
                route: routeDecision.route,
                retrievalContext: retrievalContext,
                dialogueState: dialogueState
           ) {
            let verbalizerInput = DialogueRepairVerbalizerInput(
                currentUserTurn: query,
                priorAssistantText: retrievalContext.priorAssistantText,
                conversationState: dialogueState,
                understanding: telcoUnderstanding,
                evidence: topUnit,
                route: routeDecision.route,
                act: act,
                handoff: DialogueRepairActDeriver.handoff(for: routeDecision.route),
                requiresConfirmation: routeDecision.requiresConfirmation ?? false
            )
            let verbalized = await dialogueRepairVerbalizer.verbalize(verbalizerInput)
            finalText = Self.withRuntimeOwnedConfirmation(
                verbalized.text,
                requiresConfirmation: routeDecision.requiresConfirmation
            )
            finalSource = .dialogueRepair
            finalComposerMs += verbalized.latencyMs
            logger.info(
                "dialogue_repair_v4_used act=\(act.rawValue, privacy: .public) fallback=\(verbalized.usedFallback, privacy: .public) extraction=\(verbalized.extractionMode, privacy: .public)"
            )
        }

        logger.info(
            "composer route=\(routeDecision.route.wireName, privacy: .public) pid=\(topUnit?.pageID ?? "<none>", privacy: .public) link=\(topUnit?.linkID ?? "<none>", privacy: .public) confirm=\(routeDecision.requiresConfirmation ?? false, privacy: .public) tool=\(routeDecision.executableToolIntent?.toolID ?? "<none>", privacy: .public) fallback=\(answer.usedFallback, privacy: .public) source=\(finalSource.rawValue, privacy: .public)"
        )

        let deepLink = answer.renderedLinks.first
        return VerizonDispatchResult(
            text: finalText,
            lane: lane,
            source: finalSource,
            deepLink: deepLink,
            retrievedChunk: nil,
            kbEntry: nil,
            totalMs: elapsed(t0),
            retrievalMs: retrievalMs,
            routePolicyMs: routePolicyMs,
            composerMs: finalComposerMs,
            composerRoute: routeDecision.route,
            requiresConfirmation: routeDecision.requiresConfirmation,
            executableToolIntent: routeDecision.executableToolIntent,
            citedRAGUnit: topUnit
        )
    }

    private struct ComposerRouteDecision: Sendable, Equatable {
        let route: ComposerRoute
        let requiresConfirmation: Bool?
        let executableToolIntent: ToolIntent?
    }

    /// Step 5b Pre-flight Fix A wiring. Source of truth for the
    /// `tool_action` vs `answer_plus_action` vs `rag_answer` split when
    /// composer is the answer layer. Mirrors
    /// `scripts/vz/eval/multi_turn_acceptance.py::derive_route` exactly.
    ///
    /// Decision tree:
    ///
    /// 1. **No evidence** → `noRagAnswer` (no confirmation theatre).
    /// 2. **Evidence + affordance is `view` / `navigate`** → `ragAnswer`.
    ///    View-only pages never produce a tool offer, even when their
    ///    `link_id` aliases to a registered tool. This is what keeps
    ///    "how do parental controls work?" (landing on 13.00, a view
    ///    page) on `ragAnswer` rather than `answerPlusAction`.
    /// 3. **Evidence + alias-resolved tool + `imperativeOnly` AND mood
    ///    is not actionImperative** → `ragAnswer`. Carves out parental-
    ///    controls questions from the standard question→answerPlusAction
    ///    rule.
    /// 4. **Evidence + alias-resolved tool + mood**:
    ///     * `actionImperative` → `toolAction` (with `requiresConfirmation`).
    ///     * `question`         → `answerPlusAction` (one-tap alternative).
    ///     * `navigateImperative` / `statement` → `ragAnswer`.
    /// 5. **Evidence + no tool resolvable** → `ragAnswer`.
    ///
    /// **The ToolRegistry + ToolAliasMap together are the FINAL arbiter
    /// for confirmation.** Mood and affordance alone never produce a
    /// confirmation handshake.
    private func deriveComposerRoute(
        query: String,
        evidence: RAGUnit?,
        understanding: TelcoSharedUnderstanding?,
        toolRegistry: ToolRegistry,
        aliasMap: ToolAliasMap?
    ) -> ComposerRouteDecision {
        guard let unit = evidence else {
            // Retrieval returned nothing — composer's no_rag_answer
            // path. Per guardrail #2, KeywordKBExtractor is NOT
            // consulted here.
            return ComposerRouteDecision(
                route: .noRagAnswer,
                requiresConfirmation: nil,
                executableToolIntent: nil
            )
        }
        // Affordance gate: view/navigate pages never produce a tool
        // offer. Keeps view-only landings on rag_answer even when their
        // link_id aliases to a registered tool.
        if let affordance = unit.actionAffordance,
           affordance == "view" || affordance == "navigate" {
            return ComposerRouteDecision(
                route: .ragAnswer,
                requiresConfirmation: false,
                executableToolIntent: nil
            )
        }
        // Resolve through the alias map first, then look the tool up
        // by its canonical tool_id. Fall back to the legacy direct-match
        // path (link_id == tool_id) when no alias map is wired — this
        // keeps the dispatcher working in test contexts that don't
        // bother to construct the map.
        var tool: Tool?
        var imperativeOnly = false
        if let aliasMap, let alias = aliasMap.alias(forLinkID: unit.linkID) {
            if let intent = ToolIntent(toolID: alias.toolID) {
                tool = toolRegistry.tool(for: intent)
                imperativeOnly = alias.imperativeOnly
            }
        } else if let intent = ToolIntent(toolID: unit.linkID) {
            tool = toolRegistry.tool(for: intent)
        }
        guard let resolvedTool = tool,
              let resolvedIntent = ToolIntent(toolID: resolvedTool.id) else {
            return ComposerRouteDecision(
                route: .ragAnswer,
                requiresConfirmation: false,
                executableToolIntent: nil
            )
        }

        if understanding?.routingLane.isConfident(.localAnswer) == true {
            return ComposerRouteDecision(
                route: .ragAnswer,
                requiresConfirmation: false,
                executableToolIntent: nil
            )
        }

        if understanding?.routingLane.isConfident(.localTool) == true {
            return ComposerRouteDecision(
                route: .toolAction,
                requiresConfirmation: resolvedIntent.requiresConfirmation,
                executableToolIntent: resolvedIntent
            )
        }

        let mood = inferQueryMood(query)

        // Imperative-only carve-out: shared link_ids (notably "home" on
        // the 13.xx parental-controls pages) don't get the standard
        // question→answerPlusAction upgrade. Informational queries
        // stay informational.
        if imperativeOnly && mood != .actionImperative {
            return ComposerRouteDecision(
                route: .ragAnswer,
                requiresConfirmation: false,
                executableToolIntent: nil
            )
        }

        switch mood {
        case .actionImperative:
            return ComposerRouteDecision(
                route: .toolAction,
                requiresConfirmation: resolvedIntent.requiresConfirmation,
                executableToolIntent: resolvedIntent
            )
        case .question:
            guard unit.queryTargetsTaskObjective(query) else {
                return ComposerRouteDecision(
                    route: .ragAnswer,
                    requiresConfirmation: false,
                    executableToolIntent: nil
                )
            }
            return ComposerRouteDecision(
                route: .answerPlusAction,
                requiresConfirmation: resolvedIntent.requiresConfirmation,
                executableToolIntent: resolvedIntent
            )
        case .navigateImperative, .statement:
            return ComposerRouteDecision(
                route: .ragAnswer,
                requiresConfirmation: false,
                executableToolIntent: nil
            )
        }
    }

    private nonisolated static func withRuntimeOwnedConfirmation(
        _ text: String,
        requiresConfirmation: Bool?
    ) -> String {
        guard requiresConfirmation == true else { return text }
        let lower = text.lowercased()
        guard !lower.contains("reply 'yes'") && !lower.contains("reply yes") else {
            return text
        }
        return "\(text) Reply 'yes' to confirm."
    }

    private nonisolated static func shouldUseDialogueRepairVerbalizer(
        act: DialogueRepairAct,
        route: ComposerRoute,
        retrievalContext: RetrievalContext,
        dialogueState: DialogueRepairConversationState
    ) -> Bool {
        let priorAssistant = retrievalContext.priorAssistantText?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let hasPriorTurn = priorAssistant.map { !$0.isEmpty } ?? false
        let hasRepairState = hasPriorTurn ||
            dialogueState.pendingConfirmation ||
            dialogueState.frustrationCount > 0

        switch act {
        case .navigationFollowup, .stepFocus, .failedAttempt, .repairCannotFind:
            return hasRepairState
        case .clarification:
            return route == .clarify
        case .noAnswerRepair:
            return route == .noRagAnswer ||
                route == .outOfScope ||
                route == .accountNav ||
                route == .liveAgent
        case .topicPivotRepair:
            return hasPriorTurn
        }
    }
}
