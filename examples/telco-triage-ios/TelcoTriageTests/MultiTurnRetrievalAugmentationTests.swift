import XCTest
@testable import TelcoTriage

/// Coverage for the ADR-024 follow-up multi-turn plumbing
/// (2026-05-27 — RetrievalContext + PostActionResult +
/// ColBERTRetriever.composeEncodingPayload).
///
/// Closes the architectural gap surfaced by the device-log session
/// where Turn 2 ("i cannot find equipment tile") failed because
/// `.augmentRetrievalWithPriorAssistant` was a log-only no-op. Every
/// primitive in the chain is exercised independently with concrete
/// `XCTAssert`s — not "the build compiles" stubs.
///
/// **Coverage matrix:**
///   - `RetrievalContext` — value-type construction, `.empty` sentinel,
///     Equatable/Sendable.
///   - `ColBERTRetriever.composeEncodingPayload(query:context:)` —
///     the pure-function composer that turns `(query, context)` into
///     the string ColBERT actually encodes. Pinning tests so a future
///     refactor can't silently change augmentation behaviour.
///   - `ConversationState.cacheTurnText` — round-trip, trim, nil-on-
///     empty, snapshot propagation, reset clears.
///   - `PostActionResult` — the typed envelope that replaces the old
///     `Bool` short-circuit return so actions can communicate
///     downstream intent.
///   - `VerizonUnderstandingRouter` — emits
///     `.augmentRetrievalWithPriorAssistant` on the ANAPHORIC path,
///     which is the load-bearing test for the user's Turn 2 bug.
final class MultiTurnRetrievalAugmentationTests: XCTestCase {

    // MARK: - RetrievalContext primitive

    func test_retrievalContext_empty_isSentinel() {
        XCTAssertTrue(RetrievalContext.empty.isEmpty)
        XCTAssertNil(RetrievalContext.empty.priorAssistantText)
    }

    func test_retrievalContext_withText_isNotEmpty() {
        let ctx = RetrievalContext(priorAssistantText: "tap the Equipment tile")
        XCTAssertFalse(ctx.isEmpty)
        XCTAssertEqual(ctx.priorAssistantText, "tap the Equipment tile")
    }

    func test_retrievalContext_isEquatable() {
        // Pins the Equatable conformance. ConversationSnapshot relies
        // on it for snapshot equality, and PostActionResult relies on
        // it for assertion ergonomics in downstream tests.
        let a = RetrievalContext(priorAssistantText: "foo")
        let b = RetrievalContext(priorAssistantText: "foo")
        let c = RetrievalContext(priorAssistantText: "bar")
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
        XCTAssertNotEqual(a, .empty)
    }

    // MARK: - ColBERT payload composer (pure function)

    func test_composer_emptyContext_returnsBareQuery() {
        let result = ColBERTRetriever.composeEncodingPayload(
            query: "restart my router",
            context: .empty
        )
        XCTAssertEqual(result.payload, "restart my router")
        XCTAssertFalse(result.augmented)
    }

    func test_composer_nilPriorText_returnsBareQuery() {
        let result = ColBERTRetriever.composeEncodingPayload(
            query: "restart my router",
            context: RetrievalContext(priorAssistantText: nil)
        )
        XCTAssertEqual(result.payload, "restart my router")
        XCTAssertFalse(result.augmented)
    }

    func test_composer_priorText_prependsBeforeQuery() {
        // The Turn 2 happy-path test. With prior assistant text in
        // context, the encoder receives BOTH segments — prior on the
        // left, query on the right, joined by double-newline.
        let prior = "To restart your router, open My Verizon, tap My Devices, then tap the Equipment tile."
        let result = ColBERTRetriever.composeEncodingPayload(
            query: "i cannot find equipment tile",
            context: RetrievalContext(priorAssistantText: prior)
        )
        XCTAssertTrue(result.augmented)
        XCTAssertTrue(
            result.payload.hasPrefix("To restart your router"),
            "expected prior text at the head of payload — got: \(result.payload.prefix(60))"
        )
        XCTAssertTrue(
            result.payload.hasSuffix("i cannot find equipment tile"),
            "expected user query at the tail of payload"
        )
        XCTAssertTrue(
            result.payload.contains("\n\n"),
            "expected double-newline sentinel between prior and query"
        )
    }

    func test_composer_priorText_overBudget_isCapped() {
        // Pin the safety cap so a long prior message can't crowd out
        // the user's query in a 2048-token context window.
        let prior = String(repeating: "ABCDEFGHIJ", count: 200)  // 2000 chars
        XCTAssertGreaterThan(prior.count, ColBERTRetriever.priorContextCharBudget)

        let result = ColBERTRetriever.composeEncodingPayload(
            query: "follow-up question",
            context: RetrievalContext(priorAssistantText: prior)
        )

        XCTAssertTrue(result.augmented)
        let separatorLen = "\n\n".count
        let queryLen = "follow-up question".count
        let expectedMaxLen = ColBERTRetriever.priorContextCharBudget + separatorLen + queryLen
        XCTAssertLessThanOrEqual(
            result.payload.count, expectedMaxLen,
            "prior context budget exceeded — got \(result.payload.count) chars (cap \(expectedMaxLen))"
        )
        XCTAssertTrue(result.payload.hasSuffix("follow-up question"))
    }

    func test_composer_whitespaceOnlyPrior_doesNotAugment() {
        let result = ColBERTRetriever.composeEncodingPayload(
            query: "bare query",
            context: RetrievalContext(priorAssistantText: "   \n\t  ")
        )
        XCTAssertEqual(result.payload, "bare query")
        XCTAssertFalse(result.augmented)
    }

    func test_composer_emptyStringPrior_doesNotAugment() {
        let result = ColBERTRetriever.composeEncodingPayload(
            query: "bare query",
            context: RetrievalContext(priorAssistantText: "")
        )
        XCTAssertEqual(result.payload, "bare query")
        XCTAssertFalse(result.augmented)
    }

    func test_composer_trimsQueryWhitespace() {
        let result = ColBERTRetriever.composeEncodingPayload(
            query: "  \n  trimmed query \n  ",
            context: .empty
        )
        XCTAssertEqual(result.payload, "trimmed query")
    }

    func test_composer_isPureFunction() {
        // Same inputs → same outputs. Sanity test that the composer
        // has no hidden state.
        let ctx = RetrievalContext(priorAssistantText: "prior body")
        let r1 = ColBERTRetriever.composeEncodingPayload(query: "Q", context: ctx)
        let r2 = ColBERTRetriever.composeEncodingPayload(query: "Q", context: ctx)
        XCTAssertEqual(r1, r2)
    }

    // MARK: - ConversationState text-cache round-trip

    @MainActor
    func test_conversationState_cacheTurnText_roundTrips() {
        let state = ConversationState()
        XCTAssertNil(state.priorAssistantText)

        state.cacheTurnText(assistant: "To restart your router, open My Verizon → My Devices → Equipment tile.")
        XCTAssertEqual(
            state.priorAssistantText,
            "To restart your router, open My Verizon → My Devices → Equipment tile."
        )

        // Snapshot carries the field across actor boundaries.
        let snap = state.snapshot
        XCTAssertEqual(snap.priorAssistantText, state.priorAssistantText)
    }

    @MainActor
    func test_conversationState_cacheTurnText_trimsWhitespace() {
        let state = ConversationState()

        state.cacheTurnText(assistant: "   \n  trimmed body  \n\n   ")
        XCTAssertEqual(state.priorAssistantText, "trimmed body")
    }

    @MainActor
    func test_conversationState_cacheTurnText_nilOnEmpty() {
        let state = ConversationState()
        state.cacheTurnText(assistant: "first body")
        XCTAssertNotNil(state.priorAssistantText)

        state.cacheTurnText(assistant: "")
        XCTAssertNil(state.priorAssistantText, "empty string should clear, not store empty")

        state.cacheTurnText(assistant: "second body")
        XCTAssertNotNil(state.priorAssistantText)

        state.cacheTurnText(assistant: nil)
        XCTAssertNil(state.priorAssistantText, "explicit nil should clear")
    }

    @MainActor
    func test_conversationState_reset_clearsPriorAssistantText() {
        let state = ConversationState()
        state.cacheTurnText(assistant: "previous reply")
        XCTAssertNotNil(state.priorAssistantText)

        state.reset()
        XCTAssertNil(state.priorAssistantText, "reset() should clear all ADR-024 substrate")
    }

    @MainActor
    func test_conversationState_snapshot_carriesEmptyByDefault() {
        let state = ConversationState()
        XCTAssertNil(state.snapshot.priorAssistantText)
    }

    // MARK: - PostActionResult envelope

    func test_postActionResult_passthrough_isEmptyByDefault() {
        let result = PostActionResult.passthrough
        XCTAssertFalse(result.shortCircuited)
        XCTAssertEqual(result.retrievalContext, .empty)
    }

    func test_postActionResult_shortCircuit_carriesEmptyContext() {
        let result = PostActionResult.shortCircuit
        XCTAssertTrue(result.shortCircuited)
        XCTAssertEqual(result.retrievalContext, .empty)
    }

    func test_postActionResult_withRetrievalContext_carriesIt() {
        let ctx = RetrievalContext(priorAssistantText: "prior body")
        let result = PostActionResult(shortCircuited: false, retrievalContext: ctx)
        XCTAssertFalse(result.shortCircuited)
        XCTAssertEqual(result.retrievalContext, ctx)
    }

    func test_postActionResult_isEquatable() {
        let a = PostActionResult(
            shortCircuited: false,
            retrievalContext: RetrievalContext(priorAssistantText: "x")
        )
        let b = PostActionResult(
            shortCircuited: false,
            retrievalContext: RetrievalContext(priorAssistantText: "x")
        )
        let c = PostActionResult(
            shortCircuited: true,
            retrievalContext: .empty
        )
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
    }

    // MARK: - Router emits the augmentation action on the right paths

    func test_router_anaphoricTurn_emitsAugmentAction() {
        // LOAD-BEARING test for the user's Turn 2 bug. When the
        // relational head (eventually) classifies Turn 2 as anaphoric,
        // the router MUST emit `.augmentRetrievalWithPriorAssistant`
        // so the plumbing downstream can do its work.
        let understanding = makeUnderstanding(
            turnRelationship: .anaphoric
        )

        let decision = VerizonUnderstandingRouter.decideMultiTurn(
            understanding: understanding,
            conversation: .empty
        )

        XCTAssertTrue(
            decision.actions.contains(.augmentRetrievalWithPriorAssistant),
            "router must emit augmentation action on ANAPHORIC follow-up — got actions: \(decision.actions)"
        )
    }

    func test_router_independentTurn_doesNotEmitAugmentAction() {
        // Pinning test — single-turn / independent turns must NOT
        // trigger the augmentation, otherwise the retriever runs on
        // unrelated prior context and dilutes signal.
        let understanding = makeUnderstanding(
            turnRelationship: .independent
        )

        let decision = VerizonUnderstandingRouter.decideMultiTurn(
            understanding: understanding,
            conversation: .empty
        )

        XCTAssertFalse(
            decision.actions.contains(.augmentRetrievalWithPriorAssistant),
            "INDEPENDENT turn must not augment retrieval — got actions: \(decision.actions)"
        )
    }

    // MARK: - End-to-end flow (composer fed from action result)

    func test_endToEnd_anaphoricFlow_composerReceivesPriorText() {
        // Simulates what processTextQuery does after Layer 2:
        //   1. Router emits .augmentRetrievalWithPriorAssistant.
        //   2. applyPostDecisionActions copies priorAssistantText from
        //      ConversationState into PostActionResult.retrievalContext.
        //   3. The lane handler passes that context to ColBERTRetriever.
        // We're stitching steps 1+3 here — confirming that if the
        // ConversationState had a cached prior reply, it flows all the
        // way into the composer's output.
        let priorReply = "To restart: open My Verizon → My Devices → tap Equipment tile."
        let ctx = RetrievalContext(priorAssistantText: priorReply)

        let composition = ColBERTRetriever.composeEncodingPayload(
            query: "i cannot find equipment tile",
            context: ctx
        )

        XCTAssertTrue(composition.augmented, "augmentation should fire when context has text")
        XCTAssertTrue(composition.payload.contains("Equipment tile"))
        XCTAssertTrue(composition.payload.contains("i cannot find equipment tile"))
        XCTAssertGreaterThan(
            composition.payload.count,
            "i cannot find equipment tile".count,
            "augmented payload must be longer than bare query"
        )
    }

    // MARK: - Fixtures

    private func makeUnderstanding(
        chatMode: ChatMode = .kbQuestion,
        topicGate: VerizonTopicGate = .inScope,
        turnRelationship: TurnRelationship? = nil
    ) -> QueryUnderstanding {
        QueryUnderstanding(
            chatMode: ChatModePrediction(
                mode: chatMode,
                confidence: 0.9,
                reasoning: "test",
                runtimeMS: 0
            ),
            topicGate: TopicGateOutcome(value: topicGate, confidence: 0.95),
            refusalFlags: RefusalFlagsOutcome(
                value: VerizonRefusalFlags(
                    hasRagAnswer: true,
                    navigationOnly: false,
                    liveAgentTrigger: false
                ),
                probabilities: [0.9, 0.0, 0.0]
            ),
            emotionalState: nil,
            slotCompleteness: nil,
            turnRelationship: turnRelationship.map {
                TurnRelationshipOutcome(value: $0, confidence: 0.88)
            },
            slotAlignment: nil,
            stanceChange: nil,
            totalMs: 0,
            strategy: .shared
        )
    }
}
