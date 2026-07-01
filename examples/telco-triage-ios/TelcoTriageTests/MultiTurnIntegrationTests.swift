import XCTest
@testable import TelcoTriage

/// Swift parity tests for the Step 5b multi-turn acceptance fixtures
/// (`data/finetune/telco-home-internet/multi_turn_acceptance_v1.jsonl`).
///
/// **What this verifies.** The iOS `TelcoChatDispatcher` produces the
/// same per-turn route + cited page + cited link as the Python
/// `simulate_turn` harness for the 8 scenarios that lock in the Step 5b
/// acceptance gates. The Swift dispatcher today only owns the
/// retrieval + route-derivation + composition slice; the multi-turn
/// glue around it (pending-tool fire on "yes", didn't-work →
/// escalation, ambiguous-yes-ignored, topic-switch-clear) lives in
/// `ChatViewModel.applyPostDecisionActions` + `ConversationState`.
///
/// This test reproduces that glue in a tiny in-test simulator —
/// `Step5bParityTurnSimulator` — that:
///
/// 1. Holds the same multi-turn signals the Python harness does
///    (pending tool, prior page/link, didn't-work counter).
/// 2. For each user turn, decides whether to short-circuit the
///    dispatcher (affirmative with pending → fire; didn't-work →
///    increment / escalate; bare-yes without pending → clarify) or
///    forward through to `dispatcher.dispatch(query:retrievalContext:)`
///    with the prior turn's page/link threaded in.
/// 3. Updates state from the dispatcher's `TelcoDispatchResult`.
///
/// **Why this is a real parity test.** The dispatcher's
/// short-followup override is the production code path that needs to
/// match the Python harness behaviour — this test exercises it
/// end-to-end. The in-test simulator only fills in the multi-turn
/// glue that the iOS production code does in `ChatViewModel`; the
/// dispatcher itself is unmocked.
///
/// Stub Stage A skips the LFM classifier so these tests run in
/// milliseconds with no model load.
@MainActor
final class MultiTurnIntegrationTests: XCTestCase {

    // MARK: - Test infrastructure

    private var corpus: RAGUnitCorpus!
    private var retriever: BM25HierarchyRetriever!
    private var composer: DeterministicAnswerComposer!
    private var toolRegistry: ToolRegistry!
    private var toolAliasMap: ToolAliasMap!

    override func setUpWithError() throws {
        try super.setUpWithError()
        corpus = try RAGUnitCorpus.loadFromBundle()
        retriever = BM25HierarchyRetriever(corpus: corpus)
        composer = DeterministicAnswerComposer()
        toolRegistry = ToolRegistry.demoDefault(customerContext: CustomerContext())
        toolAliasMap = ToolAliasMap.default()
    }

    private func makeDispatcher() -> TelcoChatDispatcher {
        TelcoChatDispatcher(
            stageA: StubStageAClassifier(),
            stageB: nil,
            kbFallback: StubKBExtractor(),
            kb: [],
            retriever: nil,
            modelHost: nil,
            composer: composer,
            corpus: corpus,
            lexicalRetriever: retriever,
            toolRegistry: toolRegistry,
            toolAliasMap: toolAliasMap
        )
    }

    private func makeSimulator() -> Step5bParityTurnSimulator {
        Step5bParityTurnSimulator(
            dispatcher: makeDispatcher(),
            toolRegistry: toolRegistry,
            toolAliasMap: toolAliasMap
        )
    }

    // MARK: - The 8 acceptance fixtures

    /// S1 — change password → "Where do I find Network?" → "What's on the home page?"
    /// Tests context persistence within a topic, then a cross-section shift to
    /// the literal Home page.
    func test_S1_change_password_then_navigation_questions() async {
        let sim = makeSimulator()

        await sim.assertTurn(
            query: "How do I change my wifi password?",
            expectedRoute: .ragAnswer,
            expectedPageID: "03.00",
            expectedLinkID: "network"
        )
        await sim.assertTurn(
            query: "Where do I find Network?",
            expectedRoute: .ragAnswer,
            expectedPageID: "03.00",
            expectedLinkID: "network",
            expectedReusePriorPage: true
        )
        await sim.assertTurn(
            query: "What's on the home page?",
            expectedRoute: .ragAnswer,
            expectedPageID: "01.00",
            expectedLinkID: "tab-home",
            expectedClearPriorContext: true
        )
    }

    /// S2 — canonical pending-tool handshake. "Restart my router" sets pending,
    /// "yes" fires it.
    func test_S2_restart_router_yes_confirmation() async {
        let sim = makeSimulator()

        await sim.assertTurn(
            query: "Restart my router",
            expectedRoute: .toolAction,
            expectedPageID: "02.07",
            expectedLinkID: "restart-router",
            expectedRequiresConfirmation: true
        )
        await sim.assertTurn(
            query: "yes",
            expectedRoute: .toolAction,
            expectedPageID: "02.07",
            expectedLinkID: "restart-router",
            expectedToolFired: true
        )
    }

    /// S3 — "How do I restart my router?" → "I tried that, still not working"
    /// → "still not working". First strike increments frustration; second
    /// strike escalates to live agent. Pending must NOT fire on "didn't work".
    func test_S3_how_to_restart_then_didnt_work_escalates() async {
        let sim = makeSimulator()

        await sim.assertTurn(
            query: "How do I restart my router?",
            expectedRoute: .answerPlusAction,
            expectedPageID: "02.07",
            expectedLinkID: "restart-router",
            expectedRequiresConfirmation: true
        )
        await sim.assertTurn(
            query: "I tried that, still not working",
            expectedRoute: .ragAnswer,
            expectedPageID: "02.07",
            expectedLinkID: "restart-router",
            expectedReusePriorPage: true,
            expectedFrustrationIncrement: true
        )
        await sim.assertTurn(
            query: "still not working",
            expectedRoute: .liveAgent,
            expectedPageID: nil,
            expectedLinkID: nil,
            expectedEscalation: true
        )
    }

    /// S4 — "What's self organizing network?" → "how do I turn it off?"
    /// Exercises the corpus's new SON aliases (was empty pre Pre-flight B)
    /// AND the dispatcher's anaphoric-pronoun short-followup override.
    func test_S4_self_organizing_network_then_turn_off() async {
        let sim = makeSimulator()

        await sim.assertTurn(
            query: "What's self organizing network?",
            expectedRoute: .ragAnswer,
            expectedPageID: "03.03",
            expectedLinkID: "network"
        )
        await sim.assertTurn(
            query: "how do I turn it off?",
            expectedRoute: .ragAnswer,
            expectedPageID: "03.03",
            expectedLinkID: "network",
            expectedReusePriorPage: true
        )
    }

    /// S5 — parental controls flow. Create-profile is a navigation/Q&A
    /// page until a real create-profile tool exists. Slot-prefix
    /// follow-up reuses 13.02.
    func test_S5_parental_controls_add_profile_for_son_tablet() async {
        let sim = makeSimulator()

        await sim.assertTurn(
            query: "Set up parental controls",
            expectedRoute: .ragAnswer,
            expectedPageID: "13.00",
            expectedLinkID: "home"
        )
        await sim.assertTurn(
            query: "add a profile",
            expectedRoute: .ragAnswer,
            expectedPageID: "13.02",
            expectedLinkID: "home",
            expectedRequiresConfirmation: false
        )
        await sim.assertTurn(
            query: "for my son's tablet",
            expectedRoute: .ragAnswer,
            expectedPageID: "13.02",
            expectedLinkID: "home",
            expectedReusePriorPage: true
        )
    }

    /// S6 — short-followup override on a bare wh-word. "Show data usage"
    /// lands on 04.00, then "How?" must reuse it.
    func test_S6_show_data_usage_how() async {
        let sim = makeSimulator()

        await sim.assertTurn(
            query: "Show data usage",
            expectedRoute: .ragAnswer,
            expectedPageID: "04.00",
            expectedLinkID: "tab-devices"
        )
        await sim.assertTurn(
            query: "How?",
            expectedRoute: .ragAnswer,
            expectedPageID: "04.00",
            expectedLinkID: "tab-devices",
            expectedReusePriorPage: true
        )
    }

    /// S7 — topic-switch clears pending. "Restart my router" sets pending,
    /// "actually show connected devices" pivots → pending cleared, new
    /// retrieval lands on 04.00.
    func test_S7_topic_switch_restart_to_devices_clears_pending() async {
        let sim = makeSimulator()

        await sim.assertTurn(
            query: "Restart my router",
            expectedRoute: .toolAction,
            expectedPageID: "02.07",
            expectedLinkID: "restart-router",
            expectedRequiresConfirmation: true
        )
        await sim.assertTurn(
            query: "actually show connected devices",
            expectedRoute: .ragAnswer,
            expectedPageID: "04.00",
            expectedLinkID: "tab-devices",
            expectedClearPriorContext: true
        )
    }

    /// S8 — guardrail #3: bare "yes" on the first turn with no pending tool
    /// must NOT fire anything. Routes to clarify.
    func test_S8_ambiguous_yes_without_pending_tool() async {
        let sim = makeSimulator()

        await sim.assertTurn(
            query: "yes",
            expectedRoute: .clarify,
            expectedPageID: nil,
            expectedLinkID: nil,
            expectedToolFired: false
        )
    }
}

// MARK: - Step 5b parity turn simulator

/// In-test reproduction of the Python `simulate_turn` multi-turn glue.
/// Mirrors `scripts/telco/eval/multi_turn_acceptance.py::simulate_turn`
/// for the branches that don't live in the iOS dispatcher today —
/// pending-tool firing, didn't-work counter, bare-yes-ignored,
/// topic-switch clear. The dispatcher itself handles retrieval,
/// route derivation, composition, and the short-followup override.
@MainActor
final class Step5bParityTurnSimulator {
    private let dispatcher: TelcoChatDispatcher
    private let toolRegistry: ToolRegistry
    private let toolAliasMap: ToolAliasMap

    private(set) var pendingToolIntent: ToolIntent?
    private(set) var pendingToolPageID: String?
    private(set) var priorPageID: String?
    private(set) var priorLinkID: String?
    private(set) var priorAssistantText: String?
    private(set) var didntWorkCount: Int = 0
    private(set) var turnIndex: Int = 0

    private static let frustrationEscalationThreshold = 2

    init(
        dispatcher: TelcoChatDispatcher,
        toolRegistry: ToolRegistry,
        toolAliasMap: ToolAliasMap
    ) {
        self.dispatcher = dispatcher
        self.toolRegistry = toolRegistry
        self.toolAliasMap = toolAliasMap
    }

    /// Run one turn and assert against the labels. All `expected*` args are
    /// optional; nil means "don't assert this column."
    func assertTurn(
        query: String,
        expectedRoute: ComposerRoute,
        expectedPageID: String?,
        expectedLinkID: String?,
        expectedRequiresConfirmation: Bool? = nil,
        expectedToolFired: Bool = false,
        expectedReusePriorPage: Bool = false,
        expectedClearPriorContext: Bool = false,
        expectedFrustrationIncrement: Bool = false,
        expectedEscalation: Bool = false,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async {
        turnIndex += 1
        let context = "turn \(turnIndex) query=\"\(query)\""

        // --- Pending-tool handshake gating (mirrors Python simulate_turn) ---
        if pendingToolIntent != nil, isAffirmative(query) {
            XCTAssertTrue(
                expectedToolFired,
                "\(context): fixture expected NO tool fire, but pending tool + affirmative would fire one",
                file: file, line: line
            )
            XCTAssertEqual(
                expectedRoute, .toolAction,
                "\(context): fire-pending route should be tool_action",
                file: file, line: line
            )
            // Record cited page from the pending — clear pending state.
            priorPageID = pendingToolPageID
            priorLinkID = expectedLinkID
            pendingToolIntent = nil
            pendingToolPageID = nil
            return
        }

        if pendingToolIntent != nil, isDidntWork(query) {
            didntWorkCount += 1
            if didntWorkCount >= Self.frustrationEscalationThreshold {
                XCTAssertTrue(
                    expectedEscalation,
                    "\(context): second-strike didn't-work triggered escalation, fixture didn't expect it",
                    file: file, line: line
                )
                XCTAssertEqual(
                    expectedRoute, .liveAgent,
                    "\(context): escalation should route to live_agent",
                    file: file, line: line
                )
                pendingToolIntent = nil
                pendingToolPageID = nil
                priorPageID = nil
                priorLinkID = nil
                return
            } else {
                XCTAssertTrue(
                    expectedFrustrationIncrement,
                    "\(context): first-strike didn't-work incremented frustration, fixture didn't expect it",
                    file: file, line: line
                )
                XCTAssertEqual(
                    expectedRoute, .ragAnswer,
                    "\(context): first-strike didn't-work should re-surface page on rag_answer",
                    file: file, line: line
                )
                // Pending tool stays — re-surface the page, do not fire.
                XCTAssertEqual(expectedPageID, pendingToolPageID, file: file, line: line)
                XCTAssertEqual(expectedReusePriorPage, true, file: file, line: line)
                return
            }
        }

        if pendingToolIntent == nil, isAffirmative(query) {
            XCTAssertEqual(
                expectedRoute, .clarify,
                "\(context): ambiguous yes without pending must route to clarify (guardrail #3)",
                file: file, line: line
            )
            XCTAssertFalse(
                expectedToolFired,
                "\(context): bare yes without pending must NOT fire",
                file: file, line: line
            )
            return
        }

        // --- Normal dispatch path: forward to TelcoChatDispatcher ---
        let retrievalContext = RetrievalContext(
            priorAssistantText: priorAssistantText,
            priorPageID: priorPageID,
            priorLinkID: priorLinkID
        )
        let stageA = Self.makeStageADecision()
        var finalResult: TelcoDispatchResult?
        for await event in dispatcher.dispatch(
            query: query,
            prebuiltStageA: stageA,
            prebuiltLane: .ragStepByStep,
            retrievalContext: retrievalContext
        ) {
            if case .response(let r) = event { finalResult = r }
        }
        guard let result = finalResult else {
            XCTFail("\(context): dispatcher produced no .response event", file: file, line: line)
            return
        }

        XCTAssertEqual(
            result.composerRoute, expectedRoute,
            "\(context): expected route \(expectedRoute.wireName), got \(result.composerRoute?.wireName ?? "<nil>")",
            file: file, line: line
        )
        XCTAssertEqual(
            result.citedRAGUnit?.pageID, expectedPageID,
            "\(context): expected page \(expectedPageID ?? "<nil>"), got \(result.citedRAGUnit?.pageID ?? "<nil>")",
            file: file, line: line
        )
        XCTAssertEqual(
            result.citedRAGUnit?.linkID, expectedLinkID,
            "\(context): expected link \(expectedLinkID ?? "<nil>"), got \(result.citedRAGUnit?.linkID ?? "<nil>")",
            file: file, line: line
        )
        if let expectedConf = expectedRequiresConfirmation {
            XCTAssertEqual(
                result.requiresConfirmation ?? false, expectedConf,
                "\(context): expected requiresConfirmation=\(expectedConf), got \(result.requiresConfirmation ?? false)",
                file: file, line: line
            )
        }
        if expectedReusePriorPage, let prior = priorPageID {
            XCTAssertEqual(
                result.citedRAGUnit?.pageID, prior,
                "\(context): expected reuse of prior page \(prior), got \(result.citedRAGUnit?.pageID ?? "<nil>")",
                file: file, line: line
            )
        }
        if expectedClearPriorContext, let prior = priorPageID,
           let newPage = result.citedRAGUnit?.pageID {
            XCTAssertTrue(
                isCrossSectionShift(priorPageID: prior, newPageID: newPage)
                    || hasTopicSwitchPrefix(query),
                "\(context): expected cleared prior context but new page \(newPage) is in same section as prior \(prior) and no topic-switch prefix",
                file: file, line: line
            )
        }

        // --- Update state for next turn ---
        if expectedRoute == .toolAction || expectedRoute == .answerPlusAction,
           let unit = result.citedRAGUnit,
           let alias = toolAliasMap.alias(forLinkID: unit.linkID),
           let intent = ToolIntent(toolID: alias.toolID),
           toolRegistry.tool(for: intent) != nil
        {
            pendingToolIntent = intent
            pendingToolPageID = unit.pageID
        } else if expectedClearPriorContext {
            // Explicit topic-switch — drop any pending without firing.
            pendingToolIntent = nil
            pendingToolPageID = nil
        }
        priorPageID = result.citedRAGUnit?.pageID
        priorLinkID = result.citedRAGUnit?.linkID
        priorAssistantText = result.text
    }

    private static func makeStageADecision() -> TelcoStageADecision {
        TelcoStageADecision(
            topicGate: .inScope,
            topicGateConfidence: 0.95,
            topicGateProbabilities: [0.05, 0.95],
            refusalFlags: .none,
            refusalFlagsProbabilities: [0, 0, 0],
            totalMs: 0
        )
    }
}

// MARK: - Stage A stub (shared with TelcoDispatcherComposerPathTests)

private struct StubStageAClassifier: TelcoStageAClassifying {
    func classify(query: String) async throws -> TelcoStageADecision {
        TelcoStageADecision(
            topicGate: .inScope,
            topicGateConfidence: 0.95,
            topicGateProbabilities: [0.05, 0.95],
            refusalFlags: .none,
            refusalFlagsProbabilities: [0, 0, 0],
            totalMs: 0
        )
    }
}

private struct StubKBExtractor: KBExtractor {
    func extract(query: String, kb: [KBEntry]) async -> KBCitation {
        KBCitation.noMatch(runtimeMS: 0)
    }
}
