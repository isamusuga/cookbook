import XCTest
@testable import TelcoTriage

/// Coverage for ADR-023 Phase 2 — end-to-end recovery paths that the
/// `ConversationStateTests` unit suite can't exercise:
///
///  - Pending-clarification → slot-fill from short bare-noun answer
///    → tool re-fire with combined args.
///  - Pending-tool-confirmation → "yes" replay path.
///  - Frustration counter → `EscalateOnFrustrationNBA` attachment.
///  - Topic-switch fall-through (recovery doesn't mis-fire on
///    unrelated follow-ups).
///
/// Backed by `TestChatHarness` so the assertions are about user-visible
/// state (assistant messages, attached NBAs, conversation counters)
/// rather than internal call shapes.
@MainActor
final class ConversationRecoveryTests: XCTestCase {

    // MARK: - Pending clarification recovery (the multi-turn slot fill)

    func test_pendingClarification_recoveredFromBareNounAnswer() async {
        // The canonical Phase 2 flow:
        //   Turn 1: user says "pause internet" — tool action lane, but
        //           the target_device slot is empty (no
        //           ImperativeToolDetector match without a device name).
        //   Turn 2: user types "kitchen tablet" — recovered as the
        //           missing slot and the tool re-fires.
        //
        // Note: the simplest way to seed turn 1 in a test harness is
        // to set the pendingClarification directly via ConversationState
        // — we're testing the recovery code path, not the seed
        // mechanism (which is covered by ConversationStateTests).
        let state = ConversationState()
        let harness = TestChatHarness(conversationState: state)
        await harness.whenToolPrompt(returns: """
        {"tool_id":"toggle-parental-controls","arguments":{"action":"pause_internet","target_device":"kitchen tablet"},"reasoning":"recovered slot","requires_confirmation":true,"confidence":0.95}
        """)

        // Simulate the prior turn having asked for the device.
        state.recordTurn(
            userMessage: "pause internet",
            assistantLane: .toolAction,
            toolDecision: nil,
            missingSlots: [.device],
            pendingIntent: .toggleParentalControls,
            originalQuery: "pause internet"
        )
        XCTAssertNotNil(state.pendingClarification)

        // The reply is the slot answer.
        await harness.send("kitchen tablet")

        XCTAssertNil(state.pendingClarification,
                     "successful recovery must clear pendingClarification")
        let decision = harness.lastToolDecision
        XCTAssertNotNil(decision, "recovery must produce a tool decision")
        XCTAssertEqual(decision?.toolID, "toggle-parental-controls")
        let targetArg = decision?.arguments.first(where: { $0.label == "Target Device" })
        XCTAssertEqual(targetArg?.value, "kitchen tablet")
    }

    func test_pendingClarification_doesNotRecoverOnTopicSwitch() async {
        // The user changed their mind and asked something unrelated.
        // The pending must be CLEARED (no stale state) but the new
        // query must route normally via the classifier — not be
        // hijacked into the original intent.
        let state = ConversationState()
        let harness = TestChatHarness(conversationState: state)
        await harness
            .whenModeIs(.kbQuestion, matching: "speed test")
            .whenChatPrompt(contains: "Reference:", returns: "Speed tests measure your line.")

        state.recordTurn(
            userMessage: "pause internet",
            assistantLane: .toolAction,
            toolDecision: nil,
            missingSlots: [.device],
            pendingIntent: .toggleParentalControls,
            originalQuery: "pause internet"
        )
        XCTAssertNotNil(state.pendingClarification)

        await harness.send("how do I run a speed test")

        // Pending was cleared (the user changed topics).
        XCTAssertNil(state.pendingClarification)
        // The new query routed normally — no toggle-parental-controls
        // decision attached.
        XCTAssertNil(harness.lastToolDecision?.toolID == "toggle-parental-controls" ? () : nil,
                     "topic-switch reply must NOT hijack into the original intent")
    }

    // MARK: - Pending tool confirmation recovery (bare "yes" path)

    func test_pendingToolConfirmation_recoveredFromBareYes() async {
        // The assistant proposed a tool last turn. The user types
        // "yes" — that should fire the same tool, not route as OOS.
        let state = ConversationState()
        let harness = TestChatHarness(conversationState: state)
        await harness.whenChatPrompt(
            contains: "Customer-facing summary:",
            returns: "Router restart initiated. Devices will reconnect shortly."
        )

        // Pre-seed a pending tool confirmation (mirrors the state the
        // ChatViewModel would have set after a prior tool-proposal turn).
        let primary = ToolDecision(
            intent: .restartRouter,
            toolID: "restart-router",
            displayName: "Restart Router",
            icon: "arrow.clockwise",
            description: "Reboots the gateway.",
            arguments: [],
            confidence: 0.95,
            reasoning: nil,
            requiresConfirmation: true,
            isDestructive: true
        )
        state.recordTurn(
            userMessage: "restart my router",
            assistantLane: .toolAction,
            toolDecision: primary
        )
        XCTAssertNotNil(state.pendingToolConfirmation)
        let routerBefore = harness.customerContext.profile.equipment.first { $0.kind == .router }?.lastReboot

        await harness.send("yes", timeout: 4.0)

        XCTAssertNil(state.pendingToolConfirmation,
                     "successful confirmation recovery must clear pending")
        XCTAssertNil(harness.lastToolDecision,
                     "bare yes after pending tool must execute, not render a second proposal")
        XCTAssertTrue(
            harness.lastAssistantMessage?.text.contains("Router restart initiated") ?? false
        )
        let routerAfter = harness.customerContext.profile.equipment.first { $0.kind == .router }?.lastReboot
        XCTAssertNotEqual(routerAfter, routerBefore)
    }

    // MARK: - Counter-based escalation NBA

    func test_escalateOnFrustration_firesAfterTwoLiveAgentRequests() {
        // Pure-function NBA check — the escalation chip must fire on
        // the snapshot signal alone, no head needed. Uses the empty
        // QueryUnderstanding to prove the head signal isn't required.
        let snap = ConversationSnapshot(liveAgentRequestCount: 2)
        let understanding = Self.emptyUnderstanding()
        let nba = EscalateOnFrustrationNBA()

        XCTAssertTrue(nba.matchesUnderstanding(
            understanding,
            lane: .telco(.ragStepByStep),
            toolIntent: nil,
            conversation: snap
        ))
    }

    func test_escalateOnFrustration_doesNotFireOnOneRequest() {
        // Single live-agent mention can be exploratory — only the
        // SECOND mention is the escalation signal.
        let snap = ConversationSnapshot(liveAgentRequestCount: 1)
        let understanding = Self.emptyUnderstanding()
        let nba = EscalateOnFrustrationNBA()

        XCTAssertFalse(nba.matchesUnderstanding(
            understanding,
            lane: .telco(.ragStepByStep),
            toolIntent: nil,
            conversation: snap
        ))
    }

    func test_escalateOnFrustration_firesAfterTwoDidntWorkContinuations() {
        let snap = ConversationSnapshot(didntWorkCount: 2)
        let understanding = Self.emptyUnderstanding()
        let nba = EscalateOnFrustrationNBA()

        XCTAssertTrue(nba.matchesUnderstanding(
            understanding,
            lane: .telco(.ragStepByStep),
            toolIntent: nil,
            conversation: snap
        ))
    }

    func test_escalateOnFrustration_neverFiresWhenAlreadyEscalating() {
        // Don't double-offer the chip — when the router already
        // landed on .liveAgentEscalation the live-agent template IS
        // the answer.
        let snap = ConversationSnapshot(liveAgentRequestCount: 5)
        let understanding = Self.emptyUnderstanding()
        let nba = EscalateOnFrustrationNBA()

        XCTAssertFalse(nba.matchesUnderstanding(
            understanding,
            lane: .telco(.liveAgentEscalation),
            toolIntent: nil,
            conversation: snap
        ))
    }

    func test_escalateOnFrustration_legacyHeadPathStillFiresWithoutSnapshot() {
        // ADR-022 (pre-Phase-2) behaviour: when the emotional_state head
        // says frustrated, the chip fires even without conversation
        // state. Source-compatible defaults preserved.
        let understanding = Self.understandingWithEmotional(.frustrated)
        let nba = EscalateOnFrustrationNBA()

        XCTAssertTrue(nba.matchesUnderstanding(
            understanding,
            lane: .telco(.ragStepByStep),
            toolIntent: nil,
            conversation: nil
        ))
    }

    // MARK: - Engine integration: counter signal beats keyword path

    func test_engine_returnsEscalationNBAWhenCounterFires() async {
        // bestMatchForUnderstanding must surface the
        // EscalateOnFrustrationNBA when conversation snapshot carries
        // the counter signal, even with an empty understanding.
        let context = CustomerContext(profile: .demo)
        let engine = NextBestActionEngine(
            registry: .default,
            customerContext: context
        )
        let understanding = Self.emptyUnderstanding()
        let snap = ConversationSnapshot(liveAgentRequestCount: 3)

        let match = engine.bestMatchForUnderstanding(
            understanding,
            lane: .telco(.ragStepByStep),
            toolIntent: nil,
            conversation: snap
        )

        XCTAssertEqual(match?.id, "escalate-on-frustration")
    }

    // MARK: - Fixtures

    private static func emptyUnderstanding() -> QueryUnderstanding {
        QueryUnderstanding(
            chatMode: nil,
            topicGate: nil,
            refusalFlags: nil,
            emotionalState: nil,
            slotCompleteness: nil,
            totalMs: 0,
            strategy: .composite
        )
    }

    private static func understandingWithEmotional(
        _ value: EmotionalState
    ) -> QueryUnderstanding {
        QueryUnderstanding(
            chatMode: nil,
            topicGate: nil,
            refusalFlags: nil,
            emotionalState: EmotionalStateOutcome(value: value, confidence: 0.9),
            slotCompleteness: nil,
            totalMs: 0,
            strategy: .composite
        )
    }
}
