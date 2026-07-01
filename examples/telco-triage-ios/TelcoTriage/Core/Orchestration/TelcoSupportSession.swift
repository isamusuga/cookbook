import Foundation
import os.log

/// UI-independent result for one support turn.
///
/// Host apps can render this into any chat, voice, or orchestration surface.
/// The result deliberately carries structured route, citation, tool, timing,
/// and blackboard fields so clients do not need to import the cookbook chat UI.
public struct TelcoSupportTurnResult: Sendable, Equatable {
    public let query: String
    public let text: String
    public let lane: TelcoLane
    public let source: TelcoDispatchResult.Source
    public let route: ComposerRoute?
    public let citation: TelcoSupportCitation?
    public let deepLink: String?
    public let requiresConfirmation: Bool
    public let executableToolIntent: ToolIntent?
    public let retrievalCandidates: [TelcoRetrievalCandidate]
    public let timing: TelcoSupportTiming
    public let blackboard: TelcoDialogueBlackboard
    public let dispatchResult: TelcoDispatchResult

    public init(
        query: String,
        text: String,
        lane: TelcoLane,
        source: TelcoDispatchResult.Source,
        route: ComposerRoute?,
        citation: TelcoSupportCitation?,
        deepLink: String?,
        requiresConfirmation: Bool,
        executableToolIntent: ToolIntent?,
        retrievalCandidates: [TelcoRetrievalCandidate],
        timing: TelcoSupportTiming,
        blackboard: TelcoDialogueBlackboard,
        dispatchResult: TelcoDispatchResult
    ) {
        self.query = query
        self.text = text
        self.lane = lane
        self.source = source
        self.route = route
        self.citation = citation
        self.deepLink = deepLink
        self.requiresConfirmation = requiresConfirmation
        self.executableToolIntent = executableToolIntent
        self.retrievalCandidates = retrievalCandidates
        self.timing = timing
        self.blackboard = blackboard
        self.dispatchResult = dispatchResult
    }
}

public struct TelcoSupportCitation: Sendable, Equatable {
    public let label: String
    public let pageID: String?
    public let linkID: String?
    public let url: String?

    public init(label: String, pageID: String?, linkID: String?, url: String?) {
        self.label = label
        self.pageID = pageID
        self.linkID = linkID
        self.url = url
    }
}

public struct TelcoSupportTiming: Sendable, Equatable {
    public let turnRelationMs: Int
    public let sharedUnderstandingMs: Int
    public let retrievalMs: Int?
    public let policyMs: Int?
    public let composerMs: Int?
    public let totalMs: Int

    public init(
        turnRelationMs: Int,
        sharedUnderstandingMs: Int,
        retrievalMs: Int?,
        policyMs: Int?,
        composerMs: Int?,
        totalMs: Int
    ) {
        self.turnRelationMs = turnRelationMs
        self.sharedUnderstandingMs = sharedUnderstandingMs
        self.retrievalMs = retrievalMs
        self.policyMs = policyMs
        self.composerMs = composerMs
        self.totalMs = totalMs
    }
}

public enum TelcoSupportSessionError: Error, LocalizedError, Sendable {
    case dispatcherFailed(String)
    case missingDispatcherResponse

    public var errorDescription: String? {
        switch self {
        case .dispatcherFailed(let message):
            return message
        case .missingDispatcherResponse:
            return "Telco support dispatcher finished without a response."
        }
    }
}

/// Headless support-session facade for SDK-style integrations.
///
/// `TelcoSupportSession` owns conversation state and calls the same Core
/// dispatcher, RAG, policy, and composer path used by the sample chat UI.
/// The UI is intentionally outside this type: clients call `handle(_:)`,
/// render the returned value, and execute confirmed tools through their own
/// application layer.
public final actor TelcoSupportSession {
    private var blackboard: TelcoDialogueBlackboard
    private let dispatcher: TelcoChatDispatcher
    private let understandingClassifier: TelcoSharedUnderstandingClassifying?
    private let relationalStrategy: RelationalHeadsStrategy
    private let logger = Logger(
        subsystem: "ai.liquid.demos.telcotriage",
        category: "TelcoSupportSession"
    )

    public init(
        dispatcher: TelcoChatDispatcher,
        understandingClassifier: TelcoSharedUnderstandingClassifying? = nil,
        relationalStrategy: RelationalHeadsStrategy = UnavailableRelationalStrategy(),
        blackboard: TelcoDialogueBlackboard = TelcoDialogueBlackboard()
    ) {
        self.dispatcher = dispatcher
        self.understandingClassifier = understandingClassifier
        self.relationalStrategy = relationalStrategy
        self.blackboard = blackboard
    }

    public func snapshot() -> TelcoDialogueBlackboard {
        blackboard
    }

    public func reset() {
        blackboard = TelcoDialogueBlackboard()
    }

    public func recordPendingToolExecuted() {
        blackboard = TelcoDialogueBlackboardReducer.recordToolExecuted(on: blackboard)
    }

    public func recordPendingToolCancelled(reasonCode: String = "user_declined") {
        blackboard = TelcoDialogueBlackboardReducer.recordToolCancelled(
            on: blackboard,
            reasonCode: reasonCode
        )
    }

    public func handle(_ query: String) async throws -> TelcoSupportTurnResult {
        let turnStart = Date()
        let priorUserText = blackboard.lastUserTurn
        let priorAssistantText = blackboard.lastAssistantSummary
        let runtimeState = relationalRuntimeState()

        let (turnRelation, relationMs) = await classifyTurnRelation(
            query,
            priorAssistantText: priorAssistantText,
            priorUserText: priorUserText,
            runtimeState: runtimeState
        )
        let (understanding, understandingMs) = await classifySharedUnderstanding(query)

        blackboard = TelcoDialogueBlackboardReducer.reduce(
            blackboard,
            userTurn: query,
            observedRelation: turnRelation,
            understanding: understanding
        )

        let dispatchResult = try await dispatch(
            query: query,
            understanding: understanding,
            relation: blackboard.lastTurnRelation
        )

        blackboard = recordDispatch(
            dispatchResult,
            assistantText: dispatchResult.text,
            on: blackboard
        )

        let timing = makeTiming(
            dispatchResult,
            relationMs: relationMs,
            understandingMs: understandingMs,
            totalWallMs: Int(Date().timeIntervalSince(turnStart) * 1000)
        )

        return TelcoSupportTurnResult(
            query: query,
            text: dispatchResult.text,
            lane: dispatchResult.lane,
            source: dispatchResult.source,
            route: dispatchResult.composerRoute,
            citation: makeCitation(from: dispatchResult),
            deepLink: dispatchResult.deepLink,
            requiresConfirmation: dispatchResult.requiresConfirmation ?? false,
            executableToolIntent: dispatchResult.executableToolIntent,
            retrievalCandidates: dispatchResult.retrievalCandidates,
            timing: timing,
            blackboard: blackboard,
            dispatchResult: dispatchResult
        )
    }

    private func classifyTurnRelation(
        _ query: String,
        priorAssistantText: String?,
        priorUserText: String?,
        runtimeState: RelationalRuntimeState
    ) async -> (TelcoTurnRelation?, Int) {
        do {
            let outcomes = try await relationalStrategy.classifyFromText(
                currentUserQuery: query,
                priorAssistantText: priorAssistantText,
                priorUserText: priorUserText,
                runtimeState: runtimeState
            )
            return (outcomes.telcoTurnRelation?.value, Int(outcomes.runtimeMs.rounded()))
        } catch {
            logger.error("turn relation failed: \(error.localizedDescription, privacy: .public)")
            return (nil, 0)
        }
    }

    private func classifySharedUnderstanding(
        _ query: String
    ) async -> (TelcoSharedUnderstanding?, Int) {
        guard let understandingClassifier else {
            return (nil, 0)
        }
        do {
            let understanding = try await understandingClassifier.classify(query: query)
            return (understanding, Int(understanding.totalMs.rounded()))
        } catch {
            logger.error("shared understanding failed: \(error.localizedDescription, privacy: .public)")
            return (nil, 0)
        }
    }

    private func dispatch(
        query: String,
        understanding: TelcoSharedUnderstanding?,
        relation: TelcoTurnRelation?
    ) async throws -> TelcoDispatchResult {
        var finalResult: TelcoDispatchResult?
        var finalErrorMessage: String?

        for await event in dispatcher.dispatchComposer(
            query: query,
            retrievalContext: blackboard.retrievalContext,
            telcoUnderstanding: understanding,
            dialogueState: dialogueRepairState(),
            turnRelation: relation,
            policyState: policyStateSnapshot()
        ) {
            switch event {
            case .response(let result):
                finalResult = result
            case .failed(let message):
                finalErrorMessage = message
            default:
                break
            }
        }

        if let finalResult {
            return finalResult
        }
        if let finalErrorMessage {
            throw TelcoSupportSessionError.dispatcherFailed(finalErrorMessage)
        }
        throw TelcoSupportSessionError.missingDispatcherResponse
    }

    private func dialogueRepairState() -> DialogueRepairConversationState {
        DialogueRepairConversationState(
            priorPageID: blackboard.priorPageID,
            priorLinkID: blackboard.priorLinkID,
            pendingTool: blackboard.pendingToolConfirmation?.toolID,
            frustrationCount: blackboard.frustrationCount,
            pendingConfirmation: blackboard.pendingToolConfirmation != nil
        )
    }

    private func policyStateSnapshot() -> TelcoDialogueStateSnapshot {
        let activeTask = blackboard.activeTaskID
        let repairAttempts = activeTask
            .map { blackboard.failedAttemptCountByTask[$0] ?? 0 }
            ?? blackboard.frustrationCount

        return TelcoDialogueStateSnapshot(
            hasActiveTask: blackboard.priorPageID != nil,
            priorPageID: blackboard.priorPageID,
            priorLinkID: blackboard.priorLinkID,
            pendingToolID: blackboard.pendingToolConfirmation?.toolID,
            repairAttemptsOnActiveTask: repairAttempts,
            frustrationCount: blackboard.frustrationCount,
            hasPriorAssistantTurn: blackboard.lastAssistantSummary != nil,
            priorRouteWasClarify: blackboard.lastPolicyDecision?.route == .clarify
        )
    }

    private func relationalRuntimeState() -> RelationalRuntimeState {
        let pendingClarification = blackboard.pendingClarification.map { pending in
            let slots = pending.missingSlots.map(\.rawValue).sorted()
            return slots.isEmpty ? pending.source.rawValue : slots.joined(separator: ",")
        }
        let pendingTool = blackboard.pendingToolConfirmation?.toolID
        return RelationalRuntimeState(
            priorRoute: blackboard.lastPolicyDecision?.route.wireName,
            priorPageID: blackboard.priorPageID,
            priorLinkID: blackboard.priorLinkID,
            pendingTool: pendingTool,
            pendingConfirmation: pendingTool != nil,
            pendingClarification: pendingClarification,
            frustrationCount: blackboard.frustrationCount
        )
    }

    private func recordDispatch(
        _ result: TelcoDispatchResult,
        assistantText: String,
        on current: TelcoDialogueBlackboard
    ) -> TelcoDialogueBlackboard {
        let citedUnit = result.citedRAGUnit
        let policyDecision = result.composerRoute.map {
            TelcoPolicyDecision(
                route: $0,
                requiresConfirmation: result.requiresConfirmation ?? false,
                handoff: $0 == .liveAgent ? "live_agent" : nil
            )
        }

        var next = TelcoDialogueBlackboardReducer.recordRetrievalAndPolicy(
            on: current,
            retrievalCandidates: result.retrievalCandidates,
            selectedPageID: citedUnit?.pageID,
            selectedLinkID: citedUnit?.linkID,
            selectedTitle: citedUnit?.displayLabel,
            policyDecision: policyDecision
        )

        if let intent = result.executableToolIntent,
           result.requiresConfirmation == true {
            let pending = TelcoPendingTool(
                toolID: intent.toolID,
                intent: intent,
                pageID: citedUnit?.pageID,
                linkID: citedUnit?.linkID
            )
            next = TelcoDialogueBlackboardReducer.setPendingTool(pending, on: next)
        }

        return TelcoDialogueBlackboardReducer.recordResponse(assistantText, on: next)
    }

    private func makeCitation(from result: TelcoDispatchResult) -> TelcoSupportCitation? {
        if let unit = result.citedRAGUnit {
            return TelcoSupportCitation(
                label: unit.displayLabel,
                pageID: unit.pageID,
                linkID: unit.linkID,
                url: unit.canonicalURL
            )
        }
        if let chunk = result.retrievedChunk {
            return TelcoSupportCitation(
                label: chunk.title,
                pageID: chunk.pageID,
                linkID: nil,
                url: nil
            )
        }
        if let entry = result.kbEntry {
            return TelcoSupportCitation(
                label: entry.topic,
                pageID: entry.id,
                linkID: nil,
                url: entry.deepLinks.first?.url
            )
        }
        return nil
    }

    private func makeTiming(
        _ result: TelcoDispatchResult,
        relationMs: Int,
        understandingMs: Int,
        totalWallMs: Int
    ) -> TelcoSupportTiming {
        let dispatchMs = result.totalMs > 0 ? Int(result.totalMs.rounded()) : totalWallMs
        return TelcoSupportTiming(
            turnRelationMs: relationMs,
            sharedUnderstandingMs: understandingMs,
            retrievalMs: result.retrievalMs.map { Int($0.rounded()) },
            policyMs: result.routePolicyMs.map { Int($0.rounded()) },
            composerMs: result.composerMs.map { Int($0.rounded()) },
            totalMs: relationMs + understandingMs + dispatchMs
        )
    }
}
