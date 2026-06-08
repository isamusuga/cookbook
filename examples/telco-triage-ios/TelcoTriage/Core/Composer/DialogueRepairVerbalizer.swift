import Foundation
import os.log

public enum DialogueRepairAct: String, Codable, Sendable, Equatable {
    case clarification
    case navigationFollowup = "navigation_followup"
    case stepFocus = "step_focus"
    case failedAttempt = "failed_attempt"
    case repairCannotFind = "repair_cannot_find"
    case topicPivotRepair = "topic_pivot_repair"
    case noAnswerRepair = "no_answer_repair"
}

public enum DialogueRepairHandoff: String, Codable, Sendable, Equatable {
    case none
    case cloud
    case human
}

public struct DialogueRepairConversationState: Codable, Sendable, Equatable {
    public let priorPageID: String?
    public let priorLinkID: String?
    public let pendingTool: String?
    public let frustrationCount: Int
    public let pendingConfirmation: Bool

    public init(
        priorPageID: String? = nil,
        priorLinkID: String? = nil,
        pendingTool: String? = nil,
        frustrationCount: Int = 0,
        pendingConfirmation: Bool = false
    ) {
        self.priorPageID = priorPageID
        self.priorLinkID = priorLinkID
        self.pendingTool = pendingTool
        self.frustrationCount = frustrationCount
        self.pendingConfirmation = pendingConfirmation
    }

    enum CodingKeys: String, CodingKey {
        case priorPageID = "prior_page_id"
        case priorLinkID = "prior_link_id"
        case pendingTool = "pending_tool"
        case frustrationCount = "frustration_count"
        case pendingConfirmation = "pending_confirmation"
    }

    public static let empty = DialogueRepairConversationState()
}

public struct DialogueRepairVerbalizerInput: Sendable, Equatable {
    public let currentUserTurn: String
    public let priorAssistantText: String?
    public let conversationState: DialogueRepairConversationState
    public let understanding: TelcoSharedUnderstanding?
    public let evidence: RAGUnit?
    public let route: ComposerRoute
    public let act: DialogueRepairAct
    public let handoff: DialogueRepairHandoff
    public let requiresConfirmation: Bool

    public init(
        currentUserTurn: String,
        priorAssistantText: String?,
        conversationState: DialogueRepairConversationState,
        understanding: TelcoSharedUnderstanding?,
        evidence: RAGUnit?,
        route: ComposerRoute,
        act: DialogueRepairAct,
        handoff: DialogueRepairHandoff,
        requiresConfirmation: Bool
    ) {
        self.currentUserTurn = currentUserTurn
        self.priorAssistantText = priorAssistantText
        self.conversationState = conversationState
        self.understanding = understanding
        self.evidence = evidence
        self.route = route
        self.act = act
        self.handoff = handoff
        self.requiresConfirmation = requiresConfirmation
    }
}

public struct DialogueRepairVerbalizerResult: Sendable, Equatable {
    public let text: String
    public let rawOutput: String?
    public let usedFallback: Bool
    public let extractionMode: String
    public let latencyMs: Double

    public init(
        text: String,
        rawOutput: String?,
        usedFallback: Bool,
        extractionMode: String,
        latencyMs: Double
    ) {
        self.text = text
        self.rawOutput = rawOutput
        self.usedFallback = usedFallback
        self.extractionMode = extractionMode
        self.latencyMs = latencyMs
    }
}

public protocol DialogueRepairVerbalizing: Sendable {
    func verbalize(_ input: DialogueRepairVerbalizerInput) async -> DialogueRepairVerbalizerResult
}

/// Response-only wrapper for the Telco dialogue-repair v4 LoRA.
///
/// First-principles contract:
/// - the model may propose only customer-facing response text;
/// - Swift owns route, source ids, links, handoff, confirmation, and unsafe-action;
/// - invalid or over-broad model text falls back to a deterministic repair template.
public final class DialogueRepairVerbalizer: DialogueRepairVerbalizing, @unchecked Sendable {
    private let backend: AdapterInferenceBackend
    private let adapterPath: String
    private let logger = Logger(
        subsystem: "ai.liquid.demos.telcotriage",
        category: "DialogueRepairV4"
    )

    public init(backend: AdapterInferenceBackend, adapterPath: String) {
        self.backend = backend
        self.adapterPath = adapterPath
    }

    public static func bundled(
        backend: AdapterInferenceBackend,
        bundle: Bundle = .main
    ) -> DialogueRepairVerbalizer? {
        guard let adapter = TelcoModelBundle.dialogueRepairV4AdapterPath(in: bundle) else {
            return nil
        }
        return DialogueRepairVerbalizer(backend: backend, adapterPath: adapter)
    }

    public func verbalize(_ input: DialogueRepairVerbalizerInput) async -> DialogueRepairVerbalizerResult {
        let started = CFAbsoluteTimeGetCurrent()
        let payload: String
        do {
            payload = try Self.payloadJSON(for: input)
        } catch {
            let fallback = Self.runtimeFallback(for: input)
            return DialogueRepairVerbalizerResult(
                text: fallback,
                rawOutput: nil,
                usedFallback: true,
                extractionMode: "payload_encode_failed",
                latencyMs: Self.elapsed(started)
            )
        }

        let raw: String
        do {
            raw = try await backend.generate(
                messages: [.user(payload)],
                adapterPath: adapterPath,
                maxTokens: 96,
                stopSequences: ["<|tool_call_end|>", "<|im_end|>"]
            )
        } catch {
            logger.error("dialogue repair v4 failed: \(error.localizedDescription, privacy: .public)")
            let fallback = Self.runtimeFallback(for: input)
            return DialogueRepairVerbalizerResult(
                text: fallback,
                rawOutput: nil,
                usedFallback: true,
                extractionMode: "generation_failed",
                latencyMs: Self.elapsed(started)
            )
        }

        let (candidate, mode) = Self.extractResponseCandidate(from: raw)
        let validated = candidate.flatMap { Self.validatedResponse($0, for: input) }
        let text: String
        let usedFallback: Bool
        if let validated {
            text = Self.withRuntimeOwnedConfirmation(validated, input: input)
            usedFallback = false
        } else {
            text = Self.runtimeFallback(for: input)
            usedFallback = true
        }

        logger.info(
            "dialogue_repair_v4 act=\(input.act.rawValue, privacy: .public) mode=\(mode, privacy: .public) fallback=\(usedFallback, privacy: .public) ms=\(String(format: "%.0f", Self.elapsed(started)), privacy: .public)"
        )
        return DialogueRepairVerbalizerResult(
            text: text,
            rawOutput: raw,
            usedFallback: usedFallback,
            extractionMode: mode,
            latencyMs: Self.elapsed(started)
        )
    }

    private static func elapsed(_ start: CFAbsoluteTime) -> Double {
        (CFAbsoluteTimeGetCurrent() - start) * 1000
    }

    private static func payloadJSON(for input: DialogueRepairVerbalizerInput) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(Payload(input: input))
        return String(decoding: data, as: UTF8.self)
    }

    private static func extractResponseCandidate(from raw: String) -> (String?, String) {
        if let response = quotedField("response", in: raw) {
            return (response.trimmingCharacters(in: .whitespacesAndNewlines), "response_field")
        }

        return (nil, "missing_response_field")
    }

    private static func quotedField(_ field: String, in text: String) -> String? {
        let pattern = "(?s)\\b\(NSRegularExpression.escapedPattern(for: field))\\s*=\\s*\"((?:\\\\.|[^\"\\\\])*)\""
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let match = regex.firstMatch(in: text, range: range),
              match.numberOfRanges >= 2,
              let fieldRange = Range(match.range(at: 1), in: text) else {
            return nil
        }
        return unescapeModelString(String(text[fieldRange]))
    }

    private static func unescapeModelString(_ value: String) -> String {
        let json = "\"\(value)\""
        if let data = json.data(using: .utf8),
           let decoded = try? JSONDecoder().decode(String.self, from: data) {
            return decoded
        }
        return value
            .replacingOccurrences(of: "\\\"", with: "\"")
            .replacingOccurrences(of: "\\\\", with: "\\")
    }

    private static func validatedResponse(
        _ candidate: String,
        for input: DialogueRepairVerbalizerInput
    ) -> String? {
        let text = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        guard !text.hasPrefix("[") && !text.hasPrefix("{") && !text.hasPrefix("(") else {
            return nil
        }
        let wordCount = text.split(whereSeparator: \.isWhitespace).count
        guard wordCount <= 55 else { return nil }

        let lower = text.lowercased()
        let forbiddenAlways = [
            "telcohome://", "http://", "https://", "tel://", "source_page_id",
            "source_link_id", "unsafe_action", "telco.dialogue_reply",
            "<|tool_call", "](", "forbids_contact", "unsafe_contact",
            "unsafe_", "turn_style", "language=", "act=", "handoff="
        ]
        guard !forbiddenAlways.contains(where: lower.contains) else { return nil }
        guard lower.range(
            of: #"\b[a-z_][a-z0-9_]*\s*\([^)]*="#,
            options: .regularExpression
        ) == nil else {
            return nil
        }
        guard lower.range(of: #"\b\d{2}\.\d{2}\b"#, options: .regularExpression) == nil else {
            return nil
        }

        if input.handoff == .none {
            let noHandoffWords = ["support agent", "representative", "call us", "cloud", "live account"]
            guard !noHandoffWords.contains(where: lower.contains) else { return nil }
        }

        if input.requiresConfirmation {
            let executionClaims = [
                "i restarted", "i've restarted", "i have restarted",
                "i ran", "i've run", "done", "completed", "successfully"
            ]
            guard !executionClaims.contains(where: lower.contains) else { return nil }
        }

        return text
    }

    private static func withRuntimeOwnedConfirmation(
        _ response: String,
        input: DialogueRepairVerbalizerInput
    ) -> String {
        guard input.requiresConfirmation else { return response }
        let lower = response.lowercased()
        guard !lower.contains("reply 'yes'") && !lower.contains("reply yes") else {
            return response
        }
        return "\(response) Reply 'yes' to confirm."
    }

    public static func runtimeFallback(for input: DialogueRepairVerbalizerInput) -> String {
        let title = input.evidence?.displayLabel ?? input.evidence?.title ?? "that setting"
        switch input.handoff {
        case .human:
            return "A support agent is the right next step for \(title)."
        case .cloud:
            return "\(title) needs live account or network data, so open the app to continue."
        case .none:
            break
        }

        if input.requiresConfirmation {
            return "I can help with \(title). Reply 'yes' to confirm."
        }

        switch input.act {
        case .clarification:
            if input.understanding?.missingSlots.activeLabels.contains(.missingDevice) == true {
                return "Which device are you trying to manage?"
            }
            if input.understanding?.missingSlots.activeLabels.contains(.missingLocation) == true {
                return "Which room or location are you working on?"
            }
            return "What would you like help with?"
        case .failedAttempt, .repairCannotFind:
            if let step = input.evidence?.steps.first {
                return "Let's stay on \(title). Start with: \(step.trimmingCharacters(in: CharacterSet(charactersIn: ".")))."
            }
            return "What do you see instead of \(title)?"
        case .stepFocus:
            let steps = input.evidence?.steps ?? []
            if !steps.isEmpty {
                let index = min(1, steps.count - 1)
                return "The step to focus on is: \(steps[index].trimmingCharacters(in: CharacterSet(charactersIn: ".")))."
            }
            return "Which step are you trying to complete?"
        case .navigationFollowup:
            if let step = input.evidence?.steps.first {
                return "Start here: \(step.trimmingCharacters(in: CharacterSet(charactersIn: ".")))."
            }
            return "I found the local guidance for \(title)."
        case .topicPivotRepair:
            return "I switched to \(title)."
        case .noAnswerRepair:
            return "I do not have enough local guidance for that. What are you trying to change or fix?"
        }
    }

    private struct Payload: Encodable {
        let currentUserTurn: String
        let history: [HistoryTurn]
        let conversationState: DialogueRepairConversationState
        let understanding: UnderstandingPayload
        let responseConstraints: ResponseConstraintsPayload
        let selectedEvidence: EvidencePayload
        let policyDecision: PolicyPayload

        init(input: DialogueRepairVerbalizerInput) {
            currentUserTurn = input.currentUserTurn
            history = input.priorAssistantText.map { [HistoryTurn(role: "assistant", text: $0)] } ?? []
            conversationState = input.conversationState
            understanding = UnderstandingPayload(input.understanding)
            responseConstraints = ResponseConstraintsPayload(input: input)
            selectedEvidence = EvidencePayload(input.evidence)
            policyDecision = PolicyPayload(
                act: input.act,
                handoff: input.handoff,
                requiresConfirmation: input.requiresConfirmation
            )
        }

        enum CodingKeys: String, CodingKey {
            case currentUserTurn = "current_user_turn"
            case history
            case conversationState = "conversation_state"
            case understanding
            case responseConstraints = "response_constraints"
            case selectedEvidence = "selected_evidence"
            case policyDecision = "policy_decision"
        }
    }

    private struct HistoryTurn: Encodable {
        let role: String
        let text: String
    }

    private struct UnderstandingPayload: Encodable {
        let supportIntent: String
        let routingLane: String
        let requiredTool: String
        let cloudRequirements: [String]
        let escalationRisk: String
        let transcriptQuality: String
        let missingSlots: [String]

        init(_ understanding: TelcoSharedUnderstanding?) {
            supportIntent = understanding?.supportIntent.label.rawValue ?? "troubleshooting"
            routingLane = understanding?.routingLane.label.rawValue ?? "local_answer"
            requiredTool = understanding?.requiredTool.label.rawValue ?? "no_tool"
            cloudRequirements = understanding?.cloudRequirements.activeLabels.map(\.rawValue) ?? []
            escalationRisk = understanding?.escalationRisk.label.rawValue ?? "low"
            transcriptQuality = understanding?.transcriptQuality.label.rawValue ?? "clean"
            missingSlots = understanding?.missingSlots.activeLabels.map(\.rawValue) ?? []
        }

        enum CodingKeys: String, CodingKey {
            case supportIntent = "support_intent"
            case routingLane = "routing_lane"
            case requiredTool = "required_tool"
            case cloudRequirements = "cloud_requirements"
            case escalationRisk = "escalation_risk"
            case transcriptQuality = "transcript_quality"
            case missingSlots = "missing_slots"
        }
    }

    private struct ResponseConstraintsPayload: Encodable {
        let mustAskConfirmation: Bool
        let mustNotGiveProceduralGuidance: Bool
        let mustNotExecuteTool: Bool
        let handoffStyle: String
        let requiredHandoffPhrase: String?
        let maxWords: Int
        let forbiddenTopics: [String]

        init(input: DialogueRepairVerbalizerInput) {
            mustAskConfirmation = input.act == .clarification ||
                input.understanding?.needsClarification == true
            mustNotGiveProceduralGuidance = input.evidence == nil
            mustNotExecuteTool = true
            handoffStyle = input.handoff.rawValue
            requiredHandoffPhrase = switch input.handoff {
            case .human: "support agent is the right next step"
            case .cloud: "needs live account or network data"
            case .none: nil
            }
            maxWords = 55
            var topics = ["links", "markdown", "internal_page_ids", "tool_execution_claims", "account_facts"]
            if input.handoff == .none {
                topics.append(contentsOf: ["agents", "cloud", "live_account_data", "handoff_claims"])
            }
            forbiddenTopics = topics
        }

        enum CodingKeys: String, CodingKey {
            case mustAskConfirmation = "must_ask_confirmation"
            case mustNotGiveProceduralGuidance = "must_not_give_procedural_guidance"
            case mustNotExecuteTool = "must_not_execute_tool"
            case handoffStyle = "handoff_style"
            case requiredHandoffPhrase = "required_handoff_phrase"
            case maxWords = "max_words"
            case forbiddenTopics = "forbidden_topics"
        }
    }

    private struct EvidencePayload: Encodable {
        let pageID: String?
        let linkID: String?
        let title: String?
        let canonicalURL: String?
        let section: String?
        let actionAffordance: String?
        let steps: [String]
        let facts: [String]

        init(_ evidence: RAGUnit?) {
            pageID = evidence?.pageID
            linkID = evidence?.linkID
            title = evidence?.displayLabel ?? evidence?.title
            canonicalURL = evidence?.canonicalURL
            section = evidence?.section
            actionAffordance = evidence?.actionAffordance
            steps = evidence?.steps ?? []
            facts = evidence.map { Self.facts(from: $0) } ?? []
        }

        private static func facts(from evidence: RAGUnit) -> [String] {
            let sentences = evidence.body
                .components(separatedBy: CharacterSet(charactersIn: ".\n"))
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty && $0.count <= 180 }
            return Array(sentences.prefix(3))
        }

        enum CodingKeys: String, CodingKey {
            case pageID = "page_id"
            case linkID = "link_id"
            case title
            case canonicalURL = "canonical_url"
            case section
            case actionAffordance = "action_affordance"
            case steps
            case facts
        }
    }

    private struct PolicyPayload: Encodable {
        let act: DialogueRepairAct
        let handoff: DialogueRepairHandoff
        let requiresConfirmation: Bool

        enum CodingKeys: String, CodingKey {
            case act
            case handoff
            case requiresConfirmation = "requires_confirmation"
        }
    }
}

public enum DialogueRepairActDeriver {
    public static func derive(
        query: String,
        route: ComposerRoute,
        evidence: RAGUnit?,
        retrievalContext: RetrievalContext,
        understanding: TelcoSharedUnderstanding?
    ) -> DialogueRepairAct? {
        let lowered = query.lowercased()
        let hasPriorAssistant = retrievalContext.priorAssistantText
            .map { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty } ?? false
        let hasPriorEvidence = hasPriorAssistant && retrievalContext.priorPageID != nil

        if route == .clarify || understanding?.needsClarification == true {
            return .clarification
        }
        if route == .liveAgent || route == .accountNav ||
            route == .noRagAnswer || route == .outOfScope {
            return .noAnswerRepair
        }
        if ConversationStateRecorder.isDidntWorkContinuation(query) ||
            lowered.contains("not starting") ||
            lowered.contains("won't start") ||
            lowered.contains("doesn't start") {
            return hasPriorEvidence ? .failedAttempt : nil
        }
        if lowered.contains("can't find") ||
            lowered.contains("cannot find") ||
            lowered.contains("not able to find") ||
            lowered.contains("where is") ||
            lowered.contains("where's") ||
            lowered.contains("which button") {
            return hasPriorEvidence ? .repairCannotFind : nil
        }

        guard hasPriorEvidence, let evidence, retrievalContext.priorPageID == evidence.pageID else {
            return nil
        }
        if lowered == "how" || lowered == "how?" ||
            lowered.contains("how do i") ||
            lowered.contains("tell me how") {
            return .navigationFollowup
        }
        if lowered.contains("button") ||
            lowered.contains("step") ||
            lowered.contains("tile") {
            return .stepFocus
        }
        return nil
    }

    public static func handoff(for route: ComposerRoute) -> DialogueRepairHandoff {
        switch route {
        case .liveAgent: return .human
        case .accountNav: return .cloud
        default: return .none
        }
    }
}
