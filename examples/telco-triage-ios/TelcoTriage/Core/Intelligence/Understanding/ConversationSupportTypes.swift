import Foundation

/// Sendable point-in-time projection of chat/session state.
///
/// Kept in Core because policy, routing, and host-owned clients need the value
/// shape without importing the observable demo `ConversationState`.
public struct ConversationSnapshot: Sendable, Equatable {
    public let liveAgentRequestCount: Int
    public let didntWorkCount: Int
    public let userTurnCount: Int
    public let hasPendingClarification: Bool
    public let hasPendingToolConfirmation: Bool
    public let priorIntent: ToolIntent?
    public let priorLane: UnderstandingLane?
    public let priorAssistantText: String?
    public let hasPriorAssistantHidden: Bool
    public let hasPriorUserHidden: Bool

    public init(
        liveAgentRequestCount: Int = 0,
        didntWorkCount: Int = 0,
        userTurnCount: Int = 0,
        hasPendingClarification: Bool = false,
        hasPendingToolConfirmation: Bool = false,
        priorIntent: ToolIntent? = nil,
        priorLane: UnderstandingLane? = nil,
        priorAssistantText: String? = nil,
        hasPriorAssistantHidden: Bool = false,
        hasPriorUserHidden: Bool = false
    ) {
        self.liveAgentRequestCount = liveAgentRequestCount
        self.didntWorkCount = didntWorkCount
        self.userTurnCount = userTurnCount
        self.hasPendingClarification = hasPendingClarification
        self.hasPendingToolConfirmation = hasPendingToolConfirmation
        self.priorIntent = priorIntent
        self.priorLane = priorLane
        self.priorAssistantText = priorAssistantText
        self.hasPriorAssistantHidden = hasPriorAssistantHidden
        self.hasPriorUserHidden = hasPriorUserHidden
    }

    public static let empty = ConversationSnapshot()
}

/// A pending clarification question awaiting the user's answer.
public struct PendingClarification: Sendable, Equatable {
    public enum Source: String, Sendable, Equatable {
        case ragClarification
        case missingSlot
    }

    public let askedAt: Date
    public let source: Source
    public let intent: ToolIntent?
    public let missingSlots: Set<Slot>
    public let originalQuery: String

    public init(
        askedAt: Date,
        source: Source,
        intent: ToolIntent?,
        missingSlots: Set<Slot>,
        originalQuery: String
    ) {
        self.askedAt = askedAt
        self.source = source
        self.intent = intent
        self.missingSlots = missingSlots
        self.originalQuery = originalQuery
    }
}

/// Pure conversation helpers used by Core routing/policy and by the demo
/// observable chat state. No UI dependency, no mutable state, no model calls.
public enum ConversationStateRecorder {
    public static func isLiveAgentRequest(_ message: String) -> Bool {
        let normalized = normalize(message)
        return liveAgentPatterns.firstMatch(
            in: normalized,
            range: NSRange(normalized.startIndex..., in: normalized)
        ) != nil
    }

    public static func isDidntWorkContinuation(_ message: String) -> Bool {
        let normalized = normalize(message)
        return didntWorkPatterns.firstMatch(
            in: normalized,
            range: NSRange(normalized.startIndex..., in: normalized)
        ) != nil
    }

    public static func isGenericHelpRequest(_ message: String) -> Bool {
        let normalized = normalize(message)
        return genericHelpPatterns.firstMatch(
            in: normalized,
            range: NSRange(normalized.startIndex..., in: normalized)
        ) != nil
    }

    public static func isContextualActionRequest(_ message: String) -> Bool {
        let normalized = normalize(message)
        return contextualActionPatterns.firstMatch(
            in: normalized,
            range: NSRange(normalized.startIndex..., in: normalized)
        ) != nil
    }

    public static func isBareAffirmative(_ message: String) -> Bool {
        let normalized = normalize(message)
        return bareAffirmativePatterns.firstMatch(
            in: normalized,
            range: NSRange(normalized.startIndex..., in: normalized)
        ) != nil
    }

    public static func isBareNegative(_ message: String) -> Bool {
        let normalized = normalize(message)
        return bareNegativePatterns.firstMatch(
            in: normalized,
            range: NSRange(normalized.startIndex..., in: normalized)
        ) != nil
    }

    private static func normalize(_ message: String) -> String {
        message
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
    }

    private static let liveAgentPatterns: NSRegularExpression = {
        // swiftlint:disable:next force_try
        try! NSRegularExpression(
            pattern: #"""
            (?xi)
            \b(
                (?:live\s+)?(?:agent|representative|rep)\b
              | (?:chat\s+(?:with\s+)?(?:a\s+)?(?:agent|rep|person|human))
              | (?:talk\s+(?:to\s+|with\s+)(?:a\s+)?(?:agent|rep|person|human|real\s+person))
              | (?:speak\s+(?:to\s+|with\s+)(?:a\s+)?(?:agent|rep|person|human|real\s+person))
              | (?:connect\s+(?:me|to\s+(?:a\s+)?(?:agent|rep|person|human|real\s+person)))
              | (?:need\s+(?:a\s+)?(?:real\s+)?(?:agent|rep|person|human|representative)(?:\s+(?:for|to)\s+help)?)
              | (?:live\s+person)
              | (?:(?:some\s+one|someone|somebody|anyone)\s+(?:can\s+)?help(?:\s+me)?)
              | (?:can\s+(?:some\s+one|someone|somebody|anyone)\s+help(?:\s+me)?)
              | (?:transfer\s+me)
              | (?:get\s+me\s+(?:a\s+)?(?:agent|rep|person|human))
              | (?:real\s+person)
              | (?:customer\s+service)
            )\b
            """#,
            options: [.allowCommentsAndWhitespace]
        )
    }()

    private static let didntWorkPatterns: NSRegularExpression = {
        // swiftlint:disable:next force_try
        try! NSRegularExpression(
            pattern: #"""
            (?xi)
            \b(
                (?:didn[''’]?t|did\s+not)\s+work
              | (?:doesn[''’]?t|does\s+not)\s+work
              | (?:not\s+working)
              | (?:still\s+(?:broken|down|not\s+working|doesn[''’]?t|isn[''’]?t))
              | (?:tried\s+that)
              | (?:that\s+didn[''’]?t\s+(?:help|work))
              | (?:no\s+luck)
              | (?:same\s+(?:problem|issue))
            )\b
            """#,
            options: [.allowCommentsAndWhitespace]
        )
    }()

    private static let genericHelpPatterns: NSRegularExpression = {
        // swiftlint:disable:next force_try
        try! NSRegularExpression(
            pattern: #"""
            ^\s*(?:
                (?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?help(?:\s+me)?
                    (?:\s+(?:with\s+)?(?:this|that|it|the\s+steps?))?
              | (?:i\s+)?need\s+(?:some\s+)?help
                    (?:\s+(?:with\s+)?(?:this|that|it|the\s+steps?))?
              | (?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?
                    (?:(?:walk\s+me\s+through)|(?:guide\s+me))
                    (?:\s+(?:this|that|it|the\s+steps?))?
              | (?:what\s+now|now\s+what)
            )\s*[?.!]?\s*$
            """#,
            options: [.allowCommentsAndWhitespace, .caseInsensitive]
        )
    }()

    private static let contextualActionPatterns: NSRegularExpression = {
        // swiftlint:disable:next force_try
        try! NSRegularExpression(
            pattern: #"""
            ^\s*(?:
                (?:(?:can|could|would|will)\s+you\s+)?
                    (?:(?:do|perform|handle|run|start|take\s+care\s+of)\s+(?:it|this|that))
                    (?:\s+for\s+me)?
              | (?:please\s+)?(?:do|perform|handle|run|start)\s+(?:it|this|that)
                    (?:\s+for\s+me)?
              | (?:please\s+)?(?:take\s+care\s+of)\s+(?:it|this|that)
                    (?:\s+for\s+me)?
            )\s*[?.!]?\s*$
            """#,
            options: [.allowCommentsAndWhitespace, .caseInsensitive]
        )
    }()

    private static let bareAffirmativePatterns: NSRegularExpression = {
        // swiftlint:disable:next force_try
        try! NSRegularExpression(
            pattern: #"""
            ^\s*(?:
                yes|yeah|yep|yup|sure|ok|okay|k|fine
              | (?:go\s+ahead)|(?:do\s+it)|(?:please\s+do)
              | (?:sounds\s+good)|(?:that\s+works)
            )\s*[.!]?\s*$
            """#,
            options: [.allowCommentsAndWhitespace, .caseInsensitive]
        )
    }()

    private static let bareNegativePatterns: NSRegularExpression = {
        // swiftlint:disable:next force_try
        try! NSRegularExpression(
            pattern: #"""
            ^\s*(?:
                no|nope|nah|cancel|stop
              | (?:do\s+not)|don[''’]?t|dont
              | (?:never\s+mind)|(?:no\s+thanks)|(?:no\s+thank\s+you)
            )\s*[.!]?\s*$
            """#,
            options: [.allowCommentsAndWhitespace, .caseInsensitive]
        )
    }()
}
