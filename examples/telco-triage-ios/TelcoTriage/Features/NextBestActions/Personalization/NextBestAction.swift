import Foundation

/// A personalization recommendation the assistant can surface. NBAs are the
/// bridge between on-device intelligence and bottom-line impact — every
/// card here maps to a number Telco finance will actually book: upsell
/// revenue, retention-driven churn avoidance, cost savings for the
/// customer that buys loyalty, or proactive support that keeps the
/// NPS number up.
///
/// Protocol-based so new actions register without changes to the engine,
/// the Plan view, or the chat. See `NextBestActionRegistry.default`.
public protocol NextBestAction: Sendable {
    var id: String { get }
    var category: NBACategory { get }
    var headline: String { get }
    var body: String { get }
    var icon: String { get }
    var acceptLabel: String { get }
    var declineLabel: String { get }
    /// One-line value tag rendered on the card, e.g. "+$12/mo" or
    /// "Save $96/yr" or "No action needed".
    var impactTag: String? { get }

    /// True if this NBA is relevant for the given customer right now.
    func isEligible(for profile: CustomerProfile) -> Bool

    /// Higher = more prominent. Engine sorts descending. Signals that
    /// affect ARPU more (upsell accepted, retention saved) score higher.
    func priorityScore(for profile: CustomerProfile) -> Double

    /// Optional: query keywords that make this NBA contextually relevant
    /// when attached to a chat reply. Nil means no chat attachment.
    var chatAttachmentKeywords: [String]? { get }

    /// ADR-022 §4.3 Layer 4: optional hook the NBA engine consults when
    /// a chat turn has produced a `QueryUnderstanding` vector. NBAs that
    /// fire on subjective signal (frustration, missing slots, urgency)
    /// implement this; profile-only NBAs (PlanOptimize, MeshUpgrade)
    /// leave the default `false`. Pure function — no I/O, no state.
    ///
    /// Engine semantics: when this returns `true` AND `isEligible(for:)`
    /// also returns `true`, the NBA is a candidate for `bestMatchForChat`.
    /// The understanding-aware path takes precedence over the keyword
    /// path so a frustration signal beats a keyword overlap.
    ///
    /// **ADR-023 Phase 2**: the optional `conversation` parameter lets
    /// session-scoped signals (repeated live-agent requests, repeated
    /// "didn't work" continuations) drive NBA eligibility WITHOUT
    /// needing a head signal. NBAs that only care about the current
    /// turn ignore it; NBAs that escalate on accumulated friction
    /// (EscalateOnFrustrationNBA) read counter values directly.
    ///
    /// **Sendable contract**: `conversation` is a `ConversationSnapshot`
    /// (immutable value type) rather than the live `@MainActor`
    /// `ConversationState`. This keeps the protocol Sendable and lets
    /// matchers stay pure-function across actor boundaries.
    func matchesUnderstanding(
        _ understanding: QueryUnderstanding,
        lane: UnderstandingLane,
        toolIntent: ToolIntent?,
        conversation: ConversationSnapshot?
    ) -> Bool
}

/// Default: no understanding-aware match. Profile-only NBAs inherit
/// this and never fire on QueryUnderstanding signal alone.
public extension NextBestAction {
    func matchesUnderstanding(
        _ understanding: QueryUnderstanding,
        lane: UnderstandingLane,
        toolIntent: ToolIntent?,
        conversation: ConversationSnapshot?
    ) -> Bool {
        false
    }

    /// Legacy 3-arg shim. Source-compatible bridge for call sites that
    /// don't yet pass a `ConversationSnapshot`. Forwards to the canonical
    /// 4-arg method with `conversation: nil`.
    func matchesUnderstanding(
        _ understanding: QueryUnderstanding,
        lane: UnderstandingLane,
        toolIntent: ToolIntent?
    ) -> Bool {
        matchesUnderstanding(
            understanding,
            lane: lane,
            toolIntent: toolIntent,
            conversation: nil
        )
    }
}

public enum NBACategory: String, Sendable, Codable, CaseIterable {
    case upsell
    case retention
    case planOptimize
    case boltOn
    case proactiveSupport

    public var displayName: String {
        switch self {
        case .upsell: return "Upsell"
        case .retention: return "Retention"
        case .planOptimize: return "Plan fit"
        case .boltOn: return "Add-on"
        case .proactiveSupport: return "Proactive"
        }
    }
}

public struct NBAOutcome: Sendable, Equatable {
    public enum Verdict: String, Sendable { case accepted, declined, snoozed }
    public let actionID: String
    public let verdict: Verdict
    public let recordedAt: Date

    public init(actionID: String, verdict: Verdict, recordedAt: Date = Date()) {
        self.actionID = actionID
        self.verdict = verdict
        self.recordedAt = recordedAt
    }
}
