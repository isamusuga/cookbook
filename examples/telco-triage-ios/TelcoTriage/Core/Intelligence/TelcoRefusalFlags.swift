import Foundation

/// 3-flag multi-label sigmoid output for the Telco RAG assistant.
///
/// Each flag is independent — multiple flags can fire on the same query.
/// The deterministic router (`TelcoRagRouter`) combines them with the
/// topic-gate and intent signals to pick a `TelcoLane`. See
/// `docs/architecture-decisions/ADR-021-telco-home-internet-rag-assistant.md` §2.1, §3.
///
/// Parallels the `RefusalFlags` dataclass in `scripts/telco/schemas.py`.
public struct TelcoRefusalFlags: Sendable, Equatable, Hashable, Codable {
    /// True when the RAG corpus contains an answer the model is confident
    /// it can ground a response against. Drives the RAG step-by-step lane.
    public let hasRagAnswer: Bool

    /// True when the query is account- or billing-related and should be
    /// answered via an in-app deep-link rather than a generated response.
    /// Drives the nav-only deep-link lane (ADR-021 §3.2).
    public let navigationOnly: Bool

    /// True when the query matches a live-agent escalation criterion
    /// (explicit human request, outage, security incident, technician
    /// scheduling, etc.). See ADR-021 §3.1 for the full trigger list.
    public let liveAgentTrigger: Bool

    public init(
        hasRagAnswer: Bool,
        navigationOnly: Bool,
        liveAgentTrigger: Bool
    ) {
        self.hasRagAnswer = hasRagAnswer
        self.navigationOnly = navigationOnly
        self.liveAgentTrigger = liveAgentTrigger
    }

    /// Convenience: all flags off. Useful for greeting / OOS lanes where
    /// the sigmoid head is never queried.
    public static let none = TelcoRefusalFlags(
        hasRagAnswer: false,
        navigationOnly: false,
        liveAgentTrigger: false
    )
}
