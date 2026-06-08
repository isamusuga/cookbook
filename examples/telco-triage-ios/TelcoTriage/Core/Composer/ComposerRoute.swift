import Foundation

/// The 9 routes the Step 5 answer-layer evaluation considers. Mirrors
/// `ALL_ROUTES` in `scripts/telco/answer_composer.py`.
///
/// The dispatcher derives the route from `(queryMood, retrievedUnit,
/// ToolRegistry)` via guardrail #3 in the Step 6 plan — the composer
/// trusts whatever route it's handed and never re-routes.
public enum ComposerRoute: String, Sendable, Equatable, CaseIterable {
    case ragAnswer = "rag_answer"
    case answerPlusAction = "answer_plus_action"
    case toolAction = "tool_action"
    case accountNav = "account_nav"
    case liveAgent = "live_agent"
    case clarify
    case outOfScope = "out_of_scope"
    case noRagAnswer = "no_rag_answer"
    case greeting

    /// Stable wire name for telemetry / trace logs. Matches the
    /// `expected_route` field in `golden_conversations_v2.jsonl`.
    public var wireName: String { rawValue }

    /// True for routes that need a selected RAG unit to compose a
    /// grounded response. `ragAnswer`, `answerPlusAction`,
    /// `toolAction`. Everything else uses fixed templates.
    public var requiresEvidence: Bool {
        switch self {
        case .ragAnswer, .answerPlusAction, .toolAction: return true
        default: return false
        }
    }
}
