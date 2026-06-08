import Foundation

/// Routing outcome for a single conversation turn.
///
/// Six customer-facing lanes plus the `clarification` multi-turn case.
/// See `docs/architecture-decisions/ADR-021-telco-home-internet-rag-assistant.md` §3.
///
/// Parallels the `Lane` IntEnum in `scripts/telco/schemas.py`.
public enum TelcoLane: Int, Sendable, CaseIterable, Codable {
    /// User said hi / hello / thanks. Friendly template response.
    case greeting = 0

    /// In-scope query with a RAG-grounded answer. Generates a
    /// `Go to [Link](telcohome://...) > Step > Step` response via the
    /// 1.2B LM + GBNF grammar.
    case ragStepByStep = 1

    /// Account- or billing-related question routed to an in-app
    /// deep-link rather than a generated answer. No LLM near financial
    /// data (ADR-021 §3.2).
    case navOnlyDeeplink = 2

    /// In-scope but the RAG corpus has no answer. Offers escalation
    /// to a Telco support agent.
    case unknownFeature = 3

    /// Explicit live-agent request or operational urgency. Triggers
    /// the cloud escalation flow (ADR-021 §3.1).
    case liveAgentEscalation = 4

    /// Out-of-scope query (weather, prompt injection, generic chitchat).
    /// Telco's canonical refusal template.
    case oosRefusal = 5

    /// Multi-turn special case: retrieval is ambiguous between two or
    /// more pages. The router emits a clarification prompt; the next
    /// turn resolves the choice via shared state (ADR-021 §3.3).
    case clarification = 6

    /// Stable wire string used in logs and metadata.
    public var wireName: String {
        switch self {
        case .greeting: return "greeting"
        case .ragStepByStep: return "rag_step_by_step"
        case .navOnlyDeeplink: return "nav_only_deeplink"
        case .unknownFeature: return "unknown_feature"
        case .liveAgentEscalation: return "live_agent_escalation"
        case .oosRefusal: return "oos_refusal"
        case .clarification: return "clarification"
        }
    }

    /// True when this lane requires the 1.2B generator + GBNF stage to
    /// run. All other lanes use template responses and skip generation
    /// entirely (ADR-021 §3, lane latency table).
    public var requiresGeneration: Bool {
        self == .ragStepByStep
    }
}
