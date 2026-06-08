import Foundation

/// 3-class emotional-state classification produced by the v2 understanding
/// layer (ADR-022 §4.3). Drives proactive escalation NBAs and engineering
/// analytics; never used by the routing decision tree itself.
///
/// **Schema commitments** (frozen — must not be re-ordered without a
/// head re-export; the label2id map in `telco_emotional_state_label_schema.json`
/// is the wire contract):
///
///  - `neutral`     index 0 — calm, informational, transactional. Default.
///  - `frustrated`  index 1 — multiple turns on the same issue, "this isn't
///                            working", explicit dissatisfaction. NBA layer
///                            offers "Talk to a human?" chip.
///  - `urgent`      index 2 — outage language, ALL CAPS, "ASAP", deadline.
///                            Escalation policy upgrades to live-agent lane
///                            even if the trained `live_agent_trigger` flag
///                            is not set, per ADR-022 §4.3 design principle
///                            on workflow-level upgrades.
///
/// Why a head and not a regex: subjective signals like frustration don't
/// reduce to keyword patterns. "ARE YOU KIDDING ME" and "this is the third
/// time I've called" are both `frustrated` but share no lexical features.
/// The 50-conversation production probe set (Chintan-labeled) is the
/// ground-truth seed; teachers (GPT-4o + Sonnet) expand to ~500/class.
public enum EmotionalState: Int, Sendable, Equatable, Codable, CaseIterable {
    case neutral = 0
    case frustrated = 1
    case urgent = 2

    /// Stable wire string for logs, metadata, training data, and the
    /// label2id map. NEVER changed without a head re-export.
    public var wireName: String {
        switch self {
        case .neutral:    return "neutral"
        case .frustrated: return "frustrated"
        case .urgent:     return "urgent"
        }
    }

    /// Human-readable label for the engineering-mode trace card.
    public var displayName: String {
        switch self {
        case .neutral:    return "Neutral"
        case .frustrated: return "Frustrated"
        case .urgent:     return "Urgent"
        }
    }

    /// True for states that should bias the workflow toward proactive
    /// escalation. The router itself is unaware — only the NBA layer
    /// and the workflow upgrade in `TelcoUnderstandingRouter.escalationOverride`
    /// consult this.
    public var warrantsProactiveEscalation: Bool {
        switch self {
        case .neutral:                return false
        case .frustrated, .urgent:    return true
        }
    }

    /// Lookup by wire string (matches the head's id2label JSON values).
    /// Defensive default — unknown labels collapse to `.neutral` rather
    /// than crashing on a meta JSON drift between train + ship.
    public static func from(wireName: String) -> EmotionalState {
        EmotionalState.allCases.first { $0.wireName == wireName } ?? .neutral
    }
}
