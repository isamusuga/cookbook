import Foundation

/// Layer 4 of ADR-022 — NBA cards that consume the `QueryUnderstanding`
/// vector. These are the surfaces that pay back the ADR's "free
/// leverage" argument: the same forward pass that picked the lane also
/// surfaced emotional state + slot completeness; the NBA layer turns
/// those signals into user-visible behaviour without a second LFM call.
///
/// **Why protocols, not closures**: keeps the NBA framework's
/// observability model intact — every NBA is a `NextBestAction`, listed
/// by the engine, scored, tracked through `NBAOutcome`. The
/// understanding-aware path is just an additional eligibility hook,
/// not a parallel architecture.

// MARK: - Escalate on frustration

/// Surfaces a "Talk to a human?" chip beneath the assistant reply when
/// the `emotional_state` head reports frustrated or urgent AND the
/// turn isn't ALREADY heading to a live agent. The NBA gives the user
/// an explicit affordance to escalate — the router itself does NOT
/// silently change the lane on emotional signal alone (per ADR-022
/// §4.3 principle #1).
///
/// **Eligibility logic** (see `matchesUnderstanding`):
///  - `emotionalState ∈ {frustrated, urgent}` — the head flagged
///    affective stress, OR
///  - `conversation.liveAgentRequestCount ≥ 2` — user has explicitly
///    asked for a human twice in this session (ADR-023 Phase 2 — the
///    counter-based path catches the calm-but-repeating power user
///    that the emotional head misses), OR
///  - `conversation.didntWorkCount ≥ 2` — user has reported the
///    suggestion didn't work twice (same accumulator pattern; affect
///    may be neutral but the friction is real).
///  - `lane != .liveAgentEscalation` — we don't double-offer when the
///    router already escalated (that'd just look broken).
///  - Profile gate (`isEligible(for:)`) is always true — frustration
///    is universal across the customer base; the engine still requires
///    `isEligible` to return true alongside.
public struct EscalateOnFrustrationNBA: NextBestAction {
    public let id = "escalate-on-frustration"
    public let category: NBACategory = .proactiveSupport
    public let icon = "person.wave.2.fill"
    public let acceptLabel = "Talk to a person"
    public let declineLabel = "Stay self-service"
    public let chatAttachmentKeywords: [String]? = nil  // signal-driven, not keyword-driven

    public init() {}

    public var headline: String { "Want a real person on this?" }
    public var body: String {
        "If this isn't moving fast enough, I can hand you to a live Telco agent — they'll see this conversation when they pick up."
    }
    public var impactTag: String? { "Live escalation" }

    public func isEligible(for profile: CustomerProfile) -> Bool {
        // Profile-level filter: only customers who have live-agent
        // routing enabled (always-on for this POC; future versions
        // may gate by tier / outage status). Keep the function pure
        // and always-true so the understanding signal is what fires
        // the card.
        _ = profile
        return true
    }

    public func priorityScore(for profile: CustomerProfile) -> Double {
        // Engine sorts descending — escalation chips on frustrated
        // turns should outrank upsells.
        _ = profile
        return 0.95
    }

    public func matchesUnderstanding(
        _ understanding: QueryUnderstanding,
        lane: UnderstandingLane,
        toolIntent: ToolIntent?,
        conversation: ConversationSnapshot?
    ) -> Bool {
        _ = toolIntent
        // Don't offer the chip when the router already escalated; the
        // live-agent template is the answer in that case. Guard first
        // — both the head-driven AND counter-driven paths bail on
        // this lane.
        if case .telco(.liveAgentEscalation) = lane {
            return false
        }

        // Path A — head signal (the original ADR-022 trigger). Fires
        // when the trained emotional_state head reports frustrated /
        // urgent. Independent of conversation state.
        if let emotional = understanding.emotionalState?.value,
           emotional.warrantsProactiveEscalation {
            return true
        }

        // Path B — counter signal (ADR-023 Phase 2). Fires when the
        // user has repeated friction in this session, regardless of
        // affect. Catches the ~20% of Telco conversations where the
        // escalation request is calm but persistent.
        if let conversation,
           conversation.liveAgentRequestCount >= Self.liveAgentEscalationThreshold {
            return true
        }
        if let conversation,
           conversation.didntWorkCount >= Self.didntWorkEscalationThreshold {
            return true
        }

        return false
    }

    /// Threshold for counter-driven escalation. Two requests is the
    /// floor — single live-agent mentions can be exploratory ("can I
    /// also talk to a person?") and don't warrant pushing the chip.
    /// The second mention is the actual escalation signal.
    public static let liveAgentEscalationThreshold: Int = 2

    /// Threshold for "didn't work" escalation. Same floor as above —
    /// one report is the first natural follow-up; the second is the
    /// continuing-friction signal.
    public static let didntWorkEscalationThreshold: Int = 2
}

// MARK: - Clarify missing slot

/// Surfaces a focused clarification chip when the `slot_completeness`
/// head reports that a required slot for the chosen tool is missing.
/// The chip carries a per-tool clarification question (see
/// `ToolSlotRequirements.clarificationQuestion`) so the user can
/// answer in one tap-and-type round-trip instead of editing tool
/// arguments after the fact.
///
/// **Eligibility logic** (see `matchesUnderstanding`):
///  - Lane is `.toolAction` — only relevant when the workflow is
///    about to fire a tool.
///  - `toolIntent != nil` — the selector picked a tool (if not, the
///    workflow already falls back to a KB lookup).
///  - `slot_completeness` head present (v2 bundled) — without the
///    head we don't know what's missing.
///  - `ToolSlotRequirements.needsClarification(for:given:)` returns
///    true — the tool's required slots are not all flagged.
public struct ClarifyMissingSlotNBA: NextBestAction {
    public let id = "clarify-missing-slot"
    public let category: NBACategory = .proactiveSupport
    public let icon = "questionmark.bubble.fill"
    public let acceptLabel = "Answer"
    public let declineLabel = "Skip"
    public let chatAttachmentKeywords: [String]? = nil

    public init() {}

    public var headline: String { "Quick question to get this right" }
    public var body: String {
        // The actual clarification question is per-tool; the engine
        // doesn't know which tool when constructing the card list, so
        // we render a generic body here. ChatViewModel substitutes
        // the per-tool question via `ToolSlotRequirements.clarificationQuestion`
        // when rendering the attached chip.
        "I have most of what I need, but one piece is unclear. Mind clarifying so I don't pause the wrong thing?"
    }
    public var impactTag: String? { "Avoid wrong action" }

    public func isEligible(for profile: CustomerProfile) -> Bool {
        _ = profile
        return true
    }

    public func priorityScore(for profile: CustomerProfile) -> Double {
        _ = profile
        // Slightly under escalation — frustration is a stronger signal
        // than a missing slot (slot can be filled by retry, frustration
        // can't).
        return 0.90
    }

    public func matchesUnderstanding(
        _ understanding: QueryUnderstanding,
        lane: UnderstandingLane,
        toolIntent: ToolIntent?,
        conversation: ConversationSnapshot?
    ) -> Bool {
        _ = conversation  // unused — slot signal is per-turn, not session-scoped
        // Only fires on the tool-action lane.
        guard case .toolAction = lane else { return false }
        guard let intent = toolIntent else { return false }
        // Pure-function decision delegated to the shared registry.
        return ToolSlotRequirements.needsClarification(
            for: intent,
            given: understanding.slotCompleteness?.value
        )
    }
}
