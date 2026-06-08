import Foundation

/// The Layer 1 output of ADR-022's understanding pipeline.
///
/// One LFM forward pass (≈150 ms) over the `telco-shared-clf-v2` adapter
/// fans out into N classifier heads. Each head is a tiny linear projection
/// over the same hidden state — adding the 5th head on the same forward
/// pass costs ~1 ms vs ~150 ms for a fresh forward pass. This is the
/// "free leverage" §3 of the ADR calls out.
///
/// Every head output is **optional**. Production rolls v2 heads in
/// gradually: today only `chatMode` + `topicGate` + `refusalFlags` are
/// trained (carry over from PR #30); `emotionalState` and `slotCompleteness`
/// land after the Phase 1/2 H100 retrain. The downstream `TelcoUnderstandingRouter`
/// only consults the heads that ARE present — it never invents a
/// fallback value when a head is missing.
///
/// **Design principle #1 from ADR-022 §4.3**:
/// > heads INFORM decisions, deterministic routers DECIDE — and decision
/// > routers must be pure functions over head outputs, never confidence-
/// > threshold overrides of them.
///
/// The router trusts the top-1 of every head it consults. Confidence is
/// surfaced into the trace for engineering review — never used as a gate.
///
/// `Sendable` so the actor-isolated `QueryUnderstandingClassifier` can
/// hand it across isolation domains without copies-with-mutation hazards.
/// `Equatable` so the test suite can assert structural equality against
/// fixture vectors without bespoke matchers.
public struct QueryUnderstanding: Sendable, Equatable {

    // MARK: - Head outputs (each optional — see file header)

    /// 4-class top-level routing: kbQuestion / toolAction / personalSummary
    /// / outOfScope. Trained on the existing `chat-mode-router-v2` data,
    /// re-bundled into the v2 shared backbone.
    public let chatMode: ChatModePrediction?

    /// 3-class Telco-specific scope gate: in_scope / out_of_scope /
    /// greeting. Trained on the Telco RAG corpus probe-set. Catches
    /// in-domain queries the generic `chatMode` head might mis-route.
    public let topicGate: TopicGateOutcome?

    /// 3-flag multi-label routing signals: has_rag_answer / navigation_only /
    /// live_agent_trigger. Sigmoid — flags are independent. Drives nav-only,
    /// live-agent, unknown-feature lane decisions.
    public let refusalFlags: RefusalFlagsOutcome?

    /// 3-class emotional state: neutral / frustrated / urgent. Drives
    /// proactive escalation NBAs. **Fresh schema** for v2 — not present
    /// in PR #30. Until the head ships, this stays nil and any NBA that
    /// depends on it stays dormant. No regex/keyword fallback —
    /// emotional state is irreducibly subjective and warrants the model.
    public let emotionalState: EmotionalStateOutcome?

    /// 4-flag multi-label slot presence: has_device / has_location /
    /// has_time / has_account_ref. Drives clarification questions for
    /// the tool-action lane. **Fresh schema** for v2. Until the head
    /// ships, the workflow falls back to today's behaviour (LFMToolSelector
    /// extracts whatever args it can; clarification stub renders).
    public let slotCompleteness: SlotCompletenessOutcome?

    // MARK: - ADR-024 — pairwise relational heads (Layer 1')

    /// 5-way categorical relationship between this turn and the prior
    /// assistant reply. See `TurnRelationship` for the taxonomy.
    /// Nil on the first turn of a session OR when the relational
    /// adapter isn't bundled — router defaults nil to INDEPENDENT.
    public let turnRelationship: TurnRelationshipOutcome?

    /// Per-slot binary: does this turn fill each slot? Replaces the
    /// bare-noun regex heuristic in clarification recovery when the
    /// relational head is bundled.
    public let slotAlignment: SlotAlignmentOutcome?

    /// 3-way: is the user continuing, reverting, or overriding the
    /// prior turn's intent / parameters?
    public let stanceChange: StanceChangeOutcome?

    // MARK: - Timing + provenance

    /// Wall-clock for the entire Layer 1 pass — adapter swap(s),
    /// forward pass(es), head projection(s), enum mapping. Surfaced in
    /// the engineering trace so the shared-backbone vs composite-fallback
    /// strategy tradeoff stays measurable.
    public let totalMs: Double

    /// Which classifier strategy produced this vector. Critical for
    /// debugging — `composite` means we paid 2-3 forward passes (one for
    /// each head's private adapter); `shared` means we paid 1. The
    /// engineering trace surfaces this so a reviewer can see why a
    /// particular turn was slow.
    public let strategy: ClassifierStrategy

    public init(
        chatMode: ChatModePrediction? = nil,
        topicGate: TopicGateOutcome? = nil,
        refusalFlags: RefusalFlagsOutcome? = nil,
        emotionalState: EmotionalStateOutcome? = nil,
        slotCompleteness: SlotCompletenessOutcome? = nil,
        turnRelationship: TurnRelationshipOutcome? = nil,
        slotAlignment: SlotAlignmentOutcome? = nil,
        stanceChange: StanceChangeOutcome? = nil,
        totalMs: Double = 0,
        strategy: ClassifierStrategy = .unavailable
    ) {
        self.chatMode = chatMode
        self.topicGate = topicGate
        self.refusalFlags = refusalFlags
        self.emotionalState = emotionalState
        self.slotCompleteness = slotCompleteness
        self.turnRelationship = turnRelationship
        self.slotAlignment = slotAlignment
        self.stanceChange = stanceChange
        self.totalMs = totalMs
        self.strategy = strategy
    }
}

// MARK: - Head outcome wrappers
//
// Every outcome carries the typed label + the head's reported confidence
// (or per-class probabilities for multi-label). The router never reads
// `confidence`; it reads `value`. Tests can construct these directly
// without spinning up a classifier.

/// Topic-gate head output. Mirrors `TelcoStageADecision.topicGate` +
/// `topicGateConfidence` but in a self-contained shape callers can
/// build in tests.
public struct TopicGateOutcome: Sendable, Equatable {
    public let value: TelcoTopicGate
    public let confidence: Double

    public init(value: TelcoTopicGate, confidence: Double) {
        self.value = value
        self.confidence = confidence
    }
}

/// Refusal-flags head output. Carries the typed flags + per-class
/// sigmoid probabilities for engineering inspection. The 3 flags are
/// independent — `liveAgentTrigger` can fire with `navigationOnly` and
/// the router resolves precedence (live-agent beats nav-only per
/// ADR-021 §3).
public struct RefusalFlagsOutcome: Sendable, Equatable {
    public let value: TelcoRefusalFlags
    /// Per-flag sigmoid scores [has_rag_answer, navigation_only,
    /// live_agent_trigger]. Surfaced in the trace; never consulted by
    /// the router.
    public let probabilities: [Double]

    public init(value: TelcoRefusalFlags, probabilities: [Double]) {
        self.value = value
        self.probabilities = probabilities
    }
}

/// Emotional-state head output. **New for v2.** A 3-class subjective
/// signal — see ADR-022 §7 for the labeller-disagreement mitigation
/// (use the production 50-conversation probe set as ground truth,
/// accept noisy training labels in exchange for better-than-no-signal).
public struct EmotionalStateOutcome: Sendable, Equatable {
    public let value: EmotionalState
    public let confidence: Double

    public init(value: EmotionalState, confidence: Double) {
        self.value = value
        self.confidence = confidence
    }
}

/// Slot-completeness head output. **New for v2.** Multi-label sigmoid
/// — the four slot kinds are independent. The router never reads this
/// to pick a lane; the WORKFLOW (specifically the tool-action handler)
/// reads it to decide whether to ask a clarification question BEFORE
/// running the tool. See `ToolSlotRequirements` for the per-tool gate.
public struct SlotCompletenessOutcome: Sendable, Equatable {
    public let value: SlotCompleteness
    /// Per-slot sigmoid scores [has_device, has_location, has_time,
    /// has_account_ref]. Order matches the trained wire contract —
    /// must not be reordered without a head re-export.
    public let probabilities: [Double]

    public init(value: SlotCompleteness, probabilities: [Double]) {
        self.value = value
        self.probabilities = probabilities
    }
}

// MARK: - Provenance

/// Which strategy the `QueryUnderstandingClassifier` used to produce a
/// vector. Surfaced in the trace; never affects routing.
///
/// - `shared`: one adapter swap + one forward pass + N head projections.
///   The architectural target for v2 (telco-shared-clf-v2 bundled).
/// - `composite`: per-head adapter swaps + per-head forward passes.
///   Today's PR #30 path (telco-topic-gate + telco-refusal-flags private
///   adapters) plus the existing `LFMChatModeRouter` for chat_mode.
/// - `unavailable`: the classifier couldn't load (degraded build).
///   Caller should fall back to existing piecemeal routing.
public enum ClassifierStrategy: String, Sendable, Equatable, Codable {
    case shared
    case composite
    case unavailable

    public var displayName: String {
        switch self {
        case .shared:      return "Shared backbone (v2)"
        case .composite:   return "Composite (v1)"
        case .unavailable: return "Unavailable"
        }
    }
}
