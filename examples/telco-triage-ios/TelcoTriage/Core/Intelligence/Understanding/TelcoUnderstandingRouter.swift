import Foundation

/// Layer 2 of the ADR-022 architecture: the **pure-function router**
/// that maps a `QueryUnderstanding` + optional retrieval signal into a
/// `TelcoLane`. Zero I/O. Zero state. Zero LFM calls. Sub-millisecond.
///
/// **This is the architecturally-load-bearing component of ADR-022.**
/// The router is the contract between the heads (which INFORM) and the
/// workflow (which ACTS). Every routing decision is reproducible from
/// inputs alone — fix a bug here and you fix it for every query the
/// model ever sees, with a regression test for proof.
///
/// **Design principle #1 from ADR-022 §4.3** (codified in tests):
/// > Heads inform; deterministic routers decide. No threshold overrides
/// > trained head output. If a head's confidence is low, the router
/// > still trusts its top-1 — the trace surfaces the low confidence
/// > for engineering review, NOT for runtime gating.
///
/// **Decision order** (matches §4.3 and supersedes `TelcoRagRouter`):
///
///  1. Topic gate short-circuits (greeting / OOS) — covers Layer 1's
///     finest-grained scope signal.
///  2. **Workflow upgrade**: `emotional_state ∈ {urgent}` AND the trained
///     `live_agent_trigger` flag is also set → escalate. This is the ONE
///     place the emotional head is allowed to upgrade a routing decision
///     (per ADR-022 §4.3 design principle), but only when corroborated
///     by the trained refusal-flags signal. A frustrated/urgent user who
///     hasn't explicitly asked for a human still gets a polite chip from
///     the NBA layer — we don't unilaterally escalate on subjective
///     signal alone.
///  3. Explicit `live_agent_trigger` flag → escalation (mirrors
///     `TelcoRagRouter` order).
///  4. `navigation_only` flag → in-app deep link.
///  5. `chat_mode` Layer 1 categorical: tool_action / personal_summary
///     map to non-RAG lanes carried as new lane cases.
///  6. RAG-eligible path with no flagged answer → unknown-feature.
///  7. RAG-eligible with retrieval below floor → unknown-feature.
///  8. RAG-eligible with ambiguous retrieval → clarification.
///  9. Default → step-by-step RAG.
///
/// **Lanes returned**: this router emits the existing `TelcoLane` set
/// PLUS two new "carrier" values that previously lived implicitly in
/// ChatViewModel's switch (`toolAction`, `personalSummary`). The new
/// lanes are added to `UnderstandingLane` (a superset of `TelcoLane`)
/// so the workflow handler can pattern-match exhaustively.
public enum TelcoUnderstandingRouter {

    /// Confidence floor for retrieval — same number as the legacy
    /// `TelcoRagRouter.retrievalConfidenceFloor` per ADR-021 §3.
    /// Kept here as a separate constant because the v2 router OWNS
    /// the policy; we don't want the legacy file's constant to drift
    /// independently.
    public static let retrievalConfidenceFloor: Double = 0.45

    /// Ambiguity gap — same number as the legacy router. Two
    /// candidates within this gap with otherwise plausible scores
    /// trigger clarification instead of step-by-step.
    public static let ambiguityGapThreshold: Double = 0.10

    /// High-confidence ceiling — above this, ignore the gap (a
    /// strong top-1 wins even with a close runner-up).
    public static let highConfidenceCeiling: Double = 0.75

    // MARK: - Public entry points

    /// Production entry point. Accepts the full understanding vector
    /// + optional retrieval signal. The retrieval argument is optional
    /// because non-RAG lanes (greeting, OOS, tool action, personal
    /// summary) don't need it.
    public static func decide(
        understanding: QueryUnderstanding,
        retrieval: ColBERTRetrievalResult? = nil
    ) -> UnderstandingLane {
        // ----------------------------------------------------------------
        // Step 1: topic-gate short-circuits. The Telco-trained topic
        // gate is the finest in-domain scope detector we have; trust it
        // first when present, fall through to `chat_mode` when absent.
        // ----------------------------------------------------------------
        if let topic = understanding.topicGate?.value {
            switch topic {
            case .greeting:
                return .telco(.greeting)
            case .outOfScope:
                return .telco(.oosRefusal)
            case .inScope:
                break  // Continue to flag-based routing below.
            }
        } else if let mode = understanding.chatMode?.mode {
            // No topic gate → fall back to chat_mode's coarse scope.
            switch mode {
            case .outOfScope:
                return .telco(.oosRefusal)
            case .personalSummary:
                return .personalSummary
            case .toolAction:
                return .toolAction
            case .kbQuestion:
                break  // Continue to flag-based routing below.
            }
        } else {
            // Neither head present — degraded build. Safe default is
            // OOS refusal so we don't silently engage downstream stages
            // with no scope signal at all.
            return .telco(.oosRefusal)
        }

        // ----------------------------------------------------------------
        // Step 2: emotional workflow upgrade. ONLY when both the
        // emotional head says urgent AND the trained refusal_flags head
        // confirms live_agent_trigger. We never let subjective signal
        // alone override the trained routing decision — that's the
        // exact failure mode the ADR-021 §0 deletion fixed.
        // ----------------------------------------------------------------
        if understanding.emotionalState?.value == .urgent,
           understanding.refusalFlags?.value.liveAgentTrigger == true {
            return .telco(.liveAgentEscalation)
        }

        // ----------------------------------------------------------------
        // Step 3: explicit live-agent flag wins over nav-only, mirrors
        // ADR-021 §3 ordering ("talk to a person about my bill" must
        // escalate, not silently navigate).
        // ----------------------------------------------------------------
        if understanding.refusalFlags?.value.liveAgentTrigger == true {
            return .telco(.liveAgentEscalation)
        }

        // ----------------------------------------------------------------
        // Step 4: navigation-only flag → deep link.
        // ----------------------------------------------------------------
        if understanding.refusalFlags?.value.navigationOnly == true {
            return .telco(.navOnlyDeeplink)
        }

        // ----------------------------------------------------------------
        // Step 5: chat_mode categorical for non-question lanes. Done
        // AFTER topic_gate so a Telco-trained `topic_gate=in_scope`
        // dominates a coarse `chat_mode=tool_action` — the topic gate
        // says "this is a Telco question worth a RAG answer".
        // ----------------------------------------------------------------
        if let mode = understanding.chatMode?.mode {
            switch mode {
            case .toolAction:
                return .toolAction
            case .personalSummary:
                return .personalSummary
            case .kbQuestion, .outOfScope:
                break  // .kbQuestion falls through to the RAG branch.
                       // .outOfScope was handled by topic_gate above.
            }
        }

        // ----------------------------------------------------------------
        // Step 6-8: RAG-answerable branch. The dispatcher trusts
        // `has_rag_answer` + the live retrieval score, not heuristics.
        // ----------------------------------------------------------------
        let hasRagAnswerFlag = understanding.refusalFlags?.value.hasRagAnswer ?? true

        if !hasRagAnswerFlag {
            return .telco(.unknownFeature)
        }

        // Resolve retrieval. If the retriever didn't return a result
        // (degraded ColBERT, cold start), fall back to a permissive
        // synthetic confidence — same as the legacy `TelcoRagRouter`
        // overload that takes an optional retrieval result. The
        // engineering trace will surface the missing signal so the
        // degradation isn't silent.
        let (confidence, gap) = resolveRetrieval(
            understanding: understanding,
            retrieval: retrieval
        )

        if confidence < retrievalConfidenceFloor {
            return .telco(.unknownFeature)
        }

        if gap < ambiguityGapThreshold && confidence < highConfidenceCeiling {
            return .telco(.clarification)
        }

        return .telco(.ragStepByStep)
    }

    /// Resolve the retrieval (confidence, gap) pair. Defensive against
    /// nil retrieval — uses a permissive synthetic when `has_rag_answer`
    /// is true (preserves PR #30 pre-Phase-1b behaviour). Pulled out so
    /// `decide(_:)` stays readable.
    private static func resolveRetrieval(
        understanding: QueryUnderstanding,
        retrieval: ColBERTRetrievalResult?
    ) -> (Double, Double) {
        if let retrieval, !retrieval.hits.isEmpty {
            return (Double(retrieval.topConfidence), Double(retrieval.topGap))
        }
        let hasRag = understanding.refusalFlags?.value.hasRagAnswer ?? true
        return hasRag ? (0.95, 0.5) : (0.0, 0.0)
    }

    // MARK: - ADR-024 — multi-turn fusion

    /// Conservative probability floor for STANCE_REVERT before we act
    /// on it. Misfires on REVERT are uniquely damaging (would clear
    /// pending state and slot accumulation on a healthy turn), so we
    /// require strong head confidence per ADR-024 §9 risk row.
    public static let stanceRevertConfidenceFloor: Double = 0.85

    /// **The fused multi-turn router** (ADR-024 §4.6). Takes the full
    /// `QueryUnderstanding` vector + `ConversationSnapshot` + optional
    /// retrieval signal, returns a complete `RoutingDecision` —
    /// `lane` + `actions` the workflow executes (`firePendingTool`,
    /// `accumulateSlot`, etc.).
    ///
    /// Pure function. Zero I/O. Reproducible from inputs alone — same
    /// principle as `decide(understanding:retrieval:)`, just at one
    /// level higher in the call graph. Single-turn behavior recovered
    /// when relational fields are nil OR conversation snapshot is
    /// empty.
    ///
    /// **Decision priority** (first match wins):
    ///  1. Pending tool confirmation + AFFIRMATIVE_CONTINUATION
    ///  2. Pending clarification + CLARIFICATION_ANSWER + matching slot
    ///  3. STANCE_REVERT — clear pending state, treat as fresh
    ///  4. NEGATIVE_CONTINUATION — keep prior lane, increment counter
    ///  5. ANAPHORIC — augment retrieval (existing lane decision logic)
    ///  6. AFFIRMATIVE_CONTINUATION (without pending tool) — fresh classify
    ///     with frustration-counter decrement signal
    ///  7. Default: existing single-turn `decide(understanding:retrieval:)`
    public static func decideMultiTurn(
        understanding: QueryUnderstanding,
        conversation: ConversationSnapshot,
        pendingClarification: PendingClarification? = nil,
        pendingToolConfirmation: ToolDecision? = nil,
        retrieval: ColBERTRetrievalResult? = nil
    ) -> RoutingDecision {
        // ----------------------------------------------------------------
        // Step 1 — Pending tool confirmation + AFFIRMATIVE
        // The user said "yes/ok/go ahead" AND we have an unresolved
        // tool proposal. Fire it.
        // ----------------------------------------------------------------
        if let pendingTool = pendingToolConfirmation,
           let rel = understanding.turnRelationship?.value,
           rel == .affirmativeContinuation {
            // Note: firePendingTool short-circuits the action loop
            // inside ChatViewModel — `clearPendingToolConfirmation`
            // and `clearPendingClarification` are kept in the actions
            // list as a defence-in-depth declaration (the router's
            // contract is that BOTH pending pointers are stale after
            // a fire). The handler also clears them explicitly before
            // re-dispatching so a future refactor of the loop order
            // can't drop the cleanup.
            return RoutingDecision(
                lane: .toolAction,
                actions: [
                    .firePendingTool(pendingTool),
                    .clearPendingToolConfirmation,
                    .clearPendingClarification
                ],
                reasoning: "AFFIRMATIVE_CONTINUATION on pending tool — fire."
            )
        }

        // ----------------------------------------------------------------
        // Step 2 — Pending clarification + CLARIFICATION_ANSWER
        // Verify slot_alignment confirms the reply fills the missing
        // slot. If yes → accumulate + clear pending. If no → degrade
        // to anaphoric (fall through to step 5).
        // ----------------------------------------------------------------
        if let pending = pendingClarification,
           let rel = understanding.turnRelationship?.value,
           rel == .clarificationAnswer {
            // The head says "this is an answer." Verify which slot it
            // fills via slot_alignment (when bundled) — otherwise the
            // bare-noun heuristic in ChatViewModel handles it post-route.
            let fillsMissing = understanding.slotAlignment?.value.filledSlots
                .intersection(pending.missingSlots) ?? []
            if !fillsMissing.isEmpty, let intent = pending.intent {
                return RoutingDecision(
                    lane: .toolAction,
                    actions: [
                        .accumulateSlotsFromAlignment(intent: intent, slots: fillsMissing),
                        .clearPendingClarification
                    ],
                    reasoning: "CLARIFICATION_ANSWER fills missing slot(s): \(fillsMissing.map(\.rawValue).sorted().joined(separator: ","))"
                )
            }
            // Head fired but slot alignment empty → either head was
            // wrong OR the bundled bundle has no slot_alignment yet.
            // Fall through; ChatViewModel's bare-noun heuristic
            // (preserved as the degraded path) handles the recovery.
        }

        // ----------------------------------------------------------------
        // Step 3 — STANCE_REVERT (with confidence floor)
        // User explicitly cancels prior decision. Clear pending state
        // and slot accumulation for the prior intent, then classify
        // q_t fresh as if it were a new conversation.
        // ----------------------------------------------------------------
        if let stance = understanding.stanceChange,
           stance.value == .revert,
           stance.confidence >= stanceRevertConfidenceFloor {
            let baseLane = decide(understanding: understanding, retrieval: retrieval)
            var actions: [PostDecisionAction] = [
                .clearPendingClarification,
                .clearPendingToolConfirmation
            ]
            if let prior = conversation.priorIntent {
                actions.append(.clearSlotStore(intent: prior))
            }
            return RoutingDecision(
                lane: baseLane,
                actions: actions,
                reasoning: "STANCE_REVERT @ \(String(format: "%.2f", stance.confidence)) — fresh classification."
            )
        }

        // ----------------------------------------------------------------
        // Step 3.5 — STANCE_OVERRIDE (same confidence floor as REVERT)
        // User refines a parameter while keeping the prior intent
        // ("actually for my daughter's tablet, not my son's"). The
        // slot_alignment head should fire positive on the new slot;
        // we accumulate the slot for the PRIOR intent and keep the
        // prior lane so the workflow re-proposes with refined args.
        //
        // No-op when prior intent is missing OR slot_alignment is
        // silent — degrades to single-turn baseline rather than
        // mis-fire on insufficient signal.
        // ----------------------------------------------------------------
        if let stance = understanding.stanceChange,
           stance.value == .override,
           stance.confidence >= stanceRevertConfidenceFloor {
            var actions: [PostDecisionAction] = []
            if let prior = conversation.priorIntent,
               let alignment = understanding.slotAlignment?.value {
                let filled = alignment.filledSlots
                if !filled.isEmpty {
                    actions.append(.accumulateSlotsFromAlignment(
                        intent: prior, slots: filled
                    ))
                }
            }
            // Keep prior lane when available — the user is staying in
            // the same flow, just correcting parameters. Fall back to
            // single-turn lane when no prior context exists.
            let lane = conversation.priorLane
                ?? decide(understanding: understanding, retrieval: retrieval)
            return RoutingDecision(
                lane: lane,
                actions: actions,
                reasoning: "STANCE_OVERRIDE @ \(String(format: "%.2f", stance.confidence)) — same intent, refined slots."
            )
        }

        // ----------------------------------------------------------------
        // Step 4 — NEGATIVE_CONTINUATION
        // "didn't work" / "still broken". Keep the prior lane (don't
        // re-propose the same thing). The NBA layer escalates when
        // didntWorkCount crosses the threshold; the router just keeps
        // the user in the current lane.
        // ----------------------------------------------------------------
        if let rel = understanding.turnRelationship?.value,
           rel == .negativeContinuation {
            let lane = conversation.priorLane
                ?? decide(understanding: understanding, retrieval: retrieval)
            return RoutingDecision(
                lane: lane,
                actions: [.traceNegativeContinuation, .suppressIntentRepeat],
                reasoning: "NEGATIVE_CONTINUATION — keeping prior lane, escalation via NBA."
            )
        }

        // ----------------------------------------------------------------
        // Step 5 — ANAPHORIC
        // The user references an entity the assistant mentioned. Run
        // the normal single-turn lane decision but add an action that
        // augments retrieval with prior assistant context (so ColBERT
        // recall on the referenced term improves).
        // ----------------------------------------------------------------
        if let rel = understanding.turnRelationship?.value,
           rel == .anaphoric {
            let baseLane = decide(understanding: understanding, retrieval: retrieval)
            return RoutingDecision(
                lane: baseLane,
                actions: [.augmentRetrievalWithPriorAssistant],
                reasoning: "ANAPHORIC — augmenting retrieval with prior assistant context."
            )
        }

        // ----------------------------------------------------------------
        // Step 6 — AFFIRMATIVE_CONTINUATION (no pending tool)
        // "great, thanks" / "that worked" — positive signal. Decrement
        // any pending frustration markers, classify q_t fresh.
        // ----------------------------------------------------------------
        if let rel = understanding.turnRelationship?.value,
           rel == .affirmativeContinuation {
            let baseLane = decide(understanding: understanding, retrieval: retrieval)
            return RoutingDecision(
                lane: baseLane,
                actions: [.traceAffirmativeContinuation],
                reasoning: "AFFIRMATIVE_CONTINUATION (no pending tool) — positive session signal."
            )
        }

        // ----------------------------------------------------------------
        // Step 7 — Default / INDEPENDENT
        // First turn of a session, nil relational signal, or
        // independent classification. Existing single-turn router.
        // ----------------------------------------------------------------
        let lane = decide(understanding: understanding, retrieval: retrieval)
        return RoutingDecision(
            lane: lane,
            actions: [],
            reasoning: relationshipReason(understanding)
        )
    }

    private static func relationshipReason(_ understanding: QueryUnderstanding) -> String {
        if let rel = understanding.turnRelationship?.value, rel == .independent {
            return "INDEPENDENT — single-turn classification."
        }
        return "Single-turn baseline (no relational signal available)."
    }

    // MARK: - Back-compat bridge

    /// Bridge that consumes a `TelcoStageADecision` (the PR #30
    /// shape) and produces an `UnderstandingLane`. Useful while the
    /// shared-backbone v2 retrain is in flight — Stage A still
    /// produces the same two heads, we just wrap them into the
    /// `QueryUnderstanding` vector.
    ///
    /// Pre-supplies `chatMode` only when the caller provides one
    /// (e.g., from `LFMChatModeRouter`). When chatMode is nil and
    /// topic_gate fires `.inScope`, the router routes as if the
    /// query were a KB question (the existing PR #30 default for
    /// the `.kbQuestion` chat branch).
    public static func decide(
        stageA: TelcoStageADecision,
        chatMode: ChatModePrediction? = nil,
        retrieval: ColBERTRetrievalResult? = nil
    ) -> UnderstandingLane {
        let understanding = QueryUnderstanding(
            chatMode: chatMode,
            topicGate: TopicGateOutcome(
                value: stageA.topicGate,
                confidence: stageA.topicGateConfidence
            ),
            refusalFlags: RefusalFlagsOutcome(
                value: stageA.refusalFlags,
                probabilities: stageA.refusalFlagsProbabilities.map(Double.init)
            ),
            emotionalState: nil,
            slotCompleteness: nil,
            totalMs: stageA.totalMs,
            strategy: .composite
        )
        return decide(understanding: understanding, retrieval: retrieval)
    }
}

/// Lane returned by the v2 understanding router. A superset of the
/// existing `TelcoLane` plus two carrier values for the chat-mode
/// branches the dispatcher used to handle implicitly via the
/// ChatViewModel switch.
///
/// **Why a separate enum**: the legacy `TelcoLane` lives in a file
/// the Telco dispatcher still owns, and adding `toolAction` /
/// `personalSummary` to it would force every existing call site to
/// handle them. The cleaner migration is a new outer enum that maps
/// 1:1 to `TelcoLane` for the existing 7 cases and adds the two
/// chat-mode carriers separately. The dispatcher only sees
/// `.telco(...)`; the chat layer pattern-matches the carriers.
public enum UnderstandingLane: Sendable, Equatable {
    /// Maps directly to a `TelcoLane`. The dispatcher handles these.
    case telco(TelcoLane)
    /// Tool-action lane — handled by `ChatViewModel.runToolProposal`.
    case toolAction
    /// Personal-summary lane — handled by `ChatViewModel.runPersonalizedSummary`.
    case personalSummary

    /// Stable wire string for logs + telemetry. Adds the two new
    /// carriers to the existing `TelcoLane.wireName` namespace.
    public var wireName: String {
        switch self {
        case .telco(let lane):     return lane.wireName
        case .toolAction:            return "tool_action"
        case .personalSummary:       return "personal_summary"
        }
    }

    /// True when this lane requires LFM generation downstream.
    /// `.toolAction` triggers the LFMToolSelector or ImperativeToolDetector
    /// fast-path (both produce a structured selection without an
    /// open-ended generation). `.personalSummary` runs deterministic
    /// Swift composition today (no LFM). Only `.telco(.ragStepByStep)`
    /// runs Stage B.
    public var requiresStageBGeneration: Bool {
        switch self {
        case .telco(let lane):  return lane.requiresGeneration
        case .toolAction,
             .personalSummary:    return false
        }
    }

    /// ADR-023 post-screenshot fix (2026-05-26): the legacy 4-case
    /// `RoutingPath` (still used by the `RoutingSummary` trace card)
    /// MUST be derived from the **resolved lane**, not from the
    /// `chat_mode` head's vote. When the generative chat_mode router
    /// fails (parses as 0.00 → outOfScope default) but `topic_gate`
    /// rescues the turn into a RAG lane, the card was telling users
    /// `INTENT: out_of_scope` while showing a perfect RAG answer above
    /// it. The lane is what the dispatcher actually executed; the
    /// trace must reflect that.
    ///
    /// Mapping (closed-world — every case exhaustive):
    ///  - `.toolAction`               → `.toolCall`
    ///  - `.personalSummary`          → `.personalized`
    ///  - `.telco(.oosRefusal)`     → `.outOfScope`
    ///  - `.telco(.liveAgentEscalation)` → `.outOfScope` (template
    ///    response, not a RAG answer — fits the "declined" surface)
    ///  - all other telco lanes (ragStepByStep, unknownFeature,
    ///    clarification, navOnlyDeeplink, greeting) → `.answerWithRAG`
    ///    since they all produce content the user reads (article,
    ///    clarification, deep link, hello).
    public var routingPath: RoutingPath {
        switch self {
        case .toolAction:
            return .toolCall
        case .personalSummary:
            return .personalized
        case .telco(let v):
            switch v {
            case .oosRefusal, .liveAgentEscalation:
                return .outOfScope
            case .ragStepByStep, .unknownFeature, .clarification,
                 .navOnlyDeeplink, .greeting:
                return .answerWithRAG
            }
        }
    }
}

// MARK: - ADR-024 RoutingDecision + PostDecisionAction

/// ADR-024 — the fully-resolved routing decision the workflow
/// executes. Carries the chosen lane PLUS the side-effects the
/// workflow must perform (fire pending tool, accumulate a slot,
/// augment retrieval, etc.). Pure data — the workflow handler
/// inspects actions and dispatches.
///
/// Source compat: when `actions.isEmpty`, the decision is
/// indistinguishable from the legacy single-turn router's lane
/// output — single-turn call sites can keep ignoring `actions`.
public struct RoutingDecision: Sendable, Equatable {
    public let lane: UnderstandingLane
    public let actions: [PostDecisionAction]
    public let reasoning: String

    public init(
        lane: UnderstandingLane,
        actions: [PostDecisionAction] = [],
        reasoning: String = ""
    ) {
        self.lane = lane
        self.actions = actions
        self.reasoning = reasoning
    }
}

/// ADR-024 — side-effects the workflow performs after the router
/// resolves. Explicit list so EVERY side-effect is reproducible
/// from inputs alone (same testability principle as the lane
/// resolution itself).
public enum PostDecisionAction: Sendable, Equatable {
    /// Fire the pending tool — used when AFFIRMATIVE_CONTINUATION
    /// arrives on a pending tool confirmation.
    case firePendingTool(ToolDecision)

    /// Clear the pending-tool-confirmation pointer. Always paired
    /// with `firePendingTool` OR with a STANCE_REVERT.
    case clearPendingToolConfirmation

    /// Clear the pending-clarification pointer. Used on
    /// successful recovery and on STANCE_REVERT.
    case clearPendingClarification

    /// Accumulate slot values from a successful slot_alignment
    /// match. ChatViewModel reads slot values out of the
    /// understanding vector + the user query.
    case accumulateSlotsFromAlignment(intent: ToolIntent, slots: Set<Slot>)

    /// Clear the slot-accumulation store for an intent — used on
    /// STANCE_REVERT to discard prior accumulated slots.
    case clearSlotStore(intent: ToolIntent)

    /// Trace marker emitted on NEGATIVE_CONTINUATION. The actual
    /// didntWorkCount mutation happens inside
    /// `ConversationStateRecorder.isDidntWorkContinuation` during
    /// `recordTurn` — this case exists only so the engineering trace
    /// can show the router's classification. Renamed from
    /// `incrementDidntWorkCounter` (2026-05-27) to prevent
    /// future double-count refactors per the post-ship code review.
    case traceNegativeContinuation

    /// Trace marker emitted on AFFIRMATIVE_CONTINUATION (no pending
    /// tool path). Counters are append-only today; positive signal
    /// has no decrement path. Renamed from
    /// `decrementFrustrationCounters` (2026-05-27) so the no-op
    /// semantics is in the case name. When a per-session
    /// `affirmativeCount` field lands, surface it through a real
    /// mutator — do not repurpose this case.
    case traceAffirmativeContinuation

    /// Augment ColBERT retrieval with tokens from the prior
    /// assistant reply — improves recall on anaphoric references.
    case augmentRetrievalWithPriorAssistant

    /// Suppress the same intent from being proposed again on this
    /// turn — used on NEGATIVE_CONTINUATION so the assistant
    /// doesn't keep proposing "restart your router" after the
    /// user already said it didn't work.
    case suppressIntentRepeat
}

/// Outcome of `ChatViewModel.applyPostDecisionActions`. Replaces the
/// previous `Bool` (was: "did we short-circuit?") so actions can
/// communicate richer downstream intent through the type system rather
/// than via log statements (the prior approach for
/// `.augmentRetrievalWithPriorAssistant`, which was a no-op).
///
/// Two facets:
///   - `shortCircuited` — when true, `processTextQuery` must return
///     immediately (the action handled the dispatch itself, e.g.
///     `firePendingTool` rerouted through `runToolProposal`).
///   - `retrievalContext` — the augmentation envelope to pass to the
///     downstream lane handler's retrieval call. `.empty` for the
///     common case; populated when `.augmentRetrievalWithPriorAssistant`
///     fired AND the conversation state has a prior assistant message
///     cached.
public struct PostActionResult: Sendable, Equatable {
    public let shortCircuited: Bool
    public let retrievalContext: RetrievalContext

    public init(
        shortCircuited: Bool = false,
        retrievalContext: RetrievalContext = .empty
    ) {
        self.shortCircuited = shortCircuited
        self.retrievalContext = retrievalContext
    }

    public static let passthrough = PostActionResult()
    public static let shortCircuit = PostActionResult(shortCircuited: true)
}
