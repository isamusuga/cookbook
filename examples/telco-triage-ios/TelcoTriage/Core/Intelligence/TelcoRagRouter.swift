import Foundation

/// Inputs to the Telco deterministic router for one conversation turn.
///
/// Aggregates the signals produced by the single LFM2.5-350M forward
/// pass (topic gate, refusal flags) plus the ColBERT retrieval
/// confidence. The router never invokes the model; it only reads these
/// signals and picks a lane.
///
/// `intent` was originally a required field (10-class macro intent
/// classifier head) but was **DROPPED May 2026** after probe scoring
/// revealed 43.6% real-production accuracy vs 81% synthetic-eval
/// accuracy. The field remains as an OPTIONAL signal for backwards
/// compatibility with call sites and tests; the router itself no
/// longer consults it. See ADR-021 §6.5 for the postmortem.
///
/// See `docs/architecture-decisions/ADR-021-telco-home-internet-rag-assistant.md` §2.2, §3, §6.5.
public struct TelcoRouterInputs: Sendable, Equatable {
    /// Topic gate output.
    public let topicGate: TelcoTopicGate

    /// Macro intent (10-class). **Ignored by the router as of May 2026** —
    /// kept optional for backward compatibility with call sites that
    /// still produce it (e.g., for link-resolver fallback selection in
    /// `TelcoLinkResolver.levelOneFallback`). The router decision
    /// flows purely through `topicGate` + `flags` + retrieval signals.
    public let intent: TelcoMacroIntent?

    /// Multi-label sigmoid output.
    public let flags: TelcoRefusalFlags

    /// Top-1 ColBERT retrieval confidence in `[0.0, 1.0]`. The router
    /// uses this to decide between RAG step-by-step (high confidence)
    /// and clarification (ambiguous) lanes.
    public let topRetrievalConfidence: Double

    /// Gap between top-1 and top-2 retrieval scores. A small gap with
    /// high absolute scores indicates ambiguity worth clarifying.
    public let retrievalTopGap: Double

    public init(
        topicGate: TelcoTopicGate,
        intent: TelcoMacroIntent? = nil,
        flags: TelcoRefusalFlags,
        topRetrievalConfidence: Double,
        retrievalTopGap: Double
    ) {
        self.topicGate = topicGate
        self.intent = intent
        self.flags = flags
        self.topRetrievalConfidence = topRetrievalConfidence
        self.retrievalTopGap = retrievalTopGap
    }
}

/// 3-class topic gate output. Distinct from the macro intent — topic
/// gate runs first and short-circuits the pipeline for OOS / greeting.
public enum TelcoTopicGate: Int, Sendable, CaseIterable, Codable {
    case inScope = 0
    case outOfScope = 1
    case greeting = 2
}

/// Deterministic policy that picks a `TelcoLane` from router inputs.
///
/// Pure function (no I/O, no state). Sub-millisecond. The model emits
/// signals; this router owns the decision. Per
/// `liquid-models-architecture` §11 (Ottoguard precedent):
/// *"anything checkable in O(1) belongs in code, not a model."*
///
/// Implements the routing policy from
/// `docs/architecture-decisions/ADR-021-telco-home-internet-rag-assistant.md` §3.
public enum TelcoRagRouter {
    /// Confidence threshold below which retrieval is considered to have
    /// no RAG answer, even if the sigmoid says `hasRagAnswer = true`.
    /// Tuned at 0.45 per the Ottoguard precedent — adjustable after
    /// Phase 0 probe-set calibration.
    public static let retrievalConfidenceFloor: Double = 0.45

    /// Maximum gap between top-1 and top-2 retrieval scores below
    /// which we treat the retrieval as ambiguous and route to
    /// clarification (when both scores are otherwise plausible).
    public static let ambiguityGapThreshold: Double = 0.10

    /// Picks the lane for a single turn.
    ///
    /// Decision order matches §3:
    ///   1. greeting → greeting lane
    ///   2. OOS → OOS refusal
    ///   3. explicit live-agent flag → escalation
    ///   4. navigation-only flag → in-app deep-link
    ///   5. intent is not RAG-answerable → unknown-feature refusal
    ///   6. retrieval below floor → unknown-feature refusal
    ///   7. retrieval is ambiguous → clarification
    ///   8. default → RAG step-by-step
    ///
    /// Ordering matters: live-agent fires before nav-only so an explicit
    /// "talk to a person" question about billing escalates rather than
    /// silently navigating away.
    public static func route(_ inputs: TelcoRouterInputs) -> TelcoLane {
        switch inputs.topicGate {
        case .greeting:
            return .greeting
        case .outOfScope:
            return .oosRefusal
        case .inScope:
            break
        }

        if inputs.flags.liveAgentTrigger {
            return .liveAgentEscalation
        }

        if inputs.flags.navigationOnly {
            return .navOnlyDeeplink
        }

        // Removed May 2026: intent-based "isRAGAnswerable" fallback. The
        // signal was redundant with refusal_flags (the model emits
        // hasRagAnswer / navigationOnly / liveAgentTrigger directly) and
        // the 10-class macro_intent head failed probe-set scoring at
        // 43.6% so we no longer ship it. See ADR-021 §6.5.

        if !inputs.flags.hasRagAnswer {
            return .unknownFeature
        }

        if inputs.topRetrievalConfidence < retrievalConfidenceFloor {
            return .unknownFeature
        }

        if inputs.retrievalTopGap < ambiguityGapThreshold
            && inputs.topRetrievalConfidence < 0.75
        {
            return .clarification
        }

        return .ragStepByStep
    }

    /// Convenience overload that takes the Stage A classifier output
    /// AND the real ColBERT retrieval result. This is the production
    /// path per ADR-021 §11.4. The synthetic-retrieval shim from
    /// Phase 3 has been removed — every router call now consults real
    /// retrieval scores.
    ///
    /// When `retrieval` is nil (the dispatcher couldn't load ColBERT
    /// or it errored out), the route falls back to the topic-gate +
    /// refusal-flags-only decision tree, using a permissive synthetic
    /// confidence when `hasRagAnswer` is true. This preserves the
    /// pre-Phase-1b behavior so the chat tab keeps answering when
    /// retrieval is degraded — surfacing the degradation via the
    /// engineering trace, not via silent failures.
    ///
    /// Tests covering this overload double as regression tests for the
    /// underlying `route(_:)` decision tree.
    public static func route(
        stageA: TelcoStageADecision,
        retrieval: ColBERTRetrievalResult?
    ) -> TelcoLane {
        let confidence: Double
        let gap: Double
        if let retrieval, let topConfidence = retrieval.hits.first.map({ _ in retrieval.topConfidence }) {
            confidence = Double(topConfidence)
            gap = Double(retrieval.topGap)
        } else {
            // Retrieval failed or unavailable. Fall back to the
            // permissive synthetic that pre-Phase-1b shipped — so
            // chat tab still answers, but engineering trace will
            // show the missing retrieval signal so the failure mode
            // isn't silent (web research: 73% of production RAG
            // failures are silent retrieval failures — surfacing the
            // missing signal is the explicit anti-pattern fix).
            confidence = stageA.refusalFlags.hasRagAnswer ? 0.95 : 0.0
            gap = stageA.refusalFlags.hasRagAnswer ? 0.5 : 0.0
        }
        return route(TelcoRouterInputs(
            topicGate: stageA.topicGate,
            intent: nil,
            flags: stageA.refusalFlags,
            topRetrievalConfidence: confidence,
            retrievalTopGap: gap
        ))
    }

    /// Legacy overload kept for tests and the engineering probe view
    /// that don't have retrieval data. Routes as if retrieval succeeded
    /// with high confidence when `hasRagAnswer` is true. Production
    /// chat path uses `route(stageA:retrieval:)` above.
    public static func route(stageA: TelcoStageADecision) -> TelcoLane {
        return route(stageA: stageA, retrieval: nil)
    }
}
