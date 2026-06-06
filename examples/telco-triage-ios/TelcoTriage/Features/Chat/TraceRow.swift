import SwiftUI

/// Permanent 4-cell trace row shown under every assistant message.
/// Renders in a single horizontal scroll so narrow devices still
/// surface every cell without truncation.
///
/// Cells (left to right):
///  - POLICY   : active telco policy lane, or legacy intent when the
///               shared telco classifier is unavailable
///  - SOURCE   : tool id (if any) or "RAG: page-id"
///  - LATENCY  : wall time plus understanding/retrieval/policy/composer split
///  - EGRESS   : "0 bytes ✓" — always, on every path. The demo never
///               leaves the device.
///
/// Source of truth is `CallTrace` on the assistant message. Missing
/// fields render as an em-dash rather than disappearing — the layout
/// stays consistent across turns.
struct TraceRow: View {
    let trace: CallTrace
    let routingPath: RoutingPath

    @Environment(\.brand) private var brand

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            cell(title: policyTitle, value: policyText, subtitle: policySubtitle,
                 tint: intentConfidenceTint)
            cell(title: "Source", value: sourceText, subtitle: sourceSubtitle)
            cell(title: "Latency", value: latencyText, subtitle: tokensText)
            cell(title: "Egress", value: "0 bytes", subtitle: "on-device ✓",
                 tint: brand.success)
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 10)
        .background(brand.surfaceElevated.opacity(0.6),
                    in: RoundedRectangle(cornerRadius: 10))
    }

    /// Confidence band for the intent cell. Visually reinforces the
    /// calibrated-confidence story at a glance without forcing the
    /// demoer to read the numeric score on every turn.
    ///
    /// Thresholds match `ToolDecisionCard.confidenceTint` (>=0.8 success,
    /// >=0.5 warning, <0.5 danger). If a third surface adopts these
    /// bands, extract a shared utility.
    var intentConfidenceTint: Color {
        if trace.telcoUnderstanding != nil {
            guard let route = trace.composerRoute else { return brand.textPrimary }
            switch route {
            case let value where value == ComposerRoute.liveAgent.wireName ||
                value == ComposerRoute.outOfScope.wireName:
                return brand.danger
            case let value where value == ComposerRoute.accountNav.wireName ||
                value == ComposerRoute.clarify.wireName ||
                value == ComposerRoute.noRagAnswer.wireName:
                return brand.warning
            default:
                return brand.success
            }
        }
        switch ConfidenceBand.classify(trace.chatModeConfidence) {
        case .neutral: return brand.textPrimary
        case .high:    return brand.success
        case .medium:  return brand.warning
        case .low:     return brand.danger
        }
    }

    // MARK: - Private view helpers

    private func cell(title: String, value: String, subtitle: String, tint: Color? = nil) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title.uppercased())
                .font(.caption2).fontWeight(.semibold)
                .foregroundStyle(brand.textSecondary)
            Text(value)
                .font(brand.monoFont)
                .fontWeight(.semibold)
                .foregroundStyle(tint ?? brand.textPrimary)
            Text(subtitle)
                .font(.caption2)
                .foregroundStyle(brand.textSecondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var policyTitle: String {
        trace.telcoUnderstanding == nil ? "Intent" : "Policy"
    }

    private var policyText: String {
        if let telco = trace.telcoUnderstanding {
            return trace.composerRoute ?? telco.routingLane.label.rawValue
        }
        return trace.chatMode?.rawValue ?? "—"
    }

    private var policySubtitle: String {
        if let telco = trace.telcoUnderstanding {
            let confidence = String(format: "%.2f", telco.routingLane.confidence)
            let ms = trace.telcoUnderstandingMS ?? Int(telco.totalMs.rounded())
            return "\(telco.routingLane.label.rawValue) · \(confidence) · \(ms)ms"
        }
        guard let c = trace.chatModeConfidence, let ms = trace.chatModeRuntimeMS else {
            return "—"
        }
        return String(format: "%.2f · %dms", c, ms)
    }

    private var sourceText: String {
        switch routingPath {
        case .toolCall:
            if let reasoningConf = trace.toolSelectionConfidence {
                return String(format: "tool · %.2f", reasoningConf)
            }
            return "tool"
        case .answerWithRAG:
            if let pageID = trace.composerCitedPageID {
                return "rag:\(pageID)"
            }
            if let kbID = trace.topKBMatchID {
                return "rag:\(kbID)"
            }
            return "rag"
        case .personalized:
            return "profile"
        case .outOfScope:
            return "unknown"
        }
    }

    private var sourceSubtitle: String {
        switch routingPath {
        case .answerWithRAG:
            if let linkID = trace.composerRenderedLinkID {
                return "evidence · \(linkID)"
            }
            if let score = trace.topKBScore {
                return String(format: "kb hit %.2f", score)
            }
            return "no kb hit"
        case .toolCall:
            return "lfm tool selector"
        case .personalized:
            return "customer context"
        case .outOfScope:
            return "off-topic, declined"
        }
    }

    private var latencyText: String {
        "\(trace.totalMS)ms"
    }

    private var tokensText: String {
        if trace.telcoUnderstandingMS != nil || trace.retrievalMS != nil || trace.routePolicyMS != nil || trace.composerMS != nil {
            let understanding = formatStageMS(trace.telcoUnderstandingMS)
            let retrieval = formatStageMS(trace.retrievalMS)
            let policy = formatStageMS(trace.routePolicyMS)
            let composer = formatStageMS(trace.composerMS ?? trace.inferenceMS)
            return "LFM \(understanding) · RAG \(retrieval)\npolicy \(policy) · compose \(composer)"
        }
        let inTok = trace.inputTokens ?? 0
        let outTok = trace.outputTokens ?? 0
        if inTok == 0 && outTok == 0 { return "—" }
        return "\(inTok) in · \(outTok) out"
    }

    private func formatStageMS(_ value: Int?) -> String {
        guard let value else { return "—" }
        return value == 0 ? "<1ms" : "\(value)ms"
    }
}

/// Pure confidence-band decision — extracted so the tint threshold
/// logic is testable without SwiftUI `Color` equality (which is
/// platform-dependent and flaky in unit tests).
///
/// `nil` and `0.0` both map to `.neutral` so a backend failure
/// (LFMChatModeRouter returns 0.0 on throw) doesn't paint the cell
/// red as if the model disagreed — it's "no signal," not a low score.
///
/// NaN is also neutral by virtue of `c > 0` being false for NaN under
/// IEEE 754.
public enum ConfidenceBand: Equatable {
    case neutral
    case low
    case medium
    case high

    public static func classify(_ confidence: Double?) -> ConfidenceBand {
        guard let c = confidence, c > 0 else { return .neutral }
        if c >= 0.8 { return .high }
        if c >= 0.5 { return .medium }
        return .low
    }
}
