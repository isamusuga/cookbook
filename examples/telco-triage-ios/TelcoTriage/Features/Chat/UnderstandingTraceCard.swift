import SwiftUI

/// ADR-022 §4.3 Layer 4 — engineering-mode trace card surfacing the
/// full 5-head `QueryUnderstanding` vector for a single chat turn.
///
/// Renders beneath the existing `TraceRow` when `CallTrace.understanding`
/// is non-nil and the user is in engineering mode. Each head occupies
/// one cell. Heads that the v2 bundle doesn't include (or that weren't
/// consulted on this turn) render as an em-dash so the layout stays
/// stable across the v1→v2 rollout.
///
/// Visual conventions match `TraceRow` so the two cards read as a single
/// stack — same brand palette, same mono font, same em-dash for missing.
///
/// Confidence is surfaced for engineering inspection but never gates
/// routing (per ADR-022 §4.3 design principle #1).
struct UnderstandingTraceCard: View {
    let understanding: QueryUnderstanding

    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            header
            grid
            footer
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 10)
        .background(
            brand.surfaceElevated.opacity(0.6),
            in: RoundedRectangle(cornerRadius: 10)
        )
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 6) {
            Image(systemName: "brain.head.profile")
                .font(.caption2)
                .foregroundStyle(brand.textSecondary)
            Text("UNDERSTANDING · \(understanding.strategy.displayName.uppercased())")
                .font(.caption2).fontWeight(.semibold)
                .foregroundStyle(brand.textSecondary)
            Spacer()
            Text(String(format: "%.0f ms", understanding.totalMs))
                .font(brand.monoFont)
                .font(.caption2)
                .foregroundStyle(brand.textSecondary)
        }
    }

    // MARK: - 5-cell grid (chat_mode, topic_gate, refusal_flags, emotional, slots)

    private var grid: some View {
        // 5 cells in a 2-row layout that wraps gracefully on narrow
        // devices — same approach as TraceRow's horizontal arrangement
        // but tilted to vertical when widths shrink.
        LazyVGrid(
            columns: Array(repeating: GridItem(.flexible(), spacing: 8, alignment: .topLeading), count: 3),
            alignment: .leading,
            spacing: 8
        ) {
            cell(
                title: "Chat mode",
                value: chatModeValue,
                subtitle: chatModeSubtitle,
                tint: confidenceTint(understanding.chatMode?.confidence)
            )
            cell(
                title: "Topic gate",
                value: topicGateValue,
                subtitle: topicGateSubtitle,
                tint: confidenceTint(understanding.topicGate?.confidence)
            )
            cell(
                title: "Refusal flags",
                value: refusalFlagsValue,
                subtitle: refusalFlagsSubtitle
            )
            cell(
                title: "Emotional",
                value: emotionalValue,
                subtitle: emotionalSubtitle,
                tint: emotionalTint
            )
            cell(
                title: "Slots",
                value: slotsValue,
                subtitle: slotsSubtitle
            )
        }
    }

    // MARK: - Footer

    private var footer: some View {
        HStack(spacing: 4) {
            Image(systemName: "info.circle")
                .font(.system(size: 9))
                .foregroundStyle(brand.textSecondary.opacity(0.7))
            Text("Heads INFORM. Pure-function router DECIDES. ADR-022.")
                .font(.system(size: 9))
                .foregroundStyle(brand.textSecondary.opacity(0.7))
        }
    }

    // MARK: - Cell builder

    private func cell(
        title: String,
        value: String,
        subtitle: String,
        tint: Color? = nil
    ) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title.uppercased())
                .font(.caption2).fontWeight(.semibold)
                .foregroundStyle(brand.textSecondary)
            Text(value)
                .font(brand.monoFont)
                .fontWeight(.semibold)
                .foregroundStyle(tint ?? brand.textPrimary)
                .lineLimit(2)
                .minimumScaleFactor(0.7)
            Text(subtitle)
                .font(.caption2)
                .foregroundStyle(brand.textSecondary)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Cell content

    private var chatModeValue: String {
        understanding.chatMode?.mode.rawValue ?? "—"
    }
    private var chatModeSubtitle: String {
        guard let c = understanding.chatMode?.confidence else { return "—" }
        return String(format: "%.2f", c)
    }

    private var topicGateValue: String {
        guard let outcome = understanding.topicGate else { return "—" }
        switch outcome.value {
        case .inScope:    return "in_scope"
        case .outOfScope: return "out_of_scope"
        case .greeting:   return "greeting"
        }
    }
    private var topicGateSubtitle: String {
        guard let c = understanding.topicGate?.confidence else { return "—" }
        return String(format: "%.2f", c)
    }

    private var refusalFlagsValue: String {
        guard let outcome = understanding.refusalFlags else { return "—" }
        var parts: [String] = []
        if outcome.value.hasRagAnswer { parts.append("rag") }
        if outcome.value.navigationOnly { parts.append("nav") }
        if outcome.value.liveAgentTrigger { parts.append("agent") }
        return parts.isEmpty ? "none" : parts.joined(separator: "·")
    }
    private var refusalFlagsSubtitle: String {
        guard let probs = understanding.refusalFlags?.probabilities,
              probs.count == 3 else { return "—" }
        return String(format: "%.2f/%.2f/%.2f", probs[0], probs[1], probs[2])
    }

    private var emotionalValue: String {
        understanding.emotionalState?.value.displayName.lowercased() ?? "—"
    }
    private var emotionalSubtitle: String {
        guard let c = understanding.emotionalState?.confidence else { return "—" }
        return String(format: "%.2f", c)
    }
    private var emotionalTint: Color? {
        guard let state = understanding.emotionalState?.value else { return nil }
        switch state {
        case .neutral:    return nil  // default text colour
        case .frustrated: return brand.warning
        case .urgent:     return brand.danger
        }
    }

    private var slotsValue: String {
        guard let outcome = understanding.slotCompleteness else { return "—" }
        let presentSlots = outcome.value.presentSlots
        if presentSlots.isEmpty { return "none" }
        return presentSlots
            .map(\.rawValue)
            .map { $0.replacingOccurrences(of: "has_", with: "") }
            .sorted()
            .joined(separator: "·")
    }
    private var slotsSubtitle: String {
        guard let probs = understanding.slotCompleteness?.probabilities,
              probs.count == 4 else { return "—" }
        return String(format: "d·l·t·a %.2f/%.2f/%.2f/%.2f",
                      probs[0], probs[1], probs[2], probs[3])
    }

    // MARK: - Confidence tint (matches TraceRow's bands)

    private func confidenceTint(_ confidence: Double?) -> Color? {
        switch ConfidenceBand.classify(confidence) {
        case .neutral: return nil
        case .high:    return brand.success
        case .medium:  return brand.warning
        case .low:     return brand.danger
        }
    }
}

/// ADR-026 engineering-mode card for the active Telco Triage runtime:
/// one shared LFM2.5-350M pass with `telco-shared-clf-v1`, followed by
/// nine classifier heads. This is intentionally separate from
/// `UnderstandingTraceCard`, which renders the legacy ADR-022 composite
/// vector.
struct TelcoUnderstandingTraceDisclosure: View {
    let understanding: TelcoSharedUnderstanding
    @Binding var isExpanded: Bool

    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                isExpanded.toggle()
            } label: {
                HStack(spacing: 7) {
                    Image(systemName: "slider.horizontal.3")
                        .font(.caption2)
                        .foregroundStyle(brand.textSecondary)
                    VStack(alignment: .leading, spacing: 1) {
                        Text("LFM UNDERSTANDING · SINGLE FORWARD PASS")
                            .font(.caption2)
                            .fontWeight(.semibold)
                            .foregroundStyle(brand.textSecondary)
                        Text("\(understanding.routingLane.label.rawValue) · \(understanding.requiredTool.label.rawValue)")
                            .font(brand.monoFont)
                            .fontWeight(.semibold)
                            .foregroundStyle(brand.textPrimary)
                            .lineLimit(1)
                            .minimumScaleFactor(0.75)
                    }
                    Spacer()
                    Text(String(format: "%.0f ms", understanding.totalMs))
                        .font(brand.monoFont)
                        .font(.caption2)
                        .foregroundStyle(brand.textSecondary)
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .foregroundStyle(brand.textSecondary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if isExpanded {
                grid
                footer
            }
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 10)
        .background(
            brand.surfaceElevated.opacity(0.6),
            in: RoundedRectangle(cornerRadius: 10)
        )
    }

    private var grid: some View {
        LazyVGrid(
            columns: Array(repeating: GridItem(.flexible(), spacing: 8, alignment: .topLeading), count: 3),
            alignment: .leading,
            spacing: 8
        ) {
            cell(title: "Intent",
                 value: understanding.supportIntent.label.rawValue,
                 subtitle: confidence(understanding.supportIntent.confidence))
            cell(title: "Lane",
                 value: understanding.routingLane.label.rawValue,
                 subtitle: confidence(understanding.routingLane.confidence),
                 tint: laneTint)
            cell(title: "Tool",
                 value: understanding.requiredTool.label.rawValue,
                 subtitle: confidence(understanding.requiredTool.confidence))
            cell(title: "Cloud",
                 value: activeLabels(understanding.cloudRequirements.activeLabels),
                 subtitle: "multi-label")
            cell(title: "Complexity",
                 value: understanding.issueComplexity.label.rawValue,
                 subtitle: confidence(understanding.issueComplexity.confidence))
            cell(title: "Escalation",
                 value: understanding.escalationRisk.label.rawValue,
                 subtitle: confidence(understanding.escalationRisk.confidence),
                 tint: escalationTint)
            cell(title: "PII",
                 value: understanding.piiRisk.label.rawValue,
                 subtitle: confidence(understanding.piiRisk.confidence),
                 tint: piiTint)
            cell(title: "Transcript",
                 value: understanding.transcriptQuality.label.rawValue,
                 subtitle: confidence(understanding.transcriptQuality.confidence))
            cell(title: "Missing slots",
                 value: activeLabels(understanding.missingSlots.activeLabels),
                 subtitle: "values TBD")
        }
    }

    private var footer: some View {
        HStack(alignment: .top, spacing: 4) {
            Image(systemName: "info.circle")
                .font(.system(size: 9))
                .foregroundStyle(brand.textSecondary.opacity(0.7))
            Text("One LFM2.5-350M pass computes all heads. Heads inform; deterministic policy decides. Slot value extraction is TBD.")
                .font(.system(size: 9))
                .foregroundStyle(brand.textSecondary.opacity(0.7))
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func cell(
        title: String,
        value: String,
        subtitle: String,
        tint: Color? = nil
    ) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title.uppercased())
                .font(.caption2)
                .fontWeight(.semibold)
                .foregroundStyle(brand.textSecondary)
            Text(value)
                .font(brand.monoFont)
                .fontWeight(.semibold)
                .foregroundStyle(tint ?? brand.textPrimary)
                .lineLimit(2)
                .minimumScaleFactor(0.65)
            Text(subtitle)
                .font(.caption2)
                .foregroundStyle(brand.textSecondary)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func confidence(_ value: Double) -> String {
        String(format: "%.2f", value)
    }

    private func activeLabels<Label>(_ labels: [Label]) -> String
    where Label: RawRepresentable, Label.RawValue == String {
        if labels.isEmpty { return "none" }
        return labels
            .map(\.rawValue)
            .joined(separator: "·")
    }

    private var laneTint: Color? {
        switch understanding.routingLane.label {
        case .localAnswer, .localTool:
            return brand.success
        case .cloudAssist:
            return brand.warning
        case .humanEscalation:
            return understanding.routingLane.confidence >= TelcoPolicyThreshold.hardDecision
                ? brand.danger
                : brand.warning
        case .blocked:
            return understanding.routingLane.confidence >= TelcoPolicyThreshold.hardDecision
                ? brand.danger
                : brand.warning
        }
    }

    private var escalationTint: Color? {
        switch understanding.escalationRisk.label {
        case .low:
            return nil
        case .frustrated:
            return brand.warning
        case .churnRisk, .complaint, .urgent:
            return understanding.escalationRisk.confidence >= TelcoPolicyThreshold.hardDecision
                ? brand.danger
                : brand.warning
        }
    }

    private var piiTint: Color? {
        if understanding.piiRisk.label == .safe { return nil }
        return understanding.piiRisk.confidence >= TelcoPolicyThreshold.piiBlock
            ? brand.danger
            : brand.warning
    }
}
