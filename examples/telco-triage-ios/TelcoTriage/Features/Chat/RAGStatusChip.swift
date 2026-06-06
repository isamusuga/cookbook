import SwiftUI

/// Engineering-mode chip that surfaces the boot-time load status of the
/// local AI answer path. The customer UI hides raw diagnostics; this
/// chip is for build validation and quick simulator/device triage.
///
/// Renders nothing when status is .live AND we're in customer mode —
/// the chip is an engineering signal, not a customer-facing badge.
struct RAGStatusChip: View {
    let status: RAGStackStatus
    let isEngineeringMode: Bool

    @Environment(\.brand) private var brand
    @State private var showingDiagnostic = false

    var body: some View {
        // Customers don't need to see raw runtime diagnostics or raw
        // reasons. Engineering mode keeps the full tap-to-diagnose chip.
        if !isEngineeringMode {
            customerStatus
        } else {
            diagnosticButton
        }
    }

    @ViewBuilder
    private var customerStatus: some View {
        switch status {
        case .live, .degraded:
            EmptyView()
        case .notInitialized:
            HStack(spacing: 6) {
                Image(systemName: "hourglass")
                    .font(.system(size: 11, weight: .semibold))
                Text("Preparing local guide")
                    .font(.caption)
                    .fontWeight(.semibold)
            }
            .foregroundStyle(brand.textSecondary)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(brand.surfaceElevated, in: Capsule())
            .overlay(Capsule().stroke(brand.border, lineWidth: 1))
            .accessibilityLabel("Preparing local guide")
        }
    }

    private var diagnosticButton: some View {
        Button {
            showingDiagnostic = true
        } label: {
            HStack(spacing: 6) {
                Image(systemName: iconName)
                    .font(.system(size: 11, weight: .semibold))
                Text(status.summary)
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(backgroundColor)
            .foregroundStyle(foregroundColor)
            .clipShape(Capsule())
            .overlay(
                Capsule().stroke(borderColor, lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Local AI runtime status. \(status.summary). Tap for diagnostic.")
        .sheet(isPresented: $showingDiagnostic) {
            RAGStatusDiagnosticSheet(status: status)
        }
    }

    private var iconName: String {
        switch status {
        case .live: return "checkmark.circle.fill"
        case .degraded: return "exclamationmark.triangle.fill"
        case .notInitialized: return "hourglass"
        }
    }

    private var backgroundColor: Color {
        switch status {
        case .live: return Color.green.opacity(0.15)
        case .degraded: return Color.orange.opacity(0.18)
        case .notInitialized: return Color.gray.opacity(0.15)
        }
    }

    private var foregroundColor: Color {
        switch status {
        case .live: return .green
        case .degraded: return .orange
        case .notInitialized: return .gray
        }
    }

    private var borderColor: Color {
        switch status {
        case .live: return Color.green.opacity(0.4)
        case .degraded: return Color.orange.opacity(0.5)
        case .notInitialized: return Color.gray.opacity(0.3)
        }
    }
}

/// Modal sheet shown when the chip is tapped. Renders the full
/// `RAGStackStatus.diagnosticText` plus a dismiss button. No
/// remediation buttons — fixes happen out-of-band (re-run
/// bootstrap-models.sh, clean install, etc.). The sheet's only job
/// is to make the underlying failure mode visible.
private struct RAGStatusDiagnosticSheet: View {
    let status: RAGStackStatus

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    HStack(spacing: 10) {
                        Image(systemName: status.isLive
                              ? "checkmark.seal.fill"
                              : "exclamationmark.octagon.fill")
                            .font(.system(size: 28))
                            .foregroundStyle(status.isLive ? .green : .orange)
                        Text(status.isLive
                             ? "Local AI Runtime: LIVE"
                             : "Local AI Runtime: DEGRADED")
                            .font(.system(size: 17, weight: .semibold))
                    }

                    Text(status.diagnosticText)
                        .font(.system(size: 13, design: .monospaced))
                        .foregroundStyle(.primary.opacity(0.9))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)

                    Text("This status is captured once at app launch. To re-check, fully restart the app after applying a fix.")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                        .padding(.top, 8)
                }
                .padding()
            }
            .navigationTitle("Runtime Status")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }
}
