import SwiftUI

/// Carrier-neutral Liquid telco theme. The customer-facing demo should feel
/// like a premium reference app a carrier can imagine reskinning, so the
/// surface is intentionally light, quiet, and not tied to any one operator.
public struct TelcoTriageTheme: BrandTheme {
    public let id = "telco-triage"
    public let displayName = "Liquid Telco"
    public let tagline = "On-device support by Liquid AI"

    public let primary = Color(red: 0.07, green: 0.07, blue: 0.08)
    public let onPrimary = Color.white
    public let accent = Color(red: 0.33, green: 0.43, blue: 0.92)
    public let surfaceBackground = Color(red: 0.965, green: 0.969, blue: 0.976)
    public let surfaceElevated = Color.white
    public let border = Color.black.opacity(0.10)
    public let textPrimary = Color(red: 0.07, green: 0.09, blue: 0.13)
    public let textSecondary = Color(red: 0.31, green: 0.35, blue: 0.42)

    public let success = Color(red: 0.12, green: 0.7, blue: 0.4)
    public let warning = Color(red: 0.95, green: 0.65, blue: 0.0)
    public let danger = Color(red: 0.85, green: 0.2, blue: 0.2)
    public let info = Color(red: 0.2, green: 0.5, blue: 0.9)

    public let titleFont = Font.system(.title3, design: .default).weight(.bold)
    public let bodyFont = Font.system(.body)
    public let monoFont = Font.system(.caption, design: .monospaced)

    public let bubbleCornerRadius: CGFloat = 18
    public let cardCornerRadius: CGFloat = 16

    public let appName = "Liquid Telco"
    public let appSubtitle = "On-device support"
    public let assistantName = "Telco Assistant"
    public let chatPlaceholder = "Ask about Wi-Fi, devices, or your router"
    public let deepLinkScheme = "liquid-telco-triage"
    public let wordmarkSystemImage = "antenna.radiowaves.left.and.right"

    public var welcomeGreeting: @Sendable (String) -> String {
        { name in
            if name.isEmpty {
                return "Hi - I can help with home internet questions, Wi-Fi settings, connected devices, and safe router actions. Everything here runs on-device."
            }
            return "Hi \(name) - I can help with home internet questions, Wi-Fi settings, connected devices, and safe router actions. Everything here runs on-device."
        }
    }

    public init() {}
}

public typealias TelcoTheme = TelcoTriageTheme
