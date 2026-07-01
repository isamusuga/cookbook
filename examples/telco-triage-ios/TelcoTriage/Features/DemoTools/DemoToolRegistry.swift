import Foundation

/// Demo-app convenience factory for the sample Verizon tools.
///
/// Production integrations should construct `ToolRegistry(tools:)` with
/// host-owned tools instead of depending on the cookbook `CustomerContext`.
extension ToolRegistry {
    @MainActor
    public static func demoDefault(customerContext: CustomerContext) -> ToolRegistry {
        ToolRegistry(tools: [
            RestartRouterTool(customerContext: customerContext),
            RunSpeedTestTool(),
            CheckConnectionTool(customerContext: customerContext),
            EnableWPSTool(),
            RunDiagnosticsTool(customerContext: customerContext),
            ScheduleTechnicianTool(customerContext: customerContext),
            ToggleParentalControlsTool(customerContext: customerContext),
            RebootExtenderTool(customerContext: customerContext),
        ])
    }

    @MainActor
    @available(*, deprecated, message: "Use demoDefault(customerContext:) in the demo app, or ToolRegistry(tools:) in host apps.")
    public static func `default`(customerContext: CustomerContext) -> ToolRegistry {
        demoDefault(customerContext: customerContext)
    }
}
