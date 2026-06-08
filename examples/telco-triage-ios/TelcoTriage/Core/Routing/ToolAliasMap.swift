import Foundation

/// Swift port of `scripts/telco/eval/multi_turn_acceptance.py::ToolAliasMap`
/// (Step 5b Pre-flight Fix A).
///
/// The corpus `link_id` doesn't always match the `ToolIntent.toolID` that
/// the iOS dispatcher would resolve via `ToolIntent(toolID:)`. Two real-world
/// gaps the Step 5b harness exposed:
///
/// * Corpus `link_id="speed-test"` (page `01.02`) — no `ToolIntent` accepts
///   the short form; the canonical id is `"run-speed-test"`. Without this
///   alias, "run a speed test" routes to `.ragAnswer` instead of
///   `.toolAction`, silently breaking the demo's speed-test card.
/// Shared `link_id`s are intentionally not aliased. For example,
/// `link_id="home"` covers many parental-controls pages, but the only
/// registered parental-controls tool pauses/restores internet access
/// for a device. It cannot create a profile. Aliasing the shared link
/// would manufacture a confirmation handshake for a capability that
/// does not exist.
///
/// **Critically**, `share-wifi` is NOT aliased. The corpus has a
/// `share-wifi` `link_id` on page 03.02 with affordance `tool_action`,
/// but no `ToolIntent` is registered. The user's Step 5b brief is
/// explicit: "share-wifi only if a real executable tool exists; otherwise
/// keep as rag_answer/deep link". Adding the alias here would create a
/// confirmation handshake for a non-existent tool.

/// A single `link_id → tool_id` entry, with an optional gate that
/// suppresses the tool offer for non-imperative queries.
public struct ToolAlias: Sendable, Equatable {
    /// Corpus `link_id` (as found on `RAGUnit.linkID`).
    public let linkID: String
    /// Canonical `ToolIntent.toolID` (hyphenated) to resolve through
    /// `ToolIntent(toolID:)`.
    public let toolID: String
    /// When `true`, the alias only resolves to a tool for
    /// `inferQueryMood(query) == .actionImperative`. Question form stays
    /// on `.ragAnswer` rather than being upgraded to
    /// `.answerPlusAction`. Used for shared `link_id`s that cover both
    /// tool-action pages and informational pages (e.g. `link_id="home"`
    /// across the 13.xx parental-controls subpages).
    public let imperativeOnly: Bool

    public init(linkID: String, toolID: String, imperativeOnly: Bool = false) {
        self.linkID = linkID
        self.toolID = toolID
        self.imperativeOnly = imperativeOnly
    }
}

/// Read-only lookup table: corpus `link_id` → `ToolAlias`. Built once at
/// `AppState.buildLFMStack` time and threaded into the dispatcher.
public final class ToolAliasMap: Sendable {
    private let aliasesByLinkID: [String: ToolAlias]

    public init(aliases: [ToolAlias]) {
        var dict: [String: ToolAlias] = [:]
        dict.reserveCapacity(aliases.count)
        for alias in aliases {
            dict[alias.linkID] = alias
        }
        aliasesByLinkID = dict
    }

    /// Returns the alias for the given corpus `link_id`, or `nil` if no
    /// alias exists. Callers pair this with `ToolRegistry.tool(for:)` to
    /// get the actual `Tool` instance.
    public func alias(forLinkID linkID: String?) -> ToolAlias? {
        guard let linkID else { return nil }
        return aliasesByLinkID[linkID]
    }

    /// The production map — byte-equivalent to
    /// `build_default_alias_map()` in the Python harness. Mirrors the
    /// 8 registered `ToolIntent` cases plus the `speed-test` non-identity
    /// alias. Shared navigation links are left out unless there is an
    /// exact executable capability behind them.
    @MainActor
    public static func `default`() -> ToolAliasMap {
        ToolAliasMap(aliases: [
            ToolAlias(linkID: "restart-router", toolID: "restart-router"),
            ToolAlias(linkID: "speed-test", toolID: "run-speed-test"),
            ToolAlias(linkID: "run-speed-test", toolID: "run-speed-test"),
            ToolAlias(linkID: "check-connection", toolID: "check-connection"),
            ToolAlias(linkID: "enable-wps", toolID: "enable-wps"),
            ToolAlias(linkID: "run-diagnostics", toolID: "run-diagnostics"),
            ToolAlias(linkID: "schedule-technician", toolID: "schedule-technician"),
            ToolAlias(linkID: "reboot-extender", toolID: "reboot-extender"),
        ])
    }
}
