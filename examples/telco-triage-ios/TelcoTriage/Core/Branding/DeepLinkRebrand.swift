import Foundation

/// Translates the internal `telcohome://` URL scheme — baked into Stage B's
/// training distribution, the RAG corpus, the GBNF grammar, and the
/// page-link table — into the user-facing brand scheme at the UI
/// boundary. The internal scheme is preserved end-to-end through Stage A
/// + ColBERT + Stage B + KB fallback, so the trained model stays on its
/// training distribution and format-compliance gates keep passing. Only
/// the rendered Markdown the user reads gets rebranded.
///
/// Design note (from session-054 architectural correction):
/// An earlier attempt bulk-renamed `telcohome://` → `liquid-telco-triage://`
/// in 177 places across data, prompt, grammar, and tests (commit
/// 5b75e55). That worked but pushed Stage B off its training
/// distribution — format compliance started failing on every RAG turn
/// and the dispatcher fell back to KeywordKBExtractor as a workaround,
/// silently degrading the user experience. The correct abstraction is
/// internal-representation vs display-representation: keep the data in
/// the trained scheme, rebrand at the render boundary.
public enum DeepLinkRebrand {
    /// Rewrites `telcohome://X` → `<brand-scheme>://X` for display. Use
    /// when rendering message text that may contain a Stage B deep link
    /// in Markdown form. Idempotent — applies the brand scheme even if
    /// the text already uses it.
    public static func forDisplay(_ text: String, brand: BrandTheme) -> String {
        let target = "\(brand.deepLinkScheme)://"
        guard target != Self.internalScheme else { return text }
        return text.replacingOccurrences(
            of: Self.internalScheme,
            with: target
        )
    }

    /// Rewrites a stored deep-link URL to the display brand. Used by
    /// the DeepLinkChip subtitle so the URL the user sees matches the
    /// brand they've been promised, even though Stage B + the page-link
    /// table stored it in the internal scheme.
    public static func forDisplay(url: String, brand: BrandTheme) -> String {
        forDisplay(url, brand: brand)
    }

    /// Canonical internal scheme. Stage B was trained on this. The
    /// page-link table, GBNF grammar, RAG corpus all use it. Renaming
    /// it requires retraining Stage B.
    public static let internalScheme = "telcohome://"
}
