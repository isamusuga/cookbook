import Foundation

/// Boot-time load result for the local Telco Triage answer path.
/// Captured once at app launch and surfaced in engineering mode so an
/// operator can see whether the on-device understanding layer, corpus,
/// BM25 retriever, and deterministic composer are wired, or whether
/// the app fell back to a degraded path.
///
/// "Live" means the canonical RAG units loaded and the composer
/// retriever can answer from them. The `embedDim` field remains for the
/// retired ColBERT path and is `0` for BM25/composer. "Degraded" carries
/// a one-line reason string so the tap-to-expand sheet can show the
/// exact missing artifact or initialization failure.
public enum RAGStackStatus: Equatable, Sendable {
    case live(chunkCount: Int, embedDim: Int)
    case degraded(reason: String)
    case notInitialized

    public var isLive: Bool {
        if case .live = self { return true }
        return false
    }

    /// Canonical-unit count when the composer corpus is loaded.
    public var corpusUnitCount: Int? {
        if case .live(let n, _) = self { return n }
        return nil
    }

    /// One-line degraded reason, if boot failed to wire the composer path.
    public var degradedReason: String? {
        if case .degraded(let reason) = self { return reason }
        return nil
    }

    /// One-line label suitable for the engineering header chip.
    /// Keep this short — the full reason can render in a sheet on tap.
    public var summary: String {
        switch self {
        case .live(let n, _):
            return "AI: LFM→RAG→composer (\(n) units)"
        case .degraded(let reason):
            // Trim long error messages so the chip stays single-line.
            // The tap-to-expand sheet shows the full text.
            let max = 60
            let short = reason.count > max
                ? String(reason.prefix(max)) + "…"
                : reason
            return "AI: degraded — \(short)"
        case .notInitialized:
            return "AI: initializing…"
        }
    }

    /// Full multi-line text for the diagnostic sheet that opens when
    /// the operator taps the chip. Always informative, including for
    /// the live case (so engineering mode can confirm corpus size).
    public var diagnosticText: String {
        switch self {
        case .live(let n, let d):
            let retrievalDetail = d > 0
                ? "Retrieval: ColBERT legacy path, embed dim \(d)."
                : "Retrieval: BM25HierarchyRetriever over canonical units."
            return """
            Local AI runtime is live.

            Understanding: LFM2.5-350M + telco-shared-clf-v1 + 9 classifier heads.
            Corpus: \(n) canonical units.
            \(retrievalDetail)
            Answer layer: deterministic composer.

            Stage B, ColBERT, chat-mode-router-v2, topic/refusal LoRAs, \
            and the relational LoRA are not part of the normal customer \
            answer path. They remain legacy / evaluation-only surfaces.
            """
        case .degraded(let reason):
            return """
            Local AI runtime is not fully wired.

            Reason: \(reason)

            Common fixes:
              1. Confirm `rag-units-v1.json` is present in the app bundle.
              2. Confirm `telco-shared-clf-v1` and the nine classifier \
                 heads are present in the app bundle.
              3. Delete the app from the device, clean the build folder \
                 (Cmd+Shift+K), and reinstall — a stale install can \
                 reference pbxproj entries the bundle never received.
            """
        case .notInitialized:
            return "Local AI runtime not yet initialized — boot still in progress."
        }
    }
}
