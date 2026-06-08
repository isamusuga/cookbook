import Accelerate
import Foundation
import os.log

/// One scored chunk returned by `ColBERTRetriever.retrieve`.
public struct ColBERTHit: Sendable, Equatable {
    public let chunk: ColBERTChunk
    public let score: Float

    public init(chunk: ColBERTChunk, score: Float) {
        self.chunk = chunk
        self.score = score
    }
}

/// Full retrieval result for one query.
///
/// `hits` are sorted high → low by MaxSim score. `topConfidence` and
/// `topGap` are the inputs `TelcoRagRouter.route(stageA:retrieval:)`
/// reads to decide between `.ragStepByStep`, `.clarification`, and
/// `.unknownFeature`. Per ADR §11.4.3, these replace the synthetic
/// values that `untilRetrievalLandsSyntheticRetrieval` was returning.
public struct ColBERTRetrievalResult: Sendable, Equatable {
    public let hits: [ColBERTHit]
    public let queryTokens: Int
    public let elapsedMs: Double

    public init(hits: [ColBERTHit], queryTokens: Int, elapsedMs: Double) {
        self.hits = hits
        self.queryTokens = queryTokens
        self.elapsedMs = elapsedMs
    }

    /// Top-1 ColBERT MaxSim score, normalized to `[0, 1]` by dividing
    /// by the query length. Unit-vector MaxSim sums up to `queryTokens`
    /// in the best case, so dividing gives a per-token average cosine.
    public var topConfidence: Float {
        guard let top = hits.first, queryTokens > 0 else { return 0 }
        return top.score / Float(queryTokens)
    }

    /// Gap between top-1 and top-2 confidence. A small gap with high
    /// absolute scores indicates ambiguity — the router routes to
    /// `.clarification` in that case (ADR §11.3 / TelcoRagRouter).
    public var topGap: Float {
        guard hits.count >= 2, queryTokens > 0 else { return 1.0 }
        return (hits[0].score - hits[1].score) / Float(queryTokens)
    }
}

/// Errors raised during ColBERT retrieval.
public enum ColBERTRetrievalError: Error, LocalizedError {
    case backendNotLoaded
    case backendFailure(underlying: Error)
    case projectionFailure(underlying: Error)
    case emptyQuery

    public var errorDescription: String? {
        switch self {
        case .backendNotLoaded:
            return "ColBERT backend not loaded — swap to ColBERT GGUF before calling retrieve"
        case .backendFailure(let e): return "ColBERT backend: \(e.localizedDescription)"
        case .projectionFailure(let e): return "ColBERT projection: \(e.localizedDescription)"
        case .emptyQuery: return "empty query"
        }
    }
}

/// Late-interaction (ColBERT MaxSim) retriever over the bundled
/// `ColBERTIndex`. Doesn't own a backend — caller provides one already
/// loaded with `lfm2-colbert-350m-Q4_K_M.gguf`. Per ADR §11.4.3
/// (single-backend topology), the caller is `TelcoChatDispatcher`
/// which `backend.unload()` + `backend.loadModel(path: colbertPath)`
/// before invoking `retrieve` and swaps back afterward.
///
/// MaxSim math (cosine on unit-normalized per-token vectors):
///
///     score(Q, D) = Σ_{q ∈ Q} max_{d ∈ D} (q · d)
///
/// In matrix form, a single `cblas_sgemm` computes `S = Q · D^T` once,
/// then we take the row-wise max and sum. Sub-millisecond for our
/// scale (~10 query tokens × ~120 doc tokens × 128 dim per chunk;
/// 149 chunks total).
public final class ColBERTRetriever: @unchecked Sendable {
    public let index: ColBERTIndex
    public let projection: ColBERTProjection
    public let topK: Int

    private let logger = Logger(
        subsystem: "ai.liquid.demos.telcotriage",
        category: "ColBERTRetriever"
    )

    public init(
        index: ColBERTIndex,
        projection: ColBERTProjection,
        topK: Int = 5
    ) {
        self.index = index
        self.projection = projection
        self.topK = topK
        // Sanity: projection out == index embed dim.
        precondition(
            projection.outFeatures == index.embedDim,
            "projection output (\(projection.outFeatures)) ≠ index embedDim (\(index.embedDim))"
        )
    }

    /// Encode the query with the already-loaded ColBERT backbone +
    /// the Dense projection, then score against every chunk in the
    /// index. Returns top-K hits sorted by score.
    ///
    /// **Multi-turn augmentation (ADR-024 follow-up 2026-05-27).**
    /// When `context.priorAssistantText` is non-nil, it's prepended to
    /// the user's query (sentinel-separated) before tokenization. The
    /// router emits `.augmentRetrievalWithPriorAssistant` on anaphoric /
    /// negative-continuation follow-ups; this is the consumer side.
    /// With `context == .empty` the behaviour is byte-identical to the
    /// previous single-arg signature — back-compat is implicit.
    public func retrieve(
        query: String,
        context: RetrievalContext = .empty,
        via backend: LlamaBackend
    ) async throws -> ColBERTRetrievalResult {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw ColBERTRetrievalError.emptyQuery }

        // Augment the encoding payload via the static composer. Pure
        // function — same `(query, context)` always yields the same
        // payload. Exposed as `composeEncodingPayload(...)` so unit
        // tests can pin the augmentation contract without spinning up
        // a real backend (actors aren't subclassable). Separation of
        // concerns: composer = how to build the encoder input;
        // backend = how to tokenize and embed it.
        let composition = Self.composeEncodingPayload(
            query: trimmed,
            context: context
        )
        let encodingPayload = composition.payload
        let augmented = composition.augmented

        let t0 = CFAbsoluteTimeGetCurrent()

        // 1) Query encoding: per-token hidden states from llama.cpp.
        // Existing `allTokenEmbeddings` returns flat [numTokens * hiddenDim]
        // fp32. ColBERT typically prefixes a [Q] marker token; the
        // LFM2-ColBERT chat_template handles this when we go through
        // the template path. For now we encode the raw text — PyLate's
        // is_query=True uses a query-specific template that we'd need
        // to replicate exactly for parity. A small format gap; calibrate
        // with the probe-set faithfulness/R@5 results in Phase 4.
        let raw: (embeddings: [Float], numTokens: Int, hiddenDim: Int)
        do {
            raw = try await backend.allTokenEmbeddings(prompt: encodingPayload, clearCache: true)
        } catch {
            throw ColBERTRetrievalError.backendFailure(underlying: error)
        }

        guard raw.numTokens > 0 else { throw ColBERTRetrievalError.emptyQuery }
        guard raw.hiddenDim == projection.inFeatures else {
            throw ColBERTRetrievalError.projectionFailure(
                underlying: ColBERTProjectionError.dimensionMismatch(
                    expected: projection.inFeatures,
                    found: raw.hiddenDim
                )
            )
        }

        // 2) Apply 1024→128 projection + L2 normalization per token.
        let queryVecs: [Float]
        do {
            queryVecs = try projection.project(
                hiddenStates: raw.embeddings,
                numTokens: raw.numTokens
            )
        } catch {
            throw ColBERTRetrievalError.projectionFailure(underlying: error)
        }

        // 3) MaxSim score against every chunk. Naive linear scan over
        // the 149-chunk v1 corpus is fine — ~1 ms total at this scale.
        // Hierarchical pruning (PLAID-style) is a future optimization
        // when the corpus grows past ~1k chunks.
        var hits: [ColBERTHit] = []
        hits.reserveCapacity(index.count)
        for i in 0..<index.count {
            guard let chunkSlice = index.chunkVectors(at: i) else { continue }
            let score = Self.maxSim(
                query: queryVecs,
                queryTokens: raw.numTokens,
                doc: chunkSlice.vectors,
                docTokens: chunkSlice.nTokens,
                embedDim: projection.outFeatures
            )
            hits.append(ColBERTHit(chunk: index.chunks[i], score: score))
        }

        // Sort high → low, take top-K.
        hits.sort { $0.score > $1.score }
        let topHits = Array(hits.prefix(topK))

        let elapsedMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        let result = ColBERTRetrievalResult(
            hits: topHits,
            queryTokens: raw.numTokens,
            elapsedMs: elapsedMs
        )

        if let top = topHits.first {
            logger.info(
                "retrieve top=\(top.chunk.chunkID, privacy: .public) score=\(String(format: "%.3f", top.score), privacy: .public) conf=\(String(format: "%.3f", result.topConfidence), privacy: .public) gap=\(String(format: "%.3f", result.topGap), privacy: .public) elapsed=\(String(format: "%.1f", elapsedMs), privacy: .public)ms augmented=\(augmented, privacy: .public)"
            )
        }
        return result
    }

    /// Character cap for the `RetrievalContext.priorAssistantText`
    /// prefix. Roughly 4 chars/token ≈ 200 tokens of prior context
    /// before the user's query — leaves > 1500 tokens for the query
    /// itself in a 2048-ctx window. Empirically tuned: prior assistant
    /// replies in our corpus are 200-600 chars (step-by-step
    /// instructions), so 800 char cap rarely trims real signal.
    public static let priorContextCharBudget = 800

    /// Pure-function composer: assemble the encoding payload the
    /// ColBERT backbone receives, given the user's query plus the
    /// optional cross-turn `RetrievalContext`. Lives in its own
    /// function so unit tests can pin the augmentation rules without
    /// touching a real `LlamaBackend` actor.
    ///
    /// Rules (in order):
    ///   1. If `context.priorAssistantText` is nil / empty / whitespace-
    ///      only → return the bare query, `augmented = false`.
    ///   2. Otherwise trim the prior, cap at `priorContextCharBudget`
    ///      characters, join with a double newline before the query.
    ///      Returns the joined payload, `augmented = true`.
    ///
    /// Why double-newline: it's the only sentinel a generic BPE
    /// tokenizer maps consistently across the LFM2 vocab without
    /// requiring a special template token. The retriever's
    /// `is_query=True` template gap (see line 153 docs) means we don't
    /// emit a `[Q]` marker today either — both segments are encoded as
    /// regular text.
    public struct EncodingPayloadComposition: Equatable {
        public let payload: String
        public let augmented: Bool
    }

    public static func composeEncodingPayload(
        query: String,
        context: RetrievalContext
    ) -> EncodingPayloadComposition {
        let trimmedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let prior = context.priorAssistantText?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !prior.isEmpty else {
            return EncodingPayloadComposition(payload: trimmedQuery, augmented: false)
        }
        let capped = String(prior.prefix(priorContextCharBudget))
        return EncodingPayloadComposition(
            payload: "\(capped)\n\n\(trimmedQuery)",
            augmented: true
        )
    }

    /// MaxSim score between query per-token vectors `Q` and document
    /// per-token vectors `D`, both row-major, unit-normalized.
    ///
    ///     S = Q · D^T   →   shape [queryTokens, docTokens]
    ///     score = Σ_q max_d S[q, d]
    ///
    /// One BLAS call + one O(queryTokens × docTokens) scan. For our
    /// scale (~10 × ~120) this is hundreds of microseconds.
    static func maxSim(
        query: [Float],
        queryTokens: Int,
        doc: [Float],
        docTokens: Int,
        embedDim: Int
    ) -> Float {
        // Pairwise similarity matrix S = Q · D^T, shape [Q, D]
        var sim = [Float](repeating: 0, count: queryTokens * docTokens)
        query.withUnsafeBufferPointer { qPtr in
            doc.withUnsafeBufferPointer { dPtr in
                sim.withUnsafeMutableBufferPointer { sPtr in
                    cblas_sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasTrans,
                        Int32(queryTokens),
                        Int32(docTokens),
                        Int32(embedDim),
                        1.0,
                        qPtr.baseAddress,
                        Int32(embedDim),
                        dPtr.baseAddress,
                        Int32(embedDim),
                        0.0,
                        sPtr.baseAddress,
                        Int32(docTokens)
                    )
                }
            }
        }

        // Row-wise max via vDSP, then sum the maxes.
        var totalScore: Float = 0
        sim.withUnsafeBufferPointer { sPtr in
            guard let base = sPtr.baseAddress else { return }
            for q in 0..<queryTokens {
                var rowMax: Float = 0
                vDSP_maxv(
                    base.advanced(by: q * docTokens),
                    1,
                    &rowMax,
                    vDSP_Length(docTokens)
                )
                totalScore += rowMax
            }
        }
        return totalScore
    }
}
