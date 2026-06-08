import Foundation

/// Faithfulness measurement between a Stage B response and the chunk
/// it was supposed to ground in. Implements the v1 of ADR-021 §11.4.2
/// — token-overlap pre-filter only. Reverse-retrieval via ColBERT and
/// HHEM-2.1 NLI grading are documented v2 + v3 paths that require
/// topology changes (dual-backend resident OR a third bundled model)
/// outside the scope of Phase 3.
///
/// First-principles question: did Stage B answer FROM the retrieved
/// chunk, or did it paraphrase from training memorization?
///
/// Token overlap is a weak-but-cheap signal:
///   - Stage B that grounded in the chunk → text contains chunk
///     keywords (the link, step names, button labels — verbatim or
///     near-verbatim).
///   - Stage B that paraphrased from training → uses different
///     wording; overlap drops below a calibrated floor.
///
/// Per RAGAS production guidance ([Digital Applied 2026]):
///   - Faithfulness below 70% = unsafe for production
///   - 70-90% = usable with explicit "verify source" UX
///   - >90% = citation-trust UX
///
/// The token-overlap threshold here is intentionally LOW (~0.1
/// Jaccard) so it catches only egregious drift. False-positive
/// flags on borderline cases are worse than ungrounded answers
/// for v1 — when we have a real reverse-retrieval path we can
/// raise the threshold.
public struct FaithfulnessScore: Sendable, Equatable {
    /// Jaccard overlap between response 2-grams and chunk 2-grams.
    /// Range `[0, 1]`. Strong-faithful Telco responses score
    /// ~0.30-0.50; drift cases land ~0.05.
    public let bigramJaccard: Double

    /// Whether the response passed the v1 faithfulness gate.
    /// Today: `bigramJaccard >= floor`. v2 will compound this with
    /// reverse-retrieval MaxSim and (optionally) HHEM NLI.
    public let isFaithful: Bool

    /// The floor used for the decision. Surfaced so engineering trace
    /// can show what threshold a borderline turn was judged against.
    public let floor: Double

    public init(bigramJaccard: Double, floor: Double) {
        self.bigramJaccard = bigramJaccard
        self.floor = floor
        self.isFaithful = bigramJaccard >= floor
    }
}

public enum FaithfulnessScorer {
    /// v1 floor: ≥0.10 Jaccard 2-gram overlap. Calibrated against a
    /// hand-labeled sample: faithful Telco step-format responses
    /// against their grounding chunk score ~0.30-0.50; clear drift
    /// cases (Stage B paraphrased from training) score ~0.05. The
    /// 0.10 floor catches egregious drift without false-positiving
    /// on terse responses (a 3-step instruction has few 2-grams,
    /// so even faithful ones can score modestly).
    ///
    /// Tighten when reverse-retrieval lands in v2 — that primary
    /// signal will let us raise this to ~0.20 without false positives.
    public static let defaultFloor: Double = 0.10

    /// Score Stage B's response against the chunk it was supposed
    /// to ground in. When `chunk` is nil (ungrounded mode), returns
    /// a `isFaithful = true` result — there's no ground truth to be
    /// unfaithful to.
    public static func score(
        response: String,
        chunk: ColBERTChunk?,
        floor: Double = defaultFloor
    ) -> FaithfulnessScore {
        guard let chunk else {
            return FaithfulnessScore(bigramJaccard: 1.0, floor: floor)
        }
        let jaccard = bigramJaccard(
            response: response,
            chunkText: chunk.body
        )
        return FaithfulnessScore(bigramJaccard: jaccard, floor: floor)
    }

    /// Jaccard similarity between the bigram sets of two strings.
    ///
    /// Normalization:
    ///   - Lowercase
    ///   - Strip punctuation (replace with space)
    ///   - Collapse whitespace
    ///   - Split on whitespace into unigrams
    ///   - Generate 2-grams from consecutive unigrams
    ///   - Filter out stopword-pair bigrams (e.g., "of the", "to a")
    ///     which inflate spurious overlap on any English text
    ///
    /// Returns 0 when either side has no bigrams (very short response
    /// or empty chunk). Internal — exposed for tests.
    static func bigramJaccard(response: String, chunkText: String) -> Double {
        let responseBigrams = bigrams(response)
        let chunkBigrams = bigrams(chunkText)
        guard !responseBigrams.isEmpty, !chunkBigrams.isEmpty else { return 0 }
        let intersection = responseBigrams.intersection(chunkBigrams)
        let union = responseBigrams.union(chunkBigrams)
        return Double(intersection.count) / Double(union.count)
    }

    /// Set of 2-grams after normalization + stopword-pair filtering.
    static func bigrams(_ text: String) -> Set<String> {
        let normalized = text
            .lowercased()
            .replacingOccurrences(
                of: #"[^a-z0-9\s]"#,
                with: " ",
                options: .regularExpression
            )
        let unigrams = normalized
            .split(whereSeparator: { $0.isWhitespace })
            .map(String.init)
        guard unigrams.count >= 2 else { return [] }
        var result: Set<String> = []
        for i in 0..<(unigrams.count - 1) {
            let a = unigrams[i]
            let b = unigrams[i + 1]
            // Drop stopword-pair bigrams that inflate spurious overlap
            // on any English text. Conservative list — common
            // function-word pairs only.
            if stopwords.contains(a) && stopwords.contains(b) { continue }
            result.insert("\(a) \(b)")
        }
        return result
    }

    private static let stopwords: Set<String> = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "to", "of", "in", "on", "at", "by", "for", "with", "from",
        "and", "or", "but", "if", "then", "so", "as",
        "it", "this", "that", "these", "those",
        "you", "your", "yours",
    ]
}
