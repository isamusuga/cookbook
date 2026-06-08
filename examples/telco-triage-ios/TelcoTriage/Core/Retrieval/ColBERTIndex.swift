import Foundation
import os.log

/// One chunk from the bundled Telco RAG corpus.
///
/// Mirrors the per-chunk record in `rag-chunks-v1.json`. The body is the
/// verbatim section text; the contextual prefix is the Anthropic-style
/// preamble that tells ColBERT what document and section the chunk
/// belongs to (ADR-021 §11.4.1). The deep_link is the canonical
/// telcohome:// route for the parent page — what Stage B will emit and
/// what `TelcoLinkResolver` validates.
public struct ColBERTChunk: Sendable, Equatable, Hashable, Identifiable {
    public var id: String { chunkID }

    public let chunkID: String
    public let pageID: String
    public let section: String
    public let title: String
    public let contextualPrefix: String
    public let body: String
    /// Optional because a handful of corpus pages (template detail
    /// pages like Equipment Details that cover router/gateway/extender
    /// variants) don't have one canonical telcohome:// route. Chunks
    /// without a deep link are still retrievable but the Stage B
    /// grounded-generation prompt will need to handle the no-link
    /// branch — answer descriptively without claiming a deep link.
    public let deepLink: String?
    public let deepLinkLabel: String?
    public let nTokens: Int

    public init(
        chunkID: String,
        pageID: String,
        section: String,
        title: String,
        contextualPrefix: String,
        body: String,
        deepLink: String?,
        deepLinkLabel: String?,
        nTokens: Int
    ) {
        self.chunkID = chunkID
        self.pageID = pageID
        self.section = section
        self.title = title
        self.contextualPrefix = contextualPrefix
        self.body = body
        self.deepLink = deepLink
        self.deepLinkLabel = deepLinkLabel
        self.nTokens = nTokens
    }
}

/// Errors raised when loading or validating the bundled RAG index.
public enum ColBERTIndexError: Error, LocalizedError {
    case missingResource(String)
    case invalidMagic(expected: String, found: String)
    case unsupportedVersion(Int)
    case chunkCountMismatch(metadata: Int, binary: Int)
    case truncatedBinary(expected: Int, actual: Int)
    case malformedMetadata(String)

    public var errorDescription: String? {
        switch self {
        case .missingResource(let name): return "RAG resource missing: \(name)"
        case .invalidMagic(let exp, let got): return "RAG index magic mismatch: expected \(exp), got \(got)"
        case .unsupportedVersion(let v): return "RAG index version unsupported: \(v)"
        case .chunkCountMismatch(let m, let b):
            return "RAG chunk count mismatch: metadata=\(m), binary=\(b)"
        case .truncatedBinary(let exp, let act):
            return "RAG index binary truncated: expected \(exp) bytes, got \(act)"
        case .malformedMetadata(let msg): return "RAG metadata malformed: \(msg)"
        }
    }
}

/// Bundled, pre-encoded ColBERT corpus index.
///
/// Loads two artifacts at construction:
///   - `rag-chunks-v1.json` — per-chunk typed metadata (this struct's
///     `chunks` array).
///   - `rag-index-v1.bin` — per-token fp16 vectors, custom binary
///     layout. Slurped into memory on init (4 MB for the 149-chunk
///     v1 corpus, well within mobile budget).
///
/// **Binary layout** (mirrors `scripts/telco/build_rag_index.py` writer):
///
///   HEADER (16 bytes)
///     magic        : 4 bytes ASCII "TLRG"
///     version      : u32 little-endian
///     n_chunks     : u32 little-endian
///     embed_dim    : u32 little-endian (128 for LFM2-ColBERT-350M)
///
///   OFFSET_TABLE (n_chunks × 8 bytes)
///     For each chunk: (offset_u32, n_tokens_u32)
///
///   VECTORS (concatenated fp16 per-token vectors)
///
/// `ColBERTRetriever` consumes this via `chunkVectors(at:)` which
/// returns the per-token slice for a given chunk index without
/// copying.
public final class ColBERTIndex: @unchecked Sendable {
    public let chunks: [ColBERTChunk]
    public let embedDim: Int
    public let corpusVersion: Int
    public let modelID: String

    /// Raw fp16 vectors, contiguous: chunk 0 [tokens × embedDim] then
    /// chunk 1 [tokens × embedDim], etc. Read-only after construction.
    private let vectorBuffer: [Float]

    /// Per-chunk view: each entry is `(start_offset_in_floats, n_tokens)`.
    /// `vectorBuffer[start ..< start + n_tokens * embedDim]` is the
    /// chunk's flattened per-token vectors.
    private let chunkOffsets: [(start: Int, nTokens: Int)]

    private let logger = Logger(
        subsystem: "ai.liquid.demos.telcotriage",
        category: "ColBERTIndex"
    )

    public init(
        chunksURL: URL,
        indexURL: URL
    ) throws {
        // ---- 1. Load + parse metadata JSON ----
        let metaData = try Data(contentsOf: chunksURL)
        guard let json = try JSONSerialization.jsonObject(with: metaData) as? [String: Any]
        else { throw ColBERTIndexError.malformedMetadata("rag-chunks-v1.json is not a JSON object") }
        guard let version = json["version"] as? Int else {
            throw ColBERTIndexError.malformedMetadata("missing version")
        }
        guard version == 1 else { throw ColBERTIndexError.unsupportedVersion(version) }
        let embedDim = (json["embed_dim"] as? Int) ?? 128
        let modelID = (json["model_id"] as? String) ?? "unknown"

        guard let rawChunks = json["chunks"] as? [[String: Any]] else {
            throw ColBERTIndexError.malformedMetadata("missing chunks array")
        }

        var parsed: [ColBERTChunk] = []
        parsed.reserveCapacity(rawChunks.count)
        for raw in rawChunks {
            guard
                let chunkID = raw["chunk_id"] as? String,
                let pageID = raw["page_id"] as? String,
                let section = raw["section"] as? String,
                let title = raw["title"] as? String,
                let prefix = raw["contextual_prefix"] as? String,
                let body = raw["body"] as? String,
                let nTokens = raw["n_tokens"] as? Int
            else {
                throw ColBERTIndexError.malformedMetadata(
                    "chunk record missing required keys: \(raw["chunk_id"] ?? "?")"
                )
            }
            // deep_link is intentionally optional — a handful of corpus
            // pages (template detail pages) don't have a canonical
            // telcohome:// route. See ColBERTChunk.deepLink for the
            // semantic.
            parsed.append(
                ColBERTChunk(
                    chunkID: chunkID,
                    pageID: pageID,
                    section: section,
                    title: title,
                    contextualPrefix: prefix,
                    body: body,
                    deepLink: raw["deep_link"] as? String,
                    deepLinkLabel: raw["deep_link_label"] as? String,
                    nTokens: nTokens
                )
            )
        }
        self.chunks = parsed
        self.embedDim = embedDim
        self.corpusVersion = version
        self.modelID = modelID

        // ---- 2. Load + parse binary index ----
        let binData = try Data(contentsOf: indexURL)
        let headerSize = 16
        let perChunkOffsetBytes = 8
        guard binData.count >= headerSize else {
            throw ColBERTIndexError.truncatedBinary(expected: headerSize, actual: binData.count)
        }

        let magic = binData.subdata(in: 0..<4)
        let magicStr = String(data: magic, encoding: .ascii) ?? "?"
        guard magicStr == "TLRG" else {
            throw ColBERTIndexError.invalidMagic(expected: "TLRG", found: magicStr)
        }

        // Header fields, little-endian u32
        let binVersion = Self.readU32(binData, at: 4)
        let nChunksFromBin = Int(Self.readU32(binData, at: 8))
        let embedDimFromBin = Int(Self.readU32(binData, at: 12))

        guard binVersion == 1 else {
            throw ColBERTIndexError.unsupportedVersion(Int(binVersion))
        }
        guard nChunksFromBin == parsed.count else {
            throw ColBERTIndexError.chunkCountMismatch(
                metadata: parsed.count, binary: nChunksFromBin
            )
        }
        guard embedDimFromBin == embedDim else {
            throw ColBERTIndexError.malformedMetadata(
                "embed_dim mismatch: meta=\(embedDim), binary=\(embedDimFromBin)"
            )
        }

        // Parse offset table
        let offsetsStart = headerSize
        let offsetsEnd = offsetsStart + nChunksFromBin * perChunkOffsetBytes
        guard binData.count >= offsetsEnd else {
            throw ColBERTIndexError.truncatedBinary(expected: offsetsEnd, actual: binData.count)
        }

        var offsetRecords: [(start: Int, nTokens: Int)] = []
        offsetRecords.reserveCapacity(nChunksFromBin)
        var floatCursor = 0  // counts FLOATS into the vectors region
        for i in 0..<nChunksFromBin {
            let recOffset = offsetsStart + i * perChunkOffsetBytes
            let byteOffset = Int(Self.readU32(binData, at: recOffset))
            let nTokens = Int(Self.readU32(binData, at: recOffset + 4))

            // Verify the header's byte offset matches what we'd compute
            // from the cumulative float count so far. Floats are fp16
            // on disk (2 bytes each); vectors region starts after the
            // header + offset table.
            let expectedByteOffset = offsetsEnd + floatCursor * MemoryLayout<UInt16>.size
            guard byteOffset == expectedByteOffset else {
                throw ColBERTIndexError.malformedMetadata(
                    "chunk \(i) offset mismatch: header=\(byteOffset), computed=\(expectedByteOffset)"
                )
            }
            // `start` indexes into the fp32-expanded vectorBuffer below,
            // which is also in floats — same cursor value.
            offsetRecords.append((start: floatCursor, nTokens: nTokens))
            floatCursor += nTokens * embedDim
        }
        self.chunkOffsets = offsetRecords

        // ---- 3. Parse vectors region (fp16 → fp32 expansion) ----
        let totalFloats = floatCursor
        let expectedVectorsBytes = totalFloats * MemoryLayout<UInt16>.size
        let actualVectorsBytes = binData.count - offsetsEnd
        guard actualVectorsBytes == expectedVectorsBytes else {
            throw ColBERTIndexError.truncatedBinary(
                expected: offsetsEnd + expectedVectorsBytes,
                actual: binData.count
            )
        }

        var floats = [Float](repeating: 0, count: totalFloats)
        binData.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            let fp16Start = base.advanced(by: offsetsEnd)
                .bindMemory(to: UInt16.self, capacity: totalFloats)
            // Expand fp16 → fp32 via Accelerate's vImageConvert_Planar16FtoPlanarF.
            // For simplicity here we go through Float16; Accelerate is
            // an option for the v2 perf pass.
            for i in 0..<totalFloats {
                let bits = fp16Start[i]
                let f16 = Float16(bitPattern: bits)
                floats[i] = Float(f16)
            }
        }
        self.vectorBuffer = floats

        logger.info(
            "loaded: \(parsed.count, privacy: .public) chunks, embed_dim=\(embedDim, privacy: .public), \(totalFloats, privacy: .public) total floats (\(binData.count / 1024 / 1024, privacy: .public) MB on disk)"
        )
    }

    /// Convenience: load both artifacts from the main bundle. Returns
    /// nil when either is missing (fresh clones without bootstrap).
    public static func bundled(in bundle: Bundle = .main) throws -> ColBERTIndex {
        guard let chunksURL = bundle.url(forResource: "rag-chunks-v1", withExtension: "json") else {
            throw ColBERTIndexError.missingResource("rag-chunks-v1.json")
        }
        guard let indexURL = bundle.url(forResource: "rag-index-v1", withExtension: "bin") else {
            throw ColBERTIndexError.missingResource("rag-index-v1.bin")
        }
        return try ColBERTIndex(chunksURL: chunksURL, indexURL: indexURL)
    }

    /// Number of chunks in the corpus.
    public var count: Int { chunks.count }

    /// Per-token vector slice for the chunk at the given index. Caller
    /// gets a 1-D view of length `nTokens * embedDim`; reshape on
    /// consume. Bounds-checked, returns nil on out-of-range.
    public func chunkVectors(at index: Int) -> (vectors: [Float], nTokens: Int)? {
        guard chunks.indices.contains(index) else { return nil }
        let off = chunkOffsets[index]
        let end = off.start + off.nTokens * embedDim
        guard end <= vectorBuffer.count else { return nil }
        let slice = Array(vectorBuffer[off.start..<end])
        return (vectors: slice, nTokens: off.nTokens)
    }

    /// Per-token vector slice for a chunk by `chunkID`. Slower than
    /// `at:` — for repeated calls, cache the index from `chunks.firstIndex`.
    public func chunkVectors(forChunkID chunkID: String) -> (vectors: [Float], nTokens: Int)? {
        guard let i = chunks.firstIndex(where: { $0.chunkID == chunkID }) else { return nil }
        return chunkVectors(at: i)
    }

    // MARK: - Binary parsing helpers

    private static func readU32(_ data: Data, at offset: Int) -> UInt32 {
        return data.withUnsafeBytes { raw -> UInt32 in
            guard let base = raw.baseAddress else { return 0 }
            return base.advanced(by: offset).load(as: UInt32.self).littleEndian
        }
    }
}
