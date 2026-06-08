import Accelerate
import Foundation
import os.log

/// Errors raised when loading the bundled ColBERT projection matrix.
public enum ColBERTProjectionError: Error, LocalizedError {
    case missingResource(String)
    case invalidMagic(expected: String, found: String)
    case unsupportedVersion(Int)
    case truncatedBinary(expected: Int, actual: Int)
    case dimensionMismatch(expected: Int, found: Int)

    public var errorDescription: String? {
        switch self {
        case .missingResource(let name): return "ColBERT projection resource missing: \(name)"
        case .invalidMagic(let exp, let got):
            return "ColBERT projection magic mismatch: expected \(exp), got \(got)"
        case .unsupportedVersion(let v):
            return "ColBERT projection version unsupported: \(v)"
        case .truncatedBinary(let exp, let act):
            return "ColBERT projection binary truncated: expected \(exp) bytes, got \(act)"
        case .dimensionMismatch(let exp, let got):
            return "ColBERT projection dim mismatch: expected \(exp), got \(got)"
        }
    }
}

/// The 1024→128 Dense projection extracted from LFM2-ColBERT-350M's
/// `1_Dense` module. Applied in Swift via Accelerate's BLAS after
/// `LlamaBackend.allTokenEmbeddings` returns 1024-dim per-token
/// hidden states.
///
/// We split the projection out of the GGUF (rather than baking it as
/// an LM head) so the GGUF stays a standard `Lfm2Model` convertible
/// by llama.cpp's `convert_hf_to_gguf.py`. The 256 KB projection
/// ships as a tiny separate asset.
///
/// **Binary layout** (mirrors `scripts/telco/build_rag_index.py`):
///
///   HEADER (16 bytes)
///     magic           : 4 bytes ASCII "TLPJ"
///     version         : u32 little-endian
///     out_features    : u32 little-endian (128)
///     in_features     : u32 little-endian (1024)
///
///   WEIGHTS
///     out_features × in_features × fp16, row-major
///
/// Output is L2-normalized per token (ColBERT MaxSim is cosine over
/// the projected space).
public final class ColBERTProjection: @unchecked Sendable {
    public let outFeatures: Int  // 128
    public let inFeatures: Int   // 1024

    /// Row-major projection matrix [out × in], fp32 expanded from fp16.
    private let weights: [Float]

    private let logger = Logger(
        subsystem: "ai.liquid.demos.telcotriage",
        category: "ColBERTProjection"
    )

    public init(url: URL) throws {
        let data = try Data(contentsOf: url)
        let headerSize = 16
        guard data.count >= headerSize else {
            throw ColBERTProjectionError.truncatedBinary(
                expected: headerSize, actual: data.count
            )
        }

        let magic = String(data: data.subdata(in: 0..<4), encoding: .ascii) ?? "?"
        guard magic == "TLPJ" else {
            throw ColBERTProjectionError.invalidMagic(expected: "TLPJ", found: magic)
        }

        let version = Self.readU32(data, at: 4)
        let outFeat = Int(Self.readU32(data, at: 8))
        let inFeat = Int(Self.readU32(data, at: 12))
        guard version == 1 else {
            throw ColBERTProjectionError.unsupportedVersion(Int(version))
        }

        let weightCount = outFeat * inFeat
        let expectedBytes = headerSize + weightCount * MemoryLayout<UInt16>.size
        guard data.count == expectedBytes else {
            throw ColBERTProjectionError.truncatedBinary(
                expected: expectedBytes, actual: data.count
            )
        }

        // Expand fp16 → fp32 row-major. Caching as fp32 trades 2× memory
        // (~1 MB) for cblas_sgemm with native fp32, which is the fast
        // path in Accelerate. fp16 GEMM via vDSP_BNNS is an option for
        // a v2 perf pass if 1 MB starts to matter.
        var floats = [Float](repeating: 0, count: weightCount)
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            let fp16Start = base.advanced(by: headerSize)
                .bindMemory(to: UInt16.self, capacity: weightCount)
            for i in 0..<weightCount {
                floats[i] = Float(Float16(bitPattern: fp16Start[i]))
            }
        }

        self.outFeatures = outFeat
        self.inFeatures = inFeat
        self.weights = floats
        logger.info(
            "loaded projection: [\(outFeat, privacy: .public) × \(inFeat, privacy: .public)] from \(data.count, privacy: .public) bytes"
        )
    }

    public static func bundled(in bundle: Bundle = .main) throws -> ColBERTProjection {
        guard let url = bundle.url(forResource: "colbert-projection-v1", withExtension: "bin") else {
            throw ColBERTProjectionError.missingResource("colbert-projection-v1.bin")
        }
        return try ColBERTProjection(url: url)
    }

    /// Project [numTokens × inFeatures] hidden states down to
    /// [numTokens × outFeatures], then L2-normalize each row.
    ///
    /// Math: `projected = hiddenStates @ weights^T`, then `row /= |row|`.
    /// `cblas_sgemm` with `NoTrans, Trans` does the matmul in one call.
    ///
    /// - Parameter hiddenStates: flat row-major [numTokens × inFeatures]
    /// - Parameter numTokens: number of token rows
    /// - Returns: flat row-major [numTokens × outFeatures], unit-norm rows
    public func project(hiddenStates: [Float], numTokens: Int) throws -> [Float] {
        guard hiddenStates.count == numTokens * inFeatures else {
            throw ColBERTProjectionError.dimensionMismatch(
                expected: numTokens * inFeatures, found: hiddenStates.count
            )
        }
        var out = [Float](repeating: 0, count: numTokens * outFeatures)

        // C = A @ B^T where:
        //   A = hiddenStates,  shape [M=numTokens, K=inFeatures], lda=K
        //   B = weights,       shape [N=outFeatures, K=inFeatures], ldb=K
        //   C = out,           shape [M=numTokens, N=outFeatures], ldc=N
        // cblas_sgemm(order, transA, transB, M, N, K, alpha, A, lda,
        //             B, ldb, beta, C, ldc)
        hiddenStates.withUnsafeBufferPointer { aPtr in
            weights.withUnsafeBufferPointer { bPtr in
                out.withUnsafeMutableBufferPointer { cPtr in
                    cblas_sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasTrans,
                        Int32(numTokens),
                        Int32(outFeatures),
                        Int32(inFeatures),
                        1.0,
                        aPtr.baseAddress,
                        Int32(inFeatures),
                        bPtr.baseAddress,
                        Int32(inFeatures),
                        0.0,
                        cPtr.baseAddress,
                        Int32(outFeatures)
                    )
                }
            }
        }

        // L2-normalize each row (ColBERT MaxSim is cosine on unit vectors).
        Self.l2NormalizeRows(&out, numRows: numTokens, dim: outFeatures)
        return out
    }

    /// In-place row-wise L2 normalization. Each row of length `dim`
    /// becomes a unit vector. vDSP makes this two passes per row
    /// (sum-of-squares, then divide).
    static func l2NormalizeRows(_ matrix: inout [Float], numRows: Int, dim: Int) {
        matrix.withUnsafeMutableBufferPointer { mPtr in
            guard let base = mPtr.baseAddress else { return }
            for r in 0..<numRows {
                let rowStart = base.advanced(by: r * dim)
                var sumSq: Float = 0
                vDSP_svesq(rowStart, 1, &sumSq, vDSP_Length(dim))
                let norm = sqrt(sumSq)
                if norm > 1e-8 {
                    var inv = 1.0 / norm
                    vDSP_vsmul(rowStart, 1, &inv, rowStart, 1, vDSP_Length(dim))
                }
                // else: zero vector, leave alone — MaxSim against zero
                // contributes nothing, no harm done.
            }
        }
    }

    private static func readU32(_ data: Data, at offset: Int) -> UInt32 {
        return data.withUnsafeBytes { raw -> UInt32 in
            guard let base = raw.baseAddress else { return 0 }
            return base.advanced(by: offset).load(as: UInt32.self).littleEndian
        }
    }
}
