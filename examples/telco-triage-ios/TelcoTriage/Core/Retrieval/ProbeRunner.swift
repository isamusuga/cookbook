import Foundation
import os.log

/// One probe-set entry as parsed from `probe-set-v1.json`.
public struct TelcoProbe: Sendable, Equatable, Codable {
    public let query: String
    /// Expected `TelcoLane.wireName` (e.g. `"rag_step_by_step"`).
    public let expectedLane: String
    /// Optional: when set, the retrieved chunk's page_id must start
    /// with this prefix (e.g. `"03."` for any Network-section page).
    /// Catches cross-section retrieval drift even when the lane is
    /// correct.
    public let expectedPagePrefix: String?

    enum CodingKeys: String, CodingKey {
        case query
        case expectedLane = "expected_lane"
        case expectedPagePrefix = "expected_page_prefix"
    }
}

/// One probe's outcome.
public struct TelcoProbeResult: Sendable {
    public let probe: TelcoProbe
    public let actualLane: TelcoLane?
    public let actualPageID: String?
    public let actualSource: TelcoDispatchResult.Source?
    public let latencyMs: Double
    public let lanePass: Bool
    public let pagePass: Bool
    public let error: String?

    /// Overall pass: lane matched AND (no page expectation OR page matched).
    public var passed: Bool { lanePass && pagePass }
}

/// Aggregate result of running the full probe set.
public struct TelcoProbeReport: Sendable {
    public let total: Int
    public let lanePassed: Int
    public let pagePassed: Int
    public let results: [TelcoProbeResult]
    public let totalElapsedMs: Double

    public var laneAccuracy: Double {
        guard total > 0 else { return 0 }
        return Double(lanePassed) / Double(total)
    }

    public var pageAccuracy: Double {
        let denom = results.filter { $0.probe.expectedPagePrefix != nil }.count
        guard denom > 0 else { return 0 }
        return Double(pagePassed) / Double(denom)
    }

    public var summary: String {
        let lane = String(format: "%.0f%%", laneAccuracy * 100)
        let page = String(format: "%.0f%%", pageAccuracy * 100)
        return "lane=\(lanePassed)/\(total) (\(lane)) " +
            "page=\(pagePassed)/\(results.filter { $0.probe.expectedPagePrefix != nil }.count) (\(page)) " +
            "elapsed=\(Int(totalElapsedMs))ms"
    }
}

/// Errors raised when loading or running the probe set.
public enum TelcoProbeRunnerError: Error, LocalizedError {
    case missingResource(String)
    case malformedJSON(String)

    public var errorDescription: String? {
        switch self {
        case .missingResource(let name): return "probe set resource missing: \(name)"
        case .malformedJSON(let msg): return "probe set JSON malformed: \(msg)"
        }
    }
}

/// Runs the bundled probe set against a live `TelcoChatDispatcher`
/// and produces a structured report. Per ADR-021 §11.5 L3, this is
/// the on-device continuous-integration gate for retrieval quality.
///
/// Engineering settings will surface a "Run probe" button that calls
/// this and renders the report. Tests can also invoke with a stub
/// dispatcher to verify scoring math without spinning the model.
public final class TelcoProbeRunner: @unchecked Sendable {
    private let dispatcher: TelcoChatDispatcher
    private let probes: [TelcoProbe]
    private let logger = Logger(
        subsystem: "ai.liquid.demos.telcotriage",
        category: "ProbeRunner"
    )

    public init(dispatcher: TelcoChatDispatcher, probes: [TelcoProbe]) {
        self.dispatcher = dispatcher
        self.probes = probes
    }

    /// Load the bundled probe set. Returns nil-throwing path so the
    /// engineering settings can gracefully report "probe set not
    /// bundled" rather than crashing.
    public static func loadBundledProbes(
        in bundle: Bundle = .main
    ) throws -> [TelcoProbe] {
        guard let url = bundle.url(forResource: "probe-set-v1", withExtension: "json") else {
            throw TelcoProbeRunnerError.missingResource("probe-set-v1.json")
        }
        let data = try Data(contentsOf: url)
        do {
            // The JSON has a top-level { version, probes: [...] } envelope.
            let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            guard let probesRaw = root?["probes"] as? [[String: Any]] else {
                throw TelcoProbeRunnerError.malformedJSON("missing 'probes' array")
            }
            let probesData = try JSONSerialization.data(withJSONObject: probesRaw)
            let probes = try JSONDecoder().decode([TelcoProbe].self, from: probesData)
            return probes
        } catch let err as TelcoProbeRunnerError {
            throw err
        } catch {
            throw TelcoProbeRunnerError.malformedJSON(error.localizedDescription)
        }
    }

    /// Convenience: load bundled probes and construct a runner.
    public static func bundled(
        dispatcher: TelcoChatDispatcher,
        in bundle: Bundle = .main
    ) throws -> TelcoProbeRunner {
        let probes = try loadBundledProbes(in: bundle)
        return TelcoProbeRunner(dispatcher: dispatcher, probes: probes)
    }

    /// Run every probe through the dispatcher; collect results.
    /// Sequential (not concurrent) because the dispatcher serializes
    /// on the shared backend anyway. ~1-2 s per probe on warm caches
    /// for RAG turns; instant for template lanes.
    public func run() async -> TelcoProbeReport {
        let runStart = CFAbsoluteTimeGetCurrent()
        var results: [TelcoProbeResult] = []
        results.reserveCapacity(probes.count)

        for probe in probes {
            results.append(await runOne(probe))
        }

        let totalMs = (CFAbsoluteTimeGetCurrent() - runStart) * 1000
        let lanePassed = results.filter { $0.lanePass }.count
        let pagePassed = results.filter { $0.pagePass && $0.probe.expectedPagePrefix != nil }.count
        let report = TelcoProbeReport(
            total: results.count,
            lanePassed: lanePassed,
            pagePassed: pagePassed,
            results: results,
            totalElapsedMs: totalMs
        )
        logger.info("probe report: \(report.summary, privacy: .public)")
        return report
    }

    private func runOne(_ probe: TelcoProbe) async -> TelcoProbeResult {
        let t0 = CFAbsoluteTimeGetCurrent()
        var finalResult: TelcoDispatchResult?
        var lastLane: TelcoLane?
        var failure: String?
        for await event in dispatcher.dispatch(query: probe.query) {
            switch event {
            case .laneSelected(let lane):
                lastLane = lane
            case .response(let r):
                finalResult = r
            case .failed(let msg):
                failure = msg
            default:
                break
            }
        }
        let elapsedMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        let actualLane = finalResult?.lane ?? lastLane
        let actualPage = finalResult?.retrievedChunk?.pageID
        let lanePass = actualLane?.wireName == probe.expectedLane
        let pagePass: Bool
        if let prefix = probe.expectedPagePrefix {
            pagePass = actualPage?.hasPrefix(prefix) ?? false
        } else {
            pagePass = true  // no expectation = trivially satisfied
        }

        return TelcoProbeResult(
            probe: probe,
            actualLane: actualLane,
            actualPageID: actualPage,
            actualSource: finalResult?.source,
            latencyMs: elapsedMs,
            lanePass: lanePass,
            pagePass: pagePass,
            error: failure
        )
    }
}
