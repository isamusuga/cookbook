import Foundation

/// Swift port of `scripts/vz/answer_composer.py::DeterministicComposer`.
///
/// **Doctrine** (Step 5 decision record):
///
/// > The generator should verbalize already-selected evidence; it
/// > should not discover evidence, invent links, or decide citations.
///
/// The composer trusts its inputs unconditionally:
///
/// * `evidence` — the canonical RAG unit (picked by the retriever or
///   by the test oracle). The composer never re-ranks and never opens
///   `body` for new claims.
/// * `route` — already decided by the dispatcher
///   (`VerizonChatDispatcher.deriveRoute`) using the ToolRegistry
///   gate (guardrail #3). The composer never re-routes.
/// * `requiresConfirmation` — already decided. The composer toggles
///   a single confirmation clause; it NEVER invokes any tool.
///
/// **Link guarantee**: every `vzhome://` URL the composer emits comes
/// from `evidence.canonicalURL`. The only other URLs allowed are the
/// fixed external set (My Verizon, Verizon internet, tel:).
public enum AnswerComposerConstants {
    public static let verizonInternetURL = "https://www.verizon.com/home/internet"
    public static let myVerizonURL = "https://m.vzw.com/wMM0jUjn"
    public static let liveAgentPhone = "tel://+18009220204"
}

public protocol AnswerComposing: Sendable {
    func compose(
        query: String,
        route: ComposerRoute,
        evidence: RAGUnit?,
        requiresConfirmation: Bool?,
        history: [ConversationTurnSnippet],
        expectedPolicyLinkID: String?
    ) -> ComposedAnswer
}

public struct DeterministicAnswerComposer: AnswerComposing {
    public let name: String

    public init(name: String = "deterministic_composer") {
        self.name = name
    }

    public func compose(
        query: String,
        route: ComposerRoute,
        evidence: RAGUnit?,
        requiresConfirmation: Bool? = nil,
        history: [ConversationTurnSnippet] = [],
        expectedPolicyLinkID: String? = nil
    ) -> ComposedAnswer {
        _ = history  // unused — composer does not condition on history
        _ = expectedPolicyLinkID
        let start = CFAbsoluteTimeGetCurrent()

        var usedFallback = false
        var citedPageID: String? = evidence?.pageID
        var expectedLinkURL: String? = evidence?.canonicalURL

        var text = ""
        var hasStepChain = false

        switch route {
        case .greeting:
            text = "Hello! How can I assist you with Verizon Home Internet today?"
            expectedLinkURL = nil
        case .outOfScope:
            text = "I'm here to help with topics related to Verizon Home Internet. Please try asking a different question."
            expectedLinkURL = nil
        case .noRagAnswer:
            text = "It looks like I don't have specific information about that. You can check [Verizon Home Internet](\(AnswerComposerConstants.verizonInternetURL)) for more details."
            expectedLinkURL = AnswerComposerConstants.verizonInternetURL
        case .liveAgent:
            text = "I can connect you with a Verizon support agent. [Call us](\(AnswerComposerConstants.liveAgentPhone)) for support."
            expectedLinkURL = AnswerComposerConstants.liveAgentPhone
        case .clarify:
            let hint = clarifyHintFromQuery(query)
            text = "Could you clarify what you're asking about? Are you referring to \(hint)?"
            expectedLinkURL = nil
        case .accountNav:
            text = "To manage your account, please use the My Verizon App: [My Verizon](\(AnswerComposerConstants.myVerizonURL))."
            expectedLinkURL = AnswerComposerConstants.myVerizonURL
        case .ragAnswer, .answerPlusAction, .toolAction:
            if let unit = evidence {
                let task = taskPhrase(from: unit)
                let label = unit.displayLabel
                let url = unit.canonicalURL
                let intro = introForRoute(route, task: task)
                if let focused = renderFocusedStepAnswer(query: query, unit: unit, route: route) {
                    text = focused
                    hasStepChain = true
                } else if !unit.steps.isEmpty {
                    let chain = renderStepChain(label: label, url: url, steps: unit.steps)
                    text = "\(intro)\n\n\(chain)."
                    hasStepChain = true
                } else {
                    text = renderGroundedSummaryAnswer(
                        unit: unit,
                        route: route
                    )
                }
                let needsConfirmClause = (requiresConfirmation ?? false) &&
                    (route == .toolAction || route == .answerPlusAction)
                if needsConfirmClause {
                    text = "\(text)\n\n\(confirmationClause())"
                }
            } else {
                // Composer can't ground itself without evidence. Safe
                // fallback — the verizon.com URL is the canonical
                // citation for this state.
                text = "It looks like I don't have specific information about that. You can check [Verizon Home Internet](\(AnswerComposerConstants.verizonInternetURL)) for more details."
                usedFallback = true
                citedPageID = nil
                expectedLinkURL = AnswerComposerConstants.verizonInternetURL
            }
        }

        let (urls, labels) = extractRenderedLinks(text)
        let latency = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        return ComposedAnswer(
            text: text,
            route: route,
            citedPageID: citedPageID,
            renderedLinks: urls,
            renderedLinkLabels: labels,
            expectedLinkURL: expectedLinkURL,
            requiresConfirmation: requiresConfirmation,
            actionFired: false,
            latencyMs: latency,
            strategy: name,
            hasStepChain: hasStepChain,
            usedFallback: usedFallback
        )
    }
}

// MARK: - Rendering helpers (mirror Python module-private helpers)

/// Render a markdown link in the canonical Verizon Home assistant style.
public func renderLink(label: String, url: String) -> String {
    let safeLabel = label
        .replacingOccurrences(of: "[", with: "(")
        .replacingOccurrences(of: "]", with: ")")
        .trimmingCharacters(in: .whitespacesAndNewlines)
    return "[\(safeLabel)](\(url))"
}

/// Render `[label](url) > step1 > step2 > ...`. Mirrors Python.
public func renderStepChain(label: String, url: String, steps: [String]) -> String {
    var parts: [String] = [renderLink(label: label, url: url)]
    let whitespace = CharacterSet.whitespacesAndNewlines
    let arrowReplace = "(?:^|[^>])>"  // unused; replicate Python's safer logic below
    _ = arrowReplace
    for step in steps {
        var clean = step.trimmingCharacters(in: whitespace)
        if clean.hasSuffix(".") {
            clean = String(clean.dropLast())
        }
        clean = clean.replacingOccurrences(of: "\n", with: " ")
        // Collapse runs of whitespace.
        clean = clean.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
        // Never let a step introduce a "`>`"-shaped separator inside
        // its own text — match Python's heuristic.
        clean = clean.replacingOccurrences(of: " > ", with: " - ")
        clean = clean.replacingOccurrences(of: ">", with: " ")
        if clean.isEmpty { continue }
        parts.append(clean)
    }
    return parts.joined(separator: " > ")
}

/// Extract markdown link URLs + labels from the rendered text. Returns
/// `(urls, labels)` parallel arrays. Mirrors Python.
public func extractRenderedLinks(_ text: String) -> ([String], [String]) {
    var urls: [String] = []
    var labels: [String] = []
    guard let regex = try? NSRegularExpression(pattern: #"\[([^\]\n]+)\]\(([^)\s]+)\)"#, options: []) else {
        return ([], [])
    }
    let ns = text as NSString
    let range = NSRange(location: 0, length: ns.length)
    regex.enumerateMatches(in: text, options: [], range: range) { match, _, _ in
        guard let m = match, m.numberOfRanges >= 3 else { return }
        labels.append(ns.substring(with: m.range(at: 1)).trimmingCharacters(in: .whitespaces))
        urls.append(ns.substring(with: m.range(at: 2)).trimmingCharacters(in: .whitespaces))
    }
    return (urls, labels)
}

/// Mirrors `_task_phrase_from_evidence`.
private func taskPhrase(from unit: RAGUnit) -> String {
    if let taskID = unit.taskID, !taskID.isEmpty {
        return taskID.replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: "_", with: " ")
    }
    if let label = unit.citationLabel, !label.isEmpty {
        return label.lowercased()
    }
    let title = unit.title.lowercased()
    if title.hasSuffix(".") { return String(title.dropLast()) }
    return title.isEmpty ? "do this" : title
}

/// Mirrors `_intro_for_route`.
private func introForRoute(_ route: ComposerRoute, task: String) -> String {
    let customerTask = customerFacingTaskPhrase(task)
    switch route {
    case .toolAction:
        return "I can \(customerTask) for you."
    case .answerPlusAction:
        return "To \(customerTask), follow these steps. I can also do it after you confirm:"
    default:
        return "To \(customerTask):"
    }
}

private func customerFacingTaskPhrase(_ task: String) -> String {
    switch task {
    case "restart router":
        return "restart your router"
    case "run speed test", "speed test":
        return "run a speed test"
    case "toggle parental controls":
        return "update parental controls"
    case "reboot extender":
        return "reboot your extender"
    default:
        return task
    }
}

/// Render a useful grounded answer for pages that have source text but
/// no canonical `steps` array. This keeps deterministic composition
/// from leaking corpus labels like "I found the relevant page for X"
/// while preserving the hard guarantee that every visible claim is
/// extracted from the selected `RAGUnit`.
private func renderGroundedSummaryAnswer(
    unit: RAGUnit,
    route: ComposerRoute
) -> String {
    let label = unit.displayLabel
    let link = renderLink(label: label, url: unit.canonicalURL)
    let lead = leadSentence(from: unit)
        ?? "This section covers \(humanizedLabel(label))."
    let facts = sourceOrderedFactSentences(from: unit, excluding: lead, limit: 4)

    var sections: [String] = [lead]
    if !facts.isEmpty {
        sections.append("Key details:\n" + facts.map { "- \($0)" }.joined(separator: "\n"))
    }

    let cta: String
    switch route {
    case .toolAction:
        cta = "Open \(label) to continue: \(link)."
    case .answerPlusAction:
        cta = "You can continue in \(label): \(link)."
    default:
        cta = "Open \(label) for the exact screen: \(link)."
    }
    sections.append(cta)
    return sections.joined(separator: "\n\n")
}

private func leadSentence(from unit: RAGUnit) -> String? {
    let sentences = sourceSentences(from: unit.body)
    let titleTokens = Set(BM25Tokenizer.tokenize(unit.title))
    let labelTokens = Set(BM25Tokenizer.tokenize(unit.displayLabel))
    let objectiveTokens = titleTokens.union(labelTokens)

    for sentence in sentences.prefix(5) {
        let sentenceTokens = Set(BM25Tokenizer.tokenize(sentence))
        if !sentenceTokens.intersection(objectiveTokens).isEmpty {
            return sentence
        }
    }
    return sentences.first
}

private func sourceOrderedFactSentences(
    from unit: RAGUnit,
    excluding lead: String,
    limit: Int
) -> [String] {
    let sentences = sourceSentences(from: unit.body)
    guard let leadIndex = sentences.firstIndex(of: lead) else {
        return Array(sentences.filter { $0 != lead }.prefix(limit))
    }
    let afterLead = sentences.dropFirst(leadIndex + 1)
    if !afterLead.isEmpty {
        return Array(afterLead.prefix(limit))
    }
    return Array(sentences.filter { $0 != lead }.prefix(limit))
}

private func sourceSentences(from body: String) -> [String] {
    let decoded = decodeSourceText(body)
    let normalized = decoded
        .replacingOccurrences(of: "\r", with: "\n")
        .replacingOccurrences(of: "\n", with: ". ")
        .replacingOccurrences(of: "  ", with: " ")

    return normalized
        .components(separatedBy: ". ")
        .map { raw -> String in
            var clean = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            while let last = clean.last, last == "." || last == ":" {
                clean.removeLast()
                clean = clean.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            return clean
        }
        .filter { sentence in
            let tokens = BM25Tokenizer.tokenize(sentence)
            return tokens.count >= 3 && !sentence.lowercased().hasSuffix(" page")
        }
}

private func decodeSourceText(_ text: String) -> String {
    text
        .replacingOccurrences(of: "&quot;", with: "\"")
        .replacingOccurrences(of: "&amp;", with: "&")
        .replacingOccurrences(of: "&#39;", with: "'")
        .replacingOccurrences(of: "&apos;", with: "'")
}

private func humanizedLabel(_ label: String) -> String {
    label
        .replacingOccurrences(of: "-", with: " ")
        .replacingOccurrences(of: "_", with: " ")
        .lowercased()
}

/// If a user asks about one step inside the active task, render that
/// grounded step instead of replaying the whole procedure. This is a
/// local span-selection pass over `RAGUnit.steps`, not an additional
/// intent heuristic or free-form generation.
private func renderFocusedStepAnswer(
    query: String,
    unit: RAGUnit,
    route: ComposerRoute
) -> String? {
    guard !unit.steps.isEmpty else { return nil }
    guard !unit.queryTargetsTaskObjective(query) else { return nil }
    let qTokens = Set(BM25Tokenizer.tokenize(query))
    guard !qTokens.isEmpty else { return nil }

    var best: (index: Int, precision: Double, overlap: Int)?
    for (index, step) in unit.steps.enumerated() {
        let stepTokens = Set(BM25Tokenizer.tokenize(step))
        let overlap = qTokens.intersection(stepTokens).count
        guard overlap > 0 else { continue }
        let precision = Double(overlap) / Double(qTokens.count)
        if best == nil ||
            precision > best!.precision ||
            (precision == best!.precision && overlap > best!.overlap) {
            best = (index, precision, overlap)
        }
    }
    guard let match = best, match.precision >= 0.5 else { return nil }

    let label = unit.displayLabel
    let url = unit.canonicalURL
    let task = taskPhrase(from: unit)
    let intro = route == .toolAction
        ? "For \(task), this is the step to focus on:"
        : "For \(label), this is the step to focus on:"
    let step = unit.steps[match.index]
    let chain = renderStepChain(label: label, url: url, steps: [step])
    return "\(intro)\n\n\(chain)."
}

/// Mirrors `_confirmation_clause`.
private func confirmationClause() -> String {
    return "Want me to do this for you? Reply 'yes' to confirm."
}

/// Mirrors `_clarify_hint_from_query` — cheap heuristic that fills the
/// clarify template's `…are you asking about X?` slot.
private func clarifyHintFromQuery(_ query: String) -> String {
    var q = query
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased()
    while let last = q.last, last == "?" || last == "." {
        q = String(q.dropLast())
    }
    q = q.trimmingCharacters(in: .whitespacesAndNewlines)
    if q.isEmpty { return "your Verizon Home Internet service" }
    let leadingDeterminers = ["the ", "a ", "an ", "my ", "our "]
    for det in leadingDeterminers {
        if q.hasPrefix(det) {
            q = String(q.dropFirst(det.count))
            break
        }
    }
    if q.count > 80 { q = String(q.prefix(80)) }
    return q
}
