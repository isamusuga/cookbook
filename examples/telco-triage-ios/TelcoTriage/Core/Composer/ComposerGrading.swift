import Foundation

/// Swift mirror of the grading helpers in `scripts/telco/answer_composer.py`.
///
/// These are used by:
///
/// * `AnswerComposerLinkValidityTests` / `…CitationCorrectnessTests` /
///   `…NoAutoExecuteTests` / `…RefusalTemplateTests`.
/// * A DEBUG-only self-audit assertion inside the dispatcher — if the
///   composer ever emits a rendered link that isn't in
///   `RAGUnitCorpus.allCanonicalURLs ∪ approvedExternalURLs`, we
///   `assertionFailure` so the bug is caught at development time. Ships
///   disabled in RELEASE.
public enum ComposerGrading {
    public static let approvedExternalURLs: Set<String> = [
        AnswerComposerConstants.myTelcoURL,
        AnswerComposerConstants.telcoInternetURL,
        AnswerComposerConstants.liveAgentPhone,
    ]

    /// Coarse format gate. Mirrors `is_format_compliant`.
    public static func isFormatCompliant(_ answer: ComposedAnswer) -> Bool {
        let text = answer.text
        switch answer.route {
        case .ragAnswer, .answerPlusAction, .toolAction:
            if answer.usedFallback {
                return text.contains("telco.com/home/internet")
            }
            return !answer.renderedLinks.isEmpty
        case .accountNav, .liveAgent:
            return !answer.renderedLinks.isEmpty
        case .greeting, .outOfScope, .clarify:
            // No telcohome:// link allowed on these routes.
            return !answer.renderedLinks.contains { $0.hasPrefix("telcohome://") }
        case .noRagAnswer:
            return !answer.renderedLinks.isEmpty
        }
    }

    /// Every rendered link must resolve to a known canonical URL or
    /// one of the approved external URLs. Mirrors `is_link_valid`.
    public static func isLinkValid(
        _ answer: ComposedAnswer,
        knownTelcoHomeURLs: Set<String>,
        knownExternalURLs: Set<String> = approvedExternalURLs
    ) -> Bool {
        for url in answer.renderedLinks {
            if knownExternalURLs.contains(url) { continue }
            let bare = url.split(separator: "?").first.map(String.init) ?? url
            if knownTelcoHomeURLs.contains(url) || knownTelcoHomeURLs.contains(bare) {
                continue
            }
            return false
        }
        return true
    }

    /// First rendered URL matches the expected URL. Mirrors
    /// `is_citation_correct`. Query-suffix equivalence accepted.
    public static func isCitationCorrect(_ answer: ComposedAnswer) -> Bool {
        guard let expected = answer.expectedLinkURL else {
            return answer.renderedLinks.isEmpty
        }
        guard let rendered = answer.renderedLinks.first else { return false }
        if rendered == expected { return true }
        let renderedBare = rendered.split(separator: "?").first.map(String.init) ?? rendered
        let expectedBare = expected.split(separator: "?").first.map(String.init) ?? expected
        return renderedBare == expectedBare
    }

    /// No telcohome:// rendered link outside the selected evidence's
    /// canonicalURL. Mirrors `is_grounded`.
    public static func isGrounded(_ answer: ComposedAnswer, evidence: RAGUnit?) -> Bool {
        if let evidence {
            let canonical = evidence.canonicalURL
            let canonicalBare = evidence.canonicalURLBare
            for url in answer.renderedLinks {
                guard url.hasPrefix("telcohome://") else { continue }
                let bare = url.split(separator: "?").first.map(String.init) ?? url
                if url != canonical && bare != canonicalBare {
                    return false
                }
            }
            return true
        }
        // No evidence — composer fallback. Allowed iff no telcohome://
        // link rendered.
        return !answer.renderedLinks.contains { $0.hasPrefix("telcohome://") }
    }

    /// tool_action must carry the "confirm" handshake. Composer must
    /// never set `actionFired = true`. Mirrors `is_action_safe`.
    public static func isActionSafe(_ answer: ComposedAnswer) -> Bool {
        if answer.actionFired { return false }
        guard answer.route == .toolAction else { return true }
        let lowered = answer.text.lowercased()
        return lowered.contains("confirm") || lowered.contains("yes to confirm")
    }

    /// Refusal / escalation / clarify / greeting / nav templates pin
    /// to the canonical wording. Mirrors `is_refusal_template_correct`.
    public static func isRefusalTemplateCorrect(_ answer: ComposedAnswer) -> Bool {
        let text = answer.text
        let lowered = text.lowercased()
        switch answer.route {
        case .greeting:
            return text.contains("Hello! How can I assist you with Telco Home Internet today?")
        case .outOfScope:
            return text.contains("I'm here to help with topics related to Telco Home Internet. Please try asking a different question.")
        case .liveAgent:
            return lowered.contains("support agent") && text.contains(AnswerComposerConstants.liveAgentPhone)
        case .accountNav:
            return lowered.contains("my telco") && text.contains(AnswerComposerConstants.myTelcoURL)
        case .noRagAnswer:
            return (lowered.contains("don't have specific information") || lowered.contains("don't have information"))
                && text.contains(AnswerComposerConstants.telcoInternetURL)
        case .clarify:
            return (lowered.contains("clarify") || lowered.contains("could you"))
                && !answer.renderedLinks.contains { $0.hasPrefix("telcohome://") }
        default:
            return true
        }
    }
}
