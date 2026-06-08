import Foundation

/// 10-class macro intent over the Telco Home Internet domain.
///
/// Wire contract with the trained classifier head bundled in
/// `Resources/Models/telco-macro-intent_classifier_weights.bin`. Integer
/// raw values are part of the contract — do not renumber after a model
/// has been trained against them. See
/// `docs/architecture-decisions/ADR-021-telco-home-internet-rag-assistant.md` §2.1.
///
/// Parallels the `MacroIntent` IntEnum in `scripts/telco/schemas.py`. The
/// two must stay in lock-step.
public enum TelcoMacroIntent: Int, Sendable, CaseIterable, Codable {
    case network = 0
    case equipment = 1
    case devices = 2
    case homePage = 3
    case parental = 4
    case digitalSecureHome = 5
    case discover = 6
    case accountOOS = 7
    case billingOOS = 8
    case liveAgent = 9

    /// Stable wire string used in logs and metadata. Distinct from the
    /// Swift case name so renaming the case (e.g. for style) does not
    /// break logged data.
    public var wireName: String {
        switch self {
        case .network: return "network"
        case .equipment: return "equipment"
        case .devices: return "devices"
        case .homePage: return "home_page"
        case .parental: return "parental"
        case .digitalSecureHome: return "digital_secure_home"
        case .discover: return "discover"
        case .accountOOS: return "account_oos"
        case .billingOOS: return "billing_oos"
        case .liveAgent: return "live_agent"
        }
    }

    /// True when this intent maps to an in-scope RAG-answerable topic.
    /// `accountOOS`, `billingOOS`, and `liveAgent` are in-scope at the
    /// topic-gate level but do not have RAG answers — they route to
    /// navigation-only or live-agent lanes.
    public var isRAGAnswerable: Bool {
        switch self {
        case .network, .equipment, .devices, .homePage,
             .parental, .digitalSecureHome, .discover:
            return true
        case .accountOOS, .billingOOS, .liveAgent:
            return false
        }
    }
}
