import XCTest
@testable import TelcoTriage

/// Swift parity tests for the Step 5b dispatcher heuristics. Mirrors
/// `scripts/telco/tests/test_multi_turn_acceptance.py` so any drift between
/// the Python harness (where the 8 acceptance gates live) and the iOS
/// dispatcher gets caught at build time.
///
/// The tests stay narrow — one assertion per case, exhaustive over the
/// public surface. End-to-end multi-turn flows live in
/// `MultiTurnIntegrationTests.swift` (the Step α.7 harness) once the
/// dispatcher rewiring is verified at the unit level.
final class MultiTurnHeuristicsTests: XCTestCase {

    // MARK: - isAffirmative

    func test_isAffirmative_recognizes_bare_forms() {
        let positive = ["yes", "Yes", "YES", "yep", "yeah", "yup", "y", "OK", "okay", "sure", "do it", "go ahead", "confirm"]
        for text in positive {
            XCTAssertTrue(isAffirmative(text), "expected isAffirmative(\(text)) == true")
        }
    }

    func test_isAffirmative_rejects_directive_and_nonaffirmative() {
        let negative = [
            "yes please restart it",   // carries a new directive — not bare
            "nope",
            "maybe",
            "I'd like to",
            "yes but actually no",
            "",
        ]
        for text in negative {
            XCTAssertFalse(isAffirmative(text), "expected isAffirmative(\(text)) == false")
        }
    }

    // MARK: - isDidntWork

    func test_isDidntWork_recognizes_failure_family() {
        let positive = [
            "I tried that, still not working",
            "still not working",
            "didn't work",
            "did not work",
            "I already tried that",
            "still broken",
            "no luck",
        ]
        for text in positive {
            XCTAssertTrue(isDidntWork(text), "expected isDidntWork(\(text)) == true")
        }
    }

    func test_isDidntWork_rejects_neutral_phrases() {
        let negative = ["worked great", "thanks", "let's try", "ok"]
        for text in negative {
            XCTAssertFalse(isDidntWork(text), "expected isDidntWork(\(text)) == false")
        }
    }

    // MARK: - hasTopicSwitchPrefix

    func test_hasTopicSwitchPrefix_recognizes_pivots() {
        let positive = [
            "actually show connected devices",
            "instead, restart my router",
            "never mind, just show me devices",
            "wait, change the password",
            "no, show me data usage",
            "let's switch to parental controls",
        ]
        for text in positive {
            XCTAssertTrue(hasTopicSwitchPrefix(text), "expected hasTopicSwitchPrefix(\(text)) == true")
        }
    }

    func test_hasTopicSwitchPrefix_rejects_neutral_starts() {
        let negative = ["show me devices", "restart router", "yes"]
        for text in negative {
            XCTAssertFalse(hasTopicSwitchPrefix(text), "expected hasTopicSwitchPrefix(\(text)) == false")
        }
    }

    // MARK: - isShortFollowup

    func test_isShortFollowup_recognizes_short_forms() {
        let positive = [
            "How?",              // bare wh-word
            "Why?",
            "Where?",
            "how do I turn it off?",  // anaphoric pronoun + few content tokens
            "Can you tell me how to do it",
            "for my son's tablet",    // slot-accrual preposition
            "with my tablet",
        ]
        for text in positive {
            XCTAssertTrue(isShortFollowup(text), "expected isShortFollowup(\(text)) == true")
        }
    }

    func test_isShortFollowup_rejects_full_queries() {
        let negative = [
            "show me my connected devices",
            "How do I restart my router?",
            "restart the router please",
        ]
        for text in negative {
            XCTAssertFalse(isShortFollowup(text), "expected isShortFollowup(\(text)) == false")
        }
    }

    func test_toolSupportPageContext_maps_action_to_rag_page() {
        XCTAssertEqual(ChatViewModel.supportPageContext(for: .toggleParentalControls)?.pageID, "13.00")
        XCTAssertEqual(ChatViewModel.supportPageContext(for: .toggleParentalControls)?.linkID, "home")
        XCTAssertEqual(ChatViewModel.supportPageContext(for: .restartRouter)?.pageID, "02.07")
        XCTAssertEqual(ChatViewModel.supportPageContext(for: .runSpeedTest)?.pageID, "01.02")
    }

    // MARK: - isCrossSectionShift

    func test_isCrossSectionShift_different_sections() {
        XCTAssertTrue(isCrossSectionShift(priorPageID: "03.00", newPageID: "04.00"))
        XCTAssertTrue(isCrossSectionShift(priorPageID: "02.07", newPageID: "13.02"))
    }

    func test_isCrossSectionShift_same_section() {
        XCTAssertFalse(isCrossSectionShift(priorPageID: "13.00", newPageID: "13.02"))
        XCTAssertFalse(isCrossSectionShift(priorPageID: "03.00", newPageID: "03.03"))
    }

    func test_isCrossSectionShift_nil_inputs() {
        XCTAssertFalse(isCrossSectionShift(priorPageID: nil, newPageID: "03.00"))
        XCTAssertFalse(isCrossSectionShift(priorPageID: "03.00", newPageID: nil))
        XCTAssertFalse(isCrossSectionShift(priorPageID: nil, newPageID: nil))
    }
}

/// Swift parity tests for ``ToolAliasMap`` (Step 5b Pre-flight Fix A).
@MainActor
final class ToolAliasMapTests: XCTestCase {

    func test_default_map_resolves_speed_test_alias() {
        let map = ToolAliasMap.default()
        let alias = map.alias(forLinkID: "speed-test")
        XCTAssertNotNil(alias)
        XCTAssertEqual(alias?.toolID, "run-speed-test")
        XCTAssertFalse(alias?.imperativeOnly ?? true)
    }

    func test_default_map_resolves_restart_router_identity() {
        let map = ToolAliasMap.default()
        let alias = map.alias(forLinkID: "restart-router")
        XCTAssertNotNil(alias)
        XCTAssertEqual(alias?.toolID, "restart-router")
        XCTAssertFalse(alias?.imperativeOnly ?? true)
    }

    func test_default_map_does_not_alias_shared_parental_controls_home_link() {
        let map = ToolAliasMap.default()
        XCTAssertNil(
            map.alias(forLinkID: "home"),
            "home is a shared navigation link across 13.xx pages; aliasing it to toggle-parental-controls creates a false create-profile tool"
        )
    }

    func test_default_map_does_not_alias_share_wifi() {
        // User spec: share-wifi only if a real executable tool exists.
        // No ToolIntent is registered for share-wifi — so adding an
        // alias here would manufacture a confirmation handshake for a
        // tool that can't fire. The map must NOT carry it.
        let map = ToolAliasMap.default()
        XCTAssertNil(map.alias(forLinkID: "share-wifi"))
    }

    func test_unknown_link_id_returns_nil() {
        let map = ToolAliasMap.default()
        XCTAssertNil(map.alias(forLinkID: "no-such-tool"))
        XCTAssertNil(map.alias(forLinkID: nil))
    }

    func test_default_map_includes_exact_registered_tool_links_plus_speed_test_short_form() {
        let map = ToolAliasMap.default()
        let expectedLinkIDs: Set<String> = [
            "restart-router",
            "speed-test",            // non-identity alias
            "run-speed-test",
            "check-connection",
            "enable-wps",
            "run-diagnostics",
            "schedule-technician",
            "reboot-extender",
        ]
        for linkID in expectedLinkIDs {
            XCTAssertNotNil(map.alias(forLinkID: linkID), "alias missing for link_id=\(linkID)")
        }
        XCTAssertNil(map.alias(forLinkID: "home"))
    }
}
