import XCTest
@testable import TelcoTriage

@MainActor
final class TelcoSupportSessionTests: XCTestCase {
    private var corpus: RAGUnitCorpus!
    private var retriever: BM25HierarchyRetriever!
    private var toolRegistry: ToolRegistry!

    override func setUpWithError() throws {
        try super.setUpWithError()
        corpus = try RAGUnitCorpus.loadFromBundle()
        retriever = BM25HierarchyRetriever(corpus: corpus)
        toolRegistry = ToolRegistry.demoDefault(customerContext: CustomerContext())
    }

    func test_headlessSessionReturnsStructuredRAGAnswerWithoutChatViewModel() async throws {
        let session = makeSession()

        let result = try await session.handle("how do I restart my router?")

        XCTAssertEqual(result.route, .answerPlusAction)
        XCTAssertEqual(result.executableToolIntent, .restartRouter)
        XCTAssertEqual(result.citation?.pageID, "02.07")
        XCTAssertEqual(result.blackboard.priorPageID, "02.07")
        XCTAssertEqual(result.blackboard.pendingToolConfirmation?.toolID, "restart-router")
        XCTAssertFalse(result.text.isEmpty)
    }

    func test_headlessSessionDoesNotTrapIndependentTopicSwitchOnPriorTask() async throws {
        let session = makeSession()

        _ = try await session.handle("how do I restart my router?")
        let result = try await session.handle("what is my network SSID?")

        XCTAssertNotEqual(result.citation?.pageID, "02.07")
        XCTAssertNotEqual(result.blackboard.priorPageID, "02.07")
        XCTAssertTrue(
            result.text.localizedCaseInsensitiveContains("network")
                || result.text.localizedCaseInsensitiveContains("SSID")
                || result.citation?.label.localizedCaseInsensitiveContains("Wi-Fi") == true
        )
    }

    private func makeSession() -> TelcoSupportSession {
        let dispatcher = TelcoChatDispatcher(
            stageA: nil,
            stageB: nil,
            kbFallback: KeywordKBExtractor(),
            kb: [],
            retriever: nil,
            modelHost: nil,
            composer: DeterministicAnswerComposer(),
            corpus: corpus,
            lexicalRetriever: retriever,
            toolRegistry: toolRegistry,
            toolAliasMap: ToolAliasMap.default(),
            dialogueRepairVerbalizer: nil
        )
        return TelcoSupportSession(dispatcher: dispatcher)
    }
}
