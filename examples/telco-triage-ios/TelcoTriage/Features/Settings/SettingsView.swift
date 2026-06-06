import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.brand) private var brand

    @State private var showResetConfirm = false

    var body: some View {
        NavigationStack {
            Form {
                modeSection
                kbSection
                modelsSection
                sessionSection
                aboutSection
            }
            .navigationTitle("Settings")
            .alert("Reset session?", isPresented: $showResetConfirm) {
                Button("Reset", role: .destructive, action: resetSession)
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Clears the token ledger, latency stats, and PII counts. Conversation history is not affected.")
            }
        }
    }

    private var modeSection: some View {
        Section(header: Text("Experience mode")) {
            Picker("Mode", selection: $appState.appMode) {
                Text("Customer").tag(AppMode.customer)
                Text("Engineering").tag(AppMode.engineering)
            }
            .pickerStyle(.segmented)
            Text(appState.appMode == .customer
                 ? "Clean chat experience. Traces and confidence scores are hidden."
                 : "Full instrumentation. Trace rows, tool cards, and latency counters visible.")
                .font(.caption).foregroundStyle(brand.textSecondary)
        }
    }

    private var kbSection: some View {
        Section(header: Text("Local AI runtime")) {
            LabeledContent("Status", value: appState.ragStatus.isLive ? "Live" : "Degraded")
            LabeledContent("Understanding", value: "LFM2.5-350M shared pass")
            LabeledContent("Policy", value: "9 heads + thresholds")
            LabeledContent("Corpus", value: "rag-units-v1.json")
            LabeledContent(
                "Units",
                value: appState.ragStatus.corpusUnitCount.map(String.init) ?? "—"
            )
            LabeledContent("Retriever", value: "BM25 hierarchy")
            LabeledContent("Answer layer", value: "Composer + V4 repair")
            if let reason = appState.ragStatus.degradedReason {
                LabeledContent("Reason", value: reason)
            }
            Text("Normal demo turns use one LFM2.5-350M forward pass for route, tool, cloud, handoff, safety, and transcript signals. BM25 retrieves canonical support units; deterministic policy owns links, citations, and confirmation. V4 verbalizes bounded multi-turn repair wording when bundled. Slot value extraction is TBD.")
                .font(.caption).foregroundStyle(brand.textSecondary)
        }
    }

    @ViewBuilder
    private var modelsSection: some View {
        if appState.appMode == .engineering {
            Section(header: Text("Active on-device artifacts")) {
                LabeledContent("Base", value: TelcoModelBundle.baseModelName)
                LabeledContent(
                    "Understanding adapter",
                    value: TelcoModelBundle.adr015TelcoStackBundled()
                        ? TelcoModelBundle.sharedClfAdapterName
                        : "not bundled"
                )
                LabeledContent("Classifier heads", value: "9 packaged heads")
                LabeledContent("Tool policy", value: "required_tool head + registry")
                LabeledContent(
                    "Repair verbalizer",
                    value: TelcoModelBundle.dialogueRepairV4AdapterPath() == nil
                        ? "not bundled"
                        : TelcoModelBundle.dialogueRepairV4AdapterName
                )
                LabeledContent("Slot values", value: "TBD")
                Text("Customer Q&A does not run the legacy chat router, ColBERT, or Stage B generator. Supported actions are gated by the shared `required_tool` head, canonical RAG unit affordance, ToolRegistry, and confirmation policy. V4 can only rewrite response text; it cannot change route, source, handoff, or tool execution.")
                    .font(.caption).foregroundStyle(brand.textSecondary)
            }

            Section(header: Text("Legacy / evaluation artifacts")) {
                LabeledContent(
                    "Chat router",
                    value: inactiveArtifactStatus(
                        name: TelcoModelBundle.chatModeRouterAdapterName,
                        path: TelcoModelBundle.chatModeRouterAdapterPath()
                    )
                )
                LabeledContent(
                    "Tool selector LoRA",
                    value: inactiveArtifactStatus(
                        name: TelcoModelBundle.toolAdapterName,
                        path: TelcoModelBundle.toolAdapterPath()
                    )
                )
                NavigationLink {
                    VerizonRAGTestView()
                } label: {
                    LabeledContent(
                        "Stage B generator",
                        value: TelcoModelBundle.verizonStageBGeneratorPath() == nil
                            ? "not bundled"
                            : TelcoModelBundle.verizonStageBGeneratorName
                    )
                }
                .disabled(TelcoModelBundle.verizonStageBGeneratorPath() == nil)
                Text("These artifacts are kept only for degraded-build compatibility or offline evaluation. They are not invoked by the normal Telco Triage customer/demo answer path.")
                    .font(.caption).foregroundStyle(brand.textSecondary)
            }
        }
    }

    @ViewBuilder
    private var sessionSection: some View {
        if appState.appMode == .engineering {
            Section(header: Text("Session")) {
                LabeledContent("Tokens kept on-device", value: "\(appState.tokenLedger.totalTokensSaved)")
                LabeledContent("On-device answers", value: "\(appState.tokenLedger.messagesOnDevice)")
                LabeledContent("Tool deflections", value: "\(appState.tokenLedger.messagesDeflected)")
                Button("Reset metrics", role: .destructive) { showResetConfirm = true }
            }
        }
    }

    private var aboutSection: some View {
        Section(header: Text("About")) {
            LabeledContent("Build", value: "\(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.0.0") (\(Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "0"))")
            LabeledContent("App", value: "\(brand.appName) \(brand.appSubtitle)")
            Text("Liquid Telco Triage runs LFM2.5-350M on-device for routing, safe action decisions, and private support flows. Grounded Q&A uses BM25 composer RAG over canonical support units with explicit confirmation before supported actions.")
                .font(.caption).foregroundStyle(brand.textSecondary)
        }
    }

    private func resetSession() {
        appState.tokenLedger.reset()
        appState.sessionStats.reset()
    }

    private func inactiveArtifactStatus(name: String, path: String?) -> String {
        path == nil ? "not bundled" : "\(name) · inactive"
    }
}
