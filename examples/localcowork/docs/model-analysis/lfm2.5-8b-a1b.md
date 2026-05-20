# LFM2.5-8B-A1B Model Analysis (Stub)

**Last updated**: 2026-05-21
**Status**: New default model for LocalCowork. Benchmark run on the curated tool surface pending.
**Config**: `_models/config.yaml` entry `lfm25-8b-a1b`

This is a stub. Sections below capture what is currently known and mark unknowns with explicit `TODO:` lines. As benchmarks land, this file should grow toward the depth of `lfm2-24b-a2b-benchmark.md` and `gpt-oss-20b.md`.

---

## Architecture

| Property | Value |
|----------|-------|
| Total parameters | 8B |
| Active parameters per token | 1B (A1B) |
| Architecture | Mixture-of-Experts, LFM2.5 hybrid attention + conv |
| Training objective | GRPO (reasoning + tool use) |
| Predecessor | Publicly released LFM2-8B-A1B |
| Tool-call format | `bracket` with `<think>...</think>` reasoning prefix |
| Reasoning output | Current turn's `<think>` block flows into chat content as-is; prior turns' thinking is stripped by the chat template (`preserve_thinking` defaults false) |

LFM2.5-8B-A1B is the successor to the publicly released LFM2-8B-A1B. The 8B / 1B-active split is the same; LFM2.5 adds long-context training and GRPO reasoning to the recipe.

---

## Provenance

- **Source checkpoint**: `LiquidAI/fernando_grpo_8B_MoE_from06081_longctx_v4_rope5M_step90_762484_HF` (private)
- **Training run signal**: the source checkpoint name embeds `longctx_v4` and `rope5M`, indicating long-context training with RoPE base scaling. **Long-context wiring (context budget allocation, eviction strategy, `start-model.sh --ctx-size` bump) is deferred to a follow-up PR.** The conservative 32k `context_window` in `_models/config.yaml` reflects this deferral, not a model limit.
- **GRPO step**: step 90 (`step90_762484_HF` suffix). Behavioral consequence: the model emits `<think>...</think>` reasoning blocks before bracket-format tool calls. The bracket parser tolerates this prefix (see `bracket_tolerates_think_prefix_*` tests in `src-tauri/src/inference/tool_call_parser.rs`).

---

## Quantizations Published

| Quant | Approx. size | Purpose |
|-------|--------------|---------|
| F16 | ~16 GB | Conversion intermediate, not uploaded |
| Q8_0 | ~8.5 GB | Production default (cookbook target: 16 GB Mac) |
| Q4_K_M | ~5 GB | Lightweight option for tighter VRAM budgets |

- **Dev mirror (private, pre-release)**: [`Paulescu/LFM2.5-8B-A1B-GGUF`](https://huggingface.co/Paulescu/LFM2.5-8B-A1B-GGUF)
- **Public release target**: `LiquidAI/LFM2.5-8B-A1B-GGUF`. The cookbook branch `localcowork-lfm2.5-8b-a1b` is non-mergeable to `main` until this public repo exists; the cutover diff is the repo string in `_models/config.yaml` and `README.md`.
- **F16 is intentionally not uploaded.** It is a conversion artifact only; only Q4_K_M and Q8_0 are pushed.

`TODO`: record the `llama.cpp` commit hash used for the conversion run (filled in by issue 02 operator after the first successful run of `scripts/convert-8b-a1b.sh`).

---

## Recommended Inference Params

| Param | Value | Source |
|-------|-------|--------|
| `temperature` | `0.3` | Public LFM2-8B-A1B model card on Hugging Face |
| `max_tokens` | `8192` | Reasoning models routinely emit 200-800 thinking tokens before the tool call on simple prompts and significantly more on harder ones; the previous 4096 default risks mid-bracket truncation |
| `context_window` | `32768` | Stack convention; long-context capability deferred |
| `tool_temperature` | (unset) | Reasoning models emit reasoning + tool-call tokens within a single generation, so the per-turn dual-temperature pattern from ADR-008 doesn't apply. Successor ADR is deferred |

**Deliberately deferred** (vendor-recommended but require schema and inference-client plumbing):

- `min_p: 0.15`: vendor-recommended in the LFM2-8B-A1B model card.
- `repetition_penalty: 1.05`: vendor-recommended in the LFM2-8B-A1B model card.

Adding these requires extending the model config schema and threading the values through the Rust inference client; scoped out of the current PR to keep its surface area minimal. See the parent PRD's "Deferred to follow-up issues" list.

---

## Known Gaps

- `TODO`: single-step tool-call accuracy on the curated 20-tool surface.
- `TODO`: multi-step chain-completion rate vs LFM2-24B-A2B baseline.
- `TODO`: VRAM and latency measurements on M-series Macs (M1, M2, M3, M4) at Q8_0 and Q4_K_M.
- `TODO`: behavior comparison with vs without `<think>` blocks suppressed at decode time (does the model still emit reliable tool calls, and at what accuracy cost, if reasoning is turned off?).

These are explicit gaps, not omissions. The PRD author accepted the risk of shipping a new default model without a benchmark run on the curated tool surface, mitigated by the in-script smoke validation at conversion time (see `scripts/convert-8b-a1b.sh`).
