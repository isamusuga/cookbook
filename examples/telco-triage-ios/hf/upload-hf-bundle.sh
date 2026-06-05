#!/usr/bin/env bash
#
# Upload the prepared Telco Triage bundle to a Hugging Face model repo.
#
# Required:
#   HF_REPO_ID     Example: LiquidAI/TelcoTriage-POC
#
# Optional:
#   HF_BUNDLE_DIR  Defaults to ./.hf-bundle/telco-triage-ios
#   HF_PRIVATE     Defaults to 1. If repo does not exist, create it private.
#   HF_REVISION    Optional revision/branch.
#
# Authentication:
#   Run `hf auth login` locally, or set HF_TOKEN in your shell. Do not commit
#   tokens and do not paste them into chat logs.
#
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXAMPLE_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
BUNDLE_DIR="${HF_BUNDLE_DIR:-$EXAMPLE_DIR/.hf-bundle/telco-triage-ios}"
PRIVATE_FLAG="${HF_PRIVATE:-1}"

if [[ -z "${HF_REPO_ID:-}" ]]; then
  echo "error: HF_REPO_ID is required, for example:" >&2
  echo "  HF_REPO_ID=LiquidAI/TelcoTriage-POC $0" >&2
  exit 1
fi

if [[ ! -d "$BUNDLE_DIR" ]]; then
  echo "error: bundle directory not found: $BUNDLE_DIR" >&2
  echo "run ./hf/prepare-hf-bundle.sh first" >&2
  exit 1
fi

whoami="$(hf auth whoami 2>&1 || true)"
if [[ -z "${HF_TOKEN:-}" && "$whoami" == *"Not logged in"* ]]; then
  echo "error: Hugging Face auth not found." >&2
  echo "Run 'hf auth login' with a write/fine-grained token or set HF_TOKEN." >&2
  exit 1
fi

args=(
  upload
  "$HF_REPO_ID"
  "$BUNDLE_DIR"
  .
  --repo-type model
  --commit-message "Refresh Telco Triage current runtime pack"
  --delete "knowledge-base.json"
  --delete "chat-mode-router-v2.gguf"
  --delete "chat-mode-clf-v1.gguf"
  --delete "kb-extract-clf-v1.gguf"
  --delete "kb-extractor-v1.gguf"
  --delete "tool-selector-clf-v1.gguf"
  --delete "telco-shared-clf-v1.gguf"
  --delete "telco-relational-v1.gguf"
  --delete "vz-topic-gate-clf-v1.lora.gguf"
  --delete "vz-refusal-flags-clf-v1.lora.gguf"
  --delete "lfm2-colbert-350m-Q4_K_M.gguf"
  --delete "rag-index-v1.bin"
  --delete "rag-chunks-v1.json"
  --delete "colbert-projection-v1.bin"
)
if [[ "$PRIVATE_FLAG" == "1" || "$PRIVATE_FLAG" == "true" ]]; then
  args+=(--private)
fi
if [[ -n "${HF_REVISION:-}" ]]; then
  args+=(--revision "$HF_REVISION")
fi

hf "${args[@]}"
