#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# LocalCowork: Convert LFM2.5-8B-A1B safetensors checkpoint to GGUF
#
# Purpose (one-shot, run-on-demand, NOT invoked by CI):
#   1. Convert the safetensors checkpoint at <hf-model-dir> to F16 GGUF.
#   2. Quantize F16 → Q8_0 and F16 → Q4_K_M.
#   3. Spawn a temporary llama-server against Q8_0 and validate that a
#      tool-calling prompt produces a <think>...</think> block AND a
#      parseable <|tool_call_start|>[...]<|tool_call_end|> call.
#   4. Optionally upload Q4_K_M and Q8_0 (NOT F16) to the dev HF repo.
#
# Usage:
#   ./scripts/convert-8b-a1b.sh <hf-model-dir>            # convert + validate
#   ./scripts/convert-8b-a1b.sh <hf-model-dir> --upload   # also push to HF
#
# Required environment:
#   HF_TOKEN        HuggingFace write token. Picked up automatically by
#                   huggingface-hub / huggingface-cli; no custom auth in
#                   this script. Only required when --upload is set.
#   LLAMA_CPP_DIR   Path to local llama.cpp checkout containing
#                   convert_hf_to_gguf.py, llama-quantize, and llama-server.
#                   Defaults to $HOME/llama.cpp.
#
# llama.cpp version requirement:
#   LFM2.5-8B-A1B requires mainline llama.cpp support for the LFM2.5 MoE
#   (A1B) architecture. The first dry-run of this script verifies arch
#   support: convert_hf_to_gguf.py will exit non-zero on unsupported arch.
#   The actual commit hash used is printed in the run banner and is meant
#   to be copy-pasted into docs/model-analysis/lfm2.5-8b-a1b.md after a
#   successful run.
#
# Source / target HF repos:
#   Source (private):  LiquidAI/fernando_grpo_8B_MoE_from06081_longctx_v4_rope5M_step90_762484_HF
#   Dev target:        Paulescu/LFM2.5-8B-A1B-GGUF        (this script)
#   Release target:    LiquidAI/LFM2.5-8B-A1B-GGUF        (edit DEV_HF_REPO below at release cutover)
#
# Design note: this is a deliberate copy-and-edit of scripts/convert-to-gguf.sh.
# It runs twice in its lifetime (once now to the dev repo, once at release to
# the public repo). Abstracting into a shared library is overhead with no payback.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Constants (hand-edit at release cutover) ────────────────────────────────

MODEL_BASENAME="LFM2.5-8B-A1B"
DEV_HF_REPO="Paulescu/${MODEL_BASENAME}-GGUF"
# At release cutover, change DEV_HF_REPO to "LiquidAI/${MODEL_BASENAME}-GGUF".

F16_NAME="${MODEL_BASENAME}-F16.gguf"
Q8_NAME="${MODEL_BASENAME}-Q8_0.gguf"
Q4_NAME="${MODEL_BASENAME}-Q4_K_M.gguf"

VALIDATION_PORT=8099
VALIDATION_TEMPERATURE=0.3
VALIDATION_CTX_SIZE=8192
VALIDATION_MAX_TOKENS=4096

# ── Parse arguments ──────────────────────────────────────────────────────────

UPLOAD=false
MODEL_DIR=""

print_usage() {
    sed -n '2,/^# ────/p' "$0" | sed 's/^# \{0,1\}//'
}

for arg in "$@"; do
    case "$arg" in
        --upload) UPLOAD=true ;;
        --help|-h) print_usage; exit 0 ;;
        -*) echo "Unknown flag: $arg" >&2; exit 1 ;;
        *)
            if [ -z "$MODEL_DIR" ]; then
                MODEL_DIR="$arg"
            else
                echo "Unexpected positional arg: $arg" >&2
                exit 1
            fi
            ;;
    esac
done

if [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 <hf-model-dir> [--upload]" >&2
    exit 1
fi
if [ ! -d "$MODEL_DIR" ]; then
    echo "Model dir not found: $MODEL_DIR" >&2
    exit 1
fi
if [ "$UPLOAD" = true ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "✗ --upload requires HF_TOKEN env var (picked up by huggingface-cli)" >&2
    exit 1
fi

LLAMA_CPP="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$MODEL_DIR")/gguf}"
mkdir -p "$OUTPUT_DIR"

F16_PATH="$OUTPUT_DIR/$F16_NAME"
Q8_PATH="$OUTPUT_DIR/$Q8_NAME"
Q4_PATH="$OUTPUT_DIR/$Q4_NAME"
SERVER_LOG="$OUTPUT_DIR/.validate-server.log"

LLAMA_CPP_COMMIT=$(git -C "$LLAMA_CPP" rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo "=== LFM2.5-8B-A1B GGUF Conversion ==="
echo "  Model dir:    $MODEL_DIR"
echo "  llama.cpp:    $LLAMA_CPP (commit $LLAMA_CPP_COMMIT)"
echo "  Output dir:   $OUTPUT_DIR"
if [ "$UPLOAD" = true ]; then
    echo "  Upload:       yes → $DEV_HF_REPO (Q4_K_M, Q8_0 only)"
else
    echo "  Upload:       no (pass --upload to push)"
fi
echo

# ── Step 1: HF safetensors → F16 GGUF ────────────────────────────────────────

echo "[1/5] Converting HF safetensors → GGUF (F16)..."
python3 "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$MODEL_DIR" \
    --outfile "$F16_PATH" \
    --outtype f16

echo "  → $F16_PATH ($(du -h "$F16_PATH" | cut -f1))"
echo

# ── Step 2: F16 → Q8_0 ───────────────────────────────────────────────────────

echo "[2/5] Quantizing → Q8_0..."
"$LLAMA_CPP/llama-quantize" "$F16_PATH" "$Q8_PATH" Q8_0
echo "  → $Q8_PATH ($(du -h "$Q8_PATH" | cut -f1))"
echo

# ── Step 3: F16 → Q4_K_M ─────────────────────────────────────────────────────

echo "[3/5] Quantizing → Q4_K_M..."
"$LLAMA_CPP/llama-quantize" "$F16_PATH" "$Q4_PATH" Q4_K_M
echo "  → $Q4_PATH ($(du -h "$Q4_PATH" | cut -f1))"
echo

# ── Step 4: Spawn llama-server and validate tool-calling behavior ───────────

echo "[4/5] Validating Q8_0 with llama-server (port $VALIDATION_PORT)..."

"$LLAMA_CPP/llama-server" \
    --model "$Q8_PATH" \
    --port "$VALIDATION_PORT" \
    --ctx-size "$VALIDATION_CTX_SIZE" \
    --n-gpu-layers 99 \
    --jinja \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup_server() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup_server EXIT

echo -n "  Waiting for server"
for i in $(seq 1 60); do
    if curl -sf "http://localhost:$VALIDATION_PORT/health" > /dev/null 2>&1; then
        echo " ready"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo " timeout after 60s" >&2
        echo "  Server log: $SERVER_LOG" >&2
        exit 1
    fi
    sleep 1
    echo -n "."
done

REQUEST_BODY=$(cat <<JSON
{
  "messages": [
    {"role": "system", "content": "You are a tool-using assistant. Think step by step inside <think>...</think>, then call exactly one tool."},
    {"role": "user", "content": "List the files in my Downloads folder."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "filesystem.list_dir",
        "description": "List entries in a directory.",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {"type": "string", "description": "Absolute path of the directory to list."}
          },
          "required": ["path"]
        }
      }
    }
  ],
  "temperature": $VALIDATION_TEMPERATURE,
  "max_tokens": $VALIDATION_MAX_TOKENS
}
JSON
)

RESPONSE=$(curl -sS "http://localhost:$VALIDATION_PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$REQUEST_BODY")

if [ -z "$RESPONSE" ]; then
    echo "✗ Empty response from llama-server" >&2
    echo "  Server log: $SERVER_LOG" >&2
    exit 1
fi

# Validate response: must contain a <think>...</think> block AND a parseable
# bracket-format tool call. llama.cpp may extract either into structured
# fields (reasoning_content, tool_calls) depending on version and --jinja
# behavior; accept either the literal markers OR the structured equivalents.
VALIDATION_EXIT=0
VALIDATION_OUTPUT=$(RESPONSE_JSON="$RESPONSE" python3 -c '
import json
import os
import sys

raw = os.environ.get("RESPONSE_JSON", "")
try:
    data = json.loads(raw)
    msg = data["choices"][0]["message"]
except (KeyError, IndexError, json.JSONDecodeError) as e:
    print(f"PARSE_ERROR: {e}", file=sys.stderr)
    print(raw[:4000], file=sys.stderr)
    sys.exit(2)

content = msg.get("content") or ""
reasoning = msg.get("reasoning_content") or ""
tool_calls = msg.get("tool_calls") or []

has_think_literal = "<think>" in content and "</think>" in content
has_think_extracted = bool(reasoning.strip())
has_think = has_think_literal or has_think_extracted

has_bracket_literal = "<|tool_call_start|>" in content and "<|tool_call_end|>" in content
has_tool_calls_extracted = len(tool_calls) > 0
has_tool_call = has_bracket_literal or has_tool_calls_extracted

print("--- content ---")
print(content[:2000])
print("--- reasoning_content ---")
print(reasoning[:2000])
print("--- tool_calls ---")
print(json.dumps(tool_calls, indent=2)[:2000])
print("--- checks ---")
print(f"has_think:     {has_think}  (literal={has_think_literal}, extracted={has_think_extracted})")
print(f"has_tool_call: {has_tool_call}  (literal={has_bracket_literal}, extracted={has_tool_calls_extracted})")

if not has_think:
    sys.exit(3)
if not has_tool_call:
    sys.exit(4)
sys.exit(0)
') || VALIDATION_EXIT=$?

echo "$VALIDATION_OUTPUT" | sed 's/^/  | /'

case "$VALIDATION_EXIT" in
    0) echo "  ✓ Validation passed (<think> block + bracket tool call present)" ;;
    2) echo "✗ Could not parse llama-server response as JSON" >&2; exit 1 ;;
    3) echo "✗ Validation failed: response lacks <think>...</think> block" >&2; exit 1 ;;
    4) echo "✗ Validation failed: response lacks bracket-format tool-call markers" >&2; exit 1 ;;
    *) echo "✗ Validation failed (exit $VALIDATION_EXIT)" >&2; exit 1 ;;
esac

cleanup_server
trap - EXIT
echo

# ── Step 5: Upload (only if --upload) ───────────────────────────────────────

if [ "$UPLOAD" = true ]; then
    echo "[5/5] Uploading Q4_K_M and Q8_0 to $DEV_HF_REPO..."
    echo "      (F16 intermediate is deliberately NOT uploaded.)"
    echo

    huggingface-cli upload "$DEV_HF_REPO" "$Q8_PATH" "$Q8_NAME"
    huggingface-cli upload "$DEV_HF_REPO" "$Q4_PATH" "$Q4_NAME"

    echo
    echo "  ✓ Uploaded:"
    echo "    - $Q8_NAME"
    echo "    - $Q4_NAME"
else
    echo "[5/5] Skipping upload (no --upload flag)"
fi

echo
echo "=== Done ==="
echo "  llama.cpp commit: $LLAMA_CPP_COMMIT"
echo "  Local files:"
ls -lh "$OUTPUT_DIR/${MODEL_BASENAME}"-*.gguf
if [ "$UPLOAD" = true ]; then
    echo
    echo "  HF repo:          https://huggingface.co/$DEV_HF_REPO"
    echo "  Record the llama.cpp commit ($LLAMA_CPP_COMMIT) in"
    echo "  docs/model-analysis/lfm2.5-8b-a1b.md per issue 05."
fi
