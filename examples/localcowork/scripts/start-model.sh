#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# LocalCowork: Start Model Server
#
# Starts llama-server for a model defined in _models/config.yaml.
# Usage:
#   ./scripts/start-model.sh                          # Start the active_model from the config
#   ./scripts/start-model.sh --model lfm25-8b-a1b     # Start a specific model
#   ./scripts/start-model.sh --model lfm2-24b-a2b     # Start the predecessor
#   ./scripts/start-model.sh --vision                 # Also start vision model on port 8081
#   ./scripts/start-model.sh --check                  # Just check if model files are downloaded
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

MODELS_DIR="${LOCALCOWORK_MODELS_DIR:-$HOME/Projects/_models}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/_models/config.yaml"

# Vision model (LFM2.5-VL-1.6B): always on port 8081, independent of --model
VISION_MODEL="LFM2.5-VL-1.6B-Q8_0.gguf"
VISION_MMPROJ="mmproj-LFM2.5-VL-1.6b-Q8_0.gguf"
VISION_PORT=8081

# Read the active_model key from _models/config.yaml.
# Small Python one-liner per issue scope: no YAML-parsing dependency added,
# just regex on the top-level key.
read_active_model() {
    python3 -c '
import re, sys
with open(sys.argv[1]) as f:
    for line in f:
        m = re.match(r"^active_model:\s*(\S+)", line)
        if m:
            print(m.group(1)); sys.exit(0)
sys.exit(1)
' "$CONFIG_FILE"
}

# Map a model key from _models/config.yaml to the bits start-model.sh needs:
# the on-disk filename, the llama-server port, the context size, the HF repo
# for download instructions, and the human-readable display name.
#
# Sets MAIN_MODEL, MAIN_PORT, MAIN_CTX, MAIN_DOWNLOAD_REPO, MAIN_DISPLAY.
# Exits 1 for unknown or ollama-runtime keys.
resolve_model() {
    case "$1" in
        lfm25-8b-a1b)
            MAIN_MODEL="LFM2.5-8B-A1B-Q4_K_M.gguf"
            MAIN_PORT=8080
            MAIN_CTX=32768
            MAIN_DOWNLOAD_REPO="LiquidAI/LFM2.5-8B-A1B-GGUF"
            MAIN_DISPLAY="LFM2.5-8B-A1B"
            ;;
        lfm2-24b-a2b)
            MAIN_MODEL="LFM2-24B-A2B-Q4_K_M.gguf"
            MAIN_PORT=8080
            MAIN_CTX=32768
            MAIN_DOWNLOAD_REPO="LiquidAI/LFM2-24B-A2B-GGUF"
            MAIN_DISPLAY="LFM2-24B-A2B"
            ;;
        *)
            echo "❌ Model key '$1' is not supported by start-model.sh." >&2
            echo "   Supported keys: lfm25-8b-a1b, lfm2-24b-a2b" >&2
            echo "   (Ollama-hosted models like gpt-oss-20b start via 'ollama serve'.)" >&2
            exit 1
            ;;
    esac
}

# ── Parse arguments ──────────────────────────────────────────────────────────

START_VISION=false
CHECK_ONLY=false
MODEL_KEY=""

while [ $# -gt 0 ]; do
    case "$1" in
        --vision) START_VISION=true; shift ;;
        --check)  CHECK_ONLY=true; shift ;;
        --model)
            if [ $# -lt 2 ]; then
                echo "--model requires a key argument (e.g. --model lfm25-8b-a1b)" >&2
                exit 1
            fi
            MODEL_KEY="$2"
            shift 2
            ;;
        --help|-h)
            cat <<HELP
Usage: $0 [--model <key>] [--vision] [--check]

  --model <key>   Start the model identified by <key> in _models/config.yaml.
                  Defaults to the active_model from the config.
                  Supported keys (handled by this script): lfm25-8b-a1b, lfm2-24b-a2b.
  --vision        Also start the vision model server on port $VISION_PORT.
  --check         Check if model files exist (don't start servers).

Environment:
  LOCALCOWORK_MODELS_DIR    Model directory (default: ~/Projects/_models)
HELP
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Run '$0 --help' for usage." >&2
            exit 1
            ;;
    esac
done

# ── Resolve model selection ──────────────────────────────────────────────────

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config not found: $CONFIG_FILE" >&2
    exit 1
fi

if [ -z "$MODEL_KEY" ]; then
    if ! MODEL_KEY=$(read_active_model); then
        echo "❌ Could not read active_model from $CONFIG_FILE" >&2
        exit 1
    fi
fi

resolve_model "$MODEL_KEY"

# ── Check llama-server ───────────────────────────────────────────────────────

if ! command -v llama-server &> /dev/null; then
    echo "❌ llama-server not found."
    echo ""
    echo "Install via Homebrew (macOS):"
    echo "  brew install llama.cpp"
    echo ""
    echo "Or build from source:"
    echo "  git clone https://github.com/ggml-org/llama.cpp"
    echo "  cd llama.cpp && cmake -B build && cmake --build build --config Release"
    echo "  # Binary at: build/bin/llama-server"
    exit 1
fi

echo "✅ llama-server found: $(command -v llama-server)"

# ── Check model files ────────────────────────────────────────────────────────

echo ""
echo "Models directory: $MODELS_DIR"
echo "Selected model:   $MODEL_KEY ($MAIN_DISPLAY)"
echo ""

MAIN_PATH="$MODELS_DIR/$MAIN_MODEL"
VISION_PATH="$MODELS_DIR/$VISION_MODEL"
MMPROJ_PATH="$MODELS_DIR/$VISION_MMPROJ"

if [ -f "$MAIN_PATH" ]; then
    MAIN_SIZE=$(du -h "$MAIN_PATH" | cut -f1)
    echo "✅ Main model:   $MAIN_MODEL ($MAIN_SIZE)"
else
    echo "❌ Main model not found: $MAIN_PATH"
    echo ""
    echo "   Download $MAIN_DISPLAY from HuggingFace:"
    echo "   https://huggingface.co/$MAIN_DOWNLOAD_REPO"
    echo ""
    echo "   pip install huggingface-hub"
    echo "   python3 -c \""
    echo "     from huggingface_hub import hf_hub_download"
    echo "     hf_hub_download('$MAIN_DOWNLOAD_REPO',"
    echo "                     '$MAIN_MODEL',"
    echo "                     local_dir='$MODELS_DIR')"
    echo "   \""
    if [ "$CHECK_ONLY" = true ]; then
        echo ""
    else
        exit 1
    fi
fi

if [ -f "$VISION_PATH" ] && [ -f "$MMPROJ_PATH" ]; then
    echo "✅ Vision model:  $VISION_MODEL + mmproj"
else
    echo "⚠️  Vision model not found (optional; OCR falls back to Tesseract)"
    if [ "$START_VISION" = true ]; then
        echo ""
        echo "   Download from: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF"
        echo ""
        echo "   pip install huggingface-hub"
        echo "   python3 -c \""
        echo "     from huggingface_hub import hf_hub_download"
        echo "     for f in ['$VISION_MODEL', '$VISION_MMPROJ']:"
        echo "         hf_hub_download('LiquidAI/LFM2.5-VL-1.6B-GGUF', f,"
        echo "                         local_dir='$MODELS_DIR')"
        echo "   \""
    fi
fi

if [ "$CHECK_ONLY" = true ]; then
    exit 0
fi

# ── Start main model server ─────────────────────────────────────────────────

if [ ! -f "$MAIN_PATH" ]; then
    echo "Cannot start server; main model file missing."
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Starting $MAIN_DISPLAY on port $MAIN_PORT"
echo "═══════════════════════════════════════════════════"
echo "  Model:   $MAIN_MODEL"
echo "  Context: $MAIN_CTX tokens"
echo "  API:     http://localhost:$MAIN_PORT/v1"
echo ""

# Start main model in background
llama-server \
    --model "$MAIN_PATH" \
    --port "$MAIN_PORT" \
    --ctx-size "$MAIN_CTX" \
    --n-gpu-layers 99 \
    --flash-attn on &

MAIN_PID=$!
echo "  PID: $MAIN_PID"

# Wait for health check
echo -n "  Waiting for server..."
for i in $(seq 1 60); do
    if curl -sf "http://localhost:$MAIN_PORT/health" > /dev/null 2>&1; then
        echo " ready!"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo " timeout (60s). Check logs above for errors."
        exit 1
    fi
    sleep 1
    echo -n "."
done

# ── Start vision model server (optional) ─────────────────────────────────────

if [ "$START_VISION" = true ] && [ -f "$VISION_PATH" ] && [ -f "$MMPROJ_PATH" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "  Starting LFM2.5-VL-1.6B on port $VISION_PORT"
    echo "═══════════════════════════════════════════════════"

    llama-server \
        --model "$VISION_PATH" \
        --mmproj "$MMPROJ_PATH" \
        --port "$VISION_PORT" \
        --ctx-size 32768 &

    VISION_PID=$!
    echo "  PID: $VISION_PID"

    echo -n "  Waiting for server..."
    for i in $(seq 1 60); do
        if curl -sf "http://localhost:$VISION_PORT/health" > /dev/null 2>&1; then
            echo " ready!"
            break
        fi
        if [ "$i" -eq 60 ]; then
            echo " timeout. Vision OCR will fall back to Tesseract."
        fi
        sleep 1
        echo -n "."
    done
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Model servers running"
echo "═══════════════════════════════════════════════════"
echo "  Main:   http://localhost:$MAIN_PORT/v1  (PID $MAIN_PID)"
if [ "$START_VISION" = true ] && [ -n "${VISION_PID:-}" ]; then
    echo "  Vision: http://localhost:$VISION_PORT/v1  (PID $VISION_PID)"
fi
echo ""
echo "  In another terminal:  cargo tauri dev"
echo "  To stop:              kill $MAIN_PID${VISION_PID:+ $VISION_PID}"
echo ""

# Wait for all background processes
wait
