#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$DIR/models"

echo "=== Wake Word Trainer Setup (macOS ARM / Apple Silicon) ==="

# ── Python venv ───────────────────────────────────────────────────────────────
if [ ! -d "$DIR/.venv" ]; then
    echo "→ Creating Python virtual environment..."
    python3 -m venv "$DIR/.venv"
fi
source "$DIR/.venv/bin/activate"
pip install --upgrade pip -q

# ── PyTorch (MPS support built-in for Apple Silicon) ─────────────────────────
echo "→ Installing PyTorch (MPS-capable)..."
pip install torch torchaudio -q

# ── TensorFlow for microWakeWord (tensorflow-macos deprecated; use tensorflow) ─
echo "→ Installing TensorFlow..."
pip install tensorflow -q

# ── Training dependencies ─────────────────────────────────────────────────────
echo "→ Installing training dependencies..."
pip install -r "$DIR/requirements.txt" -q

# ── Clone & install openWakeWord (dev mode so we can patch it) ────────────────
if [ ! -d "$DIR/openWakeWord" ]; then
    echo "→ Cloning openWakeWord..."
    git clone https://github.com/dscripka/openWakeWord "$DIR/openWakeWord" -q
fi
pip install -e "$DIR/openWakeWord" -q
echo "  ✓ openWakeWord installed"

# ── Clone & install microWakeWord (for ESP32 on-device training) ──────────────
if [ ! -d "$DIR/microWakeWord" ]; then
    echo "→ Cloning microWakeWord..."
    git clone --depth 1 https://github.com/kahrendt/microWakeWord "$DIR/microWakeWord" -q
fi
# Install from source (pip wheel is incomplete); skip pendulum which fails on Python 3.13
pip install --no-deps -e "$DIR/microWakeWord" -q 2>/dev/null || true
# Install tbm_utils without deps (its pendulum dep fails to build on Python 3.13)
pip install --no-deps tbm_utils -q
pip install mmap_ninja pymicro-features webrtcvad-wheels audio_metadata -q
echo "  ✓ microWakeWord installed"

# ── TTS: macOS built-in voices + ffmpeg ──────────────────────────────────────
echo "→ Checking TTS dependencies..."
if ! command -v say &>/dev/null; then
    echo "  ERROR: 'say' command not found. This tool requires macOS."
    exit 1
fi
if ! command -v ffmpeg &>/dev/null; then
    echo "  ffmpeg not found, installing via Homebrew..."
    brew install ffmpeg
fi
echo "  ✓ macOS TTS (say) + ffmpeg ready"

# ── openWakeWord base models ──────────────────────────────────────────────────
if [ ! -f "$MODELS_DIR/embedding_model.onnx" ]; then
    echo "→ Downloading openWakeWord feature extractor models (~20 MB)..."
    curl -L --progress-bar \
        "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx" \
        -o "$MODELS_DIR/embedding_model.onnx"
    curl -L --progress-bar \
        "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx" \
        -o "$MODELS_DIR/melspectrogram.onnx"
    echo "  ✓ Base models ready"
else
    echo "  ✓ Base models already present"
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Usage:"
echo "  source .venv/bin/activate"
echo "  python train.py 'Hey Dobbi'            # quick test (500 samples, ~1h)"
echo "  python train.py 'Hey Jarvis'           # any wake word you like"
echo "  python train.py 'Hey Dobbi' --full     # production quality (2000 samples, ~13 GB downloads)"
echo ""
echo "Note: First run downloads background noise data (~2-13 GB depending on mode)."
