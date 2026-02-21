#!/bin/bash
# Correctness regression test for CB decoder changes.
# Transcribes a known audio file and compares output tokens/text to a saved baseline.
#
# Usage:
#   ./test_correctness.sh [--save-baseline]
#
# Prerequisites:
#   - Build CTranslate2 and install Python bindings
#   - Set LD_LIBRARY_PATH if needed
#   - Activate venv: source .venv/bin/activate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="${SCRIPT_DIR}/../ctranslate2-server"
BASELINE_FILE="${SCRIPT_DIR}/test_baseline.json"
MODEL_PATH="${MODEL_PATH:-/home/leonardo/Documents/projects/ctranslate2-server/models/whisper-large-v3}"
TEST_AUDIO="${TEST_AUDIO:-${SERVER_DIR}/test_audio.wav}"

# Check if test audio exists; if not, try to generate a short one with ffmpeg
if [ ! -f "$TEST_AUDIO" ]; then
    echo "WARNING: Test audio not found at $TEST_AUDIO"
    echo "Please provide a test audio file (WAV, 16kHz, mono)."
    echo "Set TEST_AUDIO=/path/to/audio.wav to override."
    exit 1
fi

# Activate venv if not already active
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
        source "${SCRIPT_DIR}/.venv/bin/activate"
    fi
fi

# Set LD_LIBRARY_PATH for CTranslate2
export LD_LIBRARY_PATH="${SCRIPT_DIR}/build:${LD_LIBRARY_PATH:-}"

PYTHON_SCRIPT=$(cat << 'PYEOF'
import sys
import json
import os
import ctranslate2
import numpy as np
from inspect import signature

model_path = sys.argv[1]
audio_path = sys.argv[2]
mode = sys.argv[3]  # "save" or "check"
baseline_path = sys.argv[4]

# Load feature extractor from model config (matches server.py exactly)
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.audio import pad_or_trim

config_path = os.path.join(model_path, "preprocessor_config.json")
fe_kwargs = {}
if os.path.isfile(config_path):
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    valid_keys = set(signature(FeatureExtractor.__init__).parameters.keys())
    fe_kwargs = {k: v for k, v in raw.items() if k in valid_keys}
extractor = FeatureExtractor(**fe_kwargs)

# Load audio
import soundfile as sf
audio, sr = sf.read(audio_path, dtype='float32')
if audio.ndim > 1:
    audio = audio.mean(axis=1)

# Compute mel features (same pipeline as server.py)
mel = extractor(audio)
mel = pad_or_trim(mel, length=extractor.nb_max_frames)
features = np.ascontiguousarray(mel[np.newaxis].astype(np.float32))

# Load model and transcribe
model = ctranslate2.models.Whisper(model_path, device="cuda", compute_type="float16")
sv = ctranslate2.StorageView.from_array(features)

# Detect language (returns list[list[tuple(token_str, prob)]])
lang_results = model.detect_language(sv)
lang_token = lang_results[0][0][0]  # e.g. "<|en|>"
print(f"Detected language: {lang_token}")

# Use generate (standard path) for deterministic baseline
# Prompt must include <|startoftranscript|> + language + task + notimestamps
# (mirrors server.py's tokenizer.sot_sequence + no_timestamps)
prompt = ["<|startoftranscript|>", lang_token, "<|transcribe|>", "<|notimestamps|>"]
print(f"Prompt: {prompt}")
result = model.generate(
    sv,
    [prompt],
    beam_size=5,
    max_length=448,
)

# generate() returns a list (one per prompt); take the first
res = result[0]
tokens = [int(t) for t in res.sequences_ids[0]]
score = float(res.scores[0]) if res.scores else 0.0

output = {
    "tokens": tokens,
    "score": round(score, 4),
    "num_tokens": len(tokens),
}

if mode == "save":
    with open(baseline_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Baseline saved: {len(tokens)} tokens, score={score:.4f}")
    print(f"First 10 tokens: {tokens[:10]}")
    print(f"Last 10 tokens: {tokens[-10:]}")
elif mode == "check":
    with open(baseline_path) as f:
        baseline = json.load(f)

    match = (tokens == baseline["tokens"])
    print(f"Token count: {len(tokens)} (baseline: {baseline['num_tokens']})")
    print(f"Score: {score:.4f} (baseline: {baseline['score']})")
    print(f"Exact match: {match}")

    if not match:
        # Find first difference
        for i, (a, b) in enumerate(zip(tokens, baseline["tokens"])):
            if a != b:
                print(f"First diff at position {i}: got {a}, expected {b}")
                break
        if len(tokens) != len(baseline["tokens"]):
            print(f"Length mismatch: {len(tokens)} vs {baseline['num_tokens']}")
        sys.exit(1)
    else:
        print("PASS: Output matches baseline exactly.")
PYEOF
)

if [ "${1:-}" = "--save-baseline" ]; then
    echo "=== Saving correctness baseline ==="
    python3 -c "$PYTHON_SCRIPT" "$MODEL_PATH" "$TEST_AUDIO" "save" "$BASELINE_FILE"
    echo "Baseline saved to $BASELINE_FILE"
else
    if [ ! -f "$BASELINE_FILE" ]; then
        echo "No baseline found. Run with --save-baseline first."
        exit 1
    fi
    echo "=== Checking correctness against baseline ==="
    python3 -c "$PYTHON_SCRIPT" "$MODEL_PATH" "$TEST_AUDIO" "check" "$BASELINE_FILE"
fi
