# Whisper Continuous Batching Server - Deployment Guide

## Overview

This server uses a **custom CTranslate2 fork** with continuous batching support for Whisper.
It runs a single model instance that handles transcription, language detection, and word-level timestamps.

## Architecture

```
server.py (FastAPI)
    └── WhisperContinuousBatcher (single model load)
            ├── worker thread    → continuous batching decode (transcription)
            └── dedicated replica → detect_language(), align() (word timestamps)
```

Key benefit: the model weights are loaded **once** (~3GB for large-v3). The replica shares the same weights with its own encoder/decoder layer objects, adding negligible memory overhead.

## Prerequisites

- Python 3.9+
- A CTranslate2 Whisper model (e.g. converted from `openai/whisper-large-v3`)
- **GCC 12** (gcc-12, g++-12) — required for C++17 + CUDA compatibility
- CMake 3.15+
- CUDA toolkit (tested with CUDA 12.x and 13.x)
- `pybind11[global]` for building the Python bindings

## Step 1: Build CTranslate2

```bash
# Clone the custom fork (--recurse-submodules pulls third-party deps)
git clone --recurse-submodules -b continuous-batching https://github.com/leochiodi/CTranslate2.git
cd CTranslate2

# Remove bundled thrust — it conflicts with CUDA toolkit's version
rm -rf third_party/thrust

# Set GCC 12 as the compiler
export CC=gcc-12
export CXX=g++-12

# Create build directory
mkdir build && cd build

# Configure (disable Intel MKL/OpenMP — not needed for GPU inference)
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CUDA=ON \
  -DWITH_MKL=OFF \
  -DOPENMP_RUNTIME=COMP \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

# Build
cmake --build . --config Release -j$(nproc)
```

### Build troubleshooting

| Issue | Fix |
|-------|-----|
| `libiomp5 not found` | Add `-DWITH_MKL=OFF -DOPENMP_RUNTIME=COMP` to cmake |
| `thrust` version conflicts | Delete `third_party/thrust` — CUDA toolkit provides it |
| C++17 errors or CUDA host compiler issues | Use GCC 12: `export CC=gcc-12 CXX=g++-12` |
| Third-party dirs empty | Run `git submodule update --init --recursive` |

## Step 2: Install the Python package

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# setuptools + pybind11 are needed to compile the Python extension
pip install setuptools pybind11[global]

# Install the CTranslate2 Python bindings
# CTRANSLATE2_ROOT tells setup.py where to find headers
# LIBRARY_PATH tells the linker where to find libctranslate2.so
cd /path/to/CTranslate2
CTRANSLATE2_ROOT=$(pwd) LIBRARY_PATH=$(pwd)/build pip install -e python/ --no-build-isolation

# Install server dependencies
pip install faster-whisper fastapi uvicorn python-multipart tokenizers torch torchaudio soundfile
```

## Step 3: Set the library path

The Python extension needs to find `libctranslate2.so` at runtime.

```bash
export LD_LIBRARY_PATH=/path/to/CTranslate2/build:$LD_LIBRARY_PATH
```

For production, either:
- Install the library system-wide: `cmake --install build/`
- Or add the `export` to your service file / Docker entrypoint

## Step 4: Get a Whisper model

**Option A** — download a pre-converted model (fastest):

```bash
pip install huggingface-hub[hf_xet]
huggingface-cli download Systran/faster-whisper-large-v3 --local-dir models/whisper-large-v3
```

**Option B** — convert from HuggingFace yourself:

```bash
pip install transformers[torch]
ct2-transformers-converter --model openai/whisper-large-v3 --output_dir models/whisper-large-v3 --quantization float16
```

## Step 5: Run the server

```bash
# Required
export WHISPER_MODEL=/path/to/models/whisper-large-v3

# Optional (shown with defaults)
export MAX_SLOTS=4              # concurrent decode slots
export DEVICE=cuda              # "cpu" or "cuda"
export DEVICE_INDEX=0           # GPU index
export COMPUTE_TYPE=float16     # "default", "float16", "int8", etc.
export BEAM_SIZE=5              # 1 = greedy
export MAX_FILE_DURATION=600    # max audio length in seconds
export REQUEST_TIMEOUT=120      # per-request timeout in seconds
export DEFAULT_LANGUAGE=fr      # fallback when language not specified
export HOTWORDS=""              # space-separated hotwords

# Start
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

## API Usage

### Transcribe audio

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "detect_language=true"
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file (wav, mp3, flac, etc.) |
| `language` | string | `null` | Force language code (e.g. `"en"`, `"fr"`) |
| `detect_language` | bool | `false` | Auto-detect language from first speech chunk |
| `vad_onset` | float | `0.9` | VAD sensitivity threshold (lower = more sensitive) |
| `enable_default_hotwords` | bool | `false` | Use the `HOTWORDS` env var for biasing |

If neither `language` nor `detect_language` is set, `DEFAULT_LANGUAGE` is used.

### Response format

```json
{
  "id": "584a7e0a85374e7a...",
  "text": "The full transcription text.",
  "duration": 35.0,
  "language": "en",
  "segments": [
    {
      "id": 1,
      "start": 0.0,
      "end": 35.0,
      "text": "The full transcription text.",
      "tokens": [440, 5193, ...],
      "avg_logprob": -0.23,
      "compression_ratio": 1.45,
      "no_speech_prob": 0.0,
      "words": [
        {"word": "The", "start": 0.0, "end": 0.24, "probability": 0.9352},
        {"word": "full", "start": 0.24, "end": 0.5, "probability": 0.9739},
        ...
      ]
    }
  ],
  "words": [
    {"word": "The", "start": 0.0, "end": 0.24, "probability": 0.9352},
    ...
  ]
}
```

### OpenAI-compatible endpoint

The server also listens on `/v1/audio/transcriptions` with the same parameters.

### Health check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

## Docker example

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv cmake git libsndfile1 \
    gcc-12 g++-12

ENV CC=gcc-12
ENV CXX=g++-12

# Build CTranslate2
COPY CTranslate2/ /opt/CTranslate2/
WORKDIR /opt/CTranslate2
RUN rm -rf third_party/thrust
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON \
             -DWITH_MKL=OFF -DOPENMP_RUNTIME=COMP && \
    cmake --build . -j$(nproc) && \
    cmake --install .
RUN ldconfig

# Install Python deps
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install setuptools pybind11[global]
RUN CTRANSLATE2_ROOT=/opt/CTranslate2 pip install -e /opt/CTranslate2/python/ --no-build-isolation
RUN pip install faster-whisper fastapi uvicorn python-multipart tokenizers torch torchaudio soundfile

# Copy server
COPY server.py /opt/server.py

WORKDIR /opt
ENV WHISPER_MODEL=/models/whisper-large-v3
ENV DEVICE=cuda
ENV MAX_SLOTS=4
ENV BEAM_SIZE=5

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Tuning

| Parameter | Effect |
|-----------|--------|
| `MAX_SLOTS=4` | More slots = higher throughput but more memory. Start with 4, increase if GPU memory allows. |
| `BEAM_SIZE=1` | Greedy decoding — fastest. `BEAM_SIZE=2-5` improves quality at cost of speed. |
| `COMPUTE_TYPE=float16` | Half precision on GPU — fastest. Use `int8` for lower memory. |
| `COMPUTE_TYPE=int8` | Quantized — lowest memory, slightly lower quality. |

## Files

| File | Description |
|------|-------------|
| `server.py` | FastAPI server — the only application file |
| `CTranslate2/` | Custom fork with continuous batching + single-model architecture |
