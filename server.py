"""
FastAPI server for Whisper transcription using CTranslate2 continuous batching.

Usage:
    WHISPER_MODEL=/path/to/ct2-model python -m uvicorn server:app --host 0.0.0.0 --port 8000

Environment variables:
    WHISPER_MODEL       Path to CTranslate2 whisper model directory (required)
    MAX_SLOTS           Max concurrent decode slots (default: 4)
    DEVICE              "cpu" or "cuda" (default: "cpu")
    DEVICE_INDEX        GPU index (default: 0)
    COMPUTE_TYPE        e.g. "default", "float16", "int8" (default: "default")
    BEAM_SIZE           Beam size, 1 = greedy (default: 5)
    MAX_FILE_DURATION   Max audio length in seconds (default: 600)
    REQUEST_TIMEOUT     Per-request timeout in seconds (default: 120)
    DEFAULT_LANGUAGE    Fallback language when not provided (default: "fr")
    HOTWORDS            Space-separated hotwords to bias recognition (default: "")
"""

import asyncio
import faulthandler
import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from inspect import signature
from typing import Optional

import numpy as np
import tokenizers
import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ctranslate2._ext import StorageView, WhisperContinuousBatcher
from faster_whisper.audio import pad_or_trim
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import get_compression_ratio
from faster_whisper.vad import VadOptions, get_speech_timestamps

try:
    from wrapper.prometheus_metrics import (
        metrics_router,
        STT_REQUESTS_INFLIGHT,
        STT_E2E_LATENCY,
        STT_AUDIO_SECONDS_PROCESSED,
        STT_RTFX_BATCH,
        STT_PREPROCESS_TIME,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

faulthandler.enable()

COMPRESSION_RATIO_THRESHOLD = 2.4

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("WHISPER_MODEL", "")
MAX_SLOTS = int(os.environ.get("MAX_SLOTS", "4"))
DEVICE = os.environ.get("DEVICE", "cpu")
DEVICE_INDEX = int(os.environ.get("DEVICE_INDEX", "0"))
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "default")
BEAM_SIZE = int(os.environ.get("BEAM_SIZE", "5"))
MAX_FILE_DURATION = float(os.environ.get("MAX_FILE_DURATION", "600"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "120"))
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "fr")
HOTWORDS = os.environ.get("HOTWORDS", "")

# ---------------------------------------------------------------------------
# Audio helpers (torchaudio)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _get_resampler(sr_in: int, sr_out: int, dtype=torch.float32):
    return torchaudio.transforms.Resample(sr_in, sr_out, dtype=dtype)


def decode_audio(raw: bytes, sr_out: int = 16000) -> np.ndarray:
    """Convert any audio bytes to mono float32 ndarray."""
    import soundfile as sf
    buf = io.BytesIO(raw)
    audio, sr = sf.read(buf, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sr_out:
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        audio_t = _get_resampler(sr, sr_out, dtype=audio_t.dtype)(audio_t)
        return audio_t.squeeze(0).numpy()
    return audio


def get_audio_duration(raw: bytes) -> float:
    """Return duration in seconds, 0.0 on failure."""
    try:
        import soundfile as sf
        info = sf.info(io.BytesIO(raw))
        return info.frames / info.samplerate
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Global state filled at startup
# ---------------------------------------------------------------------------

batcher: WhisperContinuousBatcher = None  # type: ignore[assignment]
feature_extractor: FeatureExtractor = None  # type: ignore[assignment]
hf_tokenizer: tokenizers.Tokenizer = None  # type: ignore[assignment]


def _load_feature_extractor(model_path: str) -> FeatureExtractor:
    config_path = os.path.join(model_path, "preprocessor_config.json")
    kwargs = {}
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)
        valid_keys = set(signature(FeatureExtractor.__init__).parameters.keys())
        kwargs = {k: v for k, v in raw.items() if k in valid_keys}
    return FeatureExtractor(**kwargs)


# ---------------------------------------------------------------------------
# OpenAI-compatible error helpers
# ---------------------------------------------------------------------------


def raise_oai_error(message: str, param=None, code: int = 400):
    raise HTTPException(
        status_code=code,
        detail={
            "error": {
                "message": message,
                "type": "invalid_request_error" if code < 500 else "server_error",
                "param": param,
                "code": code,
            }
        },
    )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global batcher, feature_extractor, hf_tokenizer

    if not MODEL_PATH:
        raise RuntimeError("Set WHISPER_MODEL env var to the CTranslate2 model directory")

    feature_extractor = _load_feature_extractor(MODEL_PATH)

    tok_path = os.path.join(MODEL_PATH, "tokenizer.json")
    if not os.path.isfile(tok_path):
        raise RuntimeError(f"tokenizer.json not found in {MODEL_PATH}")
    hf_tokenizer = tokenizers.Tokenizer.from_file(tok_path)

    batcher = WhisperContinuousBatcher(
        model_path=MODEL_PATH,
        max_slots=MAX_SLOTS,
        device=DEVICE,
        device_index=DEVICE_INDEX,
        compute_type=COMPUTE_TYPE,
        beam_size=BEAM_SIZE,
        patience=1.0,
        length_penalty=1.0,
        max_length=448,
        suppress_blank=True,
        suppress_tokens=[-1],
        max_initial_timestamp_index=50,
    )
    batcher.start()

    yield

    batcher.stop()


app = FastAPI(title="Whisper Continuous Batching", lifespan=lifespan)

if PROMETHEUS_AVAILABLE:
    app.include_router(metrics_router)


# ---------------------------------------------------------------------------
# Exception handlers (OpenAI-compatible)
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def validation_to_oai(request, exc):
    msgs = [
        f"{'.'.join(str(x) for x in err['loc'][1:])}: {err['msg']}"
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": " ".join(msgs),
                "type": "invalid_request_error",
                "param": None,
                "code": 400,
            }
        },
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check():
    if batcher is None:
        return Response(
            content="Not ready", status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Transcription endpoints
# ---------------------------------------------------------------------------


@app.post("/transcribe")
@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    vad_onset: Optional[float] = Form(0.9),
    detect_language: Optional[bool] = Form(False),
    enable_default_hotwords: Optional[bool] = Form(False),
):
    if PROMETHEUS_AVAILABLE:
        STT_REQUESTS_INFLIGHT.inc()
    t_start = time.time()

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise_oai_error("Empty audio file", param="file")

        loop = asyncio.get_running_loop()

        # Duration validation
        duration = await loop.run_in_executor(None, get_audio_duration, audio_bytes)
        if duration == 0.0:
            raise_oai_error(
                "File has 0 seconds duration or unsupported format.", param="file"
            )
        if duration > MAX_FILE_DURATION:
            raise_oai_error(
                f"File exceeds duration limit of {MAX_FILE_DURATION} seconds.",
                param="file",
            )

        # Resolve language
        resolved_language = language
        if resolved_language is None and not detect_language:
            resolved_language = DEFAULT_LANGUAGE

        # Resolve hotwords
        hotwords = HOTWORDS if (enable_default_hotwords and HOTWORDS) else None

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    _process_request,
                    audio_bytes,
                    resolved_language,
                    bool(detect_language),
                    vad_onset or 0.5,
                    hotwords,
                ),
                timeout=REQUEST_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise_oai_error("Transcription timed out.", code=504)

        e2e = time.time() - t_start
        logger.info("Request complete: e2e %.3fs | audio %.1fs | lang %s", e2e, duration, result["language"])
        if PROMETHEUS_AVAILABLE:
            STT_E2E_LATENCY.observe(e2e)
            STT_AUDIO_SECONDS_PROCESSED.inc(duration)
            if e2e > 0 and duration > 0:
                STT_RTFX_BATCH.observe(duration / e2e)

        return {
            "id": uuid.uuid4().hex,
            "text": result["text"],
            "duration": duration,
            "language": result["language"],
            "segments": result["segments"],
            "words": result["words"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Transcription error")
        raise_oai_error(str(e), code=500)
    finally:
        if PROMETHEUS_AVAILABLE:
            STT_REQUESTS_INFLIGHT.dec()


# ---------------------------------------------------------------------------
# Synchronous processing pipeline (runs in thread pool)
# ---------------------------------------------------------------------------


def _process_request(
    audio_bytes: bytes,
    language: Optional[str],
    detect_language: bool,
    vad_onset: float,
    hotwords: Optional[str],
) -> dict:
    """decode audio -> VAD -> (lang detect) -> mel per segment -> submit -> collect."""

    sampling_rate = feature_extractor.sampling_rate
    t_pre = time.time()

    # 1. Decode audio
    waveform = decode_audio(audio_bytes, sr_out=sampling_rate)
    t_audio_decode = time.time()

    # 2. VAD
    vad_opts = VadOptions(threshold=vad_onset)
    speech_chunks = get_speech_timestamps(
        waveform, vad_options=vad_opts, sampling_rate=sampling_rate
    )
    t_vad = time.time()
    if not speech_chunks:
        return {
            "text": "",
            "language": language or DEFAULT_LANGUAGE,
            "segments": [],
            "words": [],
        }

    # 3. Language detection from first speech chunk
    t_lang_start = time.time()
    if detect_language or language is None:
        first_chunk = waveform[speech_chunks[0]["start"] : speech_chunks[0]["end"]]
        mel = feature_extractor(first_chunk)
        mel = pad_or_trim(mel, length=feature_extractor.nb_max_frames)
        features = StorageView.from_array(mel[np.newaxis].astype(np.float32))
        # Returns list[list[tuple(lang_token, prob)]]  e.g. [[("<|en|>", 0.95), ...]]
        lang_results = batcher.detect_language(features)
        best_lang_token = lang_results[0][0][0]  # e.g. "<|en|>"
        language = best_lang_token.strip("<|>")  # -> "en"
        logger.info("Detected language: %s (prob %.2f)", language, lang_results[0][0][1])
    t_lang = time.time()

    # 4. Build tokenizer and prompt
    tokenizer = Tokenizer(
        hf_tokenizer,
        multilingual=batcher.is_multilingual,
        task="transcribe",
        language=language,
    )
    prompt = list(tokenizer.sot_sequence) + [tokenizer.no_timestamps]

    # 5. Prepend hotwords as previous-context tokens
    if hotwords:
        hw_token_ids = hf_tokenizer.encode(" " + hotwords).ids
        prompt = [tokenizer.sot_prev] + hw_token_ids + prompt

    # 6. Submit each VAD segment as a separate batcher request.
    #    This preserves per-segment timing for the response.
    req_ids = []
    segment_times = []  # (start_sec, end_sec) per segment
    segment_num_frames = []  # num_frames_raw per segment

    t_mel_start = time.time()
    for i, chunk_info in enumerate(speech_chunks):
        start_sample = chunk_info["start"]
        end_sample = chunk_info["end"]
        chunk = waveform[start_sample:end_sample]

        mel = feature_extractor(chunk)
        num_frames_raw = mel.shape[-1]
        mel = pad_or_trim(mel, length=feature_extractor.nb_max_frames)
        features = StorageView.from_array(mel[np.newaxis].astype(np.float32))

        segment_num_frames.append(num_frames_raw)
        req_ids.append(batcher.submit(features, prompt))
        segment_times.append(
            (start_sample / sampling_rate, end_sample / sampling_rate)
        )

    t_pre_done = time.time()
    if PROMETHEUS_AVAILABLE:
        STT_PREPROCESS_TIME.observe(t_pre_done - t_pre)

    logger.info(
        "Submitted %d VAD segments (%.1fs speech in %.1fs audio)",
        len(req_ids),
        sum(e - s for s, e in segment_times),
        len(waveform) / sampling_rate,
    )

    # 7. Collect results in order
    segments = []
    all_texts = []
    all_words = []
    seg_idx = 0
    tokens_per_second = sampling_rate // (feature_extractor.hop_length * 2)
    t_decode_start = time.time()
    t_align_total = 0.0
    total_tokens = 0

    for i, req_id in enumerate(req_ids):
        result = batcher.get_result(req_id)
        tokens = result.sequences_ids[0]
        total_tokens += len(tokens)
        text = tokenizer.decode(tokens)

        if not text.strip():
            continue

        cr = get_compression_ratio(text)
        if cr > COMPRESSION_RATIO_THRESHOLD:
            logger.warning(
                "Dropping segment %d: compression ratio %.1f (hallucination)", i, cr
            )
            continue

        seg_idx += 1
        start_sec, end_sec = segment_times[i]
        avg_logprob = result.scores[0] if result.scores else 0.0

        # Word-level timestamps via alignment (using cross-attention captured during decode)
        seg_words = []
        text_tokens_clean = [t for t in tokens if t < tokenizer.eot]
        if text_tokens_clean and result.attention_weights:
            num_frames = segment_num_frames[i]
            try:
                t_align_start = time.time()
                ar = batcher.align_from_attention(
                    result.attention_weights,
                    text_tokens_clean,
                    num_frames,
                    sot_sequence_length=1,
                    median_filter_width=7,
                )
                t_align_total += time.time() - t_align_start
                text_indices = np.array([p[0] for p in ar.alignments])
                time_indices = np.array([p[1] for p in ar.alignments])
                words, word_tokens = tokenizer.split_to_word_tokens(
                    text_tokens_clean + [tokenizer.eot]
                )
                word_boundaries = np.pad(
                    np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0)
                )
                jumps = np.pad(
                    np.diff(text_indices), (1, 0), constant_values=1
                ).astype(bool)
                jump_times = time_indices[jumps] / tokens_per_second
                start_times = jump_times[word_boundaries[:-1]]
                end_times = jump_times[word_boundaries[1:]]

                for wi, word in enumerate(words):
                    if not word.strip():
                        continue
                    w_start = round(start_sec + float(start_times[wi]), 3)
                    w_end = round(start_sec + float(end_times[wi]), 3)
                    # Per-token probabilities not available with align_from_attention
                    # (would require a separate decoder pass). Use avg_logprob instead.
                    prob = min(1.0, max(0.0, np.exp(avg_logprob)))
                    seg_words.append({
                        "word": word.strip(),
                        "start": w_start,
                        "end": w_end,
                        "probability": round(prob, 4),
                    })
            except Exception:
                logger.warning("Word alignment failed for segment %d", i, exc_info=True)

        segments.append(
            {
                "id": seg_idx,
                "start": round(start_sec, 3),
                "end": round(end_sec, 3),
                "text": text.strip(),
                "tokens": tokens,
                "avg_logprob": avg_logprob,
                "compression_ratio": cr,
                "no_speech_prob": 0.0,
                "words": seg_words,
            }
        )
        all_texts.append(text.strip())
        all_words.extend(seg_words)

    t_done = time.time()
    t_preprocess = t_pre_done - t_pre
    t_decode = t_done - t_decode_start
    t_total = t_done - t_pre
    audio_dur = len(waveform) / sampling_rate

    t_decode_only = t_decode - t_align_total
    logger.info(
        "Timing: audio_decode %.3fs | vad %.3fs | lang_detect %.3fs | mel+submit %.3fs | "
        "queue_wait %.3fs (%d tokens) | align %.3fs | total %.3fs | RTFx %.1f (%.1fs audio)",
        t_audio_decode - t_pre,
        t_vad - t_audio_decode,
        t_lang - t_lang_start,
        t_pre_done - t_mel_start,
        t_decode_only, total_tokens,
        t_align_total,
        t_total,
        audio_dur / t_total if t_total > 0 else 0,
        audio_dur,
    )

    return {
        "text": " ".join(all_texts),
        "language": language,
        "segments": segments,
        "words": all_words,
    }
