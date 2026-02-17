"""
Benchmark: isolate overhead sources in continuous batching.
Test with and without attention capture (alignment_heads).
"""
import json
import os
import shutil
import time
import numpy as np

from ctranslate2._ext import StorageView, WhisperContinuousBatcher

MODEL_PATH = "models/whisper-large-v3"
MODEL_PATH_NO_ATTN = "/tmp/whisper-large-v3-no-attn"
DEVICE = "cuda"
DEVICE_INDEX = 0
COMPUTE_TYPE = "float16"
BEAM_SIZE = 5
MAX_SLOTS = 4
NUM_REQUESTS = 4

# Create a model copy WITHOUT alignment_heads (disables attention capture)
if not os.path.exists(MODEL_PATH_NO_ATTN):
    shutil.copytree(MODEL_PATH, MODEL_PATH_NO_ATTN)
with open(os.path.join(MODEL_PATH_NO_ATTN, "config.json")) as f:
    cfg = json.load(f)
del cfg["alignment_heads"]
with open(os.path.join(MODEL_PATH_NO_ATTN, "config.json"), "w") as f:
    json.dump(cfg, f)

# Prepare features
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.audio import pad_or_trim
import tokenizers
from faster_whisper.tokenizer import Tokenizer

fe = FeatureExtractor(feature_size=128)
waveform = np.load("tests/data/audio/jfk.npy")
audio_duration = len(waveform) / 16000
mel = fe(waveform)
mel = pad_or_trim(mel, length=fe.nb_max_frames)
features_np = mel[np.newaxis].astype(np.float32)

hf_tok = tokenizers.Tokenizer.from_file(f"{MODEL_PATH}/tokenizer.json")
tokenizer = Tokenizer(hf_tok, multilingual=True, task="transcribe", language="en")
prompt = list(tokenizer.sot_sequence)

print(f"Audio: {audio_duration:.1f}s, {NUM_REQUESTS} requests, beam={BEAM_SIZE}, slots={MAX_SLOTS}")

def bench_batcher(model_path, label):
    batcher = WhisperContinuousBatcher(
        model_path=model_path,
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

    # Warm up (2 rounds to stabilize)
    for _ in range(2):
        wid = batcher.submit(StorageView.from_array(features_np), prompt)
        batcher.get_result(wid)

    # Timed run (5 iterations)
    times = []
    for run in range(5):
        t0 = time.time()
        req_ids = []
        for i in range(NUM_REQUESTS):
            req_ids.append(batcher.submit(StorageView.from_array(features_np), prompt))

        for rid in req_ids:
            r = batcher.get_result(rid)

        t = time.time() - t0
        times.append(t)
        ntok = len(r.sequences_ids[0])
        print(f"  {label} run {run+1}: {t:.3f}s ({ntok} tokens)")

    best = min(times)
    avg = sum(times) / len(times)
    batcher.stop()
    return best, avg

# Test 1: WITH attention capture (alignment_heads present)
print(f"\n{'='*60}")
print("WITH attention capture (alignment_heads in config)")
print(f"{'='*60}")
best_attn, avg_attn = bench_batcher(MODEL_PATH, "attn=ON ")

# Test 2: WITHOUT attention capture
print(f"\n{'='*60}")
print("WITHOUT attention capture (alignment_heads removed)")
print(f"{'='*60}")
best_no_attn, avg_no_attn = bench_batcher(MODEL_PATH_NO_ATTN, "attn=OFF")

# Also test with beam_size=1 (greedy) to see beam overhead
print(f"\n{'='*60}")
print("WITHOUT attention + GREEDY (beam_size=1)")
print(f"{'='*60}")
BEAM_SIZE_ORIG = BEAM_SIZE
BEAM_SIZE = 1

def bench_greedy(model_path, label):
    batcher = WhisperContinuousBatcher(
        model_path=model_path,
        max_slots=MAX_SLOTS,
        device=DEVICE,
        device_index=DEVICE_INDEX,
        compute_type=COMPUTE_TYPE,
        beam_size=1,
        patience=1.0,
        length_penalty=1.0,
        max_length=448,
        suppress_blank=True,
        suppress_tokens=[-1],
        max_initial_timestamp_index=50,
    )
    batcher.start()

    for _ in range(2):
        wid = batcher.submit(StorageView.from_array(features_np), prompt)
        batcher.get_result(wid)

    times = []
    for run in range(5):
        t0 = time.time()
        req_ids = []
        for i in range(NUM_REQUESTS):
            req_ids.append(batcher.submit(StorageView.from_array(features_np), prompt))
        for rid in req_ids:
            r = batcher.get_result(rid)
        t = time.time() - t0
        times.append(t)
        ntok = len(r.sequences_ids[0])
        print(f"  {label} run {run+1}: {t:.3f}s ({ntok} tokens)")

    best = min(times)
    avg = sum(times) / len(times)
    batcher.stop()
    return best, avg

best_greedy, avg_greedy = bench_greedy(MODEL_PATH_NO_ATTN, "greedy  ")

BEAM_SIZE = BEAM_SIZE_ORIG

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Config: {NUM_REQUESTS} requests × {audio_duration:.1f}s audio, max_slots={MAX_SLOTS}")
print(f"")
print(f"  beam=5 + attn capture:  best={best_attn:.3f}s  avg={avg_attn:.3f}s")
print(f"  beam=5 + NO attn:       best={best_no_attn:.3f}s  avg={avg_no_attn:.3f}s")
print(f"  beam=1 + NO attn:       best={best_greedy:.3f}s  avg={avg_greedy:.3f}s")
print(f"")
print(f"Attention capture overhead: {best_attn - best_no_attn:.3f}s ({(best_attn/best_no_attn - 1)*100:.0f}%)")
print(f"Beam search overhead (5 vs 1): {best_no_attn - best_greedy:.3f}s ({(best_no_attn/best_greedy - 1)*100:.0f}%)")
