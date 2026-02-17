"""
Clean benchmark: old vs continuous batching.
Tests beam=5 and beam=1 for both paths, with proper warmup and stabilization.
"""
import time
import gc
import numpy as np

from ctranslate2._ext import StorageView, WhisperContinuousBatcher
import ctranslate2

MODEL_PATH = "models/whisper-large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
MAX_SLOTS = 4
NUM_REQUESTS = 4
NUM_RUNS = 8

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

batch_features_np = np.repeat(features_np, NUM_REQUESTS, axis=0)

print(f"Audio: {audio_duration:.1f}s × {NUM_REQUESTS} requests, {COMPUTE_TYPE}")
print()

# =====================================================
# OLD model.generate()
# =====================================================
for beam in [5, 1]:
    model = ctranslate2.models.Whisper(MODEL_PATH, device=DEVICE, compute_type=COMPUTE_TYPE)
    bf = StorageView.from_array(batch_features_np)
    bp = [prompt] * NUM_REQUESTS

    # Warmup
    for _ in range(3):
        model.generate(bf, bp, beam_size=beam, max_length=448,
                       suppress_blank=True, suppress_tokens=[-1])

    times = []
    for run in range(NUM_RUNS):
        t0 = time.time()
        model.generate(bf, bp, beam_size=beam, max_length=448,
                       suppress_blank=True, suppress_tokens=[-1])
        times.append(time.time() - t0)

    print(f"OLD generate beam={beam}: {[f'{t:.3f}' for t in times]}")
    print(f"  best={min(times):.3f}s  median={sorted(times)[len(times)//2]:.3f}s  avg={sum(times)/len(times):.3f}s")
    del model; gc.collect()
    time.sleep(0.5)

# =====================================================
# Continuous batching
# =====================================================
for beam in [5, 1]:
    batcher = WhisperContinuousBatcher(
        model_path=MODEL_PATH, max_slots=MAX_SLOTS,
        device=DEVICE, device_index=0, compute_type=COMPUTE_TYPE,
        beam_size=beam, patience=1.0, length_penalty=1.0, max_length=448,
        suppress_blank=True, suppress_tokens=[-1], max_initial_timestamp_index=50,
    )
    batcher.start()

    # Warmup (3 rounds)
    for _ in range(3):
        ids = [batcher.submit(StorageView.from_array(features_np), prompt)
               for _ in range(NUM_REQUESTS)]
        for rid in ids:
            batcher.get_result(rid)
        time.sleep(0.1)  # Let threads settle

    times = []
    for run in range(NUM_RUNS):
        time.sleep(0.1)  # Ensure worker is idle and waiting
        t0 = time.time()
        ids = [batcher.submit(StorageView.from_array(features_np), prompt)
               for _ in range(NUM_REQUESTS)]
        for rid in ids:
            batcher.get_result(rid)
        times.append(time.time() - t0)

    print(f"CB  batch   beam={beam}: {[f'{t:.3f}' for t in times]}")
    print(f"  best={min(times):.3f}s  median={sorted(times)[len(times)//2]:.3f}s  avg={sum(times)/len(times):.3f}s")
    batcher.stop()
    time.sleep(0.5)
