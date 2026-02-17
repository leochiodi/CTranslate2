"""
Benchmark: old model.generate() vs continuous batching
Both with beam_size=5, 4 concurrent requests, 35s real French audio.
"""
import os
import time
import numpy as np
import soundfile as sf

from ctranslate2._ext import StorageView, WhisperContinuousBatcher
import ctranslate2

MODEL_PATH = os.environ.get("WHISPER_MODEL",
    "../ctranslate2-server/models/whisper-large-v3")
AUDIO_PATH = os.environ.get("AUDIO_PATH",
    "../ctranslate2-server/f1_35s.wav")
DEVICE = "cuda"
DEVICE_INDEX = 0
COMPUTE_TYPE = "float16"
BEAM_SIZE = 5
MAX_SLOTS = 4
NUM_REQUESTS = 4

# Load test audio (raw waveform) and convert to mel features
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.audio import pad_or_trim

fe = FeatureExtractor(feature_size=128)
waveform, sr = sf.read(AUDIO_PATH, dtype="float32")
if waveform.ndim > 1:
    waveform = waveform.mean(axis=1)
if sr != 16000:
    import torchaudio, torch
    waveform = torchaudio.transforms.Resample(sr, 16000)(
        torch.from_numpy(waveform).unsqueeze(0)).squeeze(0).numpy()
audio_duration = len(waveform) / 16000
print(f"Audio: {AUDIO_PATH} ({audio_duration:.1f}s)")

mel = fe(waveform)
mel = pad_or_trim(mel, length=fe.nb_max_frames)
features_np = mel[np.newaxis].astype(np.float32)  # [1, n_mels, 3000]

# Build prompt
import tokenizers
hf_tok = tokenizers.Tokenizer.from_file(f"{MODEL_PATH}/tokenizer.json")

from faster_whisper.tokenizer import Tokenizer
tokenizer = Tokenizer(hf_tok, multilingual=True, task="transcribe", language="fr")
prompt = list(tokenizer.sot_sequence)
print(f"Prompt tokens: {prompt}")

# =====================================================
# BENCHMARK 1: Old model.generate() with batch of 4
# =====================================================
print("\n" + "="*60)
print(f"OLD model.generate(): batch={NUM_REQUESTS}, beam_size={BEAM_SIZE}")
print("="*60)

model = ctranslate2.models.Whisper(MODEL_PATH, device=DEVICE, device_index=DEVICE_INDEX,
                                    compute_type=COMPUTE_TYPE)

# Batch the features: [4, n_mels, 3000]
batch_features_np = np.repeat(features_np, NUM_REQUESTS, axis=0)
batch_features = StorageView.from_array(batch_features_np)
batch_prompts = [prompt] * NUM_REQUESTS

# Warm up
print("Warming up old model...")
results = model.generate(StorageView.from_array(features_np), [prompt],
                          beam_size=BEAM_SIZE, max_length=448,
                          suppress_blank=True, suppress_tokens=[-1])
print(f"Warmup result: {tokenizer.decode(results[0].sequences_ids[0])[:80]}...")

# Timed run (3 iterations, take best)
best_old = float('inf')
for run in range(3):
    t0 = time.time()
    results_old = model.generate(batch_features, batch_prompts,
                                  beam_size=BEAM_SIZE, max_length=448,
                                  suppress_blank=True, suppress_tokens=[-1])
    t_old = time.time() - t0
    best_old = min(best_old, t_old)
    print(f"  Run {run+1}: {t_old:.3f}s")

for i, r in enumerate(results_old):
    score = r.scores[0] if r.scores else 0.0
    print(f"  req {i}: {len(r.sequences_ids[0])} tokens, score={score:.2f}")
print(f"OLD model.generate() best time: {best_old:.3f}s for {NUM_REQUESTS} requests")

del model  # free GPU memory
import gc; gc.collect()
import torch; torch.cuda.empty_cache()
time.sleep(1)

# =====================================================
# BENCHMARK 2: Continuous batching with max_slots=4
# =====================================================
print("\n" + "="*60)
print(f"NEW continuous batching: max_slots={MAX_SLOTS}, beam_size={BEAM_SIZE}")
print("="*60)

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

# Warm up
print("Warming up continuous batcher...")
wf = StorageView.from_array(features_np)
wid = batcher.submit(wf, prompt)
wr = batcher.get_result(wid)
print(f"Warmup result: {tokenizer.decode(wr.sequences_ids[0])[:80]}...")

# Timed run (3 iterations, take best)
best_cb = float('inf')
for run in range(3):
    t0 = time.time()
    req_ids = []
    for i in range(NUM_REQUESTS):
        f = StorageView.from_array(features_np)
        req_ids.append(batcher.submit(f, prompt))
    t_submit = time.time() - t0

    # Collect results
    results_cb = {}
    for rid in req_ids:
        r = batcher.get_result(rid)
        results_cb[rid] = (time.time() - t0, r)

    t_cb = time.time() - t0
    best_cb = min(best_cb, t_cb)

    print(f"  Run {run+1}: {t_cb:.3f}s (submit: {t_submit:.3f}s)")
    for rid in req_ids:
        elapsed, r = results_cb[rid]
        print(f"    req {rid}: {len(r.sequences_ids[0])} tokens, done at {elapsed:.3f}s")

print(f"NEW continuous batching best time: {best_cb:.3f}s for {NUM_REQUESTS} requests")

batcher.stop()

# =====================================================
# Summary
# =====================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Audio: {audio_duration:.1f}s × {NUM_REQUESTS} requests")
print(f"Config: beam_size={BEAM_SIZE}, max_slots={MAX_SLOTS}, {COMPUTE_TYPE}")
print(f"OLD model.generate():    {best_old:.3f}s")
print(f"NEW continuous batching: {best_cb:.3f}s")
print(f"Ratio (new/old):         {best_cb/best_old:.2f}x")
if best_cb > best_old:
    print(f"Continuous batching is {best_cb/best_old:.1f}x SLOWER")
else:
    print(f"Continuous batching is {best_old/best_cb:.1f}x FASTER")
