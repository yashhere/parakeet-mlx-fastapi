"""
Offline VAD-aware splitter:
* target 30-s chunks (±10 s)
* never cut mid-utterance (trailing silence ≥ 300 ms)
* returns List[pathlib.Path] of temp .wav files
"""

from __future__ import annotations
import tempfile, pathlib, wave
from typing import List
import numpy as np, torch
from torch.hub import load as torch_hub_load


vad_model, vad_utils = torch_hub_load("snakers4/silero-vad", "silero_vad")
get_speech_ts, _, _, _, collect_chunks = vad_utils 

SAMPLE_RATE        = 16_000
TARGET_SEC         = 60
MAX_SEC            = 70          # never exceed this in one chunk
TRAIL_SIL_MS       = 300         # keep ≥300 ms silence at cut point
THRESH             = 0.60        # stricter prob threshold

def vad_chunk(path: pathlib.Path) -> List[pathlib.Path]:
    import torchaudio
    wav, sr = torchaudio.load(str(path))
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav.mean(0).numpy()               # (T,)
    # --- 1) get speech timestamp dicts ----------------------------------
    speech_ts = get_speech_ts(
        wav, vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=THRESH,
        min_silence_duration_ms=TRAIL_SIL_MS,
    )
    if not speech_ts:                       # no speech at all
        return []

    # --- 2) collect chunks -------------------------------------
    groups, current, cur_len = [], [], 0
    for seg in speech_ts:
        seg_len = seg["end"] - seg["start"]
        if cur_len + seg_len > TARGET_SEC * SAMPLE_RATE and cur_len > 0:
            groups.append(current); current, cur_len = [], 0
        current.append(seg); cur_len += seg_len
        if cur_len > MAX_SEC * SAMPLE_RATE:
            groups.append(current); current, cur_len = [], 0
    if current:
        groups.append(current)

    # --- 3) write each group to temp .wav --------------------------------
    paths = []
    for g in groups:
        start  = g[0]["start"]
        end    = g[-1]["end"]
        clip   = wav[start:end]
        tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
            wf.writeframes(
                np.clip(clip * 32768, -32768, 32767).astype(np.int16).tobytes()
            )
        paths.append(pathlib.Path(tmp.name))
    return paths
