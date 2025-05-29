"""
Offline VAD-aware splitters
───────────────────────────
* vad_chunk            – legacy helper (loads full file, unchanged)
  - target 60-s chunks (±10 s)
  - never cut mid-utterance (trailing silence ≥ 300 ms)
  - returns List[pathlib.Path] of temp .wav files

* vad_chunk_streaming  – New low-RAM streamer
"""

from __future__ import annotations
import tempfile, wave, pathlib, numpy as np
from typing import List
import soundfile as sf


from torch.hub import load as torch_hub_load

vad_model, vad_utils = torch_hub_load("snakers4/silero-vad", "silero_vad")
get_speech_ts, _, _, VADIterator, _ = vad_utils 

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
    wav = wav.mean(0).numpy()

    speech_ts = get_speech_ts(
        wav, vad_model, sampling_rate=SAMPLE_RATE,
        threshold=THRESH, min_silence_duration_ms=TRAIL_SIL_MS,
    )
    if not speech_ts:
        return []

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

    paths = []
    for g in groups:
        start, end = g[0]["start"], g[-1]["end"]
        clip = wav[start:end]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
            wf.writeframes(
                np.clip(clip * 32768, -32768, 32767).astype(np.int16).tobytes()
            )
        paths.append(pathlib.Path(tmp.name))
    return paths

STRIPE_SEC        = 2                         # read 2-second stripes
STRIPE_FRAMES     = SAMPLE_RATE * STRIPE_SEC
MAX_CHUNK_MS      = 60_000                    # hard 60s cap
SPEECH_PAD_MS     = 120                       # same as live VAD

def _flush(buf: bytearray) -> pathlib.Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(bytes(buf))
    return pathlib.Path(tmp.name)

def vad_chunk_streaming(path: pathlib.Path) -> List[pathlib.Path]:
    """
    Stream the file in small stripes and split on VADIterator boundaries.
    Uses the SAME PyTorch Silero model, but keeps only a few seconds in RAM.
    """
    vad_iter = VADIterator(
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=THRESH,
        min_silence_duration_ms=TRAIL_SIL_MS,
        speech_pad_ms=SPEECH_PAD_MS,
    )

    paths, buf = [], bytearray()
    speech_ms  = 0

    with sf.SoundFile(path) as snd:
        while True:
            audio = snd.read(frames=STRIPE_FRAMES, dtype="int16", always_2d=False)
            if not len(audio):
                break

            # Normalise to float32 [-1,1] for VADIterator
            audio_f32 = audio.astype("float32") / 32768

            # Feed 512-sample windows
            for start in range(0, len(audio_f32), 512):
                window = audio_f32[start:start+512]
                if len(window) < 512:
                    break
                evt = vad_iter(window)
                buf.extend(audio[start:start+512].tobytes())
                speech_ms += 32

                if (evt and evt.get("end")) or speech_ms >= MAX_CHUNK_MS:
                    paths.append(_flush(buf))
                    buf.clear()
                    speech_ms = 0

    if buf:
        paths.append(_flush(buf))
    return paths
