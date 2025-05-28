"""
Audio helpers:
* ensure_mono_16k(path)  -> Path (possibly rewritten .wav)
* schedule_cleanup(background, *paths)
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, List

import torchaudio
import torchaudio.functional as AF
from fastapi import BackgroundTasks, HTTPException, status

from .config import TARGET_SR, logger


SUPPORTED_EXTS: List[str] = [".wav", ".flac", ".mp3", ".ogg", ".opus"]


def ensure_mono_16k(src: Path) -> Tuple[Path, Path]:
    """
    Down-mix and resample to mono/16 kHz.

    Returns (original_path, path_to_feed_model).
    If conversion is needed, a sibling *.wav* is written.
    """
    if src.suffix.lower() not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type {src.suffix}",
        )

    wav, sr = torchaudio.load(src)             # (ch, time) float32 −1…1
    if wav.shape[0] > 1:                       # stereo → mono
        wav = wav.mean(dim=0, keepdim=True)

    if sr != TARGET_SR:
        wav = AF.resample(wav, sr, TARGET_SR)

    if src.suffix.lower() == ".wav" and sr == TARGET_SR:
        # rewrite header to 16-bit PCM in-place
        torchaudio.save(src, wav, TARGET_SR,
                        encoding="PCM_S", bits_per_sample=16)
        return src, src

    dst = src.with_suffix(".wav")
    torchaudio.save(dst, wav, TARGET_SR,
                    encoding="PCM_S", bits_per_sample=16)
    return src, dst


def schedule_cleanup(tasks: BackgroundTasks, *paths: Path) -> None:
    for p in paths:
        tasks.add_task(p.unlink, missing_ok=True)
