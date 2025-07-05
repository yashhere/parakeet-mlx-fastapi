"""
Audio helpers:
* ensure_mono_16k(path)  -> Path (possibly rewritten .wav)
* schedule_cleanup(background, *paths)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf  # type: ignore
from fastapi import BackgroundTasks, HTTPException, status

from parakeet_service.config import TARGET_SR, logger

SUPPORTED_EXTS: List[str] = [".wav", ".flac", ".mp3", ".ogg", ".opus"]


def convert_audio_streaming(src: Path) -> Tuple[Path, Path]:
    """
    Stream audio conversion to mono/16kHz with minimal memory usage
    Processes audio in chunks to avoid loading entire file into memory
    """
    try:
        with sf.SoundFile(src, "r") as snd:
            sr_orig = snd.samplerate
            channels = snd.channels
            # Create temp output file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            dst = Path(tmp.name)

            with sf.SoundFile(
                tmp.name, "w", samplerate=16000, channels=1, subtype="PCM_16"
            ) as out:
                # Process in 10-second chunks
                chunk_size = 10 * sr_orig
                while True:
                    chunk = snd.read(int(chunk_size), dtype="float32")
                    if len(chunk) == 0:
                        break

                    # Convert to mono if needed
                    if channels > 1:
                        chunk = np.mean(chunk, axis=1)

                    # Resample if needed using librosa
                    if sr_orig != 16000:
                        chunk = librosa.resample(
                            chunk, orig_sr=sr_orig, target_sr=16000
                        )

                    out.write(chunk)

            return src, dst

    except Exception as e:
        logger.error(f"Streaming conversion failed: {e}")
        # Fallback to standard conversion
        return ensure_mono_16k_standard(src)


def ensure_mono_16k_standard(src: Path) -> Tuple[Path, Path]:
    """
    Standard full-file audio conversion (fallback)
    """
    wav, sr = librosa.load(src, sr=None, mono=False)

    # Handle stereo to mono conversion
    if wav.ndim > 1:
        wav = np.mean(wav, axis=0)

    # Resample if needed
    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)

    if src.suffix.lower() == ".wav" and sr == TARGET_SR:
        # If already correct format, can reuse the file
        return src, src

    dst = src.with_suffix(".wav")
    sf.write(dst, wav, TARGET_SR, subtype="PCM_16")
    return src, dst


def ensure_mono_16k(src: Path) -> Tuple[Path, Path]:
    """
    Down-mix and resample to mono/16 kHz using streaming when possible.
    """
    if src.suffix.lower() not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type {src.suffix}",
        )

    # For WAV files that are already mono and 16kHz, no conversion needed
    if src.suffix.lower() == ".wav":
        try:
            with sf.SoundFile(src) as snd:
                if snd.samplerate == 16000 and snd.channels == 1:
                    return src, src
        except:  # noqa
            pass

    # Use streaming conversion for other cases
    return convert_audio_streaming(src)


def schedule_cleanup(tasks: BackgroundTasks, *paths: Path) -> None:
    for p in paths:
        tasks.add_task(p.unlink, missing_ok=True)
