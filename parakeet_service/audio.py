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

from parakeet_service.config import TARGET_SR, get_logger

logger = get_logger("parakeet_service.audio")

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

            # Validate audio file
            if sr_orig <= 0:
                logger.error(f"Invalid sample rate in audio file: {sr_orig}")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Invalid audio file: sample rate must be positive",
                )

            if channels <= 0:
                logger.error(f"Invalid channel count in audio file: {channels}")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Invalid audio file: must have at least one channel",
                )

            logger.debug(f"Audio file info: {sr_orig}Hz, {channels} channels")

            # Create temp output file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            dst = Path(tmp.name)

            with sf.SoundFile(
                tmp.name, "w", samplerate=16000, channels=1, subtype="PCM_16"
            ) as out:
                # Process in 10-second chunks
                chunk_size = 10 * sr_orig
                total_frames_processed = 0

                while True:
                    chunk = snd.read(int(chunk_size), dtype="float32")
                    if len(chunk) == 0:
                        break

                    # Convert to mono if needed
                    if channels > 1:
                        chunk = np.mean(chunk, axis=1)

                    # Resample if needed using librosa
                    if sr_orig != 16000:
                        try:
                            chunk = librosa.resample(
                                chunk, orig_sr=sr_orig, target_sr=16000
                            )
                        except Exception as e:
                            logger.error(
                                f"Resampling failed at frame {total_frames_processed}: {e}"
                            )
                            raise HTTPException(
                                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail=f"Audio resampling failed: {e}",
                            ) from e

                    out.write(chunk)
                    total_frames_processed += len(chunk)

            logger.debug(
                f"Streaming conversion completed: {total_frames_processed} frames processed"
            )
            return src, dst

    except sf.LibsndfileError as e:
        logger.error(f"libsndfile error processing {src}: {e}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported or corrupted audio format: {e}",
        ) from e
    except MemoryError as e:
        logger.error(f"Insufficient memory for audio processing of {src}")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Audio file too large to process",
        ) from e
    except Exception as e:
        logger.error(f"Streaming conversion failed for {src}: {e}")
        logger.exception("Streaming conversion error details:")
        # Fallback to standard conversion
        try:
            logger.info("Attempting fallback to standard conversion")
            return ensure_mono_16k_standard(src)
        except Exception as fallback_e:
            logger.error(f"Fallback conversion also failed: {fallback_e}")
            logger.exception("Fallback conversion error details:")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Audio processing failed: {e}",
            ) from fallback_e


def ensure_mono_16k_standard(src: Path) -> Tuple[Path, Path]:
    """
    Standard full-file audio conversion (fallback)
    """
    try:
        logger.debug(f"Loading audio file with librosa: {src}")
        wav, sr = librosa.load(src, sr=None, mono=False)
        logger.debug(
            f"Loaded audio: shape={wav.shape if hasattr(wav, 'shape') else 'scalar'}, sr={sr}"
        )
    except librosa.LibrosaError as e:
        logger.error(f"Librosa failed to load audio {src}: {e}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Failed to load audio file: {e}",
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error loading audio {src}: {e}")
        logger.exception("Audio loading error details:")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Audio loading failed: {e}",
        ) from e

    try:
        # Handle stereo to mono conversion
        if wav.ndim > 1:
            logger.debug("Converting stereo to mono")
            wav = np.mean(wav, axis=0)

        # Resample if needed
        if sr != TARGET_SR:
            logger.debug(f"Resampling from {sr}Hz to {TARGET_SR}Hz")
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)

        if src.suffix.lower() == ".wav" and sr == TARGET_SR:
            # If already correct format, can reuse the file
            logger.debug("Audio file already in correct format, reusing original")
            return src, src

        dst = src.with_suffix(".wav")
        logger.debug(f"Writing processed audio to: {dst}")
        sf.write(dst, wav, TARGET_SR, subtype="PCM_16")
        return src, dst

    except Exception as e:
        logger.error(f"Audio processing failed for {src}: {e}")
        logger.exception("Audio processing error details:")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Audio processing failed: {e}",
        ) from e


def ensure_mono_16k(src: Path) -> Tuple[Path, Path]:
    """
    Down-mix and resample to mono/16 kHz using streaming when possible.
    """
    if src.suffix.lower() not in SUPPORTED_EXTS:
        logger.error(f"Unsupported file type: {src.suffix}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type {src.suffix}. Supported formats: {', '.join(SUPPORTED_EXTS)}",
        )

    # For WAV files that are already mono and 16kHz, no conversion needed
    if src.suffix.lower() == ".wav":
        try:
            with sf.SoundFile(src) as snd:
                if snd.samplerate == 16000 and snd.channels == 1:
                    logger.debug("Audio file already in correct format")
                    return src, src
        except Exception as e:
            logger.warning(
                f"Failed to check WAV file format for {src}: {e}, proceeding with conversion"
            )

    # Use streaming conversion for other cases
    logger.debug(f"Converting audio file: {src}")
    return convert_audio_streaming(src)


def schedule_cleanup(tasks: BackgroundTasks, *paths: Path) -> None:
    """Schedule cleanup of temporary files."""
    cleanup_count = 0
    for p in paths:
        if p and p.exists():
            tasks.add_task(p.unlink, missing_ok=True)
            cleanup_count += 1

    if cleanup_count > 0:
        logger.debug(f"Scheduled cleanup of {cleanup_count} temporary files")
    else:
        logger.debug("No temporary files to clean up")
