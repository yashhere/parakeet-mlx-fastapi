from __future__ import annotations

import asyncio
import tempfile
import time
import zlib
from pathlib import Path
from typing import Annotated, List, Literal, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import PlainTextResponse
from parakeet_mlx.cli import _aligned_sentence_to_dict, to_srt, to_txt, to_vtt

from parakeet_service import config
from parakeet_service.audio import ensure_mono_16k, schedule_cleanup
from parakeet_service.config import get_logger
from parakeet_service.schemas import (
    TranscriptionResponse,
    TranscriptionSegment,
    TranscriptionUsage,
    TranscriptionWord,
)

logger = get_logger("parakeet_service.routes")

router = APIRouter(tags=["speech"])


@router.get("/healthz", summary="Liveness/readiness probe")
def health(request: Request):
    # Check if model is loaded successfully
    model_loaded = getattr(request.app.state, "model_loaded", False)
    model_error = getattr(request.app.state, "model_error", None)

    if model_loaded:
        return {"status": "ok", "model_status": "loaded"}
    elif model_error:
        # Log the detailed error but don't expose it to clients
        logger.warning(f"Model loading failed: {model_error}")
        return {"status": "degraded", "model_status": "failed"}
    else:
        return {"status": "starting", "model_status": "loading"}


@router.post(
    "/audio/transcriptions",
    summary="Transcribe an audio file (OpenAI-compatible)",
    response_model=TranscriptionResponse,
    response_model_exclude_none=True,
)
async def transcribe_audio(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="The audio file to transcribe")],
    model: Annotated[
        str, Form(description="Model to use for transcription")
    ] = "parakeet",
    language: Annotated[
        Optional[str],
        Form(description="Language of the input audio (only 'en' supported)"),
    ] = None,
    prompt: Annotated[
        Optional[str], Form(description="Not supported by parakeet (ignored)")
    ] = None,
    response_format: Annotated[
        Literal["json", "text", "srt", "verbose_json", "vtt"],
        Form(description="Format of the output"),
    ] = "json",
    temperature: Annotated[
        float, Form(description="Sampling temperature between 0 and 1", ge=0.0, le=1.0)
    ] = 0.0,
    timestamp_granularities: Annotated[
        Optional[List[Literal["word", "segment"]]],
        Form(
            description="Timestamp granularities to populate (requires verbose_json format)"
        ),
    ] = None,
    chunking_strategy: Annotated[
        Literal["auto"],
        Form(description="Chunking strategy to use (always 'auto' for parakeet)"),
    ] = "auto",
    stream: Annotated[
        Optional[bool], Form(description="Not supported by parakeet (ignored)")
    ] = False,
):
    # Validate language constraint
    if language and language != "en":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Parakeet only supports English language ('en'). Other languages are not supported.",
        )

    # Validate timestamp granularities constraint
    if timestamp_granularities and response_format != "verbose_json":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="timestamp_granularities parameter requires response_format to be 'verbose_json'",
        )

    # Log ignored prompt parameter
    if prompt:
        logger.warning(
            "prompt parameter is not supported by parakeet and will be ignored"
        )

    # Log ignored stream parameter
    if stream:
        logger.warning(
            "stream parameter is not supported by parakeet and will be ignored"
        )

    # Create temp file with appropriate extension
    suffix = Path(file.filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)

    mp3_tmp_path = None

    # Stream upload directly to processing with cancellation handling
    try:
        # Use FFmpeg for MP3 files to fix header issues
        # Create temp MP3 file if needed
        if suffix.lower() == ".mp3":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_tmp:
                mp3_tmp_path = Path(mp3_tmp.name)
            # Write entire MP3 to temporary file
            with open(mp3_tmp_path, "wb") as f:
                while True:
                    try:
                        chunk = await file.read(8192)
                    except asyncio.CancelledError:
                        logger.warning("File upload cancelled during MP3 saving")
                        raise
                    except Exception as e:
                        logger.error(f"Error reading file chunk: {e}")
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Failed to read audio file: {e}",
                        ) from e
                    if not chunk:
                        break
                    f.write(chunk)

            # Update FFmpeg command to read from file
            ffmpeg_cmd = [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(mp3_tmp_path),
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                str(tmp_path),
            ]
        else:
            ffmpeg_cmd = None
            # For non-MP3, stream directly to file
            with open(tmp_path, "wb") as f:
                while True:
                    try:
                        chunk = await file.read(8192)
                    except asyncio.CancelledError:
                        logger.warning("File upload cancelled during processing")
                        raise
                    except Exception as e:
                        logger.error(f"Error reading file chunk: {e}")
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Failed to read audio file: {e}",
                        ) from e
                    if not chunk:
                        break
                    f.write(chunk)

        # Run FFmpeg if processing MP3
        if ffmpeg_cmd:
            try:
                process = await asyncio.create_subprocess_exec(
                    *ffmpeg_cmd,
                    stdout=asyncio.subprocess.DEVNULL,  # We don't need stdout
                    stderr=asyncio.subprocess.PIPE,
                )

                # Read stderr in real-time
                stderr_lines = []
                if process.stderr:  # Check if stderr is not None
                    while True:
                        line = await process.stderr.readline()
                        if not line:
                            break
                        line_str = line.decode().strip()
                        stderr_lines.append(line_str)
                        logger.debug(f"FFmpeg: {line_str}")

                # Wait for process to finish
                return_code = await process.wait()
                stderr_str = "\n".join(stderr_lines)

                if return_code != 0:
                    logger.error(f"FFmpeg failed with return code {return_code}")
                    logger.error(f"FFmpeg error output: {stderr_str}")
                    raise HTTPException(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        detail=f"Invalid audio format: {stderr_str[:200]}",
                    )
                else:
                    logger.debug("FFmpeg completed successfully")

            except FileNotFoundError as e:
                logger.error("FFmpeg not found on system")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="FFmpeg is required for MP3 processing but not found on system",
                ) from e
            except Exception as e:
                logger.error(f"FFmpeg processing error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Audio processing failed: {e}",
                ) from e

    except asyncio.CancelledError:
        # Clean up temporary files if processing was canceled
        logger.info("Request cancelled, cleaning up temporary files")
        if tmp_path.exists():
            tmp_path.unlink()
        if mp3_tmp_path and mp3_tmp_path.exists():
            mp3_tmp_path.unlink()
        raise
    except BrokenPipeError as err:
        logger.error("FFmpeg process terminated unexpectedly")
        if tmp_path.exists():
            tmp_path.unlink()
        if mp3_tmp_path and mp3_tmp_path.exists():
            mp3_tmp_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audio processing failed due to FFmpeg crash",
        ) from err
    except HTTPException:
        # Re-raise HTTP exceptions without wrapping
        raise
    except Exception as e:
        logger.error(f"Unexpected error during file processing: {e}")
        logger.exception("File processing error details:")
        if tmp_path.exists():
            tmp_path.unlink()
        if mp3_tmp_path and mp3_tmp_path.exists():
            mp3_tmp_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during file processing",
        ) from e
    finally:
        try:
            await file.close()
        except Exception as e:
            logger.warning(f"Error closing uploaded file: {e}")

    # Process audio to ensure mono 16kHz
    try:
        original, to_model = ensure_mono_16k(tmp_path)
        logger.info("transcribe(): processing audio file")
    except HTTPException:
        # Re-raise HTTP exceptions from audio processing
        raise
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        logger.exception("Audio preprocessing error details:")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Audio preprocessing failed: {e}",
        ) from e

    # Clean up temporary files
    cleanup_files = [original, to_model]
    if mp3_tmp_path:
        cleanup_files.append(mp3_tmp_path)
    schedule_cleanup(background_tasks, *cleanup_files)

    try:
        if not getattr(request.app.state, "model_loaded", False):
            error_msg = getattr(
                request.app.state, "model_error", "Model failed to load"
            )
            logger.error(f"ASR model not available: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Speech recognition service temporarily unavailable",
            )

        model = request.app.state.asr_model
        if model is None:
            logger.error("ASR model is None despite model_loaded being True")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Speech recognition service temporarily unavailable",
            )

    except AttributeError:
        logger.error("ASR model state not available in app state")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Speech recognition service temporarily unavailable",
        ) from None

    try:
        start_time = time.time()

        chunk_duration_param = 60 * 2
        overlap_duration_param = 15

        result = model.transcribe(
            str(to_model),
            chunk_duration=chunk_duration_param,
            overlap_duration=overlap_duration_param,
        )

        duration = time.time() - start_time
        merged_text = result.text

        # Determine timestamp granularities to include
        include_words = False
        include_segments = False

        audio_duration = _get_audio_duration(result, to_model)

        if timestamp_granularities:
            include_words = "word" in timestamp_granularities
            include_segments = "segment" in timestamp_granularities
        elif response_format == "verbose_json":
            # Default timestamp granularity is segment for verbose_json
            include_segments = True

        if response_format == "text":
            content = to_txt(result)
            return PlainTextResponse(content=content, media_type="text/plain")

        elif response_format == "srt":
            content = to_srt(result, highlight_words=False)
            return PlainTextResponse(content=content, media_type="application/x-subrip")

        elif response_format == "vtt":
            content = to_vtt(result, highlight_words=False)
            return PlainTextResponse(content=content, media_type="text/vtt")

        else:  # json or verbose_json
            response_data = {
                "task": "transcribe",
                "language": "english",
                "duration": audio_duration,
                "text": merged_text,
            }

            # Add usage information (rounded to nearest integer)
            response_data["usage"] = TranscriptionUsage(seconds=round(audio_duration))

            # Add words OR segments (mutually exclusive) based on timestamp_granularities
            if include_words:
                # Only include words, not segments
                words = []
                for sentence in result.sentences:
                    if hasattr(sentence, "tokens") and sentence.tokens:
                        for token in sentence.tokens:
                            words.append(
                                TranscriptionWord(
                                    word=token.text.strip(),
                                    start=round(token.start, 3),
                                    end=round(token.end, 3),
                                )
                            )
                if words:
                    response_data["words"] = words
            elif include_segments:
                # Only include segments, not words
                segments = []
                for i, sentence in enumerate(result.sentences):
                    sentence_dict = _aligned_sentence_to_dict(sentence)

                    # Extract token IDs from the sentence tokens
                    token_ids = []
                    for token in sentence.tokens:
                        token_ids.append(token.id)

                    # Convert to OpenAI-compatible segment format
                    segment = TranscriptionSegment(
                        id=i,
                        seek=0,  # parakeet-mlx doesn't provide seek offset
                        start=sentence_dict["start"],
                        end=sentence_dict["end"],
                        text=sentence_dict["text"],
                        tokens=token_ids,
                        temperature=temperature,
                        avg_logprob=-0.5,  # Default reasonable value
                        compression_ratio=len(sentence_dict["text"])
                        / len(zlib.compress(sentence_dict["text"].encode("utf-8"))),
                        no_speech_prob=0.0,
                    )
                    segments.append(segment)
                if segments:
                    response_data["segments"] = segments

            logger.info(
                "Transcription completed successfully, text length: {}, processing time: {:.2f}s".format(
                    len(merged_text), duration
                )
            )
            return TranscriptionResponse(**response_data)

    except MemoryError as e:
        logger.error("Insufficient memory for ASR processing")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Audio file too large for available memory",
        ) from e
    except TimeoutError as e:
        logger.error("ASR processing timed out")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Audio processing timed out",
        ) from e
    except Exception as exc:
        logger.error(f"ASR processing failed: {exc}")
        logger.exception("ASR processing error details:")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Speech recognition processing failed",
        ) from exc


def _calculate_audio_duration(audio_path: Path) -> float:
    """Calculate audio duration from file."""
    try:
        import wave

        with wave.open(str(audio_path), "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            return frames / sample_rate
    except Exception:
        # Fallback - return 0 if we can't determine duration
        return 0.0


def _get_audio_duration(result, audio_path: Path) -> float:
    """Get audio duration from parakeet result or calculate from file."""
    # Try to get duration from parakeet result first
    if hasattr(result, "duration") and result.duration is not None:
        return result.duration

    # If parakeet result has sentences, calculate from last sentence end time
    if hasattr(result, "sentences") and result.sentences:
        return max(sentence.end for sentence in result.sentences)

    # Fallback to calculating from audio file
    return _calculate_audio_duration(audio_path)


@router.get("/debug/cfg")
def show_cfg(request: Request):
    """Show model configuration"""
    try:
        # Get the actual model name being used (from app state or config default)
        actual_model_name = getattr(
            request.app.state, "model_name", config.DEFAULT_MODEL_NAME
        )

        config_info = {
            "model_name": actual_model_name,
            "sample_rate": config.TARGET_SR,
            "precision": config.MODEL_PRECISION,
            "framework": "MLX",
        }
        return config_info
    except Exception as e:
        logger.error(f"Error retrieving configuration: {e}")
        logger.exception("Configuration retrieval error details:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve configuration",
        ) from e
