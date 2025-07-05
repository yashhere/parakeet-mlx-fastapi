from __future__ import annotations

import asyncio
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Annotated

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

from parakeet_service import config
from parakeet_service.audio import ensure_mono_16k, schedule_cleanup
from parakeet_service.config import get_logger
from parakeet_service.schemas import TranscriptionResponse

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
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe an audio file",
)
@router.post(
    "/audio/transcriptions",
    response_model=TranscriptionResponse,
    summary="Transcribe an audio file",
)
async def transcribe_audio(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(media_type="audio/*")],
    include_timestamps: Annotated[
        bool,
        Form(description="Return char/word/segment offsets"),
    ] = False,
    should_chunk: Annotated[
        bool,
        Form(description="If true (default), enable chunking for long audio files"),
    ] = True,
):
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

    # Run ASR with parakeet-mlx (chunking handled internally)
    try:
        # Check if model loaded successfully
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
        # Use parakeet-mlx's built-in chunking if enabled
        if should_chunk:
            # Enable chunking with 60-second chunks and 15-second overlap
            result = model.transcribe(
                str(to_model), chunk_duration=60.0, overlap_duration=15.0
            )
        else:
            # Process without chunking
            result = model.transcribe(str(to_model))

        merged_text = result.text
        timestamps = None

        if include_timestamps and hasattr(result, "sentences"):
            # Convert parakeet-mlx AlignedSentence objects to compatible format
            merged = defaultdict(list)
            for sentence in result.sentences:
                # Create word-level timestamps from tokens if available
                if hasattr(sentence, "tokens"):
                    for token in sentence.tokens:
                        merged["words"].append(
                            {"text": token.text, "start": token.start, "end": token.end}
                        )

                # Add segment-level timestamps
                merged["segments"].append(
                    {
                        "text": sentence.text,
                        "start": sentence.start,
                        "end": sentence.end,
                    }
                )
            timestamps = dict(merged)

        logger.info(
            f"Transcription completed successfully, text length: {len(merged_text)}"
        )
        return TranscriptionResponse(text=merged_text, timestamps=timestamps)

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
