from __future__ import annotations
import asyncio
import tempfile
from pathlib import Path
from collections import defaultdict

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    UploadFile,
    status,
    Request,
    Form,
)

from parakeet_service import config
from parakeet_service.audio import ensure_mono_16k, schedule_cleanup
from parakeet_service.schemas import TranscriptionResponse
from parakeet_service.config import logger


router = APIRouter(tags=["speech"])


@router.get("/healthz", summary="Liveness/readiness probe")
def health():
    return {"status": "ok"}


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
    file: UploadFile = File(..., media_type="audio/*"),
    include_timestamps: bool = Form(
        False,
        description="Return char/word/segment offsets",
    ),
    should_chunk: bool = Form(
        True, description="If true (default), enable chunking for long audio files"
    ),
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
                    if not chunk:
                        break
                    f.write(chunk)

        # Run FFmpeg if processing MP3
        if ffmpeg_cmd:
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
    except asyncio.CancelledError:
        # Clean up temporary files if processing was canceled
        if tmp_path.exists():
            tmp_path.unlink()
        if mp3_tmp_path and mp3_tmp_path.exists():
            mp3_tmp_path.unlink()
        raise
    except BrokenPipeError:
        logger.error("FFmpeg process terminated unexpectedly")
        if tmp_path.exists():
            tmp_path.unlink()
        if mp3_tmp_path and mp3_tmp_path.exists():
            mp3_tmp_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audio processing failed due to FFmpeg crash",
        )
    finally:
        await file.close()

    # Process audio to ensure mono 16kHz
    original, to_model = ensure_mono_16k(tmp_path)

    logger.info("transcribe(): processing audio file")

    # Clean up temporary files
    cleanup_files = [original, to_model]
    if mp3_tmp_path:
        cleanup_files.append(mp3_tmp_path)
    schedule_cleanup(background_tasks, *cleanup_files)

    # Run ASR with parakeet-mlx (chunking handled internally)
    model = request.app.state.asr_model

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

    except Exception as exc:
        logger.exception("ASR failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return TranscriptionResponse(text=merged_text, timestamps=timestamps)


@router.get("/debug/cfg")
def show_cfg(_request: Request):
    """Show model configuration"""
    config_info = {
        "model_name": config.MODEL_NAME,
        "sample_rate": config.TARGET_SR,
        "precision": config.MODEL_PRECISION,
        "framework": "MLX",
    }
    return config_info
