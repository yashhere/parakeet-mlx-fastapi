from __future__ import annotations
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile, status, Request, Form

from .audio import ensure_mono_16k, schedule_cleanup
from .model import _to_builtin
from .schemas import TranscriptionResponse
from .config import logger

from parakeet_service.model import reset_fast_path


router = APIRouter(tags=["speech"])


@router.get("/healthz", summary="Liveness/readiness probe")
def health():
    return {"status": "ok"}


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe an audio file",
)
async def transcribe_audio(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., media_type="audio/*"),
    include_timestamps: bool = Form(
        False, description="Return char/word/segment offsets",
    ),
):
    # 1 – persist upload
    suffix = Path(file.filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        original = Path(tmp.name)
    await file.close()

    original, to_model = ensure_mono_16k(original)
    schedule_cleanup(background_tasks, original, to_model)

    # 2 – run ASR
    model = request.app.state.asr_model

    try:
        out = model.transcribe(
            [str(to_model)],
            batch_size=1,                # avoids auto-batch VRAM spikes
            timestamps=include_timestamps
        )
        if (
          not include_timestamps                     # switch back to model fast-path if timestamps turned off
          and getattr(model.cfg.decoding, "compute_timestamps", False)
        ):
          reset_fast_path(model)                    

    except RuntimeError as exc:
        logger.exception("ASR failed")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(exc)) from exc

    hyp = out[0] if isinstance(out, list) else out[0][0]
    text = getattr(hyp, "text", str(hyp))
    ts_raw = getattr(hyp, "timestamp", None) if include_timestamps else None

    return TranscriptionResponse(text=text,
                                 timestamps=_to_builtin(ts_raw) if ts_raw else None)

@router.get("/debug/cfg")
def show_cfg(request: Request):
    from omegaconf import OmegaConf
    model = request.app.state.asr_model         
    yaml_str = OmegaConf.to_yaml(model.cfg, resolve=True) 
    return yaml_str