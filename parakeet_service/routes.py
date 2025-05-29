from __future__ import annotations
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile, status, Request, Form

from .audio import ensure_mono_16k, schedule_cleanup
from .model import _to_builtin
from .schemas import TranscriptionResponse
from .config import logger

from parakeet_service.model import reset_fast_path
from parakeet_service.chunker import vad_chunk, vad_chunk_streaming


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
    should_chunk: bool = Form(True,
        description="If true (default), split long audio into "
                    "~60s VAD-aligned chunks for batching"),
):
    # 1 – persist upload
    suffix = Path(file.filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        original = Path(tmp.name)
    await file.close()

    original, to_model = ensure_mono_16k(original)

    if should_chunk:
        chunk_paths = vad_chunk_streaming(to_model) or [to_model]
    else:
        chunk_paths = [to_model]

    logger.info("transcribe(): sending %d chunks to ASR", len(chunk_paths))

    schedule_cleanup(background_tasks, original, to_model, *chunk_paths)

    # 2 – run ASR
    model = request.app.state.asr_model

    try:
        outs = model.transcribe(
            [str(p) for p in chunk_paths],
            batch_size=2,
            timestamps=include_timestamps,
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

    if isinstance(outs, tuple):
      outs = outs[0]
    texts = []
    ts_agg = [] if include_timestamps else None
    merged = defaultdict(list)

    for h in outs:
        texts.append(getattr(h, "text", str(h)))
        if include_timestamps:
            for k, v in _to_builtin(getattr(h, "timestamp", {})).items():
                merged[k].extend(v)           # concat lists

    merged_text = " ".join(texts).strip()
    timestamps  = dict(merged) if include_timestamps else None

    return TranscriptionResponse(text=merged_text, timestamps=timestamps)

@router.get("/debug/cfg")
def show_cfg(request: Request):
    from omegaconf import OmegaConf
    model = request.app.state.asr_model         
    yaml_str = OmegaConf.to_yaml(model.cfg, resolve=True) 
    return yaml_str