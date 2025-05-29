from contextlib import asynccontextmanager
import contextlib
import gc
import torch, asyncio
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict

from .config import MODEL_NAME, logger

from parakeet_service.batchworker import batch_worker


def _to_builtin(obj):
    """torch/NumPy → pure-Python (JSON-safe)."""
    import numpy as np
    import torch as th

    if isinstance(obj, (th.Tensor, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    return obj


@asynccontextmanager
async def lifespan(app):
    """Load model once per process; free GPU on shutdown."""
    logger.info("Loading %s …", MODEL_NAME)
    with torch.inference_mode():
        model_fp32 = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME, map_location="cpu")
        model_fp16 = model_fp32.to(dtype=torch.float16) # weights = 1.2 GB #TODO: make optional via envars
        del model_fp32
        model = model_fp16.cuda()
        
    gc.collect()  # should reclaim the unreferenced fp32 RAM
    torch.cuda.empty_cache() 

    app.state.asr_model = model
    logger.info("Model ready on %s", next(model.parameters()).device)

    app.state.worker = asyncio.create_task(batch_worker(model), name="batch_worker")
    logger.info("batch_worker scheduled")

    try:
        yield
    finally:
        app.state.worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await app.state.worker

        logger.info("Releasing GPU memory and shutting down worker")
        del app.state.asr_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # free cache but keep driver


def reset_fast_path(model):
    """Restore low-latency decoding flags."""
    with open_dict(model.cfg.decoding):
        if getattr(model.cfg.decoding, "compute_timestamps", False):
            model.cfg.decoding.compute_timestamps = False
        if getattr(model.cfg.decoding, "preserve_alignments", False):
            model.cfg.decoding.preserve_alignments = False
    model.change_decoding_strategy(model.cfg.decoding)