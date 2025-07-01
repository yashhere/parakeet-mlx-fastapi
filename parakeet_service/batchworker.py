import asyncio, contextlib, logging, tempfile, pathlib, time
from typing import Union, List

from parakeet_service import model as mdl

logger = logging.getLogger("batcher")
logger.setLevel(logging.DEBUG)

# -------- shared state -------------------------------------------------------
transcription_queue: asyncio.Queue[str | bytes] = asyncio.Queue()
condition = asyncio.Condition()  # wakes websocket consumers
results: dict[str, str] = {}  # path -> text


# -------- helper -------------------------------------------------------------
def _as_path(blob: Union[str, bytes]) -> str:
    """Ensures we always hand a *file path* to parakeet-mlx."""
    if isinstance(blob, str):
        return blob
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(blob)
    tmp.close()
    return tmp.name


# -------- main worker --------------------------------------------------------
async def batch_worker(model, batch_ms: float = 15.0, max_batch: int = 4):
    """Forever drain `transcription_queue` → ASR → `results`."""
    logger.info("worker started (batch ≤%d, window %.0f ms)", max_batch, batch_ms)
    logger.info("worker started with model")

    while True:
        path = await transcription_queue.get()  # blocks until 1st item
        batch: List[str] = [_as_path(path)]

        # ---------- micro-batch gathering with timeout ----------
        deadline = time.monotonic() + batch_ms / 1000
        while len(batch) < max_batch:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                nxt = await asyncio.wait_for(transcription_queue.get(), timeout)
                batch.append(_as_path(nxt))
            except asyncio.TimeoutError:
                break

        logger.debug("processing %d-file batch", len(batch))

        # ---------- inference ----------
        try:
            # Process each file individually with parakeet-mlx
            # Note: parakeet-mlx doesn't have built-in batching like NeMo
            outs = []
            for file_path in batch:
                result = model.transcribe(file_path)
                outs.append(result)
        except Exception as exc:
            logger.exception("ASR failed: %s", exc)
            for _ in batch:
                transcription_queue.task_done()
            continue

        # ---------- store results & notify ----------
        for p, result in zip(batch, outs):
            # Extract text from parakeet-mlx AlignedResult
            results[p] = result.text if hasattr(result, "text") else str(result)
            transcription_queue.task_done()  # mark done
        async with condition:
            condition.notify_all()

        # ---------- cleanup ----------
        for p in batch:
            with contextlib.suppress(FileNotFoundError):
                pathlib.Path(p).unlink(missing_ok=True)
