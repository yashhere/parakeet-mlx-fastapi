# Parakeet-TDT 0.6B v2 FastAPI STT Service

A lightweight FastAPI wrapper around NVIDIA’s Parakeet-TDT 0.6B v2 model for high-accuracy English speech-to-text, designed to follow the [OpenAI Audio API specification](https://platform.openai.com/docs/api-reference/audio) in interface and behavior. Includes a REST transcription endpoint, health checks, model-configuration debug, and an experimental WebSocket streaming endpoint.

## Features

- **RESTful transcription**:  
  - `POST /transcribe` accepts `multipart/form-data` audio uploads.  
  - Optional word/character/segment timestamps.  
  - Response model closely mirrors OpenAI’s `TranscriptionResponse` schema.

- **Health & debug**:  
  - `GET /healthz` for liveness/readiness probes.  
  - `GET /debug/cfg` to dump the active NeMo config via [OmegaConf](https://github.com/omry/omegaconf).

- **Experimental streaming**:  
  - `WebSocket /ws` ingests PCM frames, runs [Silero VAD](https://github.com/snakers4/silero-vad) and emits partial/full transcription JSON.

- **Batch worker**:  
  - Asynchronous micro-batching via an internal queue, leveraging NVIDIA’s [NeMo ASR toolkit](https://github.com/NVIDIA/NeMo) for model inference in FP16.

- **Audio preprocessing**:  
  - Automatic downmix & resample to mono 16 kHz using [Torchaudio](https://github.com/pytorch/audio).  
  - File‐type validation with FastAPI’s `UploadFile`.

## Table of Contents

- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [Running the Server](#running-the-server)  
- [Usage](#usage)  
  - [REST API](#rest-api)  
  - [WebSocket Streaming](#websocket-streaming)  
- [Architecture Overview](#architecture-overview)  
- [Contributing](#contributing)  
- [License](#license)  

## Prerequisites

- Python 3.10+  
- GPU with CUDA 11.7+ (optional; CPU fallback supported but slower)  

## Installation

```bash
git clone https://github.com/Shadowfita/parakeet-tdt-0.6b-v2-fastapi.git
cd parakeet-tdt-fastapi

# create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install --upgrade pip
pip install torch torchaudio nemo_toolkit['asr'] fastapi uvicorn python-multipart pydantic omegaconf
pip install git+https://github.com/snakers4/silero-vad.git  # Silero VAD via TorchHub
````

## Configuration

All config values live in [`config.py`](./config.py). Envronment variables coming soon.

## Running the Server

### Uvicorn

```bash
uvicorn parakeet_service.main:app \
  --host 0.0.0.0 --port 8000 \
  --log-level info \
  --workers 1
```

> **Tip:** Use `--workers=1` to avoid multiple model loads. Lifespan events handle GPU allocation and cleanup per process.

### Docker (example)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "parakeet_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Usage

### REST API

#### `GET /healthz`

```bash
curl http://localhost:8000/healthz
# {"status":"ok"}
```

#### `POST /transcribe`

Mimics OpenAI’s [Create transcription](https://platform.openai.com/docs/api-reference/audio/create):

```bash
curl -X POST http://localhost:8000/transcribe \
  -F file="@audio.wav" \
  -F include_timestamps=false
```

**Parameters** (form-data)

| Name                 | Type      | Description                         |
| -------------------- | --------- | ----------------------------------- |
| `file`               | `audio/*` | Audio upload (wav, mp3, flac, etc.) |
| `include_timestamps` | `bool`    | Whether to return offset metadata   |

**Response** (`200 OK`)

```json
{
  "text": "Transcribed text goes here.",
  "timestamps": {
    "words": [ /* word-level offsets */ ],
    "segments": [ /* segment-level offsets */ ]
  }
}
```

#### `GET /debug/cfg`

Dumps the model’s active Hydra/OmegaConf config for debugging:

```bash
curl http://localhost:8000/debug/cfg
```

### WebSocket Streaming

#### `ws://localhost:8000/ws`

A rough prototype implementing a streaming ASR over WebSocket:

* **Client → Server**: send raw 16 kHz PCM frames (int16 mono).
* **Server → Client**: JSON messages

  * `{ "status": "queued" }` when frames are queued for inference
  * `{ "text": "partial or final transcription" }` as results arrive

**Example (JavaScript)**

```js
const ws = new WebSocket("ws://localhost:8000/ws");
ws.onopen = () => /* send PCM chunks via ws.send(u8Array) */;
ws.onmessage = evt => console.log("received", JSON.parse(evt.data));
```

## Architecture Overview

1. **`main.py`**

   * FastAPI app factory, includes REST & streaming routers.
   * Lifespan hook loads the NeMo model in FP16 and starts the background batch worker.

2. **`model.py`**

   * Handles model load/unload and decoding reset for fast path.
   * Utility to convert Torch/Numpy tensors to JSON-safe types.

3. **`routes.py`**

   * Implements `/healthz`, `/transcribe`, and debug endpoints.
   * Uses `audio.py` for downmixing & resampling.

4. **`stream_routes.py`**

   * WebSocket router driving `StreamingVAD` and a global `batchworker`.

5. **`streaming_vad.py`**

   * Voice Activity Detection via [Silero VAD](https://github.com/snakers4/silero-vad).
   * Emits WAV files of utterances for batching.

6. **`batchworker.py`**

   * Central asyncio queue and micro-batch logic.
   * Invokes `model.transcribe` and notifies WebSocket consumers.

7. **`audio.py`**

   * Audio helpers: `ensure_mono_16k` and cleanup scheduling.

8. **`config.py`**

   * Centralized logging and environment variable defaults.

## Contributing

1. Fork the repository and create your feature branch.
2. Run tests (if added) and linting.
3. Submit a pull request with a clear description of your changes.

Please adhere to existing code style and add docstrings for new functionality.

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
See [LICENSE](./LICENSE) for details.
