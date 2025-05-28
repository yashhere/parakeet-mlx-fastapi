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
git clone https://github.com/<your-org>/parakeet-tdt-fastapi.git
cd parakeet-tdt-fastapi

# create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install --upgrade pip
pip install torch torchaudio nemo_toolkit['asr'] fastapi uvicorn python-multipart pydantic omegaconf
pip install git+https://github.com/snakers4/silero-vad.git  # Silero VAD via TorchHub
