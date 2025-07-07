# Parakeet-MLX FastAPI Service

[![Build Python Package](https://github.com/yashhere/parakeet-mlx-fastapi/actions/workflows/build-package.yml/badge.svg)](https://github.com/yashhere/parakeet-mlx-fastapi/actions/workflows/build-package.yml)

A production-ready, OpenAI-compatible FastAPI service for speech-to-text transcription using Nvidia's Parakeet models. This service runs on macOS systems, powered by `uv` for fast and reliable dependency management.

**⚠️ Platform Restriction: This package is only supported on macOS due to its dependency on MLX (Apple Silicon) and optimized audio processing libraries.**

## Features

- High-accuracy English speech-to-text transcription
- Word/character/segment timestamps
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's Audio API (no streaming support currently).
- RESTful API with FastAPI
- macOS native optimization with MLX

## OpenAI-Compatible API

This service exposes an OpenAI-compatible API endpoint for transcriptions. This means you can use any existing OpenAI client library (Python, Node.js, etc.) to interact with this service by simply changing the `base_url` to point to this service's address.

The API specification is available in the [openapi.yaml](openapi.yaml) file.

### Example with OpenAI Python Client

```python
from openai import OpenAI

# Point the client to the local service
client = OpenAI(base_url="http://localhost:8765/v1", api_key="dummy")

with open("path/to/your/audio.wav", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="dummy",
        file=audio_file,
    )
    print(transcription.text)
```

## Requirements

- **macOS only** (macOS 11.0+ recommended)
- **Apple Silicon Mac** (M1, M2, M3, etc.) for optimal performance
- Python 3.10+
- FFmpeg
- 4GB+ RAM (recommended)
- `uv` package manager

## Quick Start

### Install with `uv tool` (Recommended)

Install the service as a standalone tool that can be run from anywhere:

```bash
# Install directly from Git repository
uv tool install git+https://github.com/yashhere/parakeet-mlx-fastapi.git

# Or install from local directory (if you have the source)
uv tool install .

# Run the service (available globally)
parakeet-server
```

### Local Development

For local development and testing:

```bash
# 1. Clone the repository
git clone https://github.com/yashhere/parakeet-mlx-fastapi.git
cd parakeet-mlx-fastapi

# 2. Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install dependencies
uv sync

# 4. Run the service locally (without installing)
uv run parakeet-server

# 5. Run with custom options
uv run parakeet-server --host 127.0.0.1 --port 9000 --model mlx-community/parakeet-tdt-0.6b-v2 -vv
```

### Tool Management

After installing with `uv tool`, you can manage the installation:

```bash
# List all installed tools
uv tool list

# Upgrade to the latest version
uv tool upgrade parakeet-mlx-fastapi

# Uninstall the tool
uv tool uninstall parakeet-mlx-fastapi
```

### Alternative Installation Methods

#### Build and Install from Source

```bash
# Build the distribution packages
uv build

# Install the wheel package
uv pip install dist/parakeet_mlx_fastapi-0.1.0-py3-none-any.whl

# Run the installed CLI
parakeet-server
```

#### Direct Python Module Execution

```bash
# Install dependencies
uv sync

# Run the main module directly
uv run python -m parakeet_service.main
```

## Why macOS Only?

This package is restricted to macOS for several reasons:

1. **MLX Framework**: Optimized for Apple Silicon processors
2. **Large Model Downloads**: The service downloads multi-GB language models that work best in unconstrained environments
3. **Memory Management**: Better memory handling for large models on Apple Silicon

## Development with `uv`

### Setting up Development Environment

```bash
# Install dev dependencies (includes pre-commit, ruff)
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install
```

#### Running Pre-commit Manually

```bash
# Run on all files
uv run pre-commit run --all-files
```

### Adding Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Add with version constraints
uv add "fastapi>=0.100.0"
```

### Running Development

```bash
# Install dev dependencies
uv sync --extra dev

# Format and lint code
uv run ruff check .
uv run ruff format .

# Or run both linting and formatting together
uv run ruff check --fix .
```

### Creating a New Release

This project uses automated releases triggered by Git tags. The version is automatically derived from the tag using dynamic versioning.

```bash
# 1. Ensure your changes are committed and pushed to main
git add .
git commit -m "Your changes"
git push origin main

# 2. Create and push a version tag (this triggers the release workflow)
git tag v1.2.3  # Use semantic versioning (v1.2.3, v2.0.0, etc.)
git push origin v1.2.3

# The GitHub workflow will automatically:
# - Build the package with version 1.2.3
# - Create a GitHub release
# - Upload wheel and source distributions
# - Generate release notes with installation instructions
```

**Pre-release versions:**

```bash
# For beta/alpha releases
git tag v1.2.3-beta
git push origin v1.2.3-beta
# Creates: parakeet_mlx_fastapi-1.2.3b0-py3-none-any.whl
```

## API Usage

Once the service is running, it will be available at `http://localhost:8765`.

### API Documentation

- Interactive API docs: `http://localhost:8765/docs`
- OpenAPI schema: `http://localhost:8765/openapi.json`

### Basic Transcription

```bash
# Upload an audio file for transcription
curl -X POST "http://localhost:8765/transcribe" \
     -H "Content-Type: multipart/form-data" \
     -F "audio=@your_audio_file.wav"
```

## CLI Usage

The service provides a simple command-line interface with essential options.

### Basic Usage

```bash
# Start the service with default settings
parakeet-server

# Start with custom host and port
parakeet-server --host 127.0.0.1 --port 9000

# Start with a different model
parakeet-server --model mlx-community/parakeet-tdt-0.6b-v2

# Start with verbose logging
parakeet-server -vv

# Combine multiple options
parakeet-server --host 127.0.0.1 --port 9000 --model mlx-community/parakeet-tdt-0.6b-v2 -vvv
```

### CLI Options

**Available Options:**

- `--host, -h`: Host to bind the server to (default: 0.0.0.0)
- `--port, -p`: Port to bind the server to (default: 8765)
- `--model, -m`: Model name to use (default: mlx-community/parakeet-tdt-1.1b)
- `--verbose, -v`: Increase verbosity (-v for WARNING, -vv for INFO, -vvv for DEBUG)

### Environment Variables

The service also supports configuration via environment variables:

- `PARAKEET_WORKERS`: Number of worker processes (default: 1)

### CLI Help

Get help for the CLI:

```bash
# Show help and available options
parakeet-server --help
```

## Binary Distribution

### Building Binaries

```bash
# Build wheel and source distribution
uv build

# Build only wheel
uv build --wheel

# Build only source distribution
uv build --sdist
```

### Installing from Binary

```bash
# Install from local wheel
uv pip install dist/parakeet_mlx_fastapi-0.1.0-py3-none-any.whl

# Install with extras
uv pip install "parakeet-mlx-fastapi[dev]"
```

## File Structure

```
parakeet-mlx-fastapi/
├── parakeet_service/         # Main application package
│   ├── main.py               # FastAPI app and CLI entry point
│   ├── routes.py             # API routes
│   ├── stream_routes.py      # Streaming routes
│   ├── model.py              # Model handling
│   ├── config.py             # Configuration
│   └── ...
├── dist/                     # Built distribution packages
├── pyproject.toml            # Project configuration and dependencies
├── uv.lock                   # Locked dependencies
└── README.md                 # This file
```

## License

See [LICENSE](/LICENSE) file for details.
