# Easy ASR Server

A simple high-concurrency speech recognition service based on FunASR.

## Overview

Easy ASR Server provides a REST API for automatic speech recognition with support for high-concurrency workloads. It leverages the FunASR library for Voice Activity Detection (VAD) and Automatic Speech Recognition (ASR).

Key features:
- REST API endpoints for speech recognition (`/asr/recognize`) and health check (`/asr/health`)
- Support for processing multiple audio formats, automatically converting to the required format
- Multi-worker support for handling concurrent requests
- Automatic GPU detection and utilization if available
- Simple CLI-based configuration

## Installation

### Prerequisites

- Python 3.9 or higher

### Setup

#### Option 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and environment manager.

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
uv pip install -e .
```

#### Option 2: Using standard Python tools

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install the package with pip
pip install -e .

# Alternatively, install using requirements.txt
pip install -r requirements.txt
pip install -e .
```

#### Option 3: Install directly from GitHub

```bash
# Install the latest version from GitHub
pip install git+https://github.com/username/easy_asr_server.git

# Replace 'username' with the actual GitHub username/organization
```

## Usage

### Starting the Server

You can start the server in two ways:

#### Method 1: Using the run command (recommended)

```bash
# Start with default settings (host=127.0.0.1, port=8000, workers=1, device=auto)
easy-asr-server run

# Specify host and port
easy-asr-server run --host 0.0.0.0 --port 9000

# Specify the number of workers for concurrency
easy-asr-server run --workers 4

# Manually specify device
easy-asr-server run --device cpu  # Force CPU usage
easy-asr-server run --device cuda  # Force GPU usage if available

# Combine options
easy-asr-server run --host 0.0.0.0 --port 8080 --workers 2 --device auto
```

#### Method 2: Direct command

You can also run the server directly without the 'run' command:

```bash
# Start with default settings
easy-asr-server

# With options
easy-asr-server --host 0.0.0.0 --port 9000 --workers 4 --device auto
```

### API Endpoints

#### POST /asr/recognize

Processes an audio file and returns the transcribed text.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Form data with an `audio` file field

**Response:**
- Status: 200 OK
- Content-Type: application/json
- Body: `{"text": "transcribed text"}`

**Example:**

```python
import requests

# Send audio file for recognition
with open("audio.wav", "rb") as f:
    audio_data = f.read()

response = requests.post(
    "http://127.0.0.1:8000/asr/recognize",
    files={"audio": ("input_audio.wav", audio_data)}
)

if response.status_code == 200:
    print(response.json()["text"])
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

#### GET /asr/health

Checks if the ASR service is healthy and ready to accept requests.

**Response:**
- Status: 200 OK
- Content-Type: application/json
- Body: `{"status": "healthy"}`

### Error Handling

The API returns appropriate HTTP status codes for different error scenarios:

- **400 Bad Request**: Invalid input (not an audio file or unsupported format)
- **422 Unprocessable Entity**: Request format error (missing required fields)
- **500 Internal Server Error**: Server-side processing error
- **503 Service Unavailable**: Service not ready or temporarily unavailable

## Models

The service uses the following models from ModelScope:
- ASR model: `iic/SenseVoiceSmall`
- VAD model: `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`

Models are automatically downloaded on first use and cached locally.

## Development

For detailed development documentation, refer to the `dev.md` file in the repository.

## Building and Distribution

To build the package for distribution:

```bash
# Install build dependencies
pip install build twine

# Build the package
python -m build

# This will generate distribution files in the dist/ directory:
# - A source distribution (.tar.gz)
# - A wheel distribution (.whl)
```

### Publishing to PyPI

To publish the package to the Python Package Index (PyPI):

```bash
# Test the upload to TestPyPI first
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Upload to the actual PyPI
twine upload dist/*
```

### Installing from a Built Package

```bash
# Install from the wheel file
pip install dist/easy_asr_server-0.1.0-py3-none-any.whl

# Or install from the source distribution
pip install dist/easy-asr-server-0.1.0.tar.gz
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

The underlying models (`iic/SenseVoiceSmall` and `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`) are also licensed under Apache License 2.0.
