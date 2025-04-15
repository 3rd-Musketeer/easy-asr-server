# Easy ASR Server

[![Tests](https://github.com/3rd-Musketeer/easy-asr-server/actions/workflows/test.yml/badge.svg)](https://github.com/3rd-Musketeer/easy-asr-server/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://github.com/3rd-Musketeer/easy-asr-server)

A simple high-concurrency speech recognition service based on FunASR.

## Overview

Easy ASR Server provides a REST API for automatic speech recognition with support for high-concurrency workloads. It leverages the FunASR library and ModelScope models for Voice Activity Detection (VAD), Automatic Speech Recognition (ASR), and Punctuation Restoration.

Key features:
- Selectable ASR pipelines (`sensevoice`, `paraformer`) via CLI.
- REST API endpoints for speech recognition (`/asr/recognize`), health check (`/asr/health` - provides pipeline/device info), and hotword management (`GET /asr/hotwords`, `PUT /asr/hotwords`).
- Support for processing multiple audio formats, automatically converting to the required format
- Multi-worker support for handling concurrent requests
- Automatic GPU detection and utilization if available
- Simple CLI-based configuration
- Configurable hotword list via file (`--hotword-file`) or API, utilized by supported pipelines (e.g., `paraformer`).

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
pip install git+https://github.com/3rd-Musketeer/easy-asr-server.git
```

## Usage

### Starting the Server

# Since there is only one command, invoke the script directly with options.

```bash
# Start with default settings (host=127.0.0.1, port=8000, workers=1, device=auto)
easy-asr-server

# Specify host and port
easy-asr-server --host 0.0.0.0 --port 9000

# Specify the number of workers for concurrency
easy-asr-server --workers 4

# Manually specify device
easy-asr-server --device cpu  # Force CPU usage
easy-asr-server --device cuda  # Force GPU usage if available

# Specify a hotword file (one word per line)
easy-asr-server --hotword-file /path/to/hotwords.txt

# Specify the ASR pipeline (e.g., paraformer for hotword support)
easy-asr-server --pipeline paraformer

# Combine options
easy-asr-server --host 0.0.0.0 --port 8080 --workers 2 --device auto --hotword-file ./my_hotwords.txt
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
- Body: `{"status": "healthy", "pipeline": "<loaded_pipeline_type>", "device": "<configured_device>"}`

#### GET /asr/hotwords

Gets the current list of hotwords from the configured file.

**Response:**
- Status: 200 OK
- Content-Type: application/json
- Body: `["hotword1", "hotword2", ...]`

**Example:**

```python
import requests

response = requests.get("http://127.0.0.1:8000/asr/hotwords")

if response.status_code == 200:
    print(f"Current hotwords: {response.json()}")
else:
    print(f"Failed to get hotwords: {response.status_code} - {response.text}")
```

#### PUT /asr/hotwords

Updates the hotword file with a new list, overwriting the previous content. Requires the server to be started with the `--hotword-file` option.

**Request:**
- Method: PUT
- Content-Type: application/json
- Body: `["new_hotword1", "new_hotword2", ...]`

**Response:**
- Status: 204 No Content (on success)

**Example:**

```python
import requests

new_list = ["OpenAI", "ChatGPT", "FunASR"]
response = requests.put("http://127.0.0.1:8000/asr/hotwords", json=new_list)

if response.status_code == 204:
    print("Hotwords updated successfully.")
else:
    print(f"Failed to update hotwords: {response.status_code} - {response.text}")
```

### Error Handling

The API returns appropriate HTTP status codes for different error scenarios:

- **400 Bad Request**: Invalid input (not an audio file or unsupported format)
- **422 Unprocessable Entity**: Request format error (missing required fields)
- **500 Internal Server Error**: Server-side processing error
- **503 Service Unavailable**: Service not ready or temporarily unavailable

## Models

The server supports different underlying ASR pipelines, configurable via the `--pipeline` CLI option. Models are automatically downloaded from ModelScope on first use and cached locally.

Available Pipelines:

*   **`sensevoice` (Default)**:
    *   Uses `iic/SenseVoiceSmall`.
    *   Provides robust ASR with punctuation and VAD included in the main model.
    *   **Does not** currently support hotword boosting.
*   **`paraformer`**:
    *   Uses `iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404` (ASR), `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` (VAD), and `iic/punc_ct-transformer_cn-en-common-vocab471067-large` (Punctuation).
    *   **Supports** hotword boosting. Hotwords provided via the `--hotword-file` option or the `/asr/hotwords` API endpoint will be used when this pipeline is selected.

## Examples

The `src/example` directory contains scripts demonstrating how to interact with the server's API and perform basic performance testing:

*   `record_and_recognize.py`: Records audio from your microphone and sends it for recognition.
*   `send_file.py`: Sends an existing audio file for recognition.
*   `manage_hotwords.py`: Gets the current hotword list from the server or updates it using a local file.
*   `performance_test.py`: Runs concurrent requests against the server using either a local file or generated audio to measure performance.

Please refer to the `src/example/README.md` file for detailed instructions on setting up and running the client examples (`record_and_recognize.py`, `send_file.py`, `manage_hotwords.py`). See the "Performance Testing" section below for details on `performance_test.py`.

## Performance Testing

A basic performance testing script is included in `src/example/performance_test.py`. This script allows you to test the server's throughput and latency under different concurrency levels using either a specific audio file or generated sine wave audio.

**Dependencies:** The required dependencies (`requests`, `numpy`, `rich`) are included in the main package installation.

**Run the Test Script:**

Navigate to the `src/example` directory or provide the full path to the script.

```bash
# Example: Test with sample.wav using 1, 5, 10 concurrent clients, 20 repetitions each
python src/example/performance_test.py --file path/to/your/sample.wav --concurrency 1 5 10 --repetitions 20

# Example: Test with generated 1s and 3s audio, concurrency 1, 2, 4, 8, 10 repetitions each
python src/example/performance_test.py --durations 1.0 3.0 --concurrency 1 2 4 8 --repetitions 10

# Example: Test against a different server
python src/example/performance_test.py --file path/to/your/sample.wav --server http://192.168.1.100:9000 --concurrency 1 5 10

# See all options
python src/example/performance_test.py --help
```

The script will output a table summarizing the results for each tested configuration (audio source + concurrency level), including success/error counts, latency statistics (average, median, P90, P99), and throughput (requests per second/minute).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

The underlying models (
`iic/SenseVoiceSmall`, 
`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`, 
`iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404`, and
`iic/punc_ct-transformer_cn-en-common-vocab471067-large`
) are also licensed under Apache License 2.0.
