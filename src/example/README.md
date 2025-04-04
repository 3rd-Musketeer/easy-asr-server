# Easy ASR Server Examples

This directory contains example scripts that demonstrate how to use the Easy ASR Server.

## Setup

Before running the examples, make sure you have the required dependencies installed:

```bash
# Install the required packages
uv pip install requests numpy pyaudio
```

Note: PyAudio may require additional system dependencies:
- On macOS: `brew install portaudio`
- On Ubuntu/Debian: `sudo apt-get install python3-pyaudio`
- On Windows: PyAudio should install via pip without additional dependencies

## Example Scripts

### 1. Record and Recognize (record_and_recognize.py)

This script records audio from your microphone and sends it to the ASR server for recognition.

```bash
# Basic usage (records 5 seconds of audio and sends it to localhost:8000)
python record_and_recognize.py

# Specify a different server URL
python record_and_recognize.py --server http://example.com:8000

# Change the recording duration (in seconds)
python record_and_recognize.py --duration 10
```

### 2. Send Existing Audio File (send_file.py)

This script sends an existing audio file to the ASR server for recognition.

```bash
# Basic usage (send a file to localhost:8000)
python send_file.py --file path/to/your/audio.wav

# Specify a different server URL
python send_file.py --file path/to/your/audio.wav --server http://example.com:8000
```

## Troubleshooting

1. **Server Connection Issues**
   - Make sure the Easy ASR Server is running
   - Check that the server URL is correct
   - Verify network connectivity

2. **Audio Recording Issues**
   - Ensure your microphone is connected and working
   - Check system permissions for microphone access
   - Try adjusting microphone input volume

3. **Recognition Quality**
   - Ensure you're speaking clearly and not too quickly
   - Minimize background noise
   - Try using pre-recorded high-quality audio files

## Additional Notes

- The recorded audio is saved as `recorded_audio.wav` in the current directory
- The server expects audio in WAV format, but it will try to convert other formats if possible
- For best results, use audio with a 16kHz sample rate and a single channel (mono) 