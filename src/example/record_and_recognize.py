#!/usr/bin/env python3
"""
Simple audio recording and ASR recognition example for easy_asr_server

This script demonstrates how to:
1. Record audio from the microphone
2. Send the recorded audio to the easy_asr_server
3. Receive and display the recognition results

Dependencies:
- pyaudio: for audio recording (install with 'pip install "easy-asr-server[client]"')
- requests: for HTTP requests
- numpy: for audio data handling

Usage:
    python record_and_recognize.py [--server SERVER_URL] [--duration SECONDS]

Example:
    python record_and_recognize.py --server http://localhost:8000 --duration 5
"""

import argparse
import io
import wave
import time
import sys
import requests
import numpy as np

# Try to import PyAudio, which is an optional dependency
try:
    import pyaudio
except ImportError:
    print("PyAudio is required for this example but not installed.")
    print("Install it with: pip install 'easy-asr-server[client]'")
    sys.exit(1)

# Default settings
DEFAULT_SERVER_URL = "http://localhost:8000"
DEFAULT_DURATION = 5  # seconds
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024


def record_audio(duration, sample_rate=SAMPLE_RATE, channels=CHANNELS, chunk=CHUNK, format=FORMAT):
    """
    Record audio from the microphone
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz
        channels: Number of channels (1 for mono, 2 for stereo)
        chunk: Number of frames per buffer
        format: Audio format (e.g., pyaudio.paInt16)
        
    Returns:
        bytes: The recorded audio data in WAV format
    """
    p = pyaudio.PyAudio()
    
    # Open audio stream
    stream = p.open(
        format=format,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk
    )
    
    print(f"Recording for {duration} seconds...")
    
    # Record audio
    frames = []
    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
        # Show progress
        if i % int(sample_rate / chunk) == 0:
            seconds_elapsed = i / (sample_rate / chunk)
            sys.stdout.write(f"\rRecording: {seconds_elapsed:.1f}/{duration} seconds")
            sys.stdout.flush()
    
    print("\nRecording finished!")
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Convert the recorded audio to WAV format
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    
    # Get the WAV data
    wav_buffer.seek(0)
    wav_data = wav_buffer.read()
    
    return wav_data


def send_to_asr_server(audio_data, server_url):
    """
    Send audio data to the ASR server for recognition
    
    Args:
        audio_data: Audio data in WAV format
        server_url: URL of the ASR server
        
    Returns:
        str: The recognized text
    """
    # Construct the API endpoint URL
    api_url = f"{server_url}/asr/recognize"
    
    # Create the files payload
    files = {
        'audio': ('recording.wav', audio_data, 'audio/wav')
    }
    
    print(f"Sending audio to ASR server at {api_url}...")
    
    try:
        # Send the request
        response = requests.post(api_url, files=files)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse and return the result
        result = response.json()
        return result.get('text', 'No text in response')
        
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None


def main():
    """Main function to record audio and send it for recognition"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Record audio and send to ASR server')
    parser.add_argument('--server', default=DEFAULT_SERVER_URL, help=f'ASR server URL (default: {DEFAULT_SERVER_URL})')
    parser.add_argument('--duration', type=int, default=DEFAULT_DURATION, help=f'Recording duration in seconds (default: {DEFAULT_DURATION})')
    args = parser.parse_args()
    
    try:
        # Record audio
        audio_data = record_audio(args.duration)
        
        # Save the audio to a file (optional)
        with open('recorded_audio.wav', 'wb') as f:
            f.write(audio_data)
        print("Audio saved to recorded_audio.wav")
        
        # Send the audio to the ASR server
        recognized_text = send_to_asr_server(audio_data, args.server)
        
        # Display the result
        if recognized_text:
            print("\nRecognized text:")
            print("-" * 50)
            print(recognized_text)
            print("-" * 50)
            
    except KeyboardInterrupt:
        print("\nRecording interrupted.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main() 