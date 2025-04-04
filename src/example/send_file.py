#!/usr/bin/env python3
"""
Simple file-based ASR recognition example for easy_asr_server

This script demonstrates how to:
1. Send an existing audio file to the easy_asr_server
2. Receive and display the recognition results

Dependencies:
- requests: for HTTP requests

Usage:
    python send_file.py --file AUDIO_FILE [--server SERVER_URL]

Example:
    python send_file.py --file sample.wav --server http://localhost:8000
"""

import argparse
import sys
import os
import requests


def send_file_to_asr_server(file_path, server_url):
    """
    Send an audio file to the ASR server for recognition
    
    Args:
        file_path: Path to the audio file
        server_url: URL of the ASR server
        
    Returns:
        str: The recognized text
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return None
        
    # Construct the API endpoint URL
    api_url = f"{server_url}/asr/recognize"
    
    print(f"Sending file {file_path} to ASR server at {api_url}...")
    
    try:
        # Open the file in binary mode
        with open(file_path, 'rb') as f:
            # Create the files payload
            files = {
                'audio': (os.path.basename(file_path), f, 'audio/wav')
            }
            
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
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    """Main function to send an audio file for recognition"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Send audio file to ASR server')
    parser.add_argument('--file', required=True, help='Path to the audio file')
    parser.add_argument('--server', default='http://localhost:8000', help='ASR server URL (default: http://localhost:8000)')
    args = parser.parse_args()
    
    # Send the file and get the result
    recognized_text = send_file_to_asr_server(args.file, args.server)
    
    # Display the result
    if recognized_text:
        print("\nRecognized text:")
        print("-" * 50)
        print(recognized_text)
        print("-" * 50)


if __name__ == "__main__":
    main() 