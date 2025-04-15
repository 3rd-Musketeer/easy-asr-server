#!/usr/bin/env python3
"""
Example script for managing hotwords on the easy_asr_server.

This script demonstrates how to:
1. Get the current list of hotwords from the server.
2. Update the list of hotwords on the server by reading from a file.

Dependencies:
- requests: for HTTP requests

Usage:
    # Get current hotwords from default server (http://localhost:8000)
    python manage_hotwords.py get

    # Get hotwords from a specific server
    python manage_hotwords.py get --server http://your-server:port

    # Set hotwords using a file (one hotword per line)
    python manage_hotwords.py set --input-file new_hotwords.txt

    # Set hotwords on a specific server
    python manage_hotwords.py set --input-file new_hotwords.txt --server http://your-server:port
"""

import argparse
import sys
import os
import requests
import json

DEFAULT_SERVER_URL = "http://localhost:8000"

def get_current_hotwords(server_url):
    """Fetches the current hotwords from the server."""
    api_url = f"{server_url}/asr/hotwords"
    print(f"Fetching hotwords from {api_url}...")
    try:
        response = requests.get(api_url)
        response.raise_for_status() # Raise exception for bad status codes
        
        hotwords = response.json()
        print("\nCurrent Hotwords:")
        if hotwords:
            for word in hotwords:
                print(f"- {word}")
        else:
            print("(No hotwords configured or list is empty)")
            
    except requests.exceptions.RequestException as e:
        print(f"\nError getting hotwords: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status Code: {e.response.status_code}")
            try:
                detail = e.response.json().get("detail")
                if detail:
                    print(f"Detail: {detail}")
                else:
                    print(f"Response: {e.response.text}")
            except json.JSONDecodeError:
                 print(f"Response: {e.response.text}")
        sys.exit(1)


def set_new_hotwords(server_url, input_file_path):
    """Reads hotwords from a file and sends them to the server."""
    api_url = f"{server_url}/asr/hotwords"
    
    # Read hotwords from the input file
    if not os.path.isfile(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        sys.exit(1)
        
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Process lines: strip whitespace, filter empty lines
        new_hotwords_list = [line.strip() for line in lines if line.strip()]
    except IOError as e:
        print(f"Error reading input file {input_file_path}: {e}")
        sys.exit(1)

    print(f"\nRead {len(new_hotwords_list)} hotwords from {input_file_path}.")
    print(f"Sending new hotword list to {api_url}...")
    
    try:
        response = requests.put(api_url, json=new_hotwords_list)
        response.raise_for_status() # Raise exception for bad status codes (4xx, 5xx)
        
        # Status code 204 means success for PUT
        if response.status_code == 204:
             print("\nHotwords updated successfully on the server.")
        else:
             # Should not happen if raise_for_status works, but as fallback
             print(f"\nUnexpected success status code: {response.status_code}")
             print(f"Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"\nError setting hotwords: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status Code: {e.response.status_code}")
            try:
                detail = e.response.json().get("detail")
                if detail:
                    print(f"Detail: {detail}")
                else:
                    print(f"Response: {e.response.text}")
            except json.JSONDecodeError:
                 print(f"Response: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Manage hotwords on the Easy ASR Server.')
    parser.add_argument('--server', default=DEFAULT_SERVER_URL, 
                        help=f'ASR server URL (default: {DEFAULT_SERVER_URL})')
    
    subparsers = parser.add_subparsers(dest='command', required=True, 
                                       help='Sub-command to execute')
    
    # Subparser for the "get" command
    parser_get = subparsers.add_parser('get', help='Get the current hotwords')
    
    # Subparser for the "set" command
    parser_set = subparsers.add_parser('set', help='Set hotwords from a file')
    parser_set.add_argument('--input-file', required=True, 
                            help='Path to the input file containing new hotwords (one per line).')

    args = parser.parse_args()

    if args.command == 'get':
        get_current_hotwords(args.server)
    elif args.command == 'set':
        set_new_hotwords(args.server, args.input_file)

if __name__ == "__main__":
    main() 