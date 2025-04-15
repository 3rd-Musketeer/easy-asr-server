#!/usr/bin/env python3
"""
Concurrent performance testing script for easy_asr_server.

This script sends multiple concurrent requests to the ASR server's
/asr/recognize endpoint using a specified audio file and measures
response times and throughput for different concurrency levels.

Dependencies:
- requests

Usage:
    python performance_test.py --file AUDIO_FILE --concurrency C1 C2 C3 ... [--server SERVER_URL] [--repetitions N]

Example:
    # Test with concurrency levels 1, 5, 10 using sample.wav, 5 repetitions each
    python performance_test.py --file sample.wav --concurrency 1 5 10 --repetitions 5 

    # Test with concurrency levels 1, 2, 4, 8, 16 against a different server
    python performance_test.py --file my_audio.wav --server http://192.168.1.100:8000 --concurrency 1 2 4 8 16
"""

import argparse
import time
import requests
import os
import statistics
import concurrent.futures
from collections import defaultdict
import io
import wave
import struct
import numpy as np # Added for audio generation
from typing import Union, List, Dict, Any # Added Dict, Any
from rich.console import Console # Added rich imports
from rich.table import Table

# Initialize rich console
console = Console()

# --- New Function: Generate WAV Audio ---
def generate_wav_audio(duration_s: float, sample_rate: int = 16000, frequency: float = 440.0) -> bytes:
    """
    Generates a mono WAV audio byte stream containing a sine wave.

    Args:
        duration_s: Duration of the audio in seconds.
        sample_rate: Sample rate in Hz (default: 16000).
        frequency: Frequency of the sine wave in Hz (default: 440).

    Returns:
        bytes: The WAV audio data as bytes.
    """
    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0., duration_s, num_samples, endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5 # Use half of max amplitude
    data = amplitude * np.sin(2. * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    pcm_data = data.astype(np.int16)

    # Create WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data.tobytes())
    
    buffer.seek(0)
    return buffer.read()
# --- End New Function ---

def send_request(server_url: str, file_path_or_desc: str, file_content: bytes, file_name: str):
    """
    Sends a single ASR request and returns the response time in seconds.

    Args:
        server_url: URL of the ASR server.
        file_path_or_desc: Original path of the audio file or a description (used for logging).
        file_content: Content of the audio file in bytes.
        file_name: Base name of the audio file (used in the request).

    Returns:
        float: Response time in seconds, or None if the request failed.
    """
    api_url = f"{server_url}/asr/recognize"
    start_time = time.monotonic()
    try:
        files = {
            'audio': (file_name, file_content, 'audio/wav') 
        }
        # Increased timeout for potentially longer generated files
        response = requests.post(api_url, files=files, timeout=120) # 120 second timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        # Check if we got a valid JSON response with 'text'
        try:
            result = response.json()
            if 'text' not in result:
                 # print(f"Warning: Request successful but 'text' key missing in response: {result}")
                 # Treat as success for latency measurement, but could indicate server-side issue
                 pass 
        except requests.exceptions.JSONDecodeError:
            # print(f"Warning: Request successful but response was not valid JSON: {response.text[:100]}...")
            # Treat as success for latency measurement
             pass
        
        end_time = time.monotonic()
        return end_time - start_time
    except requests.exceptions.Timeout:
        return "Error: Timeout"
    except requests.exceptions.RequestException as e:
        return f"Error: Request failed ({e})"
    except Exception as e:
        return f"Error: Unexpected ({e})"

# Modified signature to use typing.Union for compatibility
def run_performance_test(
    server_url: str, 
    concurrency_levels: List[int], 
    repetitions: int,
    file_path: Union[str, None] = None, 
    durations: Union[List[float], None] = None, 
    sample_rate: int = 16000
):
    """
    Runs the performance test, collects all results, and prints a summary at the end.
    """
    
    aggregated_results = [] # Store all results here
    test_configs = [] # Store configurations tested

    console.print(f"[bold cyan]--- Starting Performance Test ---[/bold cyan]")
    console.print(f"[cyan]Server URL:[/cyan] {server_url}")
    console.print(f"[cyan]Concurrency Levels:[/cyan] {concurrency_levels}")
    console.print(f"[cyan]Repetitions per Level:[/cyan] {repetitions}")

    if file_path:
        # --- File-based Test ---
        if not os.path.exists(file_path):
            console.print(f"[bold red]Error:[/bold red] Test audio file '{file_path}' not found.")
            return

        console.print(f"[cyan]Test File:[/cyan] {file_path}")
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
            file_name = os.path.basename(file_path)
            console.print(f"Audio file loaded successfully ({len(file_content)} bytes). Loading...", end='\r')
            test_configs.append({'description': file_path, 'content': file_content, 'name': file_name})
        except Exception as e:
            console.print(f"[bold red]Error loading audio file '{file_path}':[/bold red] {e}")
            return

    elif durations:
        # --- Generated Audio Test ---
        console.print(f"[cyan]Generated Audio Durations (s):[/cyan] {durations}")
        console.print(f"[cyan]Generated Audio Sample Rate:[/cyan] {sample_rate} Hz")
        for duration in durations:
            desc = f"Generated {duration:.2f}s audio"
            console.print(f"Generating {desc}...", end='\r')
            try:
                audio_content = generate_wav_audio(duration, sample_rate)
                audio_name = f"generated_{duration:.2f}s.wav"
                test_configs.append({'description': desc, 'content': audio_content, 'name': audio_name})
            except Exception as e:
                console.print(f"[bold red]Error generating audio for duration {duration:.2f}s:[/bold red] {e}")
                # Continue to next duration
    else:
        console.print("[bold red]Error:[/bold red] No file specified and no durations specified.")
        return
    
    console.print("Test setup complete. Running tests..." + " " * 20) # Clear loading message

    # --- Run Tests for all configurations --- 
    total_tests = len(test_configs) * len(concurrency_levels)
    current_test = 0
    for config in test_configs:
        results_for_config = run_tests_for_audio(
            server_url=server_url, 
            audio_description=config['description'],
            audio_content=config['content'], 
            audio_name=config['name'],
            concurrency_levels=concurrency_levels, 
            repetitions=repetitions,
            total_tests=total_tests,
            current_test_offset=current_test
        )
        aggregated_results.extend(results_for_config)
        current_test += len(concurrency_levels)

    console.print("\n[bold cyan]--- Test Complete ---[/bold cyan]")
    # --- Print Final Summary Report --- 
    print_summary_report(aggregated_results)


def run_tests_for_audio(
    server_url: str, 
    audio_description: str,
    audio_content: bytes, 
    audio_name: str,
    concurrency_levels: List[int], 
    repetitions: int,
    total_tests: int,
    current_test_offset: int
) -> List[Dict[str, Any]]:
    """
    Helper function to execute concurrent requests for a given audio and return collected results.
    Does not print intermediate results, only progress.
    Returns a list of result dictionaries.
    """
    results_list = []

    for i, concurrency in enumerate(concurrency_levels):
        current_test_num = current_test_offset + i + 1
        console.print(f"Running Test {current_test_num}/{total_tests} (Audio: '{audio_description}', Concurrency: {concurrency})...", end="\r")
        level_start_time = time.monotonic()
        
        run_latencies = []
        run_errors = 0
        error_messages = defaultdict(int)

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(send_request, server_url, audio_description, audio_content, audio_name)
                for _ in range(concurrency * repetitions) # Submit all requests at once
            ]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if isinstance(result, float):
                    run_latencies.append(result)
                else: # It's an error string
                    run_errors += 1
                    error_messages[result] += 1 # Count specific errors

        level_end_time = time.monotonic()

        results_list.append({
            'audio_description': audio_description,
            'concurrency': concurrency,
            'latencies': run_latencies,
            'errors': run_errors,
            'error_messages': dict(error_messages), # Convert defaultdict back to dict
            'total_time': level_end_time - level_start_time
        })
    
    console.print(" " * 80, end="\r") # Clear progress line
    return results_list


# --- New Function: Print Summary Report ---
def print_summary_report(results: List[Dict[str, Any]]):
    """
    Prints a vertically formatted summary table of all test results using rich.
    Each column represents a test configuration (Audio + Concurrency).
    """
    if not results:
        console.print("[yellow]No results to display.[/yellow]")
        return

    table = Table(title="Performance Test Summary", show_header=True, header_style="bold magenta")

    # --- Setup Columns --- 
    table.add_column("Metric", style="dim", width=18) # First column for metric names
    
    # Sort results for consistent column order
    sorted_results = sorted(results, key=lambda x: (x['audio_description'], x['concurrency']))

    # Add a column for each test result
    for result in sorted_results:
        audio_desc = result['audio_description']
        # Shorten description if it's a file path
        if '/' in audio_desc or '\\' in audio_desc:
            audio_desc = os.path.basename(audio_desc)
        
        header = f"[bold]{audio_desc}[/bold]\nConc: {result['concurrency']}"
        table.add_column(header, justify="right")

    # --- Calculate Metrics for Each Result --- 
    metric_data = defaultdict(list)
    for result in sorted_results:
        latencies = result['latencies']
        errors = result['errors']
        total_time = result['total_time']
        error_messages = result['error_messages']
        success_reqs = len(latencies)

        # Calculate metrics, handling cases with 0 successes
        if success_reqs > 0:
            avg_latency = statistics.mean(latencies) * 1000
            median_latency = statistics.median(latencies) * 1000
            if success_reqs > 10:
                 p90_latency = np.quantile(latencies, 0.90) * 1000
                 p99_latency = np.quantile(latencies, 0.99) * 1000
            elif success_reqs > 1:
                 p90_latency = statistics.quantiles(latencies, n=10)[-1] * 1000
                 p99_latency = statistics.quantiles(latencies, n=100)[-1] * 1000 if success_reqs > 9 else p90_latency
            else:
                 p90_latency = p99_latency = avg_latency
            std_dev = statistics.stdev(latencies) * 1000 if success_reqs > 1 else 0.0
            throughput = success_reqs / total_time if total_time > 0 else 0.0
        else:
            avg_latency = median_latency = p90_latency = p99_latency = std_dev = throughput = 0.0

        metric_data["Success"].append(f"[green]{success_reqs}[/green]")
        metric_data["Errors"].append(f"[red]{errors}[/red]" if errors > 0 else str(errors))
        metric_data["Avg Latency (ms)"].append(f"{avg_latency:.2f}")
        metric_data["Median (P50) (ms)"].append(f"{median_latency:.2f}")
        metric_data["P90 Latency (ms)"].append(f"{p90_latency:.2f}")
        metric_data["P99 Latency (ms)"].append(f"{p99_latency:.2f}")
        metric_data["StdDev (ms)"].append(f"{std_dev:.2f}")
        metric_data["Throughput (req/s)"].append(f"[blue]{throughput:.2f}[/blue]")
        
        # Calculate throughput per minute
        throughput_min = throughput * 60
        metric_data["Throughput (req/min)"].append(f"[blue]{throughput_min:.2f}[/blue]")

        # Prepare error details string
        error_str = "-"
        if error_messages:
             error_str = "; ".join([f"{count}x '{msg}'" for msg, count in error_messages.items()])
        metric_data["Error Details"].append(f"[dim]{error_str}[/dim]")
        

    # --- Add Rows to Table --- 
    metric_order = [
        "Success", "Errors", "Avg Latency (ms)", "Median (P50) (ms)", 
        "P90 Latency (ms)", "P99 Latency (ms)", "StdDev (ms)", 
        "Throughput (req/s)", "Throughput (req/min)", "Error Details"
    ]

    for metric_name in metric_order:
        table.add_row(metric_name, *metric_data[metric_name])
        # Add separator after Errors and Throughput for readability
        if metric_name == "Errors" or metric_name == "Throughput (req/s)":
             table.add_section()

    console.print(table)
# --- End New Function ---

def main():
    parser = argparse.ArgumentParser(
        description='ASR Server Performance Test Tool. Tests using either a specified audio file or generated sine waves of specified durations.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults
    )
    parser.add_argument('--server', default='http://localhost:8000', 
                        help='ASR server URL')
    
    # File vs Generation options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', default=None,
                       help='Path to the audio file to use for testing. If not provided, use --durations.')
    group.add_argument('--durations', nargs='+', type=float, default=None,
                       help='List of audio durations (in seconds) to generate and test (e.g., 1.0 3.0 5.0). Used if --file is not specified.')

    # General options
    parser.add_argument('--concurrency', required=True, nargs='+', type=int, 
                        help='List of concurrency levels to test (e.g., 1 5 10 20)')
    parser.add_argument('--repetitions', type=int, default=10, 
                        help='Number of times to send requests for each concurrency level (total requests = concurrency * repetitions)')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Sample rate in Hz for generated audio (used only if --file is not specified)')

    args = parser.parse_args()

    # Basic validation handled by argparse group, but double-check repetitions
    if args.repetitions <= 0:
        console.print("[bold red]Error:[/bold red] Repetitions must be a positive integer.")
        parser.print_usage()
        return
        
    if args.durations:
        for d in args.durations:
             if d <=0:
                  console.print(f"[bold red]Error:[/bold red] Durations must be positive numbers. Found: {d}")
                  parser.print_usage()
                  return


    # Call the main test function with appropriate arguments
    run_performance_test(
        server_url=args.server, 
        concurrency_levels=args.concurrency, 
        repetitions=args.repetitions,
        file_path=args.file, 
        durations=args.durations, 
        sample_rate=args.sample_rate
    )

if __name__ == "__main__":
    main() 