#!/usr/bin/env python3
"""
ASR Performance Testing Script

This script sends concurrent requests to the ASR server and measures latency.
It supports sending a specified number of concurrent requests with a given audio file
and reports statistics on the latency.

Usage:
    python asr_request.py --file AUDIO_FILE --server SERVER_URL --concurrency NUM --repetitions NUM --output CSV_FILE
"""

import argparse
import asyncio
import csv
import logging
import os
import statistics
import time
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

async def send_request(
    session: aiohttp.ClientSession,
    file_path: str,
    server_url: str,
    request_id: int,
    max_retries: int = MAX_RETRIES
) -> Tuple[int, float, Optional[str]]:
    """
    Send a request to the ASR server and measure latency.
    Retries failed requests up to max_retries times.

    Args:
        session: aiohttp client session
        file_path: Path to the audio file
        server_url: URL of the ASR server
        request_id: Identifier for this request
        max_retries: Maximum number of retries for failed requests

    Returns:
        Tuple of (request_id, latency in ms, error message if any)
    """
    start_time = time.time()
    error_msg = None
    retries = 0
    
    # Read file data once to avoid repeated file I/O
    with open(file_path, 'rb') as f:
        file_data = f.read()
    
    while retries <= max_retries:
        try:
            api_url = f"{server_url}/asr/recognize"
            
            form_data = aiohttp.FormData()
            form_data.add_field(
                'audio',
                file_data,
                filename=os.path.basename(file_path),
                content_type='audio/wav'
            )
            
            async with session.post(api_url, data=form_data, timeout=60) as response:
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
                
                if response.status == 200:
                    result = await response.json()
                    logger.debug(f"Request {request_id} succeeded: {result}")
                    return request_id, latency_ms, None
                
                # Status code 503 might mean the server is still initializing
                if response.status == 503 and retries < max_retries:
                    error_text = await response.text()
                    logger.warning(f"Request {request_id} returned {response.status}, retrying ({retries+1}/{max_retries}): {error_text}")
                    retries += 1
                    await asyncio.sleep(RETRY_DELAY)
                    start_time = time.time()  # Reset start time for accurate latency measurement
                    continue
                
                # Other errors or final retry
                error_msg = f"HTTP Error {response.status}: {await response.text()}"
                logger.warning(f"Request {request_id} failed: {error_msg}")
                return request_id, latency_ms, error_msg
                
        except asyncio.TimeoutError:
            # Handle timeout specially
            if retries < max_retries:
                retries += 1
                logger.warning(f"Request {request_id} timed out, retrying ({retries}/{max_retries})")
                await asyncio.sleep(RETRY_DELAY)
                start_time = time.time()  # Reset start time
                continue
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            error_msg = "Request timed out"
            logger.warning(f"Request {request_id} timed out after {max_retries} retries")
            return request_id, latency_ms, error_msg
            
        except Exception as e:
            # For other exceptions, retry if we haven't reached max_retries
            if retries < max_retries:
                retries += 1
                logger.warning(f"Request {request_id} failed with {str(e)}, retrying ({retries}/{max_retries})")
                await asyncio.sleep(RETRY_DELAY)
                start_time = time.time()  # Reset start time
                continue
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            error_msg = str(e)
            logger.warning(f"Request {request_id} exception: {error_msg}")
            return request_id, latency_ms, error_msg


async def run_test(
    file_path: str,
    server_url: str,
    concurrency: int,
    repetitions: int
) -> List[Dict]:
    """
    Run a performance test with the specified parameters.

    Args:
        file_path: Path to the audio file
        server_url: URL of the ASR server
        concurrency: Number of concurrent requests
        repetitions: Number of test repetitions

    Returns:
        List of dictionaries with test results
    """
    results = []

    # Configure the connection pool with higher limits for high concurrency
    conn = aiohttp.TCPConnector(limit=concurrency * 2, limit_per_host=concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=120)
    
    # Create a session for connection pooling
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        for rep in range(repetitions):
            logger.info(f"Running repetition {rep+1}/{repetitions} with concurrency {concurrency}")
            
            # Create tasks for all concurrent requests
            tasks = []
            for i in range(concurrency):
                task = send_request(session, file_path, server_url, i)
                tasks.append(task)
            
            # Execute all tasks concurrently
            rep_start_time = time.time()
            responses = await asyncio.gather(*tasks)
            rep_end_time = time.time()
            
            # Calculate statistics for this repetition
            latencies = [resp[1] for resp in responses]
            errors = [resp[2] for resp in responses if resp[2] is not None]
            
            results.append({
                'repetition': rep + 1,
                'concurrency': concurrency,
                'total_time_ms': (rep_end_time - rep_start_time) * 1000,
                'min_latency_ms': min(latencies),
                'max_latency_ms': max(latencies),
                'avg_latency_ms': statistics.mean(latencies),
                'median_latency_ms': statistics.median(latencies),
                'p95_latency_ms': np.percentile(latencies, 95),
                'stddev_latency_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                'success_rate': (concurrency - len(errors)) / concurrency,
                'errors': len(errors)
            })
            
            # Wait a moment between repetitions to avoid overwhelming the server
            await asyncio.sleep(1)
    
    return results


def calculate_summary_stats(results: List[Dict]) -> Dict:
    """
    Calculate summary statistics across all repetitions.

    Args:
        results: List of result dictionaries from run_test

    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {}
    
    avg_latencies = [result['avg_latency_ms'] for result in results]
    success_rates = [result['success_rate'] for result in results]
    
    return {
        'concurrency': results[0]['concurrency'],
        'repetitions': len(results),
        'avg_latency_ms': statistics.mean(avg_latencies),
        'min_latency_ms': min(r['min_latency_ms'] for r in results),
        'max_latency_ms': max(r['max_latency_ms'] for r in results),
        'p95_latency_ms': statistics.mean([r['p95_latency_ms'] for r in results]),
        'avg_success_rate': statistics.mean(success_rates),
        'throughput_per_min': (results[0]['concurrency'] * 60 * 1000) / statistics.mean(avg_latencies)
    }


async def main():
    parser = argparse.ArgumentParser(description='ASR Server Performance Test')
    parser.add_argument('--file', required=True, help='Path to the audio file')
    parser.add_argument('--server', default='http://localhost:8000', help='ASR server URL')
    parser.add_argument('--concurrency', type=int, default=1, help='Number of concurrent requests')
    parser.add_argument('--repetitions', type=int, default=10, help='Number of test repetitions')
    parser.add_argument('--output', help='Output CSV file path')
    args = parser.parse_args()
    
    logger.info(f"Running performance test with {args.concurrency} concurrent requests, "
                f"{args.repetitions} repetitions, using {args.file}")
    
    # Verify file exists
    if not os.path.exists(args.file):
        logger.error(f"Audio file not found: {args.file}")
        return
    
    # Run the test
    results = await run_test(args.file, args.server, args.concurrency, args.repetitions)
    
    # Calculate summary
    summary = calculate_summary_stats(results)
    
    # Print summary
    print("\nTest Summary:")
    print(f"File: {args.file}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Repetitions: {args.repetitions}")
    print(f"Average Latency: {summary['avg_latency_ms']:.2f} ms")
    print(f"Min Latency: {summary['min_latency_ms']:.2f} ms")
    print(f"Max Latency: {summary['max_latency_ms']:.2f} ms")
    print(f"P95 Latency: {summary['p95_latency_ms']:.2f} ms")
    print(f"Success Rate: {summary['avg_success_rate'] * 100:.2f}%")
    print(f"Estimated Throughput: {summary['throughput_per_min']:.2f} requests/min")
    
    # Save results to CSV if output file is specified
    if args.output:
        # First, write detailed results for each repetition
        with open(args.output, 'w', newline='') as csvfile:
            fieldnames = [
                'repetition', 'concurrency', 'total_time_ms', 'min_latency_ms', 
                'max_latency_ms', 'avg_latency_ms', 'median_latency_ms', 
                'p95_latency_ms', 'stddev_latency_ms', 'success_rate', 'errors'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow(result)
            
            # Write a summary line as a JSON-like structure for easier parsing
            summary_data = {
                'repetition': 'summary',
                'concurrency': args.concurrency,
                'total_time_ms': summary.get('total_time_ms', 0),
                'min_latency_ms': summary['min_latency_ms'],
                'max_latency_ms': summary['max_latency_ms'],
                'avg_latency_ms': summary['avg_latency_ms'],
                'median_latency_ms': summary.get('median_latency_ms', 0),
                'p95_latency_ms': summary['p95_latency_ms'],
                'stddev_latency_ms': summary.get('stddev_latency_ms', 0),
                'success_rate': summary['avg_success_rate'],
                'errors': summary.get('errors', 0)
            }
            
            # Add an additional JSON-formatted summary line for easier parsing
            csvfile.write("\n# SUMMARY_JSON: ")
            csvfile.write(f"{{\"avg_latency_ms\":{summary['avg_latency_ms']:.2f},")
            csvfile.write(f"\"min_latency_ms\":{summary['min_latency_ms']:.2f},")
            csvfile.write(f"\"max_latency_ms\":{summary['max_latency_ms']:.2f},")
            csvfile.write(f"\"p95_latency_ms\":{summary['p95_latency_ms']:.2f},")
            csvfile.write(f"\"success_rate\":{summary['avg_success_rate']:.4f}}}")
            csvfile.write("\n")
            
            # Also write the row in CSV format
            writer.writerow(summary_data)
            
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main()) 