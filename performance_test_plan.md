# ASR Server Performance Test Plan

## Overview

This document outlines a simplified performance test plan for evaluating the `easy_asr_server` under various conditions. We will test and generate a table showing latency while varying:

1. **Audio Length**: Testing with different durations of audio input (up to 25s)
2. **Concurrency Level**: Testing with different numbers of simultaneous requests
3. **Worker Count**: Testing with 2, 4, and 6 workers

## Test Data Preparation

We will use the existing `sample.wav` file (2.84 seconds) as our base sample and create test files of varying lengths:

| File Name | Duration | Creation Method |
|-----------|----------|-----------------|
| sample_3s.wav | ~3s (original) | Already exists |
| sample_10s.wav | ~10s | Concatenate original sample ~3-4 times |
| sample_25s.wav | ~25s | Concatenate original sample ~9 times |

```bash
# Create the test audio files using ffmpeg
mkdir -p test_data

# Copy the original sample
cp sample.wav test_data/sample_3s.wav

# Create 10-second sample
ffmpeg -i "concat:sample.wav|sample.wav|sample.wav|sample.wav" -acodec copy test_data/sample_10s.wav

# Create 25-second sample
ffmpeg -f concat -safe 0 -i <(for i in {1..9}; do echo "file '$(pwd)/sample.wav'"; done) -c copy test_data/sample_25s.wav
```

## Test Matrix

We will measure the latency under the following combinations:

### Audio Duration × Concurrency × Workers

| Audio Duration | Concurrency Levels | Worker Counts |
|----------------|-------------------|---------------|
| 3s, 10s, 25s   | 1, 5, 10, 15, 20, 25, 30, 35 | 2, 4, 6 |

This gives us a total of 72 test scenarios (3 durations × 8 concurrency levels × 3 worker counts).

## Implementation Approach

We will implement the testing in two separate components:

### 1. Python Request Script (`asr_request.py`)

A Python script using `asyncio` and `aiohttp` that:
- Takes parameters for audio file, server URL, number of concurrent requests, and number of repetitions
- Sends the specified number of concurrent requests to the ASR server
- Measures and records the latency for each request
- Calculates statistics (average, min, max, etc.) and outputs results in CSV format

```python
# Usage example:
# python asr_request.py --file test_data/sample_10s.wav --server http://localhost:8000 --concurrency 10 --repetitions 10 --output results.csv
```

### 2. Shell Orchestration Script (`run_performance_test.sh`)

A shell script that:
- Takes care of starting/stopping the ASR server with different worker counts
- Iterates through all combinations of audio durations, concurrency levels, and worker counts
- Calls the Python request script with appropriate parameters for each test scenario
- Consolidates the results into a final summary table

```bash
# Usage example:
# ./run_performance_test.sh --host 0.0.0.0 --port 8000 --output-dir results/
```

## Expected Output

The final output will be a set of tables showing the average latency (in milliseconds) for each combination. Due to the increased number of concurrency levels, we'll organize the results by worker count and audio duration:

### Sample Result Table Format

#### 2 Workers

| Audio Duration | 1 | 5 | 10 | 15 | 20 | 25 | 30 | 35 |
|----------------|---|---|----|----|----|----|----|----|
| 3s             |   |   |    |    |    |    |    |    |
| 10s            |   |   |    |    |    |    |    |    |
| 25s            |   |   |    |    |    |    |    |    |

#### 4 Workers

| Audio Duration | 1 | 5 | 10 | 15 | 20 | 25 | 30 | 35 |
|----------------|---|---|----|----|----|----|----|----|
| 3s             |   |   |    |    |    |    |    |    |
| 10s            |   |   |    |    |    |    |    |    |
| 25s            |   |   |    |    |    |    |    |    |

#### 6 Workers

| Audio Duration | 1 | 5 | 10 | 15 | 20 | 25 | 30 | 35 |
|----------------|---|---|----|----|----|----|----|----|
| 3s             |   |   |    |    |    |    |    |    |
| 10s            |   |   |    |    |    |    |    |    |
| 25s            |   |   |    |    |    |    |    |    |

*Values in the tables represent average latency in milliseconds.*

## Test Execution Flow

The test execution will follow this flow:

1. Create test audio files
2. For each worker count (2, 4, 6):
   a. Start the ASR server with the specified number of workers
   b. For each audio duration (3s, 10s, 25s):
      i. For each concurrency level (1, 5, 10, 15, 20, 25, 30, 35):
         - Run the Python request script to send concurrent requests
         - Record results
   c. Stop the ASR server
3. Generate final summary tables

Each test scenario will be run 10 times to minimize the impact of outliers, and we'll report the average latency.

## Conclusion

This test plan will measure how the ASR server's latency is affected by audio duration, concurrent requests, and worker count. The results will help determine the optimal configuration for different usage scenarios. 