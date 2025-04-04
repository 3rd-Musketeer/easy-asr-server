# ASR Server Performance Testing

This directory contains scripts for testing the performance of the ASR server under various conditions. The tests measure latency while varying audio length, concurrency level, and worker count.

## Design Overview

The performance testing system follows a two-step process:

1. **Data Collection**: Run tests with various combinations of parameters and collect all results in a single timestamped CSV file
2. **Visualization**: Process the CSV data to generate visualizations and a markdown report

This separation keeps the code simple and modular, allowing for better flexibility in analysis and reporting.

## Prerequisites

Before running the performance tests, ensure you have the required dependencies:

```bash
# Install testing dependencies
uv pip install aiohttp numpy rich pandas matplotlib
```

You also need `ffmpeg` installed for audio file preparation:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use Chocolatey: `choco install ffmpeg`

## Scripts Overview

### 1. Data Collection (`run_performance_test.sh`)

This script orchestrates the tests with various parameter combinations:

```bash
# Usage
./run_performance_test.sh [--host HOST] [--port PORT] [--output-dir DIR]

# Example
./run_performance_test.sh --host 0.0.0.0 --port 8000 --output-dir results/
```

The script:
1. Creates test audio files of different lengths
2. Starts and stops the ASR server with different worker counts
3. Runs tests for all combinations of audio length and concurrency
4. Records all data in a single timestamped CSV file (e.g., `results_20250405_120000.csv`)

### 2. Visualization (`visualize_results.py`)

This script processes the test results and generates visualizations:

```bash
# Usage
python visualize_results.py --input RESULTS_CSV [--output-dir OUTPUT_DIR]

# Example
python visualize_results.py --input results/results_20250405_120000.csv --output-dir results/report
```

The script:
1. Reads the CSV data
2. Generates tables using the 'rich' library for console output
3. Creates a markdown report with tables and charts
4. Optionally generates graphs for deeper analysis

## Test Matrix

The tests are run with the following parameter combinations:

- **Audio Durations**: 3s, 10s, 25s
- **Concurrency Levels**: 1, 5, 10, 15, 20, 25, 30, 35
- **Worker Counts**: 2, 4, 6

This gives a total of 72 test scenarios (3 durations × 8 concurrency levels × 3 worker counts).

## Running the Tests

The complete testing workflow:

```bash
# Step 1: Run the performance tests
./run_performance_test.sh --output-dir results

# Step 2: Generate visualizations and report
python visualize_results.py --input results/results_*.csv --output-dir results/report
```

## Data Format

The CSV file contains all test data with the following columns:

- `timestamp`: When the test was run
- `worker_count`: Number of server workers
- `audio_duration`: Duration of the test audio in seconds
- `concurrency`: Number of concurrent requests
- `avg_latency_ms`: Average latency in milliseconds
- `min_latency_ms`: Minimum latency in milliseconds
- `max_latency_ms`: Maximum latency in milliseconds
- `p95_latency_ms`: 95th percentile latency
- `success_rate`: Percentage of successful requests
- `throughput_per_min`: Estimated throughput in requests per minute
- Additional metrics as needed

## Results

The visualization script generates:

1. **Console Output**: Tables displaying the results using the rich library
2. **Markdown Report**: A comprehensive report with tables and analysis
3. **Charts (Optional)**: Visual representations of the performance data

Example table in the markdown report:

### 2 Workers

| Audio Duration | 1 | 5 | 10 | 15 | 20 | 25 | 30 | 35 |
|----------------|---|---|----|----|----|----|----|----|
| 3s             | 245 | 251 | 278 | 305 | 367 | 428 | 499 | 558 |
| 10s            | 752 | 791 | 847 | 978 | 1150 | 1352 | 1587 | 1812 |
| 25s            | 1840 | 1912 | 2154 | 2498 | 2894 | 3405 | 3988 | 4431 |

*Values represent average latency in milliseconds.*

## Customizing the Tests

To modify the test parameters, edit the corresponding arrays in `run_performance_test.sh`:

```bash
# Define test parameters
AUDIO_DURATIONS=("3s" "10s" "25s")
CONCURRENCY_LEVELS=(1 5 10 15 20 25 30 35)
WORKER_COUNTS=(2 4 6)
REPETITIONS=10
```

## Troubleshooting

- If the server fails to start, check the server log files in the output directory
- Increase `SERVER_STARTUP_WAIT` in the script if the server needs more time to initialize
- For high concurrency levels, you might need to increase system limits (e.g., file descriptors) 