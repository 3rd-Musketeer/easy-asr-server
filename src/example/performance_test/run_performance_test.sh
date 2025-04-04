#!/bin/bash
# Performance Test Orchestration Script
#
# This script orchestrates the performance testing by iterating through
# combinations of audio durations, concurrency levels, and worker counts.
#
# Usage:
#   ./run_performance_test.sh [--host HOST] [--port PORT] [--output-dir DIR]
#
# Example:
#   ./run_performance_test.sh --host 0.0.0.0 --port 8000 --output-dir results/

set -e

# Default values
HOST="0.0.0.0"
PORT="8000"
OUTPUT_DIR="performance_results"
TEST_DATA_DIR="test_data"
SERVER_STARTUP_WAIT=5  # Initial wait time before first health check
MAX_RETRIES=30         # Maximum number of health check retries
RETRY_INTERVAL=5       # Seconds between health check retries

# Check for required tools
for tool in ffmpeg bc curl python; do
  if ! command -v $tool &> /dev/null; then
    echo "Error: $tool is required but not installed. Please install it and try again."
    exit 1
  fi
done

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Make sure directories exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEST_DATA_DIR"

# Create timestamped CSV file for all results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_CSV="${OUTPUT_DIR}/results_${TIMESTAMP}.csv"

# Create the CSV header
echo "timestamp,worker_count,audio_duration,concurrency,repetitions,avg_latency_ms,min_latency_ms,max_latency_ms,p95_latency_ms,success_rate,throughput_per_min" > "$RESULTS_CSV"

# Prepare test audio files
echo "Preparing test audio files..."
if [ ! -f "sample.wav" ]; then
  echo "Error: sample.wav file not found in the current directory"
  exit 1
fi
cp sample.wav "$TEST_DATA_DIR/sample_3s.wav"

echo "Creating 10-second sample..."
if ! ffmpeg -y -i "concat:sample.wav|sample.wav|sample.wav|sample.wav" -acodec copy "$TEST_DATA_DIR/sample_10s.wav" 2>/dev/null; then
  echo "Error: Failed to create 10-second sample"
  exit 1
fi

echo "Creating 25-second sample..."
# Need to create a temporary file list for ffmpeg
TEMP_FILE_LIST=$(mktemp)
for i in {1..9}; do
  echo "file '$(pwd)/sample.wav'" >> "$TEMP_FILE_LIST"
done
if ! ffmpeg -y -f concat -safe 0 -i "$TEMP_FILE_LIST" -c copy "$TEST_DATA_DIR/sample_25s.wav" 2>/dev/null; then
  echo "Error: Failed to create 25-second sample"
  rm "$TEMP_FILE_LIST"
  exit 1
fi
rm "$TEMP_FILE_LIST"

# Define test parameters
AUDIO_DURATIONS=("3s" "10s" "25s")
CONCURRENCY_LEVELS=(1 5 10)
WORKER_COUNTS=(2 4 6)
REPETITIONS=10

# Function to check if server is healthy
check_server_health() {
  # Try to connect to the health endpoint
  if curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:${PORT}/asr/health" | grep -q "200"; then
    return 0  # Server is healthy
  else
    return 1  # Server is not healthy
  fi
}

# Function to start the server and wait until it's ready
start_server() {
  local workers=$1
  echo "Starting ASR server with $workers workers..."
  easy-asr-server run --host "$HOST" --port "$PORT" --workers "$workers" --device auto > "${OUTPUT_DIR}/server_${workers}.log" 2>&1 &
  SERVER_PID=$!
  echo "Server PID: $SERVER_PID"
  
  # Check if server process is running
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Error: Server failed to start. Check ${OUTPUT_DIR}/server_${workers}.log"
    exit 1
  fi
  
  # Initial wait time
  echo "Waiting $SERVER_STARTUP_WAIT seconds before first health check..."
  sleep $SERVER_STARTUP_WAIT
  
  # Check server health with retries
  echo "Checking if server is ready..."
  local retries=0
  while ! check_server_health && [ $retries -lt $MAX_RETRIES ]; do
    retries=$((retries+1))
    echo "Server not ready yet (attempt $retries/$MAX_RETRIES). Waiting $RETRY_INTERVAL seconds..."
    sleep $RETRY_INTERVAL
  done
  
  if [ $retries -eq $MAX_RETRIES ]; then
    echo "Error: Server did not become healthy within the timeout period."
    echo "Check ${OUTPUT_DIR}/server_${workers}.log for errors."
    stop_server
    exit 1
  fi
  
  echo "Server is ready!"
}

# Function to stop the server
stop_server() {
  if [ -n "$SERVER_PID" ]; then
    echo "Stopping ASR server (PID: $SERVER_PID)..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null || true
    SERVER_PID=""
  fi
}

# Function to extract values from CSV and append to main results file
process_temp_results() {
  local temp_file=$1
  local worker_count=$2
  local audio_duration=$3
  local concurrency=$4
  
  # Debug output
  echo "Processing results from: $temp_file"
  
  # Try to extract from JSON summary if it exists
  if grep -q "# SUMMARY_JSON:" "$temp_file"; then
    # Extract the JSON line and parse values
    local json_line=$(grep "# SUMMARY_JSON:" "$temp_file" | sed 's/# SUMMARY_JSON: //')
    echo "Found JSON summary: $json_line"
    
    # Extract values using grep and sed
    local avg_lat=$(echo "$json_line" | grep -o '"avg_latency_ms":[^,}]*' | cut -d':' -f2)
    local min_lat=$(echo "$json_line" | grep -o '"min_latency_ms":[^,}]*' | cut -d':' -f2)
    local max_lat=$(echo "$json_line" | grep -o '"max_latency_ms":[^,}]*' | cut -d':' -f2)
    local p95_lat=$(echo "$json_line" | grep -o '"p95_latency_ms":[^,}]*' | cut -d':' -f2)
    local success_rate=$(echo "$json_line" | grep -o '"success_rate":[^,}]*' | cut -d':' -f2)
  else
    # Fall back to the old method if JSON not found
    echo "No JSON summary found, using CSV parsing"
    
    # Get the last line of the temporary CSV (summary stats)
    local last_line=$(tail -n 1 "$temp_file")
    echo "Last line: $last_line"
    
    # Extract avg_latency_ms directly with grep and cut (more reliable)
    local avg_lat=$(grep -o 'avg_latency_ms":[^,]*' "$temp_file" | tail -1 | cut -d':' -f2 | tr -d '", ')
    local min_lat=$(grep -o 'min_latency_ms":[^,]*' "$temp_file" | tail -1 | cut -d':' -f2 | tr -d '", ')
    local max_lat=$(grep -o 'max_latency_ms":[^,]*' "$temp_file" | tail -1 | cut -d':' -f2 | tr -d '", ')
    local p95_lat=$(grep -o 'p95_latency_ms":[^,]*' "$temp_file" | tail -1 | cut -d':' -f2 | tr -d '", ')
    local success_rate=$(grep -o 'success_rate":[^,]*' "$temp_file" | tail -1 | cut -d':' -f2 | tr -d '", ')
  fi
  
  # If values are empty or zero, use defaults
  [[ -z "$avg_lat" || "$avg_lat" == "0" ]] && avg_lat="1000"
  [[ -z "$min_lat" ]] && min_lat="0"
  [[ -z "$max_lat" ]] && max_lat="0"
  [[ -z "$p95_lat" ]] && p95_lat="0"
  [[ -z "$success_rate" ]] && success_rate="0"
  
  echo "Extracted values: avg=$avg_lat, min=$min_lat, max=$max_lat, p95=$p95_lat, success=$success_rate"
  
  # Calculate throughput (requests per minute) with error checking
  local throughput=0
  if [[ $(echo "$avg_lat > 0" | bc -l) -eq 1 ]]; then
    throughput=$(echo "scale=2; ($concurrency * 60 * 1000) / $avg_lat" | bc -l)
  fi
  
  # Add data to the main results CSV
  echo "${TIMESTAMP},${worker_count},${audio_duration},${concurrency},${REPETITIONS},${avg_lat},${min_lat},${max_lat},${p95_lat},${success_rate},${throughput}" >> "$RESULTS_CSV"
}

# Trap to ensure server is stopped if script is interrupted
trap stop_server EXIT

# Run tests for each worker count
for workers in "${WORKER_COUNTS[@]}"; do
  echo "========================================================="
  echo "Testing with $workers workers"
  echo "========================================================="
  
  # Start the server and wait until it's ready
  start_server $workers
  
  # Run tests for each audio duration and concurrency level
  for duration in "${AUDIO_DURATIONS[@]}"; do
    echo "---------------------------------------------------------"
    echo "Testing with ${duration} audio"
    echo "---------------------------------------------------------"
    
    for concurrency in "${CONCURRENCY_LEVELS[@]}"; do
      echo "Testing with concurrency level $concurrency..."
      
      # Create a temporary file for this test's results
      temp_results_file="${OUTPUT_DIR}/temp_results_w${workers}_${duration}_c${concurrency}.csv"
      
      # Run the test
      python src/example/performance_test/asr_request.py \
        --file "${TEST_DATA_DIR}/sample_${duration}.wav" \
        --server "http://${HOST}:${PORT}" \
        --concurrency $concurrency \
        --repetitions $REPETITIONS \
        --output "$temp_results_file"
      
      # Process the temporary results and add to main CSV
      process_temp_results "$temp_results_file" "$workers" "${duration//[^0-9]/}" "$concurrency"
      
      # Remove the temporary file
      rm "$temp_results_file"
    done
  done
  
  # Stop the server
  stop_server
  
  echo "Completed tests for $workers workers"
done

echo "========================================================="
echo "Performance testing completed!"
echo "Results are available in $RESULTS_CSV"
echo "========================================================="

# Run the visualization script if it exists
if [ -f "src/example/performance_test/visualize_results.py" ]; then
  echo "Generating visualization and report..."
  python src/example/performance_test/visualize_results.py --input "$RESULTS_CSV" --output-dir "${OUTPUT_DIR}/report"
else
  echo "Visualization script not found. Install the script to generate reports."
fi

echo "Done!" 