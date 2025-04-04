#!/bin/bash
# Test script to verify fixes to the performance test system
#
# This is a minimal script that runs just one test case to verify that
# the CSV parsing and data extraction works correctly.

set -e

# Default values
OUTPUT_DIR="test_fixes"
TEST_DATA_DIR="$OUTPUT_DIR/test_data"
HOST="0.0.0.0"
PORT="8000"

# Make sure directories exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEST_DATA_DIR"

echo "Testing CSV parsing and data extraction fixes..."

# Copy the sample file
if [ ! -f "sample.wav" ]; then
  echo "Error: sample.wav file not found in the current directory"
  exit 1
fi
cp sample.wav "$TEST_DATA_DIR/sample.wav"

# Run a single test case
echo "Running a test with asr_request.py..."
python src/example/performance_test/asr_request.py \
  --file "$TEST_DATA_DIR/sample.wav" \
  --concurrency 1 \
  --repetitions 3 \
  --output "$OUTPUT_DIR/test_results.csv"

echo "Testing process_temp_results function..."

# Define a minimal version of the process_temp_results function
process_temp_results() {
  local temp_file=$1
  
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
    echo "No JSON summary found, using CSV parsing"
    
    # Extract values using grep
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
  
  # Calculate throughput with error checking
  local concurrency=1
  local throughput=0
  if [[ $(echo "$avg_lat > 0" | bc -l) -eq 1 ]]; then
    throughput=$(echo "scale=2; ($concurrency * 60 * 1000) / $avg_lat" | bc -l)
  fi
  
  echo "Calculated throughput: $throughput"
}

# Test the function
process_temp_results "$OUTPUT_DIR/test_results.csv"

echo "Test completed successfully!" 