#!/bin/bash
# Complete Performance Test Script
#
# This is a simplified script that demonstrates how to run the complete
# performance test workflow with data collection and visualization.
#
# Usage:
#   ./run_complete_test.sh [--output-dir DIR]

set -e

# Default values
OUTPUT_DIR="results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
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

# Make sure the output directory exists
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "ASR Server Performance Test"
echo "=================================================="
echo 
echo "This script will run the complete performance test workflow:"
echo "1. Run performance tests (collecting data)"
echo "2. Generate visualization and reports"
echo
echo "Output directory: $OUTPUT_DIR"
echo

# Step 1: Run performance tests
echo "Step 1: Running performance tests..."
./src/example/performance_test/run_performance_test.sh --output-dir "$OUTPUT_DIR"

# At this point, the run_performance_test.sh script should have already run the visualization
# automatically, but we can do it explicitly in case it was skipped for some reason

# Check if a results CSV file exists
RESULTS_CSV=$(find "$OUTPUT_DIR" -name "results_*.csv" -type f | sort -r | head -n 1)

if [ -n "$RESULTS_CSV" ]; then
  echo
  echo "Step 2: Generating visualization and reports..."
  python src/example/performance_test/visualize_results.py --input "$RESULTS_CSV" --output-dir "${OUTPUT_DIR}/report"
else
  echo
  echo "Error: No results CSV file found in $OUTPUT_DIR"
  exit 1
fi

echo
echo "=================================================="
echo "Performance test completed!"
echo "=================================================="
echo "Results are available in: $OUTPUT_DIR"
echo "Report is available in: ${OUTPUT_DIR}/report"
echo
echo "To view the report, see the markdown file in ${OUTPUT_DIR}/report"
echo "==================================================" 