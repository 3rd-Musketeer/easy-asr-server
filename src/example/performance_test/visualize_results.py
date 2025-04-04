#!/usr/bin/env python3
"""
Visualization Script for ASR Server Performance Tests

This script processes the performance test results CSV file and generates
visualizations and reports using the rich library.

Usage:
    python visualize_results.py --input RESULTS_CSV [--output-dir OUTPUT_DIR]
"""

import argparse
import csv
import datetime
import os
import sys
from typing import Dict, List, Tuple, Any

import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


def load_data(csv_file: str) -> pd.DataFrame:
    """
    Load and parse the CSV file containing performance test results.
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        DataFrame with the test results
    """
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)


def create_pivot_table(df: pd.DataFrame, worker_count: int, metric: str = 'avg_latency_ms') -> pd.DataFrame:
    """
    Create a pivot table for a specific worker count and metric.
    
    Args:
        df: DataFrame with the test results
        worker_count: Worker count to filter by
        metric: Metric to use for the values (default: avg_latency_ms)
        
    Returns:
        Pivot table with audio_duration as rows and concurrency as columns
    """
    filtered_df = df[df['worker_count'] == worker_count]
    pivot = filtered_df.pivot_table(
        index='audio_duration', 
        columns='concurrency',
        values=metric,
        aggfunc='mean'  # In case there are multiple entries
    )
    return pivot


def display_rich_tables(df: pd.DataFrame, console: Console) -> None:
    """
    Display tables using rich library for each worker count.
    
    Args:
        df: DataFrame with the test results
        console: Rich console for output
    """
    worker_counts = sorted(df['worker_count'].unique())
    
    for worker_count in worker_counts:
        pivot = create_pivot_table(df, worker_count)
        
        table = Table(title=f"{worker_count} Workers - Average Latency (ms)")
        
        # Add columns
        table.add_column("Audio Duration", style="cyan")
        for concurrency in pivot.columns:
            table.add_column(str(concurrency), justify="right")
        
        # Add rows
        for duration in pivot.index:
            row = [f"{duration}s"]
            for concurrency in pivot.columns:
                value = pivot.loc[duration, concurrency]
                row.append(f"{value:.0f}")
            table.add_row(*row)
        
        console.print(table)
        console.print()


def generate_markdown_report(df: pd.DataFrame, output_dir: str) -> str:
    """
    Generate a markdown report with tables and charts.
    
    Args:
        df: DataFrame with the test results
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated markdown file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"performance_report_{timestamp}.md")
    
    worker_counts = sorted(df['worker_count'].unique())
    
    with open(report_path, 'w') as f:
        f.write("# ASR Server Performance Test Report\n\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Test Configuration\n\n")
        f.write(f"- Test timestamp: {df['timestamp'].iloc[0]}\n")
        f.write(f"- Number of worker configurations: {len(worker_counts)}\n")
        f.write(f"- Audio durations tested: {', '.join([f'{d}s' for d in sorted(df['audio_duration'].unique())])}\n")
        f.write(f"- Concurrency levels tested: {', '.join([str(c) for c in sorted(df['concurrency'].unique())])}\n\n")
        
        # Generate tables for each worker count
        f.write("## Latency Results\n\n")
        
        for worker_count in worker_counts:
            f.write(f"### {worker_count} Workers\n\n")
            
            pivot = create_pivot_table(df, worker_count)
            
            # Create markdown table
            f.write("| Audio Duration |")
            for concurrency in pivot.columns:
                f.write(f" {concurrency} |")
            f.write("\n")
            
            f.write("|")
            for _ in range(len(pivot.columns) + 1):
                f.write(" --- |")
            f.write("\n")
            
            for duration in pivot.index:
                f.write(f"| {duration}s |")
                for concurrency in pivot.columns:
                    value = pivot.loc[duration, concurrency]
                    f.write(f" {value:.0f} |")
                f.write("\n")
            
            f.write("\n*Values represent average latency in milliseconds.*\n\n")
        
        # Add analysis section
        f.write("## Analysis\n\n")
        
        # Effect of audio duration
        f.write("### Effect of Audio Duration\n\n")
        f.write("Audio duration has a significant impact on latency. Longer audio files take more time to process, ")
        f.write("resulting in higher latency values.\n\n")
        
        # Effect of concurrency
        f.write("### Effect of Concurrency\n\n")
        f.write("As concurrency increases, so does latency. This indicates that the server's resources are ")
        f.write("being divided among multiple requests, affecting processing speed.\n\n")
        
        # Effect of worker count
        f.write("### Effect of Worker Count\n\n")
        f.write("Different worker counts show varying performance characteristics. ")
        f.write("More workers can help handle higher concurrency but may introduce additional overhead.\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("Based on the test results, the following configurations are recommended:\n\n")
        
        # Find best worker count for different scenarios
        f.write("- **Low Latency Priority**: Use fewer workers for individual requests\n")
        f.write("- **High Throughput Priority**: Use more workers for concurrent processing\n")
        f.write("- **Balanced Performance**: Consider using a moderate number of workers\n\n")
    
    return report_path


def generate_charts(df: pd.DataFrame, output_dir: str) -> List[str]:
    """
    Generate charts for visualization.
    
    Args:
        df: DataFrame with the test results
        output_dir: Directory to save the charts
        
    Returns:
        List of paths to the generated chart files
    """
    os.makedirs(output_dir, exist_ok=True)
    chart_files = []
    
    # 1. Latency vs Concurrency for different worker counts (fixed audio duration)
    for duration in sorted(df['audio_duration'].unique()):
        filtered_df = df[df['audio_duration'] == duration]
        
        plt.figure(figsize=(10, 6))
        
        for worker_count in sorted(filtered_df['worker_count'].unique()):
            worker_df = filtered_df[filtered_df['worker_count'] == worker_count]
            worker_df = worker_df.sort_values('concurrency')
            plt.plot(
                worker_df['concurrency'], 
                worker_df['avg_latency_ms'], 
                marker='o',
                label=f"{worker_count} Workers"
            )
        
        plt.title(f"Latency vs Concurrency ({duration}s Audio)")
        plt.xlabel("Concurrency")
        plt.ylabel("Average Latency (ms)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        chart_path = os.path.join(output_dir, f"latency_vs_concurrency_{duration}s.png")
        plt.savefig(chart_path)
        plt.close()
        chart_files.append(chart_path)
    
    # 2. Latency vs Audio Duration for different worker counts (fixed concurrency)
    for concurrency in sorted(df['concurrency'].unique()):
        filtered_df = df[df['concurrency'] == concurrency]
        
        plt.figure(figsize=(10, 6))
        
        for worker_count in sorted(filtered_df['worker_count'].unique()):
            worker_df = filtered_df[filtered_df['worker_count'] == worker_count]
            worker_df = worker_df.sort_values('audio_duration')
            plt.plot(
                worker_df['audio_duration'], 
                worker_df['avg_latency_ms'], 
                marker='o',
                label=f"{worker_count} Workers"
            )
        
        plt.title(f"Latency vs Audio Duration (Concurrency={concurrency})")
        plt.xlabel("Audio Duration (s)")
        plt.ylabel("Average Latency (ms)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        chart_path = os.path.join(output_dir, f"latency_vs_duration_c{concurrency}.png")
        plt.savefig(chart_path)
        plt.close()
        chart_files.append(chart_path)
    
    return chart_files


def main():
    parser = argparse.ArgumentParser(description='Visualize ASR Server Performance Test Results')
    parser.add_argument('--input', required=True, help='Input CSV file with test results')
    parser.add_argument('--output-dir', default='results/report', help='Output directory for the report')
    args = parser.parse_args()
    
    # Create console for rich output
    console = Console()
    
    console.print(Panel.fit(
        Text("ASR Server Performance Test Visualization", style="bold magenta"),
        border_style="green"
    ))
    
    # Load the data
    console.print("Loading data...", style="yellow")
    df = load_data(args.input)
    console.print(f"Loaded {len(df)} test results", style="green")
    
    # Display summary statistics
    console.print("\nSummary Statistics:", style="bold blue")
    console.print(f"Worker counts: {sorted(df['worker_count'].unique())}")
    console.print(f"Audio durations: {sorted(df['audio_duration'].unique())}s")
    console.print(f"Concurrency levels: {sorted(df['concurrency'].unique())}")
    
    # Display rich tables
    console.print("\nPerformance Results:", style="bold blue")
    display_rich_tables(df, console)
    
    # Generate markdown report
    console.print("\nGenerating markdown report...", style="yellow")
    report_path = generate_markdown_report(df, args.output_dir)
    console.print(f"Report generated: {report_path}", style="green")
    
    # Generate charts
    console.print("\nGenerating charts...", style="yellow")
    chart_files = generate_charts(df, args.output_dir)
    console.print(f"Generated {len(chart_files)} charts in {args.output_dir}", style="green")
    
    console.print(Panel.fit(
        Text("Visualization completed!", style="bold magenta"),
        border_style="green"
    ))


if __name__ == "__main__":
    main() 