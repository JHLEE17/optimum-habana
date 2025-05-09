#!/usr/bin/env python3

import sys
import csv
import os

def convert_log_to_csv(input_file, output_file=None):
    # If output file is not specified, use input filename with .csv extension
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.csv"
    
    # Read input file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Extract values (True/False) and convert to O/X
    values = []
    for line in lines:
        if "can_use_cache: True" in line:
            values.append("X")
        elif "can_use_cache: False" in line:
            values.append("O")
    
    # Calculate number of rows needed for 28 columns
    num_cols = 28
    num_rows = (len(values) + num_cols - 1) // num_cols
    
    # Fill any remaining cells with empty strings
    while len(values) < num_rows * num_cols:
        values.append("")
    
    # Create rows for CSV
    rows = []
    for i in range(num_rows):
        start_idx = i * num_cols
        end_idx = start_idx + num_cols
        rows.append(values[start_idx:end_idx])
    
    # Write to output file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"Conversion complete. Output saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_log_to_csv.py <input_log_file> [output_csv_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    # Check if output file is provided
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_log_to_csv(input_file, output_file) 