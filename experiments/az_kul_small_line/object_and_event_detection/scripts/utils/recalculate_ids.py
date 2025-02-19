# This is a simple helper script to recalculate IDs in a given CSV file

import argparse

import pandas as pd

# Set up argument parser
parser = argparse.ArgumentParser(
    description="Script to process a CSV file by adding/modifying an ID column and saving the output."
)

# Add arguments
parser.add_argument(
    "input_file",
    type=str,
    help="Path to the input CSV file."
)
parser.add_argument(
    "output_file",
    type=str,
    help="Path to save the processed CSV file."
)
parser.add_argument(
    "--help",
    action="help",
    help="Show this help message and exit."
)

# Parse arguments
args = parser.parse_args()

# Read input file
try:
    df = pd.read_csv(args.input_file)
    df['ID'] = range(0, len(df))
    df.to_csv(args.output_file, index=False)
    print(f"File processed successfully. Output saved to: {args.output_file}")
except Exception as e:
    print(f"Error: {e}")
