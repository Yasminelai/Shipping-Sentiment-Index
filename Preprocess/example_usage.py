#!/usr/bin/env python3
"""
Example usage of the cleaning.py script for maritime text preprocessing
"""

import pandas as pd
from cleaning import TextCleaner

def clean_maritime_data(input_file: str, output_file: str, text_column: str = 'text'):
    """
    Clean maritime text data from CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        text_column: Name of the text column to clean
    """
    # Load data
    df = pd.read_csv(input_file)
    
    # Initialize cleaner
    cleaner = TextCleaner()
    
    # Clean the text
    cleaned_df = cleaner.clean_dataframe(df, text_column)
    
    # Save cleaned data
    cleaned_df.to_csv(output_file, index=False)
    
    print(f"Cleaned data saved to {output_file}")
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")

if __name__ == "__main__":
    # Example usage
    clean_maritime_data("input.csv", "output_cleaned.csv", "text")
