import pandas as pd
import numpy as np
import sys
import os

# --- Configuration ---
# NOTE: Replace 'syn.csv' with the actual path to your data file.
DATA_FILE_PATH = r'C:\Users\prana\DDoS-Agent\Datasets\Syn.csv'
OUTPUT_FILE_PATH = 'cleaned_data_for_windowing.csv'

# Define the set of columns relevant for SYN flood analysis and feature creation.
# These names are based on the common feature set in the CICDDoS2019 dataset.
RELEVANT_COLUMNS = [
    # Identifier/Timestamp
    'Timestamp',
    'Flow ID', 
    'Source IP',
    'Destination IP',
    'Source Port',
    'Destination Port',
    
    # TCP Flags (Crucial for SYN analysis)
    'FIN Flag Count',
    'SYN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'URG Flag Count',
    'ECE Flag Count',
    
    # Packet/Flow Statistics
    'Protocol', # Should be TCP (6) for SYN, but included for completeness
    'Fwd Packet Length Max', 
    'Fwd Packet Length Mean', 
    'Fwd Packet Length Std',
    'Total Fwd Packets',
    'Total Backward Packets',
    
    # Target Label
    'Label'
]


def load_and_clean_data(file_path):
    """
    Loads the raw data, selects relevant columns, handles missing/infinite values,
    and prepares the binary target variable.
    """
    print(f"Loading data from: {file_path}")
    try:
        # Load the CSV. Using 'low_memory=False' often helps with large datasets.
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please update DATA_FILE_PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        sys.exit(1)

    print(f"Initial shape: {df.shape}")

    # 1. Select Relevant Columns
    # Filter out columns that are not in the raw data to prevent errors
    cols_to_use = [col for col in RELEVANT_COLUMNS if col in df.columns]
    
    # Ensure the 'Label' column is present before proceeding
    if 'Label' not in cols_to_use:
        print("Error: 'Label' column not found in the dataset. Cannot proceed.")
        sys.exit(1)

    df = df[cols_to_use]
    print(f"Shape after column selection: {df.shape}")
    
    # 2. Convert Timestamps to datetime objects
    if 'Timestamp' in df.columns:
        # Handle potential date parsing errors by coercing to NaT
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        # Drop rows where Timestamp could not be parsed
        df.dropna(subset=['Timestamp'], inplace=True)
        print("Timestamp column converted to datetime.")
    
    # 3. Handle Missing Values and Inconsistent Values
    # Replace infinite values (common after division operations in CICIDS) with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Simple Imputation: Replace remaining NaN values with 0. 
    # For numeric features, 0 is often a reasonable placeholder for 'no activity' or 'missing value'.
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    print("Handled infinite and NaN values (imputed with 0).")
    
    # 4. Target Variable Preparation (Categorical to Numeric)
    
    # Define the attack label for SYN flood specifically (can vary slightly by dataset version)
    ATTACK_LABEL = 'Syn' 
    
    # Create the binary target feature 'Is_DDoS'
    # Any row with the attack label gets 1, all others (BENIGN) get 0.
    # Note: If the Label column contains BENIGN and a specific SYN attack type, this works.
    df['Is_DDoS'] = df['Label'].apply(lambda x: 1 if isinstance(x, str) and ATTACK_LABEL.lower() in x.lower() else 0)
    
    # Check for class balance after cleaning
    label_counts = df['Is_DDoS'].value_counts()
    print("\nTarget Class Distribution ('Is_DDoS'):")
    print(label_counts)
    
    # Drop the original categorical 'Label' column
    df.drop('Label', axis=1, inplace=True)
    
    print("\nData acquisition and initial cleaning complete.")
    return df


if __name__ == '__main__':
    # --- IMPORTANT NOTE ---
    # The actual CICDDoS2019 dataset is very large. For the sake of demonstration,
    # if you cannot run the full file, consider sampling a subset after loading
    # (e.g., df = df.sample(frac=0.1, random_state=42).reset_index(drop=True))

    processed_df = load_and_clean_data(DATA_FILE_PATH)
    
    # Convert all column names to strings and strip whitespace/special characters
    # (Important for compatibility with certain machine learning frameworks)
    processed_df.columns = processed_df.columns.astype(str).str.strip().str.replace('[^A-Za-z0-9_]+', '_', regex=True)

    # Save the cleaned, labeled data for the next windowing step
    processed_df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"\nCleaned data saved to: {OUTPUT_FILE_PATH}")
    print(f"Final data ready for windowing. Shape: {processed_df.shape}")
