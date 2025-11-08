# feature_aggregation.py
#
# Implements a sliding, overlapping time window to aggregate packet-level features
# into coarse-grained, time-series features for DDoS anomaly detection.
#
# UPDATED to use flag count columns (e.g., 'SYN_Flag_Count') instead of a single 'Flag' string column.

import pandas as pd
import numpy as np
from math import log2

# --- Configuration ---
# Define the size of the time window for aggregation (e.g., 5 seconds)
WINDOW_SIZE_SECONDS = 5
# Define the step size, controlling the overlap (e.g., 1 second means 4s overlap)
STEP_SIZE_SECONDS = 1
INPUT_FILE = r'C:\Users\prana\DDoS-Agent\Datasets\cleaned_data_for_windowing.csv'
OUTPUT_FILE = r'C:\Users\prana\DDoS-Agent\Datasets\windowed_features_for_rf.csv'

def calculate_entropy(data, feature_col):
    """
    Calculates the Shannon Entropy of a specific feature column within the window.
    High entropy indicates high diversity (e.g., many Source IPs).
    """
    if data.empty:
        return 0.0

    # Count occurrences of each unique value
    counts = data[feature_col].value_counts()
    
    # Calculate probabilities
    probabilities = counts / len(data)
    
    # Calculate Shannon Entropy: -sum(p * log2(p))
    # Add epsilon for log(0) safety
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9)) 
    return entropy

def aggregate_window_features(window_df):
    """
    Calculates all required aggregate features for a single time window.
    Uses the numeric flag count columns provided in the cleaned dataset.
    """
    # 1. Basic Counts and Rates
    packet_count = len(window_df)
    
    # Time window duration in seconds (should be close to WINDOW_SIZE_SECONDS)
    duration = (window_df['Timestamp'].max() - window_df['Timestamp'].min()).total_seconds()
    
    # Use a safe duration to avoid division by zero or errors from identical timestamps
    safe_duration = duration if duration > 0 else STEP_SIZE_SECONDS 
    
    packet_rate = packet_count / safe_duration
    
    # Total SYN packets in the window (sum of the count column)
    syn_count = window_df['SYN_Flag_Count'].sum()
    syn_rate = syn_count / safe_duration

    # 2. Entropy Features (Using actual column names: Source_IP, Destination_Port)
    ip_entropy = calculate_entropy(window_df, 'Source_IP')
    port_entropy = calculate_entropy(window_df, 'Destination_Port')

    # 3. Ratio Features
    # Avoid division by zero
    syn_ratio = syn_count / (packet_count + 1e-9)

    # 4. Statistical Flag Features
    # The mean of these counts indicates the average number of flags set per packet in the window
    fin_mean = window_df['FIN_Flag_Count'].mean()
    ack_mean = window_df['ACK_Flag_Count'].mean()
    rst_mean = window_df['RST_Flag_Count'].mean()

    # 5. Statistical Length Features (Using Fwd_Packet_Length_Mean as the aggregate)
    fwd_len_mean = window_df['Fwd_Packet_Length_Mean'].mean()
    fwd_len_std = window_df['Fwd_Packet_Length_Mean'].std() if len(window_df) > 1 else 0


    # 6. Window Label (Target Variable)
    # If ANY packet in the window is labeled as DDoS (1), the whole window is labeled DDoS.
    window_label = 1 if window_df['Is_DDoS'].any() else 0
    
    # Return the aggregated feature vector
    return {
        'Packet_Count': packet_count,
        'Packet_Rate': packet_rate,
        'Syn_Rate': syn_rate,
        'Syn_Ratio': syn_ratio,
        'IP_Entropy': ip_entropy,
        'Port_Entropy': port_entropy,
        'FIN_Flag_Mean': fin_mean,
        'ACK_Flag_Mean': ack_mean,
        'RST_Flag_Mean': rst_mean,
        'Fwd_Packet_Len_Mean': fwd_len_mean,
        'Fwd_Packet_Len_Std': fwd_len_std,
        'Window_Label': window_label # Our target variable (Y)
    }


def perform_windowing_and_aggregation():
    """
    Main function to read data, apply sliding window, and save the results.
    """
    try:
        # Load the cleaned dataset
        print(f"Loading cleaned data from {INPUT_FILE}...")
        df = pd.read_csv(INPUT_FILE)

        # Ensure Timestamp is in datetime format
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Sort by timestamp to ensure proper windowing
        df = df.sort_values(by='Timestamp').reset_index(drop=True)
        
        # Define the start and end of the entire dataset time range
        start_time = df['Timestamp'].min()
        end_time = df['Timestamp'].max()
        
        print(f"Data Time Range: {start_time} to {end_time}")
        
        aggregated_data = []
        current_start_time = start_time
        
        # Determine the time duration for steps and windows
        window_delta = pd.Timedelta(seconds=WINDOW_SIZE_SECONDS)
        step_delta = pd.Timedelta(seconds=STEP_SIZE_SECONDS)
        
        # --- Sliding Window Loop ---
        while current_start_time + window_delta <= end_time:
            window_end_time = current_start_time + window_delta
            
            # Filter the DataFrame for the current time window
            window_df = df[(df['Timestamp'] >= current_start_time) & (df['Timestamp'] < window_end_time)]
            
            # Only process windows that contain data
            if not window_df.empty:
                # Calculate the features for this window
                features = aggregate_window_features(window_df)
                
                # Add window metadata (optional, but useful for debugging)
                features['Window_Start'] = current_start_time
                features['Window_End'] = window_end_time
                
                aggregated_data.append(features)
                
            # Move the window forward by the step size (creating the overlap)
            current_start_time += step_delta
            
            # Simple progress reporting
            if (current_start_time - start_time).total_seconds() % 600 < step_delta.total_seconds():
                 print(f"Processing... {round((current_start_time - start_time).total_seconds() / (end_time - start_time).total_seconds() * 100, 2)}% complete")

        # Convert the list of feature dictionaries into the final DataFrame
        features_df = pd.DataFrame(aggregated_data)

        # Save the final feature set
        features_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n--- Aggregation Complete ---")
        print(f"Total windows generated: {len(features_df)}")
        print(f"Features saved to: {OUTPUT_FILE}")
        print(f"Distribution of Labels (0=Normal, 1=DDoS):")
        print(features_df['Window_Label'].value_counts())

    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        print("Please ensure you have run the 'data_acquisition_and_cleaning.py' script first.")
    except Exception as e:
        print(f"An unexpected error occurred during windowing: {e}")

if __name__ == '__main__':
    perform_windowing_and_aggregation()
