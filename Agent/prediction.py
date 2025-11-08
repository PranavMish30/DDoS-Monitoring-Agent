# model_prediction.py
#
# Loads a trained Random Forest model and scaler, makes predictions
# on a test dataset, and evaluates the results.

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# --- Configuration (Update these paths as necessary) ---
# Path to the data file (can be your original test set or a new one)
INPUT_FILE = r'C:\Users\prana\DDoS-Agent\Packets\aggregated-features.csv' 
# Assuming you have a separate file for final testing, or use the original one
MODEL_FILENAME = r'C:\Users\prana\DDoS-Agent\Models\random_forest_ddos_model.joblib'

def load_model_and_predict():
    """
    Loads the saved model and scaler, applies them to a test dataset,
    makes predictions, and prints the performance metrics.
    """
    if not os.path.exists(MODEL_FILENAME):
        print(f"Error: Model file '{MODEL_FILENAME}' not found.")
        print("Please ensure the training script has been run successfully.")
        return

    try:
        # 1. Load the model and scaler
        print(f"Loading model and scaler from {MODEL_FILENAME}...")
        # The model and scaler were saved as a tuple (rf_model, scaler)
        rf_model, scaler = joblib.load(MODEL_FILENAME)
        print("Model and scaler loaded successfully.")

        # 2. Load Test Data
        print(f"Loading test features from {INPUT_FILE}...")
        df_test = pd.read_csv(INPUT_FILE)
        
        # Separate Features (X) and Target (Y)
        X_test = df_test.drop(columns=['Window_Start', 'Window_End'], errors='ignore')
        # y_test = df_test['Window_Label']

        # 3. Preprocessing (Consistent with training script)
        print("Handling NaN/Inf values (matching training preprocessing)...")
        
        # Ensure 'errors=ignore' in drop() above if these columns don't exist
        
        # Replace NaN with 0
        X_test.replace([np.nan], 0, inplace=True) 
        
        # Replace Inf with a large number (using 1e5 as a default ceiling)
        # We can't use X.max().max() here as we only have the test set,
        # so we use a safe, large default. The scaler should handle clipping.
        X_test.replace([np.inf, -np.inf], 1e5, inplace=True) 

        # 4. Feature Scaling (Crucial: Use the loaded 'scaler.transform')
        print("Scaling features using the loaded scaler...")
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Prediction dataset size: {len(X_test_scaled)} windows.")

        # 5. Make Predictions
        print("Making predictions...")
        y_pred = rf_model.predict(X_test_scaled)
        
        # 6. Evaluation
        print("\n--- Model Performance Metrics on Prediction Set ---")
        
        # # Calculate individual metrics
        # precision = precision_score(y_test, y_pred, zero_division=0)
        # recall = recall_score(y_test, y_pred, zero_division=0)
        # f1 = f1_score(y_test, y_pred, zero_division=0)
        # accuracy = accuracy_score(y_test, y_pred)

        # print(f"Accuracy:  {accuracy:.4f}")
        # print(f"Precision: {precision:.4f}")
        # print(f"Recall:    {recall:.4f}")
        # print(f"F1-Score:  {f1:.4f}")

        # Detailed classification report
        # print("\nClassification Report (0=Normal, 1=DDoS):")
        # report_lines = classification_report(y_test, y_pred, output_dict=False, zero_division=0)
        # print(report_lines)
        print(len(y_pred),y_pred)
        print("\nPrediction complete.")

    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")

if __name__ == '__main__':
    load_model_and_predict()