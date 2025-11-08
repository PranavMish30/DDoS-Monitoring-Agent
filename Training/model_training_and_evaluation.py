# model_training_and_evaluation.py
#
# Implements the model training and evaluation steps using the windowed features.
# This script uses a time-sequential split and a Random Forest Classifier.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib

# --- Configuration ---
INPUT_FILE = r'C:\Users\prana\DDoS-Agent\Datasets\windowed_features_for_rf.csv'
MODEL_FILENAME = r'C:\Users\prana\DDoS-Agent\Models\random_forest_ddos_model.joblib'
TEST_SIZE_RATIO = 0.3  # Use the last 30% of time for testing

def train_and_evaluate_model():
    """
    Loads windowed features, performs time-sequential split, scales features,
    trains a Random Forest model, and evaluates its performance.
    """
    try:
        print(f"Loading windowed features from {INPUT_FILE}...")
        df = pd.read_csv(INPUT_FILE)

        # 1. Separate Features (X) and Target (Y)
        X = df.drop(columns=['Window_Label', 'Window_Start', 'Window_End'])
        y = df['Window_Label']
        feature_names = X.columns.tolist()

        # 2. Preprocessing Inconsistent Values (Step 3.1)
        print("Handling NaN/Inf values...")
        # Replace NaN with 0 (e.g., if Fwd_Packet_Len_Std was NaN due to single packet in window)
        X.replace([np.nan], 0, inplace=True) 
        # Replace Inf with a large number (if any division by zero resulted in Inf)
        # Using the max value of the respective column or 1e5 as a default ceiling
        X.replace([np.inf, -np.inf], X.max().max() if not X.empty else 1e5, inplace=True)


        # 3. Time-Sequential Splitting (Step 4.1)
        # Sort by Window_Start (already sorted, but good practice for sequential data)
        df_sorted = df.sort_values(by='Window_Start').reset_index(drop=True)
        
        # Determine split index based on ratio
        split_index = int(len(df_sorted) * (1 - TEST_SIZE_RATIO))
        
        # Split features and labels sequentially
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        print(f"Data Split: Train windows={len(X_train)}, Test windows={len(X_test)}")
        print(f"Training Time Range: {df_sorted['Window_Start'].iloc[0]} to {df_sorted['Window_Start'].iloc[split_index-1]}")
        print(f"Testing Time Range: {df_sorted['Window_Start'].iloc[split_index]} to {df_sorted['Window_Start'].iloc[-1]}")


        # 4. Feature Scaling (Step 3.2)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("Features scaled successfully.")

        # 5. Model Training (Step 4.2 - 4.4)
        print("Training Random Forest Classifier (n_estimators=100, class_weight='balanced')...")
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=15, # Limiting depth helps prevent overfitting
            class_weight='balanced', # Crucial for class imbalance
            n_jobs=-1 # Use all processors
        )
        rf_model.fit(X_train_scaled, y_train)
        print("Model training complete.")


        # 6. Prediction and Evaluation (Step 5.A)
        y_pred = rf_model.predict(X_test_scaled)

        print("\n--- Model Performance Metrics on Test Set ---")
        
        # Calculate individual metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy:  {accuracy:.4f} (Overall correctness)")
        print(f"Precision: {precision:.4f} (Minimizing False Alarms: critical for deployment)")
        print(f"Recall:    {recall:.4f} (Maximizing Detection: critical for security)")
        print(f"F1-Score:  {f1:.4f} (Balance between Precision and Recall)")

        # # Detailed classification report
        # print("\nClassification Report:")
        # print(classification_report(y_test, y_pred))

        # In model_training_and_evaluation.py, replace the print statements with this:

        # Detailed classification report
        print("\nClassification Report (0=Normal, 1=DDoS):")
        # Use to_string() or ensure the terminal output captures all lines
        report_lines = classification_report(y_test, y_pred, output_dict=False)
        print(report_lines)

        # ... (rest of the script remains the same)

        # 7. Feature Importance (Step 6.1)
        importances = rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names, 
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        print("\n--- Feature Importance ---")
        print(feature_importance_df.head(5).to_markdown(index=False))


        # # 8. Model Persistence (Step 6.3)
        # joblib.dump(rf_model, MODEL_FILENAME)
        # print(f"\nTrained Random Forest model saved to {MODEL_FILENAME}")
        # --- Update in model_training_and_evaluation.py (Step 8) ---
        # Previous: joblib.dump(rf_model, MODEL_FILENAME)
        # New: Save the model AND the scaler as a tuple.

        # 8. Model Persistence (Step 6.3)
        # Save the model and the scaler used for transformation
        joblib.dump((rf_model, scaler), MODEL_FILENAME)
        print(f"\nTrained Random Forest model and scaler saved to {MODEL_FILENAME}")

    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        print("Please ensure you have successfully run 'feature_aggregation.py' first.")
    except Exception as e:
        print(f"An unexpected error occurred during model training: {e}")

if __name__ == '__main__':
    train_and_evaluate_model()
