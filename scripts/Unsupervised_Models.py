import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score

# Configuration
OUTPUT_PATH = '../outputs/'
DATA_PATH = '../outputs/' 
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load Data 
def load_featured_data():
    
    print("Loading feature-engineered data...")
    file_path = os.path.join(DATA_PATH, "feature_engineered_data.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file {file_path} was not found. Please run the feature engineering script first.")
    
    df = pd.read_csv(file_path, parse_dates=["timestamp", "account_open_date"])
    print("Data loaded successfully.")
    return df

# Data Splitting & Preparation
def temporal_split(df, train_ratio=0.8):
    
    print(f"Performing temporal split with {train_ratio*100}% training data...")
    df_sorted = df.sort_values(by='timestamp')
    split_point = int(len(df_sorted) * train_ratio)
    
    df_train = df_sorted.iloc[:split_point]
    df_test = df_sorted.iloc[split_point:]
    
    print(f"Training set size: {len(df_train)}")
    print(f"Test set size: {len(df_test)}")
    return df_train, df_test

def prepare_data(df_train, df_test):
    
    print("Preparing data for Unsupervised models...")
    
    TARGET = 'is_suspicious'
    cols_to_drop = [
        'transaction_id', 'customer_id', 'account_id', 'sender_account', 'receiver_account',
        'timestamp', 'account_open_date', 'counterparty_account_open_date',
        'full_name', 'dob', 'national_id', 'phone', 'email', 'address', 'employer',
        'blacklist_reason', 'matched_rules', 'risk_score', 'country'
    ]
    
    X_train = df_train.drop(columns=cols_to_drop + [TARGET])
    y_train = df_train[TARGET].fillna(0)
    X_test = df_test.drop(columns=cols_to_drop + [TARGET])
    y_test = df_test[TARGET].fillna(0)

    # Encode categorical features
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
    for col in categorical_features:
        le = LabelEncoder()
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)
        le.fit(X_train[col])
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.fillna(-999))
    X_test = scaler.transform(X_test.fillna(-999))
    
    print(f"Data preparation complete. Using {X_train.shape[1]} features.")
    
    return X_train, y_train, X_test, y_test

# Model Evaluation 
def evaluate_unsupervised_model(y_true, anomaly_scores, model_name):
    
    print(f"\n Evaluating {model_name} ")
    
    auc_roc = roc_auc_score(y_true, anomaly_scores)
    auc_pr = average_precision_score(y_true, anomaly_scores)
    
    threshold = np.percentile(anomaly_scores, 90) 
    y_pred_binary = (anomaly_scores >= threshold).astype(int)
    
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0,1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"Precision (at 90th percentile threshold): {precision:.4f}")
    print(f"Recall (at 90th percentile threshold): {recall:.4f}")
    print(f"F1-Score (at 90th percentile threshold): {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR (Precision-Recall): {auc_pr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    
    metrics = {
        "model_name": model_name,
        "threshold_percentile": 90,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "false_positive_rate": fpr,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }
    
    metrics_file = os.path.join(OUTPUT_PATH, f"{model_name.lower().replace(' ', '_')}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_file}")
    
    return anomaly_scores

# Main Function
def main():
    
    df = load_featured_data()
    df_train, df_test = temporal_split(df)
    X_train, y_train, X_test, y_test = prepare_data(df_train, df_test)
    
    # Isolation Forest 
    print("\n Training Isolation Forest ")
    contamination_rate = y_train.mean() 
    iso_forest = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)
    
    # Predict anomaly scores. The model's decision_function returns scores where lower is more anomalous.
    # We invert them so that higher scores mean higher risk, for consistency.
    iso_scores = -iso_forest.decision_function(X_test)
    
    # Local Outlier Factor (LOF)
    print("\n Training Local Outlier Factor ")
    # LOF calculates an anomaly score based on local density.
    # The 'novelty=True' parameter allows us to use it for prediction on new data.
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination_rate, novelty=True, n_jobs=-1)
    lof.fit(X_train)
    
    # The decision_function returns scores where lower is more anomalous. We invert them.
    lof_scores = -lof.decision_function(X_test)
    
    # Normalize scores to be in a 0-1 range for consistent evaluation 
    scaler = MinMaxScaler()
    iso_scores_scaled = scaler.fit_transform(iso_scores.reshape(-1, 1)).flatten()
    lof_scores_scaled = scaler.fit_transform(lof_scores.reshape(-1, 1)).flatten()

    # Evaluate Models and Save Results 
    df_test_predictions = df_test.copy()
    df_test_predictions['isolation_forest_score'] = iso_scores_scaled
    df_test_predictions['local_outlier_factor_score'] = lof_scores_scaled
    
    evaluate_unsupervised_model(y_test, iso_scores_scaled, "Isolation Forest")
    evaluate_unsupervised_model(y_test, lof_scores_scaled, "Local Outlier Factor")
    
    output_file = os.path.join(OUTPUT_PATH, "test_set_with_unsupervised_predictions.csv")
    df_test_predictions.to_csv(output_file, index=False)
    print(f"\nTest set with all unsupervised model predictions saved to: {output_file}")

    print("\n Unsupervised Learning Script Finished ")

if __name__ == "__main__":
    main()
