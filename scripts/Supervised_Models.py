import pandas as pd
import numpy as np
import os
import json
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder

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
    
    print("Preparing data for ML models...")
    
    # Define target and features
    TARGET = 'is_suspicious'
    
    # Define columns to drop
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

    # Identify categorical and numerical features
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
    
    # Apply label encoding to categorical features
    for col in categorical_features:
        le = LabelEncoder()
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)
        
        # Fit on training data and transform both train and test
        le.fit(X_train[col])
        # Identify unseen labels in test set
        unseen_labels = set(X_test[col]) - set(le.classes_)
        if unseen_labels:
            # Add an "__UNKNOWN__" category
            le.classes_ = np.append(le.classes_, "__UNKNOWN__")
            # Replace unseen labels with "__UNKNOWN__"
            X_test[col] = X_test[col].apply(lambda x: x if x in le.classes_ else "__UNKNOWN__")
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

    # Handle potential NaN values in numeric columns
    X_train.fillna(-999, inplace=True)
    X_test.fillna(-999, inplace=True)
    
    print(f"Data preparation complete. Using {X_train.shape[1]} features.")
    
    return X_train, y_train, X_test, y_test

# Model Training and Evaluation
def evaluate_model(y_true, y_pred_proba, model_name, threshold=0.5):
    
    print(f"\n Evaluating {model_name} ")
    
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    auc_pr = average_precision_score(y_true, y_pred_proba)
    
    # Confusion Matrix to calculate False Positive Rate (FPR)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR (Precision-Recall): {auc_pr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    
    metrics = {
        "model_name": model_name,
        "threshold": threshold,
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
    
    # Save metrics to a JSON file
    metrics_file = os.path.join(OUTPUT_PATH, f"{model_name.lower().replace(' ', '_')}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_file}")
    
    return metrics

# Main Function 
def main():
    
    df = load_featured_data()
    df_train, df_test = temporal_split(df)
    X_train, y_train, X_test, y_test = prepare_data(df_train, df_test)
    
    # Handle Class Imbalance
    # Calculate the weight for the minority class (suspicious=1)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nCalculated scale_pos_weight for imbalance: {scale_pos_weight:.2f}")

    # Define Models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, 
            class_weight='balanced', 
            random_state=42, 
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=100,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )
    }
    
    # Training and Evaluating Each Model
    df_test_predictions = df_test.copy()
    
    for name, model in models.items():
        print(f"\n Training {name} ")
        model.fit(X_train, y_train)
        
        # Predict probabilities on the test set
        predictions_proba = model.predict_proba(X_test)[:, 1]
        
        # Store predictions
        df_test_predictions[f'{name}_risk_score'] = predictions_proba
        
        # Evaluate performance
        evaluate_model(y_test, predictions_proba, name, threshold=0.5)

    # Saving the Test Set with All Model Predictions 
    output_file = os.path.join(OUTPUT_PATH, "test_set_with_ml_predictions.csv")
    df_test_predictions.to_csv(output_file, index=False)
    print(f"\nTest set with all ML model predictions saved to: {output_file}")
    
    print("\n Supervised Learning Script Finished ")

if __name__ == "__main__":
    main()
