# train_model.py

import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from src.preprocessing import load_data, clean_data
from src.feature_engineering import add_features
from src.model_utils import train_supervised_model, evaluate_model

DATA_PATH = "synthetic_aml_transactions.csv"
MODEL_PATH = "model/supervised_model.pkl"

def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found at {DATA_PATH}")
        return

    print("📥 Loading data...")
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = add_features(df)

    if 'is_suspicious' not in df.columns:
        print("❌ 'label' column (target) not found in dataset.")
        return

    print("🔀 Splitting dataset...")
    X = df[['amount', 'hour', 'day_of_week']]
    y = df['is_suspicious']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🎯 Training model...")
    model = train_supervised_model(X_train, y_train, MODEL_PATH)

    print("📊 Evaluating model...")
    report = evaluate_model(model, X_test, y_test)
    print(report)

    print(f"✅ Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
