import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# Configuration #
OUTPUT_PATH = '../outputs/'
os.makedirs(os.path.join(OUTPUT_PATH, 'visualizations'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'explanations'), exist_ok=True)

# Cross-Paradigm Performance Benchmarking  #
def load_all_metrics():
    
    print("--- Loading all model metrics ---")
    metrics_files = [f for f in os.listdir(OUTPUT_PATH) if f.endswith('_metrics.json')]
    all_metrics = []
    for file in metrics_files:
        with open(os.path.join(OUTPUT_PATH, file), 'r') as f:
            metrics = json.load(f)
            all_metrics.append(metrics)
    print(f"Loaded metrics for {len(all_metrics)} models.")
    return pd.DataFrame(all_metrics)

def create_summary_table(metrics_df):
    
    print("\n--- Comparative Performance Summary Table ---")
    summary_cols = {
        'model_name': 'Model', 'f1_score': 'F1-Score', 'auc_roc': 'AUC-ROC',
        'auc_pr': 'AUC-PR', 'precision': 'Precision', 'recall': 'Recall',
        'false_positive_rate': 'FPR'
    }
    summary_df = metrics_df[list(summary_cols.keys())].rename(columns=summary_cols).round(4).set_index('Model')
    print(summary_df.to_string())
    summary_df.to_csv(os.path.join(OUTPUT_PATH, 'visualizations', 'model_performance_summary.csv'))
    print("\nSummary table saved to outputs/visualizations/model_performance_summary.csv")
    return summary_df

def plot_precision_recall_at_k(predictions_df, y_true):
    
    print("\n--- Generating Precision/Recall@K Curves ---")
    models = {
        'Hard-Rule Baseline': 'hard_rule_alert', 'Random Forest': 'Random Forest_risk_score',
        'XGBoost': 'XGBoost_risk_score', 'LightGBM': 'LightGBM_risk_score',
        'Isolation Forest': 'isolation_forest_score', 'Local Outlier Factor': 'local_outlier_factor_score'
    }
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    total_positives = y_true.sum()
    k_values = np.linspace(50, len(y_true), 100, dtype=int)

    for model_name, score_col in models.items():
        if score_col not in predictions_df.columns: continue
        sorted_df = predictions_df.sort_values(by=score_col, ascending=False)
        precision_at_k, recall_at_k = [], []
        for k in k_values:
            top_k = sorted_df.head(k)
            true_positives_at_k = top_k['is_suspicious'].sum()
            precision_at_k.append(true_positives_at_k / k)
            recall_at_k.append(true_positives_at_k / total_positives)
        ax1.plot(k_values, precision_at_k, label=model_name)
        ax2.plot(k_values, recall_at_k, label=model_name)

    ax1.set_title('Precision@K: Investigator Efficiency', fontsize=16)
    ax1.set_xlabel('K (Number of Top Alerts Reviewed)', fontsize=12)
    ax1.set_ylabel('Precision ("Hit Rate")', fontsize=12)
    ax1.legend(); ax1.set_ylim(0, 1.05)
    ax2.set_title('Recall@K: Detection Effectiveness', fontsize=16)
    ax2.set_xlabel('K (Number of Top Alerts Reviewed)', fontsize=12)
    ax2.set_ylabel('Recall (Fraction of Total Risk Captured)', fontsize=12)
    ax2.legend(); ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_PATH, 'visualizations', 'precision_recall_at_k_curves.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Precision/Recall@K plot saved to {plot_path}")
    plt.show()

def plot_fpr_comparison(metrics_df):
    """
    Creates and saves a bar chart comparing the False Positive Rate (FPR) of all models.
    """
    print("\n--- Generating False Positive Rate Comparison Plot ---")
    
    # Sort models by FPR for better visualization
    df_sorted = metrics_df.sort_values('false_positive_rate', ascending=False)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))
    
    ax = sns.barplot(x='false_positive_rate', y='model_name', data=df_sorted, palette='viridis')
    
    # Use a logarithmic scale to highlight the huge differences
    ax.set_xscale("log")
    
    plt.title('Comparison of False Positive Rates (Log Scale)', fontsize=18, pad=20)
    plt.xlabel('False Positive Rate (Lower is Better)', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    
    # Add annotations to the bars
    for p in ax.patches:
        width = p.get_width()
        plt.text(width * 1.1, p.get_y() + p.get_height() / 2,
                 f'{width:.4%}', 
                 va='center')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_PATH, 'visualizations', 'fpr_comparison_chart.png')
    plt.savefig(plot_path, dpi=300)
    print(f"FPR comparison plot saved to {plot_path}")
    plt.show()


# Advanced Model Explainability #
def explain_supervised_model():
    
    print("\n--- Generating SHAP Explanations for Random Forest ---")
    df = pd.read_csv(os.path.join(OUTPUT_PATH, 'feature_engineered_data.csv'))
    df_train, df_test = df.iloc[:int(len(df)*0.8)], df.iloc[int(len(df)*0.8):]
    TARGET = 'is_suspicious'
    cols_to_drop = [
        'transaction_id', 'customer_id', 'account_id', 'sender_account', 'receiver_account',
        'timestamp', 'account_open_date', 'counterparty_account_open_date', 'full_name',
        'dob', 'national_id', 'phone', 'email', 'address', 'employer', 'blacklist_reason',
        'matched_rules', 'risk_score', 'country'
    ]
    X_train = df_train.drop(columns=cols_to_drop + [TARGET]).fillna(-999)
    y_train = df_train[TARGET].fillna(0)
    X_test = df_test.drop(columns=cols_to_drop + [TARGET]).fillna(-999)

    # For each categorical column, fit the LabelEncoder on ALL possible values from both train and test sets.
    for col in X_train.select_dtypes(include=['object']).columns:
        all_labels = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
        le = LabelEncoder().fit(all_labels)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure()
    shap.summary_plot(shap_values[1], X_test, show=False, plot_size=(10, 8))
    plt.title("SHAP Summary Plot for Random Forest", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'explanations', 'shap_summary_plot.png'), dpi=300)
    plt.close()
    
    pred_probas = model.predict_proba(X_test)[:, 1]
    high_risk_idx = np.argmax(pred_probas)
    shap.force_plot(explainer.expected_value[1], shap_values[1][high_risk_idx,:], X_test.iloc[high_risk_idx,:], matplotlib=True, show=False, text_rotation=15)
    plt.savefig(os.path.join(OUTPUT_PATH, 'explanations', 'shap_force_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP summary and force plots saved to outputs/explanations/")

# Main Function #
def main():
    
    metrics_df = load_all_metrics()
    create_summary_table(metrics_df)
    plot_fpr_comparison(metrics_df) # <-- NEW FUNCTION CALL
    
    try:
        preds_rule = pd.read_csv(os.path.join(OUTPUT_PATH, 'test_set_with_rule_predictions.csv'))
        preds_ml = pd.read_csv(os.path.join(OUTPUT_PATH, 'test_set_with_ml_predictions.csv'))
        preds_unsup = pd.read_csv(os.path.join(OUTPUT_PATH, 'test_set_with_unsupervised_predictions.csv'))
        
        # Merge all predictions into one dataframe
        all_preds = pd.merge(preds_rule, preds_ml, on='transaction_id', how='left', suffixes=('', '_drop'))
        all_preds = pd.merge(all_preds, preds_unsup, on='transaction_id', how='left', suffixes=('', '_drop'))
        all_preds.drop([col for col in all_preds.columns if '_drop' in col], axis=1, inplace=True)
        y_true = all_preds['is_suspicious'].fillna(0)
        plot_precision_recall_at_k(all_preds, y_true)
    except FileNotFoundError as e:
        print(f"\nCould not generate @K curves because a prediction file was not found: {e}")

    explain_supervised_model()
    
    print("\n\n Final Analysis Script Finished ")
    print("All comparison tables and visualizations have been saved to the 'outputs' folder.")

if __name__ == "__main__":
    main()
