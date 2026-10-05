import pandas as pd
import numpy as np
import os
from datetime import timedelta

# Configuration
DATA_PATH = '../data/'
OUTPUT_PATH = '../outputs/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load and Merge Datasets
def load_and_merge_data():
    print("Loading and merging all datasets...")
    
    txns = pd.read_csv(os.path.join(DATA_PATH, "transactions.csv"), parse_dates=["timestamp"])
    cust = pd.read_csv(os.path.join(DATA_PATH, "customer_info.csv"), parse_dates=["account_open_date"])
    behav = pd.read_csv(os.path.join(DATA_PATH, "behavioral_profiles.csv"))
    wl = pd.read_csv(os.path.join(DATA_PATH, "watchlists.csv"))
    geo = pd.read_csv(os.path.join(DATA_PATH, "geo_risk.csv"))
    device = pd.read_csv(os.path.join(DATA_PATH, "device_channel.csv"))
    
    # Load the rule-based labels to join them later
    labels = pd.read_csv(os.path.join(OUTPUT_PATH, "rule_based_labels.csv"))
    
    # Start merging
    # The 'account_id' in transactions.csv is the sender's account
    df = txns.merge(cust, on="account_id", how="left")
    
    # Merge behavioral data
    df = df.merge(behav, on="account_id", how="left")
    
    # Merge watchlist data
    df = df.merge(wl, on="customer_id", how="left")
    
    # Merge geographical risk based on the receiver's location
    df = df.merge(geo, left_on="receiver_location", right_on="country", how="left")
    
    # Prepare and merge counterparty data
    # Create a 'counterparty' dataframe from the customer table to get receiver info
    counterparty_info = cust[['account_id', 'account_open_date', 'risk_category']].rename(columns={
        'account_id': 'receiver_account',
        'account_open_date': 'counterparty_account_open_date',
        'risk_category': 'counterparty_risk_category'
    })
    df = df.merge(counterparty_info, on='receiver_account', how='left')

    # Merge the labels from the rule engine 
    label_cols = ["transaction_id", "is_suspicious", "risk_score", "matched_rules"]
    df = df.merge(labels[label_cols], on="transaction_id", how="left")
    
    # Handle NaNs from left joins
    # For watchlist data, NaN means the customer is not on a list
    df['PEP_flag'].fillna(False, inplace=True)
    df['sanctioned_entity'].fillna(False, inplace=True)
    df['blacklist_reason'].fillna('None', inplace=True)
    
    # For geo risk, assume 'Low' if a country is not in the risk list
    df['FATF_risk_level'].fillna('Low', inplace=True)
    df['is_sanctioned'].fillna(False, inplace=True)
    
    print("Data loading and merging complete.")
    return df

# Feature Engineering
def engineer_features(df):
    
    print("Starting feature engineering...")
    
    # Sort data for temporal calculations
    df.sort_values(by=['account_id', 'timestamp'], inplace=True)
    
    # Transaction-Level Features
    print("  Creating transaction-level features...")
    df['is_round_value'] = (df['amount'] % 100 == 0) & (df['amount'] > 0)
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Temporal and Velocity Features
    print("  Creating temporal and velocity features...")
    df['time_since_last_txn_sec'] = df.groupby('account_id')['timestamp'].diff().dt.total_seconds()
    df['time_since_last_txn_sec'].fillna(0, inplace=True)

    # Customer-Level Aggregated & Behavioral Features
    print("  Creating rolling window and behavioral features...")
    df_indexed = df.set_index('timestamp')
    rolling_sum_24h = df_indexed.groupby('account_id').rolling('24h')['amount'].sum()
    rolling_count_24h = df_indexed.groupby('account_id').rolling('24h')['transaction_id'].count()
    rolling_avg_30d = df_indexed.groupby('account_id').rolling('30D')['amount'].mean()

    df = df.set_index(['account_id', 'timestamp'])
    df['cust_txn_count_24h'] = rolling_count_24h
    df['cust_txn_sum_24h'] = rolling_sum_24h
    df['cust_avg_txn_amt_30d'] = rolling_avg_30d
    df.reset_index(inplace=True)
    
    df['cust_txn_count_24h'].fillna(1, inplace=True)
    df['cust_txn_sum_24h'].fillna(df['amount'], inplace=True)
    df['cust_avg_txn_amt_30d'].fillna(df['amount'], inplace=True)

    df['txn_amt_to_avg_ratio'] = df['amount'] / (df['cust_avg_txn_amt_30d'] + 1e-6)
    
    # Watchlist and Sanctions Features
    print("  Creating watchlist features...")
    df['is_on_watchlist'] = (df['PEP_flag'] | df['sanctioned_entity']).astype(int)
    
    # Counterparty and Geographic Features
    print("  Creating counterparty and geographic features...")
    df['is_high_risk_country_receiver'] = (df['FATF_risk_level'] == 'High').astype(int)
    df['customer_tenure_days'] = (df['timestamp'] - df['account_open_date']).dt.days
    df['counterparty_account_age_days'] = (df['timestamp'] - df['counterparty_account_open_date']).dt.days
    df['counterparty_account_age_days'].fillna(0, inplace=True)
    
    # Network and Relationship Features
    print("  Creating network features...")
    # Check for reciprocal transactions (B -> A) within 72 hours of (A -> B)
    df['reciprocal_txn_flag'] = 0
    # For demonstration, we'll just create the column. A full implementation
    # would require iterating or complex joins.
    
    # Channel Features
    print("  Creating channel features...")
    # One-hot encode the transaction channel
    df = pd.concat([df, pd.get_dummies(df['channel'], prefix='channel')], axis=1)

    print("Feature engineering complete.")
    return df

# Main Function
def main():
    
    enriched_df = load_and_merge_data()
    featured_df = engineer_features(enriched_df)
    
    output_file = os.path.join(OUTPUT_PATH, "feature_engineered_data.csv")
    print(f"\nSaving feature-engineered data to: {output_file}")
    featured_df.to_csv(output_file, index=False)
    
    print("\n Feature Engineering Script Finished ")
    print(f"Total transactions processed: {len(featured_df)}")
    print(f"Total features created: {len(featured_df.columns)}")
    print("\nSample of the final data with new features:")
    
    # Display a sample of the most relevant old and new features
    sample_cols = [
        'transaction_id', 'amount', 'is_suspicious', 'is_round_value',
        'txn_amt_to_avg_ratio', 'is_on_watchlist', 'is_high_risk_country_receiver',
        'customer_tenure_days', 'counterparty_account_age_days', 'channel_Online'
    ]
    # Ensure all sample columns exist before trying to display them
    display_cols = [col for col in sample_cols if col in featured_df.columns]
    print(featured_df[display_cols].tail())

if __name__ == "__main__":
    main()
