import pandas as pd
import numpy as np
import os
from datetime import timedelta

# Risk Weight Configuration
RULE_WEIGHTS = {
    "Structuring": 8,
    "Rapid Fund Movement": 7,
    "Profile Inconsistency": 5,
    "Money Mules": 6,
    "Round Tripping": 6,
    "Shell Company": 4,
    "TBML Pattern": 3,
    "Dormant Activity": 4,
    "High-Risk Country": 5,
    "Third-Party Intermediary": 5
}

# Load and Merge Datasets 
def load_data():
    print(" Loading data...")
    txns = pd.read_csv("../data/transactions.csv", parse_dates=["timestamp"])
    cust = pd.read_csv("../data/customer_info.csv")
    behav = pd.read_csv("../data/behavioral_profiles.csv")
    wl = pd.read_csv("../data/watchlists.csv")
    geo = pd.read_csv("../data/geo_risk.csv")
    ext = pd.read_csv("../data/external_enrichment.csv")
    return txns, cust, behav, wl, geo, ext

def enrich(txns, cust, behav, wl, geo, ext):
    print(" Joining datasets...")
    txns["account_id"] = txns["sender_account"]
    df = txns.merge(cust, on="account_id", how="left") \
             .merge(behav, on="account_id", how="left") \
             .merge(ext, on="customer_id", how="left") \
             .merge(wl, on="customer_id", how="left") \
             .merge(geo, left_on="receiver_location", right_on="country", how="left")
    return df

# Apply AML Typology Rules
def apply_rules(df):
    print(" Applying rules...")

    df["is_suspicious"] = 0
    df["risk_score"] = 0
    df["matched_rules"] = ""

    account_ids = df["account_id"].unique().tolist()

    def add_flag(idx, rule_name):
        risk = RULE_WEIGHTS.get(rule_name, 0)
        df.at[idx, "is_suspicious"] = 1
        df.at[idx, "risk_score"] += risk 
        if df.at[idx, "matched_rules"]:
            df.at[idx, "matched_rules"] += f"; {rule_name}"
        else:
            df.at[idx, "matched_rules"] = rule_name

    for idx, row in df.iterrows():
        # Rule 1: Structuring
        day_txns = df[(df["customer_id"] == row["customer_id"]) &
                      (df["timestamp"] >= row["timestamp"] - timedelta(hours=12)) &
                      (df["timestamp"] <= row["timestamp"] + timedelta(hours=12)) &
                      (df["transaction_type"] == "Cash Deposit") &
                      (df["amount"] < 10000)]

        if len(day_txns) >= 3 and day_txns["amount"].sum() > 9000:
            add_flag(idx, "Structuring")

        # Rule 2: Rapid Fund Movement
        if row["amount"] > 20000:
            outgoing_window = df[(df["account_id"] == row["account_id"]) &
                                 (df["timestamp"] > row["timestamp"]) &
                                 (df["timestamp"] <= row["timestamp"] + timedelta(hours=12))]

            if not outgoing_window.empty and outgoing_window["amount"].sum() > 0.9 * row["amount"]:
                add_flag(idx, "Rapid Fund Movement")

        # Rule 3: Profile Inconsistency
        if row["annual_income"] < 50000 and row["amount"] > 25000:
            add_flag(idx, "Profile Inconsistency")

        # Rule 4: Money Mules
        incoming_txns = df[(df["receiver_account"] == row["receiver_account"]) &
                           (df["timestamp"] >= row["timestamp"] - timedelta(hours=12))]
        unique_senders = incoming_txns["sender_account"].nunique()
        if unique_senders > 10:
            add_flag(idx, "Money Mules")

        # Rule 5: Round-Tripping
        recent = df[(df["sender_account"] == row["receiver_account"]) &
                    (df["receiver_account"] == row["sender_account"]) &
                    (abs(df["amount"] - row["amount"]) <= 0.05 * row["amount"]) &
                    (df["timestamp"] >= row["timestamp"]) &
                    (df["timestamp"] <= row["timestamp"] + timedelta(hours=72))]
        if not recent.empty:
            add_flag(idx, "Round Tripping")

        # Rule 6: Shell Company
        if row["transaction_type"] == "Wire" and row["amount"] > 15000:
            if row["FATF_risk_level"] == "High" and pd.isnull(row["employer"]):
                add_flag(idx, "Shell Company")

        # Rule 7: TBML Pattern
        if row["transaction_type"] == "Wire" and row["channel"] == "In-Branch":
            add_flag(idx, "TBML Pattern")

        # Rule 8: Dormant Activity
        if row["account_open_date"] < "2022-01-01":
            history = df[(df["account_id"] == row["account_id"]) &
                         (df["timestamp"] < row["timestamp"]) &
                         (df["timestamp"] >= row["timestamp"] - timedelta(days=7))]

            if history["amount"].sum() > 50000 and row["txn_count_30d"] < 3:
                add_flag(idx, "Dormant Activity")

        # Rule 9: High-Risk Country
        if row["FATF_risk_level"] == "High" and row["amount"] > 5000:
            add_flag(idx, "High-Risk Country")

        # Rule 10: Third-Party Intermediary
        if row["transaction_type"] == "Wire" and row["sender_account"] not in account_ids and row["receiver_account"] not in account_ids:
            add_flag(idx, "Third-Party Intermediary")

    return df

# Main Function
def main():
    txns, cust, behav, wl, geo, ext = load_data()
    df = enrich(txns, cust, behav, wl, geo, ext)
    df = apply_rules(df)

    os.makedirs("../outputs", exist_ok=True)
    df.to_csv("../outputs/rule_based_labels.csv", index=False)

    print(f"\n Rule engine complete. {df['is_suspicious'].sum()} suspicious transactions flagged.")
    print(" Output saved to: outputs/rule_based_labels.csv")

if __name__ == "__main__":
    main()
