import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import os

fake = Faker()
os.makedirs('../data', exist_ok=True)

n_customers = 500
n_transactions = 3000
customer_ids = [1000 + i for i in range(n_customers)]
account_ids = [1000 + i for i in range(n_customers)]

# 1. Customer Info Dataset
customer_info = pd.DataFrame({
    "customer_id": customer_ids,
    "full_name": [fake.name() for _ in range(n_customers)],
    "dob": [fake.date_of_birth(minimum_age=18, maximum_age=80) for _ in range(n_customers)],
    "national_id": [fake.ssn() for _ in range(n_customers)],
    "nationality": [fake.country() for _ in range(n_customers)],
    "gender": [random.choice(["M", "F"]) for _ in range(n_customers)],
    "phone": [fake.phone_number() for _ in range(n_customers)],
    "email": [fake.email() for _ in range(n_customers)],
    "address": [fake.address().replace("\n", ", ") for _ in range(n_customers)],
    "occupation": [random.choice(["Engineer", "Doctor", "Teacher", "Business", "Student"]) for _ in range(n_customers)],
    "employer": [fake.company() for _ in range(n_customers)],
    "income_source": [random.choice(["Salary", "Freelance", "Investments", "Unknown"]) for _ in range(n_customers)],
    "annual_income": [random.randint(20000, 200000) for _ in range(n_customers)],
    "account_id": account_ids,
    "account_type": [random.choice(["Savings", "Business", "Current"]) for _ in range(n_customers)],
    "account_open_date": [fake.date_between(start_date='-10y', end_date='-1y') for _ in range(n_customers)],
    "risk_category": [random.choice(["Low", "Medium", "High"]) for _ in range(n_customers)]
})
customer_info.to_csv('../data/customer_info.csv', index=False)

# 2. Transaction Dataset (includes account_id = sender's account)
transactions = []
for i in range(n_transactions):
    sender_idx = random.randint(0, n_customers - 1)
    receiver_idx = random.randint(0, n_customers - 1)
    transactions.append({
        "transaction_id": 100000 + i,
        "timestamp": fake.date_time_between(start_date='-1y', end_date='now'),
        "account_id": account_ids[sender_idx],  # NEW LINE
        "amount": round(random.uniform(10, 10000), 2),
        "currency": random.choice(["USD", "EUR", "GBP"]),
        "sender_account": account_ids[sender_idx],
        "receiver_account": account_ids[receiver_idx],
        "sender_location": fake.country(),
        "receiver_location": fake.country(),
        "channel": random.choice(["ATM", "Online", "Mobile", "In-Branch"]),
        "transaction_type": random.choice(["Wire", "Cash Deposit", "Card Payment", "Crypto"]),
        "balance_before": round(random.uniform(1000, 50000), 2),
        "balance_after": round(random.uniform(1000, 50000), 2),
        "device_used": random.choice(["Web", "Mobile", "ATM Terminal"])
    })
transactions_df = pd.DataFrame(transactions)
transactions_df.to_csv('../data/transactions.csv', index=False)

# 3. Behavioral Profile Dataset
behavioral = pd.DataFrame({
    "account_id": account_ids,
    "avg_txn_size_30d": [round(random.uniform(100, 5000), 2) for _ in account_ids],
    "txn_count_30d": [random.randint(5, 100) for _ in account_ids],
    "std_txn_amount": [round(random.uniform(10, 2000), 2) for _ in account_ids],
    "recurring_txn": [random.choice([True, False]) for _ in account_ids],
    "geo_txn_diversity": [random.randint(1, 5) for _ in account_ids]
})
behavioral.to_csv('../data/behavioral_profiles.csv', index=False)

# 4. Watchlists / Sanctions / PEPs
watchlists = pd.DataFrame({
    "customer_id": random.sample(customer_ids, k=int(n_customers * 0.2)),
    "PEP_flag": [random.choice([True, False]) for _ in range(int(n_customers * 0.2))],
    "sanctioned_entity": [random.choice([True, False]) for _ in range(int(n_customers * 0.2))],
    "blacklist_reason": [random.choice(["OFAC", "FATF", "None"]) for _ in range(int(n_customers * 0.2))]
})
watchlists.to_csv('../data/watchlists.csv', index=False)

# 5. Case Management / Alerts
alerts = pd.DataFrame({
    "alert_id": [f"ALT{i+1000}" for i in range(300)],
    "linked_transaction": random.sample(list(transactions_df["transaction_id"]), 300),
    "alert_reason": [random.choice(["High Value", "Structuring", "Suspicious Pattern", "PEP Link"]) for _ in range(300)],
    "analyst_comments": [fake.sentence() for _ in range(300)],
    "escalation_status": [random.choice(["Pending", "Escalated", "Dismissed"]) for _ in range(300)],
    "SAR_filed": [random.choice([True, False]) for _ in range(300)],
    "regulatory_outcome": [random.choice(["Reported", "Cleared", "Investigating"]) for _ in range(300)]
})
alerts.to_csv('../data/alerts.csv', index=False)

# 6. Geographical / Jurisdiction Risk
countries = list(set(transactions_df["receiver_location"]))[:50]
geo_risk = pd.DataFrame({
    "country": countries,
    "FATF_risk_level": [random.choice(["Low", "Medium", "High"]) for _ in countries],
    "is_sanctioned": [random.choice([True, False]) for _ in countries],
    "regulatory_strength": [random.choice(["Strong", "Weak", "Moderate"]) for _ in countries]
})
geo_risk.to_csv('../data/geo_risk.csv', index=False)

# 7. Business / Merchant Dataset
business_profiles = pd.DataFrame({
    "account_id": random.sample(account_ids, 100),
    "business_name": [fake.company() for _ in range(100)],
    "registration_number": [fake.uuid4()[:8] for _ in range(100)],
    "industry": [random.choice(["Retail", "Finance", "Healthcare", "Crypto", "Import/Export"]) for _ in range(100)],
    "country": [fake.country() for _ in range(100)],
    "revenue": [random.randint(100000, 1000000) for _ in range(100)],
    "employees": [random.randint(10, 100) for _ in range(100)],
    "ownership_disclosed": [random.choice([True, False]) for _ in range(100)]
})
business_profiles.to_csv('../data/business_profiles.csv', index=False)

# 8. Device & Channel Dataset
device_data = pd.DataFrame({
    "customer_id": random.choices(customer_ids, k=500),
    "device_id": [fake.uuid4() for _ in range(500)],
    "ip_address": [fake.ipv4_public() for _ in range(500)],
    "os": [random.choice(["Windows", "Linux", "MacOS", "Android", "iOS"]) for _ in range(500)],
    "login_freq_7d": [random.randint(1, 20) for _ in range(500)],
    "mfa_enabled": [random.choice([True, False]) for _ in range(500)],
    "channel": [random.choice(["Mobile App", "Web", "ATM"]) for _ in range(500)]
})
device_data.to_csv('../data/device_channel.csv', index=False)

# 10. External Enrichment
external = pd.DataFrame({
    "customer_id": random.choices(customer_ids, k=300),
    "credit_score": [random.randint(300, 850) for _ in range(300)],
    "adverse_news": [random.choice([True, False]) for _ in range(300)],
    "black_market_flag": [random.choice([True, False]) for _ in range(300)]
})
external.to_csv('../data/external_enrichment.csv', index=False)

print(" All AML synthetic datasets created in the 'data/' folder.")
