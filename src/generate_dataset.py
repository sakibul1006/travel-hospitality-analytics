"""
generate_dataset.py
--------------------
Generates a synthetic hotel booking demand dataset that mirrors the structure
and statistical properties of real-world hotel booking datasets (e.g. the
well-known City Hotel / Resort Hotel booking demand data), but is entirely
synthetic. This avoids any real customer data / PHI-PII concerns while still
giving realistic distributions for lead_time, ADR, cancellations, seasonality,
customer segments, and deposit types.

Run: python src/generate_dataset.py
Output: data/raw/hotel_bookings_raw.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 6000

hotel_types = np.random.choice(["Resort Hotel", "City Hotel"], size=N, p=[0.35, 0.65])

# Arrival date components (2 years of data)
arrival_year = np.random.choice([2023, 2024], size=N, p=[0.48, 0.52])
arrival_month = np.random.choice(
    ["January","February","March","April","May","June","July","August",
     "September","October","November","December"], size=N
)
arrival_day = np.random.randint(1, 29, size=N)

# Lead time (days between booking and arrival) - right skewed
lead_time = np.random.exponential(scale=60, size=N).astype(int)
lead_time = np.clip(lead_time, 0, 500)

# Customer type / segment
customer_type = np.random.choice(
    ["Transient", "Transient-Party", "Contract", "Group"],
    size=N, p=[0.55, 0.25, 0.10, 0.10]
)

market_segment = np.random.choice(
    ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Complementary", "Groups"],
    size=N, p=[0.40, 0.18, 0.15, 0.12, 0.03, 0.12]
)

# Stay duration
stays_weekend_nights = np.random.poisson(1.0, size=N)
stays_week_nights = np.random.poisson(2.2, size=N)

# Deposit type - non-refundable deposits correlate with different cancel behavior
deposit_type = np.random.choice(
    ["No Deposit", "Non Refund", "Refundable"], size=N, p=[0.75, 0.20, 0.05]
)

# Average Daily Rate - seasonal effect baked in via month
month_price_factor = {
    "January": 0.85, "February": 0.85, "March": 0.95, "April": 1.0,
    "May": 1.05, "June": 1.15, "July": 1.30, "August": 1.35,
    "September": 1.10, "October": 0.95, "November": 0.85, "December": 1.20
}
base_adr = np.random.normal(100, 25, size=N)
adr = np.array([max(20, base_adr[i] * month_price_factor[arrival_month[i]] *
                     (1.25 if hotel_types[i] == "Resort Hotel" else 1.0))
                for i in range(N)])
adr = np.round(adr, 2)

# Special requests, previous cancellations, parking
total_of_special_requests = np.random.poisson(0.6, size=N)
previous_cancellations = np.random.poisson(0.15, size=N)
required_car_parking_spaces = np.random.choice([0, 1], size=N, p=[0.88, 0.12])
adults = np.random.choice([1, 2, 3, 4], size=N, p=[0.25, 0.55, 0.15, 0.05])
children = np.random.choice([0, 1, 2], size=N, p=[0.80, 0.15, 0.05])

# Agent / company IDs (with realistic missingness)
agent = np.random.choice(
    list(range(1, 30)) + [np.nan], size=N,
    p=[0.90/29]*29 + [0.10]
)
company = np.random.choice(
    list(range(100, 120)) + [np.nan]*8, size=N
)

country = np.random.choice(
    ["PRT", "GBR", "USA", "ESP", "FRA", "DEU", "ITA", "IND", "BRA", "IRL"],
    size=N, p=[0.25,0.15,0.12,0.10,0.10,0.08,0.07,0.06,0.04,0.03]
)

# --- Cancellation probability model (ground truth signal for later ML) ---
# Higher lead time, non-refund deposit(oddly correlated with fraud-cancels in
# real datasets), transient-party customers, and prior cancellations increase
# cancellation likelihood. High special requests / car parking reduce it.
logit = (
    -1.8
    + 0.010 * lead_time
    + (1.6 * (deposit_type == "Non Refund"))
    + (0.35 * (customer_type == "Transient-Party"))
    + (0.55 * previous_cancellations)
    - (0.35 * total_of_special_requests)
    - (0.30 * required_car_parking_spaces)
    + np.random.normal(0, 0.6, size=N)
)
prob_cancel = 1 / (1 + np.exp(-logit))
is_canceled = (np.random.rand(N) < prob_cancel).astype(int)

df = pd.DataFrame({
    "hotel": hotel_types,
    "is_canceled": is_canceled,
    "lead_time": lead_time,
    "arrival_date_year": arrival_year,
    "arrival_date_month": arrival_month,
    "arrival_date_day_of_month": arrival_day,
    "stays_in_weekend_nights": stays_weekend_nights,
    "stays_in_week_nights": stays_week_nights,
    "adults": adults,
    "children": children,
    "country": country,
    "market_segment": market_segment,
    "customer_type": customer_type,
    "adr": adr,
    "required_car_parking_spaces": required_car_parking_spaces,
    "total_of_special_requests": total_of_special_requests,
    "previous_cancellations": previous_cancellations,
    "deposit_type": deposit_type,
    "agent": agent,
    "company": company,
})

# --- Inject realistic messiness for the Week 1 cleaning exercise ---
# 1. Missing children values
missing_idx = np.random.choice(df.index, size=15, replace=False)
df.loc[missing_idx, "children"] = np.nan

# 2. A few negative / impossible ADR values (data entry errors)
err_idx = np.random.choice(df.index, size=8, replace=False)
df.loc[err_idx, "adr"] = -df.loc[err_idx, "adr"]

# 3. Extreme ADR outliers
out_idx = np.random.choice(df.index, size=5, replace=False)
df.loc[out_idx, "adr"] = df.loc[out_idx, "adr"] * 12

# 4. Duplicate rows
dupes = df.sample(20, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

# 5. Inconsistent country casing / whitespace (format standardization exercise)
messy_country_idx = np.random.choice(df.index, size=40, replace=False)
df.loc[messy_country_idx, "country"] = df.loc[messy_country_idx, "country"].str.lower() + " "

import os
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "hotel_bookings_raw.csv")
df.to_csv(OUT, index=False)
print(f"Generated {len(df)} rows -> {OUT}")
print(df.isna().sum()[df.isna().sum() > 0])
