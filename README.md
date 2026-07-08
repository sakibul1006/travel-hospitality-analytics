# Travel, Tourism & Hospitality — Customer Retention and Dynamic Pricing Analysis

**Infotact Solutions & Co. — Data Analytics Internship, Project 2**

An end-to-end data analytics pipeline that analyzes hotel booking data to
uncover the drivers of customer cancellations (churn), explore seasonal
demand and pricing patterns, segment customers by booking behavior, and
deliver a baseline machine-learning model that scores each booking's
cancellation risk for the Revenue Management and Marketing teams.

## Business problem

Travel and hospitality businesses lose revenue to unoptimized pricing and
unpredictable cancellations. This project builds the data foundation needed
for (a) a dynamic pricing engine, by quantifying how demand and ADR (Average
Daily Rate) move together across the season, and (b) targeted retention
campaigns, by identifying which customer segments and booking conditions
carry the highest cancellation risk.

## Repository structure

```
travel-hospitality-analytics/
├── data/
│   ├── raw/                     # hotel_bookings_raw.csv (synthetic, messy — Week 1 input)
│   └── processed/               # cleaned + dashboard-ready CSVs (pipeline output)
├── notebooks/
│   └── hotel_booking_analysis.ipynb   # full Week 1-4 analysis (cleaning → EDA → modeling)
├── reports/                     # exported chart images referenced in the final report
├── dashboard/
│   ├── Hotel_Dashboard.pbix          # Add Power BI dashboard.
│   ├── dashboard_data.json      # aggregates powering the dashboard
│   └── POWER_BI_GUIDE.md        # step-by-step guide to rebuild this as a native .pbix
├── src/
│   └── generate_dataset.py      # synthetic data generator (documents dataset provenance)
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

`data/raw/hotel_bookings_raw.csv` is a **synthetic** dataset (6,000+ rows)
generated to match the statistical structure of real-world hotel booking
demand data (lead time, ADR, deposit type, market segment, cancellations,
etc.), including realistic messiness — missing values, duplicate rows, sign
errors in `adr`, and inconsistent country-code casing — so the cleaning steps
in Week 1 are meaningful and reproducible. See `src/generate_dataset.py` for
exact generation logic. Using synthetic data avoids any PII/PHI exposure risk
while preserving the analytical challenge.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt

# 1. Regenerate the raw dataset (optional — already included in data/raw/)
python src/generate_dataset.py

# 2. Run the full analysis notebook top to bottom
jupyter nbconvert --to notebook --execute --inplace notebooks/hotel_booking_analysis.ipynb

# 3. Open the interactive dashboard
open dashboard/dashboard.html      # or just double-click the file
```

## Methodology (maps to the 4-week roadmap)

| Week | Focus | Notebook section |
|---|---|---|
| 1 | Data acquisition, cleaning, feature engineering (nulls, duplicates, sign errors, `total_nights`/`total_guests` features) | "Week 1" |
| 2 | EDA: seasonal demand vs. ADR, booking curve (lead time vs. cancellation), deposit/segment cross-tabs, correlation matrix | "Week 2" |
| 3 | Baseline predictive modeling: Logistic Regression + Random Forest, evaluated with Accuracy/Precision/Recall/ROC-AUC, feature importance | "Week 3" |
| 4 | Insight synthesis, strategic recommendations, dashboard export | "Week 4" + `dashboard/` |

## Key findings

1. **Lead time is the strongest cancellation driver.** Cancellation rate rises
   from ~22% for bookings made within a week of arrival to ~68% for bookings
   made 180+ days out.
2. **Non-refundable deposits do not reduce cancellations** in this data —
   they correlate with a *higher* cancellation rate (56% vs. ~25% for
   no-deposit bookings), consistent with known hospitality-industry research
   on speculative bookings.
3. **Peak months (Jun–Aug, Dec) command the highest ADR**, confirming room
   for further dynamic price testing during confirmed high-demand windows.
4. **"Early Planner" segments cancel roughly 1.5–2× more often** than
   "Last-Minute Booker" segments — the Marketing team should prioritize
   retention outreach (deposit reminders, itinerary confirmations) at the
   60+ day mark for these segments.
5. A **Random Forest classifier** outperforms the logistic-regression
   baseline on ROC-AUC while still exposing interpretable feature
   importances (lead time, ADR, and previous cancellations are the top
   predictors) — recommended as the production candidate.

Full metrics, confusion matrices, and ROC curves are in the notebook and
`reports/` folder.

## Dashboard

`dashboard/dashboard.html` is a self-contained interactive dashboard (KPI
cards, seasonal ADR trend, lead-time risk chart, deposit/segment breakdown,
source-market mix, and a customer-segment risk table) built from the same
cleaned dataset used in the notebook. See `dashboard/POWER_BI_GUIDE.md` to
recreate the identical report natively in Power BI Desktop against
`data/processed/hotel_bookings_dashboard_ready.csv`.

## Ethics & data privacy

No real customer or PII/PHI data is used anywhere in this project — the
dataset is entirely synthetic (see `src/generate_dataset.py`). Sensitive
connection strings are never hardcoded; any future production version should
inject credentials via environment variables as noted in `.gitignore`.
