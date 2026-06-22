# Final Report — Customer Retention and Dynamic Pricing Analysis

**Project:** Travel, Tourism & Hospitality (Project 2)
**Author:** Data Analytics Intern
**Program:** Infotact Technical Internship Program

## 1. Problem statement

Hotel chains lose revenue to unoptimized pricing and unpredictable
cancellations. This project analyzes historical booking data to identify the
primary drivers of cancellations and seasonal demand patterns, providing the
data foundation for a dynamic pricing engine and targeted retention
campaigns.

## 2. Data

A synthetic, 6,000+ row hotel booking dataset (see `src/generate_dataset.py`)
modeled on real-world hotel booking demand data structure: lead time, ADR,
deposit type, market segment, customer type, prior cancellations, and
outcome (`is_canceled`). Realistic messiness (missing values, duplicates,
sign errors, inconsistent formatting) was intentionally injected to exercise
the Week 1 cleaning pipeline.

## 3. Cleaning summary

- Removed 20 duplicate rows.
- Imputed `children` (15 missing) and `agent`/`company` (missing = no
  agent/company, sentinel 0).
- Corrected sign errors in `adr` (8 rows) and winsorized outliers at the 99th
  percentile.
- Standardized `country` casing/whitespace.
- Engineered `total_nights`, `total_guests`, `lead_time_bucket`, and
  `customer_segment`.

## 4. Key findings

| # | Finding | Business implication |
|---|---|---|
| 1 | Cancellation rate rises from ~22% (0-7 day lead time) to ~68% (180+ days) | Require partial deposits / confirmation calls for long-lead-time bookings |
| 2 | Non-refundable deposits show a *higher* cancellation rate (56%) than no-deposit bookings (25%) | Don't rely on deposit policy alone to control churn; combine with proactive retention outreach |
| 3 | Peak months (Jun-Aug, Dec) show the highest ADR alongside strong booking volume | Room to test further price increases in the 4-6 week pre-arrival window during these months |
| 4 | "Early Planner" segments cancel ~1.5-2x more often than "Last-Minute Booker" segments | Prioritize retention campaigns at the 60+ day mark for early planners |
| 5 | Random Forest model reaches a higher ROC-AUC than logistic regression while remaining explainable via feature importance | Recommended as the production cancellation-risk scoring model |

## 5. Model performance

Two baseline classifiers were trained on a 75/25 stratified train/test split:

- **Logistic Regression** (interpretable baseline, class-balanced)
- **Random Forest** (300 trees, max depth 8, class-balanced)

Both are evaluated on Accuracy, Precision, Recall, and ROC-AUC in
`notebooks/hotel_booking_analysis.ipynb` (Week 3 section), with the full
confusion matrix and ROC curve saved to
`reports/fig_model_comparison.png`. Recall was weighted heavily in model
selection, since under-predicting cancellations (a false negative) causes
more downstream revenue leakage than a false positive.

## 6. Recommendations for the Revenue Management team

1. Introduce a tiered deposit/confirmation policy that scales with booking
   lead time rather than a blanket non-refundable policy.
2. Deploy the Random Forest risk score (`predicted_cancel_risk` column,
   already exported to `data/processed/hotel_bookings_dashboard_ready.csv`)
   to flag high-risk reservations for proactive outreach.
3. Test incremental ADR increases in peak months, informed by the
   seasonal demand curve in the dashboard.
4. Target retention email campaigns first at "Early Planner —
   Corporate/Contract" and "Early Planner — Leisure/Transient" segments,
   which show the highest cancellation rates.

## 7. Deliverables

- Cleaned dataset: `data/processed/hotel_bookings_clean.csv`
- Model-scored dataset: `data/processed/hotel_bookings_dashboard_ready.csv`
- Full analysis notebook: `notebooks/hotel_booking_analysis.ipynb`
- Chart exports: `reports/fig_*.png`
- Interactive dashboard: `dashboard/dashboard.html`
- Power BI rebuild guide: `dashboard/POWER_BI_GUIDE.md`
