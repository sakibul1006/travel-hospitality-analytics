# Rebuilding this dashboard in Power BI Desktop

`dashboard.html` gives you an interactive preview immediately, in any
browser, with no software install. To produce the actual `.pbix` file your
evaluator expects, follow these steps in Power BI Desktop against the same
cleaned data:

## 1. Get data

`Home → Get Data → Text/CSV` → select `data/processed/hotel_bookings_dashboard_ready.csv`.
This is the fully cleaned, feature-engineered, model-scored dataset exported
at the end of the notebook (Week 3/4).

## 2. Data model / types

In Power Query Editor, confirm these types (they usually auto-detect
correctly, but verify):
- `arrival_date_month` → Text, then **add a Sort-By-Column** for chronological
  ordering (create an `arrival_month_number` column via `Add Column →
  Conditional Column` mapping January→1 … December→12, then sort
  `arrival_date_month` by it).
- `is_canceled` → Whole Number (0/1) — keep as number so `AVERAGE` gives you
  cancellation rate directly.
- `adr`, `lead_time`, `predicted_cancel_risk` → Decimal Number.
- `deposit_type`, `customer_type`, `market_segment`, `hotel`, `country` → Text.

## 3. Measures (DAX)

Create a new Measures table (`Modeling → New Table`, name it `_Measures`) and
add:

```DAX
Total Bookings = COUNTROWS('hotel_bookings_dashboard_ready')

Cancellation Rate =
DIVIDE(SUM('hotel_bookings_dashboard_ready'[is_canceled]), [Total Bookings])

Avg ADR = AVERAGE('hotel_bookings_dashboard_ready'[adr])

Avg Lead Time = AVERAGE('hotel_bookings_dashboard_ready'[lead_time])

High Risk Bookings =
CALCULATE([Total Bookings], 'hotel_bookings_dashboard_ready'[predicted_cancel_risk] > 0.6)
```

## 4. Report page layout (matches `dashboard.html`)

| Visual | Type | Fields |
|---|---|---|
| KPI cards | Card (x5) | `Total Bookings`, `Cancellation Rate`, `Avg ADR`, `Avg Lead Time`, `High Risk Bookings` |
| Seasonal Demand vs. ADR | Line and clustered column chart | Axis: `arrival_date_month` (sorted); Column: `Total Bookings`; Line: `Avg ADR` |
| Cancellation by Lead Time | Clustered bar chart | Axis: a `lead_time_bucket` calculated column (bin `lead_time` into 0-7 / 8-30 / 31-90 / 91-180 / 180+); Value: `Cancellation Rate` |
| Deposit Type vs. Cancellation | Clustered column | Axis: `deposit_type`; Value: `Cancellation Rate` |
| Customer Type vs. Cancellation | Bar chart | Axis: `customer_type`; Value: `Cancellation Rate` |
| Top Source Markets | Donut chart | Legend: `country`; Value: `Total Bookings` (Top N filter = 8) |
| Customer Segments table | Table/Matrix | `customer_segment` (create via calculated column mirroring the notebook logic), `Total Bookings`, `Avg ADR`, `Cancellation Rate` |

To create `lead_time_bucket` and `customer_segment` as calculated columns in
Power BI (mirroring the notebook's Python logic):

```DAX
lead_time_bucket =
SWITCH(
    TRUE(),
    'hotel_bookings_dashboard_ready'[lead_time] <= 7, "0-7 days",
    'hotel_bookings_dashboard_ready'[lead_time] <= 30, "8-30 days",
    'hotel_bookings_dashboard_ready'[lead_time] <= 90, "31-90 days",
    'hotel_bookings_dashboard_ready'[lead_time] <= 180, "91-180 days",
    "180+ days"
)
```

(`customer_segment` is already exported as a column directly from the
notebook, so no DAX is required for it — just drag it in.)

## 5. Slicers

Add a slicer on `hotel` (City Hotel / Resort Hotel) and one on
`arrival_date_year`, matching the filter chips in `dashboard.html`.

## 6. Theme

Match the palette used in `dashboard.html` for visual consistency:
- Background: `#0E1A22` / `#132430`
- Accent (gold): `#C9A25D`
- Positive/teal: `#3FBFAE`
- Risk/coral: `#E4633B`

Import via `View → Themes → Browse for themes` with a custom JSON theme file,
or apply manually per visual under `Format → Fill`.

## 7. Publish

`File → Publish → Publish to Power BI` (requires a Power BI account) if you
need a shareable web link for your evaluator; otherwise submit the `.pbix`
file alongside this repository.
