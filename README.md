# Logistics Data Analysis and Visualization

Exploratory data analysis of a simulated logistics dataset (1,500 shipments, 2025) using Python: delivery times, shipment volumes, transport costs, on-time performance and bottlenecks.

## Contents
- `logistics_analysis.py` - simulates the data, cleans it, runs the EDA and saves all charts
- `logistics_dataset.csv` - the generated dataset
- `charts/` - the 7 visualizations (histograms, box plot, correlation heatmap, scatter, monthly trend, carrier/region view, violin plot)
- `Logistics_Data_Analysis_Report.docx` - full written report with methodology, code excerpts, insights and recommendations

## How to run
```
pip install -r requirements.txt
python logistics_analysis.py
```
A fixed random seed (42) makes results reproducible.

## Key findings
- Overall on-time delivery: 91.8%
- Air is about 12% of shipments but about 48% of total transport spend
- On-time rate drops from 94.2% to 84.6% in Q4 without a rise in volume
- Carrier CargoPrime is the weakest at 87% on time
- Sea freight is the least reliable mode against its SLA (71%)

Note: the data is synthetic; the Q4 slowdown and weak carrier were built in on purpose to test the analysis.
