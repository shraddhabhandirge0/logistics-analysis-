"""Logistics data simulation, EDA and visualisation."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
sns.set_theme(style="whitegrid", context="notebook")
OUT = "charts"
import os
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# 1. DATA SIMULATION
# ---------------------------------------------------------------
N = 1500
modes = ["Road", "Rail", "Air", "Sea"]
mode_p = [0.55, 0.15, 0.12, 0.18]
regions = ["North", "South", "East", "West", "Central"]
carriers = ["FastTrack", "SwiftLine", "CargoPrime", "BlueFleet"]

df = pd.DataFrame({
    "shipment_id": [f"SH{1000+i}" for i in range(N)],
    "ship_date": pd.to_datetime("2025-01-01")
                 + pd.to_timedelta(np.random.randint(0, 365, N), unit="D"),
    "mode": np.random.choice(modes, N, p=mode_p),
    "region": np.random.choice(regions, N),
    "carrier": np.random.choice(carriers, N),
})
df["distance_km"] = np.where(
    df["mode"] == "Road", np.random.gamma(4, 120, N),
    np.where(df["mode"] == "Rail", np.random.gamma(5, 220, N),
    np.where(df["mode"] == "Air", np.random.gamma(6, 450, N),
             np.random.gamma(6, 900, N)))).round(0).clip(30, 9000)
df["weight_kg"] = np.random.lognormal(6.0, 0.8, N).round(0).clip(20, 8000)
df["volume_units"] = (df["weight_kg"] / 25 + np.random.normal(0, 4, N)).clip(1, None).round(0)

speed = {"Road": 55, "Rail": 45, "Air": 600, "Sea": 30}  # km/h effective
handling = {"Road": 6, "Rail": 14, "Air": 8, "Sea": 36}   # hours fixed
df["delivery_hours"] = (df["distance_km"] / df["mode"].map(speed)
                        + df["mode"].map(handling)
                        + np.random.gamma(2, 3, N)).round(1)
# congestion in peak months (Oct-Dec) and weak carrier
peak = df["ship_date"].dt.month.isin([10, 11, 12])
df.loc[peak, "delivery_hours"] *= np.random.uniform(1.10, 1.35, peak.sum())
df.loc[df["carrier"] == "CargoPrime", "delivery_hours"] *= 1.12
df["delivery_hours"] = df["delivery_hours"].round(1)

rate = {"Road": 0.9, "Rail": 0.55, "Air": 1.8, "Sea": 0.28}  # cost per tonne-km-ish
df["transport_cost"] = (df["distance_km"] * df["mode"].map(rate) * (df["weight_kg"] / 400) ** 0.7
                        + np.random.normal(0, 40, N)).clip(50, None).round(2)
df["cost_per_kg"] = (df["transport_cost"] / df["weight_kg"]).round(3)

sla = {"Road": 40, "Rail": 70, "Air": 30, "Sea": 280}
df["sla_hours"] = df["mode"].map(sla)
df["on_time"] = (df["delivery_hours"] <= df["sla_hours"]).astype(int)
df["month"] = df["ship_date"].dt.month
df["month_name"] = df["ship_date"].dt.strftime("%b")

# inject a few missing values / duplicates for cleaning demo
idx = np.random.choice(df.index, 25, replace=False)
df.loc[idx, "weight_kg"] = np.nan
df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)

# ---------------------------------------------------------------
# 2. CLEANING
# ---------------------------------------------------------------
stats = {}
stats["raw_rows"] = len(df)
stats["missing_weight"] = int(df["weight_kg"].isna().sum())
stats["duplicates"] = int(df.duplicated(subset="shipment_id").sum())
df = df.drop_duplicates(subset="shipment_id")
df["weight_kg"] = df.groupby("mode")["weight_kg"].transform(lambda s: s.fillna(s.median()))
df["cost_per_kg"] = (df["transport_cost"] / df["weight_kg"]).round(3)
stats["clean_rows"] = len(df)

# ---------------------------------------------------------------
# 3. EDA
# ---------------------------------------------------------------
num = ["distance_km", "weight_kg", "volume_units", "delivery_hours",
       "transport_cost", "cost_per_kg"]
desc = df[num].describe().T
desc["median"] = df[num].median()
desc["skew"] = df[num].skew()
stats["describe"] = desc.round(2).reset_index().to_dict("records")
stats["corr"] = df[num].corr().round(2).to_dict()
stats["on_time_rate"] = round(df["on_time"].mean() * 100, 1)

mode_tbl = df.groupby("mode").agg(
    shipments=("shipment_id", "count"),
    avg_hours=("delivery_hours", "mean"),
    avg_cost=("transport_cost", "mean"),
    avg_cost_kg=("cost_per_kg", "mean"),
    on_time=("on_time", "mean")).round(2)
mode_tbl["on_time"] = (mode_tbl["on_time"] * 100).round(1)
stats["mode_tbl"] = mode_tbl.reset_index().to_dict("records")

car_tbl = df.groupby("carrier").agg(
    shipments=("shipment_id", "count"),
    avg_hours=("delivery_hours", "mean"),
    on_time=("on_time", "mean")).round(2)
car_tbl["on_time"] = (car_tbl["on_time"] * 100).round(1)
stats["car_tbl"] = car_tbl.reset_index().to_dict("records")

reg_tbl = df.groupby("region").agg(
    shipments=("shipment_id", "count"),
    avg_hours=("delivery_hours", "mean"),
    on_time=("on_time", "mean")).round(2)
reg_tbl["on_time"] = (reg_tbl["on_time"] * 100).round(1)
stats["reg_tbl"] = reg_tbl.reset_index().to_dict("records")

peak_flag = df["month"].isin([10, 11, 12])
stats["peak_on_time"] = round(df[peak_flag]["on_time"].mean() * 100, 1)
stats["offpeak_on_time"] = round(df[~peak_flag]["on_time"].mean() * 100, 1)
stats["peak_hours"] = round(df[peak_flag]["delivery_hours"].mean(), 1)
stats["offpeak_hours"] = round(df[~peak_flag]["delivery_hours"].mean(), 1)

# outliers (IQR) on cost_per_kg
q1, q3 = df["cost_per_kg"].quantile([.25, .75])
iqr = q3 - q1
out = df[df["cost_per_kg"] > q3 + 1.5 * iqr]
stats["cost_outliers"] = int(len(out))
stats["cost_outlier_air_share"] = round((out["mode"] == "Air").mean() * 100, 1)

# regression for cost drivers (log-log with mode dummies)
import numpy.linalg as la
D = pd.get_dummies(df["mode"], drop_first=False).drop(columns="Road").astype(float)
X = np.column_stack([np.ones(len(df)), np.log(df["distance_km"]), np.log(df["weight_kg"]), D.values])
y = np.log(df["transport_cost"])
beta, *_ = la.lstsq(X, y, rcond=None)
pred = X @ beta
r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
names = ["intercept", "ln_distance", "ln_weight"] + list(D.columns)
stats["ols"] = {n: round(float(b), 3) for n, b in zip(names, beta)}
stats["ols"]["r2"] = round(float(r2), 3)

# ---------------------------------------------------------------
# 4. VISUALISATIONS
# ---------------------------------------------------------------
pal = sns.color_palette("deep")

# Fig 1 - histograms
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, c, t in zip(ax, ["delivery_hours", "transport_cost", "weight_kg"],
                   ["Delivery time (hours)", "Transport cost", "Shipment weight (kg)"]):
    sns.histplot(df[c], kde=True, ax=a, color=pal[0], bins=35)
    a.axvline(df[c].mean(), color="crimson", ls="--", label=f"mean {df[c].mean():.0f}")
    a.axvline(df[c].median(), color="green", ls=":", label=f"median {df[c].median():.0f}")
    a.set_title(t); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_hist.png", dpi=150); plt.close()

# Fig 2 - box plot delivery time by mode
fig, ax = plt.subplots(figsize=(8, 4.5))
order = ["Air", "Road", "Rail", "Sea"]
sns.boxplot(data=df, x="mode", y="delivery_hours", order=order, hue="mode", legend=False, palette="deep", ax=ax)
ax.set_yscale("log"); ax.set_title("Delivery time by transport mode (log scale)")
ax.set_ylabel("Delivery hours (log)"); ax.set_xlabel("")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_box_mode.png", dpi=150); plt.close()

# Fig 3 - correlation heatmap
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation matrix of key variables")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_corr.png", dpi=150); plt.close()

# Fig 4 - scatter cost vs distance
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="distance_km", y="transport_cost", hue="mode",
                size="weight_kg", sizes=(10, 120), alpha=.6, ax=ax)
ax.set_title("Transport cost vs distance (bubble size = weight)")
ax.legend(fontsize=7, loc="upper left", ncol=2)
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_scatter.png", dpi=150); plt.close()

# Fig 5 - monthly trend
m = df.groupby("month").agg(vol=("shipment_id", "count"),
                            hrs=("delivery_hours", "mean"),
                            ot=("on_time", "mean")).reset_index()
m["ot"] *= 100
fig, ax1 = plt.subplots(figsize=(9, 4.5))
ax1.bar(m["month"], m["vol"], color=pal[0], alpha=.45, label="Shipments")
ax1.set_ylabel("Shipments"); ax1.set_xlabel("Month"); ax1.set_xticks(range(1, 13))
ax2 = ax1.twinx()
ax2.plot(m["month"], m["ot"], color="crimson", marker="o", label="On-time %")
ax2.set_ylabel("On-time delivery (%)"); ax2.grid(False)
ax1.set_title("Monthly shipment volume and on-time performance")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower left")
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_trend.png", dpi=150); plt.close()

# Fig 6 - carrier on-time bars + region heatmap
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
c = df.groupby("carrier")["on_time"].mean().mul(100).sort_values()
sns.barplot(x=c.values, y=c.index, hue=c.index, legend=False, ax=ax[0], palette="deep")
ax[0].set_title("On-time rate by carrier (%)"); ax[0].set_xlabel("%"); ax[0].set_ylabel("")
for i, v in enumerate(c.values):
    ax[0].text(v + .5, i, f"{v:.1f}", va="center")
pv = df.pivot_table(index="region", columns="mode", values="delivery_hours", aggfunc="mean")
sns.heatmap(pv, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax[1])
ax[1].set_title("Avg delivery hours: region x mode")
plt.tight_layout(); plt.savefig(f"{OUT}/fig6_carrier_region.png", dpi=150); plt.close()

# Fig 7 - cost per kg by mode (violin)
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.violinplot(data=df, x="mode", y="cost_per_kg", order=["Sea", "Rail", "Road", "Air"],
               hue="mode", legend=False, palette="deep", cut=0, ax=ax)
ax.set_title("Cost per kg by transport mode"); ax.set_xlabel("")
plt.tight_layout(); plt.savefig(f"{OUT}/fig7_violin.png", dpi=150); plt.close()

df.to_csv("logistics_dataset.csv", index=False)
json.dump(stats, open("stats.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in stats.items() if k not in ("describe", "corr")}, indent=1, default=str))
print(pd.DataFrame(stats["describe"])[["index","mean","50%","std","min","max","skew"]])
print(pd.DataFrame(stats["corr"]))
