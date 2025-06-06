#COMPARISON OF IMD GRIDDED DATA AND OBSERVATION DATA

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.lines as mlines


#%% BOXPLOT TOTAL RAINFALL OVER PERIODS (PERIOD OF DATA IS BASED ON AVAILABILITY EACH STATION)

# Set the working directory
path = r"D:\NANDA\MASTER_S2\WUR_COURSES\PERIOD_5\CACT\dataimd\gridded_stationobs_rain"
os.chdir(path)

# List all CSV files (MAKE SURE ALL THE CSV FILE LOCATED IN ONE FOLDER)
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]

# Prepare container
all_data = []

# Loop through each CSV file
for file in csv_files:
    try:
        df = pd.read_csv(file, index_col=0)
        df["DATE"] = pd.to_datetime(df["DATE"])
        df["Year"] = df["DATE"].dt.year

        # Sum rainfall per year
        annual_total = df.groupby("Year")[["observed_rainfall", "modelled_rainfall"]].sum()

        # Extract and shorten location name for label
        base_name = os.path.splitext(file)[0]  # remove .csv
        short_name = "_".join(base_name.split("_")[:3])  # keep only first 3 parts
        annual_total["Location"] = short_name

        # Rename columns
        annual_total = annual_total.rename(columns={
            "observed_rainfall": "Observed_rainfall",
            "modelled_rainfall": "IMD_rainfall"
        })

        # Reshape to long format
        long_df = annual_total.reset_index().melt(
            id_vars=["Year", "Location"],
            value_vars=["Observed_rainfall", "IMD_rainfall"],
            var_name="Type",
            value_name="TotalRain"
        )
        all_data.append(long_df)
    except Exception as e:
        print(f"Skipping {file}: {e}")

# Combine all into one DataFrame
combined_df = pd.concat(all_data, ignore_index=True)

# Get data range for each location
location_range = (
    combined_df.groupby("Location")["Year"]
    .agg(["min", "max"])
    .apply(lambda row: f"{row.name}\n({row['min']}–{row['max']})", axis=1)
)

# Map location name with data range
combined_df["Location_label"] = combined_df["Location"].map(location_range)

# Plot
plt.figure(figsize=(10, 7))
sns.boxplot(
    data=combined_df, x="Location_label", y="TotalRain", hue="Type",
    hue_order=["IMD_rainfall", "Observed_rainfall"],
    palette={"IMD_rainfall": "#D55E00", "Observed_rainfall": "#0072B2"})

plt.title("Comparison of Total Annual Rainfall of Observation and IMD Data in Raichur", fontsize=18)
plt.ylabel("Total Annual Rainfall (mm)", fontsize=12)
plt.xlabel("")
plt.xticks(rotation=0, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10, title_fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


#%% CORRELATION AND RMSE IN EACH SEASON (RABI AND KHARIF)

# Set the directory containing all CSV files
path = r"D:\MASTER_S2\WUR_COURSES\PERIOD_5\CACT\dataimd\gridded_stationobs_rain"
os.chdir(path)

csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]

# Define custom seasons
def get_custom_season(month):
    if month in [11, 12, 1, 2, 3, 4]:
        return "Rabi"
    elif month in [6, 7, 8, 9, 10]:
        return "Kharif"
    else:
        return None  # Exclude May

# Prepare table for metrics
all_metrics = []

for file in csv_files:
    try:
        df = pd.read_csv(file, index_col=0)
        df["DATE"] = pd.to_datetime(df["DATE"])
        df["Month"] = df["DATE"].dt.month
        df["Year"] = df["DATE"].dt.year
        df["Season"] = df["Month"].apply(get_custom_season)

        # Adjust year for season
        df.loc[df["Month"].isin([1, 2, 3, 4]), "Year"] -= 1
        df.loc[df["Month"].isin([11,12]), "Year"] += 1
        df = df[df["Season"].notna()]  # Keep only valid seasons

        # Group totals
        seasonal_total = df.groupby(["Year", "Season"])[["observed_rainfall", "modelled_rainfall"]].sum().reset_index()

        # Mean rainfall per season
        season_order = ["Rabi", "Kharif"]
        seasonal_mean = seasonal_total.groupby("Season")[["observed_rainfall", "modelled_rainfall"]].mean().reindex(season_order)

        # Metric calculations
        metrics = []
        for season in season_order:
            sdata = seasonal_total[seasonal_total["Season"] == season]
            obs = sdata["observed_rainfall"]
            mod = sdata["modelled_rainfall"]
            r = obs.corr(mod)
            rmse = np.sqrt(np.mean((obs - mod)**2))
            perc_rmse = rmse / obs.mean() * 100
            metrics.append({
                "Station": file.replace(".csv", ""),
                "Season": season,
                "r": round(r, 2),
                "RMSE": round(rmse, 2),
                "%RMSE": round(perc_rmse, 2)
            })

        all_metrics.extend(metrics)

        # Plotting (optional)
        plt.figure(figsize=(9, 6))
        x_labels = season_order
        obs = seasonal_mean["observed_rainfall"]
        mod = seasonal_mean["modelled_rainfall"]

        plt.plot(x_labels, obs, label="Observed Rainfall", color='red', marker='o')
        plt.plot(x_labels, mod, label="Modelled Rainfall", color='blue', marker='o')

        for i, stat in enumerate(metrics):
            y = max(obs.iloc[i], mod.iloc[i])
            plt.text(i, y + 10,
                     f"r = {stat['r']:.2f}\nRMSE = {stat['RMSE']:.1f}\n%RMSE = {stat['%RMSE']:.1f}%",
                     fontsize=10, ha='center')

        plt.ylim(0, 700)
        plt.xlabel("Season", fontsize=13)
        plt.ylabel("Mean Seasonal Total Rainfall (mm)", fontsize=13)
        plt.title(f"Seasonal Average Rainfall\n{file}", fontsize=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(fontsize=13)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Save all metrics to Excel
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_excel("seasonal_metrics_rabi_kharif.xlsx", index=False)


#%% SCATTER PLOT FROM EACH COORDINATES OF IMD AND OBSERVED RAINFALL DATA IN 2000-2016 

path = r"D:\MASTER_S2\WUR_COURSES\PERIOD_5\CACT\dataimd\gridded_stationobs_rain"
os.chdir(path)

# List all CSV files
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]

# Container
scatter_data = []

# Process each file
for file in csv_files:
    try:
        df = pd.read_csv(file, index_col=0)
        df["DATE"] = pd.to_datetime(df["DATE"])
        df["Year"] = df["DATE"].dt.year

        annual = df.groupby("Year")[["observed_rainfall", "modelled_rainfall"]].sum().reset_index()
        base_name = os.path.splitext(file)[0]
        location = "_".join(base_name.split("_")[:3])
        annual["Station"] = location

        annual = annual.rename(columns={
            "observed_rainfall": "Observed_rainfall",
            "modelled_rainfall": "Gridded_rainfall"})
        scatter_data.append(annual)

    except Exception as e:
        print(f"Skipping {file}: {e}")

# Combine data
scatter_data = pd.concat(scatter_data, ignore_index=True)

# Filter to uniform period 2000–2016
scatter_data = scatter_data[(scatter_data['Year'] >= 2000) & (scatter_data['Year'] <= 2016)]

# === Statistics ===
slope, intercept = np.polyfit(scatter_data["Observed_rainfall"], scatter_data["Gridded_rainfall"], 1)
equation = f"y = {slope:.2f}x + {intercept:.2f}"
r, p_value = pearsonr(scatter_data["Observed_rainfall"], scatter_data["Gridded_rainfall"])
rmse = np.sqrt(mean_squared_error(scatter_data["Observed_rainfall"], scatter_data["Gridded_rainfall"]))
p_text = "<<0.05" if p_value < 0.001 else "<0.05" if p_value < 0.05 else f"{p_value:.3f}"

# === Plot ===
plt.figure(figsize=(10, 7))

# Scatter points
plt.scatter(
    scatter_data["Observed_rainfall"],
    scatter_data["Gridded_rainfall"],
    color="#0072B2",
    alpha=0.7,
    s=50,
    label="All Coordinates")

# Regression line
sns.regplot(
    data=scatter_data,
    x="Observed_rainfall",
    y="Gridded_rainfall",
    scatter=False,
    color="black",
    line_kws={"linestyle": "dashed"},
    ci=None)

# Manual legend
reg_handle = mlines.Line2D([], [], color='black', linestyle='dashed', label=f"Regression Line ({equation})")
plt.legend(handles=[
    plt.Line2D([], [], marker='o', color='w', markerfacecolor="#0072B2", linestyle='', label='All Coordinates'),
    reg_handle
], fontsize=10, loc='upper left')

# Add metrics
plt.text(0.95, 0.05, f"r = {r:.2f}\nRMSE = {rmse:.2f}\np = {p_text}",
         transform=plt.gca().transAxes,
         ha='right', va='bottom',
         fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

# Labels
plt.xlabel("Observed Annual Total Rainfall (mm)", fontsize=12)
plt.ylabel("IMD Annual Total Rainfall (mm)", fontsize=12)
plt.title("Scatter Plot of Observed vs. IMD Rainfall per Station-Year (2000–2016)", fontsize=14)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
