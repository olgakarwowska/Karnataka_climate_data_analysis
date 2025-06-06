import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import seaborn as sns

rain_chik_imd = pd.read_csv(r'D:\MASTER_S2\WUR_COURSES\PERIOD_5\CACT\dataimd\rain\clipped_chikkaballapur_rain.csv', sep=',')
rain_chik_iped = pd.read_csv(r'D:\MASTER_S2\WUR_COURSES\PERIOD_5\CACT\IPED_Mean\clipped_chikkaballapur_IPED.csv', sep=',')

rain_raichur_imd = pd.read_csv(r'D:\MASTER_S2\WUR_COURSES\PERIOD_5\CACT\dataimd\rain\clipped_raichur_rain.csv', sep=',')
rain_raichur_iped = pd.read_csv(r'D:\MASTER_S2\WUR_COURSES\PERIOD_5\CACT\IPED_Mean\clipped_raichur_IPED.csv', sep=',')

#%% MERGED IPED-GRIDDED RAINFALL OF CHIKKABALLAPUR IN ONE CSV

# Convert 'time' columns to datetime if not already
rain_chik_imd['time'] = pd.to_datetime(rain_chik_imd['time'])
rain_chik_iped['time'] = pd.to_datetime(rain_chik_iped['time'])

# Filter IMD data to match IPED's date range (1991–2023)
imd_filtered_chik = rain_chik_imd[
    (rain_chik_imd['time'] >= '1991-01-01') & 
    (rain_chik_imd['time'] <= '2023-12-31')]

# Select only coordinates present in IPED data
coord_iped_chik = rain_chik_iped[['lat', 'lon']].drop_duplicates()
imd_filtered_chik = imd_filtered_chik.merge(coord_iped_chik, on=['lat', 'lon'])

# align both datasets for merge (optional: sort first)
rain_chik_iped_sorted = rain_chik_iped.sort_values(by=['lat', 'lon', 'time'])
imd_chik_filtered_sorted = imd_filtered_chik.sort_values(by=['lat', 'lon', 'time'])

# Merge IMD and IPED data on coordinates + time
combined_chik = pd.merge(
    imd_chik_filtered_sorted,
    rain_chik_iped_sorted,
    on=['lat', 'lon', 'time'],
    how='inner',
    suffixes=('_imd', '_iped'))

# Optional: drop unnecessary columns like spatial_ref
combined_chik = combined_chik.drop(columns=['spatial_ref_imd', 'spatial_ref_iped'], errors='ignore')

#%% MERGED IPED-GRIDDED RAINFALL OF RAICHUR IN ONE CSV

# Convert 'time' columns to datetime if not already
rain_raichur_imd['time'] = pd.to_datetime(rain_raichur_imd['time'])
rain_raichur_iped['time'] = pd.to_datetime(rain_raichur_iped['time'])

# Filter IMD data to match IPED's date range (1991–2023)
imd_filtered_raichur = rain_raichur_imd[
    (rain_raichur_imd['time'] >= '1991-01-01') & 
    (rain_raichur_imd['time'] <= '2023-12-31')]

# Select only coordinates present in IPED data
coord_iped_raichur = rain_raichur_iped[['lat', 'lon']].drop_duplicates()
imd_filtered_raichur = imd_filtered_raichur.merge(coord_iped_raichur, on=['lat', 'lon'])

# align both datasets for merge (optional: sort first)
rain_raichur_iped_sorted = rain_raichur_iped.sort_values(by=['lat', 'lon', 'time'])
imd_raichur_filtered_sorted = imd_filtered_raichur.sort_values(by=['lat', 'lon', 'time'])

# Merge IMD and IPED data on coordinates + time
combined_raichur = pd.merge(
    imd_raichur_filtered_sorted,
    rain_raichur_iped_sorted,
    on=['lat', 'lon', 'time'],
    how='inner',
    suffixes=('_imd', '_iped'))

# Optional: drop unnecessary columns like spatial_ref
combined_raichur = combined_raichur.drop(columns=['spatial_ref_imd', 'spatial_ref_iped'], errors='ignore')

#%% SAVE TO CSV THE COMBINED RAINFALL IMD AND IPED

combined_chik.to_csv("combined_rainfall_chikkaballapur_1991_2023.csv", index=False)
combined_raichur.to_csv("combined_rainfall_raichur_1991_2023.csv", index=False)


#%% BOXPLOT TOTAL ANNUAL RAINFALL IN RAICHUR AND CHIKKABALLAPUR OVER PERIODS 

# Load datasets
raichur = pd.read_csv(r'D:\MASTER_S2\WUR_COURSES\PERIOD_5\CACT\IPED_Mean\combined_rainfall_raichur_1991_2023.csv')
chik = pd.read_csv(r'D:\MASTER_S2\WUR_COURSES\PERIOD_5\CACT\IPED_Mean\combined_rainfall_chikkaballapur_1991_2023.csv')

# Add region labels
raichur['Region'] = 'Raichur'
chik['Region'] = 'Chikkaballapur'

# Combine processing into a function
def process_annual(df):
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    df['Location'] = df['lat'].astype(str) + ", " + df['lon'].astype(str)
    annual = df.groupby(['Region', 'Location', 'year'])[['rain', 'pcp']].sum().reset_index()
    long_df = annual.melt(id_vars=['Region', 'Location', 'year'], 
                          value_vars=['rain', 'pcp'], 
                          var_name='Type', value_name='TotalRain')
    long_df['Type'] = long_df['Type'].replace({'rain': 'IMD_rainfall', 'pcp': 'IPED_rainfall'})
    return long_df

# Process both
long_raichur = process_annual(raichur)
long_chik = process_annual(chik)

# Plot Raichur
plt.figure(figsize=(7, 5))
sns.boxplot(data=long_raichur, x='Location', y='TotalRain', hue='Type',
            palette={'IMD_rainfall': '#D55E00', 'IPED_rainfall': '#0072B2'})
plt.title("Raichur: Total Annual Rainfall Comparison (IMD vs IPED)", fontsize=14)
plt.ylabel("Total Annual Rainfall (mm)", fontsize=12)
plt.xlabel("Coordinate (lat, lon)", fontsize=12)
plt.xticks(rotation=0, ha="right", fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.tight_layout()
plt.grid(True)
plt.show()

# Plot Chikkaballapur
plt.figure(figsize=(7,5))
sns.boxplot(data=long_chik, x='Location', y='TotalRain', hue='Type',
            palette={'IMD_rainfall': '#D55E00', 'IPED_rainfall': '#0072B2'})
plt.title("Chikkaballapur: Total Annual Rainfall Comparison (IMD vs IPED)", fontsize=14)
plt.ylabel("Total Annual Rainfall (mm)", fontsize=12)
plt.xlabel("Coordinate (lat, lon)", fontsize=12)
plt.xticks(rotation=0, ha="right", fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.tight_layout()
plt.grid(True)
plt.show()

#%% SCATTER PLOT FROM EACH COORDINATES OF IMD AND OBSERVED RAINFALL DATA IN 1991-2023

# === Load and preprocess RAICHUR ===
raichur = pd.read_csv('combined_rainfall_raichur_1991_2023.csv')
raichur['time'] = pd.to_datetime(raichur['time'])
raichur = raichur.dropna(subset=['rain', 'pcp'])
raichur['year'] = raichur['time'].dt.year

# Sum rainfall per coordinate per year
annual_raichur = raichur.groupby(['year', 'lat', 'lon'])[['rain', 'pcp']].sum().reset_index()

# Prepare data for RAICHUR
x_r = annual_raichur['rain']   # Gridded
y_r = annual_raichur['pcp']    # IPED

# Statistics
r_r, p_r = pearsonr(x_r, y_r)
p_text_r = "<<0.05" if p_r < 0.001 else "<0.05" if p_r < 0.05 else f"{p_r:.3f}"
rmse_r = np.sqrt(mean_squared_error(x_r, y_r))
slope_r, intercept_r = np.polyfit(x_r, y_r, 1)
reg_line_r = np.poly1d([slope_r, intercept_r])
reg_label_r = f"Regression: y = {slope_r:.2f}x + {intercept_r:.2f}"

# Plot RAICHUR
plt.figure(figsize=(7, 5))
plt.scatter(x_r, y_r, color='#0072B2', label='All Coordinates', alpha=0.7)
plt.plot(np.sort(x_r), reg_line_r(np.sort(x_r)), color='black', linestyle='--', label=reg_label_r)
plt.text(0.95, 0.05, f"r = {r_r:.2f}\nRMSE = {rmse_r:.2f} mm\np = {p_text_r}",
         transform=plt.gca().transAxes, ha='right', va='bottom', fontsize=11,
         bbox=dict(facecolor='white', alpha=0.7))
plt.title("Raichur: IPED vs. IMD Rainfall (Annual Total)", fontsize=14)
plt.xlabel("IMD Total Annual Rainfall (mm)", fontsize=12)
plt.ylabel("IPED Total Annual Rainfall (mm)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# === Load and preprocess CHIKKABALLAPUR ===
chikka = pd.read_csv('combined_rainfall_chikkaballapur_1991_2023.csv')
chikka['time'] = pd.to_datetime(chikka['time'])
chikka = chikka.dropna(subset=['rain', 'pcp'])
chikka['year'] = chikka['time'].dt.year

# Sum rainfall per coordinate per year
annual_chikka = chikka.groupby(['year', 'lat', 'lon'])[['rain', 'pcp']].sum().reset_index()

# Prepare data for CHIKKABALLAPUR
x_c = annual_chikka['rain']   # Gridded
y_c = annual_chikka['pcp']    # IPED

# Statistics
r_c, p_c = pearsonr(x_c, y_c)
p_text_c = "<<0.05" if p_c < 0.001 else "<0.05" if p_c < 0.05 else f"{p_c:.3f}"
rmse_c = np.sqrt(mean_squared_error(x_c, y_c))
slope_c, intercept_c = np.polyfit(x_c, y_c, 1)
reg_line_c = np.poly1d([slope_c, intercept_c])
reg_label_c = f"Regression: y = {slope_c:.2f}x + {intercept_c:.2f}"

# Plot CHIKKABALLAPUR
plt.figure(figsize=(7, 5))
plt.scatter(x_c, y_c, color='#D55E00', label='All Coordinates', alpha=0.7)
plt.plot(np.sort(x_c), reg_line_c(np.sort(x_c)), color='black', linestyle='--', label=reg_label_c)
plt.text(0.95, 0.05, f"r = {r_c:.2f}\nRMSE = {rmse_c:.2f} mm\np = {p_text_c}",
         transform=plt.gca().transAxes, ha='right', va='bottom', fontsize=11,
         bbox=dict(facecolor='white', alpha=0.7))
plt.title("Chikkaballapur: IPED vs. IMD Rainfall (Annual Total)", fontsize=14)
plt.xlabel("IMD Total Annual Rainfall (mm)", fontsize=12)
plt.ylabel("IPED Total Annual Rainfall (mm)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#%% CORR AND RMSE IN RABI AND KHARIF SEASON
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load Data ===
combined_raichur = pd.read_csv(r'D:\MASTER_S2\WUR_COURSES\PERIOD_5\CACT\IPED_Mean\combined_rainfall_raichur_1991_2023.csv')
combined_chik = pd.read_csv(r'D:\MASTER_S2\WUR_COURSES\PERIOD_5\CACT\IPED_Mean\combined_rainfall_chikkaballapur_1991_2023.csv')

# === Define Season Function ===
def get_custom_season(month):
    if month in [11, 12, 1, 2, 3, 4]:
        return "Rabi"
    elif month in [6, 7, 8, 9, 10]:
        return "Kharif"
    else:
        return None  # Exclude May

season_order = ["Rabi", "Kharif"]

# === Organize Input ===
stations_data_coord = {
    "Raichur": combined_raichur,
    "Chikkaballapur": combined_chik
}

all_metrics_coords = []

# === Process Each Coordinate ===
for station_name, df in stations_data_coord.items():
    coords = df[['lat', 'lon']].drop_duplicates().values
    for lat, lon in coords:
        try:
            sub_df = df[(df['lat'] == lat) & (df['lon'] == lon)].copy()
            sub_df["DATE"] = pd.to_datetime(sub_df["time"])
            sub_df["Month"] = sub_df["DATE"].dt.month
            sub_df["Year"] = sub_df["DATE"].dt.year
            sub_df["Season"] = sub_df["Month"].apply(get_custom_season)

            # Adjust season year
            sub_df.loc[sub_df["Month"].isin([1, 2, 3, 4]), "Year"] -= 1
            sub_df.loc[sub_df["Month"].isin([11, 12]), "Year"] += 1
            sub_df = sub_df[sub_df["Season"].notna()]

            # Rename columns
            sub_df.rename(columns={"rain": "observed_rainfall", "pcp": "modelled_rainfall"}, inplace=True)

            # Group totals
            seasonal_total = sub_df.groupby(["Year", "Season"])[["observed_rainfall", "modelled_rainfall"]].sum().reset_index()
            seasonal_mean = seasonal_total.groupby("Season")[["observed_rainfall", "modelled_rainfall"]].mean().reindex(season_order)

            # Compute metrics
            metrics = []
            for season in season_order:
                sdata = seasonal_total[seasonal_total["Season"] == season]
                if not sdata.empty:
                    obs = sdata["observed_rainfall"]
                    mod = sdata["modelled_rainfall"]
                    r = obs.corr(mod)
                    rmse = np.sqrt(np.mean((obs - mod) ** 2))
                    perc_rmse = rmse / obs.mean() * 100
                    metrics.append({
                        "Station": station_name,
                        "Latitude": lat,
                        "Longitude": lon,
                        "Season": season,
                        "r": round(r, 2),
                        "RMSE": round(rmse, 2),
                        "%RMSE": round(perc_rmse, 2)})

            all_metrics_coords.extend(metrics)

            # === Plotting ===
            if not seasonal_mean.isnull().values.any():
                plt.figure(figsize=(9, 6))
                x_labels = season_order
                obs = seasonal_mean["observed_rainfall"]
                mod = seasonal_mean["modelled_rainfall"]

                plt.plot(x_labels, obs, label="Observed Rainfall", color='red', marker='o')
                plt.plot(x_labels, mod, label="Modelled Rainfall", color='blue', marker='o')

                for i, season in enumerate(season_order):
                    stat = next((m for m in metrics if m["Season"] == season), None)
                    if stat:
                        y = max(obs.iloc[i], mod.iloc[i])
                        plt.text(i, y + 10,
                                 f"r = {stat['r']:.2f}\nRMSE = {stat['RMSE']:.1f}\n%RMSE = {stat['%RMSE']:.1f}%",
                                 fontsize=10, ha='center')

                plt.ylim(0, max(obs.max(), mod.max()) + 100)
                plt.xlabel("Season", fontsize=13)
                plt.ylabel("Mean Seasonal Total Rainfall (mm)", fontsize=13)
                plt.title(f"{station_name} ({lat}, {lon})", fontsize=14)
                plt.xticks(fontsize=13)
                plt.yticks(fontsize=13)
                plt.legend(fontsize=13)
                plt.grid(True)
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Error processing {station_name} at ({lat}, {lon}): {e}")

# === Save Metrics Table ===
metrics_coords_df = pd.DataFrame(all_metrics_coords)
metrics_coords_df.to_excel("seasonal_metrics_by_coordinate.xlsx", index=False)
print("Completed. Metrics saved. Plots displayed.")