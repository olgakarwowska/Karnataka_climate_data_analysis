# -*- coding: utf-8 -*-
"""
Created on Thu May 15 11:16:03 2025

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% Function to load and prepare dataset
def load_rain_data(path):
    df = pd.read_csv(path)
    df["date_time"] = pd.to_datetime(df["time"])
    df.drop(columns=["time"], inplace=True)
    return df

#%% Load datasets
raichur_path = "C:\\Users\\Administrator\\Documents\\MEE\\Design of Climate Change Mitigation and Adaptation Strategies\\Clipped data of Raichur\\clipped_raichur_rain.csv"
chikkaballapur_path = "C:\\Users\\Administrator\\Documents\\MEE\\Design of Climate Change Mitigation and Adaptation Strategies\\Clipped data of Chikkaballapur\\clipped_chikkaballapur_rain_5_points.csv"

rain_raichur = load_rain_data(raichur_path)
rain_chikkaballapur = load_rain_data(chikkaballapur_path)

#%% Optional: Use average over all coordinates
def average_daily_rainfall(df):
    """
    Returns a DataFrame with average daily rainfall across all locations.
    """
    df_avg = df.groupby("date_time")["rain"].mean().reset_index()
    return df_avg

# Get averaged rainfall data
rain_raichur_avg = average_daily_rainfall(rain_raichur)
rain_chik_avg = average_daily_rainfall(rain_chikkaballapur)

#%% Function to plot rainfall time series
def plot_rain_timeseries(df, location_name="Unknown"):
    df = df.sort_values('date_time')
    df.plot(x="date_time", y="rain", title=f"Average Daily Rainfall - {location_name}")
    plt.xlabel("Year")
    plt.ylabel("Rainfall (mm)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage:
plot_rain_timeseries(rain_raichur_avg, location_name="Raichur")
plot_rain_timeseries(rain_chik_avg, location_name="Chikkaballapur")

#%% Function to plot rainy days per year + trendline
def plot_rainy_days_trend(df, location_name="Unknown", threshold=2.5):
    df = df.copy()
    df = df[df["rain"] >= threshold]
    df["year"] = df["date_time"].dt.year

    rainy_days_per_year = df.groupby("year")["date_time"].nunique().reset_index()
    rainy_days_per_year.columns = ["year", "amount_of_days"]

    year = rainy_days_per_year["year"]
    days = rainy_days_per_year["amount_of_days"]

    # Trendline
    z = np.polyfit(year, days, 1)
    p = np.poly1d(z)
    trend = p(year)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(year, days, label="Rainy Days")
    plt.plot(year, trend, color="red", linewidth=2, label="Trendline")
    plt.title(f"Rainy Days per Year - {location_name}")
    plt.xlabel("Year")
    plt.ylabel(f"Days with Rain (≥ {threshold} mm)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage:
plot_rainy_days_trend(rain_raichur_avg, location_name="Raichur")
plot_rainy_days_trend(rain_chik_avg, location_name="Chikkaballapur")