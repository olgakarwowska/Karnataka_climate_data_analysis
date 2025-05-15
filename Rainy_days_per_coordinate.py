# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% Choose location
location = "Chikkaballapur"  # Options: "Chikkaballapur" or "Raichur"

#%% Load data
if location == "Raichur":
    rain_data = pd.read_csv("C:\\Users\\Administrator\\Documents\\MEE\\Design of Climate Change Mitigation and Adaptation Strategies\\Clipped data of Raichur\\clipped_raichur_rain.csv")
    lat, lon = 16.5, 77 # Coordinate options: lat = 16.25 or 16.5 - lon = 76.5, 76.75 or 77
elif location == "Chikkaballapur":
    rain_data = pd.read_csv("C:\\Users\\Administrator\\Documents\\MEE\\Design of Climate Change Mitigation and Adaptation Strategies\\Clipped data of Chikkaballapur\\clipped_chikkaballapur_rain_5_points.csv")
    lat, lon = 13.5, 77.5 # Coordinate options: lat = 13.5 or 13.75 - lon = 77.5, 77.75 or 78

#%% Set datetime and clean
rain_data["date_time"] = pd.to_datetime(rain_data["time"])
del rain_data["time"]

#%% Filter specific coordinates
rain_coor = rain_data[(rain_data["lat"] == lat) & (rain_data["lon"] == lon)]

#%% Plot rainfall over time
rain_coor.plot(x="date_time", y="rain")
plt.title(f"Rainfall at ({lat}, {lon}) - {location}")
plt.xlabel("Time (years)")
plt.ylabel("Rainfall (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Rainy days analysis (rain ≥ 2.5 mm)
rainy_days = rain_coor[rain_coor["rain"] >= 2.5].copy()
rainy_days["year"] = rainy_days["date_time"].dt.year
rainy_days_per_year = rainy_days.groupby("year")["date_time"].nunique().reset_index()
rainy_days_per_year.columns = ["year", "amount_of_days"]

# Fit trendline
year = rainy_days_per_year["year"]
days = rainy_days_per_year["amount_of_days"]
z = np.polyfit(year, days, 1)
p = np.poly1d(z)
trend = p(year)

#%% Plot rainy days with trendline
plt.bar(year, days, label="Rainy Days")
plt.plot(year, trend, color="red", linewidth=2, label="Trendline")
plt.title(f"Rainy Days per Year - {location} - coordinates {lat, lon}")
plt.xlabel("Year")
plt.ylabel("Number of Rainy Days")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
