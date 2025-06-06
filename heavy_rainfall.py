#chikkaballapur

import xarray as xr
import geopandas as gpd
import rioxarray
import os
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

rain_chikkaballapur = pd.read_csv('clipped_chikkaballapur_rain.csv', sep=',')
start_time = pd.Timestamp('1901-01-01')
rain_chikkaballapur['time']= pd.to_datetime(rain_chikkaballapur['time'])
rain_chikkaballapur = rain_chikkaballapur.dropna(subset=['rain'])
#rain_raichur.to_csv('rain_raichur.csv', index=False)


#%% MONTHLY SUM RAINFALL OVER 1901-2024 (6 PLOT SEPEARTELY FOR EACH DATAPOINT)

import pandas as pd
import matplotlib.pyplot as plt
import calendar

# === Load the rainfall data ===
rain_chikkaballapur = pd.read_csv('clipped_chikkaballapur_rain.csv', sep=',')

# Convert 'time' to datetime and extract month
rain_chikkaballapur['time'] = pd.to_datetime(rain_chikkaballapur['time'])
rain_chikkaballapur['month'] = rain_chikkaballapur['time'].dt.month

# Get unique lon-lat points
unique_points = rain_chikkaballapur[['lon', 'lat']].drop_duplicates()

# Loop through each grid point and plot
for i, (idx, point) in enumerate(unique_points.iterrows()):
    lon = point['lon']
    lat = point['lat']

    # Filter data for this point
    point_data = rain_chikkaballapur[(rain_chikkaballapur['lon'] == lon) & (rain_chikkaballapur['lat'] == lat)]

    # Group by month and calculate average
    monthly_sum= point_data.groupby('month')['rain'].sum()

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 13), monthly_sum.values, marker='o', linestyle='-', color='blue')
    plt.xticks(range(1, 13), calendar.month_abbr[1:])
    plt.title(f"Monthly Sum Rainfall (1901–2024)\nPoint ({lon}, {lat})")
    plt.xlabel("Month")
    plt.ylabel("Average Rainfall (mm)")
    plt.ylim(0,21000)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
#%% MONTHLY SUM RAINFALL OVER 1902-2024  (1 PLOT CONTAIN OF 6 LINE OF EACH DATAPOINT)
plt.figure(figsize=(12, 6))

# Loop through each point and plot on the same axes
for i, (_, point) in enumerate(unique_points.iterrows()):
    lon, lat = point['lon'], point['lat']
    point_data = rain_chikkaballapur[(rain_chikkaballapur['lon'] == lon) & (rain_chikkaballapur['lat'] == lat)]
#    
    # Monthly average
    monthly_sum = point_data.groupby('month')['rain'].sum() 
    # Plot this line
    plt.plot(
        range(1, 13),
        monthly_sum.values,
        marker='o',
        label=f"({lon:.2f}, {lat:.2f})"
    )

# Final plot formatting
plt.xticks(range(1, 13), calendar.month_abbr[1:])
plt.ylim(0, 8)
plt.xlabel("Month")
plt.ylabel("Sum Rainfall (mm)")
plt.title("Monthly Sum Rainfall (1901–2024) for 6 Grid Points")
plt.legend(title="Grid Point (lon, lat)")
plt.grid(True)
plt.tight_layout()
plt.show()

#%% MONTHLY AVG RAINFALL OVER 1902-2024  (1 PLOT CONTAIN OF 6 LINE OF EACH DATAPOINT)

import pandas as pd

# Ensure datetime
rain_chikkaballapur['time'] = pd.to_datetime(rain_chikkaballapur['time'])

# Extract year and month
rain_chikkaballapur['year'] = rain_chikkaballapur['time'].dt.year
rain_chikkaballapur['month'] = rain_chikkaballapur['time'].dt.month

#calculate sum rain every month every year in each datapoint
monthly_sum_rain = (
    rain_chikkaballapur
    .groupby(['lat', 'lon', 'year', 'month'])['rain']
    .sum()
    .reset_index()
    .rename(columns={'rain': 'sum_monthly_rain'})
)

# Calculate monthly average rainfall over 1901-2024 for each point
long_term_monthly_avg = (
    monthly_sum_rain
    .groupby(['lat', 'lon', 'month'])['sum_monthly_rain']
    .mean()
    .reset_index()
    .rename(columns={'sum_monthly_rain': 'long_term_sum_rain'}))


plt.figure(figsize=(10, 6))
# Plot one line per unique (lat, lon)
for _, point in long_term_monthly_avg[['lat', 'lon']].drop_duplicates().iterrows():
    lat, lon = point['lat'], point['lon']
    point_data = long_term_monthly_avg[
        (long_term_monthly_avg['lat'] == lat) &
        (long_term_monthly_avg['lon'] == lon)
    ]
    plt.plot(
        point_data['month'],
        point_data['long_term_sum_rain'],
        marker='o',
       label=f"({lon}, {lat})")
plt.xticks(ticks=range(1, 13), labels=calendar.month_abbr[1:])
plt.xlabel("Month")
plt.ylabel("Long-Term Sum Rainfall (mm)")
plt.title("Monthly Rainfall Pattern (1901–2024) for 6 Grid Points")
plt.grid(True)
plt.legend(title="Grid Point (lon, lat)")
plt.tight_layout()
plt.show()


#%% COUNTING HEAVY RAIN

# Step 1: Add a flag for heavy rain days 
#rain_chikkaballapur['is_heavy_rain'] = rain_chikkaballapur['rain'].between(64.5, 115.5) #for 64.5-115.5 mm
rain_chikkaballapur['is_heavy_rain'] = rain_chikkaballapur['rain']>25 #for 64.5-115.5 mm

# Step 2: Group by month, year, lat, lon and count heavy rain days
heavy_rain_days = (rain_chikkaballapur
    .groupby(['lat', 'lon', 'year', 'month'])['is_heavy_rain']
    .sum().reset_index()
    .rename(columns={'is_heavy_rain': 'heavy_rain_days'}))

heavy_rain_yearly = (
    heavy_rain_days
    .groupby(['lat', 'lon', 'year'])['heavy_rain_days']
    .sum()
    .reset_index())

# Step 2: Create one combined plot
plt.figure(figsize=(12, 6))

# Loop through each point and plot
for _, point in heavy_rain_yearly[['lat', 'lon']].drop_duplicates().iterrows():
    lat, lon = point['lat'], point['lon']
    point_data = heavy_rain_yearly[
        (heavy_rain_yearly['lat'] == lat) & (heavy_rain_yearly['lon'] == lon)
    ]
    plt.plot(
        point_data['year'],
        point_data['heavy_rain_days'],
        marker='o',
        markersize=2,
        linewidth=1,
        label=f"({lon}, {lat})")

# Formatting
plt.xlabel("Year")
plt.ylabel("Heavy Rain Days")
plt.title("Yearly Heavy Rain Days\nfor Each Grid Point (1901–2024)")
plt.legend(title="Grid Point (lon, lat)")
plt.grid(True)
plt.tight_layout()
plt.show()

#%% PLOT HEAVY RAIN EVERY YEAR IN EACH DATA POINT (1 PLOT WITH 6 SUBPLOT)
import matplotlib.pyplot as plt

# Get unique grid points
unique_points = heavy_rain_yearly[['lat', 'lon']].drop_duplicates()

# Create subplots: 3 rows x 2 columns
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(14, 10), sharex=True, sharey=True)
axs = axs.flatten()

# Plot each point in its own subplot
for i, (_, point) in enumerate(unique_points.iterrows()):
    lat, lon = point['lat'], point['lon']
    point_data = heavy_rain_yearly[
        (heavy_rain_yearly['lat'] == lat) & (heavy_rain_yearly['lon'] == lon)
    ]

    axs[i].plot(
        point_data['year'],
        point_data['heavy_rain_days'],
        marker='o',
        markersize=2,
        linewidth=1,
        label=f"({lon}, {lat})"
    )
    axs[i].set_title(f"Grid: ({lon}, {lat})")
    axs[i].grid(True)
    axs[i].legend(fontsize='small')

# Set shared labels
fig.suptitle("Yearly Heavy Rain Days (64.5–115.5 mm) for Each Grid Point (1901–2024)", fontsize=14)
fig.supxlabel("Year")
fig.supylabel("Heavy Rain Days")
fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for title
plt.show()



#%% PLOT HEAVY RAINFALL DAYS IN ALL DATAPOINTS EVERY YEAR 64.5-115.5 mm or >25,,
import matplotlib.pyplot as plt

# Step 1: Group by year only (sum across all points)
total_heavy_rain_per_year = (
    heavy_rain_yearly
    .groupby('year')['heavy_rain_days']
    .sum()
    .reset_index()
)

# Step 2: Plot
plt.figure(figsize=(10, 5))
plt.plot(
    total_heavy_rain_per_year['year'],
    total_heavy_rain_per_year['heavy_rain_days'],
    marker='o',
    linewidth=2,
    color='blue'
)

# Formatting
plt.xlabel("Year")
plt.ylabel("Total Heavy Rain Days (All Points)")
plt.title("Yearly Total Heavy Rain Days Across All Grid Points")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Data
x = total_heavy_rain_per_year['year']
y = total_heavy_rain_per_year['heavy_rain_days']

# Linear regression
coeffs = np.polyfit(x, y, 1)
trend = np.poly1d(coeffs)
slope, intercept = coeffs
equation = f"Trendline (y = {slope:.4f}x + {intercept:.2f})"

# Plot
plt.figure(figsize=(12, 7))
plt.plot(x, y, linewidth=2, color='blue', label='Heavy Rain Days (>25 mm)')
plt.plot(x, trend(x), color='red', linestyle='--', linewidth=2, label=equation)

# No equation text on plot — removed this line:
# plt.text(...)

plt.xlabel("Year", fontsize=13)
plt.ylabel("Total Heavy Rain Days (Day)", fontsize=13)
plt.xticks(ticks=range(x.min()-1, x.max() + 1, 10), fontsize=13)
plt.yticks(fontsize=12)  # Change 12 to your desired size
plt.title("Yearly Total Heavy Rain Days in Chikkaballapur", fontsize=18, pad =15)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

