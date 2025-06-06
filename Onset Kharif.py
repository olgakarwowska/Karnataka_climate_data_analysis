# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 11:52:26 2025

@author: jonas
"""
# This script regards chapter 3.2.3 of the report

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

#%% read in data, remove NaN values
rain = pd.read_csv("insert path to region file")

start_time = pd.Timestamp('1901-01-01')
rain['time']= pd.to_datetime(rain['time'])
rain['day_of_year'] = rain['time'].dt.dayofyear
#remove nan values
rain = rain.dropna(subset=['rain'])


#%% Effective onset Kharif

rain_mon_onset = rain[
    (rain["time"].dt.month.isin([6,7,8,9,10,11,12]))]


#Group dataset by time and mean rain
rain_p_d_mon_onset = rain_mon_onset.groupby("time")["rain"].mean().reset_index(name="daily_rain")
rain_p_d_mon_onset['year']= rain_p_d_mon_onset['time'].dt.year

# create collumn for start monsoon
rain_p_d_mon_onset['onset Kharif'] = 0

for year, group in rain_p_d_mon_onset.groupby('year'):
    group= group.sort_values('time').copy()
    group['cumulative_rain']= group['daily_rain'].cumsum()
    
    treshold_day = group[group['cumulative_rain']>70].head(1)
    
    if not treshold_day.empty: 
        rain_p_d_mon_onset.loc[treshold_day.index[0], 'onset Kharif'] = 1


rain_mon_onset= rain_mon_onset.merge(
    rain_p_d_mon_onset[['time', 'onset Kharif']], on='time', how = 'left'
    )
rain_mon_onset['onset Kharif']= rain_mon_onset['onset Kharif'].fillna(0).astype(int)



# Filter rows where start == 1
onset_days = rain_mon_onset[rain_mon_onset['onset Kharif'] == 1].copy()

# Compute day of year
onset_days['day'] = onset_days['time'].dt.dayofyear
onset_days['year'] = onset_days['time'].dt.year




# Prepare x and y
x = onset_days['year']
y = onset_days['day']

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Create trendline function
trendline = slope * x + intercept

if p_value < 0.01:
    p_str = "p << 0.05"
elif p_value< 0.05:
    p_str = "p < 0.05"
else:
    p_str = f"p = {p_value:.3f}"

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(x, y, color='#0072B2', label='Onset Day', zorder=3)
plt.plot(x, trendline, color='#000000', linestyle='--',
         label=f'Trendline: y = {slope:.2f}x + {intercept:.2f}\n({p_str})')

# Labels and legend
plt.title('Onset Kharif Chikkaballapur (First Day Cumulative Rain > 70mm)', fontsize='16')
plt.xlabel('Year', fontsize= '12')
plt.ylabel('Day of Year', fontsize='12')
plt.grid(True, zorder=0)

plt.legend()
plt.tight_layout()
plt.show()
