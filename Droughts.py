# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 12:32:07 2025

@author: jonas
"""

import xarray as xr
import geopandas as gpd
import rioxarray
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import matplotlib.dates as mdates
from scipy.stats import binomtest


#%% read in data, remove NaN values
rain = pd.read_csv("insert path to region file")

start_time = pd.Timestamp('1901-01-01')
rain['time']= pd.to_datetime(rain['time'])
rain['day_of_year'] = rain['time'].dt.dayofyear
#remove nan values
rain = rain.dropna(subset=['rain'])

#%% FREQUENCY OF DROUGHTS
# Thought process: aggregate data to yearly values --> create trendline (kind of moving average). --> create collumn in df with the value of the regression line at each year
#--> create collumn in df with 75% of the value of reg line. --> create collumn in df with binary if value of that year is lower than the 75% treshold == 1
#
#


#creation of yearly values dataset
rain['year']= ['time'].dt.year
yearly_rainfall = rain.groupby('year')['rain'].sum().reset_index()


yearly_rain_fit = np.polyfit(yearly_rainfall['year'], yearly_rainfall['rain'],1)
yearly_rain_trendline = np.polyval(yearly_rain_fit, yearly_rainfall['year'])

yearly_slope= yearly_rain_fit[0]
yearly_intercept = yearly_rain_fit[1]

trend_yearlyrain = f"y = {yearly_slope:.2f}x + {yearly_intercept:.2f}"

#CHANGE THAT the moving average changes along with the dataset

yearly_rainfall['moving average'] = (yearly_rainfall['year'] * yearly_slope + yearly_intercept)
yearly_rainfall['drought_treshold'] = (yearly_rainfall['moving average'] * 0.75)
yearly_rainfall['drought'] = (yearly_rainfall['rain'] < yearly_rainfall['drought_treshold']).astype(int)



yearly_rainfall['moving average'] = (yearly_rainfall['year'] * 5.72-7352.24)
yearly_rainfall['drought_treshold'] = (yearly_rainfall['moving average'] * 0.75)
yearly_rainfall['drought'] = (yearly_rainfall['rain'] < yearly_rainfall['drought_treshold']).astype(int)



######PLOTTING

plt.figure(figsize=(10, 5))
plt.scatter(yearly_rainfall["year"], yearly_rainfall["drought"], marker='o', color='#0072B2', zorder=3)
plt.ylabel("Drought", fontsize=12)
plt.xlabel("Year", fontsize=12)
plt.title("Drought occurence Raichur", fontsize=14)
plt.yticks([0, 1], ['False', 'True'])
plt.grid(True, zorder=0)
plt.tight_layout()
plt.show()

#%%
#solution to above using bionomial approach 


# Sort and reset
yearly_rainfall = yearly_rainfall.sort_values('year').reset_index(drop=True)

# Define window size
window_size = 30

#Step 1: Get the baseline p from the first 30 years or last 30 years
first_30 = yearly_rainfall.iloc[:window_size]
baseline_droughts = first_30['drought'].sum()
#last_30 = yearly_rainfall.iloc[-window_size:]
#baseline_droughts = last_30['drought'].sum()
p_null = baseline_droughts / window_size

# states the baseline drought probability in the console, adjust "first"/"last" acoordingly 
print(f"Baseline drought probability (first 30 years): {p_null:.2f}") 


#Binomial test on the moving 30-year windows starting AFTER the first 30 year(reference period)
results = []

for i in range(1, len(yearly_rainfall) - window_size + 1):  # start at 1 to skip the baseline period
    window_yearly_rainfall = yearly_rainfall.iloc[i:i + window_size]
    start_year = window_yearly_rainfall['year'].iloc[0]
    end_year = window_yearly_rainfall['year'].iloc[-1]
    
    num_droughts = window_yearly_rainfall['drought'].sum()
    expected_droughts = p_null * window_size
    test = binomtest(num_droughts, n=window_size, p=p_null, alternative='two-sided')
   
    # Prints a sentence stating when it is significantly different if it is more or less droughts than baseline
    if test.pvalue < 0.05:
        if num_droughts > expected_droughts:
            significance = "significantly MORE droughts"
        else:
            significance = "significantly FEWER droughts"
        print(f"{start_year}-{end_year}: {significance} (p = {test.pvalue:.4f})")
    else:
        significance = "not significant"
    
    results.append({
        'start_year': start_year,
        'end_year': end_year,
        'num_droughts': num_droughts,
        'p_value': test.pvalue,
        'expected_droughts': expected_droughts
    })

results_df = pd.DataFrame(results)


###PLOTTING

plt.figure(figsize=(12, 6))
plt.plot(results_df['start_year'], results_df['p_value'], marker='o')
plt.axhline(0.05, color='red', linestyle='--', label='Significance Level (0.05)')
# change plot title to first of last 30 years accordingly
plt.title('Binomial Test of Drought Frequency Raichur\nvs. Baseline (First 30 Years)') # change title based on which refernece window!
plt.xlabel('Start Year of 30-Year Window')
plt.ylabel('P-Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#### If you want to see all the p-values for each 30 year window you can download csv with following lines

# print(results_df)
# results_df.to_csv(r'C:pathwherewanttosave.csv')


#%%

## create groups based on starting year to quickly see extended periods of more or less droughts, 
#ONLY checks for significant different values compared to window BEWARE of signficantly more and less both being treated equally here.




#select significant results
significant_df = results_df[results_df['p_value'] < 0.05].copy()

# enforce start year as int
significant_df['start_year'] = significant_df['start_year'].astype(int)

# create groups of starting years
years_df = significant_df[['start_year']].drop_duplicates().sort_values('start_year').reset_index(drop=True)
years_df['gap'] = years_df['start_year'].diff() != 1
years_df['group'] = years_df['gap'].cumsum()

# create summary of the groups with the length of the group 
group_summary = years_df.groupby('group').agg(
    start_year=('start_year', 'min'),
    end_year=('start_year', 'max'),
    length=('start_year', 'count')
).reset_index(drop=True)

# print the results
print("Grouped consecutive start years of significant drought windows:")
print(group_summary)