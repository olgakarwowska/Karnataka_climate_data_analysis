# -*- coding: utf-8 -*-
"""
Created on Thu May 22 12:23:02 2025

@author: Administrator
"""

#%% Libraries
import pandas as pd
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.stats import binomtest

#%% Loading the rain datasets

#Enter your own path to rainfall data of Raichur and Chikkaballapur here
raichur_path = #enter own csv file 
chikkaballapur_path = #enter own csv file

def load_rain_data(path):
    df = pd.read_csv(path)
    df["date_time"] = pd.to_datetime(df["time"])
    df.drop(columns=["time"], inplace=True)
    return df

rain_raichur = load_rain_data(raichur_path)
rain_chikkaballapur = load_rain_data(chikkaballapur_path)

#%% Average daily rainfall data over all coordinates 
def average_daily_rainfall(df):
    df_avg = df.groupby("date_time")["rain"].mean().reset_index()
    return df_avg

# Run average daily rainfall function
rain_raich = average_daily_rainfall(rain_raichur)
rain_chik = average_daily_rainfall(rain_chikkaballapur)

#%% Function to compute dry spells
def compute_dry_spells(df):
    df = df.copy()
    df.set_index("date_time", inplace=True)

    # Add year and dry day flags
    df['year'] = df.index.year
    df['is_dry'] = df['rain'] <= 2.5

    # Reset streak on dry/wet changes OR year changes
    df['streak_change'] = (df['is_dry'] != df['is_dry'].shift()) | (df['year'] != df['year'].shift())
    df['streak_id'] = df['streak_change'].cumsum()

    # Group by streak id
    grouped = df.groupby('streak_id')
    streaks = grouped.agg({
        'is_dry': 'first',
        'rain': 'count'
    }).rename(columns={'rain': 'length'}).reset_index(drop=True)

    # Add start and end dates of each streak
    streaks['start_date'] = grouped.apply(lambda x: x.index[0]).values
    streaks['end_date'] = grouped.apply(lambda x: x.index[-1]).values

    # Filter for dry streaks of at least 20 days
    dry_streaks_df = streaks[(streaks['is_dry']) & (streaks['length'] >= 20)].copy()
    dry_streaks_df['start_date'] = pd.to_datetime(dry_streaks_df['start_date'])
    dry_streaks_df['end_date'] = pd.to_datetime(dry_streaks_df['end_date'])

    # Keep only streaks fully contained within July and August of the same year
    dry_streaks_df = dry_streaks_df[
        (dry_streaks_df['start_date'].dt.month.isin([7, 8])) &
        (dry_streaks_df['end_date'].dt.month.isin([7, 8])) &
        (dry_streaks_df['start_date'].dt.year == dry_streaks_df['end_date'].dt.year)
    ]

    # Add year and month columns based on start date
    dry_streaks_df['year'] = dry_streaks_df['start_date'].dt.year
    dry_streaks_df['month'] = dry_streaks_df['start_date'].dt.month

    # Aggregate summary stats
    total_dry_streaks_per_year = dry_streaks_df.groupby('year').size()
    max_length_per_year = dry_streaks_df.groupby('year')['length'].max()

    return total_dry_streaks_per_year, max_length_per_year, dry_streaks_df

#%% Combined plot function
def plot_combined_dry_spell_summary(max1, max2, loc1="Location 1", loc2="Location 2"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot for location 1
    axes[0].scatter(max1.index, max1.values, color='#0072B2')
    axes[0].set_title(f"Dry Spell Lengths during July and August in {loc1}", fontsize=14)
    axes[0].set_ylabel("Length of Dry Spell (Days)", fontsize=12)
    axes[0].set_xlabel("Year", fontsize=12)
    axes[0].grid(True)

    # Plot for location 2
    axes[1].scatter(max2.index, max2.values, color='#0072B2')
    axes[1].set_title(f"Dry Spell Lengths during July and August in {loc2}", fontsize=14)
    axes[1].set_xlabel("Year", fontsize=12)
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

#%% Running Computing of Dry Streaks function and plotting results
total_raich, max_raich, streaks_raich = compute_dry_spells(rain_raich)
total_chik, max_chik, streaks_chik = compute_dry_spells(rain_chik)

plot_combined_dry_spell_summary(max_raich, max_chik, loc1="Raichur", loc2="Chikkaballapur")


