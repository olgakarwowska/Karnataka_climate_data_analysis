# -*- coding: utf-8 -*-
"""
Created on Thu May 15 10:49:20 2025

@author: Administrator
"""

#%% Libraries
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

#%% Set working directory and read CSV files (you need to provide actual paths)
os.chdir("C:\\Users\\Administrator\\Documents\\MEE\\Design of Climate Change Mitigation and Adaptation Strategies\\")

rain_chik = pd.read_csv("Clipped data of Chikkaballapur\\clipped_chikkaballapur_rain_5_points.csv", delimiter=",")
rain_raich = pd.read_csv("Clipped data of Raichur\\clipped_raichur_rain.csv", delimiter=",")

#%% Preprocessing
rain_chik['date_time'] = pd.to_datetime(rain_chik['time'], errors='coerce')
rain_chik.set_index(rain_chik['date_time'], inplace=True)
del rain_chik['time']

rain_raich['date_time'] = pd.to_datetime(rain_raich['time'], errors='coerce')
rain_raich.set_index(rain_raich['date_time'], inplace=True)
del rain_raich['time']

#%% Function to compute dry streaks
def compute_dry_streaks(df):
    df['is_dry'] = df['rain'] <= 2.5  # Dry day definition
    df['streak_id'] = (df['is_dry'] != df['is_dry'].shift()).cumsum()

    # Group by streak
    streaks = df.groupby('streak_id').agg({
        'is_dry': 'first',
        'date_time': ['first', 'count']
    })
    streaks.columns = ['is_dry', 'start_date', 'length']
    streaks = streaks.reset_index(drop=True)

    # Filter for dry streaks of at least 7 days
    dry_streaks_df = streaks[(streaks['is_dry']) & (streaks['length'] >= 20)].copy()

    # Extract year
    dry_streaks_df['year'] = dry_streaks_df['start_date'].dt.year

    # Aggregate results
    total_dry_streaks_per_year = dry_streaks_df.groupby('year').size()
    max_length_per_year = dry_streaks_df.groupby('year')['length'].max()

    return total_dry_streaks_per_year, max_length_per_year

#%% Plotting function
def plot_dry_spell_summary(total_dry_streaks, max_length_per_year, location="Unknown"):
    years = max_length_per_year.index

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    # Longest dry streaks
    axs[0].plot(years, max_length_per_year.values, marker='o', color='crimson')
    axs[0].set_title(f"Longest Dry Streaks in {location}")
    axs[0].set_ylabel("Days")
    axs[0].set_xlabel("Year")
    axs[0].grid(True)

    # Number of dry spells per year
    axs[1].bar(total_dry_streaks.index, total_dry_streaks.values, color='steelblue')
    axs[1].set_title(f"Number of Dry Spells (≥20 days) in {location}")
    axs[1].set_ylabel("Number of Streaks")
    axs[1].set_xlabel("Year")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

#%% Run analysis and plot for Raichur
total_raich, max_raich = compute_dry_streaks(rain_raich)
plot_dry_spell_summary(total_raich, max_raich, location="Raichur")

#%% Run analysis and plot for Chikkaballapur
total_chik, max_chik = compute_dry_streaks(rain_chik)
plot_dry_spell_summary(total_chik, max_chik, location="Chikkaballapur")
