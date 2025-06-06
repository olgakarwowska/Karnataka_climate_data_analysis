# SEASONAL RAINFALL RAICHUR AND CHIKKABALLAPUR (RABI AND KHARIF SEASON) 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load and clean data
rain_raichur = pd.read_csv('clipped_raichur_rain.csv', sep=',')
rain_raichur['time'] = pd.to_datetime(rain_raichur['time'])
rain_raichur = rain_raichur.dropna(subset=['rain'])

rain_chikkaballapur = pd.read_csv('clipped_chikkaballapur_rain.csv', sep=',')
rain_chikkaballapur['time'] = pd.to_datetime(rain_chikkaballapur['time'])
rain_chikkaballapur = rain_chikkaballapur.dropna(subset=['rain'])

# Define season function
def assign_season(df):
    def get_season(month):
        if month in [11, 12, 1, 2, 3, 4]:
            return "Rabi Season(Nov-Apr)"
        elif month in [6, 7, 8, 9, 10]:
            return "Kharif Season(Jun-Oct)"
        return None

    df['season'] = df['time'].dt.month.apply(get_season)
    df['year'] = df['time'].dt.year
    df.loc[df['time'].dt.month.isin([1, 2, 3, 4]), 'year'] -= 1 #Jan-Apr from year N
    df.loc[df['time'].dt.month.isin([11, 12]), 'year'] += 1 #Nov&Dec from year N-1
    return df

# Process and plot function
def process_and_plot(df, title):
    df = assign_season(df)
    rain_by_coord = df.groupby(['year', 'season', 'lat', 'lon'])['rain'].sum().reset_index()
    seasonal_rain = rain_by_coord.groupby(['year', 'season'])['rain'].mean().reset_index()
    seasonal_rain = seasonal_rain.pivot(index='year', columns='season', values='rain').sort_index()
    seasonal_rain = seasonal_rain[['Rabi Season(Nov-Apr)', 'Kharif Season(Jun-Oct)']]
    # Remove incomplete Rabi years (1900 and 2025)
    if 'Rabi Season(Nov-Apr)' in seasonal_rain.columns:
        seasonal_rain.loc[[1900, 2025], 'Rabi Season(Nov-Apr)'] = None


    color_map = {'Rabi Season(Nov-Apr)': '#D55E00', 'Kharif Season(Jun-Oct)': '#0072B2'}
    #color_map = {'Rabi Season(Nov-Apr)': '#E69F00', 'Kharif Season(Jun-Oct)': '#56B4E9'}
    marker_map = {'Rabi Season(Nov-Apr)': 'o', 'Kharif Season(Jun-Oct)': 's'}
    #trendline_color_map = {'Rabi Season(Nov-Apr)': '#E69F00', 'Kharif Season(Jun-Oct)': '#56B4E9'}
    trendline_color_map = {'Rabi Season(Nov-Apr)': '#D55E00', 'Kharif Season(Jun-Oct)': '#0072B2'}
    
    plt.figure(figsize=(7, 5))
    for season in seasonal_rain.columns:
        y = seasonal_rain[season]
        valid = ~y.isna()
        x = seasonal_rain.index[valid]
        y = y[valid]

        plt.plot(seasonal_rain.index, seasonal_rain[season], label=season,
                 color=color_map[season], marker=marker_map[season], markersize=5)

        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        trend = slope * x + intercept
        p_text = "<<0.05" if p_value < 0.001 else "<0.05" if p_value < 0.05 else f"{p_value:.3f}"
        plt.plot(x, trend, linestyle='--', color=trendline_color_map[season],
                 label=f"Trendline: y = {slope:.2f}x + {intercept:.2f}, p = {p_text}")

    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Total Seasonal Rainfall (mm)", fontsize=12)
    plt.title(title, fontsize=16, pad=15)
    plt.xticks(ticks=range(seasonal_rain.index.min(), seasonal_rain.index.max()+1, 5),
               rotation=90, fontsize=11)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.ylim(0, 1500)
    plt.legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    plt.show()

# Run for both regions
process_and_plot(rain_raichur, "Seasonal Total Rainfall in Raichur")
process_and_plot(rain_chikkaballapur, "Seasonal Total Rainfall in Chikkaballapur")
