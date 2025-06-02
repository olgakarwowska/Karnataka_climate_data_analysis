# Critical Climate Moments in Raichur and Chikkaballapur  
**Climate data analysis. Data sourced from IMD**

This repository contains a comprehensive analysis of long-term temperature trends and critical climate events in **Raichur** and **Chikkaballapur**, India. The analysis focuses on identifying patterns and changes in temperature extremes, seasonal variation, warming rates, and other climate signals using historical data from the **Indian Meteorological Department (IMD)**.

---

## Getting Started

Before diving into the analysis, make sure to **run the `install_requirements.ipynb` notebook**.  
This will install all the necessary Python libraries required for data processing, analysis, and visualization.

---

## Contents
## Temperature data analysis

The analysis is structured into the following steps:

1. **Data download**  
2. **Data pre-processing**  
   - Convert binary temperature data to `.csv` format  
   - Select specific data locations (e.g., `lat = ...`, `lon = ...`)  
3. **Loading the `.csv` files** as new variables  
4. **Indexing**  
   - Set `"date_time"` as index for easier time-series analysis  
5. **Resampling Tmax and Tmin data**  
   - Mean weekly, monthly, yearly values  
   - Kharif and Rabi season averages  
6. **General temperature trends**  
   - Plot Tmax and Tmin over time  
7. **Diurnal Temperature Variation (DTV)**  
   - Plot Tmax and Tmin spread  
8. **Probability distributions**  
   - 8.1 Single plot of Tmax and Tmin  
   - 8.2 Multiple plots for detailed view  
9. **Statistical analysis**  
   - Test for significant changes in Tmax/Tmin over the years  
10. **Monthly temperature change**  
    - 10.1 Single plot  
    - 10.2 Multiple plots (can adjust to weekly, etc.)  
11. **Decadal warming rate**  
    - 11.1 Function definition  
    - 11.2 Single plot  
    - 11.3 Multiple plots  
    - 11.4 Correlation in warming/cooling trends between regions  
12. **Temperature anomalies**  
    - 12.1 Prepare new dataframe and calculate climatology  
    - 12.2 Plot anomalies  
13. **Defining hot/cold days**  
    - 13.1 Function to set thresholds  
    - 13.2 Identify hot days  
    - 13.3 Calculate thresholds  
14. **Function to count hot days**  
15. **Function to count cold days**  
16. **Plot number of hot days**  
17. **Plot number of cold days**  
18. **Compare hot/cold days in both locations**  
19. **Plotting hot days**  
20. **Plotting cold days**  
21. **Variation in hot/cold days**  
    - Assess rate of change  
22. **First hot day of the year** – function definition  
23. **First cold day of the year** – function definition  
24. **Last hot day of the year** – function definition  
25. **Calculate first/last hot/cold days**  
26. **Plot first hot day**  
27. **Plot last hot day**  
28. **Plot first cold day**  
29. **Combined plot: first and last hot days**  
30. **Flag consecutive hot days in dataframe**  
31. **Function for hot streaks**  
32. **Function for cold streaks**  
33. **Analyze hot/cold streaks**  
34. **Function to plot hot streaks**  
35. **Function to plot cold streaks**  
36. **Plot hot streaks**  
37. **Plot cold streaks**

---
## Precipitation data analysis
---
## Comparitive analysis
---
## Notes

- Data source: **Indian Meteorological Department (IMD)**
- Focus regions: **Raichur** and **Chikkaballapur**, Karnataka, India
- Python notebooks and scripts are modular and well-documented for clarity and reproducibility

---

