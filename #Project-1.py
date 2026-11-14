#Project-1

    import pandas as pd  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]


# 1. Load raw data (simulated path from NOAA or Kaggle)
df_raw = pd.read_csv('raw_global_weather_energy.csv')

# 2. Data Hygiene & Standardization
df_clean = df_raw.drop_duplicates(subset=['station_id', 'date'])
df_clean['date'] = pd.to_datetime(df_clean['date'])

# Fill missing weather readings using forward-fill grouped by weather station
df_clean = df_clean.sort_values(['station_id', 'date'])
cols_to_fill = ['temp_celsius', 'wind_speed_ms', 'solar_irradiance_wm2']
df_clean[cols_to_fill] = df_clean.groupby('station_id')[cols_to_fill].ffill()

# Drop any remaining nulls if a station had no initial reading
df_clean = df_clean.dropna(subset=cols_to_fill)

# 3. Scientific Feature Engineering
# Wind Efficiency: Turbines typically generate power between 3 m/s (cut-in) and 25 m/s (cut-out)
df_clean['viable_wind_flag'] = np.where(
    (df_clean['wind_speed_ms'] >= 3.0) & (df_clean['wind_speed_ms'] <= 25.0), 1, 0
)

# Solar Thermal Efficiency: Panels lose ~0.4% efficiency per degree above 25°C
temp_penalty = np.maximum(0, (df_clean['temp_celsius'] - 25.0) * 0.004)
df_clean['adjusted_solar_yield'] = df_clean['solar_irradiance_wm2'] * (1 - temp_penalty)

# Calculate Normalized Capacity Factor (0 to 1 scale)
max_theoretical_solar = 1000.0  # Standard Test Conditions (1000 W/m2)
df_clean['solar_capacity_factor'] = (df_clean['adjusted_solar_yield'] / max_theoretical_solar).clip(0, 1)

# 4. Split & Export for Relational Database (Star Schema)
dim_stations = df_clean[['station_id', 'country', 'region', 'latitude', 'longitude']].drop_duplicates()
fact_metrics = df_clean[['station_id', 'date', 'temp_celsius', 'wind_speed_ms', 'viable_wind_flag', 'solar_capacity_factor']]

dim_stations.to_csv('dim_stations.csv', index=False)
fact_metrics.to_csv('fact_energy_metrics.csv', index=False)

print("Pipeline complete. Exported clean dimension and fact tables.")
