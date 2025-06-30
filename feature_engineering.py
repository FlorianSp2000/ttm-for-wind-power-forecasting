import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

def analyze_forecast_lead_time(df_weather: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Analyze forecast_origin - time to understand forecast lead time. Handle mixed date formats in the weather dataset (2019-2021: datetime, 2022: date only).
    
    :param df_weather: Weather dataframe with 'forecast_origin' and 'time' columns
    :param verbose: If True, print detailed information about lead time analysis
    :returns: Weather dataframe with added 'forecast_lead_hours' column
    """
    # Check format of forecast_origin column
    sample_values = df_weather['forecast_origin'].dropna().head(10)
    
    # Convert time column first
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    
    # Handle mixed forecast_origin formats
    df_weather_copy = df_weather.copy()
    
    # Try to detect which entries are date-only vs datetime
    # Date-only format: "YYYY-MM-DD" (length 10)
    # Datetime format: "YYYY-MM-DD HH:MM:SS" (length 19)
    forecast_origin_str = df_weather_copy['forecast_origin'].astype(str)
    date_only_mask = forecast_origin_str.str.len() == 10
    datetime_mask = forecast_origin_str.str.len() == 19
    
    # Convert forecast_origin handling mixed formats
    df_weather_copy['forecast_origin'] = pd.to_datetime(df_weather_copy['forecast_origin'], format='mixed')
    
    # For date-only entries (2022), assume forecast_origin time = time (lead time = 0)
    if date_only_mask.any():
        # For date-only entries, set forecast_origin to match the time column
        df_weather_copy.loc[date_only_mask, 'forecast_origin'] = df_weather_copy.loc[date_only_mask, 'time']
    
    # Calculate lead time in hours
    df_weather_copy['forecast_lead_hours'] = (df_weather_copy['time'] - df_weather_copy['forecast_origin']).dt.total_seconds() / 3600
    
    return df_weather_copy

def create_weather_features_complex(df_weather: pd.DataFrame, 
                                   spatial_config: str = 'simple', 
                                   feature_config: Optional[Dict] = None) -> pd.DataFrame:
    """Create weather features with configurable spatial resolution and feature engineering.

    :param df_weather: Raw weather data with columns: time, latitude, longitude, weather variables
    :param spatial_config: Spatial aggregation strategy. Options: 'simple' (average all 80 stations), 'coastal_inland' (separate coastal/inland features), 'latitude_bands' (separate features for each latitude)
    :param feature_config: Feature engineering configuration with keys: 'base_features' (list of weather variables), 'derived_features' (bool for wind speed features), 'temporal_features' (bool for hour/month features), 'lag_features' (list of dicts with format {col_name: [1,2,3]} for lag columns)
    :returns: Weather features with timestamp and spatial/temporal features
    """
    print(f"=== Creating weather features (spatial_config: {spatial_config}) ===")
    
    # Default feature configuration
    if feature_config is None:
        feature_config = {
            'base_features': ['u100', 'v100', 'u10', 'v10', 'msl', 't2m', 'cdir', 'blh', 'tcc', 'tp'],
            'derived_features': True,
            'temporal_features': False,
            'lag_features': []
        }

    # Make a copy to avoid modifying the original
    feature_config = feature_config.copy()
    feature_config['base_features'] = feature_config['base_features'].copy()
    
    # Analyze forecast lead time
    df_weather = analyze_forecast_lead_time(df_weather)
    
    # print(f"Latitude range: {df_weather['latitude'].min():.2f}° to {df_weather['latitude'].max():.2f}°")
    # print(f"Longitude range: {df_weather['longitude'].min():.2f}° to {df_weather['longitude'].max():.2f}°")
    
    # Create derived features first (before spatial aggregation)
    if feature_config['derived_features']:
        print("Creating derived features: wind_speed_10m, wind_speed_100m")
        df_weather['wind_speed_10m'] = np.sqrt(df_weather['u10']**2 + df_weather['v10']**2)
        df_weather['wind_speed_100m'] = np.sqrt(df_weather['u100']**2 + df_weather['v100']**2)
        feature_config['base_features'].extend(['wind_speed_10m', 'wind_speed_100m'])
    
    # Unit conversions and transformations BEFORE spatial aggregation
    df_weather['pressure_hpa'] = df_weather['msl'] / 100
    df_weather['temperature_c'] = df_weather['t2m'] - 273.15
    df_weather['wind_dir_sin'] = np.sin(np.radians(df_weather['cdir']))
    df_weather['wind_dir_cos'] = np.cos(np.radians(df_weather['cdir']))
    
    # Update feature list to include transformed features
    transformed_features = []
    for feature in feature_config['base_features']:
        transformed_features.append(feature)
        if feature == 'msl':
            transformed_features.append('pressure_hpa')
        elif feature == 't2m':
            transformed_features.append('temperature_c')
        elif feature == 'cdir':
            transformed_features.append('wind_dir_sin')
            transformed_features.append('wind_dir_cos')
    
    # Add forecast lead hours
    agg_columns = transformed_features + ['forecast_lead_hours']
    
    # Apply spatial aggregation strategy
    if spatial_config == 'simple':
        print("Spatial strategy: Simple averaging across all 80 stations")
        weather_agg = df_weather.groupby('time')[agg_columns].mean().reset_index()
        
    elif spatial_config == 'coastal_inland':
        print("Spatial strategy: Separate coastal and inland features")
        
        coastal_threshold = 53.5
        coastal_mask = df_weather['latitude'] >= coastal_threshold
        inland_mask = df_weather['latitude'] < coastal_threshold
        
        coastal_count = coastal_mask.sum()
        inland_count = inland_mask.sum()
        print(f"Coastal stations: {coastal_count} measurements")
        print(f"Inland stations: {inland_count} measurements")
        
        if coastal_count == 0:
            raise ValueError("No coastal stations found with lat >= 53.5°")
        if inland_count == 0:
            raise ValueError("No inland stations found with lat < 53.5°")
        
        # Aggregate coastal stations
        coastal_agg = df_weather[coastal_mask].groupby('time')[agg_columns].mean().reset_index()
        coastal_agg = coastal_agg.add_suffix('_coastal')
        coastal_agg = coastal_agg.rename(columns={'time_coastal': 'time'})
        
        # Aggregate inland stations  
        inland_agg = df_weather[inland_mask].groupby('time')[agg_columns].mean().reset_index()
        inland_agg = inland_agg.add_suffix('_inland')
        inland_agg = inland_agg.rename(columns={'time_inland': 'time'})
        
        # Merge coastal and inland
        weather_agg = pd.merge(coastal_agg, inland_agg, on='time', how='outer')
        
        print(f"Coastal/Inland approach created {weather_agg.shape[1] - 1} weather features")
        
    elif spatial_config == 'latitude_bands':
        print("Spatial strategy: Separate features for each latitude band")
        
        unique_lats = sorted(df_weather['latitude'].unique())
        n_bands = len(unique_lats)
        
        print(f"Found {n_bands} unique latitudes: {unique_lats}")
        
        band_dfs = []
        for i, lat in enumerate(unique_lats):
            band_mask = df_weather['latitude'] == lat
            
            if band_mask.sum() > 0:
                band_agg = df_weather[band_mask].groupby('time')[agg_columns].mean().reset_index()
                band_agg = band_agg.add_suffix(f'_lat{lat:.1f}')
                band_agg = band_agg.rename(columns={f'time_lat{lat:.1f}': 'time'})
                band_dfs.append(band_agg)
                print(f"Latitude {lat:.1f}°: {band_mask.sum()} measurements")
        
        if not band_dfs:
            raise ValueError("No valid latitude bands found")
        
        # Merge all bands
        weather_agg = band_dfs[0]
        for band_df in band_dfs[1:]:
            weather_agg = pd.merge(weather_agg, band_df, on='time', how='outer')
            
        print(f"Latitude bands approach created {weather_agg.shape[1] - 1} weather features")
        
    else:
        raise ValueError(f"Unknown spatial_config: {spatial_config}. Must be one of: 'simple', 'coastal_inland', 'latitude_bands'")
    
    # print(f"After spatial aggregation: {weather_agg.shape}")
    
    # Rename time to timestamp for consistency
    weather_agg = weather_agg.rename(columns={'time': 'timestamp'})
    # Ensure timestamp is datetime
    weather_agg['timestamp'] = pd.to_datetime(weather_agg['timestamp'])
    
    # Add temporal features if requested
    if feature_config['temporal_features']:
        print("Adding temporal features...")
        
        weather_agg['hour'] = weather_agg['timestamp'].dt.hour
        weather_agg['month'] = weather_agg['timestamp'].dt.month
        weather_agg['year'] = weather_agg['timestamp'].dt.year
        weather_agg['dayofweek'] = weather_agg['timestamp'].dt.dayofweek
        weather_agg['dayofyear'] = weather_agg['timestamp'].dt.dayofyear
        weather_agg['quarter'] = weather_agg['timestamp'].dt.quarter
        weather_agg['weekofyear'] = weather_agg['timestamp'].dt.isocalendar().week
        
        # Cyclical encoding for temporal features
        weather_agg['hour_sin'] = np.sin(2 * np.pi * weather_agg['hour'] / 24)
        weather_agg['hour_cos'] = np.cos(2 * np.pi * weather_agg['hour'] / 24)
        weather_agg['month_sin'] = np.sin(2 * np.pi * weather_agg['month'] / 12)
        weather_agg['month_cos'] = np.cos(2 * np.pi * weather_agg['month'] / 12)
        weather_agg['dayofweek_sin'] = np.sin(2 * np.pi * weather_agg['dayofweek'] / 7)
        weather_agg['dayofweek_cos'] = np.cos(2 * np.pi * weather_agg['dayofweek'] / 7)
    
    # Add lag features if requested
    if feature_config['lag_features']:
        print("Adding lag features...")
        
        # Ensure data is sorted by timestamp for proper lag calculation
        weather_agg = weather_agg.sort_values('timestamp').reset_index(drop=True)
        
        lag_count = 0
        for lag_dict in feature_config['lag_features']:
            for col_name, lag_periods in lag_dict.items():
                if col_name not in weather_agg.columns:
                    print(f"  Warning: Column '{col_name}' not found for lag features, skipping...")
                    continue
                
                for lag in lag_periods:
                    lag_col_name = f"{col_name}_lag{lag}"
                    weather_agg[lag_col_name] = weather_agg[col_name].shift(lag)
                    lag_count += 1
                
                print(f"  Created {len(lag_periods)} lag features for '{col_name}': lags {lag_periods}")
        
        print(f"Total lag features created: {lag_count}")
    
    print(f"Final weather feature set: {weather_agg.shape}")
    print(f"Feature columns: {len([col for col in weather_agg.columns if col != 'timestamp'])}")
    
    return weather_agg

def create_weather_features_simple(df_weather):
    """
    Create simple spatially averaged across all grid points weather features  

    :returns: df with timestamp and weather-specific feature columns in hourly resolution
    """
    print("=== Creating weather features ===")
    
    # Analyze forecast lead time first
    df_weather = analyze_forecast_lead_time(df_weather)
    
    # Check for time duplicates after spatial averaging
    unique_times_before = df_weather['time'].nunique()
    total_records_before = len(df_weather)
    print(f"Before spatial averaging - Total records: {total_records_before}, Unique times: {unique_times_before}")

    # weather features should be created here!
    # e.g. wind_speed_100m, wind_speed_10m
    print("Creating new features: wind_speed_10m, wind_speed_100m")
    df_weather['wind_speed_10m'] = np.sqrt(df_weather['u10']**2 + df_weather['v10']**2)
    df_weather['wind_speed_100m'] = np.sqrt(df_weather['u100']**2 + df_weather['v100']**2)

    # Simple spatial averaging over all geographic locations
    weather_agg = df_weather.groupby('time').agg({
        'wind_speed_100m': 'mean',
        'wind_speed_10m': 'mean',
        'msl': 'mean',  # Mean sea level pressure (Pa)
        't2m': 'mean',  # Temperature at 2m (K)
        'cdir': 'mean', # Wind direction (degrees)
        'u100': 'mean', # U-component wind at 100m
        'v100': 'mean', # V-component wind at 100m
        'u10': 'mean',  # U-component wind at 10m  
        'v10': 'mean',  # V-component wind at 10m
        'blh': 'mean',  # Boundary layer height
        'tcc': 'mean',  # Total cloud cover
        'tp': 'mean',   # Total precipitation
        'forecast_lead_hours': 'mean',  # Average lead time
    }).reset_index()
    
    print(f"After spatial averaging - Shape: {weather_agg.shape}")
    
    # Check for duplicates after averaging
    unique_times_after = weather_agg['time'].nunique()
    print(f"After averaging - Records: {len(weather_agg)}, Unique times: {unique_times_after}")
    
    if len(weather_agg) != unique_times_after:
        print("WARNING: Duplicates found after spatial averaging!")
        # Remove duplicates if any
        weather_agg = weather_agg.drop_duplicates(subset=['time'], keep='first')
        print(f"After removing duplicates: {len(weather_agg)} records")
    
    # Unit conversions
    # Pressure: Pa to hPa
    weather_agg['pressure_hpa'] = weather_agg['msl'] / 100
    
    # Temperature: K to Celsius  
    weather_agg['temperature_c'] = weather_agg['t2m'] - 273.15
    
    # Wind direction: degrees to sin/cos components (handles circular nature)
    weather_agg['wind_dir_sin'] = np.sin(np.radians(weather_agg['cdir']))
    weather_agg['wind_dir_cos'] = np.cos(np.radians(weather_agg['cdir']))
    
    # Rename time column to timestamp for merging
    weather_agg = weather_agg.rename(columns={'time': 'timestamp'})
    
    # Ensure timestamp is datetime
    weather_agg['timestamp'] = pd.to_datetime(weather_agg['timestamp'])
    
    print(f"Final weather features: {list(weather_agg.columns)}")
    
    return weather_agg
