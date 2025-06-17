import pandas as pd
import numpy as np
from datetime import datetime

def check_and_fix_dst_duplicates(df_wind):
    """
    Check if duplicates in 'Date from' are due to DST and fix them
    """
    print("=== Checking DST duplicates in wind data ===")
    
    # Check duplicates
    initial_count = len(df_wind)
    unique_count = df_wind["Date from"].nunique()
    duplicate_count = initial_count - unique_count
    
    print(f"Total rows: {initial_count}")
    print(f"Unique timestamps: {unique_count}")
    print(f"Duplicate timestamps: {duplicate_count}")
    
    if duplicate_count > 0:
        # Check if duplicates occur around DST transitions
        df_wind['timestamp'] = pd.to_datetime(df_wind['Date from'])
        duplicated_timestamps = df_wind[df_wind['timestamp'].duplicated(keep=False)]['timestamp'].unique()
        
        print(f"Duplicated timestamps ({len(duplicated_timestamps)} unique):")
        for ts in sorted(duplicated_timestamps)[:10]:  # Show first 10
            print(f"  {ts}")
            
        # Check if these are around DST dates (last Sunday in March, last Sunday in October)
        dst_months = [3, 10]  # March and October
        duplicated_months = pd.to_datetime(duplicated_timestamps).month
        dst_related = sum(month in dst_months for month in duplicated_months)
        
        print(f"Duplicates in DST months (Mar/Oct): {dst_related}/{len(duplicated_timestamps)}")
        
        # Fix by keeping first occurrence (usually the correct one)
        df_wind_fixed = df_wind.drop_duplicates(subset=['timestamp'], keep='first').copy()
        print(f"After removing duplicates: {len(df_wind_fixed)} rows")
        
        return df_wind_fixed
    else:
        df_wind['timestamp'] = pd.to_datetime(df_wind['Date from'])
        return df_wind

def clean_wind_data(df_wind):
    """
    Clean wind power dataframe: remove unnamed columns, keep only wind power, handle DST, resample to hourly
    """
    print("=== Cleaning wind data ===")
    
    # Remove unnamed columns
    unnamed_cols = [col for col in df_wind.columns if 'Unnamed' in str(col)]
    df_clean = df_wind.drop(columns=unnamed_cols)
    print(f"Removed {len(unnamed_cols)} unnamed columns")
    
    # Check and fix DST duplicates
    df_clean = check_and_fix_dst_duplicates(df_clean)
    
    # Clean column names (remove extra spaces)
    df_clean.columns = df_clean.columns.str.strip()
    
    # Rename wind power columns first
    column_mapping = {
        'Wind Offshore [MW]': 'wind_power_offshore',
        'Wind Onshore [MW]': 'wind_power_onshore'
    }
    df_clean = df_clean.rename(columns=column_mapping)
    
    # Check which wind power columns exist
    required_cols = ['wind_power_offshore', 'wind_power_onshore']
    existing_wind_cols = [col for col in required_cols if col in df_clean.columns]
    missing_wind_cols = [col for col in required_cols if col not in df_clean.columns]
    
    print(f"Wind power columns found: {existing_wind_cols}")
    if missing_wind_cols:
        print(f"Missing wind power columns: {missing_wind_cols}")
    
    # RESAMPLE FROM 15-MIN TO HOURLY
    print(f"Before resampling: {df_clean.shape} (15-min intervals)")
    
    # Select only numeric columns for resampling (exclude string date columns)
    df_for_resampling = df_clean[['timestamp'] + existing_wind_cols].copy()
    
    print(f"Columns for resampling: {df_for_resampling.columns.tolist()}")
    
    # Set timestamp as index for resampling
    df_for_resampling = df_for_resampling.set_index('timestamp')
    
    # Resample to hourly using mean aggregation
    df_hourly = df_for_resampling.resample('1H').mean()
    
    # Reset index to get timestamp back as column
    df_hourly = df_hourly.reset_index()
    
    print(f"After resampling to hourly: {df_hourly.shape}")
    print(f"Wind data columns: {list(df_hourly.columns)}")
    
    return df_hourly

def analyze_forecast_lead_time(df_weather):
    """
    Analyze forecast_origin - time to understand forecast lead time
    Handle mixed date formats (2019-2021: datetime, 2022: date only)
    """
    print("=== Analyzing forecast lead time ===")
    
    # Check format of forecast_origin column
    sample_values = df_weather['forecast_origin'].dropna().head(10)
    print(f"Sample forecast_origin values: {list(sample_values)}")
    
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
    
    print(f"Date-only entries (YYYY-MM-DD): {date_only_mask.sum()}")
    print(f"Datetime entries (YYYY-MM-DD HH:MM:SS): {datetime_mask.sum()}")
    print(f"Other formats: {(~date_only_mask & ~datetime_mask).sum()}")
    
    # Convert forecast_origin handling mixed formats
    df_weather_copy['forecast_origin'] = pd.to_datetime(df_weather_copy['forecast_origin'], format='mixed')
    
    # For date-only entries (2022), assume forecast_origin time = time (lead time = 0)
    if date_only_mask.any():
        print(f"Assuming forecast_origin = time for {date_only_mask.sum()} date-only entries (2022 data)")
        # For date-only entries, set forecast_origin to match the time column
        df_weather_copy.loc[date_only_mask, 'forecast_origin'] = df_weather_copy.loc[date_only_mask, 'time']
    
    # Calculate lead time in hours
    df_weather_copy['forecast_lead_hours'] = (df_weather_copy['time'] - df_weather_copy['forecast_origin']).dt.total_seconds() / 3600
    
    lead_times = df_weather_copy['forecast_lead_hours'].unique()
    print(f"Unique forecast lead times (hours): {sorted(lead_times)[:20]}")  # Show first 20
    print(f"Min lead time: {df_weather_copy['forecast_lead_hours'].min()} hours")
    print(f"Max lead time: {df_weather_copy['forecast_lead_hours'].max()} hours")
    print(f"Most common lead times:")
    print(df_weather_copy['forecast_lead_hours'].value_counts().head())
    
    # Show some examples
    print("\nExample lead time calculations:")
    sample_indices = [0, 1000, 2000] if len(df_weather_copy) > 2000 else [0, len(df_weather_copy)//2, -1]
    for idx in sample_indices:
        if idx < len(df_weather_copy):
            row = df_weather_copy.iloc[idx]
            print(f"  time={row['time']}, forecast_origin={row['forecast_origin']}, lead={row['forecast_lead_hours']:.1f}h")
    
    return df_weather_copy

def create_weather_features_simple(df_weather):
    """
    Create simple spatially averaged weather features with detailed checks

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

def merge_wind_weather_data(df_wind_clean, df_weather_agg):
    """
    Merge wind power and weather data with sanity checks. Interpolate missings
    """
    print("=== Merging wind and weather data ===")
    
    # Pre-merge checks
    print("Before merge:")
    print(f"Wind data: {df_wind_clean.shape}, date range: {df_wind_clean['timestamp'].min()} to {df_wind_clean['timestamp'].max()}")
    print(f"Weather data: {df_weather_agg.shape}, date range: {df_weather_agg['timestamp'].min()} to {df_weather_agg['timestamp'].max()}")
    
    # Merge on timestamp
    df_combined = pd.merge(df_wind_clean, df_weather_agg, 
                          on='timestamp', how='inner')
    
    # Post-merge checks
    print(f"After merge: {df_combined.shape}")
    print(f"Date range: {df_combined['timestamp'].min()} to {df_combined['timestamp'].max()}")
    
    # Check for missing values
    missing_counts = df_combined.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) > 0:
        print(f"Missing values found:\n{missing_cols}")
        
        # Get all timestamps with missing values
        missing_mask = df_combined.isnull().any(axis=1)
        missing_timestamps = df_combined.loc[missing_mask, 'timestamp'].tolist()
        print(f"\nTimestamps with missing values: {missing_timestamps}")
        
        # Always interpolate missing values
        print("\nInterpolating missing values...")
        wind_power_cols = ['wind_power_offshore', 'wind_power_onshore']
        
        for col in wind_power_cols:
            if col in df_combined.columns and df_combined[col].isnull().any():
                df_combined[col] = df_combined[col].interpolate(method='linear').fillna(method='ffill')
        
        # Show interpolated values with neighbors
        print("\nInterpolated values with neighbors:")
        for timestamp in missing_timestamps:
            idx = df_combined[df_combined['timestamp'] == timestamp].index[0]
            
            # Get neighboring rows (before and after)
            start_idx = max(0, idx - 1)
            end_idx = min(len(df_combined), idx + 2)
            neighbors = df_combined.iloc[start_idx:end_idx][['timestamp'] + wind_power_cols]
            
            print(f"\nAround {timestamp}:")
            for _, row in neighbors.iterrows():
                marker = " → " if row['timestamp'] == timestamp else "   "
                print(f"{marker}{row['timestamp']}: offshore={row[wind_power_cols[0]]:.1f}, onshore={row[wind_power_cols[1]]:.1f}")
        
        # Final check
        remaining_missing = df_combined.isnull().sum().sum()
        if remaining_missing == 0:
            print(f"\n✅ All missing values successfully interpolated")
        else:
            print(f"\n⚠️ Warning: {remaining_missing} missing values still remain")
    else:
        print("No missing values found")
    
    # Sort by timestamp to ensure proper time ordering
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
    print("Data sorted by timestamp")
    
    return df_combined
    
import matplotlib.pyplot as plt

def create_temporal_splits(df_combined, plot=False, power_column='wind_power_offshore'):
    """
    Create temporal splits: 2019+2020 train, 2021 val, 2022 test
    :param plot: bool, if True creates verification plot (default: False)
    :param power_column: str, column name to plot (default: 'wind_power_offshore')
    :returns: combined df and dict of data splits with df indices indicating split boundaries
    """
    print("=== Creating temporal splits ===")
    
    # Extract year from timestamp
    df_combined['year'] = df_combined['timestamp'].dt.year
    
    # Check available years
    available_years = sorted(df_combined['year'].unique())
    print(f"Available years in data: {available_years}")
    
    # Show data distribution by year
    year_counts = df_combined['year'].value_counts().sort_index()
    print("Data distribution by year:")
    for year, count in year_counts.items():
        print(f"  {year}: {count} records")
    
    # Create splits based on years
    train_mask = df_combined['year'].isin([2019, 2020])
    val_mask = df_combined['year'] == 2021
    test_mask = df_combined['year'] == 2022
    
    # Get indices for splits
    train_indices = df_combined[train_mask].index.tolist()
    val_indices = df_combined[val_mask].index.tolist()
    test_indices = df_combined[test_mask].index.tolist()
    
    split_config = {
        "train": [min(train_indices), max(train_indices) + 1] if train_indices else [0, 0],
        "valid": [min(val_indices), max(val_indices) + 1] if val_indices else [0, 0],
        "test": [min(test_indices), max(test_indices) + 1] if test_indices else [0, 0],
    }
    
    print("\nData split summary:")
    print(f"Train: {len(train_indices)} samples ({split_config['train']}) - years 2019-2020")
    print(f"Val:   {len(val_indices)} samples ({split_config['valid']}) - year 2021")
    print(f"Test:  {len(test_indices)} samples ({split_config['test']}) - year 2022")
    
    # Optional plot
    if plot:
        plt.figure(figsize=(15, 6))
        if train_indices:
            plt.plot(df_combined[train_mask]['timestamp'], df_combined[train_mask][power_column], 
                    color='blue', label='Train (2019-2020)', alpha=0.7)
        if val_indices:
            plt.plot(df_combined[val_mask]['timestamp'], df_combined[val_mask][power_column], 
                    color='orange', label='Validation (2021)', alpha=0.7)
        if test_indices:
            plt.plot(df_combined[test_mask]['timestamp'], df_combined[test_mask][power_column], 
                    color='green', label='Test (2022)', alpha=0.7)
        
        plt.xlabel('Time')
        plt.ylabel(power_column)
        plt.title('Temporal Data Splits')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Drop the helper year column
    df_combined = df_combined.drop(columns=['year'])
    
    return df_combined, split_config