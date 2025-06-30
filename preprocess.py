import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from feature_engineering import create_weather_features_complex, create_weather_features_simple
from tsfm_public import TimeSeriesPreprocessor

def check_and_fix_dst_duplicates(df_wind: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Check if duplicates in 'Date from' are due to DST and fix them.
    
    :param df_wind: Wind power dataframe with 'Date from' column
    :param verbose: If True, print detailed information about the process
    :returns: Cleaned dataframe with duplicates removed
    """
    if verbose:
        print("=== Checking DST duplicates in wind data ===")
    
    # Check duplicates
    initial_count = len(df_wind)
    unique_count = df_wind["Date from"].nunique()
    duplicate_count = initial_count - unique_count
    
    if verbose:
        print(f"Total rows: {initial_count}")
        print(f"Unique timestamps: {unique_count}")
        print(f"Duplicate timestamps: {duplicate_count}")
    
    df_wind['timestamp'] = pd.to_datetime(df_wind['Date from'])

    if duplicate_count > 0:
        # Check if duplicates occur around DST transitions
        df_wind['timestamp'] = pd.to_datetime(df_wind['Date from'])
        duplicated_timestamps = df_wind[df_wind['timestamp'].duplicated(keep=False)]['timestamp'].unique()
        
        if verbose:
            print(f"Duplicated timestamps ({len(duplicated_timestamps)} unique):")
            for ts in sorted(duplicated_timestamps)[:10]:  # Show first 10
                print(f"  {ts}")
            
        # Check if these are around DST dates (last Sunday in March, last Sunday in October)
        dst_months = [3, 10]  # March and October
        duplicated_months = pd.to_datetime(duplicated_timestamps).month
        dst_related = sum(month in dst_months for month in duplicated_months)
        
        if verbose:
            print(f"Duplicates in DST months (Mar/Oct): {dst_related}/{len(duplicated_timestamps)}")
        
        # Fix by keeping first occurrence (usually the correct one)
        df_wind = df_wind.drop_duplicates(subset=['timestamp'], keep='first').copy()
        if verbose:
            print(f"After removing duplicates: {len(df_wind)} rows")
        
    return df_wind

def clean_wind_data(df_wind: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Clean wind power dataframe: remove unnamed columns, handle DST, resample to hourly.
    
    :param df_wind: Raw wind power dataframe
    :param verbose: If True, print detailed information about the cleaning process
    :returns: Cleaned and resampled wind power dataframe
    """
    if verbose:
        print("=== Cleaning wind data ===")
    
    # Remove unnamed columns
    unnamed_cols = [col for col in df_wind.columns if 'Unnamed' in str(col)]
    df_clean = df_wind.drop(columns=unnamed_cols)
    if verbose:
        print(f"Removed {len(unnamed_cols)} unnamed columns")
    
    # Check and fix DST duplicates
    df_clean = check_and_fix_dst_duplicates(df_clean, verbose=verbose)
    
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
    
    if verbose:
        print(f"Wind power columns found: {existing_wind_cols}")
        if missing_wind_cols:
            print(f"Missing wind power columns: {missing_wind_cols}")
    
    # RESAMPLE FROM 15-MIN TO HOURLY
    if verbose:
        print(f"Before resampling: {df_clean.shape} (15-min intervals)")
    
    # Select only numeric columns for resampling (exclude string date columns)
    df_for_resampling = df_clean[['timestamp'] + existing_wind_cols].copy()
    
    if verbose:
        print(f"Columns for resampling: {df_for_resampling.columns.tolist()}")
    
    # Set timestamp as index for resampling
    df_for_resampling = df_for_resampling.set_index('timestamp')
    
    # Resample to hourly using mean aggregation
    df_hourly = df_for_resampling.resample('1H').mean()
    
    # Reset index to get timestamp back as column
    df_hourly = df_hourly.reset_index()
    
    if verbose:
        print(f"After resampling to hourly: {df_hourly.shape}")
        print(f"Wind data columns: {list(df_hourly.columns)}")
    
    return df_hourly

def merge_wind_weather_data(df_wind_clean: pd.DataFrame, df_weather_agg: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Merge wind power and weather data with sanity checks. Interpolate missings that are caused through merging due to different handling of DST in weather and power datasets.
    
    :param df_wind_clean: Cleaned wind power dataframe
    :param df_weather_agg: Aggregated weather features dataframe
    :param verbose: If True, print detailed information about the merge process
    :returns: Merged and cleaned dataframe
    """
    if verbose:
        print("=== Merging wind and weather data ===")
    
    # Pre-merge checks
    if verbose:
        print("Before merge:")
        print(f"Wind data: {df_wind_clean.shape}, date range: {df_wind_clean['timestamp'].min()} to {df_wind_clean['timestamp'].max()}")
        print(f"Weather data: {df_weather_agg.shape}, date range: {df_weather_agg['timestamp'].min()} to {df_weather_agg['timestamp'].max()}")
    
    # Merge on timestamp
    df_combined = pd.merge(df_wind_clean, df_weather_agg, 
                          on='timestamp', how='inner')
    
    # Post-merge checks
    if verbose:
        print(f"After merge: {df_combined.shape}")
        print(f"Date range: {df_combined['timestamp'].min()} to {df_combined['timestamp'].max()}")
    
    # Check for missing values
    missing_counts = df_combined.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) > 0:
        if verbose:
            print(f"Missing values found:\n{missing_cols}")
        
        # Get all timestamps with missing values
        missing_mask = df_combined.isnull().any(axis=1)
        missing_timestamps = df_combined.loc[missing_mask, 'timestamp'].tolist()
        if verbose:
            print(f"\nTimestamps with missing values: {missing_timestamps}")
        
        # Always interpolate missing values
        if verbose:
            print("\nInterpolating missing values...")
        wind_power_cols = ['wind_power_offshore', 'wind_power_onshore']
        
        for col in wind_power_cols:
            if col in df_combined.columns and df_combined[col].isnull().any():
                df_combined[col] = df_combined[col].interpolate(method='linear').fillna(method='ffill')
        
        # Show interpolated values with neighbors
        if verbose:
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
        if verbose:
            if remaining_missing == 0:
                print(f"\n✅ All missing values successfully interpolated")
            else:
                print(f"\n⚠️ Warning: {remaining_missing} missing values still remain")
    else:
        if verbose:
            print("No missing values found")
    
    # Sort by timestamp to ensure proper time ordering
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
    if verbose:
        print("Data sorted by timestamp")
    
    return df_combined
    
def create_temporal_splits(df_combined: pd.DataFrame, plot: bool = False, power_column: str = 'wind_power_offshore', verbose: bool = False) -> tuple[pd.DataFrame, dict]:
    """Create temporal splits: 2019+2020 train, 2021 val, 2022 test.
    
    :param df_combined: Combined dataframe with timestamp column
    :param plot: If True creates verification plot
    :param power_column: Column name to plot
    :param verbose: If True, print detailed information about the splits
    :returns: Tuple containing the combined dataframe and dict of data splits with df indices indicating split boundaries
    """
    if verbose:
        print("=== Creating temporal splits ===")
    
    # Extract year from timestamp
    df_combined['year'] = df_combined['timestamp'].dt.year
    
    # Check available years
    available_years = sorted(df_combined['year'].unique())
    if verbose:
        print(f"Available years in data: {available_years}")
    
    # Show data distribution by year
    year_counts = df_combined['year'].value_counts().sort_index()
    if verbose:
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
    
    if verbose:
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

def setup_ttm_preprocessor(df_combined: pd.DataFrame, observable_columns: list = None, context_length: int = 512, prediction_length: int = 24, timestamp_column: str = "timestamp", verbose: bool = False) -> tuple[object, dict, int, int]:
    """Set up TimeSeriesPreprocessor for TTM model.
    
    :param df_combined: Combined dataframe with all features
    :param observable_columns: List of observable column names
    :param context_length: Context length in hours
    :param prediction_length: Prediction length in hours
    :param timestamp_column: Name of timestamp column
    :param verbose: If True, print detailed configuration information
    :returns: Tuple containing preprocessor, column specifiers, context length, and prediction length
    """
    if observable_columns is None:
        observable_columns = []
        
    if verbose:
        print("=== Setting up TTM preprocessor ===")

    # Column specifications
    id_columns = []  # Single aggregated time series

    # Target: What we want to forecast
    target_columns = ["wind_power_offshore", "wind_power_onshore"]

    # Conditional: Variables known in past but not future
    # Since we have weather forecasts, most variables are observable rather than conditional
    conditional_columns = []

    # Control: None for this use case
    control_columns = []

    column_specifiers = {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": target_columns,
        "observable_columns": observable_columns,  # Observable: Weather variables (known in future via forecasts)
        "conditional_columns": conditional_columns,
        "control_columns": control_columns,
    }

    if verbose:
        print("TTM Column Configuration:")
        print(f"Target columns ({len(target_columns)}): {target_columns}")
        print(f"Observable columns ({len(observable_columns)}): {observable_columns}")
        print(f"Conditional columns ({len(conditional_columns)}): {conditional_columns}")
        print(f"Context length: {context_length} hours (~{context_length/24:.1f} days)")
        print(f"Prediction length: {prediction_length} hours")

    # Create preprocessor
    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=prediction_length,
        scaling=True,  # Important for different units
        encode_categorical=False,  # No categorical data
        scaler_type="standard",  # Standardize all features
    )

    return tsp, column_specifiers

def prepare_wind_power_dataset(df_wind: pd.DataFrame, df_weather: pd.DataFrame, spatial_config: str = 'simple', context_length: int = 512,
        prediction_length: int = 24, feature_config: [] = None, 
        verbose: bool = False, shuffle_feature: str = None, random_seed: int = 42,) -> dict:
    """Complete pipeline to prepare wind power dataset for TTM.
    
    :param df_wind: Wind power generation data
    :param df_weather: Weather forecast data with spatial grid points
    :param spatial_config: Spatial aggregation strategy. Options: 'simple', 'coastal_inland', 'latitude_bands'
    :param context_length: Context length for model
    :param prediction_length: Prediction horizon length  
    :param feature_config: Feature engineering configuration
    :param verbose: If True, print detailed information throughout the pipeline
    :returns: Dictionary containing 'data' (combined dataset), 'preprocessor' (TTM preprocessor), 'split_config' (train/val/test splits), 'column_specifiers', 'context_length', 'prediction_length'
    """
    print("=== WIND POWER TTM DATA PREPARATION PIPELINE ===\n")
    
    if verbose:
        print(f"Spatial config: {spatial_config}")
    
    print("Step 1: Cleaning and resampling wind power data (15min -> hourly)")
    df_wind_clean = clean_wind_data(df_wind, verbose=verbose)
    
    print("Step 2: Creating weather features with configurable resolution")
    df_weather_agg = create_weather_features_complex(
        df_weather, 
        spatial_config=spatial_config,
        feature_config=feature_config
    )
    
    print("Step 3: Merging datasets")
    df_combined = merge_wind_weather_data(df_wind_clean, df_weather_agg, verbose=verbose)
    
    if shuffle_feature:
        original_values = df_combined[shuffle_feature].values[:5].copy()
            
        np.random.seed(random_seed)
        feature_data = df_combined[shuffle_feature].values.copy()
        np.random.shuffle(feature_data)
        df_combined[shuffle_feature] = feature_data
        
        # Verify shuffling worked
        shuffled_values = df_combined[shuffle_feature].values[:5]
        shuffling_worked = not np.array_equal(original_values, shuffled_values)
        
        print(f"   🔀 Original first 5 values: {original_values}")
        print(f"   🔀 Shuffled first 5 values: {shuffled_values}")
        print(f"   🔀 Shuffling successful: {shuffling_worked}")
        
        if not shuffling_worked:
            raise ValueError(f"Shuffling of feature '{shuffle_feature}' did not change values. Please check the data.")
        else:
            print(f"   ✅ Feature '{shuffle_feature}' shuffled successfully with seed {random_seed}")

    print("Step 4: Creating temporal splits")
    df_combined, split_config = create_temporal_splits(df_combined, plot=False, verbose=verbose)
    observable_columns = [col for col in df_combined.columns if col not in ['timestamp', 'wind_power_offshore', 'wind_power_onshore']]
    
    print("Step 5: Setting up TTM preprocessor")
    tsp, column_specifiers = setup_ttm_preprocessor(df_combined, observable_columns=observable_columns, context_length=context_length, prediction_length=prediction_length, verbose=verbose)

    # Final check: ensure data is properly sorted
    if not df_combined['timestamp'].is_monotonic_increasing:
        if verbose:
            print("\nFinal sorting check: Re-sorting data by timestamp")
        df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
    else:
        if verbose:
            print("\nFinal sorting check: Data already properly sorted")

    print(f"\nPIPELINE COMPLETE")
    if verbose:
        print(f"Final dataset: {df_combined.shape}")
        weather_feature_count = len([col for col in df_combined.columns 
                                    if col not in ['timestamp', 'wind_power_offshore', 'wind_power_onshore']])
        print(f"Weather features: {weather_feature_count}")
        print(f"Spatial resolution: {spatial_config}")

    return {
        'data': df_combined,
        'preprocessor': tsp,
        'split_config': split_config,
        'column_specifiers': column_specifiers,
        'context_length': context_length,
        'prediction_length': prediction_length,
        'feature_config': feature_config
    }