import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def wind_data_summary(df, target_cols=['Wind Offshore [MW] ', 'Wind Onshore [MW]']):
    """
    Comprehensive EDA for wind power time series data
    """
    print("="*80)
    print("WIND POWER DATA - EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # ============================================================================
    # 1. DATA STRUCTURE & SCHEMA ANALYSIS
    # ============================================================================
    print("\n📊 1. DATA STRUCTURE & SCHEMA")
    print("-"*50)
    df = df.copy()
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Date range: {df.iloc[0, 0]} to {df.iloc[-1, 1]}")
    
    # Data types analysis
    print("\n🔍 Data Types:")
    dtype_summary = df.dtypes.value_counts()
    for dtype, count in dtype_summary.items():
        print(f"  {dtype}: {count} columns")
    
    # Identify and clean column names
    print("\n📋 Column Analysis:")
    energy_cols = [col for col in df.columns if '[MW]' in col]
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    date_cols = [col for col in df.columns if 'Date' in col]
    
    print(f"  Energy columns: {len(energy_cols)}")
    print(f"  Date columns: {len(date_cols)}")
    print(f"  Unnamed columns: {len(unnamed_cols)}")
    
    # ============================================================================
    # 2. DATA QUALITY ASSESSMENT
    # ============================================================================
    print("\n\n🔧 2. DATA QUALITY ASSESSMENT")
    print("-"*50)
    
    # Missing values analysis
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100
    
    print("📊 Missing Values Summary:")
    missing_summary = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing %': missing_pct
    }).sort_values('Missing %', ascending=False)
    
    # Show only columns with missing data
    if missing_summary['Missing Count'].sum() > 0:
        print(missing_summary[missing_summary['Missing Count'] > 0])
    else:
        print("  ✅ No missing values found!")
        
    # ============================================================================
    # 3. TEMPORAL ANALYSIS
    # ============================================================================
    print("\n\n📅 3. TEMPORAL ANALYSIS")
    print("-"*50)
    
    # Convert date columns and create datetime index
    try:
        # Assuming German date format DD.MM.YY HH:MM
        df['datetime'] = pd.to_datetime(df['Date from'], format='%d.%m.%y %H:%M')
        df['date'] = df['datetime'].dt.date
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        df['weekday'] = df['datetime'].dt.dayofweek
        
        print(f"✅ Successfully parsed dates")
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"  Total duration: {(df['datetime'].max() - df['datetime'].min()).days} days")
        
        # Check for gaps in time series
        expected_freq = pd.infer_freq(df['datetime'])
        print(f"  Inferred frequency: {expected_freq}")
        
        # Time interval analysis
        time_diffs = df['datetime'].diff().dropna()
        print(f"  Time intervals - Mode: {time_diffs.mode().iloc[0]}")
        print(f"  Time intervals - Unique values: {time_diffs.unique()[:5]}")
        # Count each type of interval
        interval_counts = time_diffs.value_counts().sort_index()
        
        print("📊 Time Interval Frequencies:")
        for interval, count in interval_counts.items():
            percentage = (count / len(time_diffs)) * 100
            print(f"  {interval}: {count:,} times ({percentage:.4f}%)")
        normal_15min = pd.Timedelta('15 minutes')
        irregular_mask = time_diffs != normal_15min
        irregular_datetimes = df.loc[time_diffs[irregular_mask].index, 'datetime']

        print(f"\n📅 Irregular Interval Datetimes ({len(irregular_datetimes)} total):")
        for idx, dt in irregular_datetimes.items():
            interval = time_diffs.loc[idx]
            print(f"  {dt}: {interval}")

    except Exception as e:
        print(f"❌ Error parsing dates: {e}")
        print("  Manual date parsing may be needed")
    
    # ============================================================================
    # 4. TARGET VARIABLES ANALYSIS (Wind Power)
    # ============================================================================
    print("\n\n🌪️ 4. TARGET VARIABLES ANALYSIS")
    print("-"*50)
    
    # Clean target column names (remove extra spaces)
    target_cols_clean = []
    for col in target_cols:
        if col in df.columns:
            target_cols_clean.append(col)
        else:
            # Try to find similar column
            similar = [c for c in df.columns if 'Wind' in c and 'MW' in c]
            if similar:
                print(f"  Note: Column '{col}' not found. Found: {similar}")
                target_cols_clean.extend(similar)
    
    for col in target_cols_clean:
        print(f"\n📈 {col}:")
        series = pd.to_numeric(df[col], errors='coerce')
        
        print(f"  Statistics:")
        print(f"    Count: {series.count():,}")
        print(f"    Mean: {series.mean():.2f} MW")
        print(f"    Std: {series.std():.2f} MW")
        print(f"    Min: {series.min():.2f} MW")
        print(f"    Max: {series.max():.2f} MW")
        print(f"    Median: {series.median():.2f} MW")
        
        # Check for negative values (unusual for power generation)
        negative_count = (series < 0).sum()
        if negative_count > 0:
            print(f"    ⚠️ Negative values: {negative_count} ({negative_count/len(series)*100:.2f}%)")
        
        # Check for zero values
        zero_count = (series == 0).sum()
        print(f"    Zero values: {zero_count} ({zero_count/len(series)*100:.2f}%)")
    
    # ============================================================================
    # 5. SEASONAL & TEMPORAL PATTERNS
    # ============================================================================
    print("\n\n📈 5. SEASONAL & TEMPORAL PATTERNS")
    print("-"*50)
    
    if 'datetime' in df.columns:
        # Create wind power sum for analysis
        wind_cols = [col for col in df.columns if 'Wind' in col and 'MW' in col]
        if len(wind_cols) >= 2:
            df['Total_Wind'] = pd.to_numeric(df[wind_cols[0]], errors='coerce') + \
                              pd.to_numeric(df[wind_cols[1]], errors='coerce')
            
            print("🌪️ Total Wind Power Patterns:")
            
            # Monthly patterns
            monthly_avg = df.groupby('month')['Total_Wind'].mean()
            print(f"  Highest month: {monthly_avg.idxmax()} ({monthly_avg.max():.1f} MW)")
            print(f"  Lowest month: {monthly_avg.idxmin()} ({monthly_avg.min():.1f} MW)")
            
            # Daily patterns
            hourly_avg = df.groupby('hour')['Total_Wind'].mean()
            print(f"  Peak hour: {hourly_avg.idxmax()}:00 ({hourly_avg.max():.1f} MW)")
            print(f"  Low hour: {hourly_avg.idxmin()}:00 ({hourly_avg.min():.1f} MW)")
            
            # Weekly patterns
            weekly_avg = df.groupby('weekday')['Total_Wind'].mean()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            print(f"  Peak day: {days[weekly_avg.idxmax()]} ({weekly_avg.max():.1f} MW)")
            print(f"  Low day: {days[weekly_avg.idxmin()]} ({weekly_avg.min():.1f} MW)")
    
    # ============================================================================
    # 6. CORRELATION ANALYSIS
    # ============================================================================
    print("\n\n🔗 6. CORRELATION ANALYSIS")
    print("-"*50)
    offshore_col = 'Wind Offshore [MW] '  # Note the space
    onshore_col = 'Wind Onshore [MW]'

    if offshore_col in df.columns and onshore_col in df.columns:
        offshore = pd.to_numeric(df[offshore_col], errors='coerce')
        onshore = pd.to_numeric(df[onshore_col], errors='coerce')
        
        correlation = offshore.corr(onshore)
        
        print(f"🔗 Offshore vs Onshore Wind Correlation:")
        print(f"  Pearson correlation: {correlation:.4f}")
    
    # ============================================================================
    # 7. DATA COMPLETENESS FOR MODELING
    # ============================================================================
    print("\n\n🎯 7. DATA READINESS FOR TTM MODELING")
    print("-"*50)
    
    # Calculate the test set size (last year)
    if 'datetime' in df.columns:
        total_years = df['year'].nunique()
        last_year = df['year'].max()
        test_data = df[df['year'] == last_year]
        
        print(f"📊 Time Series Split Recommendations:")
        print(f"  Total years in dataset: {total_years}")
        print(f"  Last year (test set): {last_year}")
        print(f"  Test set size: {len(test_data):,} observations")
        print(f"  Test set duration: {len(test_data) * 15} minutes = {len(test_data) / 96:.1f} days")
        print(f"  Training set size: {len(df) - len(test_data):,} observations")
    
    # Check data frequency for TTM context length
    if 'datetime' in df.columns:
        freq_minutes = time_diffs.dt.total_seconds().median() / 60
        print(f"\n⏱️ Temporal Resolution Analysis:")
        print(f"  Data frequency: {freq_minutes:.0f} minutes")
    
    return df

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_wind_eda_plots(df):
    """
    Visualizations for wind power EDA with requested changes:
    """
    print("\n\n📊 CREATING VISUALIZATIONS...")
    
    # Set up the plotting layout - now 3x2 instead of 3x3
    fig = plt.figure(figsize=(16, 12))
    
    wind_cols = [col for col in df.columns if 'Wind' in col and 'MW' in col]
    offshore_col = wind_cols[0] if len(wind_cols) > 0 else None
    onshore_col = wind_cols[1] if len(wind_cols) > 1 else None
    
    # 1. Training Period Time Series (2019-2020)
    plt.subplot(3, 2, 1)
    if 'datetime' in df.columns and offshore_col and onshore_col:
        # Filter for training period
        train_mask = (df['year'] >= 2019) & (df['year'] <= 2020)
        train_data = df[train_mask]
        
        offshore_series = pd.to_numeric(train_data[offshore_col], errors='coerce')
        onshore_series = pd.to_numeric(train_data[onshore_col], errors='coerce')
        
        plt.plot(train_data['datetime'], offshore_series, 
                label='Wind Offshore', alpha=0.7, linewidth=0.3, color='blue')
        plt.plot(train_data['datetime'], onshore_series, 
                label='Wind Onshore', alpha=0.7, linewidth=0.3)
        
        plt.title('Wind Power Time Series - Training Period Subset (2019-2020)')
        plt.xlabel('Date')
        plt.ylabel('Power (MW)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 2. Test Period Time Series (2022)
    plt.subplot(3, 2, 2)
    if 'datetime' in df.columns and offshore_col and onshore_col:
        # Filter for test period  
        test_mask = (df['year'] == 2022)
        test_data = df[test_mask]
        
        offshore_series = pd.to_numeric(test_data[offshore_col], errors='coerce')
        onshore_series = pd.to_numeric(test_data[onshore_col], errors='coerce')
        
        plt.plot(test_data['datetime'], offshore_series, 
                label='Wind Offshore', alpha=0.7, linewidth=0.3, color='blue')
        plt.plot(test_data['datetime'], onshore_series, 
                label='Wind Onshore', alpha=0.7, linewidth=0.3)
        
        plt.title('Wind Power Time Series - Test Period (2022)')
        plt.xlabel('Date')
        plt.ylabel('Power (MW)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 3. Distribution plots (unchanged)
    plt.subplot(3, 2, 3)
    if offshore_col and onshore_col:
        offshore_series = pd.to_numeric(df[offshore_col], errors='coerce').dropna()
        onshore_series = pd.to_numeric(df[onshore_col], errors='coerce').dropna()
        
        plt.hist(offshore_series, bins=50, alpha=0.6, label='Wind Offshore', density=True)
        plt.hist(onshore_series, bins=50, alpha=0.6, label='Wind Onshore', density=True)
        
        plt.title('Wind Power Distribution')
        plt.xlabel('Power (MW)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 4. Hourly patterns - OFFSHORE WIND
    plt.subplot(3, 2, 4)
    if 'hour' in df.columns and offshore_col:
        offshore_series = pd.to_numeric(df[offshore_col], errors='coerce')
        df_temp = df.copy()
        df_temp['offshore_clean'] = offshore_series
        
        hourly_avg = df_temp.groupby('hour')['offshore_clean'].mean()
        hourly_std = df_temp.groupby('hour')['offshore_clean'].std()
        
        plt.plot(hourly_avg.index, hourly_avg.values, 'bo-', label='Mean', linewidth=2)
        plt.fill_between(hourly_avg.index, 
                        hourly_avg.values - hourly_std.values,
                        hourly_avg.values + hourly_std.values, 
                        alpha=0.3, label='±1 STD')
        
        plt.title('Hourly Wind Power Pattern - OFFSHORE')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Power (MW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2))
    
    # 5. Hourly patterns - ONSHORE WIND  
    plt.subplot(3, 2, 5)
    if 'hour' in df.columns and onshore_col:
        onshore_series = pd.to_numeric(df[onshore_col], errors='coerce')
        df_temp = df.copy()
        df_temp['onshore_clean'] = onshore_series
        
        hourly_avg = df_temp.groupby('hour')['onshore_clean'].mean()
        hourly_std = df_temp.groupby('hour')['onshore_clean'].std()
        
        plt.plot(hourly_avg.index, hourly_avg.values, 'ro-', label='Mean', linewidth=2)
        plt.fill_between(hourly_avg.index, 
                        hourly_avg.values - hourly_std.values,
                        hourly_avg.values + hourly_std.values, 
                        alpha=0.3, label='±1 STD', color='red')
        
        plt.title('Hourly Wind Power Pattern - ONSHORE')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Power (MW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2))
    
    # 6. Box plot by month (unchanged)
    plt.subplot(3, 2, 6)
    if 'month' in df.columns and 'Total_Wind' in df.columns:
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_data = [df[df['month'] == m]['Total_Wind'].dropna() for m in range(1, 13)]
        
        box_plot = plt.boxplot(month_data, labels=months, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.viridis(np.linspace(0, 1, 12))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title('Wind Power Distribution by Month')
        plt.xlabel('Month')
        plt.ylabel('Power (MW)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Additional plot for data completeness explanation
def explain_data_completeness(df):
    """
    Explain and visualize data completeness calculation
    """
    print("\n\n📊 DATA COMPLETENESS EXPLANATION")
    print("="*60)
    
    if 'datetime' in df.columns:
        # print("🔍 How 'Daily Data Completeness' is calculated:")
        # print("1. Resample data to daily frequency (group by date)")
        # print("2. Count number of observations per day")
        # print("3. Calculate percentage: (actual_count / expected_count) * 100")
        # print("4. Expected count = 96 observations/day (24h * 4 obs/hour)")
        
        # Demonstrate the calculation
        df_daily = df.set_index('datetime').resample('D').count()
        completeness = df_daily.iloc[:, 0] / 96 * 100  # 96 = 24*4 (15min intervals)
        
        print(f"\n📊 Completeness Statistics:")
        print(f"  Mean daily completeness: {completeness.mean():.2f}%")
        print(f"  Minimum daily completeness: {completeness.min():.2f}%")
        print(f"  Days with <100% completeness: {(completeness < 100).sum()}")
        print(f"  Days with <90% completeness: {(completeness < 90).sum()}")
        
        # Show specific low-completeness days
        low_completeness = completeness[completeness < 95]
        if len(low_completeness) > 0:
            print(f"\n⚠️ Days with <95% completeness:")
            for date, comp in low_completeness.head(5).items():
                count = df_daily.loc[date].iloc[0]
                print(f"  {date.date()}: {comp:.1f}% ({count}/96 observations)")
        
        # Create the plot with explanation
        plt.figure(figsize=(12, 4))
        plt.plot(completeness.index, completeness.values, linewidth=1)
        plt.axhline(y=100, color='green', linestyle='--', alpha=0.7, label='100% Complete')
        plt.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% Threshold')
        plt.fill_between(completeness.index, completeness.values, 100, 
                        where=(completeness.values < 100), alpha=0.3, color='red')
        
        plt.title('Daily Data Completeness Over Time\n(Expected: 96 observations/day at 15min intervals)')
        plt.xlabel('Date')
        plt.ylabel('Completeness (%)')
        plt.ylim(90, 101)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return completeness
    
    return None

def wind_scatter_analysis(df):
    """
    Enhanced scatter plot and analysis of offshore vs onshore wind relationship
    """
    wind_cols = [col for col in df.columns if 'Wind' in col and 'MW' in col]
    
    if len(wind_cols) >= 2:
        offshore_col = wind_cols[0]  # Usually 'Wind Offshore [MW] '
        onshore_col = wind_cols[1]   # Usually 'Wind Onshore [MW]'
        
        offshore = pd.to_numeric(df[offshore_col], errors='coerce')
        onshore = pd.to_numeric(df[onshore_col], errors='coerce')
        
        # Create enhanced scatter plot
        plt.figure(figsize=(12, 5))
        
        # Main scatter plot
        # plt.subplot(1, 2, 1)
        plt.scatter(offshore, onshore, alpha=0.3, s=0.5, c='blue')
        
        # Add correlation info
        correlation = offshore.corr(onshore)
        plt.text(0.05, 0.95, f'r = {correlation:.4f}', 
                transform=plt.gca().transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.xlabel(offshore_col)
        plt.ylabel(onshore_col)
        plt.title('Offshore vs Onshore Wind Power Relationship')
        plt.grid(True, alpha=0.3)
                
        # Detailed analysis
        print("\n🔍 WIND POWER RELATIONSHIP ANALYSIS")
        print("="*60)
        
        print(f"📊 Correlation Analysis:")
        print(f"  Pearson correlation: {correlation:.4f}")
                
        return correlation
    
    return None


def plot_weather_on_real_map(weather_df):
    """
    Plot weather grid points on real geographic map using Plotly
    """
    print("🗺️ CREATING REAL MAP VISUALIZATION")
    print("="*40)
    weather_df = weather_df.copy()
    # Create wind speed features
    # TODO: Dont let the features persist here but in feature engineering function
    weather_df['wind_speed_10m'] = np.sqrt(weather_df['u10']**2 + weather_df['v10']**2)
    weather_df['wind_speed_100m'] = np.sqrt(weather_df['u100']**2 + weather_df['v100']**2)
    
    # Get unique coordinates and their statistics
    coord_stats = weather_df.groupby(['longitude', 'latitude']).agg({
        'wind_speed_100m': ['mean', 'std', 'count'],
        'wind_speed_10m': 'mean',
        'msl': 'mean',
        't2m': 'mean'
    }).round(2)
    
    coord_stats.columns = ['wind_100m_mean', 'wind_100m_std', 'data_count', 'wind_10m_mean', 'pressure_mean', 'temp_mean']
    coord_stats = coord_stats.reset_index()
    
    # Convert units for better readability
    coord_stats['pressure_hpa'] = coord_stats['pressure_mean'] / 100
    coord_stats['temp_celsius'] = coord_stats['temp_mean'] - 273.15
    
    print(f"📍 Processing {len(coord_stats)} unique grid points")
    
    # German cities for reference
    cities_df = pd.DataFrame({
        'city': ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt', 'Stuttgart', 
                'Düsseldorf', 'Hannover', 'Bremen', 'Dresden'],
        'lat': [52.5, 53.6, 48.1, 50.9, 50.1, 48.8, 51.2, 52.4, 53.1, 51.1],
        'lon': [13.4, 10.0, 11.6, 6.9, 8.7, 9.2, 6.8, 9.7, 8.8, 13.7]
    })
    
    # # Create subplots for different maps (3x1 layout)
    # fig = make_subplots(
    #     rows=1, cols=3,
    #     subplot_titles=('Wind Speed at 100m Hub Height', 'Sea Level Pressure', 'Temperature'),
    #     specs=[[{"type": "scattermapbox"}, {"type": "scattermapbox"}, {"type": "scattermapbox"}]]
    # )
    
    # # Map 1: Wind speed at 100m
    # fig.add_trace(
    #     go.Scattermapbox(
    #         lat=coord_stats['latitude'],
    #         lon=coord_stats['longitude'],
    #         mode='markers',
    #         marker=dict(
    #             size=12,
    #             color=coord_stats['wind_100m_mean'],
    #             colorscale='Plasma',
    #             showscale=True,
    #             colorbar=dict(title="Wind Speed (m/s)", x=0.32)
    #         ),
    #         text=[f"Lat: {lat:.2f}<br>Lon: {lon:.2f}<br>Wind: {wind:.2f} m/s" 
    #               for lat, lon, wind in zip(coord_stats['latitude'], coord_stats['longitude'], coord_stats['wind_100m_mean'])],
    #         hovertemplate='<b>Wind Speed 100m</b><br>%{text}<extra></extra>',
    #         name='Wind Speed'
    #     ),
    #     row=1, col=1
    # )
    
    # # Add cities to first map
    # fig.add_trace(
    #     go.Scattermapbox(
    #         lat=cities_df['lat'],
    #         lon=cities_df['lon'],
    #         mode='markers+text',
    #         marker=dict(size=6, color='red'),
    #         text=cities_df['city'],
    #         textposition="top center",
    #         textfont=dict(size=8, color='black'),
    #         hovertemplate='<b>%{text}</b><extra></extra>',
    #         name='Major Cities'
    #     ),
    #     row=1, col=1
    # )
    
    # # Map 2: Pressure
    # fig.add_trace(
    #     go.Scattermapbox(
    #         lat=coord_stats['latitude'],
    #         lon=coord_stats['longitude'],
    #         mode='markers',
    #         marker=dict(
    #             size=12,
    #             color=coord_stats['pressure_hpa'],
    #             colorscale='RdBu_r',
    #             showscale=True,
    #             colorbar=dict(title="Pressure (hPa)", x=0.66)
    #         ),
    #         text=[f"Lat: {lat:.2f}<br>Lon: {lon:.2f}<br>Pressure: {press:.1f} hPa" 
    #               for lat, lon, press in zip(coord_stats['latitude'], coord_stats['longitude'], coord_stats['pressure_hpa'])],
    #         hovertemplate='<b>Sea Level Pressure</b><br>%{text}<extra></extra>',
    #         name='Pressure'
    #     ),
    #     row=1, col=2
    # )
    
    # # Map 3: Temperature
    # fig.add_trace(
    #     go.Scattermapbox(
    #         lat=coord_stats['latitude'],
    #         lon=coord_stats['longitude'],
    #         mode='markers',
    #         marker=dict(
    #             size=12,
    #             color=coord_stats['temp_celsius'],
    #             colorscale='RdYlBu_r',
    #             showscale=True,
    #             colorbar=dict(title="Temperature (°C)", x=1.02)
    #         ),
    #         text=[f"Lat: {lat:.2f}<br>Lon: {lon:.2f}<br>Temp: {temp:.1f}°C" 
    #               for lat, lon, temp in zip(coord_stats['latitude'], coord_stats['longitude'], coord_stats['temp_celsius'])],
    #         hovertemplate='<b>Temperature 2m</b><br>%{text}<extra></extra>',
    #         name='Temperature'
    #     ),
    #     row=1, col=3
    # )
    
    # # Update layout for all maps
    # for i in range(1, 4):
    #     mapbox_num = '' if i == 1 else str(i)
    #     fig.update_layout(**{
    #         f'mapbox{mapbox_num}': dict(
    #             style='open-street-map',
    #             center=dict(lat=51.5, lon=10.0),  # Center of Germany
    #             zoom=5.5
    #         )
    #     })
    
    # fig.update_layout(
    #     height=600,
    #     width=1400,
    #     title_text="Weather Data Spatial Analysis - Germany",
    #     title_x=0.5,
    #     showlegend=False,
    #     margin=dict(l=0, r=0, t=80, b=0)
    # )
    
    # fig.show()
    
    # Create additional distribution plots
    create_weather_distributions(weather_df, coord_stats)
    
    # Summary statistics
    print(f"\n🌍 GEOGRAPHIC ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"📊 Grid Coverage:")
    print(f"  Total weather stations: {len(coord_stats)}")
    print(f"  Longitude span: {coord_stats['longitude'].min():.2f}° to {coord_stats['longitude'].max():.2f}°")
    print(f"  Latitude span: {coord_stats['latitude'].min():.2f}° to {coord_stats['latitude'].max():.2f}°")
    print(f"  Data records per station: {coord_stats['data_count'].min():,} to {coord_stats['data_count'].max():,}")
    
    # Coastal vs inland analysis
    coastal_lat = 53.5  # Approximate latitude for German coast
    coastal_stations = coord_stats[coord_stats['latitude'] >= coastal_lat]
    inland_stations = coord_stats[coord_stats['latitude'] < coastal_lat]
    
    print(f"\n🌊 Coastal vs Inland Wind Analysis:")
    print(f"  Coastal stations (lat ≥ {coastal_lat}°): {len(coastal_stations)}")
    print(f"  Inland stations (lat < {coastal_lat}°): {len(inland_stations)}")
    
    if len(coastal_stations) > 0 and len(inland_stations) > 0:
        print(f"  Average coastal wind (100m): {coastal_stations['wind_100m_mean'].mean():.2f} m/s")
        print(f"  Average inland wind (100m): {inland_stations['wind_100m_mean'].mean():.2f} m/s")
        print(f"  Coastal wind advantage: {(coastal_stations['wind_100m_mean'].mean() / inland_stations['wind_100m_mean'].mean() - 1) * 100:.1f}%")
    
    # Wind resource hotspots
    top_wind_stations = coord_stats.nlargest(3, 'wind_100m_mean')
    print(f"\n🌪️ Top Wind Resource Locations:")
    for idx, station in top_wind_stations.iterrows():
        print(f"  {station['wind_100m_mean']:.2f} m/s at ({station['latitude']:.2f}°N, {station['longitude']:.2f}°E)")
    
    return coord_stats

def create_weather_distributions(weather_df, coord_stats):
    """
    Create distribution plots for weather variables
    """
    import matplotlib.pyplot as plt
    
    # Create matplotlib figure for distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Wind speed at 100m distribution
    ax1 = axes[0, 0]
    ax1.hist(weather_df['wind_speed_100m'], bins=60, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    ax1.set_xlabel('Wind Speed at 100m (m/s)')
    ax1.set_ylabel('Density')
    ax1.set_title('Wind Speed Distribution (100m Height)')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(weather_df['wind_speed_100m'].mean(), color='red', linestyle='--', 
                label=f'Mean: {weather_df["wind_speed_100m"].mean():.1f} m/s')
    ax1.legend()
    
    # Plot 2: Wind speed at 10m vs 100m scatter
    ax2 = axes[0, 1]
    sample_size = min(10000, len(weather_df))
    sample_idx = np.random.choice(len(weather_df), sample_size, replace=False)
    ax2.scatter(weather_df.iloc[sample_idx]['wind_speed_10m'], 
                weather_df.iloc[sample_idx]['wind_speed_100m'], 
                alpha=0.3, s=1, c='blue')
    ax2.plot([0, 25], [0, 25], 'r--', label='1:1 line')
    ax2.set_xlabel('Wind Speed at 10m (m/s)')
    ax2.set_ylabel('Wind Speed at 100m (m/s)')
    ax2.set_title('Wind Speed: 10m vs 100m Height')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Calculate correlation
    corr_10_100 = weather_df['wind_speed_10m'].corr(weather_df['wind_speed_100m'])
    ax2.text(0.05, 0.95, f'r = {corr_10_100:.3f}', transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Plot 3: Pressure distribution
    ax3 = axes[0, 2]
    pressure_hpa = weather_df['msl'] / 100
    ax3.hist(pressure_hpa, bins=50, alpha=0.7, color='lightcoral', edgecolor='black', density=True)
    ax3.set_xlabel('Mean Sea Level Pressure (hPa)')
    ax3.set_ylabel('Density')
    ax3.set_title('Pressure Distribution')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(pressure_hpa.mean(), color='blue', linestyle='--', 
                label=f'Mean: {pressure_hpa.mean():.0f} hPa')
    ax3.legend()
    
    # Plot 4: Temperature distribution
    ax4 = axes[1, 0]
    temp_celsius = weather_df['t2m'] - 273.15
    ax4.hist(temp_celsius, bins=50, alpha=0.7, color='lightgreen', edgecolor='black', density=True)
    ax4.set_xlabel('Temperature at 2m (°C)')
    ax4.set_ylabel('Density')
    ax4.set_title('Temperature Distribution')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(temp_celsius.mean(), color='red', linestyle='--', 
                label=f'Mean: {temp_celsius.mean():.1f}°C')
    ax4.legend()
    
    # Plot 5: Spatial wind speed variation
    ax5 = axes[1, 1]
    ax5.scatter(coord_stats['longitude'], coord_stats['wind_100m_mean'], 
                c=coord_stats['latitude'], cmap='viridis', s=60, alpha=0.7)
    ax5.set_xlabel('Longitude (°E)')
    ax5.set_ylabel('Average Wind Speed (m/s)')
    ax5.set_title('Wind Speed vs Longitude\n(Color = Latitude)')
    ax5.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax5.collections[0], ax=ax5)
    cbar.set_label('Latitude (°N)')
    
    # Plot 6: Pressure vs Wind relationship
    ax6 = axes[1, 2]
    sample_weather = weather_df.sample(min(5000, len(weather_df)))
    ax6.scatter(sample_weather['msl']/100, sample_weather['wind_speed_100m'], 
                alpha=0.3, s=1, c='purple')
    ax6.set_xlabel('Sea Level Pressure (hPa)')
    ax6.set_ylabel('Wind Speed at 100m (m/s)')
    ax6.set_title('Pressure vs Wind Speed')
    ax6.grid(True, alpha=0.3)
    
    # Calculate correlation
    corr_pressure_wind = (weather_df['msl']/100).corr(weather_df['wind_speed_100m'])
    ax6.text(0.05, 0.95, f'r = {corr_pressure_wind:.3f}', transform=ax6.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Create interactive map showing wind and weather patterns
def create_interactive_wind_analysis(coord_stats):
    """
    Create a focused interactive map for wind analysis
    """
    # Single focused map for wind analysis
    fig = go.Figure()
    
    # Add weather stations with wind speed
    fig.add_trace(go.Scattermapbox(
        lat=coord_stats['latitude'],
        lon=coord_stats['longitude'],
        mode='markers',
        marker=dict(
            size=15,
            color=coord_stats['wind_100m_mean'],
            colorscale='Jet',
            showscale=True,
            colorbar=dict(
                title="Average Wind Speed<br>at 100m Height (m/s)"
            ),
            cmin=coord_stats['wind_100m_mean'].min(),
            cmax=coord_stats['wind_100m_mean'].max()
        ),
        text=[f"<b>Weather Station</b><br>" +
              f"Location: {lat:.2f}°N, {lon:.2f}°E<br>" +
              f"Wind 100m: {wind100:.2f} m/s<br>" +
              f"Wind 10m: {wind10:.2f} m/s<br>" +
              f"Data points: {count:,}<br>" +
              f"Pressure: {press:.1f} hPa<br>" +
              f"Temperature: {temp:.1f}°C"
              for lat, lon, wind100, wind10, count, press, temp in 
              zip(coord_stats['latitude'], coord_stats['longitude'], 
                  coord_stats['wind_100m_mean'], coord_stats['wind_10m_mean'],
                  coord_stats['data_count'], coord_stats['pressure_hpa'], 
                  coord_stats['temp_celsius'])],
        hovertemplate='%{text}<extra></extra>',
        name='Weather Stations'
    ))
    
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=51.5, lon=10.0),
            zoom=6
        ),
        height=600,
        title="Interactive Wind Resource Map - Germany<br><sub>Hover over points for detailed information</sub>",
        title_x=0.5,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    fig.show()
    
    return fig