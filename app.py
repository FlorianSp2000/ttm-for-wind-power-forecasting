import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from pathlib import Path
import json

# Set page config
st.set_page_config(
    page_title="Wind Power Forecast Evaluation Dashboard",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stDataFrame {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_available_runs(results_dir="results/"):
    """
    Scan results directory and return available runs
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return []
    
    runs = []
    for run_dir in results_path.iterdir():
        if run_dir.is_dir() and not run_dir.name.startswith('.'):
            # Check if required files exist
            metadata_file = run_dir / "metadata.csv"
            metrics_file = run_dir / "metrics.csv"
            predictions_file = run_dir / "predictions.npz"
            
            if all(f.exists() for f in [metadata_file, metrics_file, predictions_file]):
                runs.append({
                    'name': run_dir.name,
                    'path': run_dir,
                    'metadata_file': metadata_file,
                    'metrics_file': metrics_file,
                    'predictions_file': predictions_file
                })
    
    # Sort by name (timestamp first)
    runs.sort(key=lambda x: x['name'], reverse=True)
    return runs

def load_run_data(run_info):
    """
    Load all data for a specific run
    """
    # Load metadata
    metadata_df = pd.read_csv(run_info['metadata_file'])
    metadata = {}
    for _, row in metadata_df.iterrows():
        if row['section'] not in metadata:
            metadata[row['section']] = {}
        metadata[row['section']][row['parameter']] = row['value']
    
    # Load metrics
    metrics_df = pd.read_csv(run_info['metrics_file'])
    
    # Load predictions
    try:
        predictions_data = np.load(run_info['predictions_file'])
        # Convert to regular dict to ensure serializability
        predictions_data = {key: predictions_data[key] for key in predictions_data.files}
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        predictions_data = {}
    
    return metadata, metrics_df, predictions_data

def load_ground_truth(results_dir="results/"):
    """
    Load ground truth data (shared across all runs)
    """
    ground_truth_path = Path(results_dir) / "ground_truth.npz"
    if ground_truth_path.exists():
        try:
            gt_data = np.load(ground_truth_path)
            # Convert to regular dict to ensure serializability
            return {key: gt_data[key] for key in gt_data.files}
        except Exception as e:
            st.error(f"Error loading ground truth: {e}")
            return None
    return None

def clean_run_name_for_legend(run_name):
    """
    Clean run name for legend display - remove datetime and shorten
    """
    # Remove datetime prefix (format: YYYY-MM-DD_HH-MM-SS_)
    import re
    cleaned = re.sub(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_', '', run_name)
    
    # Further shorten if still too long
    if len(cleaned) > 15:
        cleaned = cleaned[:12] + '...'
    
    return cleaned

def sort_horizons_properly(metrics_df):
    """
    Sort horizons properly from 1h to 24h (numeric sorting)
    """
    horizon_data = metrics_df[metrics_df['horizon'] != 'overall'].copy()
    if horizon_data.empty:
        return horizon_data
    
    # Convert horizon to numeric for proper sorting
    horizon_data['horizon_num'] = horizon_data['horizon'].str.replace('h', '').astype(int)
    horizon_data = horizon_data.sort_values(['target', 'horizon_num'])
    return horizon_data

def style_metrics_table(df, metric_col):
    """
    Apply color gradient styling to metrics table
    """
    def highlight_errors(s):
        # Use red-blue gradient: high errors = red, low errors = blue
        norm_s = (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else s * 0
        colors = []
        for val in norm_s:
            if pd.isna(val):
                colors.append('')
            else:
                # Red for high error, blue for low error
                red = int(255 * val)
                blue = int(255 * (1 - val))
                green = int(128 * (1 - val))
                colors.append(f'background-color: rgb({red}, {green}, {blue}); color: white')
        return colors
    
    return df.style.apply(highlight_errors, subset=[metric_col])

def create_metrics_table(metrics_df, run_name):
    """
    Create a formatted metrics table for display with color coding
    """
    st.subheader(f"📊 Metrics: {run_name}")
    
    # Separate overall metrics
    overall_metrics = metrics_df[metrics_df['horizon'] == 'overall'].copy()
    horizon_metrics = sort_horizons_properly(metrics_df)
    
    # Display overall metrics prominently
    if not overall_metrics.empty:
        st.markdown("**Overall Performance:**")
        overall_cols = st.columns(len(overall_metrics))
        
        for i, (_, row) in enumerate(overall_metrics.iterrows()):
            with overall_cols[i]:
                st.metric(
                    label=f"{row['target'].replace('wind_power_', '').title()}",
                    value=f"MAE: {row['mae']:.1f} MW",
                    delta=f"RMSE: {row['rmse']:.1f} MW"
                )
    
    # Display horizon-specific metrics with color coding
    if not horizon_metrics.empty:
        for target in horizon_metrics['target'].unique():
            target_data = horizon_metrics[horizon_metrics['target'] == target].copy()
            
            # Brief target label
            st.markdown(f"**{target.replace('wind_power_', '').title()}:**")
            
            # Create display dataframes with proper column names
            display_df = target_data[['horizon', 'mae', 'rmse']].copy()
            display_df['mae'] = display_df['mae'].round(2)
            display_df['rmse'] = display_df['rmse'].round(2)
            display_df.columns = ['Horizon', 'MAE (MW)', 'RMSE (MW)']
            
            # Create two columns for side-by-side styled tables
            col1, col2 = st.columns(2)
            
            with col1:
                styled_mae = style_metrics_table(display_df[['Horizon', 'MAE (MW)']], 'MAE (MW)')
                st.dataframe(styled_mae, use_container_width=True, hide_index=True)
            
            with col2:
                styled_rmse = style_metrics_table(display_df[['Horizon', 'RMSE (MW)']], 'RMSE (MW)')
                st.dataframe(styled_rmse, use_container_width=True, hide_index=True)

def create_interactive_comparison_plot(predictions_data_list, ground_truth, run_names, 
                                     start_idx=0, length=500):
    """
    Create interactive comparison plot using Plotly
    """
    
    # Define forecast horizons (1h, 6h, 12h, 18h, 24h ahead)
    horizons = [0, 5, 11, 17, 23]  # 0-indexed
    horizon_labels = ['1h', '6h', '12h', '18h', '24h']
    warm_colors = ['#FF6B35', '#F7931E', '#FFD23F', '#EE4B2B', '#FF8C42']  # Orange/red/yellow warm palette
    
    # Prepare ground truth data
    offshore_truth = ground_truth['wind_power_offshore']
    onshore_truth = ground_truth['wind_power_onshore']
    
    plot_end = min(start_idx + length, len(offshore_truth))
    time_range = list(range(start_idx, plot_end))
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Offshore', 'Onshore'),  # Simplified since legends have full titles
        vertical_spacing=0.25  # Much more spacing to accommodate upper plot legend
    )
    
    targets = ['offshore', 'onshore']
    truth_data = [offshore_truth, onshore_truth]
    
    for target_idx, (target, truth) in enumerate(zip(targets, truth_data)):
        row = target_idx + 1
        
        # Plot ground truth
        legend_ref = "legend" if target_idx == 0 else "legend2"
        fig.add_trace(
            go.Scatter(
                x=time_range, 
                y=truth[start_idx:plot_end],
                mode='lines',
                name=f'GT',  # Shortened name since we have separate legends
                line=dict(color='#666666', width=2),  # Grey color
                opacity=0.6,  # Lower opacity for better visibility
                legend=legend_ref,
                showlegend=True
            ),
            row=row, col=1
        )
        
        # Plot predictions for each run
        for run_idx, (pred_data, run_name) in enumerate(zip(predictions_data_list, run_names)):
            pred_key = f'wind_power_{target}'
            if pred_key in pred_data:
                pred_array = pred_data[pred_key]
                
                # For multiple runs, use different line styles
                line_dash = 'solid' if run_idx == 0 else 'dash'
                opacity = 0.8 if run_idx == 0 else 0.6
                
                # Plot only selected horizons to avoid clutter when comparing runs
                selected_horizons = [0, 23] if len(run_names) > 1 else horizons
                selected_labels = ['1h', '24h'] if len(run_names) > 1 else horizon_labels
                selected_colors = ['#FF6B35', '#EE4B2B'] if len(run_names) > 1 else warm_colors  # Red-orange for comparison
                
                for h_idx, h_label, color in zip(selected_horizons, selected_labels, selected_colors):
                    horizon_hours = h_idx + 1
                    pred_start = start_idx + horizon_hours
                    pred_end = min(pred_start + len(pred_array), plot_end)
                    
                    if pred_start < plot_end and pred_start < len(pred_array):
                        pred_range = list(range(pred_start, pred_end))
                        pred_values = pred_array[start_idx:start_idx + len(pred_range), h_idx]
                        
                        clean_run_name = clean_run_name_for_legend(run_name)
                        label = f'{clean_run_name} {h_label}'
                        
                        fig.add_trace(
                            go.Scatter(
                                x=pred_range,
                                y=pred_values,
                                mode='lines',
                                name=label,
                                line=dict(color=color, width=2, dash=line_dash),
                                opacity=opacity,
                                legend=legend_ref,  # Use same legend as ground truth for this target
                                showlegend=True
                            ),
                            row=row, col=1
                        )
    
    # Update layout
    fig.update_layout(
        height=750,  # Increased height for better spacing
        title_text="Wind Power Forecast Analysis",
        title_y=0.98,  # Move title higher
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,  # Move legend lower to give title more space
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=80, b=100)  # Add margins for title and legend
    )
    
    fig.update_xaxes(title_text="Time Step (Hours)")
    fig.update_yaxes(title_text="Power (MW)")
    
    return fig

def create_horizon_error_barplots(metrics_dfs, run_names):
    """
    Create bar plots showing error vs horizon for MAE and RMSE
    """
    st.subheader("📊 Error vs Horizon Analysis")
    
    # Prepare data for plotting
    plot_data = []
    
    for run_idx, (metrics_df, run_name) in enumerate(zip(metrics_dfs, run_names)):
        horizon_data = sort_horizons_properly(metrics_df)
        
        for _, row in horizon_data.iterrows():
            target_clean = row['target'].replace('wind_power_', '').title()
            clean_run_name = clean_run_name_for_legend(run_name)
            plot_data.append({
                'Run': clean_run_name,
                'Horizon': row['horizon_num'],
                'Target': target_clean,
                'MAE': row['mae'],
                'RMSE': row['rmse'],
                'Run_Target': f"{clean_run_name} - {target_clean}"
            })
    
    if not plot_data:
        st.warning("No horizon data available for plotting")
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create subplots for MAE and RMSE
    col1, col2 = st.columns(2)
    
    with col1:
        fig_mae = px.bar(
            plot_df, 
            x='Horizon', 
            y='MAE', 
            color='Run_Target',
            title='Mean Absolute Error by Forecast Horizon',
            labels={'MAE': 'MAE (MW)', 'Horizon': 'Forecast Horizon (hours)'},
            height=500,
            barmode='group'  # Side-by-side bars instead of stacked
        )
        fig_mae.update_layout(
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            hovermode='x unified'  # Shared tooltips
        )
        st.plotly_chart(fig_mae, use_container_width=True)
    
    with col2:
        fig_rmse = px.bar(
            plot_df, 
            x='Horizon', 
            y='RMSE', 
            color='Run_Target',
            title='Root Mean Square Error by Forecast Horizon',
            labels={'RMSE': 'RMSE (MW)', 'Horizon': 'Forecast Horizon (hours)'},
            height=500,
            barmode='group'  # Side-by-side bars instead of stacked
        )
        fig_rmse.update_layout(
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            hovermode='x unified'  # Shared tooltips
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

def create_metrics_comparison(metrics_dfs, run_names):
    """
    Create side-by-side metrics comparison with color coding
    """
    st.subheader("📈 Metrics Comparison")
    
    # Get targets
    targets = metrics_dfs[0]['target'].unique()
    
    for target in targets:
        # Brief target label
        st.markdown(f"**{target.replace('wind_power_', '').title()}:**")
        
        # Create comparison dataframe
        comparison_data = []
        
        for run_idx, (metrics_df, run_name) in enumerate(zip(metrics_dfs, run_names)):
            target_data = sort_horizons_properly(metrics_df)
            target_data = target_data[target_data['target'] == target]
            
            for _, row in target_data.iterrows():
                comparison_data.append({
                    'Run': run_name[:30] + '...' if len(run_name) > 30 else run_name,
                    'Horizon': row['horizon_num'],
                    'MAE (MW)': round(row['mae'], 2),
                    'RMSE (MW)': round(row['rmse'], 2)
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Horizon')
            
            # Create pivot for better comparison
            mae_pivot = comparison_df.pivot(index='Horizon', columns='Run', values='MAE (MW)')
            rmse_pivot = comparison_df.pivot(index='Horizon', columns='Run', values='RMSE (MW)')
            
            col1, col2 = st.columns(2)
            
            with col1:
                if len(mae_pivot.columns) > 1:
                    # Color coding for comparison
                    styled_mae_comp = mae_pivot.style.background_gradient(
                        cmap='RdYlBu_r', axis=None
                    ).format(precision=2)
                    st.dataframe(styled_mae_comp, use_container_width=True)
                else:
                    st.dataframe(mae_pivot, use_container_width=True)
            
            with col2:
                if len(rmse_pivot.columns) > 1:
                    # Color coding for comparison
                    styled_rmse_comp = rmse_pivot.style.background_gradient(
                        cmap='RdYlBu_r', axis=None
                    ).format(precision=2)
                    st.dataframe(styled_rmse_comp, use_container_width=True)
                else:
                    st.dataframe(rmse_pivot, use_container_width=True)

def main():
    st.title("🌪️ Wind Power Forecast Evaluation Dashboard")
    st.markdown("Compare different model runs and analyze forecast performance across multiple horizons.")
    
    # Sidebar for run selection
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Load available runs
        results_dir = st.text_input("Results Directory", value="results/")
        available_runs = load_available_runs(results_dir)
        
        if not available_runs:
            st.error(f"No runs found in {results_dir}")
            st.stop()
        
        # Run selection
        run_options = [run['name'] for run in available_runs]
        selected_runs = st.multiselect(
            "Select Runs to Compare (max 2):",
            options=run_options,
            default=run_options[:1] if run_options else [],
            max_selections=2
        )
        
        if not selected_runs:
            st.warning("Please select at least one run to analyze.")
            st.stop()
        
        # Plot configuration
        st.subheader("📊 Plot Settings")
        start_idx = st.number_input("Start Time Step", min_value=0, value=512)
        length = st.number_input("Plot Length (hours)", min_value=100, max_value=2000, value=500)
    
    # Load data for selected runs
    selected_run_data = []
    for run_name in selected_runs:
        run_info = next(run for run in available_runs if run['name'] == run_name)
        selected_run_data.append((run_info, *load_run_data(run_info)))
    
    # Load ground truth
    ground_truth = load_ground_truth(results_dir)
    if ground_truth is None:
        st.error("Ground truth data not found. Make sure evaluation has been run.")
        st.stop()
    
    # Display results
    if len(selected_runs) == 1:
        # Single run analysis
        run_info, metadata, metrics_df, predictions_data = selected_run_data[0]
        
        # Show run info
        st.header(f"📋 Run Analysis: {selected_runs[0]}")
        
        # Metadata summary
        with st.expander("🔍 Run Configuration", expanded=False):
            config_cols = st.columns(3)
            
            with config_cols[0]:
                st.markdown("**Model Config:**")
                if 'model_config' in metadata:
                    for key, value in metadata['model_config'].items():
                        st.text(f"{key}: {value}")
            
            with config_cols[1]:
                st.markdown("**Feature Config:**")
                if 'feature_config' in metadata:
                    for key, value in metadata['feature_config'].items():
                        if not key.endswith('_columns'):
                            st.text(f"{key}: {value}")
            
            with config_cols[2]:
                st.markdown("**Experiment Info:**")
                if 'experiment_info' in metadata:
                    for key, value in metadata['experiment_info'].items():
                        st.text(f"{key}: {value}")
        
        # Metrics table with color coding
        create_metrics_table(metrics_df, selected_runs[0])
        
        # Interactive plot
        st.subheader("📈 Forecast Visualization")
        fig = create_interactive_comparison_plot([predictions_data], ground_truth, [selected_runs[0]], 
                                               start_idx, length)
        st.plotly_chart(fig, use_container_width=True)
        
        # Horizon error analysis
        create_horizon_error_barplots([metrics_df], [selected_runs[0]])
        
    else:
        # Two-run comparison
        st.header(f"⚖️ Run Comparison")
        
        # Extract data
        run_infos = [data[0] for data in selected_run_data]
        metadatas = [data[1] for data in selected_run_data]
        metrics_dfs = [data[2] for data in selected_run_data]
        predictions_datas = [data[3] for data in selected_run_data]
        
        # Show only metrics comparison (no individual tables)
        create_metrics_comparison(metrics_dfs, selected_runs)
        
        # Interactive comparison plot
        st.subheader("📈 Forecast Comparison")
        fig = create_interactive_comparison_plot(predictions_datas, ground_truth, selected_runs, 
                                               start_idx, length)
        st.plotly_chart(fig, use_container_width=True)
        
        # Horizon error analysis
        create_horizon_error_barplots(metrics_dfs, selected_runs)
        
        # Configuration comparison
        with st.expander("🔍 Configuration Comparison", expanded=False):
            comp_cols = st.columns(2)
            
            for i, (metadata, run_name) in enumerate(zip(metadatas, selected_runs)):
                with comp_cols[i]:
                    st.markdown(f"**{run_name}:**")
                    
                    if 'model_config' in metadata:
                        st.markdown("*Model:*")
                        for key, value in metadata['model_config'].items():
                            st.text(f"  {key}: {value}")
                    
                    if 'feature_config' in metadata:
                        st.markdown("*Features:*")
                        for key, value in metadata['feature_config'].items():
                            if not key.endswith('_columns'):
                                st.text(f"  {key}: {value}")

if __name__ == "__main__":
    main()