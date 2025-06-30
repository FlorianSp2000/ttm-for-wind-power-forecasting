"""
Simple Wind Power Feature Selection Results Analyzer
Analyzes experiment results to identify optimal feature configurations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import ast
import warnings
warnings.filterwarnings('ignore')

class ExperimentAnalyzer:
    """Simple analyzer for wind power feature selection experiments."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.experiments_df = None
        
        # Define the 5 base feature configurations from your shell script
        self.base_feature_configs = {
            'minimal_wind': {'u100', 'v100', 'u10', 'v10'},
            'wind_plus_temp': {'u100', 'v100', 'u10', 'v10', 't2m'},
            'wind_plus_pressure': {'u100', 'v100', 'u10', 'v10', 'msl'},
            'wind_extended': {'u100', 'v100', 'u10', 'v10', 't2m', 'msl', 'cdir'},
            'all_weather': {'u100', 'v100', 'u10', 'v10', 'msl', 't2m', 'cdir', 'blh', 'tcc', 'tp'}
        }
        
    def load_experiments(self):
        """Load all experiment configurations and results."""
        print("🔍 Loading experiment results...")
        
        experiment_dirs = [d for d in self.results_dir.iterdir() 
                          if d.is_dir() and not d.name.startswith('.')]
        
        experiments = []
        
        for exp_dir in experiment_dirs:
            try:
                metrics_file = exp_dir / "metrics.csv"
                metadata_file = exp_dir / "metadata.csv"
                
                if not (metrics_file.exists() and metadata_file.exists()):
                    continue
                
                # Load data
                metrics_df = pd.read_csv(metrics_file)
                metadata_df = pd.read_csv(metadata_file)
                
                # Parse experiment configuration
                config = self._parse_experiment_config(metadata_df)
                config['experiment_id'] = exp_dir.name
                config['experiment_dir'] = str(exp_dir)
                
                # Add performance metrics for both targets
                for target in ['wind_power_onshore', 'wind_power_offshore']:
                    target_metrics = metrics_df[metrics_df['target'] == target]
                    if target_metrics.empty:
                        continue
                    
                    overall = target_metrics[target_metrics['horizon'] == 'overall']
                    if overall.empty:
                        continue
                    
                    config[f'{target}_mae'] = overall['mae'].iloc[0]
                    config[f'{target}_rmse'] = overall['rmse'].iloc[0]
                
                experiments.append(config)
                
            except Exception as e:
                print(f"⚠️  Error processing {exp_dir.name}: {e}")
                continue
        
        self.experiments_df = pd.DataFrame(experiments)
        print(f"✅ Loaded {len(experiments)} experiments successfully")
        return self.experiments_df
    
    def _parse_experiment_config(self, metadata_df):
        """Parse experiment configuration from metadata."""
        config = {}
        
        for _, row in metadata_df.iterrows():
            section, param, value = row['section'], row['parameter'], row['value']
            
            # Get spatial configuration
            if section == 'experiment_info' and param == 'spatial_config':
                config['spatial_config'] = value
            
            # Get feature information
            elif section == 'feature_config':
                if param == 'observable_columns':
                    # Parse the feature list
                    features = ast.literal_eval(value)
                    config['all_features'] = features
                    config['num_features'] = len(features)
                    
                    # Classify the configuration
                    base_type, has_derived, has_temporal = self._classify_experiment(features)
                    config['base_feature_type'] = base_type
                    config['has_derived'] = has_derived
                    config['has_temporal'] = has_temporal
                    
                elif param == 'num_observables':
                    config['num_observables'] = int(value)
        
        # Set defaults
        defaults = {
            'spatial_config': 'unknown',
            'base_feature_type': 'unknown',
            'has_derived': False,
            'has_temporal': False,
            'num_features': 0
        }
        
        for key, default in defaults.items():
            if key not in config:
                config[key] = default
                
        return config
    
    def _normalize_feature_names(self, features):
        """Remove spatial suffixes from feature names to get base feature names."""
        normalized = set()
        
        for feature in features:
            # Remove coastal/inland suffixes
            if feature.endswith('_coastal') or feature.endswith('_inland'):
                base_feature = feature.rsplit('_', 1)[0]
                normalized.add(base_feature)
            # Remove latitude band suffixes (_lat47.2, _lat48.2, etc.)
            elif '_lat' in feature and feature.split('_lat')[1].replace('.', '').isdigit():
                base_feature = feature.split('_lat')[0]
                normalized.add(base_feature)
            else:
                # No suffix, keep as is
                normalized.add(feature)
        
        return normalized
    
    def _classify_experiment(self, features):
        """Classify experiment based on actual features used."""
        # First normalize feature names by removing spatial suffixes
        normalized_features = self._normalize_feature_names(features)
        
        # Check for derived features (added when derived_features=true)
        derived_features = {'wind_speed_10m', 'wind_speed_100m'}
        has_derived = bool(normalized_features & derived_features)
        
        # Check for temporal features (added when temporal_features=true)
        temporal_features = {
            'hour', 'month', 'year', 'dayofweek', 'dayofyear', 'quarter', 'weekofyear',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos'
        }
        has_temporal = bool(normalized_features & temporal_features)
        
        # Remove derived and temporal features to find base configuration
        base_features = normalized_features - derived_features - temporal_features
        
        # Also remove processed versions of base features that might be created
        processed_features = {'temperature_c', 'pressure_hpa', 'wind_dir_sin', 'wind_dir_cos', 'forecast_lead_hours'}
        base_features = base_features - processed_features
        
        # Find which base configuration matches (check from largest to smallest)
        order = ['all_weather', 'wind_extended', 'wind_plus_pressure', 'wind_plus_temp', 'minimal_wind']
        
        for base_type in order:
            if self.base_feature_configs[base_type].issubset(base_features):
                return base_type, has_derived, has_temporal
        
        # If no match found, check if minimal wind features are present
        minimal_required = self.base_feature_configs['minimal_wind']
        if not minimal_required.issubset(base_features):
            raise ValueError(f"Minimal wind features {minimal_required} not found in {base_features}")
        
        return 'custom', has_derived, has_temporal
    
    def get_best_configurations(self, target='wind_power_onshore', top_n=10):
        """Get best performing configurations for a target."""
        if self.experiments_df is None:
            self.load_experiments()
        
        target_mae_col = f'{target}_mae'
        if target_mae_col not in self.experiments_df.columns:
            print(f"❌ No data found for target: {target}")
            return pd.DataFrame()
        
        # Remove experiments without this target's data
        valid_data = self.experiments_df.dropna(subset=[target_mae_col])
        best = valid_data.nsmallest(top_n, target_mae_col)
        
        print(f"\n🏆 TOP {top_n} CONFIGURATIONS FOR {target.upper()}:")
        print("=" * 80)
        
        for i, (_, row) in enumerate(best.iterrows(), 1):
            print(f"{i:2d}. MAE: {row[target_mae_col]:.3f}")
            print(f"    Spatial: {row['spatial_config']:15} | Features: {row['num_features']:3d} | Type: {row['base_feature_type']}")
            print(f"    Derived: {str(row['has_derived']):5} | Temporal: {str(row['has_temporal']):5}")
            print(f"    ID: {row['experiment_id']}")
            print()
        
        return best
    
    def analyze_feature_impact(self, target='wind_power_onshore'):
        """Analyze impact of different feature configurations."""
        if self.experiments_df is None:
            self.load_experiments()
            
        target_mae_col = f'{target}_mae'
        if target_mae_col not in self.experiments_df.columns:
            return
        
        valid_data = self.experiments_df.dropna(subset=[target_mae_col])
        
        print(f"\n📊 FEATURE IMPACT ANALYSIS FOR {target.upper()}:")
        print("=" * 70)
        
        # Spatial configuration impact
        spatial_stats = valid_data.groupby('spatial_config')[target_mae_col].agg(['mean', 'std', 'count'])
        print("\n🗺️  Spatial Configuration Performance:")
        for spatial, stats in spatial_stats.iterrows():
            std_str = f" ± {stats['std']:.3f}" if not pd.isna(stats['std']) else ""
            print(f"   {spatial:15}: {stats['mean']:.3f}{std_str} (n={int(stats['count']):2d})")
        
        # Base feature type impact
        type_stats = valid_data.groupby('base_feature_type')[target_mae_col].agg(['mean', 'std', 'count'])
        print("\n🔧 Base Feature Type Performance:")
        for ftype, stats in type_stats.iterrows():
            std_str = f" ± {stats['std']:.3f}" if not pd.isna(stats['std']) else ""
            print(f"   {ftype:15}: {stats['mean']:.3f}{std_str} (n={int(stats['count']):2d})")
        
        # Derived features impact
        if valid_data['has_derived'].nunique() > 1:
            derived_stats = valid_data.groupby('has_derived')[target_mae_col].agg(['mean', 'count'])
            print("\n⚙️  Derived Features Impact:")
            for has_derived, stats in derived_stats.iterrows():
                label = "With Derived" if has_derived else "Without Derived"
                print(f"   {label:15}: {stats['mean']:.3f} (n={int(stats['count']):2d})")
            
            if True in derived_stats.index and False in derived_stats.index:
                impact = derived_stats.loc[True, 'mean'] - derived_stats.loc[False, 'mean']
                print(f"   Impact: {impact:+.3f} MAE")
        
        # Temporal features impact
        if valid_data['has_temporal'].nunique() > 1:
            temporal_stats = valid_data.groupby('has_temporal')[target_mae_col].agg(['mean', 'count'])
            print("\n📅 Temporal Features Impact:")
            for has_temporal, stats in temporal_stats.iterrows():
                label = "With Temporal" if has_temporal else "Without Temporal"
                print(f"   {label:15}: {stats['mean']:.3f} (n={int(stats['count']):2d})")
            
            if True in temporal_stats.index and False in temporal_stats.index:
                impact = temporal_stats.loc[True, 'mean'] - temporal_stats.loc[False, 'mean']
                print(f"   Impact: {impact:+.3f} MAE")
    
    def verify_experiment_counts(self):
        """Verify that experiment counts match expected distribution."""
        if self.experiments_df is None:
            self.load_experiments()
        
        print(f"\n🔍 EXPERIMENT COUNT VERIFICATION:")
        print("=" * 50)
        print(f"Total experiments: {len(self.experiments_df)}")
        print(f"Expected: 60 (3 spatial × 5 features × 2 derived × 2 temporal)")
        print()
        
        # Check spatial distribution (should be 20 each)
        spatial_counts = self.experiments_df['spatial_config'].value_counts()
        print("Spatial configuration counts:")
        for spatial, count in spatial_counts.items():
            print(f"   {spatial:15}: {count:2d} (expected: 20)")
        print()
        
        # Check base feature distribution (should be 12 each)
        feature_counts = self.experiments_df['base_feature_type'].value_counts()
        print("Base feature type counts:")
        for ftype, count in feature_counts.items():
            print(f"   {ftype:15}: {count:2d} (expected: 12)")
        print()
        
        # Check derived/temporal distribution (should be 30 each)
        derived_counts = self.experiments_df['has_derived'].value_counts()
        print("Derived feature counts:")
        for has_derived, count in derived_counts.items():
            label = "With derived" if has_derived else "Without derived"
            print(f"   {label:15}: {count:2d} (expected: 30)")
        print()
        
        temporal_counts = self.experiments_df['has_temporal'].value_counts()
        print("Temporal feature counts:")
        for has_temporal, count in temporal_counts.items():
            label = "With temporal" if has_temporal else "Without temporal"
            print(f"   {label:15}: {count:2d} (expected: 30)")
    
    def create_summary_plots(self, save_dir=None):
        """Create summary visualizations."""
        if self.experiments_df is None:
            self.load_experiments()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        targets = ['wind_power_onshore', 'wind_power_offshore']
        
        for i, target in enumerate(targets):
            target_mae_col = f'{target}_mae'
            if target_mae_col not in self.experiments_df.columns:
                continue
            
            valid_data = self.experiments_df.dropna(subset=[target_mae_col])
            
            # Performance by spatial config
            ax1 = axes[i, 0]
            spatial_stats = valid_data.groupby('spatial_config')[target_mae_col].agg(['mean', 'std'])
            bars = ax1.bar(spatial_stats.index, spatial_stats['mean'], 
                          yerr=spatial_stats['std'], capsize=5)
            ax1.set_title(f'{target.replace("_", " ").title()} - Spatial Performance')
            ax1.set_ylabel('MAE')
            ax1.grid(True, alpha=0.3)
            
            # Performance by base feature type
            ax2 = axes[i, 1]
            type_stats = valid_data.groupby('base_feature_type')[target_mae_col].agg(['mean', 'std'])
            bars = ax2.bar(range(len(type_stats)), type_stats['mean'], 
                          yerr=type_stats['std'], capsize=5)
            ax2.set_xticks(range(len(type_stats)))
            ax2.set_xticklabels(type_stats.index, rotation=45, ha='right')
            ax2.set_title(f'{target.replace("_", " ").title()} - Feature Type Performance')
            ax2.set_ylabel('MAE')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / "feature_analysis_summary.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Summary plots saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, save_dir=None):
        """Generate comprehensive summary report."""
        if self.experiments_df is None:
            self.load_experiments()
        
        report = []
        report.append("🔬 WIND POWER FEATURE SELECTION ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Total experiments: {len(self.experiments_df)}")
        report.append(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for target in ['wind_power_onshore', 'wind_power_offshore']:
            target_mae_col = f'{target}_mae'
            if target_mae_col not in self.experiments_df.columns:
                continue
            
            valid_data = self.experiments_df.dropna(subset=[target_mae_col])
            best = valid_data.loc[valid_data[target_mae_col].idxmin()]
            
            report.append(f"🏆 BEST CONFIGURATION - {target.upper()}:")
            report.append(f"   MAE: {best[target_mae_col]:.3f}")
            report.append(f"   Spatial: {best['spatial_config']}")
            report.append(f"   Base Features: {best['base_feature_type']} ({best['num_features']} total features)")
            report.append(f"   Derived: {best['has_derived']}, Temporal: {best['has_temporal']}")
            report.append(f"   Experiment: {best['experiment_id']}")
            report.append("")
        
        report_text = "\n".join(report)
        print(report_text)
        
        if save_dir:
            with open(Path(save_dir) / "analysis_report.txt", 'w') as f:
                f.write(report_text)
        
        return report_text

def main():
    parser = argparse.ArgumentParser(description='Analyze wind power feature selection results')
    parser.add_argument('results_dir', help='Directory containing experiment results')
    parser.add_argument('--output_dir', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(exist_ok=True)
    
    # Run analysis
    analyzer = ExperimentAnalyzer(args.results_dir)
    analyzer.load_experiments()
    
    # Verify experiment counts
    analyzer.verify_experiment_counts()
    
    # Analyze both targets
    for target in ['wind_power_onshore', 'wind_power_offshore']:
        analyzer.get_best_configurations(target=target)
        analyzer.analyze_feature_impact(target=target)
    
    # Create visualizations and report
    analyzer.create_summary_plots(save_dir=args.output_dir)
    analyzer.generate_report(save_dir=args.output_dir)
    
    # Save results
    if args.output_dir:
        analyzer.experiments_df.to_csv(Path(args.output_dir) / "all_experiments.csv", index=False)
        print(f"\n📁 Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()