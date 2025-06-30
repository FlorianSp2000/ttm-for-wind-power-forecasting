"""
Permutation Importance Results Analyzer
Analyzes permutation importance results to identify most valuable features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class PermutationResultsAnalyzer:
    """Analyzer for permutation feature importance results."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.permutation_results = {}
        self.validation_results = None
        
        # Feature categories for analysis
        self.feature_categories = {
            'core_wind': ['u100', 'v100', 'u10', 'v10'],
            'additional_weather': ['msl', 't2m', 'cdir', 'blh', 'tcc', 'tp'],
            'processed': ['pressure_hpa', 'temperature_c', 'wind_dir_sin', 'wind_dir_cos'],
            'derived': ['wind_speed_10m', 'wind_speed_100m'],
            'meta': ['forecast_lead_hours']
        }
        
    def load_results(self):
        """Load all permutation importance results."""
        print("🔍 Loading permutation importance results...")
        
        # Load permutation importance files
        json_files = list(self.results_dir.glob("*_permutation_importance.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                experiment_name = json_file.stem.replace('_permutation_importance', '')
                self.permutation_results[experiment_name] = data
                print(f"   ✅ Loaded {experiment_name}")
                
            except Exception as e:
                print(f"   ⚠️  Error loading {json_file.name}: {e}")
        
        # Load validation results if available
        validation_dir = self.results_dir / "validation"
        if validation_dir.exists():
            self.validation_results = self._load_validation_results(validation_dir)
            print(f"   ✅ Loaded {len(self.validation_results)} validation experiments")
        
        print(f"📊 Total permutation experiments loaded: {len(self.permutation_results)}")
        
    def _load_validation_results(self, validation_dir):
        """Load validation experiment results for comparison."""
        results = []
        
        for exp_dir in validation_dir.iterdir():
            if not exp_dir.is_dir():
                continue
                
            try:
                metrics_file = exp_dir / "metrics.csv"
                if metrics_file.exists():
                    metrics_df = pd.read_csv(metrics_file)
                    
                    # Extract overall performance for both targets
                    for target in ['wind_power_onshore', 'wind_power_offshore']:
                        target_metrics = metrics_df[metrics_df['target'] == target]
                        if not target_metrics.empty:
                            overall = target_metrics[target_metrics['horizon'] == 'overall']
                            if not overall.empty:
                                results.append({
                                    'experiment_id': exp_dir.name,
                                    'target': target,
                                    'mae': overall['mae'].iloc[0],
                                    'rmse': overall['rmse'].iloc[0]
                                })
            except Exception as e:
                continue
                
        return pd.DataFrame(results)
    
    def analyze_feature_importance_rankings(self):
        """Analyze and rank feature importance across all experiments."""
        print("\n🏆 FEATURE IMPORTANCE RANKINGS")
        print("=" * 50)
        
        # Aggregate importance scores across experiments
        importance_summary = defaultdict(lambda: defaultdict(list))
        
        for exp_name, data in self.permutation_results.items():
            importance_scores = data['importance_scores']
            
            for target in importance_scores:
                for feature, scores in importance_scores[target].items():
                    importance_summary[target][feature].append(scores['mean_importance'])
        
        # Calculate statistics and rank features
        for target in importance_summary:
            print(f"\n📊 {target.upper()} - FEATURE IMPORTANCE RANKING:")
            print("-" * 50)
            
            feature_stats = []
            for feature, scores in importance_summary[target].items():
                feature_stats.append({
                    'feature': feature,
                    'mean_importance': np.mean(scores),
                    'std_importance': np.std(scores) if len(scores) > 1 else 0,
                    'max_importance': np.max(scores),
                    'experiments': len(scores),
                    'category': self._categorize_feature(feature)
                })
            
            # Sort by mean importance
            feature_stats.sort(key=lambda x: x['mean_importance'], reverse=True)
            
            print("Rank | Feature               | Mean Imp. | Max Imp.  | Std Dev | Category    | Exps")
            print("-" * 85)
            
            for rank, stats in enumerate(feature_stats[:15], 1):  # Top 15
                print(f"{rank:4d} | {stats['feature']:20s} | "
                      f"{stats['mean_importance']:8.4f} | {stats['max_importance']:8.4f} | "
                      f"{stats['std_importance']:7.4f} | {stats['category']:10s} | {stats['experiments']:4d}")
            
            # Highlight high-impact features
            high_impact = [f for f in feature_stats if f['mean_importance'] > 0.5]
            if high_impact:
                print(f"\n🔥 HIGH-IMPACT FEATURES (>0.5 MAE degradation):")
                for f in high_impact:
                    print(f"   • {f['feature']} ({f['mean_importance']:.3f} MAE impact)")
    
    def _categorize_feature(self, feature_name):
        """Categorize feature by type."""
        for category, features in self.feature_categories.items():
            if feature_name in features:
                return category
        return 'other'
    
    def analyze_feature_categories(self):
        """Analyze importance by feature category."""
        print("\n📊 FEATURE CATEGORY ANALYSIS")
        print("=" * 40)
        
        for exp_name, data in self.permutation_results.items():
            print(f"\n🔬 Experiment: {exp_name}")
            print("-" * 30)
            
            importance_scores = data['importance_scores']
            
            for target in importance_scores:
                print(f"\n{target.upper()}:")
                
                # Group by category
                category_importance = defaultdict(list)
                for feature, scores in importance_scores[target].items():
                    category = self._categorize_feature(feature)
                    category_importance[category].append({
                        'feature': feature,
                        'importance': scores['mean_importance']
                    })
                
                # Show category summaries
                for category, features in category_importance.items():
                    if not features:
                        continue
                        
                    avg_importance = np.mean([f['importance'] for f in features])
                    max_importance = max([f['importance'] for f in features])
                    
                    print(f"  {category.upper():15s}: avg={avg_importance:.3f}, max={max_importance:.3f}")
                    
                    # Show top features in category
                    features.sort(key=lambda x: x['importance'], reverse=True)
                    for f in features[:3]:  # Top 3 in category
                        print(f"    • {f['feature']:20s}: {f['importance']:6.3f}")
    
    def compare_minimal_vs_all(self):
        """Compare minimal vs all feature set results."""
        print("\n⚔️  MINIMAL vs ALL FEATURE COMPARISON")
        print("=" * 45)
        
        # Look for experiments with minimal and all configurations
        minimal_results = {}
        all_results = {}
        
        for exp_name, data in self.permutation_results.items():
            if 'minimal' in exp_name.lower():
                minimal_results[exp_name] = data
            elif 'all' in exp_name.lower():
                all_results[exp_name] = data
        
        print(f"Found {len(minimal_results)} minimal and {len(all_results)} all-features experiments")
        
        if minimal_results and all_results:
            print("\n📈 Performance Impact of Additional Features:")
            
            # Compare baseline performance from experiments
            for target in ['wind_power_onshore', 'wind_power_offshore']:
                print(f"\n{target.upper()}:")
                
                minimal_baselines = []
                all_baselines = []
                
                for exp_name, data in minimal_results.items():
                    if target in data['baseline_metrics']:
                        minimal_baselines.append(data['baseline_metrics'][target]['overall']['mae'])
                
                for exp_name, data in all_results.items():
                    if target in data['baseline_metrics']:
                        all_baselines.append(data['baseline_metrics'][target]['overall']['mae'])
                
                if minimal_baselines and all_baselines:
                    min_avg = np.mean(minimal_baselines)
                    all_avg = np.mean(all_baselines)
                    difference = all_avg - min_avg
                    
                    print(f"  Minimal features MAE: {min_avg:.3f}")
                    print(f"  All features MAE:     {all_avg:.3f}")
                    print(f"  Difference:           {difference:+.3f} ({'worse' if difference > 0 else 'better'})")
                    
                    # Show which additional features had highest importance
                    if target in list(all_results.values())[0]['importance_scores']:
                        additional_features = []
                        for feature, scores in list(all_results.values())[0]['importance_scores'][target].items():
                            if feature not in ['u100', 'v100', 'u10', 'v10', 'forecast_lead_hours']:
                                additional_features.append((feature, scores['mean_importance']))
                        
                        additional_features.sort(key=lambda x: x[1], reverse=True)
                        print(f"  Top additional features:")
                        for feature, importance in additional_features[:5]:
                            print(f"    • {feature:20s}: {importance:6.3f}")
    
    def identify_optimal_feature_sets(self):
        """Identify optimal feature combinations based on importance."""
        print("\n🎯 OPTIMAL FEATURE SET RECOMMENDATIONS")
        print("=" * 50)
        
        for target in ['wind_power_onshore', 'wind_power_offshore']:
            print(f"\n🏆 {target.upper()} RECOMMENDATIONS:")
            print("-" * 35)
            
            # Aggregate all importance scores for this target
            all_features = defaultdict(list)
            
            for exp_name, data in self.permutation_results.items():
                if target in data['importance_scores']:
                    for feature, scores in data['importance_scores'][target].items():
                        all_features[feature].append(scores['mean_importance'])
            
            # Calculate average importance
            feature_importance = []
            for feature, scores in all_features.items():
                feature_importance.append({
                    'feature': feature,
                    'avg_importance': np.mean(scores),
                    'category': self._categorize_feature(feature)
                })
            
            feature_importance.sort(key=lambda x: x['avg_importance'], reverse=True)
            
            # Recommend feature sets
            print("📋 Recommended feature sets:")
            
            # Essential features (always include)
            essential = [f for f in feature_importance if f['category'] == 'core_wind']
            essential_names = [f['feature'] for f in essential]
            
            # High-value features (importance > 0.3)
            high_value = [f for f in feature_importance if f['avg_importance'] > 0.3 and f['category'] != 'core_wind']
            
            # Medium-value features (importance 0.1-0.3)
            medium_value = [f for f in feature_importance if 0.1 <= f['avg_importance'] <= 0.3 and f['category'] != 'core_wind']
            
            print(f"\n1. MINIMAL SET ({len(essential_names)} features):")
            print(f"   {', '.join(essential_names)}")
            
            if high_value:
                recommended = essential_names + [f['feature'] for f in high_value]
                print(f"\n2. RECOMMENDED SET ({len(recommended)} features):")
                print(f"   {', '.join(recommended)}")
                print(f"   Added: {', '.join([f['feature'] for f in high_value])}")
            
            if medium_value:
                extended = essential_names + [f['feature'] for f in high_value] + [f['feature'] for f in medium_value[:3]]
                print(f"\n3. EXTENDED SET ({len(extended)} features):")
                print(f"   {', '.join(extended)}")
                print(f"   Added: {', '.join([f['feature'] for f in medium_value[:3]])}")
            
            # Features to avoid (low or negative importance)
            low_value = [f for f in feature_importance if f['avg_importance'] < 0.05]
            if low_value:
                print(f"\n❌ FEATURES TO AVOID (low importance):")
                for f in low_value[:5]:
                    print(f"   • {f['feature']} (importance: {f['avg_importance']:.3f})")
    
    def create_summary_visualizations(self, save_dir=None):
        """Create summary visualizations of feature importance."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Feature importance ranking
        self._plot_feature_importance_ranking(axes[0, 0])
        
        # Category comparison
        self._plot_category_comparison(axes[0, 1])
        
        # Minimal vs All comparison
        self._plot_minimal_vs_all(axes[1, 0])
        
        # Feature importance distribution
        self._plot_importance_distribution(axes[1, 1])
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / "permutation_importance_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualizations saved to {save_path}")
        
        plt.show()
    
    def _plot_feature_importance_ranking(self, ax):
        """Plot feature importance ranking."""
        ax.set_title("Feature Importance Ranking")
        # Implementation depends on available data
        
    def _plot_category_comparison(self, ax):
        """Plot comparison by feature category."""
        ax.set_title("Importance by Feature Category")
        
    def _plot_minimal_vs_all(self, ax):
        """Plot minimal vs all feature set comparison."""
        ax.set_title("Minimal vs All Features")
        
    def _plot_importance_distribution(self, ax):
        """Plot distribution of importance scores."""
        ax.set_title("Importance Score Distribution")
    
    def generate_summary_report(self, save_dir=None):
        """Generate comprehensive summary report."""
        print("\n📄 GENERATING SUMMARY REPORT")
        print("=" * 35)
        
        report = []
        report.append("🔬 PERMUTATION FEATURE IMPORTANCE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Experiments analyzed: {len(self.permutation_results)}")
        report.append("")
        
        # Add key findings
        report.append("🏆 KEY FINDINGS:")
        report.append("1. Most important features for wind power forecasting")
        report.append("2. Features that add minimal value (candidates for removal)")
        report.append("3. Optimal feature set recommendations")
        report.append("4. Comparison of minimal vs comprehensive feature sets")
        report.append("")
        
        # Add recommendations
        report.append("💡 RECOMMENDATIONS:")
        report.append("1. Focus on high-importance features for production models")
        report.append("2. Remove low-importance features to reduce complexity")
        report.append("3. Test recommended feature sets with longer training")
        report.append("4. Consider feature importance differences between onshore/offshore")
        
        report_text = "\n".join(report)
        print(report_text)
        
        if save_dir:
            with open(Path(save_dir) / "permutation_analysis_report.txt", 'w') as f:
                f.write(report_text)
        
        return report_text

def main():
    parser = argparse.ArgumentParser(description='Analyze permutation feature importance results')
    parser.add_argument('results_dir', help='Directory containing permutation importance results')
    parser.add_argument('--output_dir', help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(exist_ok=True)
    
    # Run analysis
    analyzer = PermutationResultsAnalyzer(args.results_dir)
    analyzer.load_results()
    
    # Perform analyses
    analyzer.analyze_feature_importance_rankings()
    analyzer.analyze_feature_categories()
    analyzer.compare_minimal_vs_all()
    analyzer.identify_optimal_feature_sets()
    
    # Create visualizations
    analyzer.create_summary_visualizations(save_dir=args.output_dir)
    
    # Generate report
    analyzer.generate_summary_report(save_dir=args.output_dir)
    
    if args.output_dir:
        print(f"\n📁 Analysis results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()