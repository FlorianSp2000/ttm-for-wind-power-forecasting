#!/usr/bin/env python3
"""Efficient Iterative Feature Pruning for Wind Power Forecasting - trains model once per iteration, then evaluates all features using model passing."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import argparse
import copy

# Import the main experiment function
from experiment_runner import run_wind_power_experiment, parse_feature_list, str_to_bool

def run_efficient_iterative_pruning(
    spatial_config="simple",
    eval_split="valid", 
    dataset_path="data/Realised_Supply_Germany.csv",
    weather_path="data/Weather_Data_Germany.csv",
    weather_path_2="data/Weather_Data_Germany_2022.csv",
    model_path="ibm-granite/granite-timeseries-ttm-r2",
    model_type="finetuned",
    context_length=512,
    prediction_length=24,
    learning_rate=0.001,
    num_epochs=25,
    batch_size=64,
    out_dir="results/",
    dataset_name="german_wind_power", 
    seed=42,
    initial_features=None,
    derived_features=True,
    n_shuffle_runs=3,
    max_iterations=10
):
    """Run efficient iterative feature pruning - train once per iteration, evaluate all features with that model.
    
    :param initial_features: Starting feature list
    :param max_iterations: Maximum pruning iterations
    :param n_shuffle_runs: Number of shuffle runs per feature for stability
    :returns: Dictionary with pruning results for each target
    """
    
    # Ensure output directory exists
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("🌱 STARTING EFFICIENT ITERATIVE FEATURE PRUNING")
    print("=" * 55)
    print(f"Initial features: {len(initial_features)}")
    print(f"Targets: onshore and offshore wind power")
    print(f"Strategy: Train once per iteration, then evaluate all features")
    print(f"Efficiency gain: ~{len(initial_features) * n_shuffle_runs}x fewer training runs")
    print(f"Stop when: All features have importance ≥ 0.0")
    print(f"Output directory: {out_dir}")
    print("")
    
    # Track results for both targets
    pruning_results = {
        'wind_power_onshore': {
            'iterations': [],
            'feature_sets': [],
            'importance_scores': [],
            'performance_history': []
        },
        'wind_power_offshore': {
            'iterations': [],
            'feature_sets': [],
            'importance_scores': [],
            'performance_history': []
        }
    }
    
    # Initialize feature sets for each target (start with same set)
    current_features = {
        'wind_power_onshore': initial_features.copy(),
        'wind_power_offshore': initial_features.copy()
    }
    
    # Track which targets are still being pruned
    active_targets = {'wind_power_onshore', 'wind_power_offshore'}
    
    for iteration in range(max_iterations):
        print(f"\n🔄 ITERATION {iteration + 1}/{max_iterations}")
        print("=" * 40)
        
        # Create a copy of active targets to avoid "set changed during iteration" error
        current_active_targets = active_targets.copy()
        targets_to_prune = []
        
        # Run efficient permutation importance for each active target
        for target in current_active_targets:
            features = current_features[target]
            
            print(f"\n📊 Analyzing {target} ({len(features)} features)")
            print(f"Features: {features[:5]}{'...' if len(features) > 5 else ''}")
            
            # Create feature config for this iteration
            feature_config = {
                'base_features': features,
                'derived_features': derived_features,
                'temporal_features': False,
                'lag_features': []
            }
            
            # Run efficient permutation importance (train once, evaluate all features)
            importance_scores = run_efficient_permutation_analysis(
                feature_config=feature_config,
                spatial_config=spatial_config,
                eval_split=eval_split,
                dataset_path=dataset_path,
                weather_path=weather_path,
                weather_path_2=weather_path_2,
                model_path=model_path,
                model_type=model_type,
                context_length=context_length,
                prediction_length=prediction_length,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                batch_size=batch_size,
                seed=seed + iteration,  # Different seed per iteration
                n_shuffle_runs=n_shuffle_runs,
                iteration=iteration + 1
            )
            
            if target not in importance_scores:
                print(f"⚠️  No results for {target}, skipping")
                continue
            
            target_scores = importance_scores[target]
            baseline_mae = target_scores['baseline_mae']
            
            # Extract feature importance
            feature_importance = []
            for feature_name, scores in target_scores['feature_scores'].items():
                importance = scores['mean_importance']
                feature_importance.append({
                    'feature': feature_name,
                    'importance': importance,
                    'std': scores['std_importance']
                })
            
            # Sort by importance (worst first)
            feature_importance.sort(key=lambda x: x['importance'])
            
            # Store results for this iteration
            pruning_results[target]['iterations'].append(iteration + 1)
            pruning_results[target]['feature_sets'].append(features.copy())
            pruning_results[target]['importance_scores'].append(feature_importance.copy())
            pruning_results[target]['performance_history'].append(baseline_mae)
            
            # Print current iteration results
            print(f"\nIteration {iteration + 1} Results for {target}:")
            print(f"Baseline MAE: {baseline_mae:.4f}")
            print("Feature Importance (worst to best):")
            for i, f in enumerate(feature_importance):
                status = "❌ NEGATIVE" if f['importance'] < 0 else "✅ POSITIVE"
                print(f"  {f['feature']:20s}: {f['importance']:8.4f} ± {f['std']:6.4f} {status}")
            
            # Identify features to remove
            negative_features = [f for f in feature_importance if f['importance'] < 0]
            
            if negative_features:
                # Remove up to 2 worst features
                to_remove = negative_features[:min(2, len(negative_features))]
                remove_names = [f['feature'] for f in to_remove]
                
                print(f"\n🗑️  Removing {len(remove_names)} features: {remove_names}")
                
                # Update feature set for next iteration
                new_features = [f for f in features if f not in remove_names]
                current_features[target] = new_features
                targets_to_prune.append(target)
                
            else:
                print(f"\n🎯 {target} CONVERGED: All features have positive importance!")
                active_targets.discard(target)
        
        # Check if we should continue
        if not active_targets:
            print(f"\n🏁 ALL TARGETS CONVERGED after {iteration + 1} iterations")
            break
        
        print(f"\n📈 Iteration {iteration + 1} Summary:")
        for target in targets_to_prune:
            remaining = len(current_features[target])
            print(f"  {target}: {remaining} features remaining")
    
    # Final summary
    print(f"\n🎉 EFFICIENT ITERATIVE PRUNING COMPLETED")
    print("=" * 45)
    
    for target in pruning_results:
        iterations = len(pruning_results[target]['iterations'])
        if iterations > 0:
            initial_count = len(pruning_results[target]['feature_sets'][0])
            final_count = len(pruning_results[target]['feature_sets'][-1])
            final_features = pruning_results[target]['feature_sets'][-1]
            
            print(f"\n{target.upper()}:")
            print(f"  Iterations: {iterations}")
            print(f"  Features: {initial_count} → {final_count}")
            print(f"  Final set: {final_features}")
    
    return pruning_results

def run_efficient_permutation_analysis(feature_config, n_shuffle_runs=3, iteration=1, **kwargs):
    """Run efficient permutation importance analysis - train once, then evaluate all features with that model.
    
    :param feature_config: Feature configuration dict
    :param n_shuffle_runs: Number of shuffle runs per feature for stability
    :param iteration: Current iteration number (for logging)
    :returns: Dictionary with importance scores for each target
    """
    
    print(f"🔍 EFFICIENT PERMUTATION ANALYSIS - Iteration {iteration}")
    print(f"🏋️  Step 1: Train model once with {len(feature_config['base_features'])} base features")
    
    # Step 1: Train model once and get both results and the trained model
    print("🚂 Training model...")
    baseline_results = run_wind_power_experiment(
        feature_config=feature_config,
        save_results=False,
        shuffle_feature=None,
        trained_model=None,  # No model passed, so it will train
        **kwargs
    )
    
    # Extract the trained model and metrics
    trained_model = baseline_results["trained_model"]
    baseline_metrics = baseline_results['metrics']
    observable_columns = baseline_results['metadata']['feature_config']['observable_columns']
    
    print(f"✅ Model trained successfully!")
    print(f"📊 Observable columns ({len(observable_columns)}): {observable_columns[:5]}...")
    
    # Step 2: Test each feature by shuffling using the pre-trained model
    print(f"🔀 Step 2: Testing {len(observable_columns)} features with {n_shuffle_runs} shuffles each...")
    
    importance_scores = {}
    
    for target in baseline_metrics.keys():
        baseline_mae = baseline_metrics[target]['overall']['mae']
        print(f"🎯 Target {target} baseline MAE: {baseline_mae:.4f}")
        
        feature_scores = {}
        
        for i, feature_name in enumerate(observable_columns):
            print(f"  🔀 Testing {feature_name} ({i+1}/{len(observable_columns)})...", end=" ")
            
            shuffle_results = []
            
            # Run multiple shuffles for stability
            for run_idx in range(n_shuffle_runs):
                try:
                    shuffled_results = run_wind_power_experiment(
                        feature_config=feature_config,
                        save_results=False,
                        shuffle_feature=feature_name,
                        trained_model=trained_model,  # Pass the trained model, force zero-shot mode
                        seed=kwargs.get('seed', 42) + run_idx,  # Ensure different seed for each shuffle
                        **{k: v for k, v in kwargs.items() if k != 'seed'}
                    )
                    
                    shuffled_mae = shuffled_results['metrics'][target]['overall']['mae']
                    importance = shuffled_mae - baseline_mae
                    shuffle_results.append(importance)
                    
                except Exception as e:
                    print(f"Error in run {run_idx + 1}: {e}")
                    continue
            
            if shuffle_results:
                # Calculate statistics
                mean_importance = np.mean(shuffle_results)
                std_importance = np.std(shuffle_results)
                
                feature_scores[feature_name] = {
                    'mean_importance': mean_importance,
                    'std_importance': std_importance,
                    'raw_scores': shuffle_results
                }
                print(f"shuffle_results: {shuffle_results}")
                print(f"Importance: {mean_importance:.4f} ± {std_importance:.4f}")
            else:
                print("❌ No valid results")
        
        importance_scores[target] = {
            'baseline_mae': baseline_mae,
            'feature_scores': feature_scores
        }
    
    print(f"✅ Efficient permutation analysis completed!")
    print(f"🏆 Training efficiency: 1 training instead of {len(observable_columns) * n_shuffle_runs}")
    
    return importance_scores

def save_pruning_results(pruning_results, out_dir):
    """Save iterative pruning results with visualizations."""
    
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save complete results as JSON
    results_file = out_path / f"efficient_iterative_pruning_results_{timestamp}.json"
    
    # Convert numpy types for JSON serialization
    json_results = copy.deepcopy(pruning_results)
    for target in json_results:
        for iteration_scores in json_results[target]['importance_scores']:
            for feature in iteration_scores:
                if 'raw_scores' in feature:
                    feature['raw_scores'] = [float(x) for x in feature['raw_scores']]
                feature['importance'] = float(feature['importance'])
                feature['std'] = float(feature['std'])
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Create visualizations for each target
    for target in pruning_results:
        if not pruning_results[target]['iterations']:
            continue
            
        create_pruning_visualizations(target, pruning_results[target], out_path, timestamp)
    
    # Save summary CSV files
    for target in pruning_results:
        if not pruning_results[target]['iterations']:
            continue
            
        # Final feature importance
        final_scores = pruning_results[target]['importance_scores'][-1]
        final_df = pd.DataFrame(final_scores)
        final_df = final_df.sort_values('importance', ascending=False)
        
        csv_file = out_path / f"final_features_{target}_{timestamp}.csv"
        final_df.to_csv(csv_file, index=False)
    
    print(f"📁 Pruning results saved to: {out_path}")
    return out_path

def create_pruning_visualizations(target, target_results, out_path, timestamp):
    """Create visualizations showing the efficient pruning process."""
    
    iterations = target_results['iterations']
    importance_scores = target_results['importance_scores']
    performance_history = target_results['performance_history']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Efficient Iterative Feature Pruning - {target.replace("_", " ").title()}', fontsize=16)
    
    # 1. Feature count over iterations
    ax1 = axes[0, 0]
    feature_counts = [len(scores) for scores in importance_scores]
    ax1.plot(iterations, feature_counts, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Number of Features')
    ax1.set_title('Feature Count Reduction')
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance over iterations
    ax2 = axes[0, 1]
    ax2.plot(iterations, performance_history, 's-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('MAE')
    ax2.set_title('Model Performance During Pruning')
    ax2.grid(True, alpha=0.3)
    
    # 3. Final feature importance ranking
    ax3 = axes[1, 0]
    final_scores = importance_scores[-1]
    final_df = pd.DataFrame(final_scores).sort_values('importance', ascending=True)
    
    colors = ['red' if x < 0 else 'green' for x in final_df['importance']]
    bars = ax3.barh(range(len(final_df)), final_df['importance'], color=colors, alpha=0.7)
    ax3.set_yticks(range(len(final_df)))
    ax3.set_yticklabels(final_df['feature'], fontsize=10)
    ax3.set_xlabel('Feature Importance')
    ax3.set_title('Final Feature Importance')
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # 4. Training efficiency comparison
    ax4 = axes[1, 1]
    
    # Calculate training counts
    total_features = sum(len(scores) for scores in importance_scores)
    n_shuffle_runs = 3  # Assume 3 runs
    old_method_trainings = total_features * n_shuffle_runs + len(iterations)  # Old: shuffles * features + baselines
    new_method_trainings = len(iterations)  # New: just one per iteration
    
    methods = ['Old Method\n(Retrain per Feature)', 'New Method\n(Train Once per Iteration)']
    training_counts = [old_method_trainings, new_method_trainings]
    colors = ['red', 'green']
    
    bars = ax4.bar(methods, training_counts, color=colors, alpha=0.7)
    ax4.set_ylabel('Total Training Runs')
    ax4.set_title(f'Training Efficiency Comparison\n{old_method_trainings//new_method_trainings}x Improvement')
    
    # Add value labels on bars
    for bar, count in zip(bars, training_counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_counts)*0.01, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    viz_file = out_path / f"efficient_pruning_visualization_{target}_{timestamp}.png"
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualization saved: {viz_file}")

def main():
    parser = argparse.ArgumentParser(description='Run efficient iterative feature pruning analysis')
    
    # Same arguments as regular experiment runner
    parser.add_argument('--dataset_path', type=str, default='data/Realised_Supply_Germany.csv')
    parser.add_argument('--weather_path', type=str, default='data/Weather_Data_Germany.csv')
    parser.add_argument('--weather_path_2', type=str, default='data/Weather_Data_Germany_2022.csv')
    parser.add_argument('--model_path', type=str, default='ibm-granite/granite-timeseries-ttm-r2')
    parser.add_argument('--model_type', type=str, choices=['zero-shot', 'finetuned'], default='finetuned')
    
    parser.add_argument('--spatial_config', type=str, choices=['simple', 'coastal_inland', 'latitude_bands'], default='simple')
    parser.add_argument('--eval_split', type=str, choices=['valid', 'test'], default='valid')
    
    parser.add_argument('--initial_features', type=str, default='u100,v100,u10,v10,msl,t2m,cdir,blh,tcc,tp')
    parser.add_argument('--derived_features', type=str_to_bool, default=True)
    parser.add_argument('--n_shuffle_runs', type=int, default=3)
    parser.add_argument('--max_iterations', type=int, default=10)
    
    parser.add_argument('--context_length', type=int, default=512)
    parser.add_argument('--prediction_length', type=int, default=24)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=64)
    
    parser.add_argument('--out_dir', type=str, default='results/')
    parser.add_argument('--dataset_name', type=str, default='german_wind_power')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Parse initial features
    initial_features = parse_feature_list(args.initial_features)
    
    # Run efficient iterative pruning
    results = run_efficient_iterative_pruning(
        spatial_config=args.spatial_config,
        eval_split=args.eval_split,
        dataset_path=args.dataset_path,
        weather_path=args.weather_path,
        weather_path_2=args.weather_path_2,
        model_path=args.model_path,
        model_type=args.model_type,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
        dataset_name=args.dataset_name,
        seed=args.seed,
        initial_features=initial_features,
        derived_features=args.derived_features,
        n_shuffle_runs=args.n_shuffle_runs,
        max_iterations=args.max_iterations
    )
    
    # Save results and create visualizations
    save_pruning_results(results, args.out_dir)

if __name__ == "__main__":
    main()