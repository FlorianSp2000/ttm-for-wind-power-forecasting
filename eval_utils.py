import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def denormalize_ttm_predictions_simple(predictions_np: np.ndarray, tsp: object) -> dict:
    """Simple denormalization for TTM predictions.
    
    :param predictions_np: Raw model predictions in normalized space, shape (n_windows, horizon, n_channels)
    :param tsp: Fitted preprocessor with target_scaler_dict
    :returns: Denormalized predictions for each target
    """

    # Get target scaler
    target_scaler = tsp.target_scaler_dict['0']

    # Extract target predictions (first 2 channels)
    n_targets = len(tsp.target_columns)
    target_preds = predictions_np[:, :, :n_targets]  # (n_windows, 24, 2)

    # Reshape and denormalize
    n_windows, horizon, _ = target_preds.shape
    reshaped = target_preds.reshape(-1, n_targets)
    denormalized = target_scaler.inverse_transform(reshaped)
    final_shape = denormalized.reshape(n_windows, horizon, n_targets)

    # Return as dictionary
    result = {}
    for i, col in enumerate(tsp.target_columns):
        result[col] = final_shape[:, :, i]

    return result

def extract_experiment_metadata(result: dict, ttm_model: object, dset_test: object, predictions_np: np.ndarray,
                               batch_size: int = 64, model_type: str = "zero-shot", model_path: str = None,
                               dataset_name: str = "german_wind_power", spatial_config: str = "simple", run_name: str = "") -> dict:
    """Extract experiment-specific metadata for evaluation pipeline tracking.
    
    :param result: Output from prepare_wind_power_dataset()
    :param ttm_model: The TTM model instance
    :param dset_test: Test dataset
    :param predictions_np: Raw model predictions
    :param batch_size: Batch size used for inference
    :param model_type: Model type like "zero-shot", "fine-tuned", etc.
    :param model_path: Path to model weights
    :param dataset_name: Name of dataset (configurable)
    :param spatial_config: Spatial coverage description (configurable)
    :returns: Experiment-specific metadata only
    """

    print("🔍 EXTRACTING EXPERIMENT METADATA")
    print("="*45)

    tsp = result['preprocessor']

    # 1. Feature configuration (varies between experiments)
    feature_config = {
        'target_columns': tsp.target_columns.copy(),
        'observable_columns': getattr(tsp, 'observable_columns', []).copy(),
        'conditional_columns': getattr(tsp, 'conditional_columns', []).copy(),
        'num_targets': len(tsp.target_columns),
        'num_observables': len(getattr(tsp, 'observable_columns', [])),
        'num_total_features': len(tsp.target_columns) + len(getattr(tsp, 'observable_columns', [])) + len(getattr(tsp, 'conditional_columns', [])),
        'scaling_enabled': getattr(tsp, 'scaling', True),
        'scaler_type': getattr(tsp, 'scaler_type', 'standard')
    }

    # 2. Model and training hyperparameters (varies between experiments)
    model_config = {
        'model_type': model_type,
        'model_path': model_path,
        'context_length': result['context_length'],
        'prediction_length': result['prediction_length'],
        'batch_size': batch_size,
        'freq_prefix_tuning': getattr(ttm_model.config, 'freq_prefix_tuning', False),
        'resolution_prefix_tuning': getattr(ttm_model.config, 'resolution_prefix_tuning', False)
    }

    # 3. Prediction metadata (varies between experiments)
    prediction_info = {
        'prediction_shape': list(predictions_np.shape),
        'num_windows': predictions_np.shape[0],
        'horizon_length': predictions_np.shape[1],
        'num_channels': predictions_np.shape[2],
        'test_windows': len(dset_test)
    }

    # 4. Experiment tracking info
    experiment_info = {
        'timestamp': datetime.now().isoformat(),
        'experiment_id': generate_experiment_id(model_config, feature_config),
        'run_name': run_name,
        'dataset_name': dataset_name,
        'spatial_config': spatial_config
    }

    # Combine all metadata (experiment-specific only)
    metadata = {
        'feature_config': feature_config,
        'model_config': model_config,
        'prediction_info': prediction_info,
        'experiment_info': experiment_info
    }

    # Print summary
    # print_metadata_summary(metadata)

    return metadata


def generate_experiment_id(model_config, feature_config):
    """Generate timestamp-based experiment ID"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp

def generate_run_name(model_type, model_config, feature_config, spatial_config):
    """Generate human-readable run name"""
    return f"{model_type}_cxt{model_config['context_length']}_pred{model_config['prediction_length']}_feat{feature_config['num_total_features']}_sp:{spatial_config}"

def print_metadata_summary(metadata):
    """Print a concise summary of the extracted metadata"""

    print(f"\n📋 EXPERIMENT METADATA SUMMARY")
    print("-" * 40)

    exp = metadata['experiment_info']
    features = metadata['feature_config']
    model = metadata['model_config']
    pred = metadata['prediction_info']

    print(f"🆔 Experiment ID: {exp['experiment_id']}")
    print(f"🏷️  Run Name: {exp['run_name']}")
    print(f"📅 Timestamp: {exp['timestamp'][:19]}")
    print(f"🗺️  Coverage: {exp['spatial_config']}")

    print(f"\n🎯 Features:")
    print(f"   Targets: {features['num_targets']} {features['target_columns']}")
    print(f"   Observables: {features['num_observables']}")
    print(f"   Total features: {features['num_total_features']}")
    print(f"   Scaling: {features['scaler_type'] if features['scaling_enabled'] else 'none'}")

    print(f"\n🤖 Model: {model['model_type']}")
    print(f"   Context: {model['context_length']}h ({model['context_length']/24:.1f} days)")
    print(f"   Prediction: {model['prediction_length']}h")
    print(f"   Batch size: {model['batch_size']}")

    print(f"\n🔮 Predictions: {pred['prediction_shape']}")
    print(f"   ({pred['num_windows']:,} windows, {pred['horizon_length']}h horizon, {pred['num_channels']} channels)")


def compute_multihorizon_metrics(predictions_denormalized: dict, ground_truth: dict, target_columns: list, horizons: list = None) -> dict:
    """Compute comprehensive metrics for all forecast horizons.
    
    :param predictions_denormalized: Output from denormalize_ttm_predictions_simple()
    :param ground_truth: Ground truth arrays for each target
    :param target_columns: List of target column names
    :param horizons: Forecast horizons to evaluate (in hours, 1-indexed). If None, uses all 24 hours
    :returns: Metrics organized by target and horizon
    """
    print(f"\nCOMPUTING MULTI-HORIZON METRICS")
    print("-" * 40)

    # Default to all 24 forecast horizons
    if horizons is None:
        horizons = list(range(1, 25))  # 1h, 2h, 3h, ..., 24h ahead

    metrics = {}

    for target in target_columns:
        if target not in predictions_denormalized or target not in ground_truth:
            continue

        pred_array = predictions_denormalized[target]  # (n_windows, 24)
        truth_array = ground_truth[target]

        target_metrics = {}

        print(f"\n  {target}:")

        for h in horizons:
            h_idx = h - 1  # Convert to 0-indexed

            if h_idx >= pred_array.shape[1]:
                continue

            # Extract h-hour ahead predictions
            pred_h = pred_array[:, h_idx]

            # Align with ground truth (account for forecast horizon offset)
            max_len = min(len(truth_array) - h, len(pred_h))

            if max_len <= 0:
                continue

            aligned_truth = truth_array[h:h + max_len]
            aligned_pred = pred_h[:max_len]

            # Calculate metrics (only MAE and RMSE)
            mae = np.mean(np.abs(aligned_truth - aligned_pred))
            rmse = np.sqrt(np.mean((aligned_truth - aligned_pred)**2))

            target_metrics[f'{h}h'] = {
                'mae': float(mae),
                'rmse': float(rmse)
            }

            # Print every 4th hour to avoid clutter
            # if h % 4 == 1 or h == 24:
            #     print(f"    {h:>2}h ahead: MAE={mae:6.1f} MW, RMSE={rmse:6.1f} MW")

        # Overall metrics (average across all horizons)
        if target_metrics:
            overall_mae = np.mean([m['mae'] for m in target_metrics.values()])
            overall_rmse = np.mean([m['rmse'] for m in target_metrics.values()])

            target_metrics['overall'] = {
                'mae': float(overall_mae),
                'rmse': float(overall_rmse)
            }

            print(f"    Overall: MAE={overall_mae:6.1f} MW, RMSE={overall_rmse:6.1f} MW")

        metrics[target] = target_metrics

    return metrics

def get_evaluation_data(metadata: dict, predictions_np: np.ndarray, df_combined: pd.DataFrame, tsp: object, split_config: dict,
                          out_dir: str = "results/", save_results: bool = True, eval_split: str = "test") -> dict:
    """Prepare data, compute metrics, and save evaluation results.
    
    :param metadata: Experiment metadata
    :param predictions_np: Raw model predictions
    :param df_combined: Full combined dataset
    :param tsp: Fitted preprocessor
    :param split_config: Train/val/test split configuration
    :param out_dir: Base output directory
    :param save_results: Whether to save results to disk
    :param eval_split: Which split to use for evaluation ('valid' or 'test')
    :returns: Complete evaluation results
    """

    print(f"\n🔧 PREPARING EVALUATION DATA")
    print("-" * 30)

    # Extract data based on specified split
    if eval_split == "valid":
        split_start, split_end = split_config['valid']
    elif eval_split == "test":
        split_start, split_end = split_config['test']
    else:
        raise ValueError(f"eval_split must be 'val' or 'test', got '{eval_split}'")
    
    df_eval = df_combined.iloc[split_start:split_end].copy()
    # Prepare ground truth arrays
    ground_truth = {}
    for target in metadata['feature_config']['target_columns']:
        if target in df_eval.columns:
            ground_truth[target] = df_eval[target].values

    # Denormalize predictions (function should be defined in jupyter cell above)
    denormalized_predictions = denormalize_ttm_predictions_simple(predictions_np, tsp)

    # Compute metrics
    metrics = compute_multihorizon_metrics(
        denormalized_predictions,
        ground_truth,
        metadata['feature_config']['target_columns']
    )

    evaluation_results = {
        'metadata': metadata,
        'metrics': metrics,
        'predictions_raw': predictions_np,
        'predictions_denormalized': denormalized_predictions,
        'ground_truth': ground_truth,
        'test_timestamps': df_eval['timestamp'].values
    }

    if save_results:
        save_evaluation_results(evaluation_results, out_dir)

    print(f"\n✅ Evaluation complete:")
    print(f"   Targets evaluated: {list(metrics.keys())}")
    print(f"   Horizons per target: {len([k for k in list(metrics.values())[0].keys() if k != 'overall'])}")

    return evaluation_results

def save_evaluation_results(evaluation_results: dict, out_dir: str = "results/") -> Path:
    """Save evaluation results to structured directory.
    
    :param evaluation_results: Complete evaluation results from get_evaluation_data()
    :param out_dir: Base output directory
    :returns: Path to experiment directory
    """

    print(f"\n💾 SAVING EVALUATION RESULTS")
    print("-" * 30)

    metadata = evaluation_results['metadata']
    metrics = evaluation_results['metrics']

    # Create base output directory
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)

    # Create experiment-specific directory
    exp_id = metadata['experiment_info']['experiment_id']
    run_name = metadata['experiment_info']['run_name']
    exp_dir = out_path / f"{exp_id}_{run_name}"
    exp_dir.mkdir(exist_ok=True)

    print(f"📁 Experiment directory: {exp_dir}")

    # 1. Save metadata as structured CSV
    metadata_df = flatten_metadata_to_dataframe(metadata)
    metadata_path = exp_dir / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"   ✅ Metadata saved: {metadata_path.name}")

    # 2. Save metrics as structured CSV
    metrics_df = flatten_metrics_to_dataframe(metrics)
    metrics_path = exp_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"   ✅ Metrics saved: {metrics_path.name}")

    # 3. Save predictions as npz
    predictions_path = exp_dir / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        raw_predictions=evaluation_results['predictions_raw'],
        **evaluation_results['predictions_denormalized'],  # Save each target separately
        timestamps=evaluation_results['test_timestamps']
    )
    print(f"   ✅ Predictions saved: {predictions_path.name}")

    # 4. Save ground truth once (check if exists)
    ground_truth_path = out_path / "ground_truth.npz"
    if not ground_truth_path.exists():
        np.savez_compressed(
            ground_truth_path,
            **evaluation_results['ground_truth'],
            timestamps=evaluation_results['test_timestamps']
        )
        print(f"   ✅ Ground truth saved: {ground_truth_path.name}")
    else:
        print(f"   ⏭️  Ground truth exists: {ground_truth_path.name}")

    print(f"\n📁 All results saved in: {exp_dir}")
    return exp_dir

def flatten_metadata_to_dataframe(metadata: dict) -> pd.DataFrame:
    """Convert nested metadata dict to flat DataFrame for easy CSV reading.
    
    :param metadata: Nested metadata dictionary
    :returns: Flattened DataFrame with columns: section, parameter, value
    """
    rows = []

    # Flatten each section
    for section_name, section_data in metadata.items():
        if isinstance(section_data, dict):
            for key, value in section_data.items():
                if isinstance(value, (list, tuple)):
                    # Convert lists to string representation
                    value = str(value)

                rows.append({
                    'section': section_name,
                    'parameter': key,
                    'value': value
                })

    return pd.DataFrame(rows)

def flatten_metrics_to_dataframe(metrics: dict) -> pd.DataFrame:
    """Convert nested metrics dict to flat DataFrame for easy analysis.
    
    :param metrics: Nested metrics dictionary organized by target and horizon
    :returns: Flattened DataFrame with metrics as columns and target_horizon as rows
    """
    rows = []

    for target, target_metrics in metrics.items():
        for horizon, horizon_metrics in target_metrics.items():
            if isinstance(horizon_metrics, dict):
                for metric_name, metric_value in horizon_metrics.items():
                    rows.append({
                        'target': target,
                        'horizon': horizon,
                        'metric': metric_name,
                        'value': metric_value
                    })

    df = pd.DataFrame(rows)

    # Pivot for easier reading: columns = metrics, rows = target_horizon combinations
    if not df.empty:
        df_pivot = df.pivot_table(
            index=['target', 'horizon'],
            columns='metric',
            values='value',
            fill_value=None
        ).reset_index()

        # Flatten column names
        df_pivot.columns.name = None

        return df_pivot

    return df