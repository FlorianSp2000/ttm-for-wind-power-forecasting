import math
import os
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
import warnings
import argparse
import json
warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"

from tsfm_public import TimeSeriesPreprocessor, TrackingCallback, count_parameters, get_datasets
from tsfm_public.toolkit.get_model import get_model
from preprocess import prepare_wind_power_dataset
from eval_utils import (extract_experiment_metadata, generate_run_name, get_evaluation_data)

def run_wind_power_experiment(spatial_config="simple", eval_split="test", 
                             dataset_path="data/Realised_Supply_Germany.csv",
                             weather_path="data/Weather_Data_Germany.csv",
                             weather_path_2="data/Weather_Data_Germany_2022.csv",
                             model_path="ibm-granite/granite-timeseries-ttm-r2",
                             model_type="finetuned",
                             context_length=512, prediction_length=24,
                             learning_rate=0.001, num_epochs=5, batch_size=64,
                             out_dir="results/", dataset_name="german_wind_power",
                             seed=42, feature_config=None,
                             save_results=True, shuffle_feature=None, trained_model=None,
                             use_exog=True
                             ):
    """Run complete wind power forecasting experiment: data loading, training, and evaluation.
    
    :param model_type: Either "zero-shot" (skip training) or "finetuned" (train model)
    :param feature_config: Feature engineering configuration dict
    :param shuffle_feature: Feature name to shuffle for permutation importance (None for normal run)
    """
    
    set_seed(seed)
    
    # Data loading
    df = pd.read_csv(dataset_path, on_bad_lines="skip")
    weather = pd.read_csv(weather_path)
    weather2022 = pd.read_csv(weather_path_2)
    weather = pd.concat([weather, weather2022], axis=0).reset_index(drop=True)
    
    print(f"🔧 EXPERIMENT_RUNNER: Calling prepare_wind_power_dataset with shuffle_feature='{shuffle_feature}'")

    # Data preprocessing with feature configuration
    result = prepare_wind_power_dataset(df.copy(), weather.copy(), 
        spatial_config=spatial_config, context_length=context_length, 
        prediction_length=prediction_length, feature_config=feature_config, 
        verbose=False, shuffle_feature=shuffle_feature, random_seed=seed)

    df_combined = result['data']
    tsp = result['preprocessor']
    split_config = result['split_config']
    
    # Model and feature configuration
    model_config = {
        'model_type': model_type,
        'model_path': model_path,
        'context_length': context_length,
        'prediction_length': prediction_length,
    }
    feature_summary = {
        'target_columns': tsp.target_columns.copy(),
        'num_targets': len(tsp.target_columns),
        'num_observables': len(getattr(tsp, 'observable_columns', [])),
        'num_total_features': len(tsp.target_columns) + len(getattr(tsp, 'observable_columns', [])) + len(getattr(tsp, 'conditional_columns', [])),
        'feature_config': feature_config or {}
    }

    # Add shuffle info to the print
    shuffle_info = f" | Shuffling: {shuffle_feature}" if shuffle_feature else ""
    print(f"Running {model_type} experiment - {spatial_config} spatial - {feature_summary['num_total_features']} features{shuffle_info}")
    
    if trained_model is not None:
        model = trained_model
        model.eval()
        # Force zero-shot mode when model is passed
        model_type = "zero-shot"
    else:
        print("🆕 Loading new model from checkpoint")
        if use_exog:
            exog_args = dict(
                num_input_channels=tsp.num_input_channels,
                decoder_mode="mix_channel",
                prediction_channel_indices=tsp.prediction_channel_indices,
                exogenous_channel_indices=tsp.exogenous_channel_indices,
                fcm_context_length=1,
                fcm_use_mixer=True,
                fcm_mix_layers=2,
                enable_forecast_channel_mixing=True,
                fcm_prepend_past=True,
            )
        else:
            exog_args = {}
        
        model = get_model(
            model_path,
            context_length=context_length,
            prediction_length=prediction_length,
            freq_prefix_tuning=False,
            prefer_l1_loss=False,
            prefer_longer_context=True,
            **exog_args    # TO USE EXOGENOUS FEATURES
            )
    
    dset_train, dset_valid, dset_test = get_datasets(
        tsp, df_combined, split_config, 
        use_frequency_token=model.config.resolution_prefix_tuning
    )
    
    eval_dataset = dset_test if eval_split == "test" else dset_valid

    run_name = generate_run_name(model_config['model_type'], model_config, feature_summary, spatial_config)
    # Training (only if finetuned)
    if model_type == "finetuned":
        print(
            "Number of params before freezing backbone",
            count_parameters(model),
        )

        # Freeze the backbone of the model
        for param in model.backbone.parameters():
            param.requires_grad = False

        print(
            "Number of params after freezing the backbone",
            count_parameters(model),
        )
        temp_dir = tempfile.mkdtemp(prefix="ttm_experiment_")
        
        training_args = TrainingArguments(
            run_name=run_name,
            output_dir=temp_dir,
            overwrite_output_dir=True,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            do_eval=True,
            eval_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size // 2, # Prevent OOM on eval
            dataloader_num_workers=4,
            report_to=None,
            save_strategy="epoch",
            logging_strategy="epoch",
            logging_dir=os.path.join(temp_dir, "logs"),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=seed,
        )

        # Callbacks and optimizer
        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0)
        try:
            tracking_callback = TrackingCallback()
            callbacks = [early_stopping_callback, tracking_callback]
        except:
            callbacks = [early_stopping_callback]
        
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        scheduler = OneCycleLR(optimizer, learning_rate, epochs=num_epochs, 
                              steps_per_epoch=math.ceil(len(dset_train) / batch_size))
        
        # Training
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dset_train,
            eval_dataset=dset_valid,
            callbacks=callbacks,
            optimizers=(optimizer, scheduler),
        )
        
        train_result = trainer.train()
        print(f"✅ Training completed! Final loss: {train_result.training_loss:.4f}")
            
    elif model_type == "zero-shot":
        if trained_model is not None:
            print("📊 EXPERIMENT_RUNNER: Using passed trained model (skipping training)")
        else:
            print("📊 EXPERIMENT_RUNNER: Using zero-shot model (skipping training)")
        # Create minimal trainer for prediction only
        trainer = Trainer(model=model)
    
    else:
        raise ValueError(f"model_type must be 'zero-shot' or 'finetuned', got '{model_type}'")
    
    # Prediction and evaluation
    predictions_dict = trainer.predict(eval_dataset)
    predictions_np = predictions_dict.predictions[0]
    
    metadata = extract_experiment_metadata(
        result=result,
        ttm_model=model,
        dset_test=eval_dataset,
        predictions_np=predictions_np,
        batch_size=batch_size,
        model_type=model_config['model_type'],
        model_path=model_path,
        dataset_name=dataset_name,
        spatial_config=spatial_config,
        run_name=run_name,
    )
    
    evaluation_results = get_evaluation_data(
        metadata, predictions_np, df_combined, tsp, split_config,
        out_dir=out_dir, save_results=save_results, eval_split=eval_split
    )
    
    evaluation_results["trained_model"] = model
    evaluation_results["trainer"] = trainer

    if save_results:
        print(f"📁 Results saved: {metadata['experiment_info']['experiment_id']}")
    else:
        # Return results for permutation importance analysis
        return evaluation_results 

def parse_feature_list(feature_string):
    """Parse comma-separated feature string into list."""
    if not feature_string or feature_string.lower() == 'none':
        return []
    return [f.strip() for f in feature_string.split(',') if f.strip()]


def str_to_bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='Run wind power forecasting experiment')
    
    # Data and model configuration
    parser.add_argument('--dataset_path', type=str, default='data/Realised_Supply_Germany.csv',
                       help='Path to wind power dataset')
    parser.add_argument('--weather_path', type=str, default='data/Weather_Data_Germany.csv',
                       help='Path to weather data')
    parser.add_argument('--weather_path_2', type=str, default='data/Weather_Data_Germany_2022.csv',
                       help='Path to additional weather data')
    parser.add_argument('--model_path', type=str, default='ibm-granite/granite-timeseries-ttm-r2',
                       help='TTM model path')
    parser.add_argument('--model_type', type=str, choices=['zero-shot', 'finetuned'], default='finetuned',
                       help='Model type: zero-shot or finetuned')
    parser.add_argument('--save_results', type=str_to_bool, default=True,
                       help='Whether to save results to disk')
    parser.add_argument('--shuffle_feature', type=str, default=None,
                       help='Feature to shuffle for permutation importance (internal use)')

    
    # Experiment configuration
    parser.add_argument('--spatial_config', type=str, choices=['simple', 'coastal_inland', 'latitude_bands'], 
                       default='simple', help='Spatial aggregation strategy')
    parser.add_argument('--eval_split', type=str, choices=['valid', 'test'], default='test',
                       help='Dataset split for evaluation')
    
    # Feature configuration
    parser.add_argument('--base_features', type=str, 
                       default='u100,v100,u10,v10,msl,t2m,cdir,blh,tcc,tp',
                       help='Comma-separated list of base weather features')
    parser.add_argument('--derived_features', type=str_to_bool, default=True,
                       help='Include derived features (wind speed, direction)')
    parser.add_argument('--temporal_features', type=str_to_bool, default=False,
                       help='Include temporal features (hour, month)')
    parser.add_argument('--lag_features', type=str_to_bool, default=False,
                       help='Include lag features')
    
    # Training hyperparameters
    parser.add_argument('--context_length', type=int, default=512,
                       help='Context length for TTM model')
    parser.add_argument('--prediction_length', type=int, default=24,
                       help='Prediction horizon length')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training and evaluation')
    
    # Output configuration
    parser.add_argument('--out_dir', type=str, default='results/',
                       help='Output directory for results')
    parser.add_argument('--dataset_name', type=str, default='german_wind_power',
                       help='Dataset name for metadata')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create feature configuration
    feature_config = {
        'base_features': parse_feature_list(args.base_features),
        'derived_features': args.derived_features,
        'temporal_features': args.temporal_features,
        'lag_features': [] if not args.lag_features else [{'wind_speed_10m': [1, 2, 3]}]
    }
    
    # Run experiment
    run_wind_power_experiment(
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
        save_results=args.save_results,
        shuffle_feature=args.shuffle_feature,
        feature_config=feature_config
    )


if __name__ == "__main__":
    main()